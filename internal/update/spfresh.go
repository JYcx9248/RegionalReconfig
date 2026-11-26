package update

import (
	"context"
	"fmt"
	"math"
	"sort"

	"rtier/internal/epoch"
	"rtier/internal/nodeclient"
	"rtier/internal/placement"
)

// Policies of a major epoch (Updater.Policy). Both split a long list by a balanced 2-means into
// two children in its partition, each member in one child, and remove a short list.
//
//   - threshold (the default): a merged list's postings go to the list of their nearest live head;
//     nothing else moves.
//   - spfresh: SPFresh's LIRE (SOSP '23), below.
//
// SPFresh's LIRE as its paper describes it (§3.2-3.3, §4.2.1) and, where the paper leaves a point
// open, as its code does it (github.com/SPFresh/SPFresh, ExtraDynamicSearcher.h: Split's
// CollectReAssign, CheckIsNeedReassign, MergePostings, Reassign):
//
//   - After the splits, each split checks the vectors near it in SPFresh's order: the members of
//     its children, then those of the ReassignRange lists whose heads are nearest the old head
//     (SPFresh's ReassignK), nearest first. A split queues a vector once, in the first list where
//     one of SPFresh's conditions holds. A vector in a split's child is checked in that child
//     only: one that stays there does not take the other child too.
//   - A merge (§3.2) pairs a list due to merge with its nearest list: the first of the 64 heads
//     nearest its own (SPFresh's InternalResultNum) that keeps both under PostingLimit and is
//     neither split nor in another merge of the epoch, as the code chooses. The shorter of the two
//     goes (the one due to merge on a tie), its postings appended to the other. A posting of it
//     that is farther from the kept head than from its own old one is queued in the kept list
//     (the code's test of "the vectors of the deleted posting", §3.3). A list with no such partner
//     does not merge.
//   - A queued vector is inserted again by the index's own rule (ASSIGN's, Assign in
//     node_engine.cpp) if a list it was queued in is no longer among the lists the rule gives it:
//     it joins every list of its new set and leaves every other list that holds it, which the
//     owners find (LISTS_OF). SPFresh bumps its version instead, so that its old copies are
//     dropped when their postings are next split or merged (§4.2.1); here they go in the epoch.
//
// The rule is the index's build's. On an index built with RNG pruning (fusion_build --rng, as
// SPFresh's evaluation builds SPANN's) it is SPANN's RNG selection, SPFresh's RNGSelection: the
// nearest heads in order, each skipped if it is closer to one already taken than to the vector,
// up to the replica count. On FusionANNS's index it is the 8 nearest heads within the closure.
//
// SPFresh's own differences, kept out: it decides on raw vectors, we on PQ codes (the router's,
// so that no data node computes); its code checks a child's member in the neighboring lists too
// when the member is no closer to the old head than to its child's head -- by FusionANNS's rule
// such a member would often take the other child as well (the two new heads are close to each
// other, and nothing prunes the second), while the RNG rule would prune it, so whether to follow
// the code there on an RNG index is open; its code splits by SPTAG's balanced k-means and then
// gives every member to the nearer new head (keeping the old head when a new one is it), ours
// keeps the balanced halves -- both the paper's even split by balanced clustering; it acts on
// every insert and every search that meets a short posting, we once per batch; and here a list
// takes part in one merge per epoch.
const (
	PolicyThreshold = "threshold"
	PolicySPFresh   = "spfresh"
)

// ReassignStats is what SPFresh's reassign did in one major epoch.
type ReassignStats struct {
	Lists      int `json:"lists"`            // neighboring lists read, each once
	Checked    int `json:"checked"`          // postings checked against SPFresh's conditions, splits' and merges'
	Candidates int `json:"candidates"`       // vectors queued: by each split, the first posting that met them; by each merge
	Moved      int `json:"moved"`            // vectors inserted again
	Added      int `json:"added"`            // postings added for them
	Dropped    int `json:"dropped"`          // postings they left, planned ones taken out of the plan too
	Stale      int `json:"stale"`            // of those, in lists the epoch read nothing else from (SPFresh's stale copies)
	StaleParts int `json:"stale_partitions"` // partitions staged only to drop those
	Merged     int `json:"merged"`           // merges: a list appended to its nearest list with room
	NoPartner  int `json:"no_partner"`       // lists due to merge that no list had room for, left as they are
}

// childCandidate and neighborCandidate are SPFresh's conditions (CheckIsNeedReassign), on squared
// distances: a member of a split's child is checked if the old head is closer to it than its
// child's head; a member of a neighboring list, if a new head is closer to it than both the old
// head and its own list's head.
func childCandidate(dOld, dChild float32) bool { return dOld < dChild }

func neighborCandidate(dOld, dNew0, dNew1, dOwn float32) bool {
	d := min(dNew0, dNew1)
	return d < dOld && d < dOwn
}

// codesBatch bounds one ASSIGN_CODES or HEAD_DISTANCES call (its frame).
const codesBatch = 4096

// assignCodes is ASSIGN_CODES over codes (m bytes each), in frames of codesBatch codes.
func (u *Updater) assignCodes(ctx context.Context, codes []byte, m int) ([][]uint32, error) {
	var out [][]uint32
	for i := 0; i < len(codes); i += codesBatch * m {
		j := min(len(codes), i+codesBatch*m)
		sets, err := u.Router.AssignCodes(ctx, codes[i:j], m)
		if err != nil {
			return nil, err
		}
		out = append(out, sets...)
	}
	return out, nil
}

// check is one posting SPFresh's reassign checks: vector k of list members (its code, location
// and holder there), and the heads its conditions compare.
type check struct {
	from  *listMembers
	k     int
	split int       // the split it is checked for
	list  uint32    // the list it is checked in
	heads [4]uint32 // old head, the two new heads, its own list's head
	child bool      // a member of the split's child, else of a neighboring list
}

// reassign runs SPFresh's reassign for the splits mj decided (their members sl, each member's
// side) and for its merges (pairs), on the router's edited graph: see PolicySPFresh. The moved
// vectors go into their new sets and out of every other list (relocate).
func (u *Updater) reassign(ctx context.Context, t *epoch.Table, mj *Major, sl []listMembers, sides [][]uint8,
	pairs []spMerge) error {
	st := mj.Reassign
	if st == nil {
		st = &ReassignStats{}
		mj.Reassign = st
	}
	if len(mj.Splits) == 0 && len(pairs) == 0 {
		return nil
	}
	removed := map[uint32]bool{}
	for _, pe := range mj.Parts {
		for _, l := range pe.Removed {
			removed[l] = true
		}
	}
	// The postings the plan adds so far: the children's, and those a merge appends.
	planned := map[uint32]map[uint32]bool{}
	for _, pe := range mj.Parts {
		for i, l := range pe.Lists {
			if planned[l] == nil {
				planned[l] = map[uint32]bool{}
			}
			planned[l][pe.IDs[i]] = true
		}
	}

	// The neighboring lists: those whose heads are nearest each old head, live and neither
	// created nor removed by this epoch, each read once from its owner.
	neighbors := make([][]uint32, len(mj.Splits))
	read := map[uint32]*listMembers{}
	if u.ReassignRange > 0 && len(mj.Splits) > 0 {
		parents := make([]uint32, len(mj.Splits))
		for i, s := range mj.Splits {
			parents[i] = s.List
		}
		nb, err := u.Router.HeadNeighbors(ctx, parents, u.ReassignRange, max(u.ReassignRange, 128))
		if err != nil {
			return fmt.Errorf("update: the lists near the split ones: %w", err)
		}
		byOwner := map[placement.NodeID][]uint32{}
		for i := range mj.Splits {
			for _, l := range nb[i] {
				if _, created := mj.Lists[l]; created || removed[l] {
					continue
				}
				neighbors[i] = append(neighbors[i], l)
				if _, ok := read[l]; !ok {
					read[l] = nil
					o := t.Owner(t.PartitionOf(l, u.Manifest))
					byOwner[o] = append(byOwner[o], l)
				}
			}
		}
		owners := make([]placement.NodeID, 0, len(byOwner))
		for o := range byOwner {
			owners = append(owners, o)
		}
		got := make([][]listMembers, len(owners))
		if err := parallel(ctx, len(owners), func(ctx context.Context, i int) error {
			for _, l := range byOwner[owners[i]] {
				lm, m, err := u.readList(ctx, t, l)
				if err != nil {
					return err
				}
				if m != 0 && m != mj.M {
					return fmt.Errorf("update: list %d has %d-byte PQ codes, the split lists %d", l, m, mj.M)
				}
				got[i] = append(got[i], lm)
			}
			return nil
		}); err != nil {
			return err
		}
		for i, o := range owners {
			for j, l := range byOwner[o] {
				read[l] = &got[i][j]
			}
		}
		st.Lists = len(read)
	}

	// Each split's checks in SPFresh's order (CollectReAssign): its children's members, then the
	// members of the lists nearest its old head, nearest first. A vector in a child of this
	// epoch's splits is checked in that child only.
	inChild := map[uint32]bool{}
	for i := range mj.Splits {
		for _, id := range sl[i].IDs {
			inChild[id] = true
		}
	}
	var checks []check
	for i, s := range mj.Splits {
		for k := range sl[i].IDs {
			c := s.Children[sides[i][k]]
			checks = append(checks, check{from: &sl[i], k: k, split: i, list: c, child: true,
				heads: [4]uint32{s.List, s.Children[0], s.Children[1], c}})
		}
		for _, l := range neighbors[i] {
			lm := read[l]
			for k, id := range lm.IDs {
				if !inChild[id] {
					checks = append(checks, check{from: lm, k: k, split: i, list: l,
						heads: [4]uint32{s.List, s.Children[0], s.Children[1], l}})
				}
			}
		}
	}

	// SPFresh's conditions, on the distances from each checked posting's code to the old head,
	// the new heads and its own list's head. A split queues a vector once, as SPFresh's code
	// does: the first list where it meets them is the one its reassign checks.
	type cand struct {
		code  []byte
		lists []uint32 // the list each split queued it in
	}
	cands := map[uint32]*cand{}
	var order []uint32 // candidate vectors, in the order first met
	queued := make([]map[uint32]bool, len(mj.Splits))
	for i := range queued {
		queued[i] = map[uint32]bool{}
	}
	for i := 0; i < len(checks); {
		var batch []check
		for ; i < len(checks) && len(batch) < codesBatch; i++ {
			if c := checks[i]; !queued[c.split][c.from.IDs[c.k]] {
				batch = append(batch, c)
			}
		}
		if len(batch) == 0 {
			break
		}
		codes := make([]byte, 0, len(batch)*mj.M)
		heads := make([]uint32, 0, 4*len(batch))
		for _, c := range batch {
			codes = append(codes, c.from.code(c.k, mj.M)...)
			heads = append(heads, c.heads[:]...)
		}
		d, err := u.Router.HeadDistances(ctx, codes, mj.M, heads, 4)
		if err != nil {
			return fmt.Errorf("update: distances to the split heads: %w", err)
		}
		for j, c := range batch {
			id := c.from.IDs[c.k]
			if queued[c.split][id] {
				continue // queued by a check before it in this batch
			}
			st.Checked++
			dd := d[4*j : 4*j+4]
			ok := neighborCandidate(dd[0], dd[1], dd[2], dd[3])
			if c.child {
				ok = childCandidate(dd[0], dd[3])
			}
			if !ok {
				continue
			}
			queued[c.split][id] = true
			st.Candidates++
			cd := cands[id]
			if cd == nil {
				cd = &cand{code: c.from.code(c.k, mj.M)}
				cands[id] = cd
				order = append(order, id)
			}
			cd.lists = append(cd.lists, c.list)
		}
	}
	// A merge's checks (MergePostings): each posting of the list that went, if it is farther from
	// the kept list's head than from its own old one, queued in the kept list.
	for _, x := range pairs {
		g := x.goneM
		for i := 0; i < len(g.IDs); i += codesBatch {
			j := min(len(g.IDs), i+codesBatch)
			heads := make([]uint32, 0, 2*(j-i))
			for k := i; k < j; k++ {
				heads = append(heads, x.gone, x.kept)
			}
			d, err := u.Router.HeadDistances(ctx, g.codes[i*mj.M:j*mj.M], mj.M, heads, 2)
			if err != nil {
				return fmt.Errorf("update: distances to merged lists' heads: %w", err)
			}
			for k := i; k < j; k++ {
				st.Checked++
				if d[2*(k-i)+1] <= d[2*(k-i)] {
					continue
				}
				id := g.IDs[k]
				st.Candidates++
				cd := cands[id]
				if cd == nil {
					cd = &cand{code: g.code(k, mj.M)}
					cands[id] = cd
					order = append(order, id)
				}
				cd.lists = append(cd.lists, x.kept)
			}
		}
	}
	if len(order) == 0 {
		return nil
	}

	// Each candidate's lists by the rule; it moves if a list it was queued in is not one.
	codes := make([]byte, 0, len(order)*mj.M)
	for _, id := range order {
		codes = append(codes, cands[id].code...)
	}
	sets, err := u.assignCodes(ctx, codes, mj.M)
	if err != nil {
		return fmt.Errorf("update: reassigning: %w", err)
	}
	moved := map[uint32][]uint32{} // vector -> its new set
	for i, id := range order {
		in := map[uint32]bool{}
		for _, l := range sets[i] {
			in[l] = true
		}
		for _, l := range cands[id].lists {
			if !in[l] {
				moved[id] = sets[i]
				break
			}
		}
	}
	st.Moved = len(moved)
	if len(moved) == 0 {
		return nil
	}

	return u.relocate(ctx, t, mj, moved, removed, planned, sl, read, pairs)
}

// relocate moves every vector of moved into its new set (moved's value), adding to mj: into each
// list of the set it is not in, out of each list the set lacks -- every list that holds it at the
// epoch's versions, which the owners find (LISTS_OF), and every list the plan puts it in
// (planned: those postings leave the plan). A list removed by the epoch takes its postings along.
// The PQ codes go to owners that have none of the vector. sl, read and pairs are the lists the
// epoch read (for a posting of each moved vector, and the stats).
func (u *Updater) relocate(ctx context.Context, t *epoch.Table, mj *Major, moved map[uint32][]uint32,
	removed map[uint32]bool, planned map[uint32]map[uint32]bool, sl []listMembers, read map[uint32]*listMembers,
	pairs []spMerge) error {
	st := mj.Reassign
	ids := make([]uint32, 0, len(moved))
	for id := range moved {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	// A posting the epoch read of each: its location, holder and PQ code.
	type posting struct {
		loc    uint32
		holder uint16
		code   []byte
	}
	src := map[uint32]posting{}
	seen := map[uint32]bool{} // the lists the epoch read
	take := func(l uint32, lm *listMembers) {
		seen[l] = true
		for k, id := range lm.IDs {
			if _, ok := moved[id]; !ok {
				continue
			}
			if _, ok := src[id]; !ok {
				src[id] = posting{lm.Locs[k], lm.Holders[k], lm.code(k, mj.M)}
			}
		}
	}
	for i := range sl {
		take(mj.Splits[i].List, &sl[i])
	}
	readLists := make([]uint32, 0, len(read))
	for l := range read {
		readLists = append(readLists, l)
	}
	sort.Slice(readLists, func(i, j int) bool { return readLists[i] < readLists[j] })
	for _, l := range readLists {
		take(l, read[l])
	}
	for _, x := range pairs {
		take(x.gone, x.goneM)
		take(x.kept, x.keptM)
	}
	held, err := u.listsOf(ctx, t, ids)
	if err != nil {
		return err
	}
	plannedIn := map[uint32][]uint32{} // moved vector -> the lists the plan puts it in
	for l, in := range planned {
		for id := range in {
			if _, ok := moved[id]; ok {
				plannedIn[id] = append(plannedIn[id], l)
			}
		}
	}
	before := map[int]bool{}
	for p := range mj.Parts {
		before[p] = true
	}
	touched, staleParts := map[int]bool{}, map[int]bool{}
	unplan := map[uint32]map[uint32]bool{}
	mj.Reassigned = map[uint32][]uint32{}
	for _, id := range ids {
		set, w := moved[id], src[id]
		if w.code == nil && mj.M > 0 {
			return fmt.Errorf("update: vector %d moves, but the epoch read no posting of it", id)
		}
		mj.Reassigned[id] = set
		inSet := map[uint32]bool{}
		for _, l := range set {
			inSet[l] = true
		}
		owners := map[placement.NodeID]bool{} // the nodes that have its PQ code
		var base []uint32                     // the lists that hold it and stay
		for _, l := range held[id] {
			owners[t.Owner(t.PartitionOf(l, u.Manifest))] = true
			if !removed[l] {
				base = append(base, l)
			}
		}
		sort.Slice(base, func(i, j int) bool { return base[i] < base[j] })
		inBase := map[uint32]bool{}
		for _, l := range base {
			inBase[l] = true
		}
		pl := plannedIn[id]
		sort.Slice(pl, func(i, j int) bool { return pl[i] < pl[j] })
		for _, l := range pl {
			if inSet[l] {
				continue
			}
			if unplan[l] == nil {
				unplan[l] = map[uint32]bool{}
			}
			unplan[l][id] = true
			delete(planned[l], id)
			st.Dropped++
		}
		for _, l := range base {
			if inSet[l] {
				continue
			}
			p := t.PartitionOf(l, u.Manifest)
			mj.part(p).drop(l, id)
			st.Dropped++
			if seen[l] {
				touched[p] = true
			} else {
				st.Stale++
				staleParts[p] = true
			}
		}
		for _, l := range set {
			if inBase[l] || planned[l][id] {
				continue
			}
			p := mj.partitionOf(t, u.Manifest, l)
			mj.part(p).add(l, id, w.loc, w.holder)
			touched[p] = true
			st.Added++
			if planned[l] == nil {
				planned[l] = map[uint32]bool{}
			}
			planned[l][id] = true
			if o := t.Owner(p); !owners[o] {
				mj.sendCode(o, id, w.code)
				owners[o] = true
			}
		}
	}
	for p := range staleParts {
		if !before[p] && !touched[p] {
			st.StaleParts++
		}
	}
	// Out of the plan: the postings of vectors that left lists the plan put them in.
	for _, pe := range mj.Parts {
		n := 0
		for i, l := range pe.Lists {
			if unplan[l][pe.IDs[i]] {
				continue
			}
			pe.Lists[n], pe.IDs[n], pe.Locs[n], pe.Holders[n] = l, pe.IDs[i], pe.Locs[i], pe.Holders[i]
			n++
		}
		pe.Lists, pe.IDs, pe.Locs, pe.Holders = pe.Lists[:n], pe.IDs[:n], pe.Locs[:n], pe.Holders[:n]
	}
	for i := range mj.Splits {
		s := &mj.Splits[i]
		for c := 0; c < 2; c++ {
			s.Sizes[c] = len(planned[s.Children[c]])
		}
	}
	return nil
}

// listsOf is every list that holds each of the vectors ids at t's versions, from the owners
// (LISTS_OF), each over all its partitions.
func (u *Updater) listsOf(ctx context.Context, t *epoch.Table, ids []uint32) (map[uint32][]uint32, error) {
	byOwner := map[placement.NodeID][]int{}
	for p := 0; p < u.Manifest.NumPartitions; p++ {
		byOwner[t.Owner(p)] = append(byOwner[t.Owner(p)], p)
	}
	owners := make([]placement.NodeID, 0, len(byOwner))
	for o := range byOwner {
		owners = append(owners, o)
	}
	got := make([][]nodeclient.Posting, len(owners))
	if err := parallel(ctx, len(owners), func(ctx context.Context, i int) error {
		ps := byOwner[owners[i]]
		vs := make([]uint32, len(ps))
		for k, p := range ps {
			vs[k] = t.Version(p)
		}
		x, err := u.client(t.Nodes[owners[i]].NodeAddr).ListsOf(ctx, ps, vs, ids)
		if err != nil {
			return fmt.Errorf("update: the lists of the moved vectors on node %d: %w", owners[i], err)
		}
		got[i] = x
		return nil
	}); err != nil {
		return nil, err
	}
	out := map[uint32][]uint32{}
	for _, x := range got {
		for _, p := range x {
			out[p.ID] = append(out[p.ID], p.List)
		}
	}
	return out, nil
}

// spMerge is one of SPFresh's merges: of a list due to merge and its partner, the one that goes
// and the one that takes its postings, with their members as read.
type spMerge struct {
	gone, kept   uint32
	goneM, keptM *listMembers
}

// mergeNeighbors is how many of the heads nearest its own a list due to merge looks among for
// its partner (SPFresh's InternalResultNum).
const mergeNeighbors = 64

// noList: no list (Merge.Into of the threshold policy's merges).
const noList = math.MaxUint32

// spfreshMerges pairs each list of merges (its members in ml) with its partner: the first of the
// mergeNeighbors heads nearest its own on the router's graph whose list is not split, due to
// merge or another merge's partner in the epoch, and keeps the two under PostingLimit (0: no
// limit). The shorter of the two goes (the list due to merge on a tie).
func (u *Updater) spfreshMerges(ctx context.Context, t *epoch.Table, mj *Major, splits, merges []uint32,
	ml []listMembers) ([]spMerge, error) {
	st := mj.Reassign
	if st == nil {
		st = &ReassignStats{}
		mj.Reassign = st
	}
	if len(merges) == 0 {
		return nil, nil
	}
	nb, err := u.Router.HeadNeighbors(ctx, merges, mergeNeighbors, 2*mergeNeighbors)
	if err != nil {
		return nil, fmt.Errorf("update: the lists near the ones to merge: %w", err)
	}
	var lengths map[uint32]int
	if u.PostingLimit > 0 {
		if lengths, err = u.listLengths(ctx, t); err != nil {
			return nil, err
		}
	}
	taken := map[uint32]bool{}
	for _, l := range append(append([]uint32(nil), splits...), merges...) {
		taken[l] = true
	}
	var out []spMerge
	for i, l := range merges {
		a := &ml[i]
		partner := uint32(noList)
		for _, b := range nb[i] {
			if !taken[b] && (u.PostingLimit <= 0 || len(a.IDs)+lengths[b] < u.PostingLimit) {
				partner = b
				break
			}
		}
		if partner == noList {
			st.NoPartner++
			continue
		}
		taken[partner] = true
		bm, m, err := u.readList(ctx, t, partner)
		if err != nil {
			return nil, err
		}
		if m > 0 {
			if mj.M > 0 && m != mj.M {
				return nil, fmt.Errorf("update: list %d has %d-byte PQ codes, the others %d", partner, m, mj.M)
			}
			mj.M = m
		}
		if len(a.IDs) > len(bm.IDs) {
			out = append(out, spMerge{gone: partner, kept: l, goneM: &bm, keptM: a})
		} else {
			out = append(out, spMerge{gone: l, kept: partner, goneM: a, keptM: &bm})
		}
	}
	st.Merged = len(out)
	return out, nil
}

// planMerges appends each merge's list that goes to the one kept: the postings the kept list
// lacks (a vector in both keeps one), their PQ codes sent along where the kept list's owner is
// another node; the list that goes is removed.
func (u *Updater) planMerges(t *epoch.Table, mj *Major, pairs []spMerge) {
	for _, x := range pairs {
		g, k := x.goneM, x.keptM
		pg := mj.part(g.p)
		pg.Removed = append(pg.Removed, x.gone)
		mj.Merges = append(mj.Merges, Merge{List: x.gone, Partition: g.p, Postings: len(g.IDs), Into: x.kept})
		has := map[uint32]bool{}
		for _, id := range k.IDs {
			has[id] = true
		}
		for i, id := range g.IDs {
			if has[id] {
				continue
			}
			mj.part(k.p).add(x.kept, id, g.Locs[i], g.Holders[i])
			if to := t.Owner(k.p); to != t.Owner(g.p) {
				mj.sendCode(to, id, g.code(i, mj.M))
			}
		}
	}
}
