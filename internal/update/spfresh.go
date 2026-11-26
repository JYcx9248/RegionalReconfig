package update

import (
	"context"
	"fmt"
	"sort"

	"rtier/internal/epoch"
	"rtier/internal/placement"
)

// Policies of a major epoch (Updater.Policy). Both split a long list by a balanced 2-means into
// two children in its partition, each member in one child, and remove a short list.
//
//   - threshold (the default): a merged list's postings go to the list of their nearest live head;
//     nothing else moves.
//   - spfresh: SPFresh's LIRE (SOSP '23) as its code does it (github.com/SPFresh/SPFresh,
//     ExtraDynamicSearcher.h: Split's CollectReAssign, CheckIsNeedReassign, Reassign). After the
//     splits, each split checks the vectors near it in SPFresh's order: the members of its
//     children, then those of the ReassignRange lists whose heads are nearest the old head
//     (SPFresh's ReassignK), nearest first. A split queues a vector once, in the first list where
//     one of SPFresh's conditions holds, and the vector is inserted again by the index's own rule
//     (ASSIGN's, Assign in node_engine.cpp) if that list is no longer among the lists the rule
//     gives it; then it joins every list of its new set and leaves every list it was seen in
//     that is not in the set. A vector in a split's child is checked in that child only: one
//     that stays there does not take the other child too. A merged list's postings go into
//     every list the rule gives them.
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
// the code there on an RNG index is open; a reassign there bumps the vector's version, so that every old
// copy is dropped when its posting is next split or merged, here the copies outside the lists
// read stay (still found by searches, as SPFresh's stale copies are until then); its merge
// appends a short posting to its nearest neighbor, ours reassigns the postings by the rule; and
// it acts on every insert, we once per batch.
const (
	PolicyThreshold = "threshold"
	PolicySPFresh   = "spfresh"
)

// ReassignStats is what SPFresh's reassign did in one major epoch.
type ReassignStats struct {
	Lists      int `json:"lists"`      // neighboring lists read, each once
	Checked    int `json:"checked"`    // postings checked against SPFresh's conditions
	Candidates int `json:"candidates"` // vectors queued: by each split, the first posting that met them
	Moved      int `json:"moved"`      // vectors inserted again
	Added      int `json:"added"`      // postings planned for them, in lists they were not seen in (a list not read may hold one already: staging keeps one)
	Dropped    int `json:"dropped"`    // postings they left
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
// side), on the router's edited graph: see PolicySPFresh. It adds to mj the postings the moved
// vectors gain and drops the ones they leave -- in the children it takes them out of the plan --
// and the PQ codes owners that lacked them need.
func (u *Updater) reassign(ctx context.Context, t *epoch.Table, mj *Major, sl []listMembers, sides [][]uint8) error {
	st := &ReassignStats{}
	mj.Reassign = st
	if len(mj.Splits) == 0 {
		return nil
	}
	removed := map[uint32]bool{}
	for _, pe := range mj.Parts {
		for _, l := range pe.Removed {
			removed[l] = true
		}
	}
	// The postings the plan adds so far: the children's, and the merged lists'.
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
	if u.ReassignRange > 0 {
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
	if len(order) == 0 {
		return nil
	}

	// Each candidate's lists by the rule; it moves if a list a split queued it in is not one.
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

	// Where each moved vector was seen -- the lists read (the base version) and the created lists
	// the plan fills -- with its location, holder and code from one of its postings; and which
	// owners have its code: those of the lists it was read in or planned for from its parent.
	type copies struct {
		loc    uint32
		holder uint16
		code   []byte
		lists  []uint32
		owners map[placement.NodeID]bool
	}
	where := map[uint32]*copies{}
	note := func(lm *listMembers, k int, list uint32, p int) {
		id := lm.IDs[k]
		if _, ok := moved[id]; !ok {
			return
		}
		w := where[id]
		if w == nil {
			w = &copies{loc: lm.Locs[k], holder: lm.Holders[k], code: lm.code(k, mj.M), owners: map[placement.NodeID]bool{}}
			where[id] = w
		}
		w.lists = append(w.lists, list)
		w.owners[t.Owner(p)] = true
	}
	for i, s := range mj.Splits {
		for k := range sl[i].IDs {
			note(&sl[i], k, s.Children[sides[i][k]], s.Partition)
		}
	}
	lists := make([]uint32, 0, len(read))
	for l := range read {
		lists = append(lists, l)
	}
	sort.Slice(lists, func(i, j int) bool { return lists[i] < lists[j] })
	for _, l := range lists {
		lm := read[l]
		for k := range lm.IDs {
			note(lm, k, l, t.PartitionOf(l, u.Manifest))
		}
	}

	// The edits: every list of the new set it was not seen in, and out of every list it was
	// seen in that the set lacks. In a created list, out of the plan.
	unplan := map[uint32]map[uint32]bool{} // created list -> vectors taken out of it
	ids := make([]uint32, 0, len(moved))
	for id := range moved {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	for _, id := range ids {
		set, w := moved[id], where[id]
		inSet := map[uint32]bool{}
		for _, l := range set {
			inSet[l] = true
		}
		seen := map[uint32]bool{}
		for _, l := range w.lists {
			seen[l] = true
			if inSet[l] {
				continue
			}
			st.Dropped++
			if _, created := mj.Lists[l]; created {
				if unplan[l] == nil {
					unplan[l] = map[uint32]bool{}
				}
				unplan[l][id] = true
				delete(planned[l], id)
				continue
			}
			mj.part(t.PartitionOf(l, u.Manifest)).drop(l, id)
		}
		for _, l := range set {
			if seen[l] || planned[l][id] {
				continue
			}
			p := mj.partitionOf(t, u.Manifest, l)
			mj.part(p).add(l, id, w.loc, w.holder)
			st.Added++
			if planned[l] == nil {
				planned[l] = map[uint32]bool{}
			}
			planned[l][id] = true
			if o := t.Owner(p); !w.owners[o] {
				mj.sendCode(o, id, w.code)
			}
		}
	}
	// Out of the plan: the created lists' postings of vectors that left them.
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
