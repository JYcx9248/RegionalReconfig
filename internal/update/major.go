package update

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"rtier/internal/epoch"
	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
	"rtier/internal/placement"
	"rtier/internal/protocol"
)

// Major epochs (U14): posting lists split and merged, and the navigation graph patched to match.
// A major epoch is a batch with its own lease: the update controller decides it on the lists'
// PQ codes -- no raw vector is read but the new heads' -- and on its own graph, and the data
// nodes execute it with the mechanisms reconfigurations use:
//
//   - a split replaces a list by two in the same partition: a balanced 2-means of its members'
//     PQ codes, each child's head the member closest to its mean, as the build chooses heads;
//     every member goes to its side's child, and the parent's head is marked deleted (it stays
//     a routing point, never a result);
//   - a merge removes a list and its head; each member goes to the list of its nearest live
//     head on the edited graph, in whatever partition. A posting that changes owner moves as
//     (vector ID, location, holder) with its PQ code: its raw vector follows on demand, one
//     hop, from the node that holds it;
//   - the graph: the router (the update controller's own rtier_node) inserts the new heads and
//     marks the removed ones deleted, and exports the bytes that changed, a patch. Every entry
//     stages it next to its graph and merges it after the grace period (the controller's
//     CommitUpdate), so no node ever holds two versions of the graph.

// MajorDue picks the lists a major epoch changes, from the list lengths the owners reported
// when a batch was staged (Batch.Lengths): lists longer than splitLen split, the longest first;
// lists shorter than mergeLen merge, the shortest first (an empty list too); at most max in all
// (0: no limit). A threshold of 0 turns that kind off.
func MajorDue(lengths map[uint32]uint32, splitLen, mergeLen uint32, max int) (splits, merges []uint32) {
	type lc struct{ list, n uint32 }
	var long, short []lc
	for l, n := range lengths {
		switch {
		case splitLen > 0 && n > splitLen && n >= 2:
			long = append(long, lc{l, n})
		case mergeLen > 0 && n < mergeLen:
			short = append(short, lc{l, n})
		}
	}
	sort.Slice(long, func(i, j int) bool {
		return long[i].n > long[j].n || long[i].n == long[j].n && long[i].list < long[j].list
	})
	sort.Slice(short, func(i, j int) bool {
		return short[i].n < short[j].n || short[i].n == short[j].n && short[i].list < short[j].list
	})
	for _, x := range long {
		if max > 0 && len(splits) >= max {
			break
		}
		splits = append(splits, x.list)
	}
	for _, x := range short {
		if max > 0 && len(splits)+len(merges) >= max {
			break
		}
		merges = append(merges, x.list)
	}
	return splits, merges
}

// Split is a list replaced by two, in its partition.
type Split struct {
	List      uint32
	Partition int
	Children  [2]uint32 // the new lists
	Heads     [2]uint32 // their heads' vector IDs
	Sizes     [2]int    // their postings
}

// Merge is a list removed, its postings sent to their nearest heads' lists.
type Merge struct {
	List      uint32
	Partition int
	Postings  int
}

// PartEdits is what the next version of one partition changes (nodeclient.Edits, without the
// deletes).
type PartEdits struct {
	Created, Removed []uint32
	Lists, IDs, Locs []uint32
	Holders          []uint16
}

func (e *PartEdits) add(list, id, loc uint32, holder uint16) {
	e.Lists, e.IDs, e.Locs, e.Holders = append(e.Lists, list), append(e.IDs, id), append(e.Locs, loc), append(e.Holders, holder)
}

// CodeSet is PQ codes (M bytes each) of the vectors IDs.
type CodeSet struct {
	IDs   []uint32
	Codes []byte
}

// Major is a decided major epoch. Its decisions are data -- the edits of every partition it
// changes and the graph patch -- so that whoever applies it applies the same one: the owners,
// and the oracle of the end-to-end tests.
type Major struct {
	Lease  *protocol.UpdateLease
	Graph  uint32 // the graph version the patch makes
	Patch  []byte // from the lease epoch's graph version to Graph
	M      int    // PQ code bytes
	Splits []Split
	Merges []Merge
	Lists  map[uint32]int     // the lists it creates, and their partitions
	Parts  map[int]*PartEdits // by partition
	// Codes: by node, the PQ codes of the postings that move to it from another owner.
	Codes map[placement.NodeID]*CodeSet
	// Staged: by owner, the next version of each partition it staged (ExecuteMajor).
	Staged  map[placement.NodeID][]protocol.PartitionVersion
	patched []placement.NodeID // entries that staged the patch
}

func (mj *Major) part(p int) *PartEdits {
	e, ok := mj.Parts[p]
	if !ok {
		e = &PartEdits{}
		mj.Parts[p] = e
	}
	return e
}

// Versions is every version the major epoch staged, by partition.
func (mj *Major) Versions() []protocol.PartitionVersion {
	var out []protocol.PartitionVersion
	for _, pvs := range mj.Staged {
		out = append(out, pvs...)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Partition < out[j].Partition })
	return out
}

// Edits is the nodeclient form of partition p's edits.
func (mj *Major) Edits(p int) nodeclient.Edits {
	pe := mj.Parts[p]
	e := nodeclient.Edits{Lists: pe.Lists, IDs: pe.IDs, Locs: pe.Locs, Holders: pe.Holders, Removed: pe.Removed}
	for _, c := range pe.Created {
		e.Created = append(e.Created, nodeclient.CreatedList{List: c, Partition: p})
	}
	return e
}

// PatchPath is where dir keeps the patch that makes graph version v of index generation gen
// (graph versions count within a generation: a full rebuild brings a graph of its own).
func PatchPath(dir string, gen, v uint32) string {
	if gen > 0 {
		dir = filepath.Join(dir, fmt.Sprintf("gen-%d", gen))
	}
	return filepath.Join(dir, fmt.Sprintf("patch-%d.bin", v))
}

// follow makes the updater work on t's index generation (U14 full rebuild): the generation's
// manifest, and list IDs counted above its lists. Generations only move forward.
func (u *Updater) follow(t *epoch.Table) error {
	u.Alloc.generation(t.Generation)
	if t.Generation == u.gen {
		return nil
	}
	if t.Generation < u.gen {
		return fmt.Errorf("update: epoch %d runs index generation %d, the updater %d", t.Epoch, t.Generation, u.gen)
	}
	m, err := partitioning.LoadManifest(t.GenParts)
	if err != nil {
		return fmt.Errorf("update: index generation %d: %w", t.Generation, err)
	}
	u.Manifest, u.gen = m, t.Generation
	return nil
}

// PrepareMajor begins a major epoch that splits the lists splits and merges the lists merges:
// it takes the controller's lease, brings the router's graph to the epoch's version, decides
// the splits and merges, and edits the router's graph, keeping the patch. If it fails, it gives
// the lease up and brings the router back.
func (u *Updater) PrepareMajor(ctx context.Context, splits, merges []uint32) (mj *Major, err error) {
	if len(splits)+len(merges) == 0 {
		return nil, errors.New("update: a major epoch needs a list to split or merge")
	}
	seen := map[uint32]bool{}
	for _, l := range append(append([]uint32(nil), splits...), merges...) {
		if seen[l] {
			return nil, fmt.Errorf("update: list %d is named twice", l)
		}
		seen[l] = true
	}
	if u.Router == nil {
		return nil, errors.New("update: no router: a major epoch edits the update controller's own navigation graph")
	}
	lease, err := u.Controller.BeginUpdate(ctx, protocol.BeginUpdateReq{})
	if err != nil {
		return nil, err
	}
	edited := false
	defer func() {
		if err != nil {
			if edited {
				err = errors.Join(err, u.syncRouter(context.WithoutCancel(ctx), lease.Table))
			}
			err = errors.Join(err, u.abort(ctx, lease))
		}
	}()
	t := lease.Table
	if err := u.follow(t); err != nil {
		return nil, err
	}
	if err := u.syncRouter(ctx, t); err != nil {
		return nil, err
	}
	mj = &Major{Lease: lease, Graph: t.Graph + 1, Lists: map[uint32]int{}, Parts: map[int]*PartEdits{},
		Codes: map[placement.NodeID]*CodeSet{}, Staged: map[placement.NodeID][]protocol.PartitionVersion{}}

	// The members of every list the epoch changes, as their owners hold them: postings with
	// their holders, and PQ codes.
	type members struct {
		p     int
		lp    nodeclient.ListPostings
		codes []byte
	}
	read := func(l uint32) (members, error) {
		if _, created := t.Lists[l]; !created && int(l) >= u.Manifest.NumLists {
			return members{}, fmt.Errorf("update: list %d does not exist", l)
		}
		p := t.PartitionOf(l, u.Manifest)
		nc := u.client(t.Nodes[t.Owner(p)].NodeAddr)
		lp, err := nc.ListGet(ctx, l, t.Version(p))
		if err != nil {
			return members{}, fmt.Errorf("update: list %d of partition %d: %w", l, p, err)
		}
		out := members{p: p, lp: lp}
		if len(lp.IDs) > 0 {
			m, codes, err := nc.PQGet(ctx, lp.IDs)
			if err != nil {
				return members{}, fmt.Errorf("update: the PQ codes of list %d: %w", l, err)
			}
			mj.M, out.codes = m, codes
		}
		return out, nil
	}
	sl := make([]members, len(splits))
	for i, l := range splits {
		if sl[i], err = read(l); err != nil {
			return nil, err
		}
		if len(sl[i].lp.IDs) < 2 {
			return nil, fmt.Errorf("update: list %d holds %d postings, too few to split", l, len(sl[i].lp.IDs))
		}
	}
	ml := make([]members, len(merges))
	for i, l := range merges {
		if ml[i], err = read(l); err != nil {
			return nil, err
		}
	}

	// Splits: two children per list, in its partition.
	children := u.Alloc.TakeLists(2*len(splits), listFloor(t, u.Manifest))
	var labels, dead []uint32
	var vecs []byte
	for i, l := range splits {
		s := sl[i]
		side, heads, err := u.Router.SplitList(ctx, s.codes, mj.M)
		if err != nil {
			return nil, fmt.Errorf("update: splitting list %d: %w", l, err)
		}
		sp := Split{List: l, Partition: s.p, Children: [2]uint32{children[2*i], children[2*i+1]}}
		pe := mj.part(s.p)
		pe.Removed = append(pe.Removed, l)
		for c := 0; c < 2; c++ {
			j := heads[c]
			sp.Heads[c] = s.lp.IDs[j]
			v, err := u.rawVector(ctx, t, s.lp.Locs[j], s.lp.Holders[j])
			if err != nil {
				return nil, fmt.Errorf("update: the head of list %d's child: %w", l, err)
			}
			labels, vecs = append(labels, sp.Children[c]), append(vecs, v...)
			pe.Created = append(pe.Created, sp.Children[c])
			mj.Lists[sp.Children[c]] = s.p
		}
		for k := range s.lp.IDs {
			pe.add(sp.Children[side[k]], s.lp.IDs[k], s.lp.Locs[k], s.lp.Holders[k])
			sp.Sizes[side[k]]++
		}
		mj.Splits = append(mj.Splits, sp)
		dead = append(dead, l)
	}
	dead = append(dead, merges...)

	// The graph: the new heads in, the removed ones marked deleted.
	if mj.Patch, err = u.Router.EditGraph(ctx, mj.Graph, labels, vecs, dead); err != nil {
		return nil, fmt.Errorf("update: editing the graph: %w", err)
	}
	edited = true

	// Merges: every posting to the list of its nearest live head on the edited graph.
	ef := u.MergeEF
	if ef <= 0 {
		ef = 64
	}
	for i, l := range merges {
		s := ml[i]
		pe := mj.part(s.p)
		pe.Removed = append(pe.Removed, l)
		mj.Merges = append(mj.Merges, Merge{List: l, Partition: s.p, Postings: len(s.lp.IDs)})
		if len(s.lp.IDs) == 0 {
			continue
		}
		targets, err := u.Router.NearestHeads(ctx, s.codes, mj.M, ef)
		if err != nil {
			return nil, fmt.Errorf("update: merging list %d: %w", l, err)
		}
		for k, tl := range targets {
			tp, created := mj.Lists[tl]
			if !created {
				tp = t.PartitionOf(tl, u.Manifest)
			}
			mj.part(tp).add(tl, s.lp.IDs[k], s.lp.Locs[k], s.lp.Holders[k])
			if to := t.Owner(tp); to != t.Owner(s.p) {
				cs, ok := mj.Codes[to]
				if !ok {
					cs = &CodeSet{}
					mj.Codes[to] = cs
				}
				cs.IDs = append(cs.IDs, s.lp.IDs[k])
				cs.Codes = append(cs.Codes, s.codes[k*mj.M:(k+1)*mj.M]...)
			}
		}
	}
	return mj, nil
}

// ExecuteMajor has the owners stage mj's lists and every entry its graph patch, keeps the
// patch, and has the controller flip to it; the controller merges the patch on the entries
// after the grace period. If the staging fails, it drops what was staged, gives the lease up
// and brings the router back to the epoch's graph.
func (u *Updater) ExecuteMajor(ctx context.Context, mj *Major) (*protocol.UpdateReply, error) {
	err := u.stageMajor(ctx, mj)
	if err == nil && u.PatchDir != "" {
		// Kept before the commit: a router started later is brought to the committed version
		// from these files.
		err = savePatch(PatchPath(u.PatchDir, mj.Lease.Table.Generation, mj.Graph), mj.Patch)
	}
	if err != nil {
		u.undoMajor(ctx, mj)
		return nil, errors.Join(err, u.abort(ctx, mj.Lease))
	}
	// If the commit fails before the flip, the router is ahead of the epoch: the next batch's
	// syncRouter brings it back.
	return u.Controller.CommitUpdate(ctx, protocol.CommitUpdateReq{Token: mj.Lease.Token,
		FromEpoch: mj.Lease.Table.Epoch, Versions: mj.Versions(), Graph: mj.Graph, Lists: mj.Lists})
}

// AbortMajor gives up a major epoch PrepareMajor decided and ExecuteMajor did not run: the
// router goes back to the epoch's graph and the lease is given up. The list IDs it took are not
// handed out again.
func (u *Updater) AbortMajor(ctx context.Context, mj *Major) error {
	u.undoMajor(ctx, mj)
	return u.abort(ctx, mj.Lease)
}

func (u *Updater) stageMajor(ctx context.Context, mj *Major) error {
	t := mj.Lease.Table
	// 1. The PQ codes of the postings that move to another owner.
	var targets []placement.NodeID
	for o := range mj.Codes {
		targets = append(targets, o)
	}
	if err := parallel(ctx, len(targets), func(ctx context.Context, i int) error {
		cs := mj.Codes[targets[i]]
		if _, _, err := u.client(t.Nodes[targets[i]].NodeAddr).PQPut(ctx, cs.IDs, mj.M, cs.Codes); err != nil {
			return fmt.Errorf("update: PQ codes to node %d: %w", targets[i], err)
		}
		return nil
	}); err != nil {
		return err
	}
	// 2. Every owner stages the next version of its partitions the epoch changes, in one call:
	// all or nothing on that node.
	byOwner := map[placement.NodeID][]int{}
	for p := range mj.Parts {
		byOwner[t.Owner(p)] = append(byOwner[t.Owner(p)], p)
	}
	var owners []placement.NodeID
	for o := range byOwner {
		owners = append(owners, o)
	}
	var mu sync.Mutex
	if err := parallel(ctx, len(owners), func(ctx context.Context, i int) error {
		o := owners[i]
		ps := byOwner[o]
		sort.Ints(ps)
		var vs []nodeclient.VersionStage
		var e nodeclient.Edits
		for _, p := range ps {
			v := t.Version(p)
			vs = append(vs, nodeclient.VersionStage{Partition: p, Base: v, Next: v + 1})
			pe := mj.Edits(p)
			e.Lists, e.IDs, e.Locs = append(e.Lists, pe.Lists...), append(e.IDs, pe.IDs...), append(e.Locs, pe.Locs...)
			e.Holders = append(e.Holders, pe.Holders...)
			e.Created, e.Removed = append(e.Created, pe.Created...), append(e.Removed, pe.Removed...)
		}
		res, err := u.client(t.Nodes[o].NodeAddr).StageVersions(ctx, vs, e)
		if err != nil {
			return fmt.Errorf("update: staging partitions %v on node %d: %w", ps, o, err)
		}
		mu.Lock()
		for k, r := range res {
			if r.Changed {
				mj.Staged[o] = append(mj.Staged[o], protocol.PartitionVersion{Partition: r.Partition, Version: vs[k].Next})
			}
		}
		mu.Unlock()
		return nil
	}); err != nil {
		return err
	}
	// 3. Every entry stages the patch next to its graph.
	return parallel(ctx, len(t.Entries), func(ctx context.Context, i int) error {
		id := t.Entries[i]
		if err := u.client(t.Nodes[id].NodeAddr).PatchStage(ctx, mj.Patch); err != nil {
			return fmt.Errorf("update: the graph patch on node %d: %w", id, err)
		}
		mu.Lock()
		mj.patched = append(mj.patched, id)
		mu.Unlock()
		return nil
	})
}

// undoMajor drops what stageMajor put on the nodes, under the lease: the staged versions and
// patches, the PQ codes staged for the moved postings (codes the new versions referenced are
// cached once those go), and brings the router back to the epoch's graph.
func (u *Updater) undoMajor(ctx context.Context, mj *Major) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), time.Minute)
	defer cancel()
	t := mj.Lease.Table
	for o, pvs := range mj.Staged {
		for _, pv := range pvs {
			if _, err := u.client(t.Nodes[o].NodeAddr).DropVersion(ctx, pv.Partition, pv.Version); err != nil {
				log.Printf("update: undo: dropping partition %d at version %d on node %d: %v", pv.Partition, pv.Version, o, err)
			}
		}
	}
	for _, id := range mj.patched {
		if _, err := u.client(t.Nodes[id].NodeAddr).PatchDrop(ctx, mj.Graph); err != nil {
			log.Printf("update: undo: dropping the graph patch on node %d: %v", id, err)
		}
	}
	for o := range mj.Codes {
		if _, err := u.client(t.Nodes[o].NodeAddr).PQRelease(ctx); err != nil {
			log.Printf("update: undo: releasing PQ codes on node %d: %v", o, err)
		}
	}
	if err := u.syncRouter(ctx, t); err != nil {
		log.Printf("update: undo: the router's graph: %v", err)
	}
}

// syncRouter brings the router to t's index generation and graph version: the generation's
// graph after a full rebuild (U14), then the patches up to t.Graph from PatchDir, after
// reloading the generation's graph (GraphPath for the deployed build) if the router is ahead --
// an edit of a major epoch that did not commit.
func (u *Updater) syncRouter(ctx context.Context, t *epoch.Table) error {
	in, err := u.Router.Info(ctx)
	if err != nil {
		return fmt.Errorf("update: router: %w", err)
	}
	want := t.Graph
	graph := u.GraphPath
	if t.Generation > 0 {
		graph = filepath.Join(t.GenIndex, "graph.hnsw")
	}
	have := in.Graph.Version
	switch {
	case in.Generation != t.Generation:
		if in.Generation > t.Generation {
			return fmt.Errorf("update: the router runs index generation %d, epoch %d generation %d", in.Generation, t.Epoch, t.Generation)
		}
		if err := u.Router.SwitchGeneration(ctx, t.Generation, t.GenIndex, t.GenParts); err != nil {
			return fmt.Errorf("update: moving the router to index generation %d: %w", t.Generation, err)
		}
		if err := u.Router.LoadGraph(ctx, graph); err != nil {
			return fmt.Errorf("update: loading index generation %d's graph on the router: %w", t.Generation, err)
		}
		have = 0
	case have == want && in.Graph.Patch == 0:
		return nil
	case have > want || in.Graph.Patch != 0:
		if graph == "" {
			return fmt.Errorf("update: the router's graph is at version %d, the epoch's at %d, and there is no graph to reload", have, want)
		}
		if err := u.Router.LoadGraph(ctx, graph); err != nil {
			return fmt.Errorf("update: reloading the router's graph: %w", err)
		}
		have = 0
	}
	for v := have + 1; v <= want; v++ {
		if u.PatchDir == "" {
			return fmt.Errorf("update: the router's graph is at version %d, the epoch's at %d, and there is no patch directory", have, want)
		}
		path, err := filepath.Abs(PatchPath(u.PatchDir, t.Generation, v))
		if err != nil {
			return err
		}
		if err := u.Router.PatchLoad(ctx, path); err != nil {
			return fmt.Errorf("update: bringing the router's graph to version %d: %w", want, err)
		}
		if err := u.Router.PatchMerge(ctx, v); err != nil {
			return fmt.Errorf("update: bringing the router's graph to version %d: %w", want, err)
		}
	}
	return nil
}

// rawVector reads the vector at loc from the node that holds it.
func (u *Updater) rawVector(ctx context.Context, t *epoch.Table, loc uint32, holder uint16) ([]byte, error) {
	n, ok := t.Nodes[placement.NodeID(holder)]
	if !ok {
		return nil, fmt.Errorf("the vector at location %d is held by node %d, which is not in epoch %d", loc, holder, t.Epoch)
	}
	_, v, err := u.client(n.NodeAddr).RawGet(ctx, []uint32{loc})
	return v, err
}

// listFloor is the first list ID above every list of t: the build's and the created ones.
func listFloor(t *epoch.Table, m *partitioning.Manifest) uint32 {
	n := uint32(m.NumLists)
	for l := range t.Lists {
		if l+1 > n {
			n = l + 1
		}
	}
	return n
}

func savePatch(path string, patch []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, patch, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
