package update

import (
	"compress/gzip"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"os"
	"sort"

	"rtier/internal/placement"
	"rtier/internal/protocol"
)

// A Quake round can be decided once and run again later (U14): PrepareQuake keeps what it decided
// as a QuakePlan (Major.QuakePlan), WriteQuakePlan saves it, and PrepareQuakePlan runs it on the
// state it was decided on, deciding nothing. Quake runs its maintenance apart from its queries
// and reports its time apart ("The current implementation executes searches, updates, and
// maintenance serially", §8.2; maintenance time "separately from update latency, as maintenance
// can be conducted in the background", §7.2): a run that executes a saved plan measures what
// this mechanism adds -- reading the lists, the graph edit, the staging and the flip -- and not
// the policy's decisions, which a planning run makes once for every repetition.
//
// The plan holds what the round's end needs, by list and vector ID: the deleted lists, the splits
// (each list's children and their heads), the members of every other list the round changes after
// it (the children's and the refined lists'), the receivers of deleted lists' members that the
// refinement did not read (Gains), the lists left empty, and the heads the refinement moved in
// place. No epoch, version, location or holder: those come from the cluster when the plan runs.
// It also holds a digest of every list the round read, and the list allocator's position: a plan
// runs only where both are the same, so that its lists hold what they held when it was decided.

const quakePlanFormat = 1

// QuakePlan is one round of Quake's maintenance as decided.
type QuakePlan struct {
	Format  int              `json:"format"`
	Stats   QuakeStats       `json:"stats"` // what the round found and did when it was decided
	Deletes []uint32         `json:"deletes,omitempty"`
	Splits  []QuakePlanSplit `json:"splits,omitempty"`
	// Final: every list the round changes but deletes or splits, with its members after the round.
	Final map[uint32][]uint32 `json:"final,omitempty"`
	// Gains: lists the refinement did not read that receive deleted lists' members.
	Gains     map[uint32][]uint32 `json:"gains,omitempty"`
	Emptied   []uint32            `json:"emptied,omitempty"`    // lists the round removes, left empty
	Moved     []uint32            `json:"moved,omitempty"`      // heads the refinement moved in place
	MovedVecs []byte              `json:"moved_vecs,omitempty"` // their vectors, element bytes each
	// Read: every list the round read, as it was then.
	Read map[uint32]ListDigest `json:"read"`
	// NextList, ListGen: the list allocator before the round (Allocator.NextList).
	NextList uint64 `json:"next_list"`
	ListGen  uint32 `json:"list_gen"`
}

// QuakePlanSplit is a split list, its two children and their heads (the 2-means' means, as
// element bytes: 2 x the vectors' size).
type QuakePlanSplit struct {
	List     uint32    `json:"list"`
	Children [2]uint32 `json:"children"`
	Heads    []byte    `json:"heads"`
}

// ListDigest is a list's members, by count and a digest of their sorted IDs (FNV-1a, 64 bits).
type ListDigest struct {
	Postings int    `json:"postings"`
	FNV      uint64 `json:"fnv"`
}

func digestOf(ids []uint32) ListDigest {
	s := append([]uint32(nil), ids...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	h := fnv.New64a()
	var b [4]byte
	for _, id := range s {
		binary.LittleEndian.PutUint32(b[:], id)
		h.Write(b[:])
	}
	return ListDigest{Postings: len(ids), FNV: h.Sum64()}
}

// WriteQuakePlan saves a plan as gzip'd JSON (to path, by a rename: no half-written plan).
func WriteQuakePlan(path string, p *QuakePlan) error {
	f, err := os.Create(path + ".part")
	if err != nil {
		return err
	}
	zw := gzip.NewWriter(f)
	err = json.NewEncoder(zw).Encode(p)
	err = errors.Join(err, zw.Close(), f.Close())
	if err == nil {
		err = os.Rename(path+".part", path)
	}
	if err != nil {
		os.Remove(path + ".part")
		return fmt.Errorf("update: writing the Quake plan %s: %w", path, err)
	}
	return nil
}

// ReadQuakePlan reads a plan WriteQuakePlan saved.
func ReadQuakePlan(path string) (*QuakePlan, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("update: the Quake plan %s: %w", path, err)
	}
	var p QuakePlan
	if err := json.NewDecoder(zr).Decode(&p); err != nil {
		return nil, fmt.Errorf("update: the Quake plan %s: %w", path, err)
	}
	if p.Format != quakePlanFormat {
		return nil, fmt.Errorf("update: the Quake plan %s has format %d, want %d", path, p.Format, quakePlanFormat)
	}
	return &p, nil
}

// PrepareQuakePlan begins a major epoch that runs a round of Quake's maintenance decided before:
// it takes the lease, brings the router's graph to the epoch's version, reads every list the plan
// read and checks that each holds what it held then, takes the same list IDs for the children,
// edits the router's graph as the round did (the children's heads in, the deleted and split
// lists' out; then the emptied lists' out and the moved heads moved), and builds the same edits --
// it decides nothing. If the state is not the one the plan was decided on, it gives the lease up
// and returns an error, nothing changed.
func (u *Updater) PrepareQuakePlan(ctx context.Context, plan *QuakePlan) (mj *Major, st *QuakeStats, err error) {
	if plan == nil || plan.Format != quakePlanFormat {
		return nil, nil, errors.New("update: no Quake plan of this format")
	}
	if u.Router == nil {
		return nil, nil, errors.New("update: no router: a Quake round edits the update controller's own navigation graph")
	}
	lease, err := u.Controller.BeginUpdate(ctx, protocol.BeginUpdateReq{})
	if err != nil {
		return nil, nil, err
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
		return nil, nil, err
	}
	if err := u.syncRouter(ctx, t); err != nil {
		return nil, nil, err
	}
	in, err := u.Router.Info(ctx)
	if err != nil {
		return nil, nil, fmt.Errorf("update: router: %w", err)
	}

	// The lists the round read, as they are now: the same members, or no plan.
	lists := make([]uint32, 0, len(plan.Read))
	for l := range plan.Read {
		lists = append(lists, l)
	}
	sort.Slice(lists, func(i, j int) bool { return lists[i] < lists[j] })
	members, m, err := u.readLists(ctx, t, lists)
	if err != nil {
		return nil, nil, err
	}
	for _, l := range lists {
		if got, want := digestOf(members[l].IDs), plan.Read[l]; got != want {
			return nil, nil, fmt.Errorf("update: list %d holds %d postings (digest %016x), the plan was decided on %d "+
				"(%016x): not the state it was decided on", l, got.Postings, got.FNV, want.Postings, want.FNV)
		}
	}
	vecs := map[uint32]*qvec{}
	for _, l := range lists {
		lm := members[l]
		for k, id := range lm.IDs {
			if v, ok := vecs[id]; ok && v.base != l {
				return nil, nil, fmt.Errorf("update: vector %d is in lists %d and %d: Quake's maintenance needs an index "+
					"with one list per vector (fusion_build --clustering kmeans)", id, v.base, l)
			}
			vecs[id] = &qvec{loc: lm.Locs[k], holder: lm.Holders[k], code: lm.code(k, m), base: l}
		}
	}
	stats := plan.Stats
	stats.Replayed = true
	st = &stats
	mj = &Major{Lease: lease, Graph: t.Graph + 1, M: m, Lists: map[uint32]int{}, Parts: map[int]*PartEdits{},
		Codes: map[placement.NodeID]*CodeSet{}, Staged: map[placement.NodeID][]protocol.PartitionVersion{}, Quake: st}
	have := func(l uint32) (*listMembers, error) {
		lm, ok := members[l]
		if !ok {
			return nil, fmt.Errorf("update: the plan changes list %d, which it did not read", l)
		}
		return lm, nil
	}

	// The deletes and the splits, as the round recorded them; the children under the same IDs.
	final := map[uint32][]uint32{}
	for _, l := range plan.Deletes {
		lm, err := have(l)
		if err != nil {
			return nil, nil, err
		}
		mj.Merges = append(mj.Merges, Merge{List: l, Partition: lm.p, Postings: len(lm.IDs), Into: noList})
		final[l] = nil
	}
	var children []uint32
	if len(plan.Splits) > 0 {
		children = u.Alloc.TakeLists(2*len(plan.Splits), listFloor(t, u.Manifest))
	}
	vb := in.Dim * dtypeSize(in.DType)
	var labels, splits []uint32
	var heads []byte
	for i, sp := range plan.Splits {
		if children[2*i] != sp.Children[0] || children[2*i+1] != sp.Children[1] {
			return nil, nil, fmt.Errorf("update: the list allocator hands out lists %d and %d for list %d's children, "+
				"the plan has %d and %d: not the state it was decided on", children[2*i], children[2*i+1], sp.List,
				sp.Children[0], sp.Children[1])
		}
		if len(sp.Heads) != 2*vb {
			return nil, nil, fmt.Errorf("update: list %d's children have %d bytes of heads in the plan, want 2 of %d",
				sp.List, len(sp.Heads), vb)
		}
		lm, err := have(sp.List)
		if err != nil {
			return nil, nil, err
		}
		for c := 0; c < 2; c++ {
			mj.Lists[sp.Children[c]] = lm.p
		}
		mj.Splits = append(mj.Splits, Split{List: sp.List, Partition: lm.p, Children: sp.Children,
			Heads: [2]uint32{noHead, noHead}})
		labels, heads = append(labels, sp.Children[:]...), append(heads, sp.Heads...)
		splits = append(splits, sp.List)
		final[sp.List] = nil
	}
	if dead := append(append([]uint32(nil), plan.Deletes...), splits...); len(labels)+len(dead) > 0 {
		if mj.Patch, err = u.Router.EditGraph(ctx, mj.Graph, labels, heads, dead, nil, nil); err != nil {
			return nil, nil, fmt.Errorf("update: editing the graph: %w", err)
		}
		edited = true
	}
	for l, ids := range plan.Final {
		if _, created := mj.Lists[l]; !created {
			if _, err := have(l); err != nil {
				return nil, nil, err
			}
		}
		final[l] = ids
	}
	if err := u.finishQuake(ctx, mj, t, st, members, vecs, plan.Deletes, splits, final, plan.Gains, plan.Emptied,
		plan.Moved, plan.MovedVecs, &edited); err != nil {
		return nil, nil, err
	}
	return mj, st, nil
}

// DiscardMajor gives a major epoch up that was prepared but not executed: the router back at the
// lease's epoch, the lease given up, and for a Quake round the list IDs it took handed out again
// (a plan decided and discarded can then run on the same state: PrepareQuakePlan).
func (u *Updater) DiscardMajor(ctx context.Context, mj *Major) error {
	err := u.syncRouter(context.WithoutCancel(ctx), mj.Lease.Table)
	if mj.QuakePlan != nil {
		u.Alloc.SetNextList(mj.QuakePlan.NextList, mj.QuakePlan.ListGen)
	}
	return errors.Join(err, u.abort(ctx, mj.Lease))
}
