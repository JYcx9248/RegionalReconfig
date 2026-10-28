// Package query is the per-query aggregator: given the lists an entry node navigated to,
// grouped by owner in the epoch the entry pinned, it sends the sub-queries to those owners and
// merges the answers.
//
// How the fixed re-ranking budget n is split across owners is open question U5. TwoPhase is
// implemented: it returns exactly the single-node answer for two round trips, which is what
// the experiments measure against. ProportionalQuota (one round trip, an answer that can
// differ) is still a placeholder, so the question stays open. What both build on -- grouping
// by owner, the data-node primitives (internal/nodeclient: FILTER, RERANK) and the
// deterministic (distance, ID) merge with global dedup -- is here.
package query

import (
	"container/heap"
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"

	"rtier/internal/design"
	"rtier/internal/epoch"
	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
	"rtier/internal/placement"
)

// Params are the search parameters of one query.
type Params struct {
	K      int `json:"k"`      // results returned
	NProbe int `json:"nprobe"` // posting lists probed (m)
	N      int `json:"n"`      // candidates re-ranked with raw vectors: fixed, no early stop
	EF     int `json:"ef"`     // graph search width; 0 = 2*nprobe
}

// Request is one query at the aggregator.
type Request struct {
	Vec    []byte   // query vector in the index's native element type
	Lists  []uint32 // navigated lists, closest first (at the entry; a forwarded query carries them grouped by owner)
	Params Params
}

// Result of one query. IDs are vector IDs; mapping them to RAG chunk IDs is open (U12).
type Result struct {
	Epoch      uint64
	Candidates []nodeclient.Candidate
}

// Nodes resolves a node ID to its data-node client.
type Nodes interface {
	Client(id placement.NodeID) (*nodeclient.Client, error)
}

// Group is the lists of one query that one node owns, each with the version of its partition
// in the query's epoch (Versions parallel to Lists): the owner may hold several versions of a
// partition while an update batch flips (U14).
type Group struct {
	Node     placement.NodeID
	Lists    []uint32
	Versions []uint32
}

// GroupByOwner splits lists by the owner of their partition in table t (ascending node IDs;
// list order preserved within a group), with the partitions' versions in t.
func GroupByOwner(lists []uint32, m *partitioning.Manifest, t *epoch.Table) []Group {
	idx := map[placement.NodeID]int{}
	var gs []Group
	for _, c := range lists {
		p := t.PartitionOf(c, m)
		o := t.Owner(p)
		i, ok := idx[o]
		if !ok {
			i = len(gs)
			idx[o] = i
			gs = append(gs, Group{Node: o})
		}
		gs[i].Lists = append(gs[i].Lists, c)
		gs[i].Versions = append(gs[i].Versions, t.Version(p))
	}
	sort.Slice(gs, func(i, j int) bool { return gs[i].Node < gs[j].Node })
	return gs
}

// Env is what a strategy sees for one query: the epoch its entry pinned (the groups passed
// alongside were computed in it) and how to reach the nodes. There is deliberately no table:
// an aggregator may not have installed that epoch (see Aggregator.Run).
type Env struct {
	Epoch uint64
	Nodes Nodes
}

// Strategy executes one query over its owner groups.
type Strategy interface {
	Name() string
	Execute(ctx context.Context, env *Env, req *Request, groups []Group) ([]nodeclient.Candidate, error)
}

// NewStrategy returns the strategy named name; two-phase when none is named (U5 is decided).
func NewStrategy(name string) (Strategy, error) {
	switch name {
	case "", "two-phase":
		return TwoPhase{}, nil
	case "proportional-quota":
		return ProportionalQuota{}, nil
	}
	return nil, fmt.Errorf("query: unknown strategy %q", name)
}

// TwoPhase: FILTER at every owner for its own top-n by PQ distance, merge the global top-n,
// RERANK each candidate at the owner that reported it, merge the top-k. Two round trips, and
// the work stays at n: a vector of the global top-n is in its owner's local top-n (nothing
// better than it can hide behind a cut), and the merge keeps one entry per ID, so the answer
// is exactly the single-node fixed-n one. Re-ranking at the reporting owner is also what will
// keep working once the raw-vector pages follow the lists (U1); today every node has them.
//
// One of the two answers to U5 (ProportionalQuota is the other, still open).
type TwoPhase struct{}

func (TwoPhase) Name() string { return "two-phase" }

func (TwoPhase) Execute(ctx context.Context, env *Env, req *Request, groups []Group) ([]nodeclient.Candidate, error) {
	if len(groups) == 0 {
		return nil, nil
	}
	n := req.Params.N
	if n <= 0 {
		n = req.Params.K
	}
	clients := make([]*nodeclient.Client, len(groups))
	filtered := make([][]nodeclient.Candidate, len(groups))
	errs := make([]error, len(groups))
	var wg sync.WaitGroup
	for i, g := range groups {
		wg.Add(1)
		go func(i int, g Group) {
			defer wg.Done()
			c, err := env.Nodes.Client(g.Node)
			if err != nil {
				errs[i] = err
				return
			}
			clients[i] = c
			filtered[i], _, errs[i] = c.Filter(ctx, env.Epoch, req.Vec, g.Lists, g.Versions, n)
		}(i, g)
	}
	wg.Wait()
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}

	byOwner := splitByOrigin(filtered, MergeTopK(filtered, n))
	reranked := make([][]nodeclient.Candidate, len(groups))
	for i := range groups {
		if len(byOwner[i]) == 0 {
			continue
		}
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			// The owner's lists go along: they name every candidate, so the owner knows where to
			// fetch a raw vector it lacks (U9).
			reranked[i], _, errs[i] = clients[i].Rerank(ctx, env.Epoch, req.Vec, byOwner[i], groups[i].Lists,
				groups[i].Versions, req.Params.K)
		}(i)
	}
	wg.Wait()
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	return MergeTopK(reranked, req.Params.K), nil
}

// splitByOrigin groups the merged candidates by the group that reported them, keeping each
// group's order. A vector several owners reported (FusionANNS replicates boundary vectors into
// up to 8 lists) goes to the first of them, so the split does not depend on timing.
func splitByOrigin(parts [][]nodeclient.Candidate, merged []nodeclient.Candidate) [][]uint32 {
	origin := make(map[uint32]int, len(merged))
	for i := len(parts) - 1; i >= 0; i-- {
		for _, c := range parts[i] {
			origin[c.ID] = i
		}
	}
	out := make([][]uint32, len(parts))
	for _, c := range merged {
		i := origin[c.ID]
		out[i] = append(out[i], c.ID)
	}
	return out
}

// ProportionalQuota: each owner filters and re-ranks n x (its share of the probed lists or
// candidates) locally in one round trip; the aggregator merges the top-k. Total work stays n
// but the selected set can differ from the global top-n. TODO(design) U5.
type ProportionalQuota struct{}

func (ProportionalQuota) Name() string { return "proportional-quota" }
func (ProportionalQuota) Execute(context.Context, *Env, *Request, []Group) ([]nodeclient.Candidate, error) {
	return nil, design.Undecided("U5", "proportional quotas")
}

// MergeTopK merges candidate lists: one entry per ID (the smallest distance wins; exact
// distances of the same vector are equal everywhere), ordered by (distance, ID), at most k (all
// of them if k < 0). A node returns its candidates in that order, FILTER's and RERANK's alike,
// so the merge takes them off the fronts of the lists, smallest first, and stops at k distinct
// IDs: the first time an ID comes off is at its smallest distance. That is a pass over the lists
// (their order checked) and O(k log lists), where sorting them all was O(n log n), twice per
// query: 20 us against 119 us for four lists of 200 (28 against 254 for eight). A list out of
// order is sorted (a copy) first, so the answer never depends on the order the lists come in.
func MergeTopK(parts [][]nodeclient.Candidate, k int) []nodeclient.Candidate {
	sorted, copied, total := parts, false, 0
	for i, p := range parts {
		total += len(p)
		if !sort.SliceIsSorted(p, func(a, b int) bool { return nodeclient.Less(p[a], p[b]) }) {
			if !copied {
				sorted, copied = append([][]nodeclient.Candidate(nil), parts...), true
			}
			s := append([]nodeclient.Candidate(nil), p...)
			sort.Slice(s, func(a, b int) bool { return nodeclient.Less(s[a], s[b]) })
			sorted[i] = s
		}
	}
	limit := total
	if k >= 0 && k < limit {
		limit = k
	}
	f := &fronts{parts: sorted, next: make([]int, len(sorted))}
	for i, p := range sorted {
		if len(p) > 0 {
			f.h = append(f.h, i)
		}
	}
	heap.Init(f)
	out := make([]nodeclient.Candidate, 0, limit)
	seen := make(map[uint32]struct{}, limit)
	for len(out) < limit && f.Len() > 0 {
		i := f.h[0]
		c := sorted[i][f.next[i]]
		if f.next[i]++; f.next[i] == len(sorted[i]) {
			heap.Pop(f)
		} else {
			heap.Fix(f, 0)
		}
		if _, dup := seen[c.ID]; !dup {
			seen[c.ID] = struct{}{}
			out = append(out, c)
		}
	}
	return out
}

// fronts is MergeTopK's heap: the lists with candidates left, by the first of them.
type fronts struct {
	parts [][]nodeclient.Candidate
	next  []int // next[i]: the first candidate of parts[i] not merged yet
	h     []int // indices into parts
}

func (f *fronts) Len() int { return len(f.h) }
func (f *fronts) Less(i, j int) bool {
	a, b := f.h[i], f.h[j]
	return nodeclient.Less(f.parts[a][f.next[a]], f.parts[b][f.next[b]])
}
func (f *fronts) Swap(i, j int) { f.h[i], f.h[j] = f.h[j], f.h[i] }
func (f *fronts) Push(x any)    { f.h = append(f.h, x.(int)) }
func (f *fronts) Pop() any {
	x := f.h[len(f.h)-1]
	f.h = f.h[:len(f.h)-1]
	return x
}

// Aggregator runs the scatter/gather of a query under the epoch its entry pinned.
type Aggregator struct {
	Nodes    Nodes
	Strategy Strategy
}

// ErrNoEpoch is returned before the first epoch is installed.
var ErrNoEpoch = fmt.Errorf("query: no epoch installed")

// Run executes a query its entry has routed: groups are the query's lists by owner in epoch
// e, the epoch the entry pinned. Run neither pins nor consults an epoch of this node's own.
// The entry keeps e pinned until the answer is back -- whether it aggregates itself or
// forwarded the query here -- and the entry is a node of e, so the grace period waits for the
// whole query and the owners in groups keep their partitions until it is done. That is what
// lets a node aggregate under an epoch it has not installed yet (a node that joins, right
// after the flip) or has already moved past (a node that is leaving, for a query admitted
// just before the flip). The entry checked that this node is an aggregator in e.
func (a *Aggregator) Run(ctx context.Context, e uint64, req *Request, groups []Group) (Result, error) {
	cands, err := a.Strategy.Execute(ctx, &Env{Epoch: e, Nodes: a.Nodes}, req, groups)
	if nodeclient.IsNotResident(err) {
		// With a correct grace period this cannot happen: owners keep partitions until every
		// query pinned to an older epoch finished, and this node pinned e when it read the
		// forward. It takes a failure -- a forward that outlived the entry's timeout, a node
		// that restarted -- which this work does not handle (U7, U13): the query fails.
		err = fmt.Errorf("query: a sub-query of epoch %d reached a node that no longer holds its partition: %w", e, err)
	}
	return Result{Epoch: e, Candidates: cands}, err
}
