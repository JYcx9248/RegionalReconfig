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

// Group is the lists of one query that one node owns.
type Group struct {
	Node  placement.NodeID
	Lists []uint32
}

// GroupByOwner splits lists by the owner of their partition in table t (ascending node IDs;
// list order preserved within a group).
func GroupByOwner(lists []uint32, m *partitioning.Manifest, t *epoch.Table) []Group {
	idx := map[placement.NodeID]int{}
	var gs []Group
	for _, c := range lists {
		o := t.Owner(m.PartitionOf(c))
		i, ok := idx[o]
		if !ok {
			i = len(gs)
			idx[o] = i
			gs = append(gs, Group{Node: o})
		}
		gs[i].Lists = append(gs[i].Lists, c)
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

// NewStrategy returns the strategy named name.
func NewStrategy(name string) (Strategy, error) {
	switch name {
	case "two-phase":
		return TwoPhase{}, nil
	case "proportional-quota":
		return ProportionalQuota{}, nil
	case "", "undecided":
		return nil, design.Undecided("U5", "no query strategy selected")
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
			filtered[i], _, errs[i] = c.Filter(ctx, env.Epoch, req.Vec, g.Lists, n)
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
			reranked[i], _, errs[i] = clients[i].Rerank(ctx, env.Epoch, req.Vec, byOwner[i], groups[i].Lists, req.Params.K)
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
// distances of the same vector are equal everywhere), ordered by (distance, ID), at most k.
func MergeTopK(parts [][]nodeclient.Candidate, k int) []nodeclient.Candidate {
	best := map[uint32]float32{}
	for _, p := range parts {
		for _, c := range p {
			if d, ok := best[c.ID]; !ok || c.Dist < d {
				best[c.ID] = c.Dist
			}
		}
	}
	out := make([]nodeclient.Candidate, 0, len(best))
	for id, d := range best {
		out = append(out, nodeclient.Candidate{ID: id, Dist: d})
	}
	sort.Slice(out, func(i, j int) bool { return nodeclient.Less(out[i], out[j]) })
	if k >= 0 && len(out) > k {
		out = out[:k]
	}
	return out
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
		// query pinned to an older epoch finished. What to do when it does anyway (a node
		// restarted, a bug) is U7.
		err = fmt.Errorf("%w (%v)", design.Undecided("U7", "sub-query reached a node that no longer holds the partition"), err)
	}
	return Result{Epoch: e, Candidates: cands}, err
}
