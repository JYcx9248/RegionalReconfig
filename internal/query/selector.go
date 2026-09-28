package query

import (
	"fmt"
	"sort"
	"sync/atomic"

	"rtier/internal/design"
	"rtier/internal/epoch"
	"rtier/internal/placement"
)

// Selector picks the node that aggregates a query received by entry node self. It runs after
// the entry navigated: req.Lists holds the probed lists and groups their owners under the
// entry's pinned epoch t -- the nodes the query has to reach anyway -- so the choice can be
// made among them. It must pick an aggregator of t. The chosen node runs the query with these
// groups, under t, whatever epoch it has installed itself (Aggregator.Run).
type Selector interface {
	Name() string
	Pick(t *epoch.Table, self placement.NodeID, req *Request, groups []Group) (placement.NodeID, error)
}

// NewSelector returns the selector named name.
func NewSelector(name string) (Selector, error) {
	switch name {
	case "", "owner":
		return OwnerSelector{}, nil
	case "local":
		return LocalSelector{}, nil
	case "warmup":
		return &WarmupSelector{}, nil
	case "outsource":
		return OutsourceSelector{}, nil
	}
	return nil, fmt.Errorf("query: unknown selector %q", name)
}

// OwnerSelector (the default) picks the aggregator from the query's owners:
//
//   - the entry itself when it is one of them: its own FILTER and RERANK stay on the node;
//   - the owner, when a single node holds every probed list: the query is forwarded once and
//     both phases run where the data is -- one network round trip instead of two, and the n PQ
//     candidates never cross the network;
//   - otherwise the entry: forwarding would add a hop to save one of several parallel
//     sub-queries, and would pile aggregation onto the owners of hot partitions, while client
//     queries already spread over the entries.
//
// Measured on 200K synthetic vectors, 64 partitions, nprobe 64 (cmd/rtier-overhead -owners):
// with locality-aware grouping the entry is an owner for 90% / 62% / 47% of queries at 2 / 8
// / 16 nodes, and a single node holds all lists for 20% / 5% / 5%.
type OwnerSelector struct{}

func (OwnerSelector) Name() string { return "owner" }

func (OwnerSelector) Pick(t *epoch.Table, self placement.NodeID, _ *Request, groups []Group) (placement.NodeID, error) {
	for _, g := range groups {
		if g.Node == self {
			return self, nil
		}
	}
	if len(groups) == 1 && t.IsAggregator(groups[0].Node) {
		return groups[0].Node, nil
	}
	return self, nil
}

// LocalSelector: the entry aggregates its own queries, whoever owns the data (the baseline).
type LocalSelector struct{}

func (LocalSelector) Name() string { return "local" }
func (LocalSelector) Pick(_ *epoch.Table, self placement.NodeID, _ *Request, _ []Group) (placement.NodeID, error) {
	return self, nil
}

// WarmupSelector hands the scatter/gather to a node that may aggregate in this epoch but is
// not an entry: during a scale-out that is exactly a new node whose navigation graph is still
// arriving. It is the one window where the nodes that can aggregate outnumber the nodes that
// can take client queries; outside it every aggregator is an entry, the candidate set is
// empty, and the entry aggregates its own queries -- which is what LocalSelector does. One
// answer to U6.
//
// What it moves is modest: an aggregator's own work is the fan-out and the merge, while the
// entry keeps the navigation (the expensive part, and the part that needs the graph). So this
// is the cheap half of "let the new node help before it has its graph"; the other half is
// having it take client queries and forward the navigation, which splits the work the same
// way from the other side. Whether either pays for itself is an experiment, not a claim.
type WarmupSelector struct{ next atomic.Uint64 }

func (*WarmupSelector) Name() string { return "warmup" }

func (s *WarmupSelector) Pick(t *epoch.Table, self placement.NodeID, _ *Request, _ []Group) (placement.NodeID, error) {
	var warming []placement.NodeID
	for _, id := range t.Aggregators {
		if !t.IsEntry(id) {
			warming = append(warming, id)
		}
	}
	if len(warming) == 0 {
		return self, nil
	}
	sort.Slice(warming, func(i, j int) bool { return warming[i] < warming[j] })
	return warming[s.next.Add(1)%uint64(len(warming))], nil
}

// OutsourceSelector is the load-aware version: pick the aggregator by where the work should
// go (heat, in-flight queries, locality), not just by who is warming up. TODO(design) U6.
type OutsourceSelector struct{}

func (OutsourceSelector) Name() string { return "outsource" }
func (OutsourceSelector) Pick(*epoch.Table, placement.NodeID, *Request, []Group) (placement.NodeID, error) {
	return 0, design.Undecided("U6", "load-aware aggregator outsourcing")
}
