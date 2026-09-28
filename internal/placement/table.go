// Package placement maps partitions to nodes (partition -> node, one entry per partition).
//
// Adapted from Koala (Apache-2.0): internal/keyby/partitionTable.go (bucket -> worker table,
// range serialization, Reconfigure returning owner changes), partition/partitionPolicy.go
// (computeBucketOwnerChanges), coordinator/apiServiceUtil.go (GenerateMigrationPlan) and the
// coordinator's BucketOwnerHistory. Koala's buckets are hash ranges of the key space; here a
// "bucket" is a partition, a fixed group of posting lists (see internal/partitioning).
package placement

import (
	"fmt"
	"sort"
)

// NodeID identifies a node; assigned by the controller at registration (monotonic, as Koala's
// WorkerManager.NextWorkerId).
type NodeID uint16

// Table is the placement: Owners[p] serves partition p. Single owner per partition. Replicas
// of hot partitions are out of scope for rtier (see the package comment of internal/design):
// elastic read capacity for a hot key belongs to the serverless-instance pattern of streaming
// systems, not to the reconfiguration protocol.
type Table struct {
	Owners []NodeID `json:"owners"`
}

// Clone returns a deep copy.
func (t Table) Clone() Table { return Table{Owners: append([]NodeID(nil), t.Owners...)} }

// NumPartitions is the number of partitions.
func (t Table) NumPartitions() int { return len(t.Owners) }

// PartitionsOf lists the partitions owned by n, ascending.
func (t Table) PartitionsOf(n NodeID) []int {
	var ps []int
	for p, o := range t.Owners {
		if o == n {
			ps = append(ps, p)
		}
	}
	return ps
}

// Nodes lists the distinct owners in order of first appearance.
func (t Table) Nodes() []NodeID {
	seen := map[NodeID]bool{}
	var ns []NodeID
	for _, o := range t.Owners {
		if !seen[o] {
			seen[o] = true
			ns = append(ns, o)
		}
	}
	return ns
}

// Counts returns the number of partitions per owner.
func (t Table) Counts() map[NodeID]int {
	c := map[NodeID]int{}
	for _, o := range t.Owners {
		c[o]++
	}
	return c
}

// Range is a run of consecutive partitions [Lo, Hi] with the same owner.
type Range struct {
	Node NodeID `json:"node"`
	Lo   int    `json:"lo"`
	Hi   int    `json:"hi"`
}

// Ranges compresses the table into runs (Koala's PartitionTable.Serialize).
func (t Table) Ranges() []Range {
	var rs []Range
	for p, o := range t.Owners {
		if n := len(rs); n > 0 && rs[n-1].Node == o && rs[n-1].Hi == p-1 {
			rs[n-1].Hi = p
			continue
		}
		rs = append(rs, Range{Node: o, Lo: p, Hi: p})
	}
	return rs
}

// FromRanges rebuilds a table of n partitions (Koala's DeserializePartitionTable).
func FromRanges(n int, rs []Range) (Table, error) {
	owners := make([]NodeID, n)
	set := make([]bool, n)
	for _, r := range rs {
		if r.Lo < 0 || r.Hi >= n || r.Lo > r.Hi {
			return Table{}, fmt.Errorf("placement: bad range %+v for %d partitions", r, n)
		}
		for p := r.Lo; p <= r.Hi; p++ {
			owners[p] = r.Node
			set[p] = true
		}
	}
	for p, ok := range set {
		if !ok {
			return Table{}, fmt.Errorf("placement: partition %d has no owner", p)
		}
	}
	return Table{Owners: owners}, nil
}

// Changes records partition moves: Changes[src][dst] = partitions moving from src to dst
// (Koala's bucketOwnerChanges).
type Changes map[NodeID]map[NodeID][]int

// Diff computes the moves from old to next (Koala's computeBucketOwnerChanges).
func Diff(old, next Table) Changes {
	ch := Changes{}
	for p := range old.Owners {
		a, b := old.Owners[p], next.Owners[p]
		if a == b {
			continue
		}
		if ch[a] == nil {
			ch[a] = map[NodeID][]int{}
		}
		ch[a][b] = append(ch[a][b], p)
	}
	return ch
}

// Moved returns every moved partition (ascending).
func (c Changes) Moved() []int {
	var ps []int
	for _, m := range c {
		for _, l := range m {
			ps = append(ps, l...)
		}
	}
	sort.Ints(ps)
	return ps
}

// Pull is one transfer in a migration plan: the destination pulls Partitions from From.
type Pull struct {
	From       NodeID  `json:"from"`
	Partitions []int   `json:"partitions"`
	Ranges     []Range `json:"ranges"` // the same partitions as runs, for logs
}

// Plan maps each destination to its pulls (Koala's GenerateMigrationPlan: destinations pull
// from sources). Pulls are ordered by source ID for determinism.
type Plan map[NodeID][]Pull

// MakePlan turns owner changes into per-destination pulls.
func MakePlan(ch Changes) Plan {
	plan := Plan{}
	for src, m := range ch {
		for dst, ps := range m {
			ps = append([]int(nil), ps...)
			sort.Ints(ps)
			var rs []Range
			for _, p := range ps {
				if n := len(rs); n > 0 && rs[n-1].Hi == p-1 {
					rs[n-1].Hi = p
					continue
				}
				rs = append(rs, Range{Node: src, Lo: p, Hi: p})
			}
			plan[dst] = append(plan[dst], Pull{From: src, Partitions: ps, Ranges: rs})
		}
	}
	for dst := range plan {
		sort.Slice(plan[dst], func(i, j int) bool { return plan[dst][i].From < plan[dst][j].From })
	}
	return plan
}

// History records who has owned each partition (Koala's BucketOwnerHistory), in two senses:
//
//	Owners  nodes that may still hold a copy of the partition's segment files; a node drops
//	        out when it has reclaimed (evicted) its copy.
//	Ever    nodes that have owned the partition at any point, never forgotten. With a static
//	        dataset such a node still holds the PQ codes of the partition's vectors (they are
//	        kept as a cache), so moving the partition back there costs no PQ transfer -- which
//	        is what ReversiblePolicy uses.
type History struct {
	Owners []map[NodeID]bool `json:"owners"`
	Ever   []map[NodeID]bool `json:"ever"`
}

// NewHistory starts a history from t.
func NewHistory(t Table) *History {
	h := &History{
		Owners: make([]map[NodeID]bool, len(t.Owners)),
		Ever:   make([]map[NodeID]bool, len(t.Owners)),
	}
	for p, o := range t.Owners {
		h.Owners[p] = map[NodeID]bool{o: true}
		h.Ever[p] = map[NodeID]bool{o: true}
	}
	return h
}

// Record adds the owners of t.
func (h *History) Record(t Table) {
	for p, o := range t.Owners {
		h.Owners[p][o] = true
		h.Ever[p][o] = true
	}
}

// Forget removes node n from partition p's copy set (after n has reclaimed its copy). What n
// has ever owned is kept: its PQ codes survive the eviction.
func (h *History) Forget(p int, n NodeID) { delete(h.Owners[p], n) }

// HadCopy reports whether node n has owned partition p and not reclaimed it.
func (h *History) HadCopy(p int, n NodeID) bool { return h.Owners[p][n] }

// HasOwned reports whether node n has ever owned partition p.
func (h *History) HasOwned(p int, n NodeID) bool { return h.Ever[p][n] }
