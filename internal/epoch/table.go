// Package epoch holds the versioned routing state of the regional tier and the machinery for
// switching versions without stopping queries.
//
// A Table is immutable once installed: the placement (partition -> node), the version of
// each partition's posting lists, the node addresses, and which nodes act as entries and
// aggregators. Reconfiguration copies data
// first, then flips the epoch atomically (Store), and reclaims old copies only after every
// query that routed with the old epoch has finished (Tracker, an RCU-style grace period).
// Koala has no explicit epoch: it runs one reconfiguration at a time and relies on in-band
// barriers in FIFO data channels, which request/response queries do not have.
package epoch

import (
	"fmt"

	"rtier/internal/placement"
)

// NodeInfo is how other nodes reach a node.
type NodeInfo struct {
	ID        placement.NodeID `json:"id"`
	Name      string           `json:"name"`
	NodeAddr  string           `json:"node_addr"`  // C++ data-node service (FILTER, RERANK, ...)
	QueryAddr string           `json:"query_addr"` // agent: client queries and delegated aggregation
	BulkAddr  string           `json:"bulk_addr"`  // agent: bulk transfer (separate connection)
}

// Table is one epoch of routing state.
type Table struct {
	Epoch     uint64          `json:"epoch"`
	Placement placement.Table `json:"placement"`
	// Versions[p] is the version of partition p's posting lists that queries of this epoch see:
	// an update batch (U14) stages the next version on the owner while the current one serves,
	// and both stay until the old epoch drains. Empty means every partition at version 0, which
	// is what a dataset without updates keeps.
	Versions []uint32 `json:"versions,omitempty"`
	// Graph is the version of the navigation graph that queries of this epoch search: a major
	// epoch (U14) patches the graph, and entries hold the patch next to the graph until the
	// epochs before it drain. 0 is the build's graph.
	Graph uint32 `json:"graph,omitempty"`
	// Lists maps the posting lists that major epochs created to their partitions; the build's
	// lists are in the partition manifest. A list ID is never reused.
	Lists       map[uint32]int                `json:"lists,omitempty"`
	Nodes       map[placement.NodeID]NodeInfo `json:"nodes"`
	Entries     []placement.NodeID            `json:"entries"`     // accept client queries; hold the graph
	Aggregators []placement.NodeID            `json:"aggregators"` // may run a query's scatter/gather
}

// Owner returns the node serving partition p.
func (t *Table) Owner(p int) placement.NodeID { return t.Placement.Owners[p] }

// Version returns the version of partition p's posting lists in this epoch.
func (t *Table) Version(p int) uint32 {
	if len(t.Versions) == 0 {
		return 0
	}
	return t.Versions[p]
}

// ListPartitions is how the build's lists map to partitions (partitioning.Manifest).
type ListPartitions interface{ PartitionOf(list uint32) int }

// PartitionOf returns the partition of list c in this epoch: one a major epoch created, or the
// build's.
func (t *Table) PartitionOf(c uint32, base ListPartitions) int {
	if p, ok := t.Lists[c]; ok {
		return p
	}
	return base.PartitionOf(c)
}

// Node looks up a node's addresses.
func (t *Table) Node(id placement.NodeID) (NodeInfo, bool) {
	n, ok := t.Nodes[id]
	return n, ok
}

// IsEntry reports whether id is an entry in this epoch.
func (t *Table) IsEntry(id placement.NodeID) bool { return contains(t.Entries, id) }

// IsAggregator reports whether id may aggregate in this epoch.
func (t *Table) IsAggregator(id placement.NodeID) bool { return contains(t.Aggregators, id) }

// Clone returns a deep copy (callers build the next epoch from a clone).
func (t *Table) Clone() *Table {
	c := *t
	c.Placement = t.Placement.Clone()
	c.Versions = append([]uint32(nil), t.Versions...)
	if t.Lists != nil {
		c.Lists = make(map[uint32]int, len(t.Lists))
		for k, v := range t.Lists {
			c.Lists[k] = v
		}
	}
	c.Nodes = make(map[placement.NodeID]NodeInfo, len(t.Nodes))
	for k, v := range t.Nodes {
		c.Nodes[k] = v
	}
	c.Entries = append([]placement.NodeID(nil), t.Entries...)
	c.Aggregators = append([]placement.NodeID(nil), t.Aggregators...)
	return &c
}

// Validate checks that every referenced node has addresses and that there is a version for
// every partition, if any.
func (t *Table) Validate() error {
	if t.Epoch == 0 {
		return fmt.Errorf("epoch: epoch 0 is reserved")
	}
	if n := len(t.Versions); n != 0 && n != t.Placement.NumPartitions() {
		return fmt.Errorf("epoch %d: versions for %d partitions, the placement has %d",
			t.Epoch, n, t.Placement.NumPartitions())
	}
	for c, p := range t.Lists {
		if p < 0 || p >= t.Placement.NumPartitions() {
			return fmt.Errorf("epoch %d: list %d in partition %d of %d", t.Epoch, c, p, t.Placement.NumPartitions())
		}
	}
	check := func(what string, id placement.NodeID) error {
		if _, ok := t.Nodes[id]; !ok {
			return fmt.Errorf("epoch %d: %s node %d has no addresses", t.Epoch, what, id)
		}
		return nil
	}
	for _, o := range t.Placement.Owners {
		if err := check("owner", o); err != nil {
			return err
		}
	}
	for _, id := range t.Entries {
		if err := check("entry", id); err != nil {
			return err
		}
	}
	for _, id := range t.Aggregators {
		if err := check("aggregator", id); err != nil {
			return err
		}
	}
	return nil
}

func contains(s []placement.NodeID, id placement.NodeID) bool {
	for _, x := range s {
		if x == id {
			return true
		}
	}
	return false
}
