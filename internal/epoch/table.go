// Package epoch holds the versioned routing state of the regional tier and the machinery for
// switching versions without stopping queries.
//
// A Table is immutable once installed: the placement (partition -> node), the node
// addresses, and which nodes act as entries and aggregators. Reconfiguration copies data
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
	Epoch       uint64                        `json:"epoch"`
	Placement   placement.Table               `json:"placement"`
	Nodes       map[placement.NodeID]NodeInfo `json:"nodes"`
	Entries     []placement.NodeID            `json:"entries"`     // accept client queries; hold the graph
	Aggregators []placement.NodeID            `json:"aggregators"` // may run a query's scatter/gather
}

// Owner returns the node serving partition p.
func (t *Table) Owner(p int) placement.NodeID { return t.Placement.Owners[p] }

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
	c.Nodes = make(map[placement.NodeID]NodeInfo, len(t.Nodes))
	for k, v := range t.Nodes {
		c.Nodes[k] = v
	}
	c.Entries = append([]placement.NodeID(nil), t.Entries...)
	c.Aggregators = append([]placement.NodeID(nil), t.Aggregators...)
	return &c
}

// Validate checks that every referenced node has addresses.
func (t *Table) Validate() error {
	if t.Epoch == 0 {
		return fmt.Errorf("epoch: epoch 0 is reserved")
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
