package controller

import (
	"strconv"
	"strings"
	"testing"

	"rtier/internal/config"
	"rtier/internal/epoch"
	"rtier/internal/placement"
)

func TestRoundRobinGraphSources(t *testing.T) {
	adds := []placement.NodeID{5, 6, 7}
	// Three new nodes over two entries, cursor at 0: sources 0, 1, 0.
	got := roundRobin(adds, 2, 0)
	if got[5] != 0 || got[6] != 1 || got[7] != 0 {
		t.Fatalf("roundRobin from 0: %v", got)
	}
	// The cursor carries over: the next reconfiguration starts where this one stopped, so
	// successive one-node scale-outs rotate over the entries.
	if one := roundRobin([]placement.NodeID{8}, 2, 3); one[8] != 1 {
		t.Fatalf("roundRobin from 3: %v", one)
	}
	// No live entry: every node copies from the build output.
	if none := roundRobin(adds, 0, 0); none[5] != -1 || none[7] != -1 {
		t.Fatalf("roundRobin without sources: %v", none)
	}
}

// TestResolvePlanNodes: a plan names nodes by ID or by name; a name two registrations share
// with neither of them connected, an unknown name and an unknown ID resolve to nothing.
func TestResolvePlanNodes(t *testing.T) {
	m := NewNodeManager()
	n1 := m.Add("n1", epoch.NodeInfo{}, nil)
	n2 := m.Add("n2", epoch.NodeInfo{}, nil)
	m.Add("n2", epoch.NodeInfo{}, nil) // registered again
	for key, want := range map[string]placement.NodeID{"n1": n1.ID, strconv.Itoa(int(n2.ID)): n2.ID} {
		if n, err := m.Resolve(key); err != nil || n.ID != want {
			t.Errorf("Resolve(%q) = %v, %v; want node %d", key, n, err, want)
		}
	}
	for _, key := range []string{"n2", "n9", "99", "70000"} {
		if n, err := m.Resolve(key); err == nil {
			t.Errorf("Resolve(%q) = node %d, want an error", key, n.ID)
		}
	}
}

// TestPlanWarnings: with lazy, a node that leaves stays the raw-vector source of the partitions
// it hands to nodes that have never owned them; handing a partition back to a node that has
// held it, or a protocol that copies every raw vector before the flip, leaves no such source.
func TestPlanWarnings(t *testing.T) {
	h := placement.NewHistory(placement.Table{Owners: []placement.NodeID{1, 1, 2, 2}})
	cur := placement.Table{Owners: []placement.NodeID{1, 1, 2, 3}} // node 3 took partition 3 from node 2
	h.Record(cur)
	c := &Controller{cfg: config.Controller{Protocol: "lazy"}, history: h}
	n2, n3 := &ManagedNode{ID: 2, Name: "n2"}, &ManagedNode{ID: 3, Name: "n3"}
	warnings := func(owners []placement.NodeID, removed ...*ManagedNode) []string {
		next := placement.Table{Owners: owners}
		return c.planWarnings(target{next: next, changes: placement.Diff(cur, next), removed: removed})
	}
	if w := warnings([]placement.NodeID{1, 1, 2, 2}, n3); len(w) != 0 {
		t.Errorf("partition 3 back to node 2, which held it: warnings %q", w)
	}
	w := warnings([]placement.NodeID{1, 1, 1, 1}, n2, n3)
	if len(w) != 2 || !strings.HasPrefix(w[0], "node 2 (n2) leaves") || !strings.Contains(w[0], "partitions [2]") ||
		!strings.HasPrefix(w[1], "node 3 (n3) leaves") || !strings.Contains(w[1], "partitions [3]") {
		t.Errorf("everything to node 1, which never held partitions 2 and 3: warnings %q", w)
	}
	c.cfg.Protocol = "copy-then-flip"
	if w := warnings([]placement.NodeID{1, 1, 1, 1}, n2, n3); len(w) != 0 {
		t.Errorf("copy-then-flip: warnings %q", w)
	}
}
