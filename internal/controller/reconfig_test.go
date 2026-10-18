package controller

import (
	"strconv"
	"testing"

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
