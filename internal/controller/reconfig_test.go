package controller

import (
	"testing"

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
