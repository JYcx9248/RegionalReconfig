package e2e

import (
	"reflect"
	"testing"
	"time"

	"rtier/internal/placement"
	"rtier/internal/protocol"
	"rtier/internal/query"
)

// TestScaleInDrainsLeavingNode: a scale-in returns a partition to a node that held it only
// partly. The partition goes from n1 (bootstrap) to n3 (lazy: n3 fetches only what queries
// need) to n4, and back to n3. The segment n4 writes for the move records where each raw vector
// is -- n4, n3 or n1 -- so n3 fetches what it lacks straight from its holder, and before the
// rescale returns it takes over what only n4 holds (the drain). Then n4 is stopped, and every
// answer is still exact. (With a source per posting list instead, n3 pointed to n4 for these
// lists and n4 to n3: queries timed out and both data nodes ran out of threads.) Idle: no query
// runs while the partition moves, so every vector is n1's; under load, n3 and n4 fetch as
// queries need them.
func TestScaleInDrainsLeavingNode(t *testing.T) {
	f := getFixture(t)
	for _, load := range []bool{false, true} {
		name := "idle"
		if load {
			name = "under load"
		}
		t.Run(name, func(t *testing.T) {
			cl := newCluster(t, f, "lazy", 2)
			cl.addNode("n1", query.TwoPhase{})
			cl.addNode("n2", query.TwoPhase{})
			if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
				t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
			}
			cl.addNode("n3", query.TwoPhase{})
			n4 := cl.addNode("n4", query.TwoPhase{})
			var stop func() loadStats
			if load {
				stop = startLoad(cl, 4)
				time.Sleep(200 * time.Millisecond)
			}
			reconfigure(t, cl, protocol.RescaleReq{DataNodes: 3})
			t3 := cl.ctl.Status().Table
			owners := map[string][]int{}
			for id, ps := range t3.Placement.ByNode() {
				owners[t3.Nodes[id].Name] = ps
			}
			if len(owners["n3"]) == 0 {
				t.Fatalf("placement %v: n3 took no partition", owners)
			}
			// One of n3's partitions goes on to n4, then back.
			moved := owners["n3"][0]
			four := map[string][]int{"n1": owners["n1"], "n2": owners["n2"], "n3": owners["n3"][1:], "n4": {moved}}
			reconfigure(t, cl, protocol.RescaleReq{Plan: &protocol.Plan{Owners: four}})
			if load {
				time.Sleep(300 * time.Millisecond) // n4 fetches what the queries need
			}
			out := reconfigure(t, cl, protocol.RescaleReq{Plan: &protocol.Plan{Owners: owners}})
			if !reflect.DeepEqual(out.Removed, []placement.NodeID{n4.ID()}) || !reflect.DeepEqual(out.Moved, []int{moved}) {
				t.Fatalf("scale-in reply %+v: want partition %d back from node %d", out, moved, n4.ID())
			}
			t.Logf("partition %d back at n3: %d raw vectors drained from n4 (%.3f s after the flip)",
				moved, out.DrainVectors, out.DrainSeconds)
			if load {
				time.Sleep(200 * time.Millisecond)
				st := stop()
				if st.mismatches > 0 || st.errors > 0 {
					t.Fatalf("%d mismatches, %d errors; first: %s", st.mismatches, st.errors, st.firstProblem)
				}
			}
			checkResidency(t, cl)
			cl.stopDataNode(n4)
			exactPass(t, cl, nil)
		})
	}
}
