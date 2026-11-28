package e2e

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"rtier/internal/config"
	"rtier/internal/placement"
	"rtier/internal/protocol"
	"rtier/internal/query"
)

// TestTaskMigration: every partition of n2 goes to n3, a node that joins, and n2 leaves the
// placement without being drained (U9): it stays the holder of its raw vectors, which n3 can
// only get from it (a node that joins has no page file). With FetchExtra 0 -- fetch on demand --
// the migration returns after the flip and names n2 as still holding vectors. With FetchExtra on,
// n3 brings every vector still at n2 in the background, by requests of its own while no query
// fetches -- idle, nothing else moves them -- and the migration returns once no posting names
// n2: the last look pulls nothing, and with n2 stopped every answer is still exact. Under load,
// every answer during the migration is exact too.
func TestTaskMigration(t *testing.T) {
	f := getFixture(t)
	for _, c := range []struct {
		extra int
		load  bool
	}{{0, true}, {4, false}, {4, true}} {
		name := fmt.Sprintf("FetchExtra %d, idle", c.extra)
		if c.load {
			name = fmt.Sprintf("FetchExtra %d, under load", c.extra)
		}
		t.Run(name, func(t *testing.T) {
			cl := newCluster(t, f, "lazy", 2, func(cc *config.Controller) { cc.FetchExtra = c.extra })
			cl.addNode("n1", query.TwoPhase{})
			n2 := cl.addNode("n2", query.TwoPhase{})
			if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
				t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
			}
			n3 := cl.addNode("n3", query.TwoPhase{})
			before := cl.ctl.Status().Table.Placement.ByNode()
			var stop func() loadStats
			if c.load {
				stop = startLoad(cl, 4)
				time.Sleep(200 * time.Millisecond)
			}
			out := reconfigure(t, cl, protocol.RescaleReq{Migrate: &protocol.Migrate{From: "n2", To: "n3", RequestMax: 32}})
			after := cl.ctl.Status().Table.Placement.ByNode()
			if !reflect.DeepEqual(after[n3.ID()], before[n2.ID()]) || len(after[n2.ID()]) != 0 ||
				!reflect.DeepEqual(out.Removed, []placement.NodeID{n2.ID()}) ||
				!reflect.DeepEqual(out.Added, []placement.NodeID{n3.ID()}) {
				t.Fatalf("migration n2 -> n3: placement %v (was %v), reply %+v", after, before, out)
			}
			if c.load {
				time.Sleep(200 * time.Millisecond)
				st := stop()
				if st.mismatches > 0 || st.errors > 0 {
					t.Fatalf("%d mismatches, %d errors; first: %s", st.mismatches, st.errors, st.firstProblem)
				}
			}
			checkResidency(t, cl)
			if c.extra == 0 {
				if !reflect.DeepEqual(out.Holding, []placement.NodeID{n2.ID()}) || out.MigrateSeconds != 0 || out.DrainVectors != 0 {
					t.Fatalf("migration without FetchExtra: reply %+v, want n2 holding, nothing moved in the background", out)
				}
				exactPass(t, cl, nil) // n3 fetches from n2, which still runs
				return
			}
			if len(out.Holding) != 0 || out.MigrateSeconds <= 0 || out.DrainVectors != 0 || (!c.load && out.MigrateVectors == 0) {
				t.Fatalf("migration with FetchExtra %d: reply %+v, want every vector moved in the background, the last "+
					"look pulling none", c.extra, out)
			}
			t.Logf("migration: %d raw vectors in the background, no posting names n2 %.3f s after the flip",
				out.MigrateVectors, out.MigrateSeconds)
			cl.stopDataNode(n2)
			exactPass(t, cl, nil)
		})
	}
}
