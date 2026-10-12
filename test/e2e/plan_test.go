package e2e

import (
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"rtier/internal/placement"
	"rtier/internal/protocol"
	"rtier/internal/query"
)

// TestReconfigurePlan: reconfigurations to a placement that a plan names, under load. A dry
// run changes nothing. A rebalance at the same size trades partitions between the two data
// nodes, which no policy does. A replacement -- one node leaves and an idle one joins in the
// same reconfiguration -- hands the joining node the partitions of the leaving one, and its
// graph comes from the entry that stays. Answers stay exact throughout; plans that do not
// resolve (made for an earlier epoch, naming an unknown node, leaving a partition without an
// owner) are refused before anything moves.
func TestReconfigurePlan(t *testing.T) {
	f := getFixture(t)
	for _, proto := range []string{"lazy", "copy-then-flip"} {
		t.Run(proto, func(t *testing.T) {
			cl := newCluster(t, f, proto, 2)
			n1 := cl.addNode("n1", query.TwoPhase{})
			n2 := cl.addNode("n2", query.TwoPhase{})
			if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
				t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
			}
			n3 := cl.addNode("n3", query.TwoPhase{}) // registers idle
			cur := cl.ctl.Status().Table
			a, b := cur.Placement.ByNode()[n1.ID()], cur.Placement.ByNode()[n2.ID()]

			stop := startLoad(cl, 4)
			time.Sleep(200 * time.Millisecond)

			// The plan the policy's scale-out would carry out; nothing moves.
			dry, err := cl.ctl.Reconfigure(cl.ctx, protocol.RescaleReq{DataNodes: 3, DryRun: true})
			if err != nil {
				t.Fatal(err)
			}
			if !dry.DryRun || dry.Plan == nil || len(dry.Plan.Owners) != 3 || len(dry.Moved) == 0 ||
				dry.Plan.FromEpoch != cur.Epoch || dry.ToEpoch != cur.Epoch {
				t.Fatalf("dry run %+v: want the plan of a scale-out to 3 nodes for epoch %d", dry, cur.Epoch)
			}
			if now := cl.ctl.Status().Table; now.Epoch != cur.Epoch || !reflect.DeepEqual(now.Placement, cur.Placement) {
				t.Fatalf("the dry run moved epoch %d to %d", cur.Epoch, now.Epoch)
			}

			// Rebalance at the same size, nodes by name: n1's last partition and n2's first trade
			// places.
			pa, pb := a[len(a)-1], b[0]
			swap := &protocol.Plan{FromEpoch: cur.Epoch, Owners: map[string][]int{
				"n1": append(append([]int(nil), a[:len(a)-1]...), pb),
				"n2": append(append([]int(nil), b[1:]...), pa),
			}}
			out := reconfigure(t, cl, protocol.RescaleReq{Plan: swap})
			if len(out.Added) != 0 || len(out.Removed) != 0 || !reflect.DeepEqual(out.Moved, ascending(pa, pb)) ||
				out.ToEpoch != out.FromEpoch+1 || len(out.Warnings) != 0 {
				t.Errorf("rebalance reply %+v: want partitions %d and %d traded in one flip, nobody joining or leaving",
					out, pa, pb)
			}
			wantOwners(t, cl, map[placement.NodeID][]int{n1.ID(): swap.Owners["n1"], n2.ID(): swap.Owners["n2"]})

			// Replace n2 by n3, nodes by ID.
			byNode := cl.ctl.Status().Table.Placement.ByNode()
			repl := &protocol.Plan{Owners: map[string][]int{
				strconv.Itoa(int(n1.ID())): byNode[n1.ID()],
				strconv.Itoa(int(n3.ID())): byNode[n2.ID()],
			}}
			out = reconfigure(t, cl, protocol.RescaleReq{Plan: repl})
			if !reflect.DeepEqual(out.Added, []placement.NodeID{n3.ID()}) ||
				!reflect.DeepEqual(out.Removed, []placement.NodeID{n2.ID()}) {
				t.Errorf("replacement reply %+v: want node %d added and node %d removed", out, n3.ID(), n2.ID())
			}
			if src, ok := out.GraphSources[n3.ID()]; !ok || src != n1.ID() {
				t.Errorf("node %d copied the graph from %d (%v), want node %d: the entry that stays", n3.ID(), src, ok, n1.ID())
			}
			// With lazy, n2 stays the raw-vector source of every partition it hands over: n3 has
			// never held one.
			if wantWarn := proto == "lazy"; wantWarn != (len(out.Warnings) == 1) ||
				wantWarn && !strings.Contains(out.Warnings[0], fmt.Sprintf("node %d (n2) leaves", n2.ID())) {
				t.Errorf("replacement warnings %q: want one about node %d with lazy, none otherwise", out.Warnings, n2.ID())
			}
			tbl := cl.ctl.Status().Table
			if !tbl.IsEntry(n3.ID()) || tbl.IsEntry(n2.ID()) || tbl.IsAggregator(n2.ID()) {
				t.Errorf("epoch %d: entries %v, aggregators %v: want node %d an entry and node %d gone",
					tbl.Epoch, tbl.Entries, tbl.Aggregators, n3.ID(), n2.ID())
			}
			wantOwners(t, cl, map[placement.NodeID][]int{n1.ID(): byNode[n1.ID()], n3.ID(): byNode[n2.ID()]})

			// Refused before anything moves.
			before := cl.ctl.Status().Table.Epoch
			byNode = cl.ctl.Status().Table.Placement.ByNode()
			for name, req := range map[string]protocol.RescaleReq{
				"made for an earlier epoch": {Plan: swap},
				"unknown node": {Plan: &protocol.Plan{Owners: map[string][]int{
					"n1": byNode[n1.ID()], "n9": byNode[n3.ID()]}}},
				"partition without owner": {Plan: &protocol.Plan{Owners: map[string][]int{
					"n1": byNode[n1.ID()], "n3": byNode[n3.ID()][1:]}}},
				"neither size nor plan": {},
				"both size and plan":    {DataNodes: 2, Plan: repl},
			} {
				if _, err := cl.ctl.Reconfigure(cl.ctx, req); err == nil {
					t.Errorf("%s: accepted", name)
				} else {
					t.Logf("%s: %v", name, err)
				}
			}
			if e := cl.ctl.Status().Table.Epoch; e != before {
				t.Errorf("refused plans moved epoch %d to %d", before, e)
			}

			time.Sleep(200 * time.Millisecond)
			st := stop()
			checkResidency(t, cl)
			t.Logf("%d queries, %d retries (%v), per epoch %v", st.total, st.retries, st.retryWhy, st.epochs)
			if st.mismatches > 0 || st.errors > 0 {
				t.Fatalf("%d mismatches, %d errors; first: %s", st.mismatches, st.errors, st.firstProblem)
			}
			if st.total == 0 || len(st.epochs) < 3 {
				t.Fatalf("load did not span the reconfigurations: %v", st.epochs)
			}
		})
	}
}

func ascending(ps ...int) []int {
	out := append([]int(nil), ps...)
	sort.Ints(out)
	return out
}

// wantOwners checks the current placement against the partitions each node should own.
func wantOwners(t *testing.T, cl *cluster, want map[placement.NodeID][]int) {
	t.Helper()
	for id, ps := range want {
		want[id] = ascending(ps...)
	}
	if got := cl.ctl.Status().Table.Placement.ByNode(); !reflect.DeepEqual(got, want) {
		t.Errorf("placement %v, want %v", got, want)
	}
}
