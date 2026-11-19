package e2e

import (
	"context"
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	"rtier/internal/agent"
	"rtier/internal/config"
	"rtier/internal/nodeclient"
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

// TestFetchExtraOnlyWhereNodesLeave: with FetchExtra on (U9), more raw vectors follow the fetches
// on demand only where nodes go away. A scale-out turns it on nowhere: the nodes that join fetch
// only what queries need, whether the controller places them or a plan does. In the scale-in that
// returns a partition from n4 to n3, the nodes that stay fetch from n4 with extras until the drain
// ends, and every node has them off once the rescale returns. n3's drain is held while the queries
// run again, re-ranking 64 candidates each instead of every one their lists name (as n4's did): n3
// fetches from n4 what they need, and the extras bring more of the same lists, which n4 fetched
// for the deeper re-ranking. Every answer is exact, and so after n4 is stopped.
func TestFetchExtraOnlyWhereNodesLeave(t *testing.T) {
	f := getFixture(t)
	const extra = 4
	cl := newCluster(t, f, "lazy", 2, func(c *config.Controller) { c.FetchExtra = extra })
	cl.addNode("n1", query.TwoPhase{})
	cl.addNode("n2", query.TwoPhase{})
	if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
	}
	held, release := make(chan struct{}), make(chan struct{})
	var holding, releasing sync.Once
	letGo := func() { releasing.Do(func() { close(release) }) }
	t.Cleanup(letGo) // before the cluster stops, whatever failed
	n3 := cl.addNodeWith("n3", agent.Options{Strategy: query.TwoPhase{}, BeforeDrain: func() error {
		holding.Do(func() { close(held) })
		<-release
		return nil
	}})
	n4 := cl.addNode("n4", query.TwoPhase{})
	extras := func(a *agent.Agent) nodeclient.FetchExtraStats {
		nc := nodeclient.New(a.Info().NodeAddr, 1)
		defer nc.Close()
		in, err := nc.Info(cl.ctx)
		if err != nil {
			t.Fatal(err)
		}
		return in.Fetch.Extra
	}
	all := func() []*agent.Agent { return cl.agents }

	reconfigure(t, cl, protocol.RescaleReq{DataNodes: 3})
	t3 := cl.ctl.Status().Table
	owners := map[string][]int{}
	for id, ps := range t3.Placement.ByNode() {
		owners[t3.Nodes[id].Name] = ps
	}
	if len(owners["n3"]) == 0 {
		t.Fatalf("placement %v: n3 took no partition", owners)
	}
	moved := owners["n3"][0]
	four := map[string][]int{"n1": owners["n1"], "n2": owners["n2"], "n3": owners["n3"][1:], "n4": {moved}}
	reconfigure(t, cl, protocol.RescaleReq{Plan: &protocol.Plan{Owners: four}})
	exactPass(t, cl, nil) // n3 and n4 fetch what the queries need
	for _, a := range all() {
		if e := extras(a); e.N != 0 || e.Queued != 0 {
			t.Errorf("%s after two scale-outs: extras %+v, want none", a.Info().Name, e)
		}
	}

	const shallow = 64 // candidates re-ranked per query while n3's drain is held
	oaddr, stopOracle, err := startNode(f, "--load", "all", "--graph", "index")
	if err != nil {
		t.Fatal(err)
	}
	oracle := nodeclient.New(oaddr, 1)
	want := make([][]nodeclient.Candidate, numQueries)
	for qi := range want {
		if want[qi], err = oracle.SearchLocal(cl.ctx, 0, f.queries.Row(qi),
			nodeclient.SearchParams{K: testK, NProbe: testNProbe, Rerank: shallow}); err != nil {
			t.Fatal(err)
		}
	}
	oracle.Close()
	if err := stopOracle(); err != nil {
		t.Fatal(err)
	}
	passed := make(chan error, 1)
	var during nodeclient.FetchExtraStats
	go func() {
		<-held
		err := checkDepth(cl, want, shallow)
		nc := nodeclient.New(n3.Info().NodeAddr, 1)
		for i := 0; err == nil && i < 500 && during.Fetched == 0; i++ { // the extras come in the background
			time.Sleep(10 * time.Millisecond)
			var in *nodeclient.Info
			if in, err = nc.Info(cl.ctx); err == nil {
				during = in.Fetch.Extra
			}
		}
		nc.Close()
		letGo()
		passed <- err
	}()
	out := reconfigure(t, cl, protocol.RescaleReq{Plan: &protocol.Plan{Owners: owners}})
	select {
	case err := <-passed:
		if err != nil {
			t.Fatalf("while n3's drain was held: %v", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("n3's drain never started")
	}
	if !reflect.DeepEqual(out.Removed, []placement.NodeID{n4.ID()}) || !reflect.DeepEqual(out.Moved, []int{moved}) {
		t.Fatalf("scale-in reply %+v: want partition %d back from node %d", out, moved, n4.ID())
	}
	if during.N != extra || during.Queued == 0 || during.Requests == 0 || during.Fetched == 0 ||
		during.Fetched > extra*during.Requests || during.Failed != 0 {
		t.Errorf("n3 while its drain was held: extras %+v, want requests of at most %d vectors from n4", during, extra)
	}
	for _, a := range all() {
		if e := extras(a); e.N != 0 {
			t.Errorf("%s after the scale-in: extras %+v, want them off", a.Info().Name, e)
		} else if a != n3 && e.Queued != 0 {
			t.Errorf("%s: extras %+v, want none: it fetched nothing from n4", a.Info().Name, e)
		}
	}
	t.Logf("scale-in: n3's extras %+v; %d raw vectors drained from n4", during, out.DrainVectors)
	checkResidency(t, cl)
	cl.stopDataNode(n4)
	exactPass(t, cl, nil)
}

// checkDepth sends every query once, to the current entries in turn, re-ranking n candidates, and
// checks each answer against answers: the first that failed or was wrong.
func checkDepth(cl *cluster, answers [][]nodeclient.Candidate, n int) error {
	tbl := cl.ctl.Status().Table
	for qi := 0; qi < numQueries; qi++ {
		addr := tbl.Nodes[tbl.Entries[qi%len(tbl.Entries)]].QueryAddr
		c := query.NewClient(addr, 1)
		ctx, cancel := context.WithTimeout(cl.ctx, 10*time.Second)
		res, err := c.Query(ctx, &query.Request{Vec: cl.f.queries.Row(qi),
			Params: query.Params{K: testK, NProbe: testNProbe, N: n}})
		cancel()
		c.Close()
		if err != nil {
			return fmt.Errorf("query %d at %s: %v", qi, addr, err)
		}
		if !equal(res.Candidates, answers[qi]) {
			return fmt.Errorf("query %d at %s in epoch %d, re-ranking %d: got %v, want %v", qi, addr, res.Epoch, n,
				res.Candidates, answers[qi])
		}
	}
	return nil
}
