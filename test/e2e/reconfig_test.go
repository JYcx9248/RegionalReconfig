package e2e

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"rtier/internal/agent"
	"rtier/internal/config"
	"rtier/internal/design"
	"rtier/internal/nodeclient"
	"rtier/internal/placement"
	"rtier/internal/protocol"
	"rtier/internal/query"
)

type loadStats struct {
	total, mismatches, errors, retries int64
	retryWhy                           map[string]int64 // why a node answered "unavailable"
	epochs                             map[uint64]int64
	maxLatency                         time.Duration
	firstProblem                       string
}

// unavailableReason classifies an "unavailable" answer by the check that refused the query.
func unavailableReason(err error) string {
	msg := err.Error()
	for _, r := range []string{"not an entry", "not an aggregator", "no epoch installed", "admission paused"} {
		if strings.Contains(msg, r) {
			return r
		}
	}
	return msg
}

// startLoad runs closed-loop query workers against the current entries until stop is called.
// Every answer must equal the single-node oracle exactly.
func startLoad(cl *cluster, workers int) func() loadStats {
	ctx, cancel := context.WithCancel(cl.ctx)
	var (
		mu      sync.Mutex
		st      = loadStats{epochs: map[uint64]int64{}, retryWhy: map[string]int64{}}
		wg      sync.WaitGroup
		clients = map[string]*query.Client{}
	)
	client := func(addr string) *query.Client {
		mu.Lock()
		defer mu.Unlock()
		c, ok := clients[addr]
		if !ok {
			c = query.NewClient(addr, 4)
			clients[addr] = c
		}
		return c
	}
	problem := func(s string) {
		if st.firstProblem == "" {
			st.firstProblem = s
		}
	}
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed))
			for ctx.Err() == nil {
				qi := rng.Intn(numQueries)
				req := &query.Request{Vec: cl.f.queries.Row(qi),
					Params: query.Params{K: testK, NProbe: testNProbe, N: maxRerank}}
				start := time.Now()
				var res query.Result
				var err error
				var why []string
				for time.Since(start) < 10*time.Second && ctx.Err() == nil {
					tbl := cl.ctl.Status().Table // one snapshot: its entries are among its nodes
					n := tbl.Nodes[tbl.Entries[rng.Intn(len(tbl.Entries))]]
					res, err = client(n.QueryAddr).Query(ctx, req)
					if !errors.Is(err, query.ErrUnavailable) {
						break
					}
					why = append(why, unavailableReason(err))
					time.Sleep(2 * time.Millisecond)
				}
				if ctx.Err() != nil {
					return
				}
				lat := time.Since(start)
				mu.Lock()
				st.total++
				st.retries += int64(len(why))
				for _, w := range why {
					st.retryWhy[w]++
				}
				st.maxLatency = max(st.maxLatency, lat)
				if err != nil {
					st.errors++
					problem(fmt.Sprintf("query %d: %v", qi, err))
				} else {
					st.epochs[res.Epoch]++
					if !equal(res.Candidates, cl.f.oracle[qi]) {
						st.mismatches++
						problem(fmt.Sprintf("query %d in epoch %d: got %v want %v", qi, res.Epoch, res.Candidates, cl.f.oracle[qi]))
					}
				}
				mu.Unlock()
			}
		}(int64(w + 1))
	}
	return func() loadStats {
		cancel()
		wg.Wait()
		return st
	}
}

func equal(a, b []nodeclient.Candidate) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// checkResidency verifies that every node holds exactly the partitions the table assigns to it
// (old copies were reclaimed) and that idle nodes hold none. PQ codes: one per vector; the
// codes of the node's partitions are live, and those of partitions it owned before stay cached.
// Raw vectors likewise: the node holds the vector of every vector ID a partition it has owned
// names (each vector has exactly one location), and none is still pending -- a reconfiguration
// returns only once its raw-vector streams have ended.
//
// It also checks the cache invariant: nothing is ever evicted. A scale-out only takes
// partitions away from a node and a reversible scale-in only gives back what it took, so a
// node's resident set stays what it was at the initial cluster size -- which fits its budget
// by construction, or the initial deployment would not have started. The invariant holds as
// long as the cluster never shrinks below its initial size, the nodes that leave are the ones
// that joined, and placement does not move a partition to a node that never held it (which
// heat-aware placement, the open half of U3, would).
func checkResidency(t *testing.T, cl *cluster) {
	t.Helper()
	cl.observe()
	st := cl.ctl.Status()
	for _, ns := range st.Nodes {
		info, err := nodeclient.New(ns.Info.NodeAddr, 1).Info(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		want := st.Table.Placement.PartitionsOf(ns.ID)
		got := append([]int(nil), info.Resident...)
		sort.Ints(got)
		if fmt.Sprint(got) != fmt.Sprint(want) && !(len(got) == 0 && len(want) == 0) {
			t.Errorf("node %d (%s, idle=%v) holds %v, placement says %v", ns.ID, ns.Name, ns.Idle, got, want)
		}
		live, all := uint64(len(cl.f.idsOf(want))), uint64(len(cl.f.idsOf(cl.held.partitions(ns.ID))))
		if pq := info.PQ; pq.Live != live || pq.Staged != 0 || pq.Resident != all ||
			pq.Resident != pq.Live+pq.Cached || pq.Evicted != 0 {
			t.Errorf("node %d (%s) PQ store %+v: want %d live codes (its partitions), %d in all "+
				"(every partition it has owned) and no eviction", ns.ID, ns.Name, pq, live, all)
		}
		if raw := info.Raw; raw.Present != all || raw.Pending != 0 ||
			raw.Present != raw.FromIndex+raw.Streamed+raw.Fetched {
			t.Errorf("node %d (%s) raw vectors %+v: want the %d vectors of every partition it has owned, "+
				"each installed once, none pending", ns.ID, ns.Name, raw, all)
		}
	}
}

// pqExpected is the number of PQ codes a reconfiguration from old to next has to pull: for each
// node, the vectors named by its new partitions that no partition it has ever owned named (it
// keeps those codes). perPartition is what shipping every partition with its own codes would
// move instead.
func pqExpected(f *fixture, held heldTracker, old, next placement.Table) (dedup, perPartition int64) {
	for _, n := range next.Nodes() {
		had := f.idsOf(held.partitions(n))
		var incoming []int
		for _, p := range next.PartitionsOf(n) {
			if old.Owners[p] != n {
				incoming = append(incoming, p)
				perPartition += int64(len(f.partIDs[p]))
			}
		}
		for id := range f.idsOf(incoming) {
			if _, ok := had[id]; !ok {
				dedup++
			}
		}
	}
	return dedup, perPartition
}

// rawStats reads every registered node's raw-vector counters.
func rawStats(t *testing.T, cl *cluster) map[placement.NodeID]nodeclient.RawStats {
	t.Helper()
	out := map[placement.NodeID]nodeclient.RawStats{}
	for _, ns := range cl.ctl.Status().Nodes {
		info, err := nodeclient.New(ns.Info.NodeAddr, 1).Info(cl.ctx)
		if err != nil {
			t.Fatal(err)
		}
		out[ns.ID] = info.Raw
	}
	return out
}

// rescale runs a reconfiguration and checks that exactly the missing PQ codes were pulled, and
// exactly the missing raw vectors moved -- streamed, or fetched on demand by queries (lazy
// protocol only).
func rescale(t *testing.T, cl *cluster, dataNodes int) *protocol.RescaleReply {
	t.Helper()
	cl.observe()
	before := cl.ctl.Status().Table.Placement
	rawBefore := rawStats(t, cl)
	out, err := cl.ctl.Rescale(cl.ctx, dataNodes)
	if err != nil {
		t.Fatalf("rescale to %d data nodes: %v", dataNodes, err)
	}
	want, naive := pqExpected(cl.f, cl.held, before, cl.ctl.Status().Table.Placement)
	cl.observe()
	var streamed, fetched uint64
	for id, r := range rawStats(t, cl) {
		streamed += r.Streamed - rawBefore[id].Streamed
		fetched += r.Fetched - rawBefore[id].Fetched
	}
	t.Logf("rescale to %d: epoch %d -> %d, moved %v, %d bytes, %d PQ codes (per-partition copies: %d), "+
		"raw vectors %d streamed + %d fetched on demand, phases %v, raw stream %.3f s after the flip",
		dataNodes, out.FromEpoch, out.ToEpoch, out.Moved, out.Bytes, out.PQCodes, naive, streamed, fetched,
		out.Phases, out.RawSeconds)
	if out.PQCodes != want {
		t.Errorf("rescale to %d pulled %d PQ codes, want %d (the vectors new to each destination)", dataNodes, out.PQCodes, want)
	}
	// A vector new to a destination needs its code and its raw vector: the same count. The
	// stream may bring one that a query fetched while its batch was in flight (installed once).
	if streamed+fetched != uint64(want) || uint64(out.RawVectors) < streamed {
		t.Errorf("rescale to %d installed %d raw vectors (%d streamed, reply received %d; %d fetched), want %d",
			dataNodes, streamed+fetched, streamed, out.RawVectors, fetched, want)
	}
	if out.Protocol != "lazy" && (fetched != 0 || uint64(out.RawVectors) != streamed) {
		t.Errorf("protocol %s: %d raw vectors fetched on demand, %d of %d streamed installed: want every one "+
			"copied before the flip, once", out.Protocol, fetched, streamed, out.RawVectors)
	}
	// Every data node of the new epoch holds all the codes its partitions name: none of them
	// went online before its codes had arrived.
	tbl := cl.ctl.Status().Table
	for _, id := range tbl.Placement.Nodes() {
		info, err := nodeclient.New(tbl.Nodes[id].NodeAddr, 1).Info(cl.ctx)
		if err != nil {
			t.Fatal(err)
		}
		if n := uint64(len(cl.f.idsOf(tbl.Placement.PartitionsOf(id)))); info.PQ.Live != n {
			t.Errorf("after rescale to %d: node %d has %d live PQ codes, its partitions name %d", dataNodes, id, info.PQ.Live, n)
		}
	}
	return out
}

func TestReconfigUnderLoad(t *testing.T) {
	f := getFixture(t)
	for _, protocol := range []string{"lazy", "copy-then-flip", "stop-and-copy"} {
		t.Run(protocol, func(t *testing.T) {
			cl := newCluster(t, f, protocol, 2)
			cl.addNode("n1", query.TwoPhase{})
			cl.addNode("n2", query.TwoPhase{})
			if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
				t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
			}
			cl.addNode("n3", query.TwoPhase{}) // registers idle; joins at scale-out
			checkResidency(t, cl)

			stop := startLoad(cl, 4)
			time.Sleep(300 * time.Millisecond)
			out := rescale(t, cl, 3)
			if out.ToEpoch != out.FromEpoch+2 || len(out.Moved) == 0 || out.Bytes == 0 || out.PQCodes == 0 {
				t.Errorf("scale-out reply %+v: want a data flip and an entry flip, moved partitions and PQ codes", out)
			}
			time.Sleep(300 * time.Millisecond)
			checkResidency(t, cl)
			if !cl.ctl.Status().Table.IsEntry(out.Added[0]) {
				t.Error("new node did not become an entry")
			}
			// The new node got every PQ code and raw vector over the network from the old owners,
			// once (it has no page file to read them from).
			newInfo, err := nodeclient.New(cl.ctl.Status().Table.Nodes[out.Added[0]].NodeAddr, 1).Info(cl.ctx)
			if err != nil {
				t.Fatal(err)
			}
			if pq := newInfo.PQ; pq.FromIndex != 0 || pq.Received != pq.Resident || pq.Skipped != 0 {
				t.Errorf("new node's PQ store %+v: want every code received from peers exactly once", pq)
			}
			if raw := newInfo.Raw; raw.FromIndex != 0 || raw.Present != newInfo.PQ.Resident {
				t.Errorf("new node's raw vectors %+v: want one per PQ code, all from peers", raw)
			}

			rescale(t, cl, 2) // scale-in: remaining nodes pull only codes they do not have
			time.Sleep(300 * time.Millisecond)
			checkResidency(t, cl) // the removed node keeps its codes, all cached

			// Scale out again: the node removed above rejoins and must serve as an entry (with
			// stop-and-copy it was paused while it left).
			again := rescale(t, cl, 3)
			tbl := cl.ctl.Status().Table
			if len(again.Added) != 1 || !tbl.IsEntry(again.Added[0]) {
				t.Fatalf("second scale-out reply %+v: rejoined node is not an entry", again)
			}
			res, err := query.NewClient(tbl.Nodes[again.Added[0]].QueryAddr, 1).Query(cl.ctx,
				&query.Request{Vec: f.queries.Row(0), Params: query.Params{K: testK, NProbe: testNProbe, N: maxRerank}})
			if err != nil || !equal(res.Candidates, f.oracle[0]) {
				t.Fatalf("query at the rejoined entry: %v", err)
			}
			time.Sleep(300 * time.Millisecond)
			st := stop()
			checkResidency(t, cl)

			t.Logf("%d queries, %d retries (unavailable: %v), max latency %v, per epoch %v",
				st.total, st.retries, st.retryWhy, st.maxLatency, st.epochs)
			if st.mismatches > 0 || st.errors > 0 {
				t.Fatalf("%d mismatches, %d errors; first: %s", st.mismatches, st.errors, st.firstProblem)
			}
			if st.total == 0 || len(st.epochs) < 3 {
				t.Fatalf("load did not span the reconfigurations: %v", st.epochs)
			}
		})
	}
}

// TestScaleOutFromOneNode is the smallest capacity-driven scale-out, and the path the local
// 1 -> 2 experiment takes: one data node owns every partition and answers every query, then a
// second node joins under load. Answers must stay exact across the flip, the new node must
// come up holding every PQ code its partitions name, and it must take over part of the data.
func TestScaleOutFromOneNode(t *testing.T) {
	f := getFixture(t)
	cl := newCluster(t, f, "lazy", 1)
	cl.addNode("n1", query.TwoPhase{})
	if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
	}
	cl.addNode("n2", query.TwoPhase{}) // registers idle; joins at the scale-out
	checkResidency(t, cl)
	solo := cl.ctl.Status().Table.Placement
	if len(solo.Nodes()) != 1 || len(solo.PartitionsOf(solo.Nodes()[0])) != numPartitions {
		t.Fatalf("one-node epoch: %d nodes, placement %v", len(solo.Nodes()), solo.Owners)
	}

	stop := startLoad(cl, 4)
	time.Sleep(300 * time.Millisecond)
	out := rescale(t, cl, 2)
	time.Sleep(300 * time.Millisecond)
	checkResidency(t, cl)
	st := stop()

	if len(out.Added) != 1 || len(out.Moved) == 0 || out.PQCodes == 0 {
		t.Fatalf("scale-out reply %+v: want one node added, partitions moved and PQ codes pulled", out)
	}
	tbl := cl.ctl.Status().Table
	added := out.Added[0]
	if got := len(tbl.Placement.PartitionsOf(added)); got == 0 || got == numPartitions {
		t.Errorf("the new node owns %d of %d partitions", got, numPartitions)
	}
	if !tbl.IsEntry(added) {
		t.Error("new node did not become an entry")
	}
	info, err := nodeclient.New(tbl.Nodes[added].NodeAddr, 1).Info(cl.ctx)
	if err != nil {
		t.Fatal(err)
	}
	if pq := info.PQ; pq.FromIndex != 0 || pq.Received != pq.Resident || pq.Skipped != 0 {
		t.Errorf("new node's PQ store %+v: want every code received from the first node exactly once", pq)
	}
	if raw := info.Raw; raw.FromIndex != 0 || raw.Present != info.PQ.Resident || raw.Streamed+raw.Fetched != raw.Present {
		t.Errorf("new node's raw vectors %+v: want one per PQ code, streamed or fetched from the first node", raw)
	}
	t.Logf("1 -> 2 under load: %d queries, %d retries (%v), max latency %v, per epoch %v",
		st.total, st.retries, st.retryWhy, st.maxLatency, st.epochs)
	if st.mismatches > 0 || st.errors > 0 {
		t.Fatalf("%d mismatches, %d errors; first: %s", st.mismatches, st.errors, st.firstProblem)
	}
	if st.total == 0 || len(st.epochs) < 2 {
		t.Fatalf("load did not span the reconfiguration: %v", st.epochs)
	}
}

// TestGraphSourcesRoundRobin: two nodes join at once. Each copies the navigation graph from a
// different existing entry -- assigned when the reconfiguration starts, round-robin -- instead
// of both pulling from the first one; both become entries, and answers stay exact throughout.
func TestGraphSourcesRoundRobin(t *testing.T) {
	f := getFixture(t)
	cl := newCluster(t, f, "lazy", 2)
	cl.addNode("n1", query.TwoPhase{})
	cl.addNode("n2", query.TwoPhase{})
	if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
	}
	cl.addNode("n3", query.TwoPhase{})
	cl.addNode("n4", query.TwoPhase{})
	entries := cl.ctl.Status().Table.Entries

	stop := startLoad(cl, 4)
	time.Sleep(300 * time.Millisecond)
	out := rescale(t, cl, 4)
	time.Sleep(300 * time.Millisecond)
	st := stop()
	checkResidency(t, cl)

	if len(out.Added) != 2 || len(out.GraphSources) != 2 {
		t.Fatalf("scale-out reply %+v: want two new nodes, each with a graph source", out)
	}
	used := map[placement.NodeID]bool{}
	for _, n := range out.Added {
		src, ok := out.GraphSources[n]
		isEntry := false
		for _, e := range entries {
			isEntry = isEntry || e == src
		}
		if !ok || !isEntry {
			t.Errorf("new node %d copied the graph from %d (ok=%v), not from an entry of %v", n, src, ok, entries)
		}
		used[src] = true
	}
	if len(used) != 2 {
		t.Errorf("both new nodes copied the graph from the same entry: %v", out.GraphSources)
	}
	tbl := cl.ctl.Status().Table
	for _, n := range out.Added {
		if !tbl.IsEntry(n) {
			t.Errorf("new node %d did not become an entry", n)
		}
	}
	t.Logf("2 -> 4 under load: graph sources %v, %d queries, %d retries (%v), per epoch %v",
		out.GraphSources, st.total, st.retries, st.retryWhy, st.epochs)
	if st.mismatches > 0 || st.errors > 0 {
		t.Fatalf("%d mismatches, %d errors; first: %s", st.mismatches, st.errors, st.firstProblem)
	}
}

// TestEntriesJoinAsGraphsLoad: three nodes join; one's graph loads at once, one's is held back,
// one's fails. The first must become an entry while the second is still waiting -- not after
// the slowest graph -- the second once its graph loads, in a later flip; the third stays a data
// node and aggregator and is reported, without holding up the others. Answers stay exact.
func TestEntriesJoinAsGraphsLoad(t *testing.T) {
	f := getFixture(t)
	cl := newCluster(t, f, "lazy", 2)
	cl.addNode("n1", query.TwoPhase{})
	cl.addNode("n2", query.TwoPhase{})
	if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
	}
	quick := cl.addNode("n3", query.TwoPhase{})
	release := make(chan struct{})
	slow := cl.addNodeWith("n4", agent.Options{Strategy: query.TwoPhase{}, BeforeGraph: func() error {
		<-release
		return nil
	}})
	broken := cl.addNodeWith("n5", agent.Options{Strategy: query.TwoPhase{}, BeforeGraph: func() error {
		return errors.New("injected: no graph")
	}})

	cl.observe()
	stop := startLoad(cl, 2)
	type result struct {
		out *protocol.RescaleReply
		err error
	}
	done := make(chan result, 1)
	go func() {
		out, err := cl.ctl.Rescale(cl.ctx, 5)
		done <- result{out, err}
	}()
	waitUntil(t, 60*time.Second, "the node with the quick graph is an entry", func() bool {
		return cl.ctl.Status().Table.IsEntry(quick.ID())
	})
	if tbl := cl.ctl.Status().Table; tbl.IsEntry(slow.ID()) || tbl.IsEntry(broken.ID()) {
		t.Fatalf("entries %v: only the node whose graph loaded should be one", tbl.Entries)
	}
	close(release)
	r := receive(t, done, "the scale-out")
	time.Sleep(200 * time.Millisecond)
	st := stop()

	if r.err == nil || !strings.Contains(r.err.Error(), "injected: no graph") {
		t.Fatalf("scale-out with a failing graph: got %v, want the failure reported", r.err)
	}
	if _, ok := r.out.EntryFailed[broken.ID()]; !ok || len(r.out.EntryFailed) != 1 {
		t.Errorf("entry failures %v: want exactly node %d", r.out.EntryFailed, broken.ID())
	}
	if len(r.out.EntryFlips) != 2 || r.out.EntryFlips[0].Nodes[0] != quick.ID() || r.out.EntryFlips[1].Nodes[0] != slow.ID() {
		t.Errorf("entry flips %+v: want node %d, then node %d in a later flip", r.out.EntryFlips, quick.ID(), slow.ID())
	}
	tbl := cl.ctl.Status().Table
	if !tbl.IsEntry(quick.ID()) || !tbl.IsEntry(slow.ID()) || tbl.IsEntry(broken.ID()) {
		t.Errorf("entries %v: want nodes %d and %d, not %d", tbl.Entries, quick.ID(), slow.ID(), broken.ID())
	}
	if !tbl.IsAggregator(broken.ID()) || len(tbl.Placement.PartitionsOf(broken.ID())) == 0 {
		t.Errorf("node %d, whose graph failed, must still be a data node and an aggregator", broken.ID())
	}
	checkResidency(t, cl)
	t.Logf("2 -> 5 under load: entry flips %+v, %d queries, %d retries (%v), per epoch %v",
		r.out.EntryFlips, st.total, st.retries, st.retryWhy, st.epochs)
	if st.mismatches > 0 || st.errors > 0 {
		t.Fatalf("%d mismatches, %d errors; first: %s", st.mismatches, st.errors, st.firstProblem)
	}
}

// TestReversiblePlacementReturnsHome: with the even-reversible policy, scaling out and back in
// puts every partition where it was, so the scale-in moves no PQ code and no raw vector at all
// -- the returning partitions name vectors their owner kept cached after giving them up.
func TestReversiblePlacementReturnsHome(t *testing.T) {
	f := getFixture(t)
	cl := newCluster(t, f, "lazy", 2, func(c *config.Controller) { c.Placement = "even-reversible" })
	cl.addNode("n1", query.TwoPhase{})
	cl.addNode("n2", query.TwoPhase{})
	if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
	}
	cl.addNode("n3", query.TwoPhase{})
	before := cl.ctl.Status().Table.Placement.Clone()

	stop := startLoad(cl, 4)
	time.Sleep(200 * time.Millisecond)
	out := rescale(t, cl, 3)
	time.Sleep(200 * time.Millisecond)
	in := rescale(t, cl, 2)
	time.Sleep(200 * time.Millisecond)
	checkResidency(t, cl)
	st := stop()

	if after := cl.ctl.Status().Table.Placement; !reflect.DeepEqual(after.Owners, before.Owners) {
		t.Errorf("after the round trip the placement is %v, want %v", after.Owners, before.Owners)
	}
	if out.PQCodes == 0 {
		t.Error("the scale-out pulled no PQ codes")
	}
	if in.PQCodes != 0 || in.RawVectors != 0 {
		t.Errorf("the scale-in pulled %d PQ codes and %d raw vectors, want none: every partition went home",
			in.PQCodes, in.RawVectors)
	}
	if st.mismatches > 0 || st.errors > 0 {
		t.Fatalf("%d mismatches, %d errors; first: %s", st.mismatches, st.errors, st.firstProblem)
	}
	t.Logf("round trip under load: %d queries, scale-out pulled %d codes, scale-in %d",
		st.total, out.PQCodes, in.PQCodes)
}

// TestStagingFailsCleanly: the joining node's PQ budget is too small for the partitions it
// would receive, so it can never hold all their codes. The scale-out must fail before the flip
// (the node never goes online), leave the old epoch serving exact answers, and leave no
// partition or PQ code behind on the joining node.
func TestStagingFailsCleanly(t *testing.T) {
	f := getFixture(t)
	cl := newCluster(t, f, "lazy", 2)
	cl.addNode("n1", query.TwoPhase{})
	cl.addNode("n2", query.TwoPhase{})
	if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
	}
	small := cl.addNode("small", query.TwoPhase{}, "--pq-capacity", "100")
	before := cl.ctl.Status().Table.Epoch

	stop := startLoad(cl, 2)
	time.Sleep(200 * time.Millisecond)
	_, rerr := cl.ctl.Rescale(cl.ctx, 3)
	time.Sleep(200 * time.Millisecond)
	st := stop()
	if rerr == nil || !strings.Contains(rerr.Error(), "PQ capacity exceeded") {
		t.Fatalf("scale-out onto a node with a 100-code budget: got %v, want a PQ capacity error", rerr)
	}
	if e := cl.ctl.Status().Table.Epoch; e != before {
		t.Fatalf("epoch moved from %d to %d after a failed staging", before, e)
	}
	checkResidency(t, cl) // the small node is idle: no partitions, no PQ codes
	info, err := nodeclient.New(small.Info().NodeAddr, 1).Info(cl.ctx)
	if err != nil {
		t.Fatal(err)
	}
	if info.PQ.Resident != 0 || info.PQ.Staged != 0 || len(info.Resident) != 0 {
		t.Fatalf("the small node kept PQ codes %+v and partitions %v", info.PQ, info.Resident)
	}
	// It never went online: no role and no partition in the table.
	if tbl := cl.ctl.Status().Table; tbl.IsEntry(small.ID()) || tbl.IsAggregator(small.ID()) ||
		len(tbl.Placement.PartitionsOf(small.ID())) != 0 {
		t.Fatalf("node %d is part of epoch %d although its staging failed", small.ID(), tbl.Epoch)
	}
	if st.mismatches > 0 || st.errors > 0 || st.total == 0 {
		t.Fatalf("%d queries, %d mismatches, %d errors; first: %s", st.total, st.mismatches, st.errors, st.firstProblem)
	}
	t.Logf("failed scale-out: %v; %d queries answered exactly meanwhile", rerr, st.total)
}

func TestPlaceholdersFailLoudly(t *testing.T) {
	f := getFixture(t)

	// No query strategy configured (U5): the node deploys, queries fail with ErrUndecided.
	cl := newCluster(t, f, "lazy", 1)
	cl.addNode("solo", nil)
	if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("deployment: %v (%s)", err, cl.ctl.LastError())
	}
	c := cl.entryClients()[0]
	_, err := c.Query(cl.ctx, &query.Request{Vec: f.queries.Row(0), Params: query.Params{K: 10, NProbe: 16, N: 200}})
	if !errors.Is(err, design.ErrUndecided) {
		t.Fatalf("query without a strategy: got %v, want ErrUndecided", err)
	}

	// Adapting the transfer rate to foreground latency (U8) is a placeholder: an agent asked
	// to do it refuses to start.
	cfg := config.DefaultAgent()
	cfg.Controller, cfg.NodeAddr, cfg.PartitionsDir, cfg.WorkDir = "127.0.0.1:1", "127.0.0.1:1", f.parts, t.TempDir()
	cfg.TransferAdapter = "adaptive"
	if _, err := agent.New(cfg, agent.Options{}); !errors.Is(err, design.ErrUndecided) {
		t.Fatalf("agent with the adaptive rate: got %v, want ErrUndecided", err)
	}
}

// TestRawVectorsFetchedOnDemand: with the lazy protocol a new node owns its partitions from the
// flip on without their raw vectors (U9). Its stream is held back, so every candidate it
// re-ranks must first be fetched from the old owner -- and the answers stay exact. Once the
// stream is released the node holds every vector, and the rescale reports what it streamed.
func TestRawVectorsFetchedOnDemand(t *testing.T) {
	f := getFixture(t)
	cl := newCluster(t, f, "lazy", 1)
	old := cl.addNode("n1", query.TwoPhase{})
	if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
	}
	release := make(chan struct{})
	fresh := cl.addNodeWith("n2", agent.Options{Strategy: query.TwoPhase{}, BeforeRaw: func() error {
		<-release
		return nil
	}})
	info := func(a *agent.Agent) *nodeclient.Info {
		in, err := nodeclient.New(a.Info().NodeAddr, 1).Info(cl.ctx)
		if err != nil {
			t.Fatal(err)
		}
		return in
	}

	cl.observe()
	stop := startLoad(cl, 4)
	time.Sleep(200 * time.Millisecond)
	type result struct {
		out *protocol.RescaleReply
		err error
	}
	done := make(chan result, 1)
	go func() {
		out, err := cl.ctl.Rescale(cl.ctx, 2)
		done <- result{out, err}
	}()
	waitUntil(t, 60*time.Second, "the new node fetches raw vectors on demand", func() bool {
		return info(fresh).Raw.Fetched > 0
	})
	time.Sleep(300 * time.Millisecond) // more queries re-rank on the fetch path
	mid := info(fresh).Raw
	if mid.Streamed != 0 || mid.Pending == 0 || mid.FromIndex != 0 {
		t.Errorf("while the stream is held: raw vectors %+v, want only on-demand fetches and some pending", mid)
	}
	close(release)
	r := receive(t, done, "the scale-out")
	time.Sleep(200 * time.Millisecond)
	st := stop()
	if r.err != nil {
		t.Fatal(r.err)
	}
	checkResidency(t, cl)

	end := info(fresh).Raw
	if r.out.RawVectors == 0 || uint64(r.out.RawVectors) < end.Streamed || end.Pending != 0 ||
		end.Present != end.Streamed+end.Fetched || end.Fetched < mid.Fetched || end.Fetches == 0 {
		t.Errorf("after the scale-out: raw vectors %+v, reply streamed %d: want the rest streamed, none pending",
			end, r.out.RawVectors)
	}
	if served := info(old).Raw.Served; served < end.Present {
		t.Errorf("the old owner served %d raw vectors, the new node holds %d", served, end.Present)
	}
	t.Logf("1 -> 2, stream held: %d vectors fetched on demand in %d round trips, %d streamed %.2f s after the "+
		"flip; %d queries, %d retries (%v), max latency %v, per epoch %v",
		end.Fetched, end.Fetches, end.Streamed, r.out.RawSeconds, st.total, st.retries, st.retryWhy, st.maxLatency, st.epochs)
	if st.mismatches > 0 || st.errors > 0 || st.total == 0 {
		t.Fatalf("%d queries, %d mismatches, %d errors; first: %s", st.total, st.mismatches, st.errors, st.firstProblem)
	}
}

// TestNodeShutdownUnderLoad interrupts rtier_node while requests are in flight. The node must
// exit cleanly (with a sanitizer build of the engine, without reports): connection threads
// finish before the engine is destroyed.
func TestNodeShutdownUnderLoad(t *testing.T) {
	f := getFixture(t)
	addr, stop, err := startNode(f, "--load", "all", "--graph", "index")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			nc := nodeclient.New(addr, 1)
			defer nc.Close()
			for i := 0; ctx.Err() == nil; i++ {
				q := f.queries.Row((w + i) % numQueries)
				lists, err := nc.Navigate(ctx, 1, q, testNProbe, 0)
				if err != nil {
					return // the node is going away
				}
				if _, _, err := nc.Filter(ctx, 1, q, lists, maxRerank); err != nil {
					return
				}
			}
		}(w)
	}
	time.Sleep(300 * time.Millisecond)
	err = stop()
	cancel()
	wg.Wait()
	if err != nil {
		t.Fatal(err)
	}
}
