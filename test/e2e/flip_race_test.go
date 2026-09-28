package e2e

// Queries that straddle a flip. A query runs under one epoch -- the one its entry pinned --
// from routing to the last merge, whatever epoch its aggregator has installed by then: the
// epoch is installed on the nodes in parallel, so a node that joins may get it after the
// entries that forward to it, and a node that leaves gets it while queries admitted just
// before are still running. Both races are reproduced exactly here by holding queries (in the
// selector, right after their entry pinned the epoch and navigated) or an install.

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"rtier/internal/agent"
	"rtier/internal/config"
	"rtier/internal/epoch"
	"rtier/internal/placement"
	"rtier/internal/query"
)

// gatedEF is what EF = 0 means (2 x nprobe): the same answers, but it marks the queries a gate
// holds.
const gatedEF = 2 * testNProbe

// gate is a selector that holds the marked queries until it is opened, then sends them to
// target(self). Other queries follow the owner rule.
type gate struct {
	target  func(self placement.NodeID) placement.NodeID
	arrived chan struct{}
	open    chan struct{}
}

func newGate(target func(self placement.NodeID) placement.NodeID) *gate {
	return &gate{target: target, arrived: make(chan struct{}, 16), open: make(chan struct{})}
}

func (g *gate) Name() string { return "gate" }

func (g *gate) Pick(t *epoch.Table, self placement.NodeID, req *query.Request, groups []query.Group) (placement.NodeID, error) {
	if req.Params.EF != gatedEF {
		return query.OwnerSelector{}.Pick(t, self, req, groups)
	}
	g.arrived <- struct{}{}
	<-g.open
	return g.target(self), nil
}

func request(f *fixture, q int, ef int) *query.Request {
	return &query.Request{Vec: f.queries.Row(q), Params: query.Params{K: testK, NProbe: testNProbe, N: maxRerank, EF: ef}}
}

func waitUntil(t *testing.T, d time.Duration, what string, cond func() bool) {
	t.Helper()
	for end := time.Now().Add(d); !cond(); time.Sleep(2 * time.Millisecond) {
		if time.Now().After(end) {
			t.Fatalf("timed out waiting until %s", what)
		}
	}
}

func receive[T any](t *testing.T, ch <-chan T, what string) T {
	t.Helper()
	select {
	case v := <-ch:
		return v
	case <-time.After(60 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
		panic("unreachable")
	}
}

type outcome struct {
	q   int
	res query.Result
	err error
}

// TestJoiningNodeAggregatesBeforeInstall: during a scale-out, an entry that has installed the
// new epoch forwards a query to the joining node -- an aggregator in that epoch -- which has not
// installed it yet. The joining node's install is held back; the entry's query must still be
// answered, exactly, under the new epoch.
func TestJoiningNodeAggregatesBeforeInstall(t *testing.T) {
	f := getFixture(t)
	cl := newCluster(t, f, "lazy", 2, func(c *config.Controller) { c.InitialEntries = 1 })
	var joining atomic.Uint32
	g := newGate(func(placement.NodeID) placement.NodeID { return placement.NodeID(joining.Load()) })
	n1 := cl.addNodeWith("n1", agent.Options{Strategy: query.TwoPhase{}, Selector: g}) // the entry
	cl.addNode("n2", query.TwoPhase{})
	if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
	}
	e := cl.ctl.Status().Table.Epoch
	installing, held := make(chan uint64, 1), make(chan struct{})
	n3 := cl.addNodeWith("n3", agent.Options{Strategy: query.TwoPhase{}, BeforeInstall: func(ep uint64) {
		if ep == e+1 {
			installing <- ep
			<-held
		}
	}})
	joining.Store(uint32(n3.ID()))

	done := make(chan error, 1)
	go func() { _, err := cl.ctl.Rescale(cl.ctx, 3); done <- err }()
	receive(t, installing, "the joining node to start installing the new epoch")

	entry := query.NewClient(n1.Info().QueryAddr, 2)
	waitUntil(t, 10*time.Second, "the entry routes with the new epoch", func() bool {
		res, err := entry.Query(cl.ctx, request(f, 0, 0))
		return err == nil && res.Epoch == e+1
	})
	before := n1.Counter("entry.forwarded")
	out := make(chan outcome, 1)
	go func() {
		res, err := entry.Query(cl.ctx, request(f, 1, gatedEF))
		out <- outcome{1, res, err}
	}()
	receive(t, g.arrived, "the query to reach the entry's selector")
	close(g.open)
	o := receive(t, out, "the forwarded query")
	if o.err != nil {
		t.Fatalf("query forwarded to the joining node before its install: %v", o.err)
	}
	if o.res.Epoch != e+1 || !equal(o.res.Candidates, f.oracle[o.q]) {
		t.Fatalf("epoch %d, got %v, want epoch %d and %v", o.res.Epoch, o.res.Candidates, e+1, f.oracle[o.q])
	}
	if n1.Counter("entry.forwarded") == before {
		t.Fatal("the query was not forwarded to the joining node")
	}

	close(held)
	if err := receive(t, done, "the scale-out"); err != nil {
		t.Fatal(err)
	}
	if !cl.ctl.Status().Table.IsEntry(n3.ID()) {
		t.Error("the joining node did not become an entry")
	}
}

// TestLeavingNodeFinishesAdmittedQueries: during a scale-in, the leaving node installs the epoch
// without it while queries admitted under the previous one still need it -- one it took as an
// entry and aggregates itself, one another entry forwards to it. Both are held after
// navigating until the leaving node has installed the new epoch; both must be answered,
// exactly, under the epoch they were admitted in, and the scale-in must wait for them before
// it reclaims anything.
func TestLeavingNodeFinishesAdmittedQueries(t *testing.T) {
	f := getFixture(t)
	cl := newCluster(t, f, "lazy", 3)
	var leaving atomic.Uint32
	fwd := newGate(func(placement.NodeID) placement.NodeID { return placement.NodeID(leaving.Load()) })
	self := newGate(func(self placement.NodeID) placement.NodeID { return self })
	n1 := cl.addNodeWith("n1", agent.Options{Strategy: query.TwoPhase{}, Selector: fwd})
	cl.addNode("n2", query.TwoPhase{})
	n3 := cl.addNodeWith("n3", agent.Options{Strategy: query.TwoPhase{}, Selector: self}) // leaves: highest ID
	leaving.Store(uint32(n3.ID()))
	if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
	}
	e := cl.ctl.Status().Table.Epoch

	out := make(chan outcome, 2)
	ask := func(a *agent.Agent, q int) {
		go func() {
			res, err := query.NewClient(a.Info().QueryAddr, 1).Query(cl.ctx, request(f, q, gatedEF))
			out <- outcome{q, res, err}
		}()
	}
	ask(n3, 2) // aggregated by the leaving node itself
	receive(t, self.arrived, "the query at the leaving node")
	ask(n1, 3) // forwarded by n1 to the leaving node
	receive(t, fwd.arrived, "the query at n1")
	before := n1.Counter("entry.forwarded")

	done := make(chan error, 1)
	go func() { _, err := cl.ctl.Rescale(cl.ctx, 2); done <- err }()
	leavingClient := query.NewClient(n3.Info().QueryAddr, 1)
	waitUntil(t, 30*time.Second, "the leaving node installs the epoch without it", func() bool {
		_, err := leavingClient.Query(cl.ctx, request(f, 0, 0))
		return err != nil && strings.Contains(err.Error(), "not an entry")
	})
	time.Sleep(200 * time.Millisecond) // the flip is done; the grace period must hold
	select {
	case err := <-done:
		t.Fatalf("the scale-in finished while queries of epoch %d were running (%v)", e, err)
	default:
	}

	close(self.open)
	close(fwd.open)
	for i := 0; i < 2; i++ {
		o := receive(t, out, "the held queries")
		if o.err != nil {
			t.Fatalf("query %d, admitted in epoch %d: %v", o.q, e, o.err)
		}
		if o.res.Epoch != e || !equal(o.res.Candidates, f.oracle[o.q]) {
			t.Fatalf("query %d: epoch %d, got %v, want epoch %d and %v", o.q, o.res.Epoch, o.res.Candidates, e, f.oracle[o.q])
		}
	}
	if n1.Counter("entry.forwarded") != before+1 {
		t.Error("n1 did not forward its query to the leaving node")
	}
	if err := receive(t, done, "the scale-in"); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(cl.ctx, 10*time.Second)
	defer cancel()
	if _, err := leavingClient.Query(ctx, request(f, 0, 0)); err == nil {
		t.Error("the node that left still answers client queries")
	}
}
