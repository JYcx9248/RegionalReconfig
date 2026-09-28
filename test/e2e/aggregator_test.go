package e2e

import (
	"testing"
	"time"

	"rtier/internal/config"
	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
	"rtier/internal/placement"
	"rtier/internal/query"
)

// clients resolves node IDs to data-node clients for query.Env (the test's own aggregator).
type clients map[placement.NodeID]*nodeclient.Client

func (c clients) Client(id placement.NodeID) (*nodeclient.Client, error) { return c[id], nil }

// TestAggregatorFromOwners: the entry works out the query's owners before choosing who
// aggregates. With one entry (node 1) and two data nodes, every query whose probed lists all
// sit on node 2 must be forwarded there -- and only those -- and every answer must equal the
// one the entry would have computed by aggregating itself.
func TestAggregatorFromOwners(t *testing.T) {
	f := getFixture(t)
	cl := newCluster(t, f, "lazy", 2, func(c *config.Controller) { c.InitialEntries = 1 })
	agents := []interface{ Counter(string) int64 }{
		cl.addNode("n1", query.TwoPhase{}), cl.addNode("n2", query.TwoPhase{}),
	}
	if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
	}
	tbl := cl.ctl.Status().Table
	if len(tbl.Entries) != 1 || len(tbl.Placement.Nodes()) != 2 {
		t.Fatalf("want one entry and two data nodes, got entries %v, owners %v", tbl.Entries, tbl.Placement.Nodes())
	}
	m, err := partitioning.LoadManifest(f.parts)
	if err != nil {
		t.Fatal(err)
	}
	entry := tbl.Nodes[tbl.Entries[0]]
	nav := nodeclient.New(entry.NodeAddr, 1)
	nodes := clients{}
	for id, n := range tbl.Nodes {
		nodes[id] = nodeclient.New(n.NodeAddr, 2)
	}
	env := &query.Env{Epoch: tbl.Epoch, Nodes: nodes}
	qc := query.NewClient(entry.QueryAddr, 2)

	want := 0
	for q := 0; q < numQueries; q++ {
		for _, nprobe := range []int{1, 2, testNProbe} { // few lists: often on one node
			req := &query.Request{Vec: f.queries.Row(q), Params: query.Params{K: testK, NProbe: nprobe, N: maxRerank}}
			lists, err := nav.Navigate(cl.ctx, tbl.Epoch, req.Vec, nprobe, 0)
			if err != nil {
				t.Fatal(err)
			}
			groups := query.GroupByOwner(lists, m, tbl)
			if len(groups) == 1 && groups[0].Node != entry.ID {
				want++
			}
			local := *req
			local.Lists = lists
			ref, err := query.TwoPhase{}.Execute(cl.ctx, env, &local, groups)
			if err != nil {
				t.Fatal(err)
			}
			got, err := qc.Query(cl.ctx, req)
			if err != nil {
				t.Fatalf("query %d nprobe %d: %v", q, nprobe, err)
			}
			if !equal(got.Candidates, ref) {
				t.Errorf("query %d nprobe %d: got %v, aggregating at the entry gives %v", q, nprobe, got.Candidates, ref)
			}
		}
	}
	var forwarded int64
	for _, a := range agents {
		forwarded += a.Counter("entry.forwarded")
	}
	t.Logf("%d queries, %d with every list on the non-entry node, %d forwarded", 3*numQueries, want, forwarded)
	if want == 0 {
		t.Fatal("no query had all its lists on the other node; the forwarding path went untested")
	}
	if forwarded != int64(want) {
		t.Errorf("forwarded %d queries, want exactly the %d whose lists all sit on the other node", forwarded, want)
	}
}
