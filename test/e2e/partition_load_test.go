package e2e

import (
	"testing"
	"time"

	"rtier/internal/nodeclient"
	"rtier/internal/placement"
	"rtier/internal/protocol"
	"rtier/internal/query"
)

// TestPartitionLoad (branch partition-load): each data node counts its partitions' load --
// FILTER's lists and postings, RERANK's candidates and the pages it reads -- for the partitions it
// owns and no other. Two nodes answer every query once: each probed list counts once, for its
// partition; every candidate re-ranked counts once; and every page re-ranking reads counts for one
// partition. Then a lazy scale-out moves partitions to n3, which re-ranks the vectors it fetches
// from memory (a candidate counted, no page read), and from then on only the new owner counts a
// moved partition. Every answer is exact.
func TestPartitionLoad(t *testing.T) {
	f := getFixture(t)
	cl := newCluster(t, f, "lazy", 2)
	cl.addNode("n1", query.TwoPhase{})
	cl.addNode("n2", query.TwoPhase{})
	if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
	}
	n3 := cl.addNode("n3", query.TwoPhase{}) // registers idle

	infos := func() map[placement.NodeID]*nodeclient.Info {
		out := map[placement.NodeID]*nodeclient.Info{}
		for _, ns := range cl.ctl.Status().Nodes {
			c := nodeclient.New(ns.Info.NodeAddr, 1)
			in, err := c.Info(cl.ctx)
			c.Close()
			if err != nil {
				t.Fatal(err)
			}
			out[ns.ID] = in
		}
		return out
	}
	sum := func(v []uint64) (s uint64) {
		for _, x := range v {
			s += x
		}
		return s
	}
	sub := func(a, b []uint64) []uint64 {
		out := make([]uint64, len(a))
		for i := range a {
			out[i] = a[i] - b[i]
		}
		return out
	}
	// check: what each node counted between `from` and `to`, while one pass sent every query once
	// and `owners` was the placement.
	check := func(what string, from, to map[placement.NodeID]*nodeclient.Info, owners map[placement.NodeID][]int) {
		t.Helper()
		var lists uint64
		for id, in := range to {
			b, a := from[id].Partitions, in.Partitions
			for _, v := range [][]uint64{a.Lists, a.Postings, a.Reranked, a.Pages} {
				if len(v) != numPartitions {
					t.Fatalf("%s: node %d reports %d partitions, want %d", what, id, len(v), numPartitions)
				}
			}
			dl, dp, dr, dg := sub(a.Lists, b.Lists), sub(a.Postings, b.Postings), sub(a.Reranked, b.Reranked),
				sub(a.Pages, b.Pages)
			mine := map[int]bool{}
			for _, p := range owners[id] {
				mine[p] = true
			}
			for p := 0; p < numPartitions; p++ {
				if !mine[p] && dl[p]+dp[p]+dr[p]+dg[p] != 0 {
					t.Errorf("%s: node %d counted partition %d, which it does not own: lists %d, postings %d, "+
						"reranked %d, pages %d", what, id, p, dl[p], dp[p], dr[p], dg[p])
				}
			}
			reranked := in.Fetch.RerankDisk + in.Fetch.RerankMem - from[id].Fetch.RerankDisk - from[id].Fetch.RerankMem
			if sum(dg) != a.RerankPages-b.RerankPages || sum(dr) != reranked || a.Unattributed != 0 {
				t.Errorf("%s: node %d counted %d pages of %d read, %d candidates of %d re-ranked, %d unattributed",
					what, id, sum(dg), a.RerankPages-b.RerankPages, sum(dr), reranked, a.Unattributed)
			}
			lists += sum(dl)
			t.Logf("%s: node %d lists %v, reranked %v, pages %v", what, id, dl, dr, dg)
		}
		if lists != numQueries*testNProbe {
			t.Errorf("%s: %d lists counted, want %d (%d queries probing %d each)", what, lists,
				numQueries*testNProbe, numQueries, testNProbe)
		}
	}

	before := infos()
	exactPass(t, cl, nil)
	check("2 nodes", before, infos(), cl.ctl.Status().Table.Placement.ByNode())

	reconfigure(t, cl, protocol.RescaleReq{DataNodes: 3})
	before = infos()
	exactPass(t, cl, nil)
	after := infos()
	check("3 nodes", before, after, cl.ctl.Status().Table.Placement.ByNode())
	if mem := after[n3.ID()].Fetch.RerankMem - before[n3.ID()].Fetch.RerankMem; mem == 0 {
		t.Errorf("n3 re-ranked nothing from memory: want the vectors it fetched on demand")
	}
}
