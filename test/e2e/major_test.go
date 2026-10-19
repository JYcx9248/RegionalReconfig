package e2e

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"rtier/internal/config"
	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
	"rtier/internal/placement"
	"rtier/internal/protocol"
	"rtier/internal/query"
	"rtier/internal/update"
)

// TestMajorEpoch applies a major epoch (U14) to a cluster under query load, and the same
// decisions to a single-node oracle: a list split in two inside its partition, and two lists
// merged -- each of their postings sent to the list of its nearest live head, on whatever owner,
// with its PQ code, its raw vector following on demand from the node that holds it. The update
// controller's router edits its own graph and exports a patch, which every entry stages next to
// its graph. Every answer must equal the oracle's for its epoch: queries of the epoch before the
// flip search the graph alone and read the old lists, those after it search through the patch
// and read the new ones. After the grace period every entry has merged the patch and keeps it as
// a file. Then:
//
//  1. a minor batch routes its inserts on the router's graph, which has the new heads;
//  2. a scale-out moves partitions, the created lists with them; the joining entry loads the
//     build's graph and the patch from a peer;
//  3. a scale-in returns them, and once the node that left is stopped every answer is still
//     exact.
func TestMajorEpoch(t *testing.T) {
	f := getFixture(t)
	m, err := partitioning.LoadManifest(f.parts)
	if err != nil {
		t.Fatal(err)
	}
	maxLen := m.MaxListLen + 256
	extraIDs := 1024
	extraPages := uint64(extraIDs)/m.VectorsPerPage + 2
	room := []string{"--max-vectors", strconv.Itoa(numVectors + extraIDs),
		"--max-pages", strconv.FormatUint(m.NumPages+extraPages, 10), "--max-list-len", strconv.Itoa(maxLen)}
	cl := newCluster(t, f, "lazy", 2, func(c *config.Controller) { c.AllowUpdates = true })
	cl.addNode("n1", query.TwoPhase{}, room...)
	cl.addNode("n2", query.TwoPhase{}, room...)
	if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
	}
	n3 := cl.addNode("n3", query.TwoPhase{}, room...) // registers idle; joins at scale-out
	ctx := cl.ctx

	addr, stopOracle, err := startNode(f, append([]string{"--load", "all", "--graph", "index"}, room...)...)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := stopOracle(); err != nil {
			t.Error(err)
		}
	}()
	oracle := nodeclient.New(addr, 1)
	defer oracle.Close()
	in, err := oracle.Info(ctx)
	if err != nil {
		t.Fatal(err)
	}
	alloc, err := update.NewAllocator(numVectors, in.IDCapacity, m.Locations(), in.Raw.Locations)
	if err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(f.dir, fmt.Sprintf("router-%d.log", nodeSeq.Add(1)))
	logf, err := os.Create(logPath)
	if err != nil {
		t.Fatal(err)
	}
	raddr, stopRouter, err := update.StartRouter(filepath.Join(f.bin, "rtier_node"), f.indexNoPages, f.parts, logf)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		err := stopRouter()
		logf.Close()
		out, _ := os.ReadFile(logPath)
		if err != nil || strings.Contains(string(out), "Sanitizer") || strings.Contains(string(out), "runtime error:") {
			t.Errorf("router: %v (log %s)", err, logPath)
		}
	}()
	router := nodeclient.New(raddr, 2)
	defer router.Close()
	u := &update.Updater{Controller: cl.ctl, Manifest: m, Alloc: alloc, Router: router,
		GraphPath: filepath.Join(f.indexNoPages, "graph.hnsw"), PatchDir: t.TempDir()}
	defer u.Close()
	oracleVersions := make([]uint32, numPartitions)

	stop := startLoad(cl, 4)
	time.Sleep(200 * time.Millisecond)

	// Lists queries probe first: one to split, and lists to merge at the boundary between the
	// owners -- the first list of a query whose second list the other node owns -- so that some
	// postings move to another owner.
	tbl := cl.ctl.Status().Table
	owner := func(l uint32) placement.NodeID { return tbl.Owner(tbl.PartitionOf(l, m)) }
	nav := nodeclient.New(tbl.Nodes[tbl.Entries[0]].NodeAddr, 1)
	var split uint32
	var cands []uint32
	for qi := 0; qi < numQueries; qi++ {
		lists, err := nav.Navigate(ctx, tbl.Epoch, tbl.Graph, f.queries.Row(qi), testNProbe, 0)
		if err != nil {
			t.Fatal(err)
		}
		if qi == 0 {
			split = lists[0]
		} else if l := lists[0]; l != split && owner(l) != owner(lists[1]) && !containsU32(cands, l) {
			cands = append(cands, l)
		}
	}
	nav.Close()
	if len(cands) < 2 {
		t.Fatalf("only lists %v are at the owners' boundary", cands)
	}

	// The major epoch: decided under the lease (given up and decided again with other lists
	// until a posting moves to another owner), applied to the oracle, whose answers hold from
	// the flip on, then executed on the cluster.
	var mj *update.Major
	for i := 0; i+1 < len(cands); i += 2 {
		if mj, err = u.PrepareMajor(ctx, []uint32{split}, cands[i:i+2]); err != nil {
			t.Fatal(err)
		}
		if len(mj.Codes) > 0 {
			break
		}
		if err := u.AbortMajor(ctx, mj); err != nil {
			t.Fatal(err)
		}
		mj = nil
	}
	if mj == nil {
		t.Fatalf("no merge of the lists %v moves a posting to another owner", cands)
	}
	applyMajorOracle(t, ctx, oracle, mj, oracleVersions)
	from := mj.Lease.Table.Epoch
	cl.expect(from+1, oracleAnswers(t, ctx, f, oracle))
	reply, err := u.ExecuteMajor(ctx, mj)
	if err != nil {
		t.Fatalf("major epoch: %v", err)
	}
	sp := mj.Splits[0]
	moved := 0
	for _, cs := range mj.Codes {
		moved += len(cs.IDs)
	}
	t.Logf("major epoch: list %d split into %v (%v postings), lists %d and %d merged (%d and %d postings, %d moved "+
		"to other owners); epoch %d -> %d, graph %d, %d-byte patch, partitions %v, phases %v",
		sp.List, sp.Children, sp.Sizes, mj.Merges[0].List, mj.Merges[1].List, mj.Merges[0].Postings, mj.Merges[1].Postings,
		moved, reply.FromEpoch, reply.ToEpoch, reply.Graph, len(mj.Patch), reply.Changed, reply.Phases)
	if reply.FromEpoch != from || reply.ToEpoch != from+1 || reply.Graph != 1 || len(reply.Changed) == 0 {
		t.Fatalf("major epoch reply %+v: want a flip from epoch %d to graph version 1", reply, from)
	}
	after := cl.ctl.Status().Table
	for _, c := range sp.Children {
		if p, ok := after.Lists[c]; !ok || p != sp.Partition {
			t.Errorf("epoch %d lists %v: want list %d in partition %d", after.Epoch, after.Lists, c, sp.Partition)
		}
	}
	checkGraphs(t, cl, 1)

	// 1. A minor batch on the patched graph: copies of the first queries.
	var vecs []byte
	for i := 0; i < 20; i++ {
		vecs = append(vecs, f.queries.Row(i)...)
	}
	vb := len(f.queries.Row(0))
	b, err := u.Prepare(ctx, vecs, vb, nil)
	if err != nil {
		t.Fatal(err)
	}
	newLists := 0
	for _, x := range b.Inserts {
		for _, l := range x.Lists {
			if _, ok := after.Lists[l]; ok {
				newLists++
			}
		}
	}
	applyOracle(t, ctx, oracle, b, oracleVersions)
	cl.expect(b.Lease.Table.Epoch+1, oracleAnswers(t, ctx, f, oracle))
	if _, err := u.Execute(ctx, b); err != nil {
		t.Fatalf("minor batch after the major epoch: %v", err)
	}
	if len(b.Lengths) == 0 {
		t.Errorf("the minor batch reported no list lengths")
	}
	t.Logf("minor batch: %d inserts, %d of their postings in lists the major epoch created; lengths %d lists",
		len(b.Inserts), newLists, len(b.Lengths))

	// 2. A scale-out: the joining entry gets the build's graph and the patch.
	out, err := cl.ctl.Reconfigure(ctx, protocol.RescaleReq{DataNodes: 3})
	if err != nil {
		t.Fatalf("scale-out: %v", err)
	}
	t.Logf("scale-out: epoch %d -> %d, moved %v, entries %v", out.FromEpoch, out.ToEpoch, out.Moved, cl.ctl.Status().Table.Entries)
	if !cl.ctl.Status().Table.IsEntry(n3.ID()) {
		t.Errorf("node %d did not become an entry after the scale-out", n3.ID())
	}
	checkGraphs(t, cl, 1)
	time.Sleep(200 * time.Millisecond)

	// 3. A scale-in returns the partitions.
	out, err = cl.ctl.Reconfigure(ctx, protocol.RescaleReq{DataNodes: 2})
	if err != nil {
		t.Fatalf("scale-in: %v", err)
	}
	t.Logf("scale-in: epoch %d -> %d, moved %v, %d raw vectors drained", out.FromEpoch, out.ToEpoch, out.Moved, out.DrainVectors)
	time.Sleep(300 * time.Millisecond)
	st := stop()
	t.Logf("%d queries, %d retries (unavailable: %v), max latency %v, per epoch %v",
		st.total, st.retries, st.retryWhy, st.maxLatency, st.epochs)
	if st.mismatches > 0 || st.errors > 0 {
		t.Fatalf("%d mismatches, %d errors; first: %s", st.mismatches, st.errors, st.firstProblem)
	}
	if len(st.epochs) < 4 {
		t.Fatalf("load did not span the epochs: %v", st.epochs)
	}
	cl.stopDataNode(n3)
	exactPass(t, cl, nil)
}

// applyMajorOracle applies mj's decisions to the oracle, which holds every partition at
// versions[p]: the next version of every partition it changes, then the graph patch, merged at
// once (the oracle answers for the epoch after the flip only), then the replaced versions dropped.
func applyMajorOracle(t *testing.T, ctx context.Context, nc *nodeclient.Client, mj *update.Major, versions []uint32) {
	t.Helper()
	var ps []int
	for p := range mj.Parts {
		ps = append(ps, p)
	}
	sort.Ints(ps)
	var vs []nodeclient.VersionStage
	var e nodeclient.Edits
	for _, p := range ps {
		vs = append(vs, nodeclient.VersionStage{Partition: p, Base: versions[p], Next: versions[p] + 1})
		pe := mj.Edits(p)
		e.Lists, e.IDs, e.Locs = append(e.Lists, pe.Lists...), append(e.IDs, pe.IDs...), append(e.Locs, pe.Locs...)
		e.Holders = append(e.Holders, pe.Holders...)
		e.Created, e.Removed = append(e.Created, pe.Created...), append(e.Removed, pe.Removed...)
	}
	res, err := nc.StageVersions(ctx, vs, e)
	if err != nil {
		t.Fatal(err)
	}
	if err := nc.PatchStage(ctx, mj.Patch); err != nil {
		t.Fatal(err)
	}
	if err := nc.PatchMerge(ctx, mj.Graph); err != nil {
		t.Fatal(err)
	}
	for _, r := range res {
		if r.Changed {
			if _, err := nc.DropVersion(ctx, r.Partition, versions[r.Partition]); err != nil {
				t.Fatal(err)
			}
			versions[r.Partition]++
		}
	}
}

// checkGraphs: every entry's data node has its graph at version v with no patch staged, and
// keeps the patches 1..v as files next to the graph (where a later entry's graph transfer
// takes them).
func checkGraphs(t *testing.T, cl *cluster, v uint32) {
	t.Helper()
	tbl := cl.ctl.Status().Table
	if tbl.Graph != v {
		t.Errorf("epoch %d is at graph version %d, want %d", tbl.Epoch, tbl.Graph, v)
	}
	for _, id := range tbl.Entries {
		nc := nodeclient.New(tbl.Nodes[id].NodeAddr, 1)
		in, err := nc.Info(cl.ctx)
		nc.Close()
		if err != nil {
			t.Fatal(err)
		}
		if in.Graph.Version != v || in.Graph.Patch != 0 {
			t.Errorf("entry %d: graph %+v, want version %d and no patch staged", id, in.Graph, v)
		}
		for w := uint32(1); w <= v; w++ {
			if _, err := os.Stat(filepath.Join(cl.dirs[id], "graph", fmt.Sprintf("patch-%d.bin", w))); err != nil {
				t.Errorf("entry %d does not keep the patch to graph version %d: %v", id, w, err)
			}
		}
	}
}

func containsU32(s []uint32, x uint32) bool {
	for _, y := range s {
		if y == x {
			return true
		}
	}
	return false
}
