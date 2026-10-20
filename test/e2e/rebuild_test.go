package e2e

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
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

// generation2 is a second generation of the fixture's index (U14 full rebuild): re-clustered with
// another seed and cut into another number of partitions, with the first generation's codebook,
// PQ codes and page layout -- which fusion_build reuses when those files are already in its
// output directory -- so that vector IDs, codes and raw-vector locations stay.
type generation2 struct {
	index, parts string
	partitions   int
	oracle       [][]nodeclient.Candidate // the answers of a single node built on it
}

var (
	gen2Once sync.Once
	gen2     *generation2
	gen2Err  error
)

func getGeneration2(t *testing.T, f *fixture) *generation2 {
	t.Helper()
	gen2Once.Do(func() { gen2, gen2Err = buildGeneration2(f) })
	if gen2Err != nil {
		t.Fatal(gen2Err)
	}
	return gen2
}

func buildGeneration2(f *fixture) (*generation2, error) {
	g := &generation2{index: filepath.Join(f.dir, "index-g2"), parts: filepath.Join(f.dir, "parts-g2"), partitions: 6}
	if err := os.MkdirAll(g.index, 0o755); err != nil {
		return nil, err
	}
	for _, name := range []string{"pq_codebook.bin", "pq_codes.bin", "layout_bucket.map", "vectors_bucket.bin"} {
		if err := os.Symlink(filepath.Join(f.index, name), filepath.Join(g.index, name)); err != nil {
			return nil, err
		}
	}
	if out, err := exec.Command(filepath.Join(f.bin, "fusion_build"), "--base", filepath.Join(f.dir, "base.fbin"),
		"--out", g.index, "--pq-m", "8", "--pq-train", "4000", "--threads", "2", "--seed", "11").CombinedOutput(); err != nil {
		return nil, fmt.Errorf("fusion_build: %v\n%s", err, out)
	} else if !strings.Contains(string(out), "reusing codebook") {
		return nil, fmt.Errorf("fusion_build trained another codebook:\n%s", out)
	}
	heads, err := partitioning.ReadU32Array(filepath.Join(g.index, "heads.bin"))
	if err != nil {
		return nil, err
	}
	assign := make([]uint32, len(heads))
	for c := range assign {
		assign[c] = uint32(c * g.partitions / len(heads))
	}
	assignPath := filepath.Join(f.dir, "assign-g2.bin")
	if err := partitioning.WriteU32Array(assignPath, assign); err != nil {
		return nil, err
	}
	if out, err := exec.Command(filepath.Join(f.bin, "rtier_segment"), "--index", g.index, "--assign", assignPath,
		"--out", g.parts).CombinedOutput(); err != nil {
		return nil, fmt.Errorf("rtier_segment: %v\n%s", err, out)
	}
	addr, stop, err := startNodeWith(f, g.index, g.parts, "--load", "all", "--graph", "index")
	if err != nil {
		return nil, err
	}
	nc := nodeclient.New(addr, 1)
	defer nc.Close()
	for i := 0; i < numQueries; i++ {
		res, err := nc.SearchLocal(context.Background(), 0, f.queries.Row(i),
			nodeclient.SearchParams{K: testK, NProbe: testNProbe, Rerank: maxRerank})
		if err != nil {
			stop()
			return nil, err
		}
		g.oracle = append(g.oracle, res)
	}
	return g, stop()
}

// TestFullRebuild moves a cluster under query load to the second generation of its index (U14):
// two spares take over every partition lazily, the two old nodes switch to the new generation
// while they serve raw vectors only, get the new partitions where most of their raw vectors
// are, warm up with the queries the entries kept, and take over in one flip; the spares leave.
// Every answer is exact for its epoch's generation: the first generation's oracle before the
// flip, the second's from it on. Then the cluster lives on in the new generation: a minor batch
// (the update controller's router follows to the new generation), and a scale-out to a node that
// registered with the first generation's configuration.
func TestFullRebuild(t *testing.T) {
	f := getFixture(t)
	g := getGeneration2(t, f)
	m, err := partitioning.LoadManifest(f.parts)
	if err != nil {
		t.Fatal(err)
	}
	extraIDs := 1024
	room := []string{"--max-vectors", strconv.Itoa(numVectors + extraIDs),
		"--max-pages", strconv.FormatUint(m.NumPages+uint64(extraIDs)/m.VectorsPerPage+2, 10), "--max-list-len", "4096"}
	// The rebuild's segments with holders come from the controller's bulk port, as on a cluster.
	cl := newCluster(t, f, "lazy", 2, func(c *config.Controller) {
		c.AllowUpdates = true
		c.WorkDir, c.BulkListen = t.TempDir(), "127.0.0.1:0"
	})
	n1 := cl.addNode("n1", query.TwoPhase{}, room...)
	n2 := cl.addNode("n2", query.TwoPhase{}, room...)
	if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
	}
	s1 := cl.addNode("s1", query.TwoPhase{}, room...)
	s2 := cl.addNode("s2", query.TwoPhase{}, room...)
	ctx := cl.ctx

	stop := startLoad(cl, 4)
	time.Sleep(300 * time.Millisecond) // the entries keep the queries the warm-up replays
	var flipAt uint64
	cl.ctl.BeforeGenerationFlip = func(e uint64) {
		flipAt = e
		cl.expect(e, g.oracle)
	}
	reply, err := cl.ctl.Rebuild(ctx, protocol.RebuildReq{
		Source: protocol.GenerationSource{Kind: "dir", IndexDir: g.index, PartsDir: g.parts}})
	if err != nil {
		t.Fatalf("rebuild: %v", err)
	}
	t.Logf("rebuild: epoch %d -> %d, spares %v, %d of %d vectors local, warm-up %d queries fetched %d vectors, "+
		"%d fetched in all, phases %v", reply.FromEpoch, reply.ToEpoch, reply.Spares, reply.Local, reply.Vectors,
		reply.WarmupQueries, reply.WarmupFetched, reply.RawFetched, reply.Phases)
	time.Sleep(300 * time.Millisecond)
	st := stop()
	t.Logf("%d queries, %d retries (unavailable: %v), max latency %v, per epoch %v",
		st.total, st.retries, st.retryWhy, st.maxLatency, st.epochs)
	if st.mismatches > 0 || st.errors > 0 {
		t.Fatalf("%d mismatches, %d errors; first: %s", st.mismatches, st.errors, st.firstProblem)
	}
	var before, after int64
	for e, k := range st.epochs {
		if e < flipAt {
			before += k
		} else {
			after += k
		}
	}
	if before == 0 || after == 0 {
		t.Fatalf("load did not span the flip to the new generation at epoch %d: %v", flipAt, st.epochs)
	}

	// The new generation runs on the old nodes, its partitions spread evenly; the spares are idle,
	// at the new generation, holding no list and no graph.
	tbl := cl.ctl.Status().Table
	if tbl.Generation != 1 || tbl.Epoch != flipAt || reply.ToEpoch != flipAt || tbl.Placement.NumPartitions() != g.partitions {
		t.Fatalf("epoch %d: generation %d with %d partitions (reply %+v)", tbl.Epoch, tbl.Generation,
			tbl.Placement.NumPartitions(), reply)
	}
	for _, a := range []interface{ ID() placement.NodeID }{n1, n2} {
		if !tbl.IsEntry(a.ID()) || len(tbl.Placement.PartitionsOf(a.ID())) != g.partitions/2 {
			t.Errorf("node %d: entry %v, partitions %v", a.ID(), tbl.IsEntry(a.ID()), tbl.Placement.PartitionsOf(a.ID()))
		}
	}
	for _, a := range []interface {
		ID() placement.NodeID
	}{s1, s2} {
		if _, member := tbl.Nodes[a.ID()]; member {
			t.Errorf("spare %d is still a member of epoch %d", a.ID(), tbl.Epoch)
		}
	}
	for _, s := range []string{s1.Info().NodeAddr, s2.Info().NodeAddr} {
		nc := nodeclient.New(s, 1)
		in, err := nc.Info(ctx)
		nc.Close()
		if err != nil || in.Generation != 1 || len(in.Resident) != 0 || in.GraphLoaded {
			t.Errorf("spare %s: generation %d, partitions %v, graph %v (%v)", s, in.Generation, in.Resident, in.GraphLoaded, err)
		}
	}
	if reply.Local <= 0 || reply.Local > reply.Vectors || reply.WarmupQueries == 0 {
		t.Errorf("reply: %d of %d vectors local, %d warm-up queries", reply.Local, reply.Vectors, reply.WarmupQueries)
	}
	exactPass(t, cl, nil)

	// A minor batch in the new generation: the update controller's router, started on the first
	// generation, follows, and so does the oracle -- one node built on the second generation.
	oaddr, stopOracle, err := startNodeWith(f, g.index, g.parts, append([]string{"--load", "all", "--graph", "index"}, room...)...)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := stopOracle(); err != nil {
			t.Error(err)
		}
	}()
	oracle := nodeclient.New(oaddr, 1)
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
	stop = startLoad(cl, 4)
	var vecs []byte
	for i := 0; i < 20; i++ {
		vecs = append(vecs, f.queries.Row(i)...)
	}
	b, err := u.Prepare(ctx, vecs, len(f.queries.Row(0)), []uint32{3, 33})
	if err != nil {
		t.Fatal(err)
	}
	if ri, err := router.Info(ctx); err != nil || ri.Generation != 1 {
		t.Fatalf("the router runs generation %d (%v)", ri.Generation, err)
	}
	applyOracle(t, ctx, oracle, b, make([]uint32, g.partitions))
	cl.expect(b.Lease.Table.Epoch+1, oracleAnswers(t, ctx, f, oracle))
	if _, err := u.Execute(ctx, b); err != nil {
		t.Fatalf("minor batch after the rebuild: %v", err)
	}

	// A scale-out to a node that registered with the first generation's configuration: it moves
	// to the new generation before it stages anything.
	n5 := cl.addNode("n5", query.TwoPhase{}, room...)
	cur := cl.ctl.Status().Table
	owners := map[string][]int{}
	for id, ps := range cur.Placement.ByNode() {
		owners[strconv.Itoa(int(id))] = ps[1:]
		owners[strconv.Itoa(int(n5.ID()))] = append(owners[strconv.Itoa(int(n5.ID()))], ps[0])
	}
	out, err := cl.ctl.Reconfigure(ctx, protocol.RescaleReq{Plan: &protocol.Plan{Owners: owners}})
	if err != nil {
		t.Fatalf("scale-out after the rebuild: %v", err)
	}
	t.Logf("scale-out: epoch %d -> %d, moved %v, entries %v", out.FromEpoch, out.ToEpoch, out.Moved, cl.ctl.Status().Table.Entries)
	time.Sleep(200 * time.Millisecond)
	st = stop()
	if st.mismatches > 0 || st.errors > 0 {
		t.Fatalf("after the rebuild: %d mismatches, %d errors; first: %s", st.mismatches, st.errors, st.firstProblem)
	}
	if !cl.ctl.Status().Table.IsEntry(n5.ID()) {
		t.Errorf("node %d did not become an entry", n5.ID())
	}
	exactPass(t, cl, nil)
}

// TestFullRebuildStopTheWorld is the baseline without spares: admission is paused on every entry
// while the nodes switch to the second generation where they are. Queries wait (they are told
// to retry) and every answer is still exact.
func TestFullRebuildStopTheWorld(t *testing.T) {
	f := getFixture(t)
	g := getGeneration2(t, f)
	cl := newCluster(t, f, "lazy", 2)
	n1 := cl.addNode("n1", query.TwoPhase{}, "--max-list-len", "4096")
	n2 := cl.addNode("n2", query.TwoPhase{}, "--max-list-len", "4096")
	if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
	}
	stop := startLoad(cl, 4)
	time.Sleep(200 * time.Millisecond)
	cl.ctl.BeforeGenerationFlip = func(e uint64) { cl.expect(e, g.oracle) }
	reply, err := cl.ctl.Rebuild(cl.ctx, protocol.RebuildReq{StopTheWorld: true, WorkDir: t.TempDir(),
		Source: protocol.GenerationSource{Kind: "dir", IndexDir: g.index, PartsDir: g.parts}})
	if err != nil {
		t.Fatalf("rebuild: %v", err)
	}
	t.Logf("rebuild: epoch %d -> %d, %d of %d vectors local, phases %v", reply.FromEpoch, reply.ToEpoch, reply.Local,
		reply.Vectors, reply.Phases)
	time.Sleep(200 * time.Millisecond)
	st := stop()
	t.Logf("%d queries, %d retries (unavailable: %v), max latency %v, per epoch %v",
		st.total, st.retries, st.retryWhy, st.maxLatency, st.epochs)
	if st.mismatches > 0 || st.errors > 0 {
		t.Fatalf("%d mismatches, %d errors; first: %s", st.mismatches, st.errors, st.firstProblem)
	}
	tbl := cl.ctl.Status().Table
	if tbl.Generation != 1 || len(reply.Spares) != 0 || !tbl.IsEntry(n1.ID()) || !tbl.IsEntry(n2.ID()) ||
		st.epochs[tbl.Epoch] == 0 {
		t.Fatalf("epoch %d: generation %d, entries %v, spares %v, queries per epoch %v", tbl.Epoch, tbl.Generation,
			tbl.Entries, reply.Spares, st.epochs)
	}
	exactPass(t, cl, nil)
}
