package e2e

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
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
	"rtier/internal/protocol"
	"rtier/internal/query"
	"rtier/internal/update"
	"rtier/internal/vecio"
)

// TestUpdateBatch applies update batches (U14) to a cluster under query load, and the same
// decisions to a single-node oracle. The update controller routes inserts with its own router,
// an rtier_node on the index without its page file that serves no query. Every answer must
// equal the oracle's for the data of its
// epoch -- queries of the epoch before a batch's flip read the old lists, those of the epoch after
// it the new ones -- across the batches and across reconfigurations that move partitions the
// batches changed:
//
//  1. A batch inserts copies of half the queries and fresh vectors, and deletes the nearest
//     neighbor of each query of the other half, and random vectors. From the flip on, each copy
//     is its query's nearest neighbor, and no deleted vector is in any answer. PQ codes left
//     staged on a data node before the batch (by a staging that never ended) are released when
//     the batch begins.
//  2. A batch that cannot be staged (more postings for one list than a request may gather)
//     changes nothing: no version, no PQ code left behind, the lease released.
//  3. A scale-out moves partitions the first batch changed; the new node loads them at that
//     version, from segment files their old owners write for the move.
//  4. A second batch changes partitions on their new owners, and deletes vectors the first one
//     inserted.
//  5. A scale-in returns partitions to their former owners, which lack the vectors inserted
//     meanwhile: they take them over from the node that leaves before the rescale returns (with
//     lazy, the drain; lazy-stream streams them in), and once that node is stopped every answer
//     is still exact.
//
// After each step every node holds its partitions at exactly the epoch's versions (and segment
// files only of versions it holds), live PQ codes for exactly the vectors its partitions name,
// no staged code, and no code of a deleted vector, not even one marked for later.
func TestUpdateBatch(t *testing.T) {
	f := getFixture(t)
	for _, protocol := range []string{"lazy", "lazy-stream"} {
		t.Run(protocol, func(t *testing.T) { testUpdateBatch(t, f, protocol) })
	}
}

func testUpdateBatch(t *testing.T, f *fixture, proto string) {
	m, err := partitioning.LoadManifest(f.parts)
	if err != nil {
		t.Fatal(err)
	}
	maxLen := m.MaxListLen + 256
	extraIDs := 2*maxLen + 1024
	extraPages := uint64(extraIDs)/m.VectorsPerPage + 2
	room := []string{"--max-vectors", strconv.Itoa(numVectors + extraIDs),
		"--max-pages", strconv.FormatUint(m.NumPages+extraPages, 10), "--max-list-len", strconv.Itoa(maxLen)}
	cl := newCluster(t, f, proto, 2, func(c *config.Controller) { c.AllowUpdates = true })
	cl.addNode("n1", query.TwoPhase{}, room...)
	cl.addNode("n2", query.TwoPhase{}, room...)
	if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
	}
	n3 := cl.addNode("n3", query.TwoPhase{}, room...) // registers idle; joins at scale-out
	ctx := cl.ctx

	// The oracle holds every partition and applies every batch's decisions.
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
	// The update controller's router: the graph and codebook only, no page file, no partition.
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
	u := &update.Updater{Controller: cl.ctl, Manifest: m, Alloc: alloc, Router: router}
	defer u.Close()
	oracleVersions := make([]uint32, numPartitions)
	model := newContent(f)
	var deleted []uint32
	vb := len(f.queries.Row(0))
	base, err := vecio.ReadBin(filepath.Join(f.dir, "base.fbin"), 0)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(3))

	// apply runs one batch: decided under the lease, applied to the oracle, whose answers hold
	// from the flip on, then executed on the cluster.
	apply := func(what string, vecs []byte, del []uint32) (*update.Batch, [][]nodeclient.Candidate) {
		t.Helper()
		b, err := u.Prepare(ctx, vecs, vb, del)
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		applyOracle(t, ctx, oracle, b, oracleVersions)
		after := oracleAnswers(t, ctx, f, oracle)
		from := b.Lease.Table.Epoch
		cl.expect(from+1, after)
		reply, err := u.Execute(ctx, b)
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		if reply.FromEpoch != from || reply.ToEpoch != from+1 || len(reply.Changed) == 0 {
			t.Fatalf("%s: reply %+v, want a flip from epoch %d that changes partitions", what, reply, from)
		}
		lists, _, _ := b.Postings()
		t.Logf("%s: %d inserts (%d postings), %d deletes; epoch %d -> %d, partitions %v at new versions, phases %v",
			what, len(b.Inserts), len(lists), len(b.Deleted), reply.FromEpoch, reply.ToEpoch, reply.Changed, reply.Phases)
		model.apply(b, m)
		deleted = append(deleted, b.Deleted...)
		for _, a := range after {
			for _, c := range a {
				for _, d := range b.Deleted {
					if c.ID == d {
						t.Fatalf("%s: deleted vector %d is in an answer", what, d)
					}
				}
			}
		}
		return b, after
	}

	// Codes staged for a staging that never ended (an update controller that died after PQ_PUT):
	// IDs at the end of the space, which no batch here reaches.
	tbl := cl.ctl.Status().Table
	n1 := nodeclient.New(tbl.Nodes[tbl.Placement.Nodes()[0]].NodeAddr, 1)
	leftover := []uint32{uint32(in.IDCapacity) - 3, uint32(in.IDCapacity) - 2, uint32(in.IDCapacity) - 1}
	if k, _, err := n1.PQPut(ctx, leftover, in.PQ.M, make([]byte, len(leftover)*in.PQ.M)); err != nil || k != len(leftover) {
		t.Fatalf("leftover staged codes: %d installed, %v", k, err)
	}
	n1.Close()

	stop := startLoad(cl, 4)
	time.Sleep(200 * time.Millisecond)

	// 1. Copies of the first half of the queries, and fresh vectors near the data; the nearest
	// neighbors of the second half, and random vectors, go.
	half := numQueries / 2
	var vecs []byte
	for i := 0; i < half; i++ {
		vecs = append(vecs, f.queries.Row(i)...)
	}
	for i := 0; i < 100; i++ {
		vecs = append(vecs, shifted(base.Row(rng.Intn(numVectors)), 0.05)...)
	}
	var del []uint32
	for i := half; i < numQueries; i++ {
		del = append(del, f.oracle[i][0].ID)
	}
	for i := 0; i < 50; i++ {
		del = append(del, uint32(rng.Intn(numVectors)))
	}
	b1, after1 := apply("batch 1", vecs, del)
	for i := 0; i < half; i++ {
		if after1[i][0].ID != b1.Inserts[i].ID || after1[i][0].Dist != 0 {
			t.Fatalf("batch 1: query %d's nearest neighbor is %+v, want its copy %d", i, after1[i][0], b1.Inserts[i].ID)
		}
	}
	time.Sleep(200 * time.Millisecond)
	checkContent(t, cl, model, append(append([]uint32(nil), deleted...), leftover...))

	// 2. A batch no owner can stage: every copy of one vector goes to the same lists, more of them
	// than a list may hold. Nothing may change, and nothing of it may stay on the nodes.
	before := cl.ctl.Status().Table
	resident := pqResident(t, cl)
	var copies []byte
	for i := 0; i <= maxLen; i++ {
		copies = append(copies, f.queries.Row(half)...)
	}
	bad, err := u.Prepare(ctx, copies, vb, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := u.Execute(ctx, bad); err == nil || !strings.Contains(err.Error(), "max-list-len") {
		t.Fatalf("a batch with lists too long to stage: got %v, want the owner to refuse it", err)
	}
	if st := cl.ctl.Status(); st.Table.Epoch != before.Epoch || st.ReconfigInProgress || st.UpdateInProgress {
		t.Fatalf("after the failed batch: epoch %d (was %d), reconfiguration in progress %v, update %v",
			st.Table.Epoch, before.Epoch, st.ReconfigInProgress, st.UpdateInProgress)
	}
	var badIDs []uint32
	for _, x := range bad.Inserts {
		badIDs = append(badIDs, x.ID)
	}
	checkContent(t, cl, model, append(append([]uint32(nil), deleted...), badIDs...))
	if got := pqResident(t, cl); fmt.Sprint(got) != fmt.Sprint(resident) {
		t.Errorf("PQ codes held after the failed batch %v, before it %v", got, resident)
	}

	// 3. A scale-out moves partitions at version 1.
	cur := cl.ctl.Status().Table
	out, err := cl.ctl.Reconfigure(ctx, protocol.RescaleReq{DataNodes: 3})
	if err != nil {
		t.Fatalf("scale-out: %v", err)
	}
	changed := 0
	for _, p := range out.Moved {
		if cur.Version(p) > 0 {
			changed++
		}
	}
	if changed == 0 || out.Phases["save_segments"] == 0 {
		t.Fatalf("scale-out moved %v at versions %v (phases %v): want partitions the batch changed, saved for the move",
			out.Moved, cur.Versions, out.Phases)
	}
	t.Logf("scale-out: epoch %d -> %d, moved %v, phases %v", out.FromEpoch, out.ToEpoch, out.Moved, out.Phases)
	time.Sleep(200 * time.Millisecond)
	checkContent(t, cl, model, deleted)

	// 4. A second batch, on the new placement: copies of the second half of the queries, fresh
	// vectors; the copies of the first queries and some of the first batch's fresh vectors go.
	vecs = nil
	for i := half; i < numQueries; i++ {
		vecs = append(vecs, f.queries.Row(i)...)
	}
	for i := 0; i < 50; i++ {
		vecs = append(vecs, shifted(base.Row(rng.Intn(numVectors)), -0.05)...)
	}
	del = nil
	for i := 0; i < 8; i++ {
		del = append(del, b1.Inserts[i].ID)
	}
	for i := 0; i < 20; i++ {
		del = append(del, b1.Inserts[half+i].ID)
	}
	b2, after2 := apply("batch 2", vecs, del)
	for i := half; i < numQueries; i++ {
		if after2[i][0].ID != b2.Inserts[i-half].ID {
			t.Fatalf("batch 2: query %d's nearest neighbor is %+v, want its copy %d", i, after2[i][0], b2.Inserts[i-half].ID)
		}
	}
	time.Sleep(200 * time.Millisecond)
	checkContent(t, cl, model, deleted)

	// 5. A scale-in returns partitions to their former owners.
	out, err = cl.ctl.Reconfigure(ctx, protocol.RescaleReq{DataNodes: 2})
	if err != nil {
		t.Fatalf("scale-in: %v", err)
	}
	t.Logf("scale-in: epoch %d -> %d, moved %v, phases %v, %d raw vectors drained from the leaving node",
		out.FromEpoch, out.ToEpoch, out.Moved, out.Phases, out.DrainVectors)
	if proto == "lazy" && out.DrainVectors == 0 {
		t.Errorf("lazy scale-in after batch 2 drained nothing: the vectors batch 2 inserted into the returning " +
			"partitions were only on the leaving node")
	}
	time.Sleep(300 * time.Millisecond)
	st := stop()
	checkContent(t, cl, model, deleted)
	t.Logf("%d queries, %d retries (unavailable: %v), max latency %v, per epoch %v",
		st.total, st.retries, st.retryWhy, st.maxLatency, st.epochs)
	if st.mismatches > 0 || st.errors > 0 {
		t.Fatalf("%d mismatches, %d errors; first: %s", st.mismatches, st.errors, st.firstProblem)
	}
	if len(st.epochs) < 5 {
		t.Fatalf("load did not span the batches and reconfigurations: %v", st.epochs)
	}
	// The node that left can go offline: nothing it held is needed from it any more.
	cl.stopDataNode(n3)
	exactPass(t, cl, nil)
}

// content tracks the vector IDs each partition's lists name as batches change them.
type content struct{ parts []map[uint32]bool }

func newContent(f *fixture) *content {
	c := &content{parts: make([]map[uint32]bool, numPartitions)}
	for p, ids := range f.partIDs {
		c.parts[p] = map[uint32]bool{}
		for _, id := range ids {
			c.parts[p][id] = true
		}
	}
	return c
}

func (c *content) apply(b *update.Batch, m *partitioning.Manifest) {
	for _, ids := range c.parts {
		for _, id := range b.Deleted {
			delete(ids, id)
		}
	}
	for _, in := range b.Inserts {
		for _, l := range in.Lists {
			c.parts[m.PartitionOf(l)][in.ID] = true
		}
	}
}

// count is the number of vectors the lists of partitions ps name: a node's live PQ codes.
func (c *content) count(ps []int) uint64 {
	s := map[uint32]bool{}
	for _, p := range ps {
		for id := range c.parts[p] {
			s[id] = true
		}
	}
	return uint64(len(s))
}

// applyOracle applies b's decisions to the oracle, which holds every partition at versions[p]:
// the inserts' codes and vectors, the next version of every partition the batch changes, then
// the reclaim in the controller's order -- the deleted vectors marked, the replaced versions
// dropped.
func applyOracle(t *testing.T, ctx context.Context, nc *nodeclient.Client, b *update.Batch, versions []uint32) {
	t.Helper()
	if len(b.Inserts) > 0 {
		ids, locs := make([]uint32, len(b.Inserts)), make([]uint32, len(b.Inserts))
		var codes []byte
		for i, x := range b.Inserts {
			ids[i], locs[i] = x.ID, x.Loc
			codes = append(codes, x.Code...)
		}
		if _, _, err := nc.PQPut(ctx, ids, len(b.Inserts[0].Code), codes); err != nil {
			t.Fatal(err)
		}
		if _, _, err := nc.RawAdd(ctx, locs, b.VecBytes, b.Vecs); err != nil {
			t.Fatal(err)
		}
	}
	vs := make([]nodeclient.VersionStage, len(versions))
	for p, v := range versions {
		vs[p] = nodeclient.VersionStage{Partition: p, Base: v, Next: v + 1}
	}
	lists, ids, locs := b.Postings()
	res, err := nc.StageVersions(ctx, vs, nodeclient.Edits{Lists: lists, IDs: ids, Locs: locs, Deleted: b.Deleted})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nc.ForgetVectors(ctx, b.Deleted, true); err != nil {
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

// oracleAnswers is the oracle's answer to every query of the fixture, from the newest version of
// every partition.
func oracleAnswers(t *testing.T, ctx context.Context, f *fixture, nc *nodeclient.Client) [][]nodeclient.Candidate {
	t.Helper()
	out := make([][]nodeclient.Candidate, numQueries)
	for i := range out {
		res, err := nc.SearchLocal(ctx, 0, f.queries.Row(i), nodeclient.SearchParams{K: testK, NProbe: testNProbe, Rerank: maxRerank})
		if err != nil {
			t.Fatal(err)
		}
		out[i] = res
	}
	return out
}

// checkContent: every node holds its partitions at exactly the epoch's versions, and segment
// files only of versions it holds; its live PQ codes are exactly the vectors its partitions name
// (model), none is staged, and none of the gone vectors has a code anywhere.
func checkContent(t *testing.T, cl *cluster, model *content, gone []uint32) {
	t.Helper()
	st := cl.ctl.Status()
	tbl := st.Table
	for _, ns := range st.Nodes {
		nc := nodeclient.New(ns.Info.NodeAddr, 1)
		info, err := nc.InfoSettled(cl.ctx)
		if err != nil {
			t.Fatal(err)
		}
		parts := tbl.Placement.PartitionsOf(ns.ID)
		sort.Ints(parts)
		held := map[string]bool{}
		var want [][2]uint32
		for _, p := range parts {
			want = append(want, [2]uint32{uint32(p), tbl.Version(p)})
		}
		if fmt.Sprint(info.Versions) != fmt.Sprint(want) {
			t.Errorf("node %d (%s) holds versions %v, epoch %d says %v", ns.ID, ns.Name, info.Versions, tbl.Epoch, want)
		}
		for _, pv := range info.Versions {
			held[partitioning.SegmentNameV(int(pv[0]), pv[1])] = true
		}
		if n := model.count(parts); info.PQ.Live != n || info.PQ.Staged != 0 || info.PQ.Deleted != 0 {
			t.Errorf("node %d (%s) PQ store %+v: want %d live codes (the vectors its partitions name), none staged, "+
				"none of a deleted vector", ns.ID, ns.Name, info.PQ, n)
		}
		for _, id := range gone {
			if _, _, err := nc.PQGet(cl.ctx, []uint32{id}); !nodeclient.IsPQAbsent(err) {
				t.Errorf("node %d (%s) still has the PQ code of vector %d, deleted or never staged (%v)", ns.ID, ns.Name, id, err)
				break
			}
		}
		nc.Close()
		files, err := os.ReadDir(filepath.Join(cl.dirs[ns.ID], "partitions"))
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range files {
			if !held[e.Name()] {
				t.Errorf("node %d (%s) keeps %s, a version it does not hold (it holds %v)", ns.ID, ns.Name, e.Name(), info.Versions)
			}
		}
	}
}

// pqResident is every registered node's PQ code count.
func pqResident(t *testing.T, cl *cluster) map[string]uint64 {
	t.Helper()
	out := map[string]uint64{}
	for _, ns := range cl.ctl.Status().Nodes {
		info, err := nodeclient.New(ns.Info.NodeAddr, 1).Info(cl.ctx)
		if err != nil {
			t.Fatal(err)
		}
		out[ns.Name] = info.PQ.Resident
	}
	return out
}

// shifted is a float vector moved by d in every coordinate: a fresh vector near the data.
func shifted(row []byte, d float32) []byte {
	out := make([]byte, len(row))
	for j := 0; j < len(row); j += 4 {
		v := math.Float32frombits(binary.LittleEndian.Uint32(row[j:])) + d
		binary.LittleEndian.PutUint32(out[j:], math.Float32bits(v))
	}
	return out
}
