package e2e

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"rtier/internal/config"
	"rtier/internal/epoch"
	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
	"rtier/internal/placement"
	"rtier/internal/query"
	"rtier/internal/update"
)

// TestSPFreshReassign applies a major epoch under SPFresh's policy (U14, internal/update/
// spfresh.go) to a cluster under query load, and the same decisions to a single-node oracle. Two
// lists that queries probe first split; each checks the members of its children and of the 64
// lists nearest its old head by SPFresh's conditions, a vector once (one in a child only there).
// A list at the owners' boundary merges: with its nearest list that has room, the shorter going
// into the other, whose head is then checked against the postings it took. A vector that no longer
// belongs to the list it was queued in is inserted again by the build's rule: into every list of
// its new set, on either owner (its PQ code sent along, its raw vector fetched on demand), and out
// of every other list that holds it -- postings dropped from the base version wherever the owners
// find them, or taken out of the plan. Every answer must equal the oracle's for its epoch, the
// lists hold what the plan says, every moved vector is in exactly its new lists, a merge's
// postings are in the list it went into or moved on, and no vector of one split list ends in both
// of its children. It runs
// on FusionANNS's index (the 8 nearest heads) and on the RNG-pruned one SPFresh's evaluation
// builds, where the build's rule -- inserts' and the reassign's -- is SPANN's RNG selection.
func TestSPFreshReassign(t *testing.T) {
	for _, c := range []struct {
		name string
		get  func(*testing.T) *fixture
	}{{"fusionanns", getFixture}, {"rng", getRNGFixture}} {
		t.Run(c.name, func(t *testing.T) { testSPFreshReassign(t, c.get(t)) })
	}
}

func testSPFreshReassign(t *testing.T, f *fixture) {
	m, err := partitioning.LoadManifest(f.parts)
	if err != nil {
		t.Fatal(err)
	}
	room := []string{"--max-vectors", strconv.Itoa(numVectors + 1024),
		"--max-pages", strconv.FormatUint(m.NumPages+1024/m.VectorsPerPage+2, 10),
		"--max-list-len", strconv.Itoa(m.MaxListLen + 256)}
	cl := newCluster(t, f, "lazy", 2, func(c *config.Controller) { c.AllowUpdates = true })
	cl.addNode("n1", query.TwoPhase{}, room...)
	cl.addNode("n2", query.TwoPhase{}, room...)
	if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
	}
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
	raddr, stopRouter, err := update.StartRouter(filepath.Join(f.bin, "rtier_node"), f.indexNoBootstrap, f.parts, logf)
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
		GraphPath: filepath.Join(f.indexNoBootstrap, "graph.hnsw"), PatchDir: t.TempDir(),
		Policy: update.PolicySPFresh, ReassignRange: 64, PostingLimit: m.MaxListLen + 200}
	defer u.Close()
	oracleVersions := make([]uint32, numPartitions)

	stop := startLoad(cl, 4)
	time.Sleep(200 * time.Millisecond)

	// The lists queries probe first, and those at the owners' boundary: the first list of a query
	// whose second list the other node owns.
	tbl := cl.ctl.Status().Table
	owner := func(l uint32) placement.NodeID { return tbl.Owner(tbl.PartitionOf(l, m)) }
	nav := nodeclient.New(tbl.Nodes[tbl.Entries[0]].NodeAddr, 1)
	var firsts, boundary []uint32
	for qi := 0; qi < numQueries; qi++ {
		lists, err := nav.Navigate(ctx, tbl.Epoch, tbl.Graph, f.queries.Row(qi), testNProbe, 0)
		if err != nil {
			t.Fatal(err)
		}
		if !containsU32(firsts, lists[0]) {
			firsts = append(firsts, lists[0])
		}
		if l := lists[0]; owner(l) != owner(lists[1]) && !containsU32(boundary, l) {
			boundary = append(boundary, l)
		}
	}
	nav.Close()

	// Decided under the lease, given up and decided again with other lists until the merge finds a
	// partner, the reassign drops a posting of the base version and a posting goes to another
	// owner.
	dropped := func(mj *update.Major) int {
		n := 0
		for _, pe := range mj.Parts {
			n += len(pe.DropLists)
		}
		return n
	}
	var mj *update.Major
	for i := 0; i+1 < len(firsts); i += 2 {
		splits := firsts[i : i+2]
		var merges []uint32
		for _, l := range boundary {
			if !containsU32(splits, l) {
				merges = []uint32{l}
				break
			}
		}
		if mj, err = u.PrepareMajor(ctx, splits, merges); err != nil {
			t.Fatal(err)
		}
		if dropped(mj) > 0 && len(mj.Codes) > 0 && len(mj.Merges) == 1 {
			break
		}
		if err := u.AbortMajor(ctx, mj); err != nil {
			t.Fatal(err)
		}
		mj = nil
	}
	if mj == nil {
		t.Fatalf("no split of the lists %v, with a merge, drops a posting and sends one to another owner", firsts)
	}
	rs := mj.Reassign
	if rs == nil || rs.Lists == 0 || rs.Checked == 0 || rs.Candidates == 0 || rs.Moved == 0 || rs.Added == 0 || rs.Dropped == 0 {
		t.Fatalf("reassign %+v: want lists read, postings checked, and vectors moved", rs)
	}
	// A list's postings in an epoch, from its owner.
	listIn := func(tb *epoch.Table, l uint32) []uint32 {
		p := tb.PartitionOf(l, m)
		if q, ok := tb.Lists[l]; ok {
			p = q
		}
		nc := nodeclient.New(tb.Nodes[tb.Owner(p)].NodeAddr, 1)
		defer nc.Close()
		lp, err := nc.ListGet(ctx, l, tb.Version(p))
		if err != nil {
			t.Fatalf("list %d of partition %d: %v", l, p, err)
		}
		return lp.IDs
	}
	// Each split list's members, and how many of the split and merged lists hold each vector.
	members := map[uint32][]uint32{}
	inLists := map[uint32]int{}
	for _, s := range mj.Splits {
		members[s.List] = listIn(mj.Lease.Table, s.List)
		for _, id := range members[s.List] {
			inLists[id]++
		}
	}
	for _, mg := range mj.Merges {
		members[mg.List] = listIn(mj.Lease.Table, mg.List)
		for _, id := range members[mg.List] {
			inLists[id]++
		}
	}
	applyMajorOracle(t, ctx, oracle, mj, oracleVersions)
	from := mj.Lease.Table.Epoch
	cl.expect(from+1, oracleAnswers(t, ctx, f, oracle))
	reply, err := u.ExecuteMajor(ctx, mj)
	if err != nil {
		t.Fatalf("major epoch: %v", err)
	}
	moved := 0
	for _, cs := range mj.Codes {
		moved += len(cs.IDs)
	}
	t.Logf("SPFresh's reassign %+v; splits %+v, merges %+v, %d PQ codes to other owners, %d postings dropped; "+
		"epoch %d -> %d, partitions %v", *rs, mj.Splits, mj.Merges, moved, dropped(mj), reply.FromEpoch, reply.ToEpoch,
		reply.Changed)
	if reply.FromEpoch != from || reply.ToEpoch != from+1 || reply.Graph != 1 {
		t.Fatalf("major epoch reply %+v: want a flip from epoch %d to graph version 1", reply, from)
	}
	checkUpdateFlip(t, "major epoch", reply)

	// The lists hold what the plan says: no dropped posting, every added one, and each child its
	// planned size.
	after := cl.ctl.Status().Table
	list := func(l uint32) []uint32 { return listIn(after, l) }
	for _, pe := range mj.Parts {
		for i, l := range pe.DropLists {
			if containsU32(list(l), pe.DropIDs[i]) {
				t.Errorf("list %d still holds vector %d, which the reassign moved out", l, pe.DropIDs[i])
			}
		}
		for i, l := range pe.Lists {
			if !containsU32(list(l), pe.IDs[i]) {
				t.Errorf("list %d lacks vector %d, which the epoch added", l, pe.IDs[i])
			}
		}
	}
	for _, s := range mj.Splits {
		for c := 0; c < 2; c++ {
			if n := len(list(s.Children[c])); n != s.Sizes[c] {
				t.Errorf("list %d (split of %d) holds %d postings, the plan %d", s.Children[c], s.List, n, s.Sizes[c])
			}
		}
	}
	// Every vector the reassign moved is in exactly the lists of its new set, the copies it had
	// elsewhere gone; the merged list is gone and its postings are in the list it went into, or
	// moved on.
	all := allLists(t, ctx, after, m)
	holds := map[uint32][]uint32{}
	for l, ids := range all {
		for _, id := range ids {
			holds[id] = append(holds[id], l)
		}
	}
	sorted := func(x []uint32) []uint32 {
		y := append([]uint32(nil), x...)
		sort.Slice(y, func(i, j int) bool { return y[i] < y[j] })
		return y
	}
	if len(mj.Reassigned) != rs.Moved {
		t.Errorf("%d vectors reassigned, the stats say %d", len(mj.Reassigned), rs.Moved)
	}
	for id, set := range mj.Reassigned {
		if got, want := sorted(holds[id]), sorted(set); !reflect.DeepEqual(got, want) {
			t.Errorf("vector %d is in lists %v, its new set is %v", id, got, want)
		}
	}
	for _, mg := range mj.Merges {
		if _, ok := all[mg.List]; ok {
			t.Errorf("merged list %d is still there", mg.List)
		}
		if _, ok := all[mg.Into]; !ok {
			t.Errorf("list %d, which list %d merged into, is gone", mg.Into, mg.List)
		}
		for _, id := range members[mg.List] {
			if _, moved := mj.Reassigned[id]; !moved && !containsU32(all[mg.Into], id) {
				t.Errorf("vector %d of merged list %d is not in list %d", id, mg.List, mg.Into)
			}
		}
	}
	// A vector in a child is checked in that child only: one that no other split or merged list
	// held ends in one of its split's children at most.
	for _, s := range mj.Splits {
		c0, c1 := list(s.Children[0]), list(s.Children[1])
		for _, id := range members[s.List] {
			if inLists[id] == 1 && containsU32(c0, id) && containsU32(c1, id) {
				t.Errorf("vector %d of list %d is in both its children, %d and %d", id, s.List, s.Children[0], s.Children[1])
			}
		}
	}

	time.Sleep(300 * time.Millisecond)
	st := stop()
	t.Logf("%d queries, %d retries, max latency %v, per epoch %v", st.total, st.retries, st.maxLatency, st.epochs)
	if st.mismatches > 0 || st.errors > 0 {
		t.Fatalf("%d mismatches, %d errors; first: %s", st.mismatches, st.errors, st.firstProblem)
	}
	if len(st.epochs) < 2 {
		t.Fatalf("load did not span the flip: %v", st.epochs)
	}
	exactPass(t, cl, nil)
}
