package e2e

import (
	"bytes"
	"encoding/json"
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
	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
	"rtier/internal/query"
	"rtier/internal/update"
)

// TestQuakePlanReplay decides a round of Quake's maintenance (U14, internal/update/quake.go) with
// one refinement iteration, as Quake's paper runs it (§7.2), saves its plan, gives the epoch up,
// and runs the saved plan instead on the same state (internal/update/quakeplan.go, PrepareQuakePlan),
// under query load. A plan decided on another state is refused, nothing changed; the replayed
// epoch makes the same edits as the decided one -- the same splits, deletes, lists, postings, PQ
// codes and graph patch -- the lists hold what the plan says, every vector stays in one list, and
// every answer equals the oracle's for its epoch.
func TestQuakePlanReplay(t *testing.T) {
	f := getQuakeFixture(t)
	m, err := partitioning.LoadManifest(f.parts)
	if err != nil {
		t.Fatal(err)
	}
	room := []string{"--max-vectors", strconv.Itoa(numVectors + 1024),
		"--max-pages", strconv.FormatUint(m.NumPages+1024/m.VectorsPerPage+2, 10),
		"--max-list-len", strconv.Itoa(4 * m.MaxListLen), "--graph-headroom", "1"}
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
	raddr, stopRouter, err := update.StartRouter(filepath.Join(f.bin, "rtier_node"), f.indexNoBootstrap, f.parts, logf,
		"--graph-headroom", "1")
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

	// The size-bound round of TestQuakeMaintenance, with one refinement pass: no head moves.
	lens, err := postingLengths(filepath.Join(f.index, "postings.bin"))
	if err != nil {
		t.Fatal(err)
	}
	sorted := append([]int(nil), lens...)
	sort.Ints(sorted)
	p := update.DefaultQuakeParams()
	p.MaxSize, p.MinSize = sorted[len(sorted)-3], sorted[0]+1
	p.RefineRadius, p.RefineIterations = 6, 1
	u := &update.Updater{Controller: cl.ctl, Manifest: m, Alloc: alloc, Router: router,
		GraphPath: filepath.Join(f.indexNoBootstrap, "graph.hnsw"), PatchDir: t.TempDir(), Policy: update.PolicyQuake, Quake: p}
	defer u.Close()
	oracleVersions := make([]uint32, numPartitions)

	stop := startLoad(cl, 4)
	time.Sleep(300 * time.Millisecond)

	// Decided, saved, given up.
	mj1, st1, err := u.PrepareQuake(ctx)
	if err != nil || mj1 == nil {
		t.Fatalf("deciding the round: %v (%+v)", err, st1)
	}
	if st1.Splits < 1 || st1.Deletes < 1 || st1.Refined == 0 || st1.Moved != 0 || mj1.QuakePlan == nil {
		t.Fatalf("decided round %+v: want lists split, deleted and refined, no head moved (one pass), and a plan", *st1)
	}
	path := filepath.Join(t.TempDir(), "quake-plan.json.gz")
	if err := update.WriteQuakePlan(path, mj1.QuakePlan); err != nil {
		t.Fatal(err)
	}
	plan, err := update.ReadQuakePlan(path)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := json.Marshal(mj1.QuakePlan)
	got, _ := json.Marshal(plan)
	if !bytes.Equal(want, got) {
		t.Fatal("the plan read back is not the one written")
	}
	if err := u.DiscardMajor(ctx, mj1); err != nil {
		t.Fatalf("giving the decided round up: %v", err)
	}

	// A plan decided on another state is refused, and changes nothing.
	tampered := *plan
	tampered.Read = map[uint32]update.ListDigest{}
	var some uint32
	for l, d := range plan.Read {
		tampered.Read[l] = d
		some = l
	}
	d := tampered.Read[some]
	d.Postings++
	tampered.Read[some] = d
	if _, _, err := u.PrepareQuakePlan(ctx, &tampered); err == nil || !strings.Contains(err.Error(), "not the state it was decided on") {
		t.Fatalf("a plan decided on another state: %v, want it refused", err)
	}

	// The saved plan instead: the same edits as decided.
	mj2, st2, err := u.PrepareQuakePlan(ctx, plan)
	if err != nil {
		t.Fatalf("running the saved plan: %v", err)
	}
	if !st2.Replayed || st2.Splits != st1.Splits || st2.Deletes != st1.Deletes || st2.Emptied != st1.Emptied {
		t.Errorf("replayed round %+v, decided %+v", *st2, *st1)
	}
	for name, pair := range map[string][2]any{
		"graph version": {mj1.Graph, mj2.Graph}, "graph patch": {mj1.Patch, mj2.Patch},
		"splits": {mj1.Splits, mj2.Splits}, "deletes": {mj1.Merges, mj2.Merges}, "lists created": {mj1.Lists, mj2.Lists},
		"partition edits": {mj1.Parts, mj2.Parts}, "PQ codes sent": {mj1.Codes, mj2.Codes},
	} {
		if !reflect.DeepEqual(pair[0], pair[1]) {
			t.Errorf("the replayed round's %s differ from the decided round's", name)
		}
	}
	t.Logf("decided %+v; replayed %d splits, %d deletes (with the emptied lists), %d partitions, a %d-byte patch",
		*st1, len(mj2.Splits), len(mj2.Merges), len(mj2.Parts), len(mj2.Patch))

	applyMajorOracle(t, ctx, oracle, mj2, oracleVersions)
	from := mj2.Lease.Table.Epoch
	cl.expect(from+1, oracleAnswers(t, ctx, f, oracle))
	reply, err := u.ExecuteMajor(ctx, mj2)
	if err != nil {
		t.Fatalf("executing the saved plan: %v", err)
	}
	if reply.FromEpoch != from || reply.ToEpoch != from+1 || reply.Graph != mj2.Lease.Table.Graph+1 {
		t.Fatalf("reply %+v, want a flip from epoch %d to graph version %d", reply, from, mj2.Lease.Table.Graph+1)
	}
	checkUpdateFlip(t, "replayed round", reply)

	// The lists hold what the plan says; the split, deleted and emptied lists are gone.
	lists := allLists(t, ctx, cl.ctl.Status().Table, m)
	asSet := func(ids []uint32) map[uint32]bool {
		s := map[uint32]bool{}
		for _, id := range ids {
			s[id] = true
		}
		return s
	}
	gone := append(append([]uint32(nil), plan.Deletes...), plan.Emptied...)
	for _, sp := range plan.Splits {
		gone = append(gone, sp.List)
	}
	for _, l := range gone {
		if _, ok := lists[l]; ok {
			t.Errorf("list %d is still there, which the plan removes", l)
		}
	}
	for l, ids := range plan.Final {
		if containsU32(plan.Emptied, l) {
			continue
		}
		if !reflect.DeepEqual(asSet(lists[l]), asSet(ids)) {
			t.Errorf("list %d holds %d postings, the plan %d", l, len(lists[l]), len(ids))
		}
	}
	for l, ids := range plan.Gains {
		have := asSet(lists[l])
		for _, id := range ids {
			if !have[id] {
				t.Errorf("list %d lacks vector %d, which the plan sends there", l, id)
			}
		}
	}
	one := map[uint32]uint32{}
	for l, ids := range lists {
		for _, id := range ids {
			if o, ok := one[id]; ok {
				t.Fatalf("vector %d is in lists %d and %d", id, o, l)
			}
			one[id] = l
		}
	}
	if len(one) != numVectors {
		t.Fatalf("%d vectors in the lists, want %d", len(one), numVectors)
	}

	time.Sleep(300 * time.Millisecond)
	ls := stop()
	t.Logf("%d queries, %d retries, max latency %v, per epoch %v", ls.total, ls.retries, ls.maxLatency, ls.epochs)
	if ls.mismatches > 0 || ls.errors > 0 {
		t.Fatalf("%d mismatches, %d errors; first: %s", ls.mismatches, ls.errors, ls.firstProblem)
	}
	if len(ls.epochs) < 2 {
		t.Fatalf("load did not span the flip: %v", ls.epochs)
	}
	exactPass(t, cl, nil)
}
