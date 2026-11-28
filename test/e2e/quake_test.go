package e2e

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"rtier/internal/config"
	"rtier/internal/epoch"
	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
	"rtier/internal/query"
	"rtier/internal/update"
)

// TestQuakeMaintenance runs rounds of Quake's maintenance (U14, internal/update/quake.go) as
// major epochs on Quake's index under query load, and the same decisions on a single-node oracle.
// The first round bounds the lists' lengths (Quake's size-bound mode): the longest lists split by
// a k-means of 2 into lists headed by their means, the shortest are deleted -- each member to its
// nearest live head's list, on either owner -- the lists around the children are refined in three
// passes, which move their heads in place, and lists left empty go. The second decides on Quake's
// cost model, from the window of queries the entries kept. Every answer must equal the oracle's
// for its epoch, the lists must hold what the plan says, and every vector stays in one list.
func TestQuakeMaintenance(t *testing.T) {
	f := getQuakeFixture(t)
	m, err := partitioning.LoadManifest(f.parts)
	if err != nil {
		t.Fatal(err)
	}
	room := []string{"--max-vectors", strconv.Itoa(numVectors + 1024),
		"--max-pages", strconv.FormatUint(m.NumPages+1024/m.VectorsPerPage+2, 10),
		"--max-list-len", strconv.Itoa(4 * m.MaxListLen), "--graph-headroom", "1"}
	cl := newCluster(t, f, "lazy", 2, func(c *config.Controller) { c.AllowUpdates = true })
	// Quake's cost model counts the lists of a window of recent client queries, which the entries
	// keep only when asked (QuerySample is 0 by default since the rebuild's warm-up went).
	cl.agentCfg = func(_ string, c *config.Agent) { c.QuerySample = 1024 }
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

	// The size-bound round: the lists longer than the third longest split, the shortest go.
	lens, err := postingLengths(filepath.Join(f.index, "postings.bin"))
	if err != nil {
		t.Fatal(err)
	}
	if len(lens) != 80 {
		t.Fatalf("Quake's fixture has %d lists, want 80", len(lens))
	}
	sorted := append([]int(nil), lens...)
	sort.Ints(sorted)
	p := update.DefaultQuakeParams()
	p.MaxSize, p.MinSize = sorted[len(sorted)-3], sorted[0]+1
	p.RefineRadius, p.RefineIterations = 6, 3
	u := &update.Updater{Controller: cl.ctl, Manifest: m, Alloc: alloc, Router: router,
		GraphPath: filepath.Join(f.indexNoBootstrap, "graph.hnsw"), PatchDir: t.TempDir(), Policy: update.PolicyQuake, Quake: p}
	defer u.Close()
	oracleVersions := make([]uint32, numPartitions)

	stop := startLoad(cl, 4)
	time.Sleep(300 * time.Millisecond)

	round := func(name string) (*update.Major, *update.QuakeStats) {
		t.Helper()
		mj, st, err := u.PrepareQuake(ctx)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if mj == nil {
			t.Logf("%s: no round (%s), %+v", name, st.Skipped, *st)
			return nil, st
		}
		applyMajorOracle(t, ctx, oracle, mj, oracleVersions)
		from := mj.Lease.Table.Epoch
		cl.expect(from+1, oracleAnswers(t, ctx, f, oracle))
		reply, err := u.ExecuteMajor(ctx, mj)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		moved := 0
		for _, cs := range mj.Codes {
			moved += len(cs.IDs)
		}
		t.Logf("%s: %+v; splits %+v, deletes %+v, %d PQ codes to other owners; epoch %d -> %d, graph %d, partitions %v",
			name, *st, mj.Splits, mj.Merges, moved, reply.FromEpoch, reply.ToEpoch, reply.Graph, reply.Changed)
		if reply.FromEpoch != from || reply.ToEpoch != from+1 || reply.Graph != mj.Lease.Table.Graph+1 {
			t.Fatalf("%s: reply %+v, want a flip from epoch %d to graph version %d", name, reply, from, mj.Lease.Table.Graph+1)
		}
		checkUpdateFlip(t, name, reply)
		// The lists hold what the plan says: every added posting, no dropped one; the split and
		// deleted lists are gone, and every vector is in one list.
		after := cl.ctl.Status().Table
		lists := allLists(t, ctx, after, m)
		for _, pe := range mj.Parts {
			for i, l := range pe.Lists {
				if !containsU32(lists[l], pe.IDs[i]) {
					t.Errorf("%s: list %d lacks vector %d, which the epoch added", name, l, pe.IDs[i])
				}
			}
			for i, l := range pe.DropLists {
				if containsU32(lists[l], pe.DropIDs[i]) {
					t.Errorf("%s: list %d still holds vector %d, which the epoch moved out", name, l, pe.DropIDs[i])
				}
			}
		}
		for _, sp := range mj.Splits {
			if _, ok := lists[sp.List]; ok {
				t.Errorf("%s: split list %d is still there", name, sp.List)
			}
			for c := 0; c < 2; c++ {
				if got := len(lists[sp.Children[c]]); got != sp.Sizes[c] {
					t.Errorf("%s: list %d (of %d) holds %d postings, the plan %d", name, sp.Children[c], sp.List, got, sp.Sizes[c])
				}
			}
		}
		for _, mg := range mj.Merges {
			if _, ok := lists[mg.List]; ok {
				t.Errorf("%s: deleted list %d is still there", name, mg.List)
			}
		}
		in := map[uint32]uint32{}
		for l, ids := range lists {
			for _, id := range ids {
				if o, ok := in[id]; ok {
					t.Fatalf("%s: vector %d is in lists %d and %d", name, id, o, l)
				}
				in[id] = l
			}
		}
		if len(in) != numVectors {
			t.Fatalf("%s: %d vectors in the lists, want %d", name, len(in), numVectors)
		}
		return mj, st
	}

	mj, st := round("size-bound round")
	if mj == nil || st.Splits < 1 || st.Deletes < 1 || st.Refined == 0 || st.Moved == 0 || st.Reassigned == 0 {
		t.Fatalf("size-bound round %+v: want lists split and deleted, a refinement that moved heads and vectors", st)
	}
	for _, sp := range mj.Splits {
		if sp.Sizes[0]+sp.Sizes[1] == 0 {
			t.Errorf("split of list %d left both children empty", sp.List)
		}
	}

	// The cost model, on the window of queries the entries kept; without the delete rejection, as
	// Quake's Wikipedia runs (wiki.yaml), so that what it decides is more likely to be applied here.
	u.Quake = update.DefaultQuakeParams()
	u.Quake.Window, u.Quake.DeleteRejection = 200, false
	_, st = round("cost-model round")
	if st.Window != 200 || st.ScanFraction <= 0 || st.ScanFraction >= 1 || st.Lists == 0 {
		t.Fatalf("cost-model round %+v: want a full window and a scan fraction in (0, 1)", st)
	}

	time.Sleep(300 * time.Millisecond)
	ls := stop()
	t.Logf("%d queries, %d retries, max latency %v, per epoch %v", ls.total, ls.retries, ls.maxLatency, ls.epochs)
	if ls.mismatches > 0 || ls.errors > 0 {
		t.Fatalf("%d mismatches, %d errors; first: %s", ls.mismatches, ls.errors, ls.firstProblem)
	}
	if len(ls.epochs) < 2 {
		t.Fatalf("load did not span the flips: %v", ls.epochs)
	}
	exactPass(t, cl, nil)
}

// postingLengths is the length of every list of an index's postings.bin (u64 lists, u64 ids,
// u64 offsets[lists+1], u32 ids[]).
func postingLengths(path string) ([]int, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	le := binary.LittleEndian
	if len(b) < 16 {
		return nil, fmt.Errorf("%s: truncated", path)
	}
	nl := int(le.Uint64(b))
	if len(b) < 16+8*(nl+1) {
		return nil, fmt.Errorf("%s: truncated", path)
	}
	out := make([]int, nl)
	for c := range out {
		out[c] = int(le.Uint64(b[16+8*(c+1):]) - le.Uint64(b[16+8*c:]))
	}
	return out, nil
}

// allLists is every list of every partition at tb's versions, from its owner.
func allLists(t *testing.T, ctx context.Context, tb *epoch.Table, m *partitioning.Manifest) map[uint32][]uint32 {
	t.Helper()
	out := map[uint32][]uint32{}
	for p := 0; p < m.NumPartitions; p++ {
		nc := nodeclient.New(tb.Nodes[tb.Owner(p)].NodeAddr, 1)
		lens, err := nc.ListLengths(ctx, []int{p}, []uint32{tb.Version(p)})
		if err != nil {
			nc.Close()
			t.Fatalf("the lists of partition %d: %v", p, err)
		}
		for _, l := range lens {
			lp, err := nc.ListGet(ctx, l.List, tb.Version(p))
			if err != nil {
				nc.Close()
				t.Fatalf("list %d of partition %d: %v", l.List, p, err)
			}
			if len(lp.IDs) != int(l.Postings) {
				t.Errorf("list %d: %d postings, LIST_LENGTHS says %d", l.List, len(lp.IDs), l.Postings)
			}
			out[l.List] = lp.IDs
		}
		nc.Close()
	}
	return out
}
