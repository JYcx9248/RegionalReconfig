package query

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"net"
	"reflect"
	"sort"
	"sync"
	"sync/atomic"
	"testing"

	"rtier/internal/design"
	"rtier/internal/epoch"
	"rtier/internal/frame"
	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
	"rtier/internal/placement"
)

func TestGroupByOwner(t *testing.T) {
	m := &partitioning.Manifest{NumLists: 6, NumPartitions: 3, ListPart: []uint32{0, 0, 1, 1, 2, 2}}
	tb := &epoch.Table{Epoch: 1, Placement: placement.Table{Owners: []placement.NodeID{7, 3, 7}}}
	gs := GroupByOwner([]uint32{5, 2, 0, 3}, m, tb)
	want := []Group{{Node: 3, Lists: []uint32{2, 3}, Versions: []uint32{0, 0}},
		{Node: 7, Lists: []uint32{5, 0}, Versions: []uint32{0, 0}}}
	if !reflect.DeepEqual(gs, want) {
		t.Fatalf("got %+v", gs)
	}
	// Each list carries the version of its partition in the table.
	tb.Versions = []uint32{4, 0, 9}
	want[1].Versions = []uint32{9, 4}
	if gs := GroupByOwner([]uint32{5, 2, 0, 3}, m, tb); !reflect.DeepEqual(gs, want) {
		t.Fatalf("with versions: got %+v", gs)
	}
}

func TestMergeTopK(t *testing.T) {
	c := func(id uint32, d float32) nodeclient.Candidate { return nodeclient.Candidate{ID: id, Dist: d} }
	a := []nodeclient.Candidate{c(4, 1), c(9, 2), c(1, 3)}
	b := []nodeclient.Candidate{c(9, 2), c(2, 2), c(8, 0.5)}
	got := MergeTopK([][]nodeclient.Candidate{a, b}, 4)
	want := []nodeclient.Candidate{c(8, 0.5), c(4, 1), c(2, 2), c(9, 2)} // tie on 2 broken by ID
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v", got)
	}
	// Order of inputs does not matter.
	if !reflect.DeepEqual(MergeTopK([][]nodeclient.Candidate{b, a}, 4), want) {
		t.Fatal("merge depends on input order")
	}
}

// mergeBySort is what MergeTopK computed before the heap merge: every candidate, one entry
// per ID at its smallest distance, all of them sorted.
func mergeBySort(parts [][]nodeclient.Candidate, k int) []nodeclient.Candidate {
	best := map[uint32]float32{}
	for _, p := range parts {
		for _, c := range p {
			if d, ok := best[c.ID]; !ok || c.Dist < d {
				best[c.ID] = c.Dist
			}
		}
	}
	out := make([]nodeclient.Candidate, 0, len(best))
	for id, d := range best {
		out = append(out, nodeclient.Candidate{ID: id, Dist: d})
	}
	sort.Slice(out, func(i, j int) bool { return nodeclient.Less(out[i], out[j]) })
	if k >= 0 && len(out) > k {
		out = out[:k]
	}
	return out
}

func TestMergeTopKMatchesSort(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 2000; trial++ {
		// Few distinct distances and IDs: ties, and vectors several lists report, at the same
		// distance or (another backend's rounding) a slightly different one.
		parts := make([][]nodeclient.Candidate, rng.Intn(6))
		for i := range parts {
			seen := map[uint32]bool{}
			for n := rng.Intn(30); len(parts[i]) < n; {
				id := uint32(rng.Intn(60))
				if seen[id] {
					continue
				}
				seen[id] = true
				d := float32(id%13) + float32(rng.Intn(2))*1e-6
				parts[i] = append(parts[i], nodeclient.Candidate{ID: id, Dist: d})
			}
			if rng.Intn(4) > 0 { // most come sorted, as from a node
				sort.Slice(parts[i], func(a, b int) bool { return nodeclient.Less(parts[i][a], parts[i][b]) })
			}
		}
		before := fmt.Sprint(parts)
		k := rng.Intn(70) - 5
		got, want := MergeTopK(parts, k), mergeBySort(parts, k)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("trial %d, k %d: got %v, want %v (parts %v)", trial, k, got, want, parts)
		}
		if fmt.Sprint(parts) != before {
			t.Fatalf("trial %d: the merge changed its input", trial)
		}
	}
}

func TestSplitByOrigin(t *testing.T) {
	c := func(id uint32, d float32) nodeclient.Candidate { return nodeclient.Candidate{ID: id, Dist: d} }
	// Vector 9 is a boundary vector: both owners report it, so it goes to the first one.
	parts := [][]nodeclient.Candidate{{c(4, 1), c(9, 2)}, {c(8, 0.5), c(9, 2), c(2, 3)}}
	merged := MergeTopK(parts, 3)
	if want := []nodeclient.Candidate{c(8, 0.5), c(4, 1), c(9, 2)}; !reflect.DeepEqual(merged, want) {
		t.Fatalf("merged %v", merged)
	}
	got := splitByOrigin(parts, merged)
	if want := [][]uint32{{4, 9}, {8}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("split %v", got)
	}
	// Every merged candidate is re-ranked exactly once.
	total := 0
	for _, ids := range got {
		total += len(ids)
	}
	if total != len(merged) {
		t.Fatalf("%d ids re-ranked, %d candidates merged", total, len(merged))
	}
}

func TestPlaceholders(t *testing.T) {
	// two-phase is implemented (see TestSplitByOrigin and test/e2e); the other option is open.
	for _, name := range []string{"proportional-quota"} {
		s, err := NewStrategy(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := s.Execute(nil, nil, nil, nil); !errors.Is(err, design.ErrUndecided) {
			t.Fatalf("%s: %v", name, err)
		}
	}
	if s, err := NewStrategy(""); err != nil || s.Name() != "two-phase" {
		t.Fatalf("no strategy named: %v, %v; want two-phase, the decided answer to U5", s, err)
	}
	s, _ := NewSelector("outsource")
	if _, err := s.Pick(nil, 1, nil, nil); !errors.Is(err, design.ErrUndecided) {
		t.Fatal("outsourcing must be undecided")
	}
	l, _ := NewSelector("local")
	if id, _ := l.Pick(nil, 4, nil, nil); id != 4 {
		t.Fatal("local selector")
	}
}

func TestWarmupSelector(t *testing.T) {
	s, err := NewSelector("warmup")
	if err != nil {
		t.Fatal(err)
	}
	// Steady state: every aggregator is an entry, so the entry aggregates its own queries.
	steady := &epoch.Table{Epoch: 7, Entries: []placement.NodeID{1, 2}, Aggregators: []placement.NodeID{1, 2}}
	if got, err := s.Pick(steady, 1, nil, nil); err != nil || got != 1 {
		t.Fatalf("steady state: %v %v", got, err)
	}
	// Scale-out window: nodes 3 and 4 may aggregate but have no graph yet, so they take turns.
	window := &epoch.Table{Epoch: 8, Entries: []placement.NodeID{1, 2},
		Aggregators: []placement.NodeID{1, 2, 4, 3}}
	seen := map[placement.NodeID]int{}
	for i := 0; i < 10; i++ {
		got, err := s.Pick(window, 1, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		if got != 3 && got != 4 {
			t.Fatalf("picked %d, want a node that is aggregating but not an entry", got)
		}
		seen[got]++
	}
	if seen[3] != 5 || seen[4] != 5 {
		t.Fatalf("round robin over the warming nodes: %v", seen)
	}
}

func TestOwnerSelector(t *testing.T) {
	s, err := NewSelector("")
	if err != nil || s.Name() != "owner" {
		t.Fatalf("default selector: %v %v", s, err)
	}
	tbl := &epoch.Table{Epoch: 3, Entries: []placement.NodeID{1, 2, 3}, Aggregators: []placement.NodeID{1, 2, 3}}
	pick := func(self placement.NodeID, owners ...placement.NodeID) placement.NodeID {
		var gs []Group
		for _, o := range owners {
			gs = append(gs, Group{Node: o, Lists: []uint32{uint32(o)}})
		}
		got, err := s.Pick(tbl, self, nil, gs)
		if err != nil {
			t.Fatal(err)
		}
		return got
	}
	if got := pick(1, 1, 2); got != 1 {
		t.Errorf("entry among the owners: picked %d, want the entry", got)
	}
	if got := pick(3, 2); got != 2 {
		t.Errorf("one owner, not the entry: picked %d, want the owner", got)
	}
	if got := pick(3, 1, 2); got != 3 {
		t.Errorf("several owners, entry not among them: picked %d, want the entry", got)
	}
	// An owner that may not aggregate in this epoch is never picked.
	noAgg := &epoch.Table{Epoch: 3, Entries: []placement.NodeID{1}, Aggregators: []placement.NodeID{1}}
	if got, _ := s.Pick(noAgg, 1, nil, []Group{{Node: 2}}); got != 1 {
		t.Errorf("owner without the aggregator role: picked %d, want the entry", got)
	}
}

// TestAggregateWire: a forwarded query carries the entry's grouping, not just the lists (the
// aggregator does not regroup), and malformed groups are refused.
func TestAggregateWire(t *testing.T) {
	req := &Request{Vec: []byte{1, 2, 3, 4}, Params: Params{K: 10, NProbe: 4, N: 200}}
	groups := []Group{{Node: 3, Lists: []uint32{2, 3}, Versions: []uint32{0, 1}},
		{Node: 7, Lists: []uint32{5, 0}, Versions: []uint32{2, 0}}}
	got, gotGroups, err := DecodeAggregate(EncodeAggregate(req, groups))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotGroups, groups) || got.Params != req.Params || string(got.Vec) != string(req.Vec) {
		t.Fatalf("round trip: %+v %+v", got, gotGroups)
	}
	// A group without versions travels at version 0.
	_, gotGroups, err = DecodeAggregate(EncodeAggregate(req, []Group{{Node: 3, Lists: []uint32{2, 3}}}))
	if err != nil || !reflect.DeepEqual(gotGroups[0].Versions, []uint32{0, 0}) {
		t.Fatalf("without versions: %+v %v", gotGroups, err)
	}
	for name, gs := range map[string][]Group{
		"no groups":   nil,
		"empty group": {{Node: 3}},
		"too many":    {{Node: 3, Lists: make([]uint32, MaxNProbe+1)}},
	} {
		if _, _, err := DecodeAggregate(EncodeAggregate(req, gs)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if _, err := DecodeRequest(EncodeRequest(req)); err != nil {
		t.Fatal(err)
	}
}

// fakeAgent answers every request on its query port with OK and no candidates, counting the
// connections it accepted; drop closes them all from its side.
func fakeAgent(t *testing.T) (addr string, conns *atomic.Int64, drop func()) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	conns = new(atomic.Int64)
	var (
		mu   sync.Mutex
		open []net.Conn
	)
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			conns.Add(1)
			mu.Lock()
			open = append(open, c)
			mu.Unlock()
			go func() {
				defer c.Close()
				for {
					var f frame.Frame
					if frame.Read(c, &f) != nil {
						return
					}
					resp := frame.Frame{Type: f.Type, Flags: frame.FlagResponse, ReqID: f.ReqID}
					if f.Type != OpPing {
						resp.Body = EncodeResult(nil)
					}
					if frame.Write(c, &resp) != nil {
						return
					}
				}
			}()
		}
	}()
	drop = func() {
		mu.Lock()
		defer mu.Unlock()
		for _, c := range open {
			c.Close()
		}
	}
	return ln.Addr().String(), conns, drop
}

func TestClientPreconnect(t *testing.T) {
	addr, conns, drop := fakeAgent(t)
	c := NewClient(addr, 3)
	ctx := context.Background()
	if err := c.Preconnect(ctx); err != nil {
		t.Fatal(err)
	}
	if c.Open() != 3 || conns.Load() != 3 {
		t.Fatalf("%d open, %d accepted: want 3", c.Open(), conns.Load())
	}
	// As many queries at once as the client allows find their connections open.
	req := &Request{Vec: []byte{1, 2, 3, 4}, Params: Params{K: 10, NProbe: 4, N: 200}}
	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := c.Query(ctx, req); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if err := c.Preconnect(ctx); err != nil || conns.Load() != 3 || c.Open() != 3 {
		t.Fatalf("after the queries and a second Preconnect: %v, %d accepted, %d open", err, conns.Load(), c.Open())
	}
	// A connection that fails is closed and no longer counted; the next request opens another.
	drop()
	if _, err := c.Query(ctx, req); err == nil {
		t.Fatal("query on a connection the agent closed succeeded")
	}
	if c.Open() != 2 {
		t.Fatalf("%d open after a failed query: want 2", c.Open())
	}
	c.Close()
	if c.Open() != 0 {
		t.Fatalf("%d open after Close", c.Open())
	}
	if _, err := c.Query(ctx, req); err != nil || conns.Load() != 4 || c.Open() != 1 {
		t.Fatalf("query after Close: %v, %d accepted, %d open", err, conns.Load(), c.Open())
	}
	if err := NewClient("127.0.0.1:1", 1).Preconnect(ctx); err == nil {
		t.Fatal("Preconnect to a port nobody listens on succeeded")
	}
}
