package query

import (
	"errors"
	"reflect"
	"testing"

	"rtier/internal/design"
	"rtier/internal/epoch"
	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
	"rtier/internal/placement"
)

func TestGroupByOwner(t *testing.T) {
	m := &partitioning.Manifest{NumLists: 6, NumPartitions: 3, ListPart: []uint32{0, 0, 1, 1, 2, 2}}
	tb := &epoch.Table{Epoch: 1, Placement: placement.Table{Owners: []placement.NodeID{7, 3, 7}}}
	gs := GroupByOwner([]uint32{5, 2, 0, 3}, m, tb)
	want := []Group{{Node: 3, Lists: []uint32{2, 3}}, {Node: 7, Lists: []uint32{5, 0}}}
	if !reflect.DeepEqual(gs, want) {
		t.Fatalf("got %+v", gs)
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
	if _, err := NewStrategy(""); !errors.Is(err, design.ErrUndecided) {
		t.Fatal("empty strategy must be undecided")
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
	groups := []Group{{Node: 3, Lists: []uint32{2, 3}}, {Node: 7, Lists: []uint32{5, 0}}}
	got, gotGroups, err := DecodeAggregate(EncodeAggregate(req, groups))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotGroups, groups) || got.Params != req.Params || string(got.Vec) != string(req.Vec) {
		t.Fatalf("round trip: %+v %+v", got, gotGroups)
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
