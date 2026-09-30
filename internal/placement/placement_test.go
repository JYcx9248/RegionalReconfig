package placement

import (
	"errors"
	"reflect"
	"testing"

	"rtier/internal/design"
)

func balanced(t *testing.T, tb Table, nodes []NodeID) {
	t.Helper()
	c := tb.Counts()
	if len(c) != len(nodes) {
		t.Fatalf("owners %v, want exactly %v", c, nodes)
	}
	lo, hi := tb.NumPartitions(), 0
	for _, n := range nodes {
		lo, hi = min(lo, c[n]), max(hi, c[n])
	}
	if hi-lo > 1 {
		t.Fatalf("unbalanced counts %v", c)
	}
}

func TestEvenScaleOutIn(t *testing.T) {
	var p EvenPolicy
	tb, err := p.Initial(256, []NodeID{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	balanced(t, tb, []NodeID{0, 1})

	// 2 -> 3 -> 5 nodes: only existing -> new moves, every new node fed by every donor.
	cur := tb
	for _, nodes := range [][]NodeID{{0, 1, 2}, {0, 1, 2, 3, 4}} {
		next, ch, err := p.Repartition(cur, nodes, nil)
		if err != nil {
			t.Fatal(err)
		}
		balanced(t, next, nodes)
		if !reflect.DeepEqual(Diff(cur, next), ch) {
			t.Fatalf("changes %v disagree with diff %v", ch, Diff(cur, next))
		}
		oldSet := map[NodeID]bool{}
		for _, n := range cur.Nodes() {
			oldSet[n] = true
		}
		for src, m := range ch {
			if !oldSet[src] {
				t.Fatalf("new node %d donated partitions", src)
			}
			for dst := range m {
				if oldSet[dst] {
					t.Fatalf("partitions moved between existing nodes %d -> %d", src, dst)
				}
			}
		}
		moved, ideal := len(ch.Moved()), 0
		for _, n := range nodes {
			if !oldSet[n] {
				ideal += next.Counts()[n]
			}
		}
		if moved != ideal {
			t.Fatalf("moved %d partitions, minimum is %d", moved, ideal)
		}
		cur = next
	}

	// 5 -> 3 nodes: removed nodes donate everything, remaining nodes keep theirs.
	next, ch, err := p.Repartition(cur, []NodeID{0, 2, 4}, nil)
	if err != nil {
		t.Fatal(err)
	}
	balanced(t, next, []NodeID{0, 2, 4})
	for src := range ch {
		if src != 1 && src != 3 {
			t.Fatalf("remaining node %d gave up partitions", src)
		}
	}
	if len(next.PartitionsOf(1)) != 0 || len(next.PartitionsOf(3)) != 0 {
		t.Fatal("removed nodes still own partitions")
	}
}

func TestEvenUnevenRemainder(t *testing.T) {
	// Koala gives the extra partition of 9/4 to the first node of the list (node 0), which owns
	// only 2 and would have to gain one during a scale-out: Koala stops with a fatal error.
	var p EvenPolicy
	old := Table{Owners: []NodeID{0, 0, 1, 1, 1, 2, 2, 2, 2}} // 0:2 1:3 2:4
	next, _, err := p.Repartition(old, []NodeID{0, 1, 2, 3}, nil)
	if err != nil {
		t.Fatal(err)
	}
	balanced(t, next, []NodeID{0, 1, 2, 3})
	// An unchanged node set is a no-op; mixed add+remove is rejected.
	if _, ch, err := p.Repartition(old, []NodeID{2, 1, 0}, nil); err != nil || len(ch) != 0 {
		t.Fatalf("same set: %v %v", ch, err)
	}
	if _, _, err := p.Repartition(old, []NodeID{0, 1, 3}, nil); err == nil {
		t.Fatal("mixed add/remove accepted")
	}
	if _, _, err := p.Repartition(old, []NodeID{0, 0}, nil); err == nil {
		t.Fatal("duplicate node accepted")
	}
}

func TestRangesPlanHistory(t *testing.T) {
	tb := Table{Owners: []NodeID{1, 1, 2, 2, 2, 1, 3}}
	rs := tb.Ranges()
	want := []Range{{1, 0, 1}, {2, 2, 4}, {1, 5, 5}, {3, 6, 6}}
	if !reflect.DeepEqual(rs, want) {
		t.Fatalf("ranges %v", rs)
	}
	back, err := FromRanges(7, rs)
	if err != nil || !reflect.DeepEqual(back, tb) {
		t.Fatalf("round trip: %v %v", back, err)
	}
	if _, err := FromRanges(8, rs); err == nil {
		t.Fatal("missing owner not detected")
	}

	next := Table{Owners: []NodeID{1, 4, 2, 4, 2, 4, 3}}
	plan := MakePlan(Diff(tb, next))
	if len(plan) != 1 || len(plan[4]) != 2 {
		t.Fatalf("plan %+v", plan)
	}
	if plan[4][0].From != 1 || !reflect.DeepEqual(plan[4][0].Partitions, []int{1, 5}) ||
		plan[4][1].From != 2 || !reflect.DeepEqual(plan[4][1].Partitions, []int{3}) {
		t.Fatalf("plan %+v", plan[4])
	}

	h := NewHistory(tb)
	h.Record(next)
	if !h.HadCopy(1, 1) || !h.HadCopy(1, 4) || h.HadCopy(1, 2) {
		t.Fatal("history")
	}
	h.Forget(1, 1)
	if h.HadCopy(1, 1) {
		t.Fatal("forget")
	}
	// Forget drops the copy, not the fact that node 1 owned it: its PQ codes are still there.
	if !h.HasOwned(1, 1) || !h.HasOwned(1, 4) || h.HasOwned(1, 2) {
		t.Fatal("ever-owned record")
	}
}

func TestWeightedIsPlaceholder(t *testing.T) {
	p, err := NewPolicy("weighted")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Initial(4, []NodeID{0}); !errors.Is(err, design.ErrUndecided) {
		t.Fatalf("got %v", err)
	}
}

// roundTrip scales P partitions from n nodes out to m and back to n (the controller removes
// the nodes that joined), and reports how many partitions end up with their original owner.
func roundTrip(t *testing.T, p Policy, P, n, m int) (same, moved int) {
	t.Helper()
	ids := func(k int) []NodeID {
		out := make([]NodeID, k)
		for i := range out {
			out[i] = NodeID(i)
		}
		return out
	}
	a, err := p.Initial(P, ids(n))
	if err != nil {
		t.Fatal(err)
	}
	h := NewHistory(a)
	b, _, err := p.Repartition(a, ids(m), h)
	if err != nil {
		t.Fatal(err)
	}
	h.Record(b)
	c, _, err := p.Repartition(b, ids(n), h)
	if err != nil {
		t.Fatal(err)
	}
	balanced(t, c, ids(n))
	for i := range a.Owners {
		if a.Owners[i] == c.Owners[i] {
			same++
		}
		if a.Owners[i] != b.Owners[i] {
			moved++
		}
	}
	return same, moved
}

// Koala's even policy is not reversible: the partitions that come back land wherever the
// round-robin happens to point. Only the ones that never moved are still home.
func TestEvenIsNotReversible(t *testing.T) {
	for _, c := range [][3]int{{8, 2, 4}, {64, 4, 8}, {64, 4, 5}, {16, 2, 4}, {64, 4, 16}} {
		same, moved := roundTrip(t, EvenPolicy{}, c[0], c[1], c[2])
		if same != c[0]-moved {
			t.Errorf("P=%d %d->%d->%d: %d partitions home, expected only the %d that never moved",
				c[0], c[1], c[2], c[1], same, c[0]-moved)
		}
		t.Logf("P=%d %d->%d->%d: %d/%d home (%d moved out)", c[0], c[1], c[2], c[1], same, c[0], moved)
	}
}

// ReversiblePolicy restores the placement exactly, so every returning partition finds its PQ
// codes already there.
func TestReversibleRestoresPlacement(t *testing.T) {
	for _, c := range [][3]int{{8, 2, 4}, {64, 4, 8}, {64, 4, 5}, {16, 2, 4}, {64, 8, 16}, {12, 3, 6}, {64, 4, 16}, {64, 3, 6}} {
		if same, moved := roundTrip(t, ReversiblePolicy{}, c[0], c[1], c[2]); same != c[0] {
			t.Errorf("P=%d %d->%d->%d: only %d/%d partitions came home (%d moved out)",
				c[0], c[1], c[2], c[1], same, c[0], moved)
		}
	}
	// Two round trips in a row, and a scale-in that removes a node which never joined: the
	// policy still balances and only moves partitions of the removed nodes.
	p := ReversiblePolicy{}
	a, err := p.Initial(64, []NodeID{0, 1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	h := NewHistory(a)
	cur := a
	for _, nodes := range [][]NodeID{{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3}, {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3}} {
		next, ch, err := p.Repartition(cur, nodes, h)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(Diff(cur, next), ch) {
			t.Fatalf("changes disagree with diff")
		}
		h.Record(next)
		balanced(t, next, nodes)
		cur = next
	}
	if !reflect.DeepEqual(cur, a) {
		t.Fatalf("after two round trips the placement is %v, want %v", cur.Owners, a.Owners)
	}
	// Removing a node that owns partitions nobody else has ever owned still works (the
	// partitions fall through to the round-robin pass).
	if _, _, err := p.Repartition(cur, []NodeID{0, 1}, NewHistory(cur)); err != nil {
		t.Fatal(err)
	}
}

// BlockPolicy hands each new node a donor's tail whole: 2 -> 4 on 64 partitions gives
// quarters, where round-robin gives each new node two eighths from different donors. Where a
// new node needs every donor's excess (2 -> 3) both policies agree, and a scale-in still
// returns partitions home.
func TestBlocksHandOutWholeTails(t *testing.T) {
	two := []NodeID{1, 2}
	four := []NodeID{1, 2, 3, 4}
	start, err := BlockPolicy{}.Initial(64, two)
	if err != nil {
		t.Fatal(err)
	}
	next, _, err := BlockPolicy{}.Repartition(start, four, NewHistory(start))
	if err != nil {
		t.Fatal(err)
	}
	for p, o := range next.Owners {
		if want := []NodeID{1, 3, 2, 4}[p/16]; o != want {
			t.Fatalf("2 -> 4: partition %d on node %d, want %d (whole quarters)", p, o, want)
		}
	}
	rr, _, err := ReversiblePolicy{}.Repartition(start, four, NewHistory(start))
	if err != nil {
		t.Fatal(err)
	}
	if rr.Owners[16] != 4 || rr.Owners[31] != 3 {
		t.Fatalf("round-robin 2 -> 4 changed: partition 16 on %d, 31 on %d", rr.Owners[16], rr.Owners[31])
	}
	three := []NodeID{1, 2, 3}
	b3, _, err := BlockPolicy{}.Repartition(start, three, NewHistory(start))
	if err != nil {
		t.Fatal(err)
	}
	r3, _, err := ReversiblePolicy{}.Repartition(start, three, NewHistory(start))
	if err != nil {
		t.Fatal(err)
	}
	for p := range b3.Owners {
		if b3.Owners[p] != r3.Owners[p] {
			t.Fatalf("2 -> 3: partition %d on %d with blocks, %d round-robin", p, b3.Owners[p], r3.Owners[p])
		}
	}
	h := NewHistory(start)
	h.Record(next)
	back, _, err := BlockPolicy{}.Repartition(next, two, h)
	if err != nil {
		t.Fatal(err)
	}
	for p := range back.Owners {
		if back.Owners[p] != start.Owners[p] {
			t.Fatalf("4 -> 2: partition %d on %d, started on %d", p, back.Owners[p], start.Owners[p])
		}
	}
}
