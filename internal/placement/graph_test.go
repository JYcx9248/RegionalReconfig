package placement

import (
	"errors"
	"reflect"
	"testing"
)

// gridGraph is 64 partitions on an 8x8 grid, numbered in Z order (x, then y, from the top bit),
// as recursive bisection numbers them: ranges of 32, 16 and 8 are halves, quarters and eighths.
// Grid neighbors are adjacent with weight 1.
func gridGraph() Adjacency {
	id := func(x, y int) int {
		v := 0
		for b := 2; b >= 0; b-- {
			v = v<<2 | (x>>b&1)<<1 | y>>b&1
		}
		return v
	}
	a := make(Adjacency, 64)
	for p := range a {
		a[p] = make([]float64, 64)
	}
	for x := 0; x < 8; x++ {
		for y := 0; y < 8; y++ {
			if x+1 < 8 {
				a[id(x, y)][id(x+1, y)], a[id(x+1, y)][id(x, y)] = 1, 1
			}
			if y+1 < 8 {
				a[id(x, y)][id(x, y+1)], a[id(x, y+1)][id(x, y)] = 1, 1
			}
		}
	}
	return a
}

func graphPolicy(a Adjacency) GraphPolicy {
	return GraphPolicy{Adjacency: func() (Adjacency, error) { return a, nil }}
}

func nodeIDs(n int) []NodeID {
	ids := make([]NodeID, n)
	for i := range ids {
		ids[i] = NodeID(i + 1)
	}
	return ids
}

// Every step keeps even-reversible's counts and moves (existing -> new only, each donor giving
// as many as there), and never cuts more than blocks from the same table.
func TestGraphScaleOutKeepsCountsAndMoves(t *testing.T) {
	a := gridGraph()
	g := graphPolicy(a)
	for _, chain := range [][]int{{2, 3, 4, 5, 6, 7, 8}, {2, 4, 8}, {3, 5}} {
		cur, err := g.Initial(64, nodeIDs(chain[0]))
		if err != nil {
			t.Fatal(err)
		}
		for _, n := range chain[1:] {
			nodes := nodeIDs(n)
			next, ch, err := g.Repartition(cur, nodes, NewHistory(cur))
			if err != nil {
				t.Fatal(err)
			}
			balanced(t, next, nodes)
			if !reflect.DeepEqual(Diff(cur, next), ch) {
				t.Fatalf("%v -> %d: changes disagree with the diff", chain, n)
			}
			blocks, bch, err := BlockPolicy{}.Repartition(cur, nodes, NewHistory(cur))
			if err != nil {
				t.Fatal(err)
			}
			old := map[NodeID]bool{}
			for _, o := range cur.Nodes() {
				old[o] = true
			}
			for src, m := range ch {
				given, want := 0, 0
				for dst, ps := range m {
					if !old[src] || old[dst] {
						t.Fatalf("%v -> %d: partitions %v moved %d -> %d", chain, n, ps, src, dst)
					}
					given += len(ps)
				}
				for _, ps := range bch[src] {
					want += len(ps)
				}
				if given != want {
					t.Fatalf("%v -> %d: node %d gives %d partitions, blocks %d", chain, n, src, given, want)
				}
			}
			if a.Cut(next) > a.Cut(blocks)+1e-9 {
				t.Fatalf("%v -> %d: cut %.0f, blocks %.0f", chain, n, a.Cut(next), a.Cut(blocks))
			}
			t.Logf("%v -> %d: cut %.0f, blocks %.0f", chain, n, a.Cut(next), a.Cut(blocks))
			cur = next
		}
	}
}

// Doubling: a donor's tail half is already compact, and the graph does as well as blocks
// (whole quarters, cut 16 on the grid). 2 -> 3: blocks gives the new node both donors' tails,
// two separate pieces; the graph puts it across the old boundary.
func TestGraphDoublesLikeBlocksAndBeatsTails(t *testing.T) {
	a := gridGraph()
	g := graphPolicy(a)
	start, err := g.Initial(64, nodeIDs(2))
	if err != nil {
		t.Fatal(err)
	}
	four, _, err := g.Repartition(start, nodeIDs(4), NewHistory(start))
	if err != nil {
		t.Fatal(err)
	}
	if c := a.Cut(four); c != 16 {
		t.Fatalf("2 -> 4: cut %.0f, want the 16 of whole quarters", c)
	}
	three, _, err := g.Repartition(start, nodeIDs(3), NewHistory(start))
	if err != nil {
		t.Fatal(err)
	}
	tails, _, err := ReversiblePolicy{}.Repartition(start, nodeIDs(3), NewHistory(start))
	if err != nil {
		t.Fatal(err)
	}
	if a.Cut(three) >= a.Cut(tails) {
		t.Fatalf("2 -> 3: cut %.0f, tails %.0f", a.Cut(three), a.Cut(tails))
	}
}

// Scaling back in returns every partition to a node that has owned it (the PQ codes are there,
// U1), and a single round trip restores the placement exactly.
func TestGraphRoundTrips(t *testing.T) {
	a := gridGraph()
	g := graphPolicy(a)
	for _, c := range [][2]int{{2, 3}, {2, 4}, {4, 6}, {3, 8}} {
		start, err := g.Initial(64, nodeIDs(c[0]))
		if err != nil {
			t.Fatal(err)
		}
		h := NewHistory(start)
		out, _, err := g.Repartition(start, nodeIDs(c[1]), h)
		if err != nil {
			t.Fatal(err)
		}
		h.Record(out)
		back, _, err := g.Repartition(out, nodeIDs(c[0]), h)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(back, start) {
			t.Fatalf("%d -> %d -> %d: placement %v, started at %v", c[0], c[1], c[0], back.Owners, start.Owners)
		}
	}
	// A chain out and back one node at a time: every returning partition goes home, and the
	// placement ends where it started. Partitions that moved twice (1 -> 3 -> 5) are what the
	// seed's greedy home pass gets wrong here; the matching does not.
	cur, err := g.Initial(64, nodeIDs(2))
	if err != nil {
		t.Fatal(err)
	}
	start := cur
	h := NewHistory(cur)
	for _, n := range []int{3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2} {
		next, ch, err := g.Repartition(cur, nodeIDs(n), h)
		if err != nil {
			t.Fatal(err)
		}
		balanced(t, next, nodeIDs(n))
		if len(next.Nodes()) < len(cur.Nodes()) {
			for _, m := range ch {
				for dst, ps := range m {
					for _, p := range ps {
						if !h.HasOwned(p, dst) {
							t.Fatalf("-> %d: partition %d went to node %d, which never owned it", n, p, dst)
						}
					}
				}
			}
		}
		h.Record(next)
		cur = next
	}
	if !reflect.DeepEqual(cur, start) {
		t.Fatalf("out to 8 and back: placement %v, started at %v", cur.Owners, start.Owners)
	}
}

func TestGraphIsDeterministic(t *testing.T) {
	g := graphPolicy(gridGraph())
	start, err := g.Initial(64, nodeIDs(3))
	if err != nil {
		t.Fatal(err)
	}
	x, _, err := g.Repartition(start, nodeIDs(5), nil)
	if err != nil {
		t.Fatal(err)
	}
	y, _, err := g.Repartition(start, nodeIDs(5), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(x, y) {
		t.Fatalf("two runs differ: %v and %v", x.Owners, y.Owners)
	}
}

func TestGraphNeedsTheGraph(t *testing.T) {
	start, err := EvenPolicy{}.Initial(64, nodeIDs(2))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewPolicy("even-reversible-graph", nil); err == nil {
		t.Fatal("the graph policy was built without a graph")
	}
	if p, err := NewPolicy("", func() (Adjacency, error) { return gridGraph(), nil }); err != nil || p.Name() != "even-reversible-graph" {
		t.Fatalf("default policy %v, %v", p, err)
	}
	broken := errors.New("no graph")
	g := GraphPolicy{Adjacency: func() (Adjacency, error) { return nil, broken }}
	if _, _, err := g.Repartition(start, nodeIDs(3), nil); !errors.Is(err, broken) {
		t.Fatalf("got %v, want the loader's error", err)
	}
	// Nothing to move: the graph is not needed.
	if next, ch, err := g.Repartition(start, nodeIDs(2), nil); err != nil || len(ch) != 0 || !reflect.DeepEqual(next, start) {
		t.Fatalf("same nodes: %v %v %v", next.Owners, ch, err)
	}
	small := GraphPolicy{Adjacency: func() (Adjacency, error) { return Adjacency{{0}}, nil }}
	if _, _, err := small.Repartition(start, nodeIDs(3), nil); err == nil {
		t.Fatal("a graph of the wrong size was accepted")
	}
}
