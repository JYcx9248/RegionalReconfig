package placement

import (
	"math/rand"
	"testing"
)

// evenCounts reports whether t gives every node floor(P/N) or ceil(P/N) partitions.
func evenCounts(t Table, nodes []NodeID) bool {
	np, nn := t.NumPartitions(), len(nodes)
	c := t.Counts()
	for _, n := range nodes {
		if c[n] < np/nn || c[n] > (np+nn-1)/nn {
			return false
		}
	}
	return len(c) == nn
}

// bestBalanced is the largest total weight of a count-balanced placement, by enumeration.
func bestBalanced(w [][]int64, nodes []NodeID) int64 {
	np, nn := len(w), len(nodes)
	best := int64(-1)
	owners := make([]int, np)
	var rec func(p int)
	rec = func(p int) {
		if p == np {
			cnt := make([]int, nn)
			var sum int64
			for q, i := range owners {
				cnt[i]++
				sum += w[q][i]
			}
			for _, k := range cnt {
				if k < np/nn || k > (np+nn-1)/nn {
					return
				}
			}
			best = max(best, sum)
			return
		}
		for i := 0; i < nn; i++ {
			owners[p] = i
			rec(p + 1)
		}
	}
	rec(0)
	return best
}

func TestAffinity(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	nodes := []NodeID{4, 7, 9}
	for trial := 0; trial < 40; trial++ {
		np := 3 + rng.Intn(5) // 3..7: balanced with and without a remainder
		w := make([][]int64, np)
		for p := range w {
			w[p] = make([]int64, len(nodes))
			for i := range w[p] {
				w[p][i] = int64(rng.Intn(100))
			}
		}
		tbl, err := Affinity(w, nodes)
		if err != nil {
			t.Fatal(err)
		}
		if !evenCounts(tbl, nodes) {
			t.Fatalf("trial %d: %v is not balanced", trial, tbl.Counts())
		}
		var sum int64
		for p, n := range tbl.Owners {
			for i, m := range nodes {
				if m == n {
					sum += w[p][i]
				}
			}
		}
		if want := bestBalanced(w, nodes); sum != want {
			t.Fatalf("trial %d: total weight %d, the best balanced placement has %d", trial, sum, want)
		}
	}
	// A partition goes where its weight is when the balance allows it.
	tbl, err := Affinity([][]int64{{0, 9}, {9, 0}, {0, 9}, {9, 0}}, []NodeID{1, 2})
	if err != nil || tbl.Owners[0] != 2 || tbl.Owners[1] != 1 || tbl.Owners[2] != 2 || tbl.Owners[3] != 1 {
		t.Fatalf("got %v, %v", tbl.Owners, err)
	}
	if _, err := Affinity([][]int64{{1, 2}}, []NodeID{1, 2}); err == nil {
		t.Fatal("one partition placed on two nodes")
	}
}

func TestRandomBalanced(t *testing.T) {
	nodes := []NodeID{3, 1, 2}
	a, err := RandomBalanced(11, nodes, 5)
	if err != nil || !evenCounts(a, nodes) {
		t.Fatalf("%v %v", a.Counts(), err)
	}
	b, _ := RandomBalanced(11, nodes, 5)
	c, _ := RandomBalanced(11, nodes, 6)
	same := func(x, y Table) bool {
		for p := range x.Owners {
			if x.Owners[p] != y.Owners[p] {
				return false
			}
		}
		return true
	}
	if !same(a, b) || same(a, c) {
		t.Fatal("the placement should depend on the seed, and only on it")
	}
}
