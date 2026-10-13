package placement

import (
	"fmt"
	"math/rand"
)

// Adjacency is the partition graph: Adjacency[p][q] = Adjacency[q][p] >= 0 says how strongly
// partitions p and q touch (partitioning.PartitionGraph counts navigation-graph edges between
// their heads). A table's cut -- the weight between partitions on different owners -- stands in
// for its fan-out: a query probes the lists around it, and those fall on two owners only where
// that neighborhood crosses a node boundary.
type Adjacency [][]float64

// Cut is the weight of the graph between partitions on different owners.
func (a Adjacency) Cut(t Table) float64 {
	var c float64
	for p := range t.Owners {
		for q := p + 1; q < len(t.Owners); q++ {
			if t.Owners[p] != t.Owners[q] {
				c += a[p][q]
			}
		}
	}
	return c
}

func (a Adjacency) check(numPartitions int) error {
	if len(a) != numPartitions {
		return fmt.Errorf("placement: partition graph of %d partitions for %d", len(a), numPartitions)
	}
	for p, row := range a {
		if len(row) != numPartitions {
			return fmt.Errorf("placement: partition graph row %d has %d entries, want %d", p, len(row), numPartitions)
		}
	}
	return nil
}

// GraphPolicy (U3, the default) keeps even-reversible's counts and moves and makes the choice
// they leave open with the partition graph. A scale-out moves partitions only from existing to
// new nodes, each donor giving its excess as in even-reversible -- but which of its partitions
// a donor gives, and to which new node, is picked to cut as little of the graph as possible.
// A scale-in still sends every partition it can to a node that has owned it before (so the PQ
// codes are there, U1); the graph picks among those nodes and places the partitions that
// cannot go home.
//
// Why: taking each donor's tail (even-reversible, -blocks) ignores where partitions lie, so a
// new node collects pieces at both ends of the old nodes' ranges and more queries straddle two
// owners. On BIGANN-10M with 64 spatial partitions (nprobe 64), 2 -> 3 gives 1.72 owners per
// query against 1.85 and stages 5% fewer PQ codes, since neighboring partitions share
// replicated vectors; one node at a time up to eight, 3.25 against 3.83. When the cluster
// doubles, a donor's tail half is already the compact choice and the result cuts as much as
// blocks (quarters, eighths). The blocks table also seeds the search, so the result never cuts
// more of the graph than blocks would. What no choice of donated partitions removes is that
// layouts reached by single-node steps nest -- each node's region only shrinks -- and drift
// from a fresh even split (3.25 against 2.79 at eight nodes above); doubling avoids it.
type GraphPolicy struct {
	// Adjacency returns the partition graph. The controller loads it in the background when it
	// starts; the first reconfiguration waits for it.
	Adjacency func() (Adjacency, error)
}

// graphRestarts is how many randomized greedy starts a scale-out tries besides the blocks table.
const graphRestarts = 16

func (GraphPolicy) Name() string { return "even-reversible-graph" }

func (GraphPolicy) Initial(numPartitions int, nodes []NodeID) (Table, error) {
	return EvenPolicy{}.Initial(numPartitions, nodes)
}

func (g GraphPolicy) Repartition(old Table, nodes []NodeID, past Past) (Table, Changes, error) {
	seed, ch, err := repartitionEven(old, nodes, past, true)
	if err != nil || len(ch) == 0 {
		return seed, ch, err
	}
	if g.Adjacency == nil {
		return Table{}, nil, fmt.Errorf("placement: %s needs the partition graph", g.Name())
	}
	adj, err := g.Adjacency()
	if err != nil {
		return Table{}, nil, fmt.Errorf("placement: partition graph: %w", err)
	}
	if err := adj.check(old.NumPartitions()); err != nil {
		return Table{}, nil, err
	}
	s := newGraphSearch(adj, old, nodes)
	var next Table
	if len(s.added) > 0 {
		next = s.scaleOut(seed)
	} else {
		next = s.scaleIn(seed, past)
	}
	return next, Diff(old, next), nil
}

// graphSearch holds one repartitioning: the old table, the node indexes, and the connection of
// every partition to every node under the table being improved.
type graphSearch struct {
	adj   Adjacency
	old   Table
	nodes []NodeID       // old and next nodes, columns of conn
	col   map[NodeID]int // node -> column
	added map[NodeID]bool
	gone  map[NodeID]bool // removed nodes
	own   []NodeID        // the table being improved
	conn  [][]float64     // conn[p][col] = weight from p to the partitions on that node
	scale float64         // noise of the randomized starts
}

func newGraphSearch(adj Adjacency, old Table, next []NodeID) *graphSearch {
	s := &graphSearch{adj: adj, old: old, col: map[NodeID]int{}, added: map[NodeID]bool{}, gone: map[NodeID]bool{}}
	want := map[NodeID]bool{}
	for _, n := range next {
		want[n] = true
	}
	for _, n := range old.Nodes() {
		s.col[n] = len(s.nodes)
		s.nodes = append(s.nodes, n)
		if !want[n] {
			s.gone[n] = true
		}
	}
	for _, n := range next {
		if _, ok := s.col[n]; !ok {
			s.col[n] = len(s.nodes)
			s.nodes = append(s.nodes, n)
			s.added[n] = true
		}
	}
	var total float64
	for p := range adj {
		for _, w := range adj[p] {
			total += w
		}
	}
	if len(adj) > 0 {
		s.scale = 0.05 * total / float64(len(adj))
	}
	return s
}

// load makes t the table being improved.
func (s *graphSearch) load(t Table) {
	s.own = append(s.own[:0], t.Owners...)
	s.conn = make([][]float64, len(s.own))
	for p := range s.conn {
		s.conn[p] = make([]float64, len(s.nodes))
		for q, w := range s.adj[p] {
			s.conn[p][s.col[s.own[q]]] += w
		}
	}
}

func (s *graphSearch) move(p int, to NodeID) {
	from, dst := s.col[s.own[p]], s.col[to]
	for q := range s.conn {
		s.conn[q][from] -= s.adj[q][p]
		s.conn[q][dst] += s.adj[q][p]
	}
	s.own[p] = to
}

// swapGain is the change of the cut when p and q (on different nodes) trade owners.
func (s *graphSearch) swapGain(p, q int) float64 {
	x, y := s.col[s.own[p]], s.col[s.own[q]]
	return s.conn[p][x] - s.conn[p][y] + s.conn[q][y] - s.conn[q][x] + 2*s.adj[p][q]
}

// descend applies the best allowed swap until none lowers the cut.
func (s *graphSearch) descend(allowed func(p, q int) bool) {
	for iter := 0; iter < 100*len(s.own); iter++ {
		bp, bq, best := -1, -1, -1e-9
		for p := range s.own {
			for q := p + 1; q < len(s.own); q++ {
				if s.own[p] == s.own[q] || !allowed(p, q) {
					continue
				}
				if d := s.swapGain(p, q); d < best {
					bp, bq, best = p, q, d
				}
			}
		}
		if bp < 0 {
			return
		}
		a, b := s.own[bp], s.own[bq]
		s.move(bp, b)
		s.move(bq, a)
	}
}

func (s *graphSearch) table() Table { return Table{Owners: append([]NodeID(nil), s.own...)} }

// scaleOut: the seed (blocks) fixes how many partitions each donor gives and each new node
// takes; starts are the seed and greedy fills, each improved by swaps that keep those counts.
func (s *graphSearch) scaleOut(seed Table) Table {
	gives := map[NodeID]int{}
	takes := map[NodeID]int{}
	for p, o := range seed.Owners {
		if s.added[o] {
			gives[s.old.Owners[p]]++
			takes[o]++
		}
	}
	// A partition still with its old owner may trade places with one that left that owner (the
	// donor gives another partition instead); two partitions on new nodes may trade freely.
	allowed := func(p, q int) bool {
		ap, aq := s.added[s.own[p]], s.added[s.own[q]]
		switch {
		case ap && aq:
			return true
		case ap:
			return s.own[q] == s.old.Owners[p]
		case aq:
			return s.own[p] == s.old.Owners[q]
		}
		return false
	}
	s.load(seed)
	s.descend(allowed)
	best, bestCut := s.table(), s.adj.Cut(s.table())
	rng := rand.New(rand.NewSource(1))
	for r := 0; r < graphRestarts; r++ {
		s.load(s.old)
		left := map[NodeID]int{}
		for d, n := range gives {
			left[d] = n
		}
		for _, a := range s.nodes {
			if !s.added[a] {
				continue
			}
			for k := 0; k < takes[a]; k++ {
				// The partition loosest on its donor and closest to what a already has; the
				// first pick lands on a boundary between donors and the rest grow around it.
				bp, bd := -1, 0.0
				for p, o := range s.own {
					if s.added[o] || left[o] == 0 {
						continue
					}
					d := s.conn[p][s.col[o]] - s.conn[p][s.col[a]]
					if r > 0 {
						d += rng.Float64() * s.scale * float64(1+r%4)
					}
					if bp < 0 || d < bd {
						bp, bd = p, d
					}
				}
				left[s.own[bp]]--
				s.move(bp, a)
			}
		}
		s.descend(allowed)
		if c := s.adj.Cut(s.table()); c < bestCut-1e-9 {
			best, bestCut = s.table(), c
		}
	}
	return best
}

// scaleIn: the partitions of the leaving nodes move, to the counts the seed gives each node.
// First as many as possible go to a node that has owned them (U1): a matching, because the
// seed's greedy pass can spend a node's room on a partition that had another way home -- one
// that moved twice, 1 -> 3 -> 4, may go back to 1 and leave 3 with a partition it never had.
// The graph places the rest and chooses among former owners.
func (s *graphSearch) scaleIn(seed Table, past Past) Table {
	var moved []int
	need := map[NodeID]int{}
	for p, o := range seed.Owners {
		if s.gone[s.old.Owners[p]] {
			moved = append(moved, p)
			need[o]++
		}
	}
	owned := func(p int, n NodeID) bool { return past != nil && past.HasOwned(p, n) }
	// Candidates are compared by partitions home, then by cut.
	var best Table
	bestHome, bestCut := -1, 0.0
	consider := func() {
		home := 0
		for _, p := range moved {
			if owned(p, s.own[p]) {
				home++
			}
		}
		if c := s.adj.Cut(s.table()); home > bestHome || home == bestHome && c < bestCut-1e-9 {
			best, bestHome, bestCut = s.table(), home, c
		}
	}
	// improve swaps the owners of two moved partitions while neither leaves a home it has.
	improve := func() {
		home := make([]bool, len(s.own))
		for _, p := range moved {
			home[p] = owned(p, s.own[p])
		}
		ok := func(p int, n NodeID) bool { return !home[p] || owned(p, n) }
		s.descend(func(p, q int) bool {
			return s.gone[s.old.Owners[p]] && s.gone[s.old.Owners[q]] && ok(p, s.own[q]) && ok(q, s.own[p])
		})
	}
	s.load(seed)
	improve()
	consider()

	// The matching: augmenting paths over (partition, former owner with room).
	at := map[int]NodeID{}
	load := map[NodeID][]int{}
	var try func(p int, seen map[NodeID]bool) bool
	try = func(p int, seen map[NodeID]bool) bool {
		for _, n := range s.nodes {
			if need[n] == 0 || seen[n] || !owned(p, n) {
				continue
			}
			seen[n] = true
			if len(load[n]) < need[n] {
				at[p], load[n] = n, append(load[n], p)
				return true
			}
			for i, q := range load[n] {
				if try(q, seen) {
					load[n][i], at[p] = p, n
					return true
				}
			}
		}
		return false
	}
	for _, p := range moved {
		try(p, map[NodeID]bool{})
	}
	s.load(s.old)
	var rest []int
	for _, p := range moved {
		if n, ok := at[p]; ok {
			s.move(p, n)
			need[n]--
		} else {
			rest = append(rest, p)
		}
	}
	// The rest: each to the node it is most connected to, while that node takes more.
	for len(rest) > 0 {
		bi, bn, bw := -1, NodeID(0), 0.0
		for i, p := range rest {
			for _, n := range s.nodes {
				if need[n] == 0 {
					continue
				}
				if w := s.conn[p][s.col[n]]; bi < 0 || w > bw {
					bi, bn, bw = i, n, w
				}
			}
		}
		s.move(rest[bi], bn)
		need[bn]--
		rest = append(rest[:bi], rest[bi+1:]...)
	}
	improve()
	consider()
	return best
}
