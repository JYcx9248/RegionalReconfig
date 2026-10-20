package placement

import (
	"container/heap"
	"fmt"
	"math/rand"
	"sort"
)

// Affinity places len(w) partitions on nodes, count-balanced like the even policies (every node
// gets floor(P/N) or ceil(P/N) partitions), so as to maximize the total weight of the pairs it
// picks: w[p][i] is what placing partition p on nodes[i] saves. A full rebuild (U14) weighs a
// new partition on an old node by the raw vectors the node holds already -- what it will not
// have to fetch. Exact (a min-cost flow) and deterministic.
func Affinity(w [][]int64, nodes []NodeID) (Table, error) {
	np, nn := len(w), len(nodes)
	if err := checkNodes(np, nodes); err != nil {
		return Table{}, err
	}
	var top int64
	for p, row := range w {
		if len(row) != nn {
			return Table{}, fmt.Errorf("placement: partition %d has %d weights for %d nodes", p, len(row), nn)
		}
		for _, x := range row {
			if x < 0 {
				return Table{}, fmt.Errorf("placement: negative weight %d", x)
			}
			top = max(top, x)
		}
	}
	// Source -> partition (1) -> node (1, cost top - w) -> sink (floor(P/N)); the ceil(P/N)-th
	// partition of a node goes through a hub that only P mod N may use.
	base, extra := np/nn, np%nn
	src, part0, node0 := 0, 1, 1+np
	hub, sink := 1+np+nn, 2+np+nn
	g := newFlow(sink + 1)
	for p := 0; p < np; p++ {
		g.add(src, part0+p, 1, 0)
	}
	pick := make([][]int, np) // edge index of p -> node i
	for p := 0; p < np; p++ {
		pick[p] = make([]int, nn)
		for i := 0; i < nn; i++ {
			pick[p][i] = g.add(part0+p, node0+i, 1, top-w[p][i])
		}
	}
	for i := 0; i < nn; i++ {
		if base > 0 {
			g.add(node0+i, sink, base, 0)
		}
		if extra > 0 {
			g.add(node0+i, hub, 1, 0)
		}
	}
	if extra > 0 {
		g.add(hub, sink, extra, 0)
	}
	if got := g.run(src, sink, np); got != np {
		return Table{}, fmt.Errorf("placement: placed %d of %d partitions", got, np)
	}
	owners := make([]NodeID, np)
	for p := 0; p < np; p++ {
		for i := 0; i < nn; i++ {
			if g.edges[pick[p][i]].cap == 0 {
				owners[p] = nodes[i]
			}
		}
	}
	return Table{Owners: owners}, nil
}

// RandomBalanced places numPartitions partitions on nodes at random, count-balanced as
// Affinity: the baseline a full rebuild's affinity placement is measured against.
func RandomBalanced(numPartitions int, nodes []NodeID, seed int64) (Table, error) {
	if err := checkNodes(numPartitions, nodes); err != nil {
		return Table{}, err
	}
	ns := append([]NodeID(nil), nodes...)
	sort.Slice(ns, func(i, j int) bool { return ns[i] < ns[j] })
	perm := rand.New(rand.NewSource(seed)).Perm(numPartitions)
	owners := make([]NodeID, numPartitions)
	for k, p := range perm {
		owners[p] = ns[k%len(ns)]
	}
	return Table{Owners: owners}, nil
}

// flow is a min-cost flow network: successive shortest paths, Dijkstra on reduced costs.
type flow struct {
	adj   [][]int
	edges []flowEdge
}

type flowEdge struct {
	to, cap int
	cost    int64
}

func newFlow(n int) *flow { return &flow{adj: make([][]int, n)} }

// add adds an edge and its reverse; it returns the edge's index.
func (g *flow) add(from, to, cap int, cost int64) int {
	g.adj[from] = append(g.adj[from], len(g.edges))
	g.edges = append(g.edges, flowEdge{to, cap, cost})
	g.adj[to] = append(g.adj[to], len(g.edges))
	g.edges = append(g.edges, flowEdge{from, 0, -cost})
	return len(g.edges) - 2
}

// run pushes up to want units from s to t at minimum cost (every cost is nonnegative at first,
// so zero potentials start it); it returns the units pushed.
func (g *flow) run(s, t, want int) int {
	n := len(g.adj)
	pot := make([]int64, n)
	dist := make([]int64, n)
	prev := make([]int, n)
	const inf = int64(1) << 62
	pushed := 0
	for pushed < want {
		for i := range dist {
			dist[i], prev[i] = inf, -1
		}
		dist[s] = 0
		pq := &distHeap{{s, 0}}
		for pq.Len() > 0 {
			it := heap.Pop(pq).(distItem)
			if it.d > dist[it.v] {
				continue
			}
			for _, ei := range g.adj[it.v] {
				e := g.edges[ei]
				if e.cap == 0 {
					continue
				}
				nd := it.d + e.cost + pot[it.v] - pot[e.to]
				if nd < dist[e.to] {
					dist[e.to], prev[e.to] = nd, ei
					heap.Push(pq, distItem{e.to, nd})
				}
			}
		}
		if dist[t] == inf {
			break
		}
		for i := range pot {
			if dist[i] < inf {
				pot[i] += dist[i]
			}
		}
		// Every augmenting path carries one unit: the partitions' edges have capacity 1.
		for v := t; v != s; {
			ei := prev[v]
			g.edges[ei].cap--
			g.edges[ei^1].cap++
			v = g.edges[ei^1].to
		}
		pushed++
	}
	return pushed
}

type distItem struct {
	v int
	d int64
}

type distHeap []distItem

func (h distHeap) Len() int { return len(h) }
func (h distHeap) Less(i, j int) bool {
	if h[i].d != h[j].d {
		return h[i].d < h[j].d
	}
	return h[i].v < h[j].v
}
func (h distHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *distHeap) Push(x any)   { *h = append(*h, x.(distItem)) }
func (h *distHeap) Pop() any {
	old := *h
	it := old[len(old)-1]
	*h = old[:len(old)-1]
	return it
}
