package partitioning

import (
	"fmt"
	"path/filepath"

	"rtier/internal/navgraph"
)

// PartitionGraph says how strongly every two partitions touch: g[p][q] (= g[q][p]) counts the
// navigation-graph edges between a head of p and a head of q, an edge that both heads list
// counting twice; the diagonal stays zero. Heads that are neighbors in the navigation graph get
// probed by the same queries, so a placement whose nodes cut little of this graph sends few
// queries to more than one owner (placement.GraphPolicy, U3). It depends only on the index and
// the assignment, never on queries.
func PartitionGraph(indexDir string, m *Manifest) ([][]float64, error) {
	g := make([][]float64, m.NumPartitions)
	for p := range g {
		g[p] = make([]float64, m.NumPartitions)
	}
	var bad error
	n, err := navgraph.Edges(filepath.Join(indexDir, navgraph.File), func(a, b uint32) {
		if int(a) >= m.NumLists || int(b) >= m.NumLists {
			if bad == nil {
				bad = fmt.Errorf("partitioning: navigation-graph edge %d -> %d names a list beyond the %d lists", a, b, m.NumLists)
			}
			return
		}
		if p, q := m.PartitionOf(a), m.PartitionOf(b); p != q {
			g[p][q]++
			g[q][p]++
		}
	})
	if err != nil {
		return nil, err
	}
	if bad != nil {
		return nil, bad
	}
	if n != m.NumLists {
		return nil, fmt.Errorf("partitioning: the navigation graph has %d heads, the partitions %d lists", n, m.NumLists)
	}
	return g, nil
}
