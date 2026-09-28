package partitioning

import "rtier/internal/design"

// ListPartitioner groups posting lists into partitions (list -> partition), once, before
// deployment. The output feeds engine/tools/rtier_segment --assign.
//
// TODO(design) U2: nothing is decided yet. Candidates discussed: locality in the navigation
// graph, k-means over the list centroids, hashing. The choice should come from measuring
// r(P) (replication across partitions), bytes per partition and query fan-out on the real
// datasets. Tests use a contiguous-range fake (test/e2e, scripts/testing).
type ListPartitioner interface {
	Name() string
	// Assign returns the partition of each of numLists lists.
	Assign(numLists, numPartitions int) ([]uint32, error)
}

// Undecided is the only production partitioner.
type Undecided struct{}

func (Undecided) Name() string { return "undecided" }

func (Undecided) Assign(int, int) ([]uint32, error) {
	return nil, design.Undecided("U2", "list -> partition assignment")
}
