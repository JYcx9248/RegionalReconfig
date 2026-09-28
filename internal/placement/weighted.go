package placement

import "rtier/internal/design"

// Weights would describe what a partition costs on each tier and how hot it is. Placeholder
// for open question U3; nothing fills it yet.
type Weights struct {
	HBMBytes  []uint64  `json:"hbm_bytes"`  // PQ codes per partition (depends on U1)
	DRAMBytes []uint64  `json:"dram_bytes"` // posting lists per partition
	SSDBytes  []uint64  `json:"ssd_bytes"`  // raw-vector pages per partition (depends on U1)
	Heat      []float64 `json:"heat"`       // recent access rate per partition
}

// WeightedPolicy is the placeholder for placement by per-tier bytes and access heat, including
// rebalancing without changing the node count (Koala rejects that case). TODO(design) U3.
type WeightedPolicy struct {
	W *Weights
}

func (WeightedPolicy) Name() string { return "weighted" }

func (WeightedPolicy) Initial(int, []NodeID) (Table, error) {
	return Table{}, design.Undecided("U3", "weighted initial placement")
}

func (WeightedPolicy) Repartition(Table, []NodeID, Past) (Table, Changes, error) {
	return Table{}, nil, design.Undecided("U3", "weighted repartitioning")
}
