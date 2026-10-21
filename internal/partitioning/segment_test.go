package partitioning

import (
	"path/filepath"
	"testing"
)

// A vector named by several lists, or by lists in several partitions, counts once; a partition
// that is not given does not count.
func TestDistinctIDs(t *testing.T) {
	dir := t.TempDir()
	for p, lists := range [][][]uint32{
		{{1, 2, 3}, {3, 4}},  // partition 0: 1-4, 3 twice
		{{4, 5}, {70, 1}},    // partition 1: 4 and 1 are in partition 0 too
		{{900, 901}, {1000}}, // partition 2
	} {
		s := &Segment{Partition: uint32(p), Offsets: []uint64{0}}
		for i, ids := range lists {
			s.Lists = append(s.Lists, uint32(10*p+i))
			s.IDs = append(s.IDs, ids...)
			s.Locs = append(s.Locs, ids...)
			s.Offsets = append(s.Offsets, uint64(len(s.IDs)))
		}
		if err := s.Write(filepath.Join(dir, SegmentName(p))); err != nil {
			t.Fatal(err)
		}
	}
	for _, c := range []struct {
		parts []int
		want  int
	}{{[]int{0}, 4}, {[]int{0, 1}, 6}, {[]int{1, 2}, 7}, {[]int{0, 1, 2}, 9}, {nil, 0}} {
		if got, err := DistinctIDs(dir, c.parts); err != nil || got != c.want {
			t.Errorf("DistinctIDs(%v) = %d, %v; want %d", c.parts, got, err, c.want)
		}
	}
	if _, err := DistinctIDs(dir, []int{3}); err == nil {
		t.Error("a partition without a segment counted")
	}
}
