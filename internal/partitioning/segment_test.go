package partitioning

import (
	"os"
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

// The bound is the sum of the longest lists, across partitions; a truncated segment is refused.
func TestMaxCandidates(t *testing.T) {
	dir := t.TempDir()
	var listPart []uint32
	for p, lens := range [][]int{{5, 1, 9}, {2, 7}, {0, 3}} {
		s := &Segment{Partition: uint32(p), Offsets: []uint64{0}}
		for _, n := range lens {
			s.Lists = append(s.Lists, uint32(len(listPart)))
			listPart = append(listPart, uint32(p))
			for i := 0; i < n; i++ {
				s.IDs = append(s.IDs, uint32(100*len(listPart)+i))
				s.Locs = append(s.Locs, uint32(i))
			}
			s.Offsets = append(s.Offsets, uint64(len(s.IDs)))
		}
		if err := s.Write(filepath.Join(dir, SegmentName(p))); err != nil {
			t.Fatal(err)
		}
	}
	manifest := "num_lists=7\nnum_partitions=3\nmax_list_len=9\npayload=" + PayloadListsLocations + "\n"
	if err := os.WriteFile(filepath.Join(dir, ManifestFile), []byte(manifest), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := WriteU32Array(filepath.Join(dir, ListPartFile), listPart); err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct{ nprobe, want int }{{1, 9}, {2, 16}, {3, 21}, {7, 27}, {64, 27}} {
		got, err := MaxCandidates(dir, c.nprobe)
		if err != nil || got.Max != uint64(c.want) || got.Longest != 9 || got.Lists != 7 ||
			got.Product != uint64(min(c.nprobe, 7)*9) {
			t.Errorf("MaxCandidates(%d) = %+v, %v; want %d", c.nprobe, got, err, c.want)
		}
	}
	seg := filepath.Join(dir, SegmentName(1))
	b, err := os.ReadFile(seg)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(seg, b[:len(b)-10], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := MaxCandidates(dir, 2); err == nil {
		t.Error("a truncated segment was read")
	}
}
