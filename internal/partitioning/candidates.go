package partitioning

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
)

// Candidates is what one request may gather from the posting lists of a partition directory.
type Candidates struct {
	MaxNprobe int `json:"max_nprobe"`
	// Max: the postings of the MaxNprobe longest lists. A request names at most max_nprobe lists
	// (its query's nearest heads), each once, so it gathers no more: rtier_node --max-candidates,
	// the filter's candidate buffers for lists that never change.
	Max     uint64 `json:"max_candidates"`
	Longest uint64 `json:"longest"` // the longest list's postings
	Lists   int    `json:"lists"`
	// Product: min(MaxNprobe, Lists) x Longest, the buffers rtier_node reserves without
	// --max-candidates, room for every list to grow to the longest's length.
	Product uint64 `json:"product"`
}

// MaxCandidates sums the maxNprobe longest posting lists of the partitions in dir. It reads the
// segments' list tables only, not their postings: a few seconds where the partitions hold tens of
// gigabytes.
func MaxCandidates(dir string, maxNprobe int) (Candidates, error) {
	if maxNprobe < 1 {
		return Candidates{}, fmt.Errorf("partitioning: max_nprobe %d, want at least 1", maxNprobe)
	}
	m, err := LoadManifest(dir)
	if err != nil {
		return Candidates{}, err
	}
	lens := make([]uint64, 0, m.NumLists)
	for p := 0; p < m.NumPartitions; p++ {
		if lens, err = appendListLengths(lens, filepath.Join(dir, SegmentName(p))); err != nil {
			return Candidates{}, err
		}
	}
	slices.Sort(lens)
	c := Candidates{MaxNprobe: maxNprobe, Lists: m.NumLists}
	for _, n := range lens[max(len(lens)-maxNprobe, 0):] {
		c.Max += n
	}
	if len(lens) > 0 {
		c.Longest = lens[len(lens)-1]
	}
	c.Product = uint64(min(maxNprobe, m.NumLists)) * c.Longest
	return c, nil
}

// appendListLengths appends the postings of each list of the segment at path (version 2 or 3),
// from its CSR offsets; the file's size must be what they say.
func appendListLengths(lens []uint64, path string) ([]uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	le := binary.LittleEndian
	var h [16]byte
	if _, err := io.ReadFull(f, h[:]); err != nil {
		return nil, fmt.Errorf("segment %s: %w", path, err)
	}
	if le.Uint32(h[:]) != segMagic {
		return nil, fmt.Errorf("%s is not a segment file", path)
	}
	version := le.Uint32(h[4:])
	if version != 2 && version != segVersion {
		return nil, fmt.Errorf("segment %s has version %d, expected 2 or %d", path, version, segVersion)
	}
	nl := int64(le.Uint32(h[12:]))
	if _, err := f.Seek(16+4*nl, io.SeekStart); err != nil {
		return nil, err
	}
	b := make([]byte, 8*(nl+1))
	if _, err := io.ReadFull(f, b); err != nil {
		return nil, fmt.Errorf("segment %s: %w", path, err)
	}
	if le.Uint64(b) != 0 {
		return nil, fmt.Errorf("segment %s: its offsets start at %d", path, le.Uint64(b))
	}
	prev := uint64(0)
	for i := int64(1); i <= nl; i++ {
		next := le.Uint64(b[8*i:])
		if next < prev {
			return nil, fmt.Errorf("segment %s: offsets not monotonic", path)
		}
		lens = append(lens, next-prev)
		prev = next
	}
	per := int64(8) // per posting: ID and location, and in version 3 the holder
	if version == segVersion {
		per = 10
	}
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if want := 16 + 4*nl + 8*(nl+1) + per*int64(prev) + 4; st.Size() != want {
		return nil, fmt.Errorf("segment %s: %d bytes, %d by its offsets", path, st.Size(), want)
	}
	return lens, nil
}
