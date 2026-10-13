// Package navgraph reads the edges of the navigation graph, the hnswlib index the engine builds
// over the list heads (graph.hnsw in an index directory, engine/include/fusion/index_meta.h).
//
// The file is hnswlib's saveIndex output (engine/third_party/hnswlib/hnswalg.h): a 96-byte
// header, one fixed-size record per element -- the level-0 link count (the low 16 bits of a
// uint32), maxM0 link slots of internal IDs, the vector, the label -- and then the upper levels,
// which nothing here needs. Labels are the heads' list IDs; internal IDs are insertion order,
// which a parallel build does not keep equal to the labels.
package navgraph

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// File is the navigation graph's name in an index directory.
const File = "graph.hnsw"

const headerSize = 96

type header struct {
	OffsetLevel0   uint64
	MaxElements    uint64
	Count          uint64
	SizePerElement uint64
	LabelOffset    uint64
	OffsetData     uint64
	MaxLevel       int32
	EnterPoint     uint32
	MaxM           uint64
	MaxM0          uint64
	M              uint64
	Mult           float64
	EfConstruction uint64
}

// chunk is how many bytes of element records one read takes.
const chunk = 4 << 20

// Edges calls fn(from, to) for every level-0 edge of the graph at path, by label. An edge is
// reported the way hnswlib stores it, from the element that lists it, so two elements that list
// each other give two calls. It returns the number of elements.
func Edges(path string, fn func(from, to uint32)) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	var h header
	if err := binary.Read(io.NewSectionReader(f, 0, headerSize), binary.LittleEndian, &h); err != nil {
		return 0, fmt.Errorf("navgraph: %s: header: %w", path, err)
	}
	st, err := f.Stat()
	if err != nil {
		return 0, err
	}
	links := 4 + 4*h.MaxM0
	switch {
	case h.OffsetLevel0 != 0 || h.MaxM0 == 0 || h.MaxM0 > 1<<16:
		return 0, fmt.Errorf("navgraph: %s: not an hnswlib index (offset %d, maxM0 %d)", path, h.OffsetLevel0, h.MaxM0)
	case h.SizePerElement < links+8 || h.LabelOffset < links || h.LabelOffset+8 > h.SizePerElement:
		return 0, fmt.Errorf("navgraph: %s: bad element layout (size %d, label at %d, %d link slots)",
			path, h.SizePerElement, h.LabelOffset, h.MaxM0)
	case h.Count > math.MaxUint32 || uint64(st.Size()) < headerSize+h.Count*h.SizePerElement:
		return 0, fmt.Errorf("navgraph: %s: %d elements of %d bytes do not fit in %d bytes",
			path, h.Count, h.SizePerElement, st.Size())
	}
	per := max(1, chunk/int(h.SizePerElement))
	buf := make([]byte, per*int(h.SizePerElement))
	// scan reads the element records in order, a chunk at a time.
	scan := func(each func(i uint64, rec []byte) error) error {
		for i := uint64(0); i < h.Count; i += uint64(per) {
			n := min(uint64(per), h.Count-i)
			b := buf[:n*h.SizePerElement]
			if _, err := f.ReadAt(b, int64(headerSize+i*h.SizePerElement)); err != nil {
				return fmt.Errorf("navgraph: %s: %w", path, err)
			}
			for j := uint64(0); j < n; j++ {
				if err := each(i+j, b[j*h.SizePerElement:(j+1)*h.SizePerElement]); err != nil {
					return err
				}
			}
		}
		return nil
	}
	// Labels first: an edge names its target by internal ID.
	labels := make([]uint32, h.Count)
	if err := scan(func(i uint64, rec []byte) error {
		l := binary.LittleEndian.Uint64(rec[h.LabelOffset:])
		if l > math.MaxUint32 {
			return fmt.Errorf("navgraph: %s: element %d has label %d", path, i, l)
		}
		labels[i] = uint32(l)
		return nil
	}); err != nil {
		return 0, err
	}
	err = scan(func(i uint64, rec []byte) error {
		n := uint64(binary.LittleEndian.Uint16(rec))
		if n > h.MaxM0 {
			return fmt.Errorf("navgraph: %s: element %d lists %d neighbors, at most %d fit", path, i, n, h.MaxM0)
		}
		for j := uint64(0); j < n; j++ {
			t := uint64(binary.LittleEndian.Uint32(rec[4+4*j:]))
			if t >= h.Count {
				return fmt.Errorf("navgraph: %s: element %d links to %d of %d", path, i, t, h.Count)
			}
			fn(labels[i], labels[t])
		}
		return nil
	})
	return int(h.Count), err
}
