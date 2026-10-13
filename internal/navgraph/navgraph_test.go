package navgraph

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

// writeHNSW writes an hnswlib file the way saveIndex does: header, level-0 records (link count,
// maxM0 slots, a dim-byte vector, the label), then an empty upper-level list per element.
func writeHNSW(t *testing.T, labels []uint64, links [][]uint32, maxM0, dim int) string {
	t.Helper()
	slots := 4 + 4*maxM0
	h := header{
		MaxElements: uint64(len(labels)), Count: uint64(len(labels)),
		SizePerElement: uint64(slots + dim + 8), LabelOffset: uint64(slots + dim), OffsetData: uint64(slots),
		MaxM: uint64(maxM0 / 2), MaxM0: uint64(maxM0), M: uint64(maxM0 / 2), Mult: 1, EfConstruction: 200,
	}
	var b bytes.Buffer
	if err := binary.Write(&b, binary.LittleEndian, h); err != nil {
		t.Fatal(err)
	}
	if b.Len() != headerSize {
		t.Fatalf("header is %d bytes, want %d", b.Len(), headerSize)
	}
	for i, l := range labels {
		rec := make([]byte, h.SizePerElement)
		binary.LittleEndian.PutUint32(rec, uint32(len(links[i])))
		for j, v := range links[i] {
			binary.LittleEndian.PutUint32(rec[4+4*j:], v)
		}
		for j := 0; j < dim; j++ {
			rec[slots+j] = 0xAB // the vector: must not be read as links or labels
		}
		binary.LittleEndian.PutUint64(rec[h.LabelOffset:], l)
		b.Write(rec)
	}
	for range labels {
		b.Write(make([]byte, 4)) // linkListSize 0: no upper levels
	}
	path := filepath.Join(t.TempDir(), File)
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestEdgesByLabel(t *testing.T) {
	// Internal IDs 0..3 carry labels 12, 10, 13, 11 (a parallel build's insertion order).
	labels := []uint64{12, 10, 13, 11}
	links := [][]uint32{{1, 2}, {0}, {0, 1, 3}, {}}
	path := writeHNSW(t, labels, links, 4, 8)
	var got [][2]uint32
	n, err := Edges(path, func(a, b uint32) { got = append(got, [2]uint32{a, b}) })
	if err != nil {
		t.Fatal(err)
	}
	want := [][2]uint32{{12, 10}, {12, 13}, {10, 12}, {13, 12}, {13, 10}, {13, 11}}
	sort.Slice(got, func(i, j int) bool { return got[i][0] < got[j][0] || got[i][0] == got[j][0] && got[i][1] < got[j][1] })
	sort.Slice(want, func(i, j int) bool {
		return want[i][0] < want[j][0] || want[i][0] == want[j][0] && want[i][1] < want[j][1]
	})
	if n != 4 || !reflect.DeepEqual(got, want) {
		t.Fatalf("got %d elements, edges %v; want 4, %v", n, got, want)
	}
}

// Several chunks: records straddle no chunk boundary and every element is read once.
func TestEdgesManyElements(t *testing.T) {
	const count, dim = 3000, 4096 // ~12 MB: three 4 MB chunks
	labels := make([]uint64, count)
	links := make([][]uint32, count)
	for i := range labels {
		labels[i] = uint64(count - 1 - i)
		links[i] = []uint32{uint32((i + 1) % count)}
	}
	path := writeHNSW(t, labels, links, 2, dim)
	seen := map[uint32]uint32{}
	if _, err := Edges(path, func(a, b uint32) { seen[a] = b }); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < count; i++ {
		from, to := uint32(count-1-i), uint32(count-1-(i+1)%count)
		if seen[from] != to {
			t.Fatalf("element with label %d: edge to %d, want %d", from, seen[from], to)
		}
	}
	if len(seen) != count {
		t.Fatalf("%d elements had edges, want %d", len(seen), count)
	}
}

func TestEdgesRejectsDamage(t *testing.T) {
	path := writeHNSW(t, []uint64{0, 1}, [][]uint32{{1}, {0}}, 4, 8)
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	cut := filepath.Join(t.TempDir(), "short.hnsw")
	if err := os.WriteFile(cut, b[:headerSize+10], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Edges(cut, func(uint32, uint32) {}); err == nil {
		t.Fatal("a truncated file was accepted")
	}
	bad := append([]byte(nil), b...)
	binary.LittleEndian.PutUint32(bad[headerSize+4:], 7) // element 0 links to element 7 of 2
	odd := filepath.Join(t.TempDir(), "bad.hnsw")
	if err := os.WriteFile(odd, bad, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Edges(odd, func(uint32, uint32) {}); err == nil {
		t.Fatal("a link beyond the elements was accepted")
	}
}
