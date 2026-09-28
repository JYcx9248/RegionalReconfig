// Package vecio reads and writes the big-ann-benchmarks vector formats used by the engine
// (engine/include/fusion/dataset.h): .u8bin / .i8bin / .fbin (uint32 n, uint32 dim, data)
// and .ibin ground truth (uint32 nq, uint32 k, int32 ids[nq*k], optional float dists).
//
// Queries stay in the index's native element type as raw bytes: they are sent to the data
// nodes unchanged.
package vecio

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// DType is an element type name as the engine reports it: "uint8", "int8" or "float".
type DType string

// Size in bytes of one element.
func (d DType) Size() int {
	switch d {
	case "uint8", "int8":
		return 1
	case "float":
		return 4
	}
	return 0
}

// DTypeFromPath infers the dtype from a file extension.
func DTypeFromPath(path string) (DType, error) {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".u8bin":
		return "uint8", nil
	case ".i8bin":
		return "int8", nil
	case ".fbin":
		return "float", nil
	}
	return "", fmt.Errorf("vecio: cannot infer dtype of %s", path)
}

// Vectors is a dense matrix in its native element type.
type Vectors struct {
	DType DType
	N     int
	Dim   int
	Data  []byte
}

// Row returns vector i as raw bytes.
func (v *Vectors) Row(i int) []byte {
	sz := v.Dim * v.DType.Size()
	return v.Data[i*sz : (i+1)*sz]
}

// ReadBin reads a .u8bin/.i8bin/.fbin file (at most maxN vectors if maxN > 0).
func ReadBin(path string, maxN int) (*Vectors, error) {
	dt, err := DTypeFromPath(path)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var hdr [2]uint32
	if err := binary.Read(f, binary.LittleEndian, &hdr); err != nil {
		return nil, fmt.Errorf("vecio: %s: %w", path, err)
	}
	n, dim := int(hdr[0]), int(hdr[1])
	if maxN > 0 && maxN < n {
		n = maxN
	}
	v := &Vectors{DType: dt, N: n, Dim: dim, Data: make([]byte, n*dim*dt.Size())}
	if _, err := io.ReadFull(f, v.Data); err != nil {
		return nil, fmt.Errorf("vecio: %s is truncated: %w", path, err)
	}
	return v, nil
}

// WriteBin writes v in the bin format matching its dtype.
func WriteBin(path string, v *Vectors) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	hdr := [2]uint32{uint32(v.N), uint32(v.Dim)}
	if err := binary.Write(f, binary.LittleEndian, hdr); err != nil {
		f.Close()
		return err
	}
	if _, err := f.Write(v.Data); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// FromFloat32 packs float vectors into a "float" matrix.
func FromFloat32(rows [][]float32) *Vectors {
	v := &Vectors{DType: "float", N: len(rows)}
	if len(rows) > 0 {
		v.Dim = len(rows[0])
	}
	v.Data = make([]byte, 0, v.N*v.Dim*4)
	for _, r := range rows {
		for _, x := range r {
			v.Data = binary.LittleEndian.AppendUint32(v.Data, math.Float32bits(x))
		}
	}
	return v
}

// GroundTruth holds the exact neighbors of each query.
type GroundTruth struct {
	NQ, K int
	IDs   []uint32  // NQ*K
	Dists []float32 // NQ*K, or nil
}

// ReadGroundTruth reads an .ibin ground-truth file (with or without distances).
func ReadGroundTruth(path string) (*GroundTruth, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) < 8 {
		return nil, fmt.Errorf("vecio: %s is truncated", path)
	}
	nq, k := int(binary.LittleEndian.Uint32(b)), int(binary.LittleEndian.Uint32(b[4:]))
	need := 8 + nq*k*4
	if len(b) < need {
		return nil, fmt.Errorf("vecio: %s is truncated", path)
	}
	g := &GroundTruth{NQ: nq, K: k, IDs: make([]uint32, nq*k)}
	for i := range g.IDs {
		g.IDs[i] = binary.LittleEndian.Uint32(b[8+4*i:])
	}
	if len(b) >= need+nq*k*4 {
		g.Dists = make([]float32, nq*k)
		for i := range g.Dists {
			g.Dists[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[need+4*i:]))
		}
	}
	return g, nil
}

// Recall returns |result ∩ gt top-k| / k for query q. A result that ties with the k-th
// ground-truth distance also counts (as fusion::RecallAtK does), when distances are known.
func (g *GroundTruth) Recall(q int, ids []uint32, dists []float32, k int) float64 {
	k = min(k, g.K)
	if k == 0 {
		return 0
	}
	truth := make(map[uint32]bool, k)
	for _, id := range g.IDs[q*g.K : q*g.K+k] {
		truth[id] = true
	}
	var kth float32 = -1
	if g.Dists != nil {
		kth = g.Dists[q*g.K+k-1]
	}
	hit := 0
	for i := 0; i < min(k, len(ids)); i++ {
		if truth[ids[i]] || (kth >= 0 && dists != nil && dists[i] <= kth) {
			hit++
		}
	}
	return float64(hit) / float64(k)
}
