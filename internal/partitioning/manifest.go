// Package partitioning reads the partition directory written by engine/tools/rtier_segment
// (manifest.txt, list_part.bin, part-NNNNN.seg) and holds the placeholders for how lists are
// grouped into partitions (U2) and what a partition carries (U1).
package partitioning

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	ManifestFile = "manifest.txt"
	ListPartFile = "list_part.bin"
	// PayloadListsLocations: a partition's file is its posting-list segment, and every posting
	// carries its vector's canonical location in the index's page file next to the ID (the
	// decided part of U1). PQ codes and raw vectors are node-level: they move separately,
	// deduplicated per node.
	PayloadListsLocations = "lists+locations"
	// PayloadListsOnly is the old format (IDs without locations, every data node opened the
	// whole page file); the engine refuses it.
	PayloadListsOnly = "lists-only"
)

// SegmentName is the file holding partition p's posting lists as built (version 0).
func SegmentName(p int) string { return fmt.Sprintf("part-%05d.seg", p) }

// SegmentNameV is the file holding version v of partition p's posting lists: the build's
// segment for version 0; for a later one, made by update batches (U14), the segment its owner
// writes when the partition moves.
func SegmentNameV(p int, v uint32) string {
	if v == 0 {
		return SegmentName(p)
	}
	return fmt.Sprintf("part-%05d.v%d.seg", p, v)
}

// SegmentVersionsGlob matches the files of partition p's later versions (SegmentNameV, v > 0).
func SegmentVersionsGlob(p int) string { return fmt.Sprintf("part-%05d.v*.seg", p) }

// Manifest describes a partition directory.
type Manifest struct {
	NumLists      int
	NumPartitions int
	MaxListLen    int
	Payload       string
	ListPart      []uint32 // list -> partition
	// The page file the postings' locations refer to (the index's vectors_<layout>.bin).
	NumPages       uint64
	VectorsPerPage uint64
}

// Locations is the size of the index's raw-vector address space: where the locations of
// inserted vectors (U14) begin.
func (m *Manifest) Locations() uint64 { return m.NumPages * m.VectorsPerPage }

// PartitionOf returns the partition of list c.
func (m *Manifest) PartitionOf(c uint32) int { return int(m.ListPart[c]) }

// LoadManifest reads manifest.txt and list_part.bin from dir.
func LoadManifest(dir string) (*Manifest, error) {
	kv, err := readKV(filepath.Join(dir, ManifestFile))
	if err != nil {
		return nil, err
	}
	m := &Manifest{Payload: kv["payload"]}
	if m.Payload == "" {
		m.Payload = PayloadListsOnly // same default as the engine (partition.cpp)
	}
	if m.Payload == PayloadListsOnly {
		return nil, fmt.Errorf("partitioning: %s was written without raw-vector locations (payload %s); "+
			"rerun rtier_segment", dir, m.Payload)
	}
	for key, dst := range map[string]*int{
		"num_lists": &m.NumLists, "num_partitions": &m.NumPartitions, "max_list_len": &m.MaxListLen,
	} {
		v, err := strconv.Atoi(kv[key])
		if err != nil {
			return nil, fmt.Errorf("partitioning: manifest %s: bad %s", dir, key)
		}
		*dst = v
	}
	for key, dst := range map[string]*uint64{"num_pages": &m.NumPages, "vectors_per_page": &m.VectorsPerPage} {
		if v, ok := kv[key]; ok {
			if *dst, err = strconv.ParseUint(v, 10, 64); err != nil {
				return nil, fmt.Errorf("partitioning: manifest %s: bad %s", dir, key)
			}
		}
	}
	if m.ListPart, err = ReadU32Array(filepath.Join(dir, ListPartFile)); err != nil {
		return nil, err
	}
	if len(m.ListPart) != m.NumLists {
		return nil, fmt.Errorf("partitioning: %s has %d entries, manifest says %d lists",
			ListPartFile, len(m.ListPart), m.NumLists)
	}
	for c, p := range m.ListPart {
		if int(p) >= m.NumPartitions {
			return nil, fmt.Errorf("partitioning: list %d in partition %d >= %d", c, p, m.NumPartitions)
		}
	}
	return m, nil
}

func readKV(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	kv := map[string]string{}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if k, v, ok := strings.Cut(line, "="); ok {
			kv[strings.TrimSpace(k)] = strings.TrimSpace(v)
		}
	}
	return kv, sc.Err()
}

// ReadU32Array reads the engine's uint32 array format (uint64 count, then values).
func ReadU32Array(path string) ([]uint32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) < 8 {
		return nil, fmt.Errorf("partitioning: %s is truncated", path)
	}
	n := binary.LittleEndian.Uint64(b)
	if uint64(len(b)-8) != 4*n {
		return nil, fmt.Errorf("partitioning: %s has %d bytes for %d values", path, len(b)-8, n)
	}
	v := make([]uint32, n)
	for i := range v {
		v[i] = binary.LittleEndian.Uint32(b[8+4*i:])
	}
	return v, nil
}

// WriteU32Array writes the engine's uint32 array format (the input of rtier_segment --assign).
func WriteU32Array(path string, v []uint32) error {
	b := make([]byte, 8, 8+4*len(v))
	binary.LittleEndian.PutUint64(b, uint64(len(v)))
	for _, x := range v {
		b = binary.LittleEndian.AppendUint32(b, x)
	}
	return os.WriteFile(path, b, 0o644)
}
