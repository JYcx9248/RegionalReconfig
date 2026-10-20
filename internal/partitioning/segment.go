package partitioning

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"os"
)

// Segment is one partition's posting lists, as engine/include/fusion/partition.h writes them
// (version 3: every posting with its vector's location and the node holding the vector, 0 for
// none). A full rebuild (U14) reads the next generation's segments to place its partitions and
// writes them back with holders, so that each new owner fetches a vector it lacks in one hop.
type Segment struct {
	Partition uint32
	Lists     []uint32 // ascending
	Offsets   []uint64 // CSR offsets into IDs, len(Lists)+1
	IDs       []uint32
	Locs      []uint32
	Holders   []uint16 // parallel to IDs; nil: none recorded
}

const (
	segMagic   = 0x47535452 // "RTSG" little endian
	segVersion = 3
)

// ReadSegment loads a segment file (version 2 or 3).
func ReadSegment(path string) (*Segment, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	le := binary.LittleEndian
	if len(b) < 16+8+4 {
		return nil, fmt.Errorf("segment %s is truncated", path)
	}
	if crc32.ChecksumIEEE(b[:len(b)-4]) != le.Uint32(b[len(b)-4:]) {
		return nil, fmt.Errorf("segment %s: checksum mismatch", path)
	}
	if le.Uint32(b) != segMagic {
		return nil, fmt.Errorf("%s is not a segment file", path)
	}
	version := le.Uint32(b[4:])
	if version != 2 && version != segVersion {
		return nil, fmt.Errorf("segment %s has version %d, expected 2 or %d", path, version, segVersion)
	}
	s := &Segment{Partition: le.Uint32(b[8:])}
	nl := int(le.Uint32(b[12:]))
	body := b[16 : len(b)-4]
	if len(body) < 4*nl+8*(nl+1) {
		return nil, fmt.Errorf("segment %s is truncated", path)
	}
	s.Lists = make([]uint32, nl)
	for i := range s.Lists {
		s.Lists[i] = le.Uint32(body[4*i:])
	}
	body = body[4*nl:]
	s.Offsets = make([]uint64, nl+1)
	for i := range s.Offsets {
		s.Offsets[i] = le.Uint64(body[8*i:])
	}
	body = body[8*(nl+1):]
	n := int(s.Offsets[nl])
	per := 8
	if version == segVersion {
		per = 10
	}
	if s.Offsets[0] != 0 || len(body) != per*n {
		return nil, fmt.Errorf("segment %s: %d postings in %d bytes", path, n, len(body))
	}
	for i := 0; i < nl; i++ {
		if s.Offsets[i] > s.Offsets[i+1] {
			return nil, fmt.Errorf("segment %s: offsets not monotonic", path)
		}
	}
	s.IDs, s.Locs = make([]uint32, n), make([]uint32, n)
	for i := 0; i < n; i++ {
		s.IDs[i] = le.Uint32(body[4*i:])
		s.Locs[i] = le.Uint32(body[4*n+4*i:])
	}
	if version == segVersion {
		s.Holders = make([]uint16, n)
		for i := 0; i < n; i++ {
			s.Holders[i] = le.Uint16(body[8*n+2*i:])
		}
	}
	return s, nil
}

// Write saves the segment (version 3) under a temporary name and renames it, so that a reader
// never sees half of it.
func (s *Segment) Write(path string) error {
	n := len(s.IDs)
	if len(s.Offsets) != len(s.Lists)+1 || len(s.Locs) != n || (s.Holders != nil && len(s.Holders) != n) ||
		s.Offsets[len(s.Lists)] != uint64(n) {
		return fmt.Errorf("segment of partition %d: %d lists, %d offsets, %d IDs, %d locations, %d holders",
			s.Partition, len(s.Lists), len(s.Offsets), n, len(s.Locs), len(s.Holders))
	}
	le := binary.LittleEndian
	b := make([]byte, 0, 16+4*len(s.Lists)+8*len(s.Offsets)+10*n+4)
	b = le.AppendUint32(b, segMagic)
	b = le.AppendUint32(b, segVersion)
	b = le.AppendUint32(b, s.Partition)
	b = le.AppendUint32(b, uint32(len(s.Lists)))
	for _, l := range s.Lists {
		b = le.AppendUint32(b, l)
	}
	for _, o := range s.Offsets {
		b = le.AppendUint64(b, o)
	}
	for _, id := range s.IDs {
		b = le.AppendUint32(b, id)
	}
	for _, loc := range s.Locs {
		b = le.AppendUint32(b, loc)
	}
	for i := 0; i < n; i++ {
		var h uint16
		if s.Holders != nil {
			h = s.Holders[i]
		}
		b = le.AppendUint16(b, h)
	}
	b = le.AppendUint32(b, crc32.ChecksumIEEE(b))
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
