package partitioning

import (
	"fmt"

	"rtier/internal/design"
)

// Files returns the files that make up partition p on disk, given the manifest's payload.
//
// With PayloadListsLocations a partition's files are its posting-list segment. PQ codes and raw
// vectors are not partition files: they are node-level, and the agent pulls the ones a
// destination lacks separately (the decided part of U1). Where RAG chunks live is still open
// (U1 with U12); a payload that carries them adds its files here and in
// engine/src/partition.cpp.
func Files(m *Manifest, p int) ([]string, error) { return FilesAt(m, p, 0) }

// FilesAt is Files for version v of the partition's lists (U14).
func FilesAt(m *Manifest, p int, v uint32) ([]string, error) {
	if p < 0 || p >= m.NumPartitions {
		return nil, fmt.Errorf("partitioning: partition %d out of range", p)
	}
	switch m.Payload {
	case PayloadListsLocations:
		return []string{SegmentNameV(p, v)}, nil
	}
	return nil, design.Undecided("U1", "partition payload "+m.Payload)
}
