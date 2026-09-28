// Partitions of the posting lists, for the regional tier (rtier).
//
// A partition is a fixed group of posting lists. Which partition a list belongs to is decided
// once, before deployment, and never changes (list -> partition). Which node serves a
// partition is the placement, owned by the Go controller (partition -> node, per epoch).
//
// rtier_segment writes a partition directory:
//   manifest.txt      key=value: num_lists, num_partitions, max_list_len, payload, and the
//                     geometry of the page file the locations refer to (layout, page_size,
//                     vec_bytes, vectors_per_page, num_pages)
//   list_part.bin     uint32 partition of every list (SaveU32Array format)
//   part-00000.seg    posting lists of partition 0, ...  (format below)
//
// Segment file (little endian):
//   u32 magic 'RTSG' | u32 version (2) | u32 partition | u32 num_lists
//   u32 list_ids[num_lists]            ascending
//   u64 offsets[num_lists + 1]         CSR offsets into ids
//   u32 ids[offsets[num_lists]]        vector IDs
//   u32 locs[offsets[num_lists]]       canonical location of each posting's raw vector
//   u32 crc32 of every byte before it
//
// Payload (the decided part of U1). A posting is (vector ID, location): the location is where
// the vector sits in the index's page file (fusion/layout.h, RawLayout), 8 B per posting with
// the ID. It travels with the list and never changes, so a node holds any subset of the page
// file at the same offsets without a vector -> page map, and the same vector reached through
// two lists names the same location. PQ codes and raw vectors are node-level and move
// separately, deduplicated per node (fusion/pq_store.h, fusion/raw_store.h); segments carry
// only the lists.
#pragma once

#include <string>
#include <vector>

#include "fusion/common.h"
#include "fusion/index_meta.h"
#include "fusion/layout.h"

namespace fusion {

namespace partfiles {
constexpr const char* kManifest = "manifest.txt";
constexpr const char* kListPart = "list_part.bin";
std::string SegmentName(uint32_t partition);  // "part-00042.seg"
}  // namespace partfiles

// Payloads. kListsLocations is the current one; kListsOnly (segment version 1, IDs without
// locations: every node had to open the whole page file) is only recognized to reject old
// partition directories with a clear message.
constexpr const char* kPayloadListsLocations = "lists+locations";
constexpr const char* kPayloadListsOnly = "lists-only";

struct PartitionManifest {
  uint32_t num_lists = 0;
  uint32_t num_partitions = 0;
  uint32_t max_list_len = 0;
  std::string payload = kPayloadListsLocations;
  // The page file the locations refer to: the index's vectors_<layout>.bin.
  std::string layout;
  RawLayout raw;
  std::vector<uint32_t> list_part;  // list -> partition

  static PartitionManifest Load(const std::string& dir);
  void Save(const std::string& dir) const;
};

// One partition's posting lists.
struct ListSegment {
  uint32_t partition = 0;
  std::vector<uint32_t> list_ids;  // ascending
  std::vector<uint64_t> offsets;   // list_ids.size() + 1
  std::vector<uint32_t> ids;
  std::vector<uint32_t> locs;      // parallel to ids

  // Vector IDs of `list`, or nullptr if the list is not in this segment.
  const uint32_t* Find(uint32_t list, uint32_t* len) const;
  uint64_t bytes() const {
    return list_ids.size() * 4 + offsets.size() * 8 + ids.size() * 4 + locs.size() * 4;
  }

  void Save(const std::string& path) const;
  static ListSegment Load(const std::string& path);
};

// Splits `lists` by `list_part` and writes the manifest, list_part.bin and one segment per
// partition into `out_dir`; every posting gets its vector's location in `layout` (the index's
// layout named `layout_name`). Returns the segments' sizes in bytes (indexed by partition).
std::vector<uint64_t> WritePartitions(const PostingLists& lists,
                                      const std::vector<uint32_t>& list_part,
                                      uint32_t num_partitions, const std::string& out_dir,
                                      const LayoutMap& layout, const std::string& layout_name);

uint32_t Crc32(const void* data, size_t bytes, uint32_t crc = 0);

}  // namespace fusion
