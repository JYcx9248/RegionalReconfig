// SSD tier: raw vectors packed into 4 KB pages (paper Section 3.4, "Optimized Storage
// Layout"), and the in-memory table that maps every vector to its (page, slot).
//
// Bucket layout: every vector goes to exactly one bucket, the one of its nearest head, so
// highly similar vectors share pages. A bucket's full pages are written as-is; the partial
// tails of all buckets are then bin-packed into shared pages to minimize free space.
//
// The paper packs the tails with a "max-min algorithm" but cites a paper on Elias-Fano index
// compression for it, so the exact algorithm is unspecified. We use largest-first best-fit:
// each new page takes the largest remaining tail, then repeatedly the largest tail that still
// fits. Tails are never split, so a bucket's tail stays on one page.
#pragma once

#include <vector>

#include "fusion/dataset.h"

namespace fusion {

// The geometry of a page file, and a vector's place in it as one number, its location:
//   loc = page * vectors_per_page + slot, the vector's bytes at page * page_size + slot * vec_bytes.
// rtier stores the location next to every posting (fusion/partition.h), so a node can hold any
// subset of the page file without the vector -> page map (the decided part of design question
// U1: canonical global offsets).
struct RawLayout {
  uint32_t page_size = kPageSize;
  uint32_t vec_bytes = 0;
  uint32_t vectors_per_page = 0;
  uint64_t num_pages = 0;

  uint64_t locations() const { return num_pages * vectors_per_page; }
  uint32_t page(uint32_t loc) const { return loc / vectors_per_page; }
  uint32_t slot(uint32_t loc) const { return loc % vectors_per_page; }
  uint64_t offset(uint32_t loc) const {
    return static_cast<uint64_t>(page(loc)) * page_size + static_cast<uint64_t>(slot(loc)) * vec_bytes;
  }
};

struct LayoutMap {
  uint32_t page_size = kPageSize;
  uint32_t vec_bytes = 0;
  uint32_t vectors_per_page = 0;
  uint64_t num_pages = 0;
  std::vector<uint32_t> page_of;  // vector -> page
  std::vector<uint16_t> slot_of;  // vector -> slot within the page

  uint64_t size() const { return page_of.size(); }
  uint64_t Offset(uint32_t id) const {
    return static_cast<uint64_t>(page_of[id]) * page_size;
  }
  RawLayout raw() const { return {page_size, vec_bytes, vectors_per_page, num_pages}; }
  // Location of vector id (see RawLayout); fits in 32 bits when raw().locations() does.
  uint64_t Loc(uint32_t id) const {
    return static_cast<uint64_t>(page_of[id]) * vectors_per_page + slot_of[id];
  }

  void Save(const std::string& path) const;
  static LayoutMap Load(const std::string& path);
};

struct LayoutStats {
  uint64_t full_pages = 0;
  uint64_t packed_pages = 0;
  uint64_t tails = 0;
  double fill_ratio = 0;  // used slots / total slots
};

// Groups vectors by `bucket_of` (their nearest head) and packs them into pages.
LayoutMap BuildBucketLayout(const std::vector<uint32_t>& bucket_of, uint32_t num_buckets,
                            uint32_t vec_bytes, uint32_t page_size, LayoutStats* stats);

// Baseline for ablations: vectors stored in ID order (page = id / vectors_per_page).
LayoutMap BuildSequentialLayout(uint64_t n, uint32_t vec_bytes, uint32_t page_size,
                                LayoutStats* stats);

// Writes the page file for `map`, reading vectors from `data`.
void WriteVectorPages(const VectorFile& data, const LayoutMap& map, const std::string& path);

}  // namespace fusion
