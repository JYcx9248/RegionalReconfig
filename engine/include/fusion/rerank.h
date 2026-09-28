// Heuristic re-ranking (paper Section 3.3, Algorithm 1) with redundancy-aware I/O
// deduplication (Section 3.4).
//
// The top-n candidates from the PQ filter are re-ranked with exact distances in mini-batches
// of `batch` vectors (in ascending PQ distance). After each mini-batch the change rate of the
// top-k max-heap is
//     delta = |S_n - (S_n ∩ S_{n-1})| / k
// and re-ranking stops once delta <= eps for beta consecutive mini-batches.
//
// Reading a mini-batch's raw vectors:
//   * intra-mini-batch dedup: vectors on the same SSD page cost one read;
//   * inter-mini-batch dedup: pages read by earlier mini-batches of the same query are kept in
//     a per-query DRAM buffer and not read again.
#pragma once

#include <vector>

#include "fusion/common.h"
#include "fusion/layout.h"
#include "fusion/page_reader.h"

namespace fusion {

struct RerankParams {
  uint32_t k = 10;
  uint32_t batch = 0;      // mini-batch size; 0 = k (the paper's setting)
  bool heuristic = true;   // Algorithm 1 early termination; false = re-rank all n candidates
  float eps = 0.1f;        // change-rate threshold (paper: 0.1)
  uint32_t beta = 1;       // stable mini-batches required (paper: 1)
  bool io_dedup = true;    // false = one read per vector, no reuse (ablation baseline)
};

struct RerankStats {
  uint32_t batches = 0;
  uint32_t reranked = 0;      // vectors whose exact distance was computed
  uint32_t pages_read = 0;    // SSD reads issued
  uint32_t intra_merged = 0;  // reads saved: same page within a mini-batch
  uint32_t buffer_hits = 0;   // reads saved: page already buffered by an earlier mini-batch
  uint32_t fetched = 0;       // rtier: raw vectors fetched from a peer first (warm-up, U9)
  double io_us = 0;
  double compute_us = 0;
  double fetch_us = 0;
};

// Per-worker scratch: page buffers and the per-query page table (the DRAM buffer).
class RerankScratch {
 public:
  RerankScratch(uint32_t max_candidates, uint32_t page_size);

  void NewQuery();
  // Returns the buffer index holding `page` if present in this query's buffer.
  bool Find(uint32_t page, uint32_t* buf) const;
  // Allocates a buffer for `page` (and records it in the table when `track`).
  uint32_t Allocate(uint32_t page, uint32_t batch_no, bool track);
  uint8_t* Buffer(uint32_t i) { return pages_.get() + static_cast<size_t>(i) * page_size_; }
  uint32_t loaded_in_batch(uint32_t buf) const { return buf_batch_[buf]; }

  // Working vectors reused across queries.
  std::vector<uint32_t> req_pages;
  std::vector<uint8_t*> req_bufs;
  std::vector<uint32_t> cand_buf;
  std::vector<std::pair<float, uint32_t>> heap;
  std::vector<uint32_t> prev;

 private:
  uint32_t capacity_;
  uint32_t page_size_;
  AlignedBuffer pages_;
  std::vector<uint32_t> buf_batch_;
  uint32_t next_buf_ = 0;
  // Open-addressing table page -> buffer, reset in O(1) with a generation counter.
  std::vector<uint32_t> keys_, vals_, gens_;
  uint32_t mask_ = 0;
  uint32_t gen_ = 0;
};

// Re-ranks cand[0..n_cand) (sorted by ascending PQ distance). Writes up to k results sorted by
// exact distance and returns their number. `reader` reads the index's whole page file, located
// through `layout` (fusion::Engine).
template <class T>
uint32_t HeuristicRerank(const T* query, uint32_t dim, const uint32_t* cand, uint32_t n_cand,
                         const LayoutMap& layout, PageReader* reader, RerankScratch* scratch,
                         const RerankParams& p, uint32_t* out_ids, float* out_dists,
                         RerankStats* st);

// Same for a page file that holds each candidate at its location locs[i] (rtier's data nodes:
// a sparse copy of the page file, located by the locations the posting lists carry, U1).
template <class T>
uint32_t HeuristicRerank(const T* query, uint32_t dim, const uint32_t* cand, const uint32_t* locs,
                         uint32_t n_cand, const RawLayout& layout, PageReader* reader,
                         RerankScratch* scratch, const RerankParams& p, uint32_t* out_ids,
                         float* out_dists, RerankStats* st);

}  // namespace fusion
