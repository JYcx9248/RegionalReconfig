// PQ-based candidate filtering (paper Section 3.2, steps 1 and 4-7 of Figure 7):
// build the query's distance table, deduplicate the candidate vector IDs collected from the
// top-m posting lists, compute PQ distances, and return the top-n IDs for re-ranking.
//
// Two backends implement the same interface: the GPU backend (PQ codes pinned in GPU memory,
// one CUDA stream and one pre-allocated working-memory block per worker) and a CPU backend
// that runs the identical steps on the host -- used for testing without a GPU and for the
// paper's "MI (CPU)" ablation.
#pragma once

#include <memory>

#include "fusion/common.h"
#include "fusion/pq.h"

namespace fusion {

class FilterWorker {
 public:
  virtual ~FilterWorker() = default;

  // Buffer the caller fills with candidate IDs (host memory; pinned on the GPU backend).
  // Holds up to FilterConfig::max_candidates entries.
  virtual uint32_t* candidate_buffer() = 0;

  // Slot-addressed backends (FilterConfig::slot_addressed): buffer parallel to
  // candidate_buffer() that the caller fills with the code slot of each candidate. Null for
  // backends whose codes are addressed by vector ID.
  virtual uint32_t* slot_buffer() = 0;

  // Step 1: starts building the distance table for `query` (dim floats). Asynchronous on the
  // GPU, so it overlaps with the CPU's graph traversal.
  virtual void BeginQuery(const float* query) = 0;

  // Steps 4-7: filters the first n_cand IDs of candidate_buffer(). Blocks until done and
  // returns the number of results (<= topn), sorted by ascending PQ distance and available
  // through result_ids()/result_dists(). *n_unique receives the number of distinct IDs.
  virtual uint32_t Filter(uint32_t n_cand, uint32_t topn, uint32_t* n_unique) = 0;

  virtual const uint32_t* result_ids() const = 0;
  virtual const float* result_dists() const = 0;
};

struct FilterConfig {
  int num_workers = 1;
  uint32_t max_candidates = 0;  // upper bound on IDs gathered per query
  uint32_t max_topn = 0;        // upper bound on the re-ranking count n
  int gpu_device = 0;
  // false: the code of vector `id` is at codes + id * m (fusion::Engine: every code resident).
  // true: codes are addressed by slot (rtier's node-level PQ store, fusion/pq_store.h): the
  // caller fills slot_buffer() next to candidate_buffer(), `n` is the number of slots, and
  // codes written into slots later are announced with StoreCodes. Results still carry IDs and
  // break distance ties by ID.
  bool slot_addressed = false;
};

class FilterBackend {
 public:
  virtual ~FilterBackend() = default;
  virtual const char* name() const = 0;
  virtual FilterWorker* worker(int i) = 0;
  // Bytes of device (or host, for CPU) memory reserved for PQ codes.
  virtual uint64_t code_bytes() const = 0;
  // Slot-addressed backends only: the host store `host_codes` (slot-addressed, m bytes per
  // slot) has new codes in `slots` (ascending). The GPU backend copies them to the device; the
  // CPU backend reads the host store directly. No running query may read these slots.
  virtual void StoreCodes(const uint32_t* slots, uint32_t n, const uint8_t* host_codes) = 0;
};

// `codes` (n x m bytes) must outlive the CPU backend. The GPU backend copies them to the
// device, so the caller may free the host copy afterwards. With cfg.slot_addressed, `codes` is
// the slot-addressed host store (n slots; the GPU backend may be given null and starts empty).
std::unique_ptr<FilterBackend> CreateCpuFilter(const PQCodebook& cb, const uint8_t* codes,
                                               uint64_t n, const FilterConfig& cfg);
std::unique_ptr<FilterBackend> CreateGpuFilter(const PQCodebook& cb, const uint8_t* codes,
                                               uint64_t n, const FilterConfig& cfg);
bool GpuFilterAvailable();

}  // namespace fusion
