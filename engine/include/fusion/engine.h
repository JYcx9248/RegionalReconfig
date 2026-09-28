// FusionANNS query engine (paper Figure 7).
//
// Tiers held by the engine:
//   host memory : navigation graph over heads + vector-ID posting lists + page mapping table
//   GPU memory  : PQ codes of all vectors (CPU backend: host memory)
//   SSD         : raw vectors in 4 KB pages
//
// Query workflow (one CPU worker thread per query, as in the paper):
//   1  GPU builds the query's PQ distance table (async)
//   2  CPU traverses the navigation graph for the top-m posting lists
//   3  CPU gathers the lists' vector IDs (no vector content)
//   4-7 GPU deduplicates IDs, computes PQ distances, sorts, returns the top-n IDs
//   8  CPU re-ranks with raw vectors from SSD (heuristic early stop + I/O dedup)
#pragma once

#include <memory>
#include <string>

#include "fusion/common.h"
#include "fusion/page_reader.h"
#include "fusion/rerank.h"

namespace fusion {

enum class FilterDevice { kCpu, kGpu };

struct EngineOptions {
  std::string index_dir;
  std::string layout = "bucket";  // which SSD layout to use ("bucket" or "id")
  FilterDevice device = FilterDevice::kGpu;
  int num_workers = 1;            // concurrent query threads
  uint32_t max_nprobe = 256;      // largest m Search() will be called with
  uint32_t max_rerank = 1000;     // largest n Search() will be called with
  IoBackend io = IoBackend::kAuto;
  bool direct_io = true;
  uint32_t io_depth = 64;         // max in-flight reads per worker
  int gpu_device = 0;
};

struct SearchParams {
  uint32_t k = 10;
  uint32_t nprobe = 64;   // m: posting lists taken from the navigation graph
  uint32_t rerank = 100;  // n: PQ candidates passed to re-ranking
  uint32_t graph_ef = 0;  // graph search width; 0 = 2 * nprobe
  RerankParams rr;        // heuristic re-ranking and I/O dedup (rr.k is set from k)
};

struct QueryStats {
  uint32_t lists = 0;       // posting lists probed
  uint32_t candidates = 0;  // vector IDs gathered (including replicas)
  uint32_t unique = 0;      // distinct IDs scored with PQ
  uint32_t topn = 0;        // candidates passed to re-ranking
  RerankStats rerank;
  double graph_us = 0, gather_us = 0, filter_us = 0, rerank_us = 0, total_us = 0;
};

class Engine {
 public:
  static std::unique_ptr<Engine> Open(const EngineOptions& opts);
  virtual ~Engine() = default;

  // Applies query-independent settings (graph search width). Call while no Search() runs.
  virtual void Configure(const SearchParams& p) = 0;

  // `query` points to dim() elements of dtype(). Worker `w` must be used by one thread at a
  // time; different workers may run concurrently. Returns the number of results (<= k).
  virtual uint32_t Search(int w, const void* query, const SearchParams& p, uint32_t* ids,
                          float* dists, QueryStats* st) = 0;

  virtual DType dtype() const = 0;
  virtual uint32_t dim() const = 0;
  virtual uint64_t size() const = 0;
  virtual int num_workers() const = 0;
  virtual const char* filter_name() const = 0;
  virtual IoBackend io_backend() const = 0;
  // Human-readable summary of the three tiers and their footprint.
  virtual std::string Describe() const = 0;
};

}  // namespace fusion
