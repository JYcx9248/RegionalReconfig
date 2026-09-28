// Data-node engine for the regional tier (rtier).
//
// Same three tiers as fusion::Engine, with these differences:
//   * posting lists are held per partition, and partitions can be loaded and evicted while
//     queries run (the placement changes during reconfiguration; the lists themselves never do);
//   * PQ codes (the GPU tier) are node-level: the node holds exactly one copy of the code of
//     every vector named by its resident posting lists, however many lists or partitions name
//     it (fusion/pq_store.h; the decided part of design question U1). A partition's codes come
//     from peers (PQPut, deduplicated with PQMissing beforehand) or, when bootstrapping, from the
//     index's pq_codes.bin. Codes of a partition that moves away stay as a cache (the dataset is
//     static): they are dropped only to make room for a later staging;
//   * raw vectors (the SSD tier) are node-level too: the node holds a sparse copy of the index's
//     page file, the vectors named by the partitions it has held, at their canonical locations
//     (fusion/raw_store.h, the decided part of U1). A partition that arrives by migration goes
//     online without them (U9): RERANK fetches the ones it needs from the partition's old owner
//     on demand, while the agent streams the rest in the background (RawMissing, RawPut). There
//     is no vector -> page map: the posting lists carry each vector's location;
//   * the query pipeline is exposed as primitives, so that the Go aggregator can spread one
//     query over several nodes:
//       Navigate    navigation graph -> the query's top-m list IDs        (needs the graph)
//       Filter      given list IDs   -> PQ top-n (id, PQ distance)        (lists must be resident)
//       Rerank      given vector IDs -> exact top-k (id, distance)        (raw vectors; fetched
//                                                                          first if missing)
//       SearchLocal all of the above on this node (single-node baseline and correctness oracle)
//
// The navigation graph is large (tens of GB at billion scale) and immutable. It can be loaded
// after startup (LoadGraph), so a new node can join as a data node before its graph is staged.
//
// Every result list is ordered by (distance, id): results do not depend on the order in which
// candidates were produced, which is what lets rtier compare distributed and single-node runs.
#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "fusion/common.h"
#include "fusion/engine.h"
#include "fusion/page_reader.h"
#include "fusion/pq_store.h"
#include "fusion/raw_store.h"
#include "fusion/rerank.h"

namespace fusion {

// Thrown when a request names a list whose partition is not resident on this node.
class NotResidentError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
// Thrown when Navigate / SearchLocal run before the navigation graph is loaded.
class NoGraphError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct NodeOptions {
  std::string index_dir;       // index built by fusion_build (codebook, graph; codes and pages
                               // are read only to bootstrap partitions)
  std::string partitions_dir;  // written by rtier_segment (manifest, list_part.bin, segments)
  std::string layout;          // "" = the manifest's (the page layout its locations refer to)
  std::string raw_file;        // this node's raw vectors; "" = an anonymous temporary file
  FilterDevice device = FilterDevice::kCpu;
  int num_workers = 2;
  uint32_t max_nprobe = 256;   // largest m per request
  uint32_t max_rerank = 1000;  // largest n (Filter top-n, Rerank input) per request
  uint64_t pq_capacity = 0;    // PQ codes this node may hold (its HBM budget); 0 = all vectors
  IoBackend io = IoBackend::kAuto;
  bool direct_io = true;
  uint32_t io_depth = 64;
  int gpu_device = 0;
};

// Where LoadPartition takes what the node does not hold yet.
enum class PQSource {
  // Migration: PQ codes must have been installed with PQPut (else PQMissingError); raw vectors
  // come later, from the raw peer (see LoadPartition).
  kPresent,
  // Bootstrap (the initial deployment, --load): PQ codes and raw vectors are copied from the
  // index's pq_codes.bin and vectors_<layout>.bin.
  kIndex,
};

struct PQPutResult {
  uint64_t installed = 0;  // new codes (vectors)
  uint64_t skipped = 0;    // already held (or repeated in the batch)
};
using RawPutResult = PQPutResult;

// Fetches n raw vectors (vec_bytes each, in the order of locs) from the data node at `peer`
// into out; throws on failure. The node server installs one (the peer's RAW_GET).
using RawFetcher =
    std::function<void(const std::string& peer, const uint32_t* locs, size_t n, uint8_t* out)>;

struct FilterStats {
  uint32_t lists = 0;
  uint32_t gathered = 0;  // vector IDs gathered, replicas included
  uint32_t unique = 0;    // distinct IDs scored with PQ
};

class NodeEngine {
 public:
  static std::unique_ptr<NodeEngine> Open(const NodeOptions& opts);
  virtual ~NodeEngine() = default;

  // Staging. LoadGraph/LoadPartition may run concurrently with queries.
  virtual void LoadGraph(const std::string& path) = 0;
  virtual bool has_graph() const = 0;
  // Loads partition p's posting lists. Every vector they name must end up with a PQ code on
  // this node (see PQSource); the partition's postings then hold references on those codes.
  // With kPresent, the raw vectors not on the node yet are fetched from raw_peer (a data node's
  // address, normally the partition's old owner) when a query needs them, until the agent's
  // stream has installed them (RawPut); without raw_peer they must all be here already
  // (RawMissingError).
  virtual void LoadPartition(uint32_t partition, const std::string& segment_path, PQSource pq,
                             const std::string& raw_peer = "") = 0;
  // Returns false if the partition was not resident. Queries already holding the partition
  // finish on the old copy; its PQ references are dropped when the last one lets go of it. Its
  // codes stay on the node (cached) for a later reconfiguration.
  virtual bool EvictPartition(uint32_t partition) = 0;
  virtual std::vector<uint32_t> ResidentPartitions() const = 0;

  // PQ codes (node-level store). For each group of segment files (one group per source a
  // staging pulls from): the vector IDs they name that have no code on this node and are not
  // listed for an earlier group, sorted. Asking once for all sources makes every missing code
  // come exactly once, even when several sources could send it. The codes the node already
  // has (live or cached) are protected until the partitions load or PQReleaseStaged, so that
  // making room for the missing ones never drops them.
  virtual std::vector<std::vector<uint32_t>> PQMissing(
      const std::vector<std::vector<std::string>>& groups) = 0;
  // Codes of ids (m bytes each) for a peer; PQMissingError if one is not held.
  virtual void PQGet(const uint32_t* ids, size_t n, uint8_t* out) = 0;
  // Installs codes received from a peer (staged until LoadPartition names them).
  virtual PQPutResult PQPut(const uint32_t* ids, size_t n, const uint8_t* codes) = 0;
  // Ends a staging that did not load: frees the codes installed for it and returns the codes
  // it protected to the cache.
  virtual size_t PQReleaseStaged() = 0;
  virtual PQStats pq_stats() const = 0;

  // Raw vectors (node-level, fusion/raw_store.h). For each group of segment files (one group
  // per source): the locations they name whose vector is not on this node and that no earlier
  // group lists, sorted -- what the agent streams from each source.
  virtual std::vector<std::vector<uint32_t>> RawMissing(
      const std::vector<std::vector<std::string>>& groups) = 0;
  // Which of locs[0..n) are not on this node (sorted, without duplicates): the stream asks
  // again before each batch, so that what queries fetched meanwhile is not sent twice.
  virtual std::vector<uint32_t> RawAbsent(const uint32_t* locs, size_t n) const = 0;
  // Vectors at locs for a peer (vec_bytes each). One that has not arrived yet is fetched from
  // this node's own source first; RawMissingError if there is none.
  virtual void RawGet(const uint32_t* locs, size_t n, uint8_t* out) = 0;
  // Installs vectors streamed from a peer.
  virtual RawPutResult RawPut(const uint32_t* locs, size_t n, const uint8_t* vecs) = 0;
  virtual RawStats raw_stats() const = 0;
  // How the node fetches raw vectors on demand (without one it cannot).
  virtual void SetRawFetcher(RawFetcher fetcher) = 0;

  // Primitives. `query` points to dim() elements of dtype(). Worker `w` must be used by one
  // thread at a time. ef = 0 means 2 * nprobe (as in fusion::Engine).
  virtual uint32_t Navigate(const void* query, uint32_t nprobe, uint32_t ef,
                            uint32_t* lists) = 0;
  virtual uint32_t Filter(int w, const void* query, const uint32_t* lists, uint32_t nlists,
                          uint32_t topn, uint32_t* ids, float* dists, FilterStats* st) = 0;
  // Exact distances for all `n` IDs (no early stop: fixed-n re-ranking); keeps the k best.
  // Every ID must be named by a partition loaded here (NotResidentError); raw vectors that
  // have not arrived yet are fetched first (RawMissingError if that fails). The worker waits
  // for the fetch: during a warm-up, fetching queries hold workers longer.
  virtual uint32_t Rerank(int w, const void* query, const uint32_t* ids, uint32_t n, uint32_t k,
                          uint32_t* out_ids, float* out_dists, RerankStats* st) = 0;
  // Navigate + Filter + Rerank on this node; every probed list must be resident.
  virtual uint32_t SearchLocal(int w, const void* query, const SearchParams& p, uint32_t* ids,
                               float* dists, QueryStats* st) = 0;

  virtual DType dtype() const = 0;
  virtual uint32_t dim() const = 0;
  virtual uint64_t size() const = 0;
  virtual uint32_t num_lists() const = 0;
  virtual uint32_t num_partitions() const = 0;
  virtual int num_workers() const = 0;
  virtual const NodeOptions& options() const = 0;
  virtual const char* filter_name() const = 0;
  virtual const std::string& payload() const = 0;
  virtual uint32_t vec_bytes() const = 0;
  virtual std::string Describe() const = 0;
};

}  // namespace fusion
