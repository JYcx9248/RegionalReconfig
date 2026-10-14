// Data-node engine for the regional tier (rtier).
//
// Same three tiers as fusion::Engine, with these differences:
//   * posting lists are held per partition, and partitions can be loaded and evicted while
//     queries run (the placement changes during reconfiguration). A partition may be resident
//     at several versions at once: the epoch table names the version of each partition a query
//     sees, and FILTER and RERANK say which one they mean, so that an update batch (U14) can
//     stage a partition's next lists while queries of the current epoch still read the old ones.
//     Without updates every partition stays at version 0;
//   * PQ codes (the GPU tier) are node-level: the node holds exactly one copy of the code of
//     every vector named by its resident posting lists, however many lists or partitions name
//     it (fusion/pq_store.h; the decided part of design question U1). A partition's codes come
//     from peers (PQPut, deduplicated with PQMissing beforehand) or, when bootstrapping, from the
//     index's pq_codes.bin. Codes of a partition that moves away stay as a cache (the dataset is
//     static): they are dropped only to make room for a later staging;
//   * raw vectors (the SSD tier) are node-level too: the node holds a sparse copy of the index's
//     page file, the vectors named by the partitions it has held, at their canonical locations
//     (fusion/raw_store.h, the decided part of U1). A partition that arrives by migration goes
//     online without them (U9): RERANK fetches the ones a query needs from the partition's old
//     owner on demand, and only those -- the rest stay with the old owner, which keeps every
//     vector it has held (the lazy-stream baseline streams them in too: RawMissing, RawPut).
//     There is no vector -> page map: the posting lists carry each vector's location, and the
//     node keeps one source per posting list (the old owner of its partition), so any list
//     that names a missing vector says where to fetch it -- lists overlap, no per-vector table;
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
#include <utility>
#include <vector>

#include "fusion/common.h"
#include "fusion/engine.h"
#include "fusion/page_reader.h"
#include "fusion/pq_store.h"
#include "fusion/raw_store.h"
#include "fusion/read_budget.h"
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
  uint64_t read_iops = 0;      // device reads per second this node may issue (fusion/read_budget.h,
                               // for several nodes on one SSD without cgroups); 0 = no budget
  // Room for inserted vectors (U14). IDs and raw-vector locations are only ever appended, so
  // both spaces are sized at startup with headroom: the vector IDs (0 = the index's vectors)
  // and the page file's pages (0 = the index's pages).
  uint64_t max_vectors = 0;
  uint64_t max_pages = 0;
  // Inserted vectors wait in memory, served from there, until the writer puts them on the SSD
  // after the vectors fetched for queries; RawAdd blocks while more than this many bytes wait.
  uint64_t update_buffer_bytes = 256ull << 20;
  uint64_t update_write_bps = 0;  // bytes/s the writer spends on inserted vectors; 0 = no limit
  // Longest posting list a request may gather (0 = the manifest's): inserts make lists longer,
  // and the filter's candidate buffers are sized by it.
  uint32_t max_list_len = 0;
};

// One partition of a StageVersions call: its next version is built from `base`.
struct VersionStage {
  uint32_t partition = 0;
  uint32_t base = 0;
  uint32_t next = 0;
};

struct StageResult {
  uint32_t partition = 0;
  bool changed = false;  // false: nothing named it, no version was installed
  uint64_t added = 0;    // postings added and removed
  uint64_t removed = 0;
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
// into out; lists[i] is the posting list that named locs[i] here, so that a peer still lacking
// it knows its own source (chained migrations). Throws on failure. The node server installs one
// (the peer's RAW_GET).
using RawFetcher = std::function<void(const std::string& peer, const uint32_t* locs,
                                      const uint32_t* lists, size_t n, uint8_t* out)>;

// Raw-vector locations and, for each, the posting list that names it (parallel arrays).
struct RawRefs {
  std::vector<uint32_t> locs;
  std::vector<uint32_t> lists;
};

struct FilterStats {
  uint32_t lists = 0;
  uint32_t gathered = 0;  // vector IDs gathered, replicas included
  uint32_t unique = 0;    // distinct IDs scored with PQ
};

// The on-demand fetch path (U9) as the node's stats log samples it: read without waiting for
// the writer, so that a sample shows what queries and the writer are doing while it runs.
// Cumulative, except the ones marked "now".
struct FetchStats {
  uint64_t present = 0;          // now: raw vectors on the SSD
  uint64_t cached = 0;           // now: fetched, not on the SSD yet (served from memory)
  uint64_t queued = 0;           // now: of those, waiting for the writer
  uint64_t queued_peak = 0;
  uint64_t fetched = 0;          // new vectors fetched from peers
  uint64_t fetch_calls = 0;      // fetches that asked a peer (RERANK, SEARCH_LOCAL, RAW_GET chains)
  uint64_t fetch_us = 0;         // time those callers waited for them, backpressure included
  uint64_t written = 0;          // installed on the SSD by the writer
  uint64_t write_batches = 0;
  uint64_t write_us = 0;         // writer time in RawStore::Put
  uint64_t sync_installed = 0;   // installed by the fetching caller itself (queue full)
  uint64_t sync_us = 0;
  uint64_t drain_calls = 0;      // DrainWrites calls
  uint64_t drain_waiting = 0;    // now: callers in DrainWrites
  uint64_t drain_us = 0;
  uint64_t rerank_disk = 0;      // RERANK candidates read from the SSD
  uint64_t rerank_mem = 0;       // RERANK candidates re-ranked from memory
  uint64_t added = 0;            // inserted vectors received (RawAdd, U14)
  uint64_t added_queued = 0;     // now: of those, not on the SSD yet (served from memory)
  uint64_t added_written = 0;    // installed on the SSD by the writer
  uint64_t added_wait_us = 0;    // time RawAdd callers waited for room in the buffer
  uint64_t pending_recounts = 0; // scans of the resident postings for the pending count
  uint64_t pending_us = 0;
};

class NodeEngine {
 public:
  static std::unique_ptr<NodeEngine> Open(const NodeOptions& opts);
  virtual ~NodeEngine() = default;

  // Staging. LoadGraph/LoadPartition may run concurrently with queries.
  virtual void LoadGraph(const std::string& path) = 0;
  virtual bool has_graph() const = 0;
  // Loads version `version` of partition p's posting lists, replacing that version if it is
  // resident; the partition's other versions stay. Every vector they name must end up with a PQ
  // code on this node (see PQSource); the partition's postings then hold references on those
  // codes.
  // With kPresent, raw_peer (a data node's address, normally the partition's old owner) becomes
  // the source of the partition's lists: a raw vector not on the node is fetched from it when a
  // query needs it (and only then); without raw_peer they must all be here already
  // (RawMissingError). The source of a list outlives the partition's eviction, so the node can
  // still resolve a peer's request for it (chains).
  virtual void LoadPartition(uint32_t partition, const std::string& segment_path, PQSource pq,
                             const std::string& raw_peer = "", uint32_t version = 0) = 0;
  // Evicts every version of the partition; returns false if none was resident. Queries already
  // holding the partition finish on the old copy; its PQ references are dropped when the last
  // one lets go of it. Its codes stay on the node (cached) for a later reconfiguration.
  virtual bool EvictPartition(uint32_t partition) = 0;
  // Partitions with at least one resident version, ascending.
  virtual std::vector<uint32_t> ResidentPartitions() const = 0;
  // Every resident (partition, version), ascending.
  virtual std::vector<std::pair<uint32_t, uint32_t>> ResidentVersions() const = 0;
  // Update batches (U14). For each of `parts` (resident at its base version), builds the next
  // version of its posting lists: the base lists, plus the postings (lists[i], ids[i],
  // locs[i]) that name its lists, minus the IDs in deleted. Each list stays sorted by ID; a
  // posting already there is skipped. A partition the batch does not change gets no new
  // version (changed = false); the others are installed as version `next`, replacing that
  // version if it is resident, next to the base version, which keeps serving queries of the
  // current epoch. All or nothing: NotResidentError, PQMissingError (an added vector's code
  // must be here, PQPut), RawMissingError (its raw vector too, RawAdd) or a list longer than
  // max_list_len leave every partition as it was. A posting may only name a list of `parts`.
  virtual std::vector<StageResult> StageVersions(const std::vector<VersionStage>& parts,
                                                 const uint32_t* lists, const uint32_t* ids,
                                                 const uint32_t* locs, size_t nadds,
                                                 const uint32_t* deleted, size_t ndeleted) = 0;
  // Evicts one version of a partition (after the grace period of the epoch that used it);
  // returns false if it was not resident.
  virtual bool DropVersion(uint32_t partition, uint32_t version) = 0;
  // Frees the PQ codes of deleted vectors that no resident version names any more (cached
  // copies included); returns how many were freed.
  virtual size_t ForgetVectors(const uint32_t* ids, size_t n) = 0;
  // Writes a resident version of a partition as a segment file (the format of the partition
  // directory's), so that a node the partition moves to can load that version.
  virtual void SavePartition(uint32_t partition, uint32_t version, const std::string& path) const = 0;

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
  // group lists, sorted, each with a list that names it -- what a stream brings from each
  // source (the eager baselines and lazy-stream; the lazy protocol fetches on demand only).
  virtual std::vector<RawRefs> RawMissing(const std::vector<std::vector<std::string>>& groups) = 0;
  // Which of locs[0..n) are not on this node (sorted, without duplicates): the stream asks
  // again before each batch, so that what queries fetched meanwhile is not sent twice.
  virtual std::vector<uint32_t> RawAbsent(const uint32_t* locs, size_t n) const = 0;
  // Vectors at locs for a peer (vec_bytes each); lists[i] names locs[i]. One this node lacks
  // is fetched from the source of lists[i] first; RawMissingError if the list has none.
  virtual void RawGet(const uint32_t* locs, const uint32_t* lists, size_t n, uint8_t* out) = 0;
  // Installs vectors streamed from a peer.
  virtual RawPutResult RawPut(const uint32_t* locs, size_t n, const uint8_t* vecs) = 0;
  // Installs inserted vectors at their (new) locations (U14). They are kept in memory and served
  // from there at once; the writer puts them on the SSD in the background, after the vectors
  // fetched for queries and within update_write_bps, and not at all while update writes are
  // paused. Blocks while update_buffer_bytes of them are waiting: backpressure on the sender.
  virtual RawPutResult RawAdd(const uint32_t* locs, size_t n, const uint8_t* vecs) = 0;
  // Holds back (or resumes) writing inserted vectors, e.g. while the node stages a
  // reconfiguration, whose transfers go first.
  virtual void PauseUpdateWrites(bool paused) = 0;
  // The stream without the agents in its path: of locs[0..n) (lists[i] names locs[i]), fetches
  // the vectors this node lacks -- not on the SSD and not fetched by a query already -- from the
  // source of their lists, data node to data node as RERANK does (chains included), and installs
  // them as RawPut does (counted as streamed, not as fetched on demand). RawMissingError if a
  // list has no source.
  virtual RawPutResult RawPull(const uint32_t* locs, const uint32_t* lists, size_t n) = 0;
  // A snapshot that waits for nothing, or (settle) one taken once the vectors fetched before
  // the call are on the SSD: with no query fetching, cached is then 0.
  virtual RawStats raw_stats(bool settle) const = 0;
  virtual FetchStats fetch_stats() const = 0;
  virtual ReadBudgetStats read_budget_stats() const = 0;  // zeros without --read-iops
  // How the node fetches raw vectors on demand (without one it cannot).
  virtual void SetRawFetcher(RawFetcher fetcher) = 0;

  // Primitives. `query` points to dim() elements of dtype(). Worker `w` must be used by one
  // thread at a time. ef = 0 means 2 * nprobe (as in fusion::Engine). In Filter and Rerank,
  // versions[i] is the version of lists[i]'s partition that the query sees (its epoch's);
  // NotResidentError if that version is not resident. versions = nullptr means each partition's
  // newest resident version.
  virtual uint32_t Navigate(const void* query, uint32_t nprobe, uint32_t ef,
                            uint32_t* lists) = 0;
  virtual uint32_t Filter(int w, const void* query, const uint32_t* lists, const uint32_t* versions,
                          uint32_t nlists, uint32_t topn, uint32_t* ids, float* dists,
                          FilterStats* st) = 0;
  // Exact distances for all `n` IDs (no early stop: fixed-n re-ranking); keeps the k best.
  // Every ID must be named by a partition loaded here (NotResidentError). `lists` are the
  // query's lists on this node (those its FILTER gathered from): a raw vector not on the node
  // is fetched first from the source of a list among them that names it (RawMissingError if
  // that fails). The worker waits for the fetch, not for the write: fetched vectors are
  // re-ranked from memory, kept for other queries until a background writer has put them on
  // the SSD.
  virtual uint32_t Rerank(int w, const void* query, const uint32_t* ids, uint32_t n, uint32_t k,
                          const uint32_t* lists, const uint32_t* versions, uint32_t nlists,
                          uint32_t* out_ids, float* out_dists, RerankStats* st) = 0;
  // The posting lists the index's build would have given each of n vectors (`vecs`, dim()
  // elements of dtype() each), and their PQ codes: the routing of inserted vectors (U14), by
  // the build's own rule and parameters (index meta: replicas, closure_eps, the assignment's
  // candidates and search width) on this node's navigation graph, so that an insert lands
  // where the build would have put it. lists gets assign_replicas() entries per vector, padded
  // with kInvalidId; codes gets pq_m() bytes per vector. Needs the graph (NoGraphError).
  virtual void Assign(const void* vecs, uint32_t n, uint32_t* lists, uint8_t* codes) = 0;
  virtual uint32_t assign_replicas() const = 0;
  virtual uint32_t pq_m() const = 0;
  // Size of the vector-ID space (NodeOptions::max_vectors, at least the index's vectors).
  virtual uint64_t id_capacity() const = 0;
  // Navigate + Filter + Rerank on this node, on each partition's newest version; every probed
  // list must be resident.
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
