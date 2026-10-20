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
//     There is no vector -> page map: the posting lists carry each vector's location and the
//     node that holds it (its holder: this node, or a peer), so a missing vector is one hop away;
//   * the query pipeline is exposed as primitives, so that the Go aggregator can spread one
//     query over several nodes:
//       Navigate    navigation graph -> the query's top-m list IDs        (needs the graph)
//       Filter      given list IDs   -> PQ top-n (id, PQ distance)        (lists must be resident)
//       Rerank      given vector IDs -> exact top-k (id, distance)        (raw vectors; fetched
//                                                                          first if missing)
//       SearchLocal all of the above on this node (single-node baseline and correctness oracle)
//
// The navigation graph is large (tens of GB at billion scale). It can be loaded after startup
// (LoadGraph), so a new node can join as a data node before its graph is staged. A node holds it
// once: a major epoch (U14) changes it by a patch staged next to it and merged in place later
// (fusion/graph_patch.h), and the graph is loaded with headroom for the heads patches add.
//
// A full rebuild (U14) moves a node, once it holds no partition, to another generation of the
// index (SwitchGeneration): new lists, partitions and graph, built outside the system with the
// same codebook, vector IDs and raw-vector locations -- so the PQ codes and raw vectors the node
// holds stay valid and are reused, and the node never holds two generations' lists or graphs.
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
#include "fusion/partition.h"
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
  // Room for the heads and lists that major epochs add (U14), as a fraction of the build's: the
  // graph is loaded with capacity for that many more elements, and list IDs up to the same
  // number are accepted.
  double graph_headroom = 0.2;
};

// Navigate's graph version: the newest the node has (a staged patch's, else the graph's).
constexpr uint32_t kNewestGraph = 0xFFFFFFFFu;

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
  // (list, postings) of every list of the next version whose length changed, and of every list
  // it creates: what decides the next major epoch (U14).
  std::vector<std::pair<uint32_t, uint32_t>> lengths;
};

// What StageVersions changes besides the base lists (parallel arrays; null when empty).
struct VersionEdits {
  // Postings to add: lists[i] gets (ids[i], locs[i]). holders[i] is the node that holds the
  // vector when it is not here (a posting that moves between owners: fetched from there when a
  // query needs it); kNoHolder, or this node's ID, means it must be here. holders may be null.
  const uint32_t* lists = nullptr;
  const uint32_t* ids = nullptr;
  const uint32_t* locs = nullptr;
  const uint16_t* holders = nullptr;
  size_t nadds = 0;
  // Vector IDs deleted from every list.
  const uint32_t* deleted = nullptr;
  size_t ndeleted = 0;
  // Lists the next version creates (a major epoch's split children), with their partitions.
  const uint32_t* created = nullptr;
  const uint32_t* created_parts = nullptr;
  size_t ncreated = 0;
  // Lists the next version no longer has (a split parent, a merged list), postings and all.
  const uint32_t* removed = nullptr;
  size_t nremoved = 0;
};

// One posting list as its owner holds it (GetList): every posting's holder, this node by its ID
// for the vectors it holds.
struct ListPostings {
  std::vector<uint32_t> ids;
  std::vector<uint32_t> locs;
  std::vector<uint16_t> holders;
};

struct GraphState {
  bool loaded = false;
  uint32_t version = 0;   // the version of the graph's bytes
  uint32_t patch = 0;     // the staged patch's version (0: none)
  uint64_t count = 0;     // elements (heads)
  uint64_t capacity = 0;  // room for elements
  uint64_t deleted = 0;   // marked deleted (removed heads stay as routing points)
};

// Where LoadPartition takes what the node does not hold yet.
enum class PQSource {
  // Migration: PQ codes must have been installed with PQPut (else PQMissingError); raw vectors
  // come later, from the raw peer (see LoadPartition).
  kPresent,
  // Bootstrap (the initial deployment, --load): PQ codes and raw vectors are copied from the
  // index's pq_codes.bin and vectors_<layout>.bin.
  kIndex,
  // A generation's partitions after a full rebuild (U14): the codes the node lacks are copied
  // from the generation's pq_codes.bin (one codebook: the same codes), and raw vectors come from
  // the holders the postings record, as with kPresent.
  kGenIndex,
};

struct PQPutResult {
  uint64_t installed = 0;  // new codes (vectors)
  uint64_t skipped = 0;    // already held (or repeated in the batch)
};
using RawPutResult = PQPutResult;

// Fetches n raw vectors (vec_bytes each, in the order of locs) from the data node at `peer`,
// which holds them, into out. Throws on failure. The node server installs one (the peer's
// RAW_GET).
using RawFetcher = std::function<void(const std::string& peer, const uint32_t* locs, size_t n,
                                      uint8_t* out)>;

// Raw-vector locations held by one node, each with a posting list here that names it
// (parallel arrays).
struct RawRefs {
  uint16_t holder = kNoHolder;
  std::vector<uint32_t> locs;
  std::vector<uint32_t> lists;
};

// What RawHeldBy found: postings that named the node and whose vector is here now (they name
// this node from now on), and the ones whose vector must be pulled from it.
struct HeldBy {
  uint64_t repointed = 0;
  RawRefs missing;
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
  uint64_t fetch_calls = 0;      // fetches that asked a peer (RERANK, SEARCH_LOCAL)
  uint64_t coalesced = 0;        // missing vectors a fetch waited for instead of fetching: another
                                 // query was fetching them
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

  // Staging. LoadGraph/LoadPartition may run concurrently with queries. LoadGraph loads the
  // build's graph (version 0) with room for graph_headroom more elements.
  virtual void LoadGraph(const std::string& path) = 0;
  // Full rebuild (U14): moves the node to `generation`, the index at index_dir (fusion_build's
  // output) partitioned as in partitions_dir (rtier_segment's). The node must hold no partition;
  // its graph and a staged patch are dropped, and the generation's manifest, PQ code file and
  // assignment rule replace the current ones -- the PQ codes and raw vectors stay. Refused, with
  // nothing changed, unless the generation has the same codebook, dimensions and page geometry
  // and its lists fit max_list_len.
  virtual void SwitchGeneration(uint32_t generation, const std::string& index_dir,
                                const std::string& partitions_dir) = 0;
  virtual uint32_t generation() const = 0;
  virtual bool has_graph() const = 0;
  virtual GraphState graph_state() const = 0;

  // Graph patches (U14 major epochs, fusion/graph_patch.h). StageGraphPatch stages a patch from
  // the graph's version to the next one next to the graph, which it leaves alone: Navigate of the
  // next version searches through the patch, Navigate of the current one the graph alone. A
  // patch staged again replaces the staged one.
  virtual void StageGraphPatch(const uint8_t* patch, size_t n) = 0;
  // Merges the staged patch of `version` into the graph in place, once no search of the version
  // before it can come any more (the grace period): searches through the patch go on meanwhile,
  // and searches of the old version are refused from then on. A no-op if the graph is at
  // `version` already.
  virtual void MergeGraphPatch(uint32_t version) = 0;
  // Unstages the patch of `version`; false if it was not staged.
  virtual bool DropGraphPatch(uint32_t version) = 0;
  // Writes the staged patch of `version` as a file, for a graph transfer to a later entry.
  virtual void SaveGraphPatch(uint32_t version, const std::string& path) const = 0;
  // The router's side (the update controller's own node, which holds the authoritative graph and
  // stages no patch): inserts the heads `labels` (list IDs; vecs: dim() elements of dtype() each)
  // and marks the heads `deleted` deleted -- they stay as routing points, never a result -- then
  // returns the patch from the graph's version to `version` (the next one). Refused, with
  // nothing changed, if the graph has no room left (graph_headroom), a label exists, or a
  // deleted head is unknown or deleted already.
  virtual std::vector<uint8_t> EditGraph(uint32_t version, const uint32_t* labels, const void* vecs,
                                         uint32_t nnew, const uint32_t* deleted, uint32_t ndeleted) = 0;
  // The router's decisions for a major epoch, on PQ codes (pq_m() bytes each), so that no raw
  // vector is read. SplitCodes: a balanced 2-means of n codes (n >= 2); side[i] is 0 or 1, and
  // heads[s] the index of the member closest to side s's mean, the child's head as the build
  // chooses heads. NearestHeads: the nearest head of each code on the graph (deleted heads are
  // never a result), searched with width ef.
  virtual void SplitCodes(const uint8_t* codes, uint32_t n, uint8_t* side, uint32_t heads[2]) const = 0;
  virtual void NearestHeads(const uint8_t* codes, uint32_t n, uint32_t ef, uint32_t* lists) const = 0;
  // Loads version `version` of partition p's posting lists, replacing that version if it is
  // resident; the partition's other versions stay. Every vector they name must end up with a PQ
  // code on this node (see PQSource); the partition's postings then hold references on those
  // codes.
  // Raw vectors: every key (list, vector) has one current holder. With kIndex the node copies
  // them all and holds them. With kPresent, a vector not on the node is fetched, when a query
  // needs it, from the holder its posting records (partition.h; a segment written at a hand-off
  // records one for every posting). fallback_holder stands in for postings that record none (a
  // version-2 segment); a vector missing here with no holder at all is a RawMissingError.
  virtual void LoadPartition(uint32_t partition, const std::string& segment_path, PQSource pq,
                             uint32_t version = 0, uint16_t fallback_holder = kNoHolder) = 0;
  // Evicts every version of the partition; returns false if none was resident. Queries already
  // holding the partition finish on the old copy; its PQ references are dropped when the last
  // one lets go of it. Its codes stay on the node (cached) for a later reconfiguration.
  virtual bool EvictPartition(uint32_t partition) = 0;
  // Partitions with at least one resident version, ascending.
  virtual std::vector<uint32_t> ResidentPartitions() const = 0;
  // Every resident (partition, version), ascending.
  virtual std::vector<std::pair<uint32_t, uint32_t>> ResidentVersions() const = 0;
  // A posting list of a resident version (NotResidentError if it is not here).
  virtual ListPostings GetList(uint32_t list, uint32_t version) const = 0;
  // Update batches (U14). For each of `parts` (resident at its base version), builds the next
  // version of its posting lists: the base lists without the removed ones and with the created
  // ones, plus the added postings that name its lists, minus the deleted IDs. Each list stays
  // sorted by ID; a posting already there is skipped. A partition the batch does not change gets
  // no new version (changed = false); the others are installed as version `next`, replacing that
  // version if it is resident, next to the base version, which keeps serving queries of the
  // current epoch. All or nothing: NotResidentError, PQMissingError (an added vector's code
  // must be here, PQPut), RawMissingError (its raw vector must be here, RawAdd, unless the
  // posting names another holder) or a list longer than max_list_len leave every partition as
  // it was. A posting may only name a list of `parts`.
  virtual std::vector<StageResult> StageVersions(const std::vector<VersionStage>& parts,
                                                 const VersionEdits& edits) = 0;
  // A minor batch: postings added (held here) and IDs deleted.
  std::vector<StageResult> StageVersions(const std::vector<VersionStage>& parts, const uint32_t* lists,
                                         const uint32_t* ids, const uint32_t* locs, size_t nadds,
                                         const uint32_t* deleted, size_t ndeleted) {
    VersionEdits e;
    e.lists = lists;
    e.ids = ids;
    e.locs = locs;
    e.nadds = nadds;
    e.deleted = deleted;
    e.ndeleted = ndeleted;
    return StageVersions(parts, e);
  }
  // Evicts one version of a partition (after the grace period of the epoch that used it);
  // returns false if it was not resident.
  virtual bool DropVersion(uint32_t partition, uint32_t version) = 0;
  // PQ codes of vectors no list will name again (PQStore::Forget): with deleted, the vectors
  // are deleted -- their unreferenced copies are freed now, and the ones a resident version
  // still names are freed with their last reference, never cached; without it, only the
  // unreferenced copies go. Returns how many were freed now.
  virtual size_t ForgetVectors(const uint32_t* ids, size_t n, bool deleted) = 0;
  // Writes a resident version of a partition as a segment file (the format of the partition
  // directory's), so that a node the partition moves to can load that version. Every posting
  // records its current holder: this node (SetNodes) for the vectors it holds, the recorded one
  // for the others.
  virtual void SavePartition(uint32_t partition, uint32_t version, const std::string& path) const = 0;
  // The cluster's node IDs: this node's (0: unchanged) and other data nodes' addresses, where
  // it fetches the vectors that postings say they hold. Merged into what it knew.
  virtual void SetNodes(uint16_t self, const std::vector<std::pair<uint16_t, std::string>>& peers) = 0;

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

  // Raw vectors (node-level, fusion/raw_store.h). The vectors the newest resident versions of
  // these partitions name that are not on this node, grouped by holder, each once with a list
  // naming it -- what a stream brings (the eager baselines and lazy-stream; the lazy protocol
  // fetches on demand only).
  virtual std::vector<RawRefs> RawMissing(const std::vector<uint32_t>& partitions) = 0;
  // Before a node leaves (U9): of every resident posting whose holder is `holder`, the ones
  // whose vector is here now are made to name this node, and the others are returned, to be
  // pulled from it (RawPull). Once they are, no posting here names the node.
  virtual HeldBy RawHeldBy(uint16_t holder) = 0;
  // Which of locs[0..n) are not on this node (sorted, without duplicates): the stream asks
  // again before each batch, so that what queries fetched meanwhile is not sent twice.
  virtual std::vector<uint32_t> RawAbsent(const uint32_t* locs, size_t n) const = 0;
  // Vectors at locs for a peer (vec_bytes each), from this node's own store: a peer asks the
  // holder, so there is no chain to follow. RawMissingError for one that is not here.
  virtual void RawGet(const uint32_t* locs, size_t n, uint8_t* out) = 0;
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
  // holders their postings in lists[i] record, data node to data node as RERANK does, installs
  // them as RawPut does (counted as streamed, not as fetched on demand), and makes those
  // postings name this node. RawMissingError if one has no holder.
  virtual RawPutResult RawPull(const uint32_t* locs, const uint32_t* lists, size_t n) = 0;
  // Which raw vectors the node holds (on the SSD, or fetched and not written yet): one bit per
  // location, 64 per word -- what a full rebuild places the new partitions by (U14).
  virtual std::vector<uint64_t> RawBitmap() const = 0;
  // A snapshot that waits for nothing, or (settle) one taken once the vectors fetched before
  // the call are on the SSD: with no query fetching, cached is then 0.
  virtual RawStats raw_stats(bool settle) const = 0;
  virtual FetchStats fetch_stats() const = 0;
  virtual ReadBudgetStats read_budget_stats() const = 0;  // zeros without --read-iops
  // How the node fetches raw vectors on demand (without one it cannot).
  virtual void SetRawFetcher(RawFetcher fetcher) = 0;

  // Primitives. `query` points to dim() elements of dtype(). Worker `w` must be used by one
  // thread at a time. ef = 0 means 2 * nprobe (as in fusion::Engine). Navigate searches the
  // graph version the query's epoch names (NotResidentError if this node does not have it). In
  // Filter and Rerank, versions[i] is the version of lists[i]'s partition that the query sees
  // (its epoch's); NotResidentError if that version is not resident. versions = nullptr means
  // each partition's newest resident version.
  virtual uint32_t Navigate(const void* query, uint32_t nprobe, uint32_t ef, uint32_t* lists,
                            uint32_t graph = kNewestGraph) = 0;
  virtual uint32_t Filter(int w, const void* query, const uint32_t* lists, const uint32_t* versions,
                          uint32_t nlists, uint32_t topn, uint32_t* ids, float* dists,
                          FilterStats* st) = 0;
  // Exact distances for all `n` IDs (no early stop: fixed-n re-ranking); keeps the k best.
  // Every ID must be named by a partition loaded here (NotResidentError). `lists` are the
  // query's lists on this node (those its FILTER gathered from): a raw vector not on the node
  // is fetched first, through a list among them that names it, from the holder that posting
  // records (RawMissingError if that fails), and the posting names this node from then on. A
  // vector another query is fetching already is waited for, not fetched twice. Fetched vectors
  // are re-ranked from memory, kept for other queries until a background writer has put them on
  // the SSD: a fetch waits for the holder, not for the write. The server fetches before it takes
  // the worker (FetchForRerank), so that the worker waits for neither.
  virtual uint32_t Rerank(int w, const void* query, const uint32_t* ids, uint32_t n, uint32_t k,
                          const uint32_t* lists, const uint32_t* versions, uint32_t nlists,
                          uint32_t* out_ids, float* out_dists, RerankStats* st) = 0;
  // Rerank's fetch on its own, with no engine worker (same arguments, same errors). A re-ranking
  // whose vectors are elsewhere waits for their holders, and a worker held meanwhile would be
  // lost to the node's FILTERs and other RERANKs -- all of a new node's warm-up (U9). Returns how
  // many vectors it installed; *fetch_us (may be null) gets the time it waited.
  virtual uint32_t FetchForRerank(const uint32_t* ids, uint32_t n, const uint32_t* lists,
                                  const uint32_t* versions, uint32_t nlists, double* fetch_us) = 0;
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
