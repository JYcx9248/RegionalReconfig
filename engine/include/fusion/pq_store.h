// Node-level PQ code store of an rtier data node (the decided part of design question U1).
//
// A node keeps exactly one copy of each vector's PQ code, however many of its resident posting
// lists -- of however many partitions -- contain the vector. (FusionANNS replicates boundary
// vectors into up to 8 lists, so partitions share many vectors; storing codes per partition
// would duplicate them in the scarce GPU memory.) Codes live in fixed-size slots of a buffer
// sized to the node's budget (`capacity` codes: the HBM budget); an ID -> slot map finds them.
//
// Every code is in one of three states:
//   live    named by at least one resident segment (a partition's version): one reference per
//           segment that names the vector, however many of its postings do (Ref);
//   staged  protected for a staging in progress: installed by Put, or already here and claimed
//           by Reserve, and not yet named by a loaded partition;
//   cached  no longer named by a resident partition (its partition moved away, or a version
//           an update batch replaced). Vectors are immutable (an update deletes and inserts
//           under a new ID, U14), so the code stays valid and is kept: the node held it before,
//           and a later reconfiguration may bring the partition back. Cached codes are only
//           dropped when a Put would otherwise exceed the capacity.
//
// A deleted vector's code never becomes cached: it can never be named again, so it is no
// cache. Forget frees it at once if nothing names it, and otherwise marks it, so that it is
// freed with its last reference -- the version that queries of an epoch before the delete may
// still read.
//
// With the placement rtier uses, the last case does not arise: a scale-out only takes
// partitions away from a node and a scale-in gives them back to a node that held them, so the
// resident set stays what the node held at the initial cluster size and fits the budget it was
// started with. Eviction is the safety net for a cluster that shrinks below that size, or a
// placement that hands a partition to a node that never had it.
//
// NodeEngine::LoadPartition adds a segment's references before the segment becomes visible to
// queries; they are dropped once the last query holding the segment lets go of it (after
// EvictPartition). A slot never moves while its code is here, so queries read codes by slot
// without taking the store's lock (the slots of a segment are resolved once, when it loads), and
// they find a referenced code's slot and location without it too (Locate, SlotOf): a load, a
// reclaim or a peer's PQ_GET never holds a query back. The lock guards the references and the
// slots' states; a load holds it once, over the segment's distinct vectors in ID order, and does
// the per-posting work outside it, so that several partitions load at once. A Put holds it only
// to hand out slots: it claims its IDs and writes their codes without it, so that a staging's
// PQ_PUTs install side by side.
//
// The store is also where the node looks up a vector's raw-vector location (fusion/layout.h,
// RawLayout): the segments carry it next to every posting, and Ref records it with the code.
// It stays known while the code is here -- live or cached, the dataset is static -- so a node
// needs no vector -> page map of its own (U1). A code installed by Put has no location until a
// partition that names the vector loads.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "fusion/common.h"

namespace fusion {

// A PQ code the operation needs is not on this node.
class PQMissingError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};
// Installing the codes would exceed the node's PQ capacity (its HBM budget), even after
// dropping every cached code.
class PQCapacityError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Where installed codes came from (statistics only).
enum class PQOrigin { kReceived, kIndex, kSnapshot };

struct PQStats {
  uint32_t m = 0;          // bytes per code
  uint64_t capacity = 0;   // slots (the node's PQ budget, in codes)
  uint64_t resident = 0;   // codes held now = live + staged + cached
  uint64_t live = 0;       // named by resident partitions
  uint64_t staged = 0;     // protected for a staging in progress
  uint64_t cached = 0;     // kept after their partitions moved away
  uint64_t received = 0;   // cumulative: codes installed from peers (PQ_PUT)
  uint64_t from_index = 0; // cumulative: codes installed from the index files
  uint64_t restored = 0;   // cumulative: codes installed from a warmup snapshot (Restore)
  uint64_t skipped = 0;    // cumulative: codes offered that were already held
  uint64_t evicted = 0;    // cumulative: cached codes dropped to make room
  uint64_t freed = 0;      // cumulative: codes dropped (evicted, released after a staging, deleted)
  // Codes of deleted vectors that a resident version still names (Forget marked them): freed
  // with their last reference, not cached.
  uint64_t deleted = 0;
  uint64_t served = 0;     // cumulative: codes copied out for peers (PQ_GET)
};

class PQStore {
 public:
  static constexpr uint32_t kNoSlot = 0xFFFFFFFFu;
  static constexpr uint32_t kNoLoc = 0xFFFFFFFFu;
  // Called after Put wrote new codes into `slots` (ascending) of codes(), before it publishes
  // them, without the store's lock (Puts may call it concurrently); the GPU filter copies them
  // into device memory. If it throws, the Put is undone.
  using Mirror = std::function<void(const uint32_t* slots, uint32_t n)>;

  // num_vectors: size of the ID space; m: bytes per code; capacity: slots (<= num_vectors).
  PQStore(uint64_t num_vectors, uint32_t m, uint64_t capacity);

  uint32_t m() const { return m_; }
  uint64_t capacity() const { return capacity_; }
  // capacity x m bytes, addressed by slot. The pointer never changes.
  const uint8_t* codes() const { return codes_.get(); }
  void SetMirror(Mirror mirror);

  // IDs among ids[0..n) that have no code here: sorted, without duplicates.
  std::vector<uint32_t> Missing(const uint32_t* ids, size_t n) const;
  // Same, for a staging: the codes that ARE here are protected (staged) until a partition
  // that names them loads, or ReleaseStaged, so that no Put of the same staging drops them.
  std::vector<uint32_t> Reserve(const uint32_t* ids, size_t n);
  // Installs n codes (m bytes each, in the order of ids) as staged. IDs already held, or being
  // installed by a concurrent Put, are skipped (their code is kept). Cached codes are dropped if
  // the new ones do not fit otherwise; if they still do not fit, throws PQCapacityError and
  // changes nothing. Returns the number of codes installed. Puts run concurrently, but for the
  // slots' accounting.
  size_t Put(const uint32_t* ids, size_t n, const uint8_t* codes, PQOrigin origin);
  // Copies the codes of ids[0..n) into out (n x m bytes). Throws PQMissingError if one is absent.
  void Get(const uint32_t* ids, size_t n, uint8_t* out);
  // Adds a reference to the code of each of ids[0..n) -- ascending, without duplicates: the
  // distinct vectors of one segment, which references each once -- making it live, and writes
  // its slot to slots_out (may be null). locs (may be null): each one's raw-vector location,
  // recorded with the code. Throws PQMissingError if a code is absent, std::runtime_error if a
  // location contradicts the one recorded before, std::invalid_argument for IDs out of order or
  // out of range; in every case nothing changes.
  void Ref(const uint32_t* ids, size_t n, uint32_t* slots_out, const uint32_t* locs = nullptr);
  // The slot of id's code, without the lock: only for a code the caller holds a reference to
  // (it cannot move then); kNoSlot if none is here.
  uint32_t SlotOf(uint32_t id) const;
  // Writes the raw-vector location of ids[0..n) to locs_out: kNoLoc where it is not known (no
  // code here, or no loaded partition has named the vector since its code arrived). Returns how
  // many are not known. Without the lock: exact for codes a resident segment references (a
  // query's candidates), whose slot and location cannot change.
  size_t Locate(const uint32_t* ids, size_t n, uint32_t* locs_out) const;
  // Drops the references a segment took (the slots Ref gave it, any order); a code with no
  // reference left becomes cached (or stays staged if a staging protects it). Never frees but a
  // deleted vector's code (Forget), never throws (it runs from the segment reclaimer);
  // inconsistencies are logged.
  void Unref(const uint32_t* slots, size_t n) noexcept;
  // Ends a staging that did not load (rollback): frees staged codes no partition ever named
  // and returns the claimed ones to the cache. Returns how many codes were freed.
  size_t ReleaseStaged();
  // Frees the codes of ids[0..n) that no resident posting references and no staging protects.
  // With deleted (U14: the vectors are deleted, so their copies are dead, not a cache), the
  // others -- named by a version that old queries may still read -- are marked, and each is
  // freed when its last reference goes instead of becoming cached. Without it, only the
  // unreferenced ones go (codes known to be unused, e.g. of inserts whose batch did not flip).
  // Returns how many were freed now.
  size_t Forget(const uint32_t* ids, size_t n, bool deleted);

  // A warmup snapshot of the codes (rtier_node --pq-file): every code held, in ID order, written
  // durably to path with codebook_crc, the CRC-32 of the codebook's centroids, so that codes of
  // another codebook are refused. Returns how many. Holds the store's lock while it writes: a
  // warmup saves an idle node.
  uint64_t Save(const std::string& path, uint32_t codebook_crc) const;
  // Installs the codes of a file Save wrote into this store, empty, as Put installs them
  // (staged until a partition that names them loads), and drops the file from the page cache:
  // a node that starts on a snapshot reads its own codes instead of a pass over the index's
  // pq_codes.bin. Refuses, before installing any, a file of another codebook or code size, a
  // damaged or truncated one, and more codes than the capacity. Returns how many.
  uint64_t Restore(const std::string& path, uint32_t codebook_crc);

  PQStats Stats() const;

 private:
  enum Category { kLive, kStaged, kCached };
  Category CategoryOf(uint32_t s) const;
  void Count(Category c, int64_t delta);
  void FreeLocked(uint32_t slot);
  void EvictLocked(uint64_t count);  // drops `count` cached codes; caller checked they exist

  // slot_of_ is written under the lock, except by Put, which claims an ID (kPending) and
  // publishes its slot without it; Locate and SlotOf read it without the lock. A slot is
  // published after its code is written (release), so a reader that finds it (acquire) finds the
  // code. loc_ is written under the lock and read without it by Locate.
  uint32_t SlotLocked(uint32_t id) const { return slot_of_[id].load(std::memory_order_acquire); }
  void SetSlot(uint32_t id, uint32_t s) { slot_of_[id].store(s, std::memory_order_release); }
  // The slot of id's code, or kNoSlot: none here, or a Put of it in progress (kPending).
  uint32_t Held(uint32_t id) const {
    const uint32_t s = SlotLocked(id);
    return s < capacity_ ? s : kNoSlot;
  }
  uint32_t LocLocked(uint32_t s) const { return loc_[s].load(std::memory_order_relaxed); }
  void SetLoc(uint32_t s, uint32_t loc) { loc_[s].store(loc, std::memory_order_relaxed); }

  const uint64_t n_;
  const uint32_t m_;
  const uint64_t capacity_;
  std::unique_ptr<uint8_t[]> codes_;  // capacity x m
  mutable std::mutex mu_;
  // Per vector ID; kNoSlot = no code here. Sized by the whole dataset, not by what this node
  // holds (4 GB at a billion vectors) -- the accounting cost of overlapping posting lists, and
  // the one per-node structure here that scaling out does not shrink. A sorted (id, slot) array
  // with a merge would replace it below ~50% residency. See engine/README.md, "Known limits of
  // the PQ path".
  std::unique_ptr<std::atomic<uint32_t>[]> slot_of_;
  std::vector<uint32_t> id_of_;    // per slot; kNoSlot = free
  std::vector<uint32_t> refs_;     // per slot: the resident segments that name it
  std::unique_ptr<std::atomic<uint32_t>[]> loc_;  // per slot: raw-vector location of its vector, or kNoLoc
  std::vector<uint8_t> flags_;     // per slot: kProtected | kEverReferenced | kDeleted
  std::vector<uint32_t> free_;     // freed slots, reused first
  uint64_t next_ = 0;              // slots [0, next_) have been handed out at least once
  uint64_t hand_ = 0;              // eviction sweep position
  PQStats st_;
  Mirror mirror_;
};

}  // namespace fusion
