// Node-level PQ code store of an rtier data node (the decided part of design question U1).
//
// A node keeps exactly one copy of each vector's PQ code, however many of its resident posting
// lists -- of however many partitions -- contain the vector. (FusionANNS replicates boundary
// vectors into up to 8 lists, so partitions share many vectors; storing codes per partition
// would duplicate them in the scarce GPU memory.) Codes live in fixed-size slots of a buffer
// sized to the node's budget (`capacity` codes: the HBM budget); an ID -> slot map finds them.
//
// Every code is in one of three states:
//   live    named by at least one posting of a resident partition (one reference per posting);
//   staged  protected for a staging in progress: installed by Put, or already here and claimed
//           by Reserve, and not yet named by a loaded partition;
//   cached  no longer named by a resident partition (its partition moved away). The dataset is
//           static, so the code stays valid and is kept: the node held it before, and a later
//           reconfiguration may bring the partition back. Cached codes are only dropped when a
//           Put would otherwise exceed the capacity. (Invalidating them when the dataset
//           changes is open question U14.)
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
// without taking the store's lock (the slots of a segment are resolved once, when it loads).
//
// The store is also where the node looks up a vector's raw-vector location (fusion/layout.h,
// RawLayout): the segments carry it next to every posting, and Ref records it with the code.
// It stays known while the code is here -- live or cached, the dataset is static -- so a node
// needs no vector -> page map of its own (U1). A code installed by Put has no location until a
// partition that names the vector loads.
#pragma once

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
enum class PQOrigin { kReceived, kIndex };

struct PQStats {
  uint32_t m = 0;          // bytes per code
  uint64_t capacity = 0;   // slots (the node's PQ budget, in codes)
  uint64_t resident = 0;   // codes held now = live + staged + cached
  uint64_t live = 0;       // named by resident partitions
  uint64_t staged = 0;     // protected for a staging in progress
  uint64_t cached = 0;     // kept after their partitions moved away
  uint64_t received = 0;   // cumulative: codes installed from peers (PQ_PUT)
  uint64_t from_index = 0; // cumulative: codes installed from the index files
  uint64_t skipped = 0;    // cumulative: codes offered that were already held
  uint64_t evicted = 0;    // cumulative: cached codes dropped to make room
  uint64_t freed = 0;      // cumulative: codes dropped (evicted, or released after a staging)
  uint64_t served = 0;     // cumulative: codes copied out for peers (PQ_GET)
};

class PQStore {
 public:
  static constexpr uint32_t kNoSlot = 0xFFFFFFFFu;
  static constexpr uint32_t kNoLoc = 0xFFFFFFFFu;
  // Called (under the store's lock) after Put wrote new codes into `slots` (ascending) of
  // codes(); the GPU filter copies them into device memory. If it throws, the Put is undone.
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
  // Installs n codes (m bytes each, in the order of ids) as staged. IDs already held are
  // skipped (their code is kept). Cached codes are dropped if the new ones do not fit
  // otherwise; if they still do not fit, throws PQCapacityError and changes nothing. Returns
  // the number of codes installed.
  size_t Put(const uint32_t* ids, size_t n, const uint8_t* codes, PQOrigin origin);
  // Copies the codes of ids[0..n) into out (n x m bytes). Throws PQMissingError if one is absent.
  void Get(const uint32_t* ids, size_t n, uint8_t* out);
  // Adds one reference per entry of ids[0..n) (the codes become live) and writes each entry's
  // slot to slots_out (may be null). locs (may be null): each entry's raw-vector location,
  // recorded with the code. Throws PQMissingError if a code is absent, and std::runtime_error
  // if a location contradicts the one recorded before; either way nothing changes.
  void Ref(const uint32_t* ids, size_t n, uint32_t* slots_out, const uint32_t* locs = nullptr);
  // Writes the raw-vector location of ids[0..n) to locs_out: kNoLoc where it is not known (no
  // code here, or no loaded partition has named the vector since its code arrived). Returns how
  // many are not known.
  size_t Locate(const uint32_t* ids, size_t n, uint32_t* locs_out) const;
  // Drops one reference per entry; a code with no reference left becomes cached (or stays
  // staged if a staging protects it). Never frees anything and never throws (it runs from
  // the segment reclaimer); inconsistencies are logged.
  void Unref(const uint32_t* ids, size_t n) noexcept;
  // Ends a staging that did not load (rollback): frees staged codes no partition ever named
  // and returns the claimed ones to the cache. Returns how many codes were freed.
  size_t ReleaseStaged();

  PQStats Stats() const;

 private:
  enum Category { kLive, kStaged, kCached };
  Category CategoryOf(uint32_t s) const;
  void Count(Category c, int64_t delta);
  void FreeLocked(uint32_t slot);
  void EvictLocked(uint64_t count);  // drops `count` cached codes; caller checked they exist

  const uint64_t n_;
  const uint32_t m_;
  const uint64_t capacity_;
  std::unique_ptr<uint8_t[]> codes_;  // capacity x m
  mutable std::mutex mu_;
  // Per vector ID; kNoSlot = no code here. Sized by the whole dataset, not by what this node
  // holds (4 GB at a billion vectors) -- the accounting cost of overlapping posting lists, and
  // the one per-node structure here that scaling out does not shrink. Staging path only; a
  // sorted (id, slot) array with a merge would replace it below ~50% residency. See
  // engine/README.md, "Known limits of the PQ path".
  std::vector<uint32_t> slot_of_;
  std::vector<uint32_t> id_of_;    // per slot; kNoSlot = free
  std::vector<uint32_t> refs_;     // per slot
  std::vector<uint32_t> loc_;      // per slot: raw-vector location of its vector, or kNoLoc
  std::vector<uint8_t> flags_;     // per slot: kProtected | kEverReferenced
  std::vector<uint32_t> free_;     // freed slots, reused first
  uint64_t next_ = 0;              // slots [0, next_) have been handed out at least once
  uint64_t hand_ = 0;              // eviction sweep position
  PQStats st_;
  Mirror mirror_;
};

}  // namespace fusion
