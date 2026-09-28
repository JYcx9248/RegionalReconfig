#include "fusion/pq_store.h"

#include <algorithm>
#include <cstring>

namespace fusion {
namespace {
// Marks an ID whose code is being installed by the current Put (never visible outside it).
constexpr uint32_t kPending = 0xFFFFFFFEu;
constexpr uint8_t kProtected = 1;       // a staging in progress needs this code
constexpr uint8_t kEverReferenced = 2;  // a loaded partition has named this code
}  // namespace

PQStore::PQStore(uint64_t num_vectors, uint32_t m, uint64_t capacity)
    : n_(num_vectors), m_(m), capacity_(capacity) {
  FUSION_CHECK(m >= 1, "PQ codes need at least one byte");
  FUSION_CHECK(num_vectors < kPending, "vector IDs must fit in 32 bits");
  FUSION_CHECK(capacity >= 1 && capacity <= num_vectors, "PQ capacity %llu outside [1, %llu]",
               static_cast<unsigned long long>(capacity),
               static_cast<unsigned long long>(num_vectors));
  codes_.reset(new uint8_t[capacity * m]);
  slot_of_.assign(n_, kNoSlot);
  id_of_.assign(capacity_, kNoSlot);
  refs_.assign(capacity_, 0);
  loc_.assign(capacity_, kNoLoc);
  flags_.assign(capacity_, 0);
  free_.reserve(capacity_);  // never grows past this: freeing cannot allocate
  st_.m = m;
  st_.capacity = capacity;
}

void PQStore::SetMirror(Mirror mirror) {
  std::lock_guard<std::mutex> l(mu_);
  mirror_ = std::move(mirror);
}

PQStore::Category PQStore::CategoryOf(uint32_t s) const {
  if (refs_[s] > 0) return kLive;
  return (flags_[s] & kProtected) ? kStaged : kCached;
}

void PQStore::Count(Category c, int64_t delta) {
  uint64_t& v = c == kLive ? st_.live : c == kStaged ? st_.staged : st_.cached;
  v = static_cast<uint64_t>(static_cast<int64_t>(v) + delta);
}

std::vector<uint32_t> PQStore::Missing(const uint32_t* ids, size_t n) const {
  std::vector<uint32_t> v(ids, ids + n);
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  if (!v.empty() && v.back() >= n_)
    throw std::invalid_argument(StrFormat("vector ID %u out of range", v.back()));
  std::lock_guard<std::mutex> l(mu_);
  v.erase(std::remove_if(v.begin(), v.end(), [&](uint32_t id) { return slot_of_[id] != kNoSlot; }),
          v.end());
  return v;
}

std::vector<uint32_t> PQStore::Reserve(const uint32_t* ids, size_t n) {
  std::vector<uint32_t> v(ids, ids + n);
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  if (!v.empty() && v.back() >= n_)
    throw std::invalid_argument(StrFormat("vector ID %u out of range", v.back()));
  std::vector<uint32_t> missing;
  missing.reserve(v.size());
  std::lock_guard<std::mutex> l(mu_);
  for (uint32_t id : v) {
    const uint32_t s = slot_of_[id];
    if (s == kNoSlot) {
      missing.push_back(id);
      continue;
    }
    const Category before = CategoryOf(s);
    flags_[s] |= kProtected;
    const Category after = CategoryOf(s);
    if (before != after) {
      Count(before, -1);
      Count(after, +1);
    }
  }
  return missing;
}

size_t PQStore::Put(const uint32_t* ids, size_t n, const uint8_t* codes, PQOrigin origin) {
  std::lock_guard<std::mutex> l(mu_);
  // Everything that allocates happens before the first change, so an allocation failure
  // leaves the store as it was.
  std::vector<size_t> fresh;
  fresh.reserve(n);
  std::vector<uint32_t> slots;
  slots.reserve(n);
  std::vector<uint32_t> sorted;
  if (mirror_) sorted.reserve(n);
  // Pass 1: find the new IDs (also deduplicates within the batch) without changing anything
  // visible; undone if the batch is rejected.
  auto undo_pending = [&] {
    for (size_t i : fresh) slot_of_[ids[i]] = kNoSlot;
  };
  for (size_t i = 0; i < n; ++i) {
    const uint32_t id = ids[i];
    if (id >= n_) {
      undo_pending();
      throw std::invalid_argument(StrFormat("vector ID %u out of range", id));
    }
    if (slot_of_[id] == kNoSlot) {
      slot_of_[id] = kPending;
      fresh.push_back(i);
    }
  }
  const uint64_t free_slots = capacity_ - st_.resident;
  if (fresh.size() > free_slots) {
    const uint64_t need = fresh.size() - free_slots;
    if (need > st_.cached) {
      undo_pending();
      throw PQCapacityError(StrFormat(
          "PQ capacity exceeded: %zu new codes, %llu of %llu slots free, %llu cached codes to drop",
          fresh.size(), static_cast<unsigned long long>(free_slots),
          static_cast<unsigned long long>(capacity_), static_cast<unsigned long long>(st_.cached)));
    }
    EvictLocked(need);  // cached codes make room; live and staged ones are never dropped
  }
  // Pass 2: take slots and copy the codes.
  slots.resize(fresh.size());
  for (size_t k = 0; k < fresh.size(); ++k) {
    uint32_t s;
    if (!free_.empty()) {
      s = free_.back();
      free_.pop_back();
    } else {
      s = static_cast<uint32_t>(next_++);
    }
    const uint32_t id = ids[fresh[k]];
    slot_of_[id] = s;
    id_of_[s] = id;
    refs_[s] = 0;
    loc_[s] = kNoLoc;        // known once a partition that names it loads (Ref)
    flags_[s] = kProtected;  // staged until a partition that names it loads
    std::memcpy(codes_.get() + static_cast<uint64_t>(s) * m_, codes + fresh[k] * m_, m_);
    slots[k] = s;
  }
  if (mirror_ && !slots.empty()) {
    sorted.assign(slots.begin(), slots.end());  // within the reserved capacity
    std::sort(sorted.begin(), sorted.end());
    try {
      mirror_(sorted.data(), static_cast<uint32_t>(sorted.size()));
    } catch (...) {
      for (uint32_t s : slots) {
        slot_of_[id_of_[s]] = kNoSlot;
        id_of_[s] = kNoSlot;
        flags_[s] = 0;
        free_.push_back(s);
      }
      throw;
    }
  }
  st_.resident += fresh.size();
  st_.staged += fresh.size();
  (origin == PQOrigin::kIndex ? st_.from_index : st_.received) += fresh.size();
  st_.skipped += n - fresh.size();
  return fresh.size();
}

void PQStore::Get(const uint32_t* ids, size_t n, uint8_t* out) {
  std::lock_guard<std::mutex> l(mu_);
  for (size_t i = 0; i < n; ++i) {
    const uint32_t id = ids[i];
    if (id >= n_) throw std::invalid_argument(StrFormat("vector ID %u out of range", id));
    const uint32_t s = slot_of_[id];
    if (s == kNoSlot) throw PQMissingError(StrFormat("no PQ code for vector %u on this node", id));
    std::memcpy(out + i * m_, codes_.get() + static_cast<uint64_t>(s) * m_, m_);
  }
  st_.served += n;
}

void PQStore::Ref(const uint32_t* ids, size_t n, uint32_t* slots_out, const uint32_t* locs) {
  std::lock_guard<std::mutex> l(mu_);
  size_t missing = 0;
  uint32_t first = 0;
  for (size_t i = 0; i < n; ++i) {
    const uint32_t id = ids[i];
    if (id >= n_) throw std::invalid_argument(StrFormat("vector ID %u out of range", id));
    const uint32_t s = slot_of_[id];
    if (s == kNoSlot) {
      if (missing++ == 0) first = id;
      continue;
    }
    // A vector's location is a property of the dataset: two segments naming it must agree.
    if (locs && loc_[s] != kNoLoc && loc_[s] != locs[i])
      throw std::runtime_error(StrFormat("vector %u: a segment gives raw-vector location %u, "
                                         "another gave %u", id, locs[i], loc_[s]));
  }
  if (missing)
    throw PQMissingError(StrFormat("%zu postings name vectors whose PQ code is not on this node "
                                   "(first: vector %u)", missing, first));
  for (size_t i = 0; i < n; ++i) {
    const uint32_t s = slot_of_[ids[i]];
    const Category before = CategoryOf(s);
    ++refs_[s];
    flags_[s] = static_cast<uint8_t>((flags_[s] & ~kProtected) | kEverReferenced);
    if (locs) loc_[s] = locs[i];
    if (before != kLive) {
      Count(before, -1);
      Count(kLive, +1);
    }
    if (slots_out) slots_out[i] = s;
  }
}

size_t PQStore::Locate(const uint32_t* ids, size_t n, uint32_t* locs_out) const {
  std::lock_guard<std::mutex> l(mu_);
  size_t unknown = 0;
  for (size_t i = 0; i < n; ++i) {
    const uint32_t id = ids[i];
    const uint32_t s = id < n_ ? slot_of_[id] : kNoSlot;
    locs_out[i] = s < capacity_ ? loc_[s] : kNoLoc;  // kPending never shows outside Put
    if (locs_out[i] == kNoLoc) ++unknown;
  }
  return unknown;
}

void PQStore::Unref(const uint32_t* ids, size_t n) noexcept {
  std::lock_guard<std::mutex> l(mu_);
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    const uint32_t id = ids[i];
    const uint32_t s = id < n_ ? slot_of_[id] : kNoSlot;
    if (s >= capacity_ || refs_[s] == 0) {
      ++bad;
      continue;
    }
    if (--refs_[s] == 0) {  // kept: cached (or staged, if a staging protects it)
      Count(kLive, -1);
      Count(CategoryOf(s), +1);
    }
  }
  if (bad) Log("PQ store: %zu references to drop were not held (bug)", bad);
}

size_t PQStore::ReleaseStaged() {
  std::lock_guard<std::mutex> l(mu_);
  size_t freed = 0;
  for (uint64_t i = 0; i < next_; ++i) {
    const uint32_t s = static_cast<uint32_t>(i);
    if (id_of_[s] == kNoSlot || !(flags_[s] & kProtected)) continue;
    if (refs_[s] == 0 && !(flags_[s] & kEverReferenced)) {  // installed for the staging, unused
      FreeLocked(s);
      ++freed;
      continue;
    }
    const Category before = CategoryOf(s);  // claimed from the cache (or live): unprotect
    flags_[s] = static_cast<uint8_t>(flags_[s] & ~kProtected);
    const Category after = CategoryOf(s);
    if (before != after) {
      Count(before, -1);
      Count(after, +1);
    }
  }
  return freed;
}

void PQStore::EvictLocked(uint64_t count) {
  // Sweep the slots from where the last eviction stopped (FIFO-like order) and drop cached
  // codes until `count` are gone. The caller checked that enough cached codes exist.
  uint64_t done = 0;
  for (uint64_t visited = 0; done < count && visited < next_; ++visited) {
    const uint32_t s = static_cast<uint32_t>(hand_);
    hand_ = (hand_ + 1) % next_;
    if (id_of_[s] != kNoSlot && CategoryOf(s) == kCached) {
      FreeLocked(s);
      ++st_.evicted;
      ++done;
    }
  }
}

void PQStore::FreeLocked(uint32_t s) {
  Count(CategoryOf(s), -1);
  slot_of_[id_of_[s]] = kNoSlot;
  id_of_[s] = kNoSlot;
  refs_[s] = 0;
  loc_[s] = kNoLoc;
  flags_[s] = 0;
  free_.push_back(s);
  --st_.resident;
  ++st_.freed;
}

PQStats PQStore::Stats() const {
  std::lock_guard<std::mutex> l(mu_);
  return st_;
}

}  // namespace fusion
