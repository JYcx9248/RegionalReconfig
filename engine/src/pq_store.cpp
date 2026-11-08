#include "fusion/pq_store.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>

namespace fusion {
namespace {
// Marks an ID whose code is being installed by the current Put (never visible outside it).
constexpr uint32_t kPending = 0xFFFFFFFEu;
constexpr uint8_t kProtected = 1;       // a staging in progress needs this code
constexpr uint8_t kEverReferenced = 2;  // a loaded partition has named this code
constexpr uint8_t kDeleted = 4;         // the vector is deleted (U14): freed with its last reference

// A snapshot of the codes as Save writes it (little endian): the magic, u32 version, u32 m, u32
// the codebook's CRC-32, u32 reserved, u64 the ID space it was saved with, u64 n, the n IDs
// ascending (u32), then their codes in that order (n x m bytes). No checksum over the codes: a
// pass over gigabytes would cost much of what the snapshot saves. The size, every ID and the
// codebook are checked; rsync checks a copy.
constexpr char kSnapMagic[8] = {'R', 'T', 'P', 'Q', 'S', 'N', 'A', 'P'};
constexpr uint32_t kSnapVersion = 1;
constexpr uint64_t kSnapHeader = 8 + 4 * 4 + 8 + 8;
constexpr uint64_t kSnapChunk = uint64_t{1} << 20;  // codes a write or a read

void WriteAll(int fd, const void* data, size_t n, const std::string& path) {
  const auto* p = static_cast<const uint8_t*>(data);
  while (n > 0) {
    const ssize_t r = ::write(fd, p, n);
    if (r < 0 && errno == EINTR) continue;
    FUSION_CHECK(r > 0, "write %s: %s", path.c_str(), r == 0 ? "no progress" : std::strerror(errno));
    p += r;
    n -= static_cast<size_t>(r);
  }
}

void ReadAllAt(int fd, void* data, size_t n, uint64_t off, const std::string& path) {
  auto* p = static_cast<uint8_t*>(data);
  while (n > 0) {
    const ssize_t r = ::pread(fd, p, n, static_cast<off_t>(off));
    if (r < 0 && errno == EINTR) continue;
    FUSION_CHECK(r > 0, "read %s at %llu: %s", path.c_str(), static_cast<unsigned long long>(off),
                 r == 0 ? "unexpected end of file" : std::strerror(errno));
    p += r;
    off += static_cast<uint64_t>(r);
    n -= static_cast<size_t>(r);
  }
}
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
  switch (origin) {
    case PQOrigin::kIndex: st_.from_index += fresh.size(); break;
    case PQOrigin::kSnapshot: st_.restored += fresh.size(); break;
    case PQOrigin::kReceived: st_.received += fresh.size(); break;
  }
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
      // A deleted vector's code is no cache. FreeLocked cannot allocate (free_ is reserved).
      if ((flags_[s] & kDeleted) && CategoryOf(s) == kCached) FreeLocked(s);
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
    if (after == kCached && (flags_[s] & kDeleted)) {
      FreeLocked(s);
      ++freed;
    }
  }
  return freed;
}

size_t PQStore::Forget(const uint32_t* ids, size_t n, bool deleted) {
  std::lock_guard<std::mutex> l(mu_);
  size_t freed = 0;
  for (size_t i = 0; i < n; ++i) {
    if (ids[i] >= n_) continue;
    const uint32_t s = slot_of_[ids[i]];
    if (s == kNoSlot) continue;
    if (CategoryOf(s) == kCached) {
      FreeLocked(s);
      ++freed;
    } else if (deleted && !(flags_[s] & kDeleted)) {
      flags_[s] |= kDeleted;
      ++st_.deleted;
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
  if (flags_[s] & kDeleted) --st_.deleted;
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

uint64_t PQStore::Save(const std::string& path, uint32_t codebook_crc) const {
  std::lock_guard<std::mutex> l(mu_);
  std::vector<uint32_t> ids;
  ids.reserve(st_.resident);
  for (uint64_t id = 0; id < n_; ++id)
    if (slot_of_[id] != kNoSlot) ids.push_back(static_cast<uint32_t>(id));
  std::string tmp = path + ".tmp.XXXXXX";
  const int fd = ::mkstemp(&tmp[0]);
  FUSION_CHECK(fd >= 0, "cannot create %s: %s", tmp.c_str(), std::strerror(errno));
  try {
    ::fchmod(fd, 0644);
    uint8_t hdr[kSnapHeader];
    const uint32_t h32[4] = {kSnapVersion, m_, codebook_crc, 0};
    const uint64_t h64[2] = {n_, ids.size()};
    std::memcpy(hdr, kSnapMagic, 8);
    std::memcpy(hdr + 8, h32, sizeof(h32));
    std::memcpy(hdr + 24, h64, sizeof(h64));
    WriteAll(fd, hdr, sizeof(hdr), tmp);
    WriteAll(fd, ids.data(), ids.size() * sizeof(uint32_t), tmp);
    std::vector<uint8_t> buf;
    for (size_t i = 0; i < ids.size(); i += kSnapChunk) {
      const size_t k = std::min<size_t>(kSnapChunk, ids.size() - i);
      buf.resize(k * m_);
      for (size_t j = 0; j < k; ++j)
        std::memcpy(buf.data() + j * m_, codes_.get() + static_cast<uint64_t>(slot_of_[ids[i + j]]) * m_, m_);
      WriteAll(fd, buf.data(), buf.size(), tmp);
    }
    FUSION_CHECK(::fsync(fd) == 0, "fsync %s: %s", tmp.c_str(), std::strerror(errno));
    DropFromPageCache(fd, 0, kSnapHeader + ids.size() * (sizeof(uint32_t) + uint64_t{m_}));
  } catch (...) {
    ::close(fd);
    ::unlink(tmp.c_str());
    throw;
  }
  ::close(fd);
  RenameDurably(tmp, path);
  return ids.size();
}

uint64_t PQStore::Restore(const std::string& path, uint32_t codebook_crc) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  FUSION_CHECK(fd >= 0, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  struct Closer {
    int fd;
    ~Closer() { ::close(fd); }
  } closer{fd};
  struct stat sb {};
  FUSION_CHECK(::fstat(fd, &sb) == 0, "stat %s: %s", path.c_str(), std::strerror(errno));
  const uint64_t size = static_cast<uint64_t>(sb.st_size);
  FUSION_CHECK(size >= kSnapHeader, "%s is no PQ snapshot: %llu bytes", path.c_str(),
               static_cast<unsigned long long>(size));
  uint8_t hdr[kSnapHeader];
  ReadAllAt(fd, hdr, sizeof(hdr), 0, path);
  uint32_t h32[4];
  uint64_t h64[2];
  std::memcpy(h32, hdr + 8, sizeof(h32));
  std::memcpy(h64, hdr + 24, sizeof(h64));
  FUSION_CHECK(std::memcmp(hdr, kSnapMagic, 8) == 0 && h32[0] == kSnapVersion,
               "%s is no PQ snapshot (PQ_SAVE writes them)", path.c_str());
  FUSION_CHECK(h32[1] == m_, "%s holds %u-byte PQ codes, this index %u-byte ones", path.c_str(), h32[1], m_);
  FUSION_CHECK(h32[2] == codebook_crc, "%s holds PQ codes of another codebook than this index's", path.c_str());
  const uint64_t n = h64[1];
  FUSION_CHECK(n <= size && size == kSnapHeader + n * (sizeof(uint32_t) + uint64_t{m_}),
               "%s is damaged: %llu bytes for %llu PQ codes", path.c_str(), static_cast<unsigned long long>(size),
               static_cast<unsigned long long>(n));
  {
    std::lock_guard<std::mutex> l(mu_);
    FUSION_CHECK(st_.resident == 0, "a PQ snapshot is restored into an empty store, not one with %llu codes",
                 static_cast<unsigned long long>(st_.resident));
  }
  if (n > capacity_)
    throw PQCapacityError(StrFormat("%s holds %llu PQ codes, more than this node's capacity of %llu (--pq-capacity)",
                                    path.c_str(), static_cast<unsigned long long>(n),
                                    static_cast<unsigned long long>(capacity_)));
  std::vector<uint32_t> ids(n);
  ReadAllAt(fd, ids.data(), n * sizeof(uint32_t), kSnapHeader, path);
  DropFromPageCache(fd, kSnapHeader, n * sizeof(uint32_t));
  for (uint64_t i = 0; i < n; ++i)
    FUSION_CHECK(ids[i] < n_ && (i == 0 || ids[i] > ids[i - 1]), "%s is damaged: vector ID %u at position %llu",
                 path.c_str(), ids[i], static_cast<unsigned long long>(i));
  uint64_t installed = 0;
  std::vector<uint8_t> buf;
  for (uint64_t i = 0; i < n; i += kSnapChunk) {
    const uint64_t k = std::min(kSnapChunk, n - i), off = kSnapHeader + n * sizeof(uint32_t) + i * m_;
    buf.resize(k * m_);
    ReadAllAt(fd, buf.data(), buf.size(), off, path);
    DropFromPageCache(fd, off, buf.size());
    installed += Put(ids.data() + i, k, buf.data(), PQOrigin::kSnapshot);
  }
  return installed;
}

}  // namespace fusion
