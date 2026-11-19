#include "fusion/pq_store.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>

namespace fusion {
namespace {
// Marks an ID whose code a Put is installing: its claim. Only Puts see it (to wait for it);
// everything else reads the ID as absent (Held) until the claim ends.
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

PQStore::PQStore(uint64_t num_vectors, uint32_t m, uint64_t capacity, bool host_copy)
    : n_(num_vectors), m_(m), capacity_(capacity) {
  FUSION_CHECK(m >= 1, "PQ codes need at least one byte");
  FUSION_CHECK(num_vectors < kPending, "vector IDs must fit in 32 bits");
  FUSION_CHECK(capacity >= 1 && capacity <= num_vectors, "PQ capacity %llu outside [1, %llu]",
               static_cast<unsigned long long>(capacity),
               static_cast<unsigned long long>(num_vectors));
  if (host_copy) codes_.reset(new uint8_t[capacity * m]);
  slot_of_.reset(new std::atomic<uint32_t>[n_]);
  for (uint64_t i = 0; i < n_; ++i) slot_of_[i].store(kNoSlot, std::memory_order_relaxed);
  id_of_.assign(capacity_, kNoSlot);
  refs_.assign(capacity_, 0);
  loc_.reset(new std::atomic<uint32_t>[capacity_]);
  for (uint64_t s = 0; s < capacity_; ++s) loc_[s].store(kNoLoc, std::memory_order_relaxed);
  flags_.assign(capacity_, 0);
  free_.reserve(capacity_);  // never grows past this: freeing cannot allocate
  st_.m = m;
  st_.capacity = capacity;
}

void PQStore::SetMirror(Mirror mirror) {
  std::lock_guard<std::mutex> l(mu_);
  mirror_ = std::move(mirror);
}

void PQStore::SetReader(Reader reader) {
  std::lock_guard<std::mutex> l(mu_);
  reader_ = std::move(reader);
}

PQStore::Category PQStore::CategoryOf(uint32_t s) const {
  if (refs_[s] > 0) return kLive;
  return (flags_[s] & kProtected) ? kStaged : kCached;
}

void PQStore::Count(Category c, int64_t delta) {
  uint64_t& v = c == kLive ? st_.live : c == kStaged ? st_.staged : st_.cached;
  v = static_cast<uint64_t>(static_cast<int64_t>(v) + delta);
}

// ids[0..n) ascending, without duplicates: sorted only if they are not (a segment's distinct
// vectors come sorted).
static std::vector<uint32_t> SortedUnique(const uint32_t* ids, size_t n) {
  std::vector<uint32_t> v(ids, ids + n);
  if (std::adjacent_find(v.begin(), v.end(), [](uint32_t a, uint32_t b) { return a >= b; }) != v.end()) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
  }
  return v;
}

std::vector<uint32_t> PQStore::Missing(const uint32_t* ids, size_t n) const {
  std::vector<uint32_t> v = SortedUnique(ids, n);
  if (!v.empty() && v.back() >= n_)
    throw std::invalid_argument(StrFormat("vector ID %u out of range", v.back()));
  std::lock_guard<std::mutex> l(mu_);
  v.erase(std::remove_if(v.begin(), v.end(), [&](uint32_t id) { return Held(id) != kNoSlot; }),
          v.end());
  return v;
}

std::vector<uint32_t> PQStore::Reserve(const uint32_t* ids, size_t n) {
  std::vector<uint32_t> v = SortedUnique(ids, n);
  if (!v.empty() && v.back() >= n_)
    throw std::invalid_argument(StrFormat("vector ID %u out of range", v.back()));
  std::vector<uint32_t> missing;
  missing.reserve(v.size());
  std::lock_guard<std::mutex> l(mu_);
  for (uint32_t id : v) {
    const uint32_t s = Held(id);
    if (s == kNoSlot) {  // or a Put of it in progress: listed again, a later Put waits for it
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

// Concurrent Puts run side by side (a staging installs the codes of several sources at once):
// most of a Put is a random access per ID into slot_of_, about 30 ms for 62K codes on
// CloudLab's c240g5, and under the lock a node's PQ_PUTs went one at a time (2026-11-08). So
// each new ID is claimed with a CAS (kNoSlot -> kPending), which also leaves a repeated ID to
// its first claim, in this batch or another; the lock only hands out the slots and keeps the
// accounting; the codes, the mirror and the slots' publication happen outside it. Until its
// slot is published, a claimed ID reads as absent everywhere (Held: kNoSlot), and its
// slot -- staged, never referenced -- is left alone (EvictLocked drops only cached codes,
// ReleaseStaged only published ones).
//
// So a Put that meets another's claim waits for it: its caller references the codes next. When
// it skipped them instead (2026-11-08 to 11-17), loads of a next generation's partitions, 8 at
// once, raced wherever two lacked the same code (FillFromIndex): the one whose Put met the
// other's claim could reach Ref before that claim was published, and failed (PQMissingError; a
// full rebuild's load, now and then). A Put waits only once it has published its own claims, so
// no Put waits holding one, and no two wait for each other. A claim that ends withdrawn (its
// Put failed) is claimed in the next round.
size_t PQStore::Put(const uint32_t* ids, size_t n, const uint8_t* codes, PQOrigin origin) {
  for (size_t i = 0; i < n; ++i)
    if (ids[i] >= n_) throw std::invalid_argument(StrFormat("vector ID %u out of range", ids[i]));
  std::vector<size_t> theirs, again;
  size_t installed = PutRound(ids, codes, nullptr, n, origin, &theirs);
  while (!theirs.empty()) {
    AwaitClaims(ids, theirs);
    again.swap(theirs);
    installed += PutRound(ids, codes, again.data(), again.size(), origin, &theirs);
  }
  return installed;
}

size_t PQStore::PutRound(const uint32_t* ids, const uint8_t* codes, const size_t* pos, size_t count,
                         PQOrigin origin, std::vector<size_t>* theirs) {
  // Everything that allocates happens before the first change, so an allocation failure
  // leaves the store as it was.
  std::vector<size_t> fresh;
  fresh.reserve(count);
  std::vector<uint32_t> slots;
  slots.reserve(count);
  theirs->clear();
  theirs->reserve(count);
  for (size_t k = 0; k < count; ++k) {
    const size_t i = pos ? pos[k] : k;
    uint32_t was = kNoSlot;
    if (slot_of_[ids[i]].compare_exchange_strong(was, kPending, std::memory_order_relaxed))
      fresh.push_back(i);
    else if (was == kPending)
      theirs->push_back(i);  // another Put's (or this one's, for an ID repeated in ids)
  }
  const size_t skipped = count - fresh.size() - theirs->size();
  auto unclaim = [&] {
    for (size_t i : fresh) SetSlot(ids[i], kNoSlot);
    if (!fresh.empty()) EndClaims();
  };
  Mirror mirror;
  {
    std::lock_guard<std::mutex> l(mu_);
    try {
      mirror = mirror_;  // std::function may allocate
    } catch (...) {
      unclaim();
      throw;
    }
    if (!codes_ && !mirror && !fresh.empty()) {
      unclaim();
      throw std::logic_error("PQ store without a host copy: no mirror to put the codes on the device");
    }
    const uint64_t free_slots = capacity_ - st_.resident;
    if (fresh.size() > free_slots) {
      const uint64_t need = fresh.size() - free_slots;
      if (need > st_.cached) {
        unclaim();
        throw PQCapacityError(StrFormat(
            "PQ capacity exceeded: %zu new codes, %llu of %llu slots free, %llu cached codes to drop",
            fresh.size(), static_cast<unsigned long long>(free_slots),
            static_cast<unsigned long long>(capacity_), static_cast<unsigned long long>(st_.cached)));
      }
      EvictLocked(need);  // cached codes make room; live and staged ones are never dropped
    }
    slots.resize(fresh.size());
    for (size_t k = 0; k < fresh.size(); ++k) {
      uint32_t s;
      if (!free_.empty()) {
        s = free_.back();
        free_.pop_back();
      } else {
        s = static_cast<uint32_t>(next_++);
      }
      SetLoc(s, kNoLoc);  // known once a partition that names it loads (Ref)
      id_of_[s] = ids[fresh[k]];
      refs_[s] = 0;
      flags_[s] = kProtected;  // staged until a partition that names it loads
      slots[k] = s;
    }
    st_.resident += fresh.size();
    st_.staged += fresh.size();
    switch (origin) {
      case PQOrigin::kIndex: st_.from_index += fresh.size(); break;
      case PQOrigin::kSnapshot: st_.restored += fresh.size(); break;
      case PQOrigin::kReceived: st_.received += fresh.size(); break;
    }
    st_.skipped += skipped;
  }
  if (codes_)
    for (size_t k = 0; k < fresh.size(); ++k)
      std::memcpy(codes_.get() + static_cast<uint64_t>(slots[k]) * m_, codes + fresh[k] * m_, m_);
  if (mirror && !slots.empty()) {
    try {
      // The codes in slot order, as the mirror takes them: without a host copy, their only way
      // to the device.
      std::vector<size_t> order(slots.size());
      for (size_t k = 0; k < order.size(); ++k) order[k] = k;
      std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return slots[a] < slots[b]; });
      std::vector<uint32_t> sorted(slots.size());
      std::vector<uint8_t> packed(slots.size() * m_);
      for (size_t j = 0; j < order.size(); ++j) {
        sorted[j] = slots[order[j]];
        std::memcpy(packed.data() + j * m_, codes + fresh[order[j]] * m_, m_);
      }
      mirror(sorted.data(), static_cast<uint32_t>(sorted.size()), packed.data());
    } catch (...) {
      {
        std::lock_guard<std::mutex> l(mu_);
        for (size_t k = 0; k < fresh.size(); ++k) {
          const uint32_t s = slots[k];
          SetSlot(ids[fresh[k]], kNoSlot);
          id_of_[s] = kNoSlot;
          flags_[s] = 0;
          free_.push_back(s);  // within its reserved capacity
        }
        st_.resident -= fresh.size();
        st_.staged -= fresh.size();
        switch (origin) {
          case PQOrigin::kIndex: st_.from_index -= fresh.size(); break;
          case PQOrigin::kSnapshot: st_.restored -= fresh.size(); break;
          case PQOrigin::kReceived: st_.received -= fresh.size(); break;
        }
        st_.skipped -= skipped;
      }
      EndClaims();
      throw;
    }
  }
  // Last: a reader that finds a slot (SlotLocked, Held, SlotOf: acquire) finds its code written.
  for (size_t k = 0; k < fresh.size(); ++k) SetSlot(ids[fresh[k]], slots[k]);
  if (!fresh.empty()) EndClaims();
  return fresh.size();
}

// A waiter checks the claims under claim_mu_ and sleeps on claim_cv_ without letting it go in
// between; the ender changes the claims before it takes claim_mu_, then notifies. So either the
// waiter sees them ended, or it is asleep by the time the ender has taken the lock, and wakes.
void PQStore::EndClaims() noexcept {
  { std::lock_guard<std::mutex> l(claim_mu_); }
  claim_cv_.notify_all();
}

void PQStore::AwaitClaims(const uint32_t* ids, const std::vector<size_t>& pos) {
  std::unique_lock<std::mutex> l(claim_mu_);
  claim_cv_.wait(l, [&] {
    for (size_t i : pos)
      if (slot_of_[ids[i]].load(std::memory_order_acquire) == kPending) return false;
    return true;
  });
}

void PQStore::Get(const uint32_t* ids, size_t n, uint8_t* out) {
  std::vector<uint32_t> slots(codes_ ? 0 : n);  // without a host copy: read from the device
  std::lock_guard<std::mutex> l(mu_);
  for (size_t i = 0; i < n; ++i) {
    const uint32_t id = ids[i];
    if (id >= n_) throw std::invalid_argument(StrFormat("vector ID %u out of range", id));
    const uint32_t s = Held(id);
    if (s == kNoSlot) throw PQMissingError(StrFormat("no PQ code for vector %u on this node", id));
    if (codes_)
      std::memcpy(out + i * m_, codes_.get() + static_cast<uint64_t>(s) * m_, m_);
    else
      slots[i] = s;
  }
  if (!codes_ && n > 0) {
    FUSION_CHECK(reader_, "PQ store without a host copy: no reader to read the codes from the device");
    reader_(slots.data(), static_cast<uint32_t>(n), out);
  }
  st_.served += n;
}

void PQStore::Ref(const uint32_t* ids, size_t n, uint32_t* slots_out, const uint32_t* locs) {
  for (size_t i = 0; i < n; ++i) {
    if (ids[i] >= n_) throw std::invalid_argument(StrFormat("vector ID %u out of range", ids[i]));
    if (i > 0 && ids[i] <= ids[i - 1])
      throw std::invalid_argument("PQ references: vector IDs must be ascending, without duplicates");
  }
  // In ID order, the slots of a segment's vectors follow the order their codes were installed
  // in (a staging's Puts come sorted), so both passes walk the arrays rather than jump.
  std::lock_guard<std::mutex> l(mu_);
  size_t missing = 0;
  uint32_t first = 0;
  for (size_t i = 0; i < n; ++i) {
    const uint32_t id = ids[i];
    const uint32_t s = Held(id);
    if (s == kNoSlot) {
      if (missing++ == 0) first = id;
      continue;
    }
    // A vector's location is a property of the dataset: two segments naming it must agree.
    if (locs && LocLocked(s) != kNoLoc && LocLocked(s) != locs[i])
      throw std::runtime_error(StrFormat("vector %u: a segment gives raw-vector location %u, "
                                         "another gave %u", id, locs[i], LocLocked(s)));
  }
  if (missing)
    throw PQMissingError(StrFormat("%zu vectors a segment names have no PQ code on this node "
                                   "(first: vector %u)", missing, first));
  for (size_t i = 0; i < n; ++i) {
    const uint32_t s = Held(ids[i]);  // published: a slot only goes away under the lock
    const Category before = CategoryOf(s);
    ++refs_[s];
    flags_[s] = static_cast<uint8_t>((flags_[s] & ~kProtected) | kEverReferenced);
    if (locs) SetLoc(s, locs[i]);
    if (before != kLive) {
      Count(before, -1);
      Count(kLive, +1);
    }
    if (slots_out) slots_out[i] = s;
  }
}

uint32_t PQStore::SlotOf(uint32_t id) const {
  if (id >= n_) return kNoSlot;
  return Held(id);
}

size_t PQStore::Locate(const uint32_t* ids, size_t n, uint32_t* locs_out) const {
  size_t unknown = 0;
  for (size_t i = 0; i < n; ++i) {
    const uint32_t s = SlotOf(ids[i]);
    locs_out[i] = s != kNoSlot ? LocLocked(s) : kNoLoc;
    if (locs_out[i] == kNoLoc) ++unknown;
  }
  return unknown;
}

void PQStore::Unref(const uint32_t* slots, size_t n) noexcept {
  std::lock_guard<std::mutex> l(mu_);
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    const uint32_t s = slots[i];
    if (s >= capacity_ || id_of_[s] == kNoSlot || refs_[s] == 0) {
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
    if (SlotLocked(id_of_[s]) != s) continue;  // a Put's, not published yet: it ends after this
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
    const uint32_t s = Held(ids[i]);
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
  SetSlot(id_of_[s], kNoSlot);
  id_of_[s] = kNoSlot;
  refs_[s] = 0;
  SetLoc(s, kNoLoc);
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
    if (Held(static_cast<uint32_t>(id)) != kNoSlot) ids.push_back(static_cast<uint32_t>(id));
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
    std::vector<uint32_t> slots;  // without a host copy: read from the device
    for (size_t i = 0; i < ids.size(); i += kSnapChunk) {
      const size_t k = std::min<size_t>(kSnapChunk, ids.size() - i);
      buf.resize(k * m_);
      if (codes_) {
        for (size_t j = 0; j < k; ++j)
          std::memcpy(buf.data() + j * m_, codes_.get() + static_cast<uint64_t>(Held(ids[i + j])) * m_, m_);
      } else {
        FUSION_CHECK(reader_, "PQ store without a host copy: no reader to read the codes from the device");
        slots.resize(k);
        for (size_t j = 0; j < k; ++j) slots[j] = Held(ids[i + j]);
        reader_(slots.data(), static_cast<uint32_t>(k), buf.data());
      }
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
