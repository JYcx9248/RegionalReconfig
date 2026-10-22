#include "fusion/raw_store.h"

#include <fcntl.h>
#include <linux/aio_abi.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>

#include "fusion/partition.h"  // Crc32
#include "fusion/read_budget.h"

namespace fusion {
namespace {

// The presence bits as SaveBitmap writes them (little endian): the magic, u32 version, the page
// geometry (u32 vec_bytes, u32 vectors_per_page, u32 page_size, u64 num_pages), u64 words, the
// words, then a CRC-32 of every byte before it.
constexpr char kBitsMagic[8] = {'R', 'T', 'R', 'A', 'W', 'B', 'I', 'T'};
constexpr uint32_t kBitsVersion = 1;
constexpr size_t kBitsHeader = 8 + 4 * 4 + 8 + 8;

constexpr uint32_t kChunkPages = 256;  // pages per round of direct I/O (1 MB at 4 KB pages)
constexpr uint32_t kIoDepth = 16;      // installs are background work: a modest queue depth

void PwriteAll(int fd, const uint8_t* p, size_t n, uint64_t off, const std::string& path) {
  size_t done = 0;
  while (done < n) {
    const ssize_t r = ::pwrite(fd, p + done, n - done, static_cast<off_t>(off + done));
    if (r < 0 && errno == EINTR) continue;
    FUSION_CHECK(r > 0, "write %s at %llu: %s", path.c_str(),
                 static_cast<unsigned long long>(off + done), r == 0 ? "no progress" : std::strerror(errno));
    done += static_cast<size_t>(r);
  }
}

void PreadAll(int fd, uint8_t* p, size_t n, uint64_t off, const std::string& path) {
  size_t done = 0;
  while (done < n) {
    const ssize_t r = ::pread(fd, p + done, n - done, static_cast<off_t>(off + done));
    if (r < 0 && errno == EINTR) continue;
    FUSION_CHECK(r > 0, "read %s at %llu: %s", path.c_str(),
                 static_cast<unsigned long long>(off + done),
                 r == 0 ? "unexpected end of file" : std::strerror(errno));
    done += static_cast<size_t>(r);
  }
}

// Reads n bytes only if the page cache holds them all (RWF_NOWAIT fails with EAGAIN instead of
// reading the device): a read budget charges only the reads that miss.
bool PreadCached(int fd, uint8_t* p, size_t n, uint64_t off) {
  iovec iov{p, n};
  return ::preadv2(fd, &iov, 1, static_cast<off_t>(off), RWF_NOWAIT) == static_cast<ssize_t>(n);
}

// Device reads one Get keeps in flight: enough for a query's misses at one holder (at most its N
// candidates) to go out together, and a bound on what one stream batch puts on the device.
constexpr size_t kReadsInFlight = 256;

// The presence bits saved at path, as many words as a store of layout L has: refused if they
// are of another page geometry, or mark a location L has no room for (saved with more pages).
std::vector<uint64_t> LoadBits(const std::string& path, const RawLayout& L) {
  const std::vector<uint8_t> b = ReadFile(path);
  FUSION_CHECK(b.size() >= kBitsHeader + 4 && std::memcmp(b.data(), kBitsMagic, 8) == 0,
               "%s holds no presence bits", path.c_str());
  uint32_t crc = 0;
  std::memcpy(&crc, b.data() + b.size() - 4, 4);
  FUSION_CHECK(Crc32(b.data(), b.size() - 4) == crc, "%s: checksum mismatch", path.c_str());
  uint32_t hdr[4];
  uint64_t pages = 0, words = 0;
  std::memcpy(hdr, b.data() + 8, sizeof(hdr));
  std::memcpy(&pages, b.data() + 24, 8);
  std::memcpy(&words, b.data() + 32, 8);
  FUSION_CHECK(hdr[0] == kBitsVersion, "%s: presence bits of version %u, expected %u", path.c_str(), hdr[0],
               kBitsVersion);
  FUSION_CHECK(hdr[1] == L.vec_bytes && hdr[2] == L.vectors_per_page && hdr[3] == L.page_size,
               "%s: presence bits of %u-byte vectors, %u to a %u-byte page; this node's are %u-byte, %u to a "
               "%u-byte page",
               path.c_str(), hdr[1], hdr[2], hdr[3], L.vec_bytes, L.vectors_per_page, L.page_size);
  FUSION_CHECK(words == CeilDiv(pages * hdr[2], 64) && b.size() == kBitsHeader + words * 8 + 4,
               "%s is truncated", path.c_str());
  std::vector<uint64_t> out(CeilDiv(L.locations(), 64), 0);
  const uint64_t keep = std::min<uint64_t>(words, out.size());
  std::memcpy(out.data(), b.data() + kBitsHeader, keep * 8);
  bool beyond = false;
  for (uint64_t w = keep; w < words; ++w) {
    uint64_t v = 0;
    std::memcpy(&v, b.data() + kBitsHeader + w * 8, 8);
    beyond |= v != 0;
  }
  if (const uint64_t tail = L.locations() % 64; tail && keep == out.size()) {
    beyond |= (out.back() >> tail) != 0;
  }
  FUSION_CHECK(!beyond, "%s marks vectors past this node's %llu locations", path.c_str(),
               static_cast<unsigned long long>(L.locations()));
  return out;
}

// Where the last vector the bits mark ends in the file (0: they mark none).
uint64_t EndOfLast(const std::vector<uint64_t>& bits, const RawLayout& L) {
  for (size_t w = bits.size(); w-- > 0;)
    if (bits[w]) return L.offset(static_cast<uint32_t>(w * 64 + 63 - __builtin_clzll(bits[w]))) + L.vec_bytes;
  return 0;
}

// Writes path at once: a temporary file of its own (two writers at once each rename a whole
// file), synced, renamed over it.
void WriteDurably(const std::string& path, const std::vector<uint8_t>& data) {
  std::string tmp = path + ".tmp.XXXXXX";
  const int fd = ::mkstemp(&tmp[0]);
  FUSION_CHECK(fd >= 0, "cannot create %s: %s", tmp.c_str(), std::strerror(errno));
  ::fchmod(fd, 0644);
  try {
    PwriteAll(fd, data.data(), data.size(), 0, tmp);
    FUSION_CHECK(::fsync(fd) == 0, "fsync %s: %s", tmp.c_str(), std::strerror(errno));
  } catch (...) {
    ::close(fd);
    ::unlink(tmp.c_str());
    throw;
  }
  ::close(fd);
  FUSION_CHECK(std::rename(tmp.c_str(), path.c_str()) == 0, "rename %s: %s", tmp.c_str(), std::strerror(errno));
  const size_t slash = path.rfind('/');
  const std::string dir = slash == std::string::npos ? "." : slash == 0 ? "/" : path.substr(0, slash);
  const int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (dfd >= 0) {  // the rename itself, where the file system takes a directory sync
    ::fsync(dfd);
    ::close(dfd);
  }
}

// Visits the entries of order (indices into locs, sorted by location) in runs of consecutive
// locations on one page -- contiguous bytes in the file: fn(first, count) with positions into
// order.
template <class F>
void ForEachRun(const std::vector<size_t>& order, const uint32_t* locs, const RawLayout& L, F fn) {
  for (size_t i = 0; i < order.size();) {
    size_t j = i + 1;
    while (j < order.size() && locs[order[j]] == locs[order[j - 1]] + 1 &&
           L.page(locs[order[j]]) == L.page(locs[order[i]]))
      ++j;
    fn(i, j - i);
    i = j;
  }
}

}  // namespace

// Batched direct page reads and writes on dfd_: Linux AIO, or pread/pwrite if it is not
// available.
class RawStore::DirectIo {
 public:
  DirectIo(int fd, uint32_t page_size) : fd_(fd), ps_(page_size) {
    if (syscall(__NR_io_setup, kIoDepth, &ctx_) < 0) ctx_ = 0;  // synchronous fallback
    iocbs_.resize(kIoDepth);
    ptrs_.resize(kIoDepth);
    events_.resize(kIoDepth);
  }
  ~DirectIo() {
    if (ctx_) syscall(__NR_io_destroy, ctx_);
  }

  // Reads or writes pages[i] from/to bufs[i] (page_size bytes each, page-aligned).
  void Run(bool write, const uint32_t* pages, uint8_t* const* bufs, uint32_t count,
           const std::string& path) {
    if (!ctx_) {
      for (uint32_t i = 0; i < count; ++i) {
        const uint64_t off = static_cast<uint64_t>(pages[i]) * ps_;
        if (write) {
          PwriteAll(fd_, bufs[i], ps_, off, path);
        } else {
          PreadAll(fd_, bufs[i], ps_, off, path);
        }
      }
      return;
    }
    for (uint32_t done = 0; done < count;) {
      const uint32_t batch = std::min(kIoDepth, count - done);
      for (uint32_t j = 0; j < batch; ++j) {
        iocb& cb = iocbs_[j];
        std::memset(&cb, 0, sizeof(cb));
        cb.aio_fildes = static_cast<uint32_t>(fd_);
        cb.aio_lio_opcode = write ? IOCB_CMD_PWRITE : IOCB_CMD_PREAD;
        cb.aio_buf = reinterpret_cast<uint64_t>(bufs[done + j]);
        cb.aio_nbytes = ps_;
        cb.aio_offset = static_cast<int64_t>(pages[done + j]) * ps_;
        cb.aio_data = done + j;
        ptrs_[j] = &cb;
      }
      for (uint32_t sub = 0; sub < batch;) {
        const long r = syscall(__NR_io_submit, ctx_, static_cast<long>(batch - sub), ptrs_.data() + sub);
        if (r < 0) {
          if (errno == EINTR || errno == EAGAIN) continue;
          FUSION_CHECK(false, "io_submit on %s: %s", path.c_str(), std::strerror(errno));
        }
        sub += static_cast<uint32_t>(r);
      }
      for (uint32_t got = 0; got < batch;) {
        const long r = syscall(__NR_io_getevents, ctx_, static_cast<long>(batch - got),
                               static_cast<long>(batch - got), events_.data(), nullptr);
        if (r < 0) {
          if (errno == EINTR) continue;
          FUSION_CHECK(false, "io_getevents on %s: %s", path.c_str(), std::strerror(errno));
        }
        for (long e = 0; e < r; ++e) {
          const long long res = static_cast<long long>(events_[e].res);
          const uint32_t page = pages[static_cast<uint32_t>(events_[e].data)];
          FUSION_CHECK(res == static_cast<long long>(ps_), "%s of page %u of %s: %s",
                       write ? "write" : "read", page, path.c_str(),
                       res < 0 ? std::strerror(static_cast<int>(-res)) : "short transfer");
        }
        got += static_cast<uint32_t>(r);
      }
      done += batch;
    }
  }

 private:
  int fd_;
  uint32_t ps_;
  aio_context_t ctx_ = 0;
  std::vector<iocb> iocbs_;
  std::vector<iocb*> ptrs_;
  std::vector<io_event> events_;
};

RawStore::RawStore(const std::string& path, const RawLayout& layout, bool direct, bool reuse)
    : layout_(layout), path_(path), anonymous_(path.empty()) {
  FUSION_CHECK(layout.vec_bytes > 0 && layout.vectors_per_page >= 1 &&
                   static_cast<uint64_t>(layout.vec_bytes) * layout.vectors_per_page <= layout.page_size,
               "bad raw-vector page geometry");
  FUSION_CHECK(layout.locations() < kInvalidId, "too many raw-vector locations");
  FUSION_CHECK(!(reuse && anonymous_), "raw vectors to reuse need their file");
  std::vector<uint64_t> words;  // reuse: the saved presence bits
  if (anonymous_) {
    const char* tmp = std::getenv("TMPDIR");
    std::string tmpl = std::string(tmp && *tmp ? tmp : "/tmp") + "/rtier-raw-XXXXXX";
    fd_ = ::mkstemp(&tmpl[0]);
    FUSION_CHECK(fd_ >= 0, "cannot create a temporary raw-vector file: %s", std::strerror(errno));
    path_ = tmpl;
  } else if (reuse) {
    words = LoadBits(BitmapPath(path_), layout);
    fd_ = ::open(path_.c_str(), O_RDWR | O_CLOEXEC);
    FUSION_CHECK(fd_ >= 0, "cannot open %s: %s", path_.c_str(), std::strerror(errno));
    // A copy cut short would read as zeros where its bits say vectors are.
    struct stat sb;
    const uint64_t need = EndOfLast(words, layout);
    if (::fstat(fd_, &sb) != 0 || static_cast<uint64_t>(sb.st_size) < need) {
      ::close(fd_);
      FUSION_CHECK(false, "%s ends before the last vector its presence bits mark (at byte %llu)", path_.c_str(),
                   static_cast<unsigned long long>(need));
    }
  } else {
    fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    FUSION_CHECK(fd_ >= 0, "cannot create %s: %s", path_.c_str(), std::strerror(errno));
    const std::string bits = BitmapPath(path_);
    if (::unlink(bits.c_str()) != 0 && errno != ENOENT) {
      const std::string err = std::strerror(errno);
      ::close(fd_);
      FUSION_CHECK(false, "cannot remove %s, the presence bits of the vectors %s held: %s", bits.c_str(),
                   path_.c_str(), err.c_str());
    }
  }
  // Sparse: only the vectors written take space. A reused file may have been saved with room
  // for more pages, or fewer: its bits mark none of those beyond this store's.
  const uint64_t bytes = layout.num_pages * layout.page_size;
  if (::ftruncate(fd_, static_cast<off_t>(bytes)) != 0) {
    const std::string err = std::strerror(errno);
    ::close(fd_);
    if (anonymous_) ::unlink(path_.c_str());
    FUSION_CHECK(false, "cannot size %s to %llu bytes: %s", path_.c_str(),
                 static_cast<unsigned long long>(bytes), err.c_str());
  }
  try {
    pages_ = std::make_unique<PageFile>(path_, layout.page_size, direct);
  } catch (...) {
    ::close(fd_);
    if (anonymous_) ::unlink(path_.c_str());
    throw;
  }
  if (pages_->direct() && layout.page_size % 4096 == 0) {
    dfd_ = ::open(path_.c_str(), O_RDWR | O_DIRECT | O_CLOEXEC);  // -1: buffered installs
    if (dfd_ >= 0) {
      void* p = nullptr;
      if (::posix_memalign(&p, 4096, static_cast<size_t>(kChunkPages) * layout.page_size) != 0) {
        ::close(dfd_);
        dfd_ = -1;
      } else {
        chunk_ = static_cast<uint8_t*>(p);
        io_ = std::make_unique<DirectIo>(dfd_, layout.page_size);
      }
    }
  }
  if (anonymous_) ::unlink(path_.c_str());  // both descriptors keep it alive until we exit
  const uint64_t nwords = CeilDiv(layout.locations(), 64);
  bits_.reset(new std::atomic<uint64_t>[nwords]);
  uint64_t present = 0;
  for (uint64_t i = 0; i < nwords; ++i) {
    const uint64_t w = i < words.size() ? words[i] : 0;
    bits_[i].store(w, std::memory_order_relaxed);
    present += static_cast<uint64_t>(__builtin_popcountll(w));
  }
  present_.store(present, std::memory_order_relaxed);
  restored_ = present;
}

uint64_t RawStore::SaveBitmap() {
  FUSION_CHECK(!anonymous_, "the raw vectors are in an anonymous file (no --raw-file): nothing to save them with");
  // The bits first, then the sync: a bit is set only once its vector's write returned, so the
  // sync puts every vector they mark on the device.
  const std::vector<uint64_t> words = Bitmap();
  FUSION_CHECK(::fsync(fd_) == 0, "fsync %s: %s", path_.c_str(), std::strerror(errno));
  std::vector<uint8_t> b(kBitsHeader + words.size() * 8 + 4);
  const uint32_t hdr[4] = {kBitsVersion, layout_.vec_bytes, layout_.vectors_per_page, layout_.page_size};
  const uint64_t pages = layout_.num_pages, nwords = words.size();
  std::memcpy(b.data(), kBitsMagic, 8);
  std::memcpy(b.data() + 8, hdr, sizeof(hdr));
  std::memcpy(b.data() + 24, &pages, 8);
  std::memcpy(b.data() + 32, &nwords, 8);
  std::memcpy(b.data() + kBitsHeader, words.data(), words.size() * 8);
  const uint32_t crc = Crc32(b.data(), b.size() - 4);
  std::memcpy(b.data() + b.size() - 4, &crc, 4);
  WriteDurably(BitmapPath(path_), b);
  uint64_t n = 0;
  for (uint64_t w : words) n += static_cast<uint64_t>(__builtin_popcountll(w));
  return n;
}

RawStore::~RawStore() {
  io_.reset();
  std::free(chunk_);
  if (dfd_ >= 0) ::close(dfd_);
  pages_.reset();
  if (fd_ >= 0) ::close(fd_);
}

void RawStore::CheckRange(const uint32_t* locs, size_t n) const {
  const uint64_t total = layout_.locations();
  for (size_t i = 0; i < n; ++i)
    if (locs[i] >= total)
      throw std::invalid_argument(StrFormat("raw-vector location %u out of range (%llu locations)",
                                            locs[i], static_cast<unsigned long long>(total)));
}

std::vector<uint32_t> RawStore::Missing(const uint32_t* locs, size_t n) const {
  CheckRange(locs, n);
  std::vector<uint32_t> v;
  for (size_t i = 0; i < n; ++i)
    if (!Present(locs[i])) v.push_back(locs[i]);
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  return v;
}

size_t RawStore::Put(const uint32_t* locs, size_t n, const uint8_t* vecs, RawOrigin origin) {
  CheckRange(locs, n);
  std::lock_guard<std::mutex> l(put_mu_);
  // The new ones, by location (duplicates in the batch once), so that neighbours on a page go
  // out in one write. Only Put sets bits, so what is absent now stays absent until it is done.
  std::vector<size_t> order;
  order.reserve(n);
  for (size_t i = 0; i < n; ++i)
    if (!Present(locs[i])) order.push_back(i);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return locs[a] < locs[b]; });
  order.erase(std::unique(order.begin(), order.end(),
                          [&](size_t a, size_t b) { return locs[a] == locs[b]; }),
              order.end());
  // Fetched and inserted vectors -- the engine's writer, a few scattered vectors per page -- go
  // out as whole pages with O_DIRECT. Bulk installs (bootstrap, streams) come in sorted batches that fill a
  // page over several requests: buffered, the page cache combines them per page, where whole
  // pages would rewrite, and first read back, a page per request (measured: a 64 KB stream
  // request 3.5-4.6 ms, bootstrap 3x slower).
  if (dfd_ >= 0 && (origin == RawOrigin::kFetched || origin == RawOrigin::kAdded)) {
    PutPages(locs, vecs, order);
  } else {
    PutBuffered(locs, vecs, order);
  }
  size_t installed = 0;
  for (size_t i : order) {
    const uint32_t loc = locs[i];
    const uint64_t bit = uint64_t{1} << (loc & 63);
    if (!(bits_[loc >> 6].fetch_or(bit, std::memory_order_release) & bit)) ++installed;
  }
  present_.fetch_add(installed, std::memory_order_relaxed);
  std::atomic<uint64_t>& by = origin == RawOrigin::kIndex      ? from_index_
                              : origin == RawOrigin::kStreamed ? streamed_
                              : origin == RawOrigin::kAdded    ? added_
                                                               : fetched_;
  by.fetch_add(installed, std::memory_order_relaxed);
  skipped_.fetch_add(n - installed, std::memory_order_relaxed);
  return installed;
}

// Writes the vectors of order (positions into locs/vecs, sorted by location), whole pages at a
// time. Their bits are set by the caller once every write returned.
void RawStore::PutPages(const uint32_t* locs, const uint8_t* vecs, const std::vector<size_t>& order) {
  const uint32_t vb = layout_.vec_bytes, ps = layout_.page_size, vpp = layout_.vectors_per_page;
  std::vector<uint32_t> pages, reads;
  std::vector<size_t> first;  // pages[k]'s vectors: order[first[k] .. first[k + 1])
  std::vector<uint8_t*> bufs, read_bufs;
  for (size_t i = 0; i < order.size();) {
    pages.clear();
    first.clear();
    size_t j = i;
    while (j < order.size() && pages.size() < kChunkPages) {
      const uint32_t p = layout_.page(locs[order[j]]);
      pages.push_back(p);
      first.push_back(j);
      while (j < order.size() && layout_.page(locs[order[j]]) == p) ++j;
    }
    first.push_back(j);
    // A page with a vector here already is read back, so that the write keeps it.
    reads.clear();
    read_bufs.clear();
    bufs.resize(pages.size());
    for (size_t k = 0; k < pages.size(); ++k) {
      bufs[k] = chunk_ + k * ps;
      const uint32_t base = pages[k] * vpp;
      bool held = false;
      for (uint32_t s = 0; s < vpp && !held; ++s) held = Present(base + s);
      if (held) {
        reads.push_back(pages[k]);
        read_bufs.push_back(bufs[k]);
      } else {
        std::memset(bufs[k], 0, ps);
      }
    }
    if (!reads.empty()) {
      if (budget_) budget_->Charge(reads.size());
      io_->Run(false, reads.data(), read_bufs.data(), static_cast<uint32_t>(reads.size()), path_);
    }
    for (size_t k = 0; k < pages.size(); ++k)
      for (size_t e = first[k]; e < first[k + 1]; ++e)
        std::memcpy(bufs[k] + static_cast<size_t>(layout_.slot(locs[order[e]])) * vb, vecs + order[e] * vb, vb);
    io_->Run(true, pages.data(), bufs.data(), static_cast<uint32_t>(pages.size()), path_);
    i = j;
  }
}

void RawStore::PutBuffered(const uint32_t* locs, const uint8_t* vecs, const std::vector<size_t>& order) {
  const uint32_t vb = layout_.vec_bytes;
  std::vector<uint8_t> run;
  ForEachRun(order, locs, layout_, [&](size_t first, size_t count) {
    run.resize(count * vb);
    for (size_t k = 0; k < count; ++k)
      std::memcpy(run.data() + k * vb, vecs + order[first + k] * vb, vb);
    PwriteAll(fd_, run.data(), run.size(), layout_.offset(locs[order[first]]), path_);
  });
}

std::vector<uint64_t> RawStore::Bitmap() const {
  std::vector<uint64_t> out(CeilDiv(layout_.locations(), 64));
  for (size_t w = 0; w < out.size(); ++w) out[w] = bits_[w].load(std::memory_order_acquire);
  return out;
}

uint64_t RawStore::CountAbsent(const std::vector<uint64_t>& mask) const {
  const uint64_t words = CeilDiv(layout_.locations(), 64);
  FUSION_CHECK(mask.size() == words, "location mask has %zu words, want %llu", mask.size(),
               static_cast<unsigned long long>(words));
  uint64_t n = 0;
  for (uint64_t w = 0; w < words; ++w)
    if (mask[w]) n += static_cast<uint64_t>(__builtin_popcountll(mask[w] & ~bits_[w].load(std::memory_order_acquire)));
  return n;
}

void RawStore::Get(const uint32_t* locs, size_t n, uint8_t* out) {
  CheckRange(locs, n);
  size_t absent = 0;
  uint32_t first = 0;
  for (size_t i = 0; i < n; ++i)
    if (!Present(locs[i]) && absent++ == 0) first = locs[i];
  if (absent)
    throw RawMissingError(StrFormat("%zu raw vectors are not on this node (first: location %u)",
                                    absent, first));
  const uint32_t vb = layout_.vec_bytes;
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return locs[a] < locs[b]; });
  // Repeated locations share a run position: read each distinct location once.
  std::vector<size_t> uniq;
  uniq.reserve(n);
  for (size_t i : order)
    if (uniq.empty() || locs[uniq.back()] != locs[i]) uniq.push_back(i);
  struct Run {
    size_t first, count;  // positions into uniq
    uint64_t off;
    uint32_t page;
    bool here;  // the page cache had it
  };
  std::vector<Run> runs;
  ForEachRun(uniq, locs, layout_, [&](size_t first_pos, size_t count) {
    const uint32_t loc = locs[uniq[first_pos]];
    runs.push_back({first_pos, count, layout_.offset(loc), layout_.page(loc), false});
  });
  std::vector<uint8_t> got(uniq.size() * vb);  // uniq position u at u * vb
  // A request's device reads are in flight together: a read-ahead hint for every run the page
  // cache lacks, then the reads, which find their pages arrived or arriving. Read one after
  // another, each waited out the device's latency -- a peer's RAW_GET of 200 scattered vectors
  // took 18-21 ms on BIGANN-20M -- while the device serves many reads at once. Buffered on
  // purpose: reading with O_DIRECT lost the page-cache hits and the read-ahead (engine/README.md).
  for (size_t w = 0; w < runs.size(); w += kReadsInFlight) {
    const size_t end = std::min(runs.size(), w + kReadsInFlight);
    // A page holds several runs where the asked vectors on it are not adjacent, and they come
    // in a row (runs are sorted): once one has started the page's read, the others wait for it
    // in their own read, neither probed (the page is not in the cache yet) nor charged again.
    uint32_t started = UINT32_MAX;
    for (size_t k = w; k < end; ++k) {
      Run& r = runs[k];
      if (r.page == started) continue;
      if (budget_) {
        if (PreadCached(fd_, got.data() + r.first * vb, r.count * vb, r.off)) {
          r.here = true;
          continue;
        }
        budget_->Charge(1);  // one device read per page
      }
      // Only a hint: if it fails, the read below waits out the device alone.
      ::posix_fadvise(fd_, static_cast<off_t>(r.off), static_cast<off_t>(r.count * vb), POSIX_FADV_WILLNEED);
      started = r.page;
    }
    for (size_t k = w; k < end; ++k)
      if (!runs[k].here) PreadAll(fd_, got.data() + runs[k].first * vb, runs[k].count * vb, runs[k].off, path_);
  }
  for (size_t u = 0; u < uniq.size(); ++u) std::memcpy(out + uniq[u] * vb, got.data() + u * vb, vb);
  for (size_t p = 0, u = 0; p < order.size(); ++p) {  // copies for the repeated ones
    if (locs[order[p]] != locs[uniq[u]]) ++u;
    if (order[p] != uniq[u]) std::memcpy(out + order[p] * vb, out + uniq[u] * vb, vb);
  }
  served_.fetch_add(n, std::memory_order_relaxed);
}

RawStats RawStore::Stats() const {
  RawStats s;
  s.vec_bytes = layout_.vec_bytes;
  s.locations = layout_.locations();
  s.present = present_.load(std::memory_order_relaxed);
  s.from_index = from_index_.load(std::memory_order_relaxed);
  s.streamed = streamed_.load(std::memory_order_relaxed);
  s.fetched = fetched_.load(std::memory_order_relaxed);
  s.added = added_.load(std::memory_order_relaxed);
  s.fetches = fetches_.load(std::memory_order_relaxed);
  s.skipped = skipped_.load(std::memory_order_relaxed);
  s.served = served_.load(std::memory_order_relaxed);
  s.restored = restored_;
  return s;
}

}  // namespace fusion
