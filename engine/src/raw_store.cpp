#include "fusion/raw_store.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <numeric>

namespace fusion {
namespace {

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

RawStore::RawStore(const std::string& path, const RawLayout& layout, bool direct)
    : layout_(layout), path_(path) {
  FUSION_CHECK(layout.vec_bytes > 0 && layout.vectors_per_page >= 1 &&
                   static_cast<uint64_t>(layout.vec_bytes) * layout.vectors_per_page <= layout.page_size,
               "bad raw-vector page geometry");
  FUSION_CHECK(layout.locations() < kInvalidId, "too many raw-vector locations");
  const bool anonymous = path_.empty();
  if (anonymous) {
    const char* tmp = std::getenv("TMPDIR");
    std::string tmpl = std::string(tmp && *tmp ? tmp : "/tmp") + "/rtier-raw-XXXXXX";
    fd_ = ::mkstemp(&tmpl[0]);
    FUSION_CHECK(fd_ >= 0, "cannot create a temporary raw-vector file: %s", std::strerror(errno));
    path_ = tmpl;
  } else {
    fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    FUSION_CHECK(fd_ >= 0, "cannot create %s: %s", path_.c_str(), std::strerror(errno));
  }
  // Sparse: only the vectors written take space.
  const uint64_t bytes = layout.num_pages * layout.page_size;
  if (::ftruncate(fd_, static_cast<off_t>(bytes)) != 0) {
    const std::string err = std::strerror(errno);
    ::close(fd_);
    if (anonymous) ::unlink(path_.c_str());
    FUSION_CHECK(false, "cannot size %s to %llu bytes: %s", path_.c_str(),
                 static_cast<unsigned long long>(bytes), err.c_str());
  }
  try {
    pages_ = std::make_unique<PageFile>(path_, layout.page_size, direct);
  } catch (...) {
    ::close(fd_);
    if (anonymous) ::unlink(path_.c_str());
    throw;
  }
  if (anonymous) ::unlink(path_.c_str());  // both descriptors keep it alive until we exit
  const uint64_t words = CeilDiv(layout.locations(), 64);
  bits_.reset(new std::atomic<uint64_t>[words]);
  for (uint64_t i = 0; i < words; ++i) bits_[i].store(0, std::memory_order_relaxed);
}

RawStore::~RawStore() {
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
  const uint32_t vb = layout_.vec_bytes;
  // The new ones, by location (duplicates in the batch once), so that neighbours on a page go
  // out in one write.
  std::vector<size_t> order;
  order.reserve(n);
  for (size_t i = 0; i < n; ++i)
    if (!Present(locs[i])) order.push_back(i);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return locs[a] < locs[b]; });
  order.erase(std::unique(order.begin(), order.end(),
                          [&](size_t a, size_t b) { return locs[a] == locs[b]; }),
              order.end());
  std::vector<uint8_t> run;
  ForEachRun(order, locs, layout_, [&](size_t first, size_t count) {
    run.resize(count * vb);
    for (size_t k = 0; k < count; ++k)
      std::memcpy(run.data() + k * vb, vecs + order[first + k] * vb, vb);
    PwriteAll(fd_, run.data(), run.size(), layout_.offset(locs[order[first]]), path_);
  });
  // Only now, with every write returned, do the vectors become visible to readers.
  size_t installed = 0;
  for (size_t i : order) {
    const uint32_t loc = locs[i];
    const uint64_t bit = uint64_t{1} << (loc & 63);
    if (!(bits_[loc >> 6].fetch_or(bit, std::memory_order_release) & bit)) ++installed;
  }
  present_.fetch_add(installed, std::memory_order_relaxed);
  std::atomic<uint64_t>& by = origin == RawOrigin::kIndex      ? from_index_
                              : origin == RawOrigin::kStreamed ? streamed_
                                                               : fetched_;
  by.fetch_add(installed, std::memory_order_relaxed);
  skipped_.fetch_add(n - installed, std::memory_order_relaxed);
  return installed;
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
  std::vector<uint8_t> run;
  // Repeated locations share a run position: read each distinct location once.
  std::vector<size_t> uniq;
  uniq.reserve(n);
  for (size_t i : order)
    if (uniq.empty() || locs[uniq.back()] != locs[i]) uniq.push_back(i);
  ForEachRun(uniq, locs, layout_, [&](size_t first_pos, size_t count) {
    run.resize(count * vb);
    PreadAll(fd_, run.data(), run.size(), layout_.offset(locs[uniq[first_pos]]), path_);
    for (size_t k = 0; k < count; ++k)
      std::memcpy(out + uniq[first_pos + k] * vb, run.data() + k * vb, vb);
  });
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
  s.fetches = fetches_.load(std::memory_order_relaxed);
  s.skipped = skipped_.load(std::memory_order_relaxed);
  s.served = served_.load(std::memory_order_relaxed);
  return s;
}

}  // namespace fusion
