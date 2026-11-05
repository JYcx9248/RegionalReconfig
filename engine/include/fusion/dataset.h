// Vector dataset and ground-truth file I/O.
//
// Supported formats
//  * big-ann-benchmarks "bin": .u8bin / .i8bin / .fbin  (uint32 n, uint32 dim, n*dim elements)
//  * TEXMEX "vecs": .bvecs / .fvecs (each vector prefixed by an int32 dimension)
//  * ground truth: big-ann-benchmarks .ibin-style (uint32 nq, uint32 k, int32 ids[nq*k],
//    float dists[nq*k]) and TEXMEX .ivecs (ids only)
#pragma once

#include <algorithm>
#include <cstring>
#include <numeric>
#include <optional>

#include "fusion/common.h"

namespace fusion {

// Infers dtype from the file extension. Returns nullopt for unknown extensions.
std::optional<DType> DTypeFromExtension(const std::string& path);

// A vector file read with pread, never mapped. Each range it reads is dropped from the page
// cache at once (DropFromPageCache), so a pass over a base larger than memory costs only the
// caller's buffers. Mapped, the 230 GB Amazon base ran the build's resident set and the page
// cache up to a shared server's memory. Callers stream the file in chunks (Read, ForEachChunk)
// or fetch rows by ID (ForEachRow, Gather). Thread-safe.
class VectorFile {
 public:
  // `dtype` overrides the extension-based guess. `max_n` > 0 truncates to a prefix.
  static std::unique_ptr<VectorFile> Open(const std::string& path,
                                          std::optional<DType> dtype = std::nullopt,
                                          uint64_t max_n = 0);
  ~VectorFile();
  VectorFile(const VectorFile&) = delete;
  VectorFile& operator=(const VectorFile&) = delete;

  DType dtype() const { return dtype_; }
  uint64_t size() const { return n_; }
  uint32_t dim() const { return dim_; }
  size_t vec_bytes() const { return vec_bytes_; }
  const std::string& path() const { return path_; }
  // Rows per chunk of a streamed pass: about 256 MB.
  uint64_t chunk_rows() const { return std::max<uint64_t>(1, (uint64_t{256} << 20) / vec_bytes_); }

  // Rows [b, e) into out, vec_bytes() each, back to back.
  void Read(uint64_t b, uint64_t e, uint8_t* out) const;

  // fn(b0, b1, rows) for the whole file, chunk_rows() rows at a time, in file order.
  template <class Fn>
  void ForEachChunk(Fn&& fn) const {
    const uint64_t chunk = chunk_rows();
    std::vector<uint8_t> buf(std::min(n_, chunk) * vec_bytes_);
    for (uint64_t b0 = 0; b0 < n_; b0 += chunk) {
      const uint64_t b1 = std::min(n_, b0 + chunk);
      Read(b0, b1, buf.data());
      fn(b0, b1, static_cast<const uint8_t*>(buf.data()));
    }
  }

  // fn(i, row) for each of the rows ids[0..m), in that order; the IDs must not decrease. Rows
  // close together are read in one pread, up to 64 MB across gaps up to 256 KB: a dense set of
  // IDs costs a sequential pass over its range, a sparse one a read per row.
  template <class Id, class Fn>
  void ForEachRow(const Id* ids, size_t m, Fn&& fn) const {
    ForSpans(m, [ids](size_t t) { return static_cast<uint64_t>(ids[t]); }, fn);
  }

  // Rows ids[0..m), in any order, into out: row i at out + i * vec_bytes().
  template <class Id>
  void Gather(const Id* ids, size_t m, uint8_t* out) const {
    const size_t vb = vec_bytes_;
    if (std::is_sorted(ids, ids + m)) {
      ForEachRow(ids, m, [&](size_t i, const uint8_t* row) { std::memcpy(out + i * vb, row, vb); });
      return;
    }
    std::vector<size_t> order(m);
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [ids](size_t a, size_t b) { return ids[a] < ids[b]; });
    ForSpans(m, [&](size_t t) { return static_cast<uint64_t>(ids[order[t]]); },
             [&](size_t t, const uint8_t* row) { std::memcpy(out + order[t] * vb, row, vb); });
  }

 private:
  template <class IdAt, class Fn>
  void ForSpans(size_t m, IdAt id_at, Fn&& fn) const;
  // Bytes [off, off + len) into out, then dropped from the page cache.
  void ReadBytes(uint64_t off, uint64_t len, uint8_t* out) const;
  uint64_t Offset(uint64_t id) const { return first_ + id * stride_; }

  VectorFile() = default;
  std::string path_;
  DType dtype_ = DType::kFloat;
  uint64_t n_ = 0;
  uint32_t dim_ = 0;
  size_t vec_bytes_ = 0;
  size_t stride_ = 0;   // from one row to the next: a vecs file has a 4-byte dimension before each
  uint64_t first_ = 0;  // where row 0's elements start
  int fd_ = -1;
};

template <class IdAt, class Fn>
void VectorFile::ForSpans(size_t m, IdAt id_at, Fn&& fn) const {
  constexpr uint64_t kGap = uint64_t{256} << 10, kSpan = uint64_t{64} << 20;
  std::unique_ptr<uint8_t[]> span;
  for (size_t i = 0; i < m;) {
    uint64_t prev = id_at(i);
    FUSION_CHECK(prev < n_, "row %llu of %s, which has %llu", (unsigned long long)prev,
                 path_.c_str(), (unsigned long long)n_);
    const uint64_t lo = Offset(prev);
    uint64_t hi = lo + vec_bytes_;
    size_t j = i + 1;
    for (; j < m; ++j) {
      const uint64_t id = id_at(j);
      FUSION_CHECK(id >= prev && id < n_, "rows of %s out of order or past its end (%llu after %llu)",
                   path_.c_str(), (unsigned long long)id, (unsigned long long)prev);
      const uint64_t o = Offset(id);
      if (o > hi + kGap || o + vec_bytes_ - lo > kSpan) break;
      hi = o + vec_bytes_;
      prev = id;
    }
    if (!span) span.reset(new uint8_t[std::max<uint64_t>(kSpan, vec_bytes_)]);
    ReadBytes(lo, hi - lo, span.get());
    for (size_t t = i; t < j; ++t) fn(t, static_cast<const uint8_t*>(span.get() + (Offset(id_at(t)) - lo)));
    i = j;
  }
}

// Dense in-memory matrix (queries, samples).
struct VectorSet {
  DType dtype = DType::kFloat;
  uint64_t n = 0;
  uint32_t dim = 0;
  std::vector<uint8_t> data;  // n * dim * DTypeSize(dtype) bytes, contiguous

  size_t vec_bytes() const { return static_cast<size_t>(dim) * DTypeSize(dtype); }
  const uint8_t* Get(uint64_t i) const { return data.data() + i * vec_bytes(); }
};

VectorSet LoadVectors(const std::string& path, std::optional<DType> dtype = std::nullopt,
                      uint64_t max_n = 0);
// Writes a big-ann-benchmarks bin file.
void SaveBin(const std::string& path, DType dtype, uint64_t n, uint32_t dim, const void* data);

struct GroundTruth {
  uint32_t nq = 0;
  uint32_t k = 0;
  std::vector<uint32_t> ids;  // nq * k
  std::vector<float> dists;   // nq * k (empty for .ivecs input)
};

GroundTruth LoadGroundTruth(const std::string& path);
void SaveGroundTruth(const std::string& path, const GroundTruth& gt);

// Mean recall@k over `nq` queries. result_ids[q * stride + i], i < k.
// If `result_dists` and gt.dists are available, a result that is not in the ground-truth
// top-k but ties with the k-th ground-truth distance also counts as a hit (common for
// integer datasets such as SIFT, where exact distance ties are frequent).
double RecallAtK(const GroundTruth& gt, const uint32_t* result_ids, const float* result_dists,
                 uint32_t stride, uint64_t nq, uint32_t k);

}  // namespace fusion
