// Vector dataset and ground-truth file I/O.
//
// Supported formats
//  * big-ann-benchmarks "bin": .u8bin / .i8bin / .fbin  (uint32 n, uint32 dim, n*dim elements)
//  * TEXMEX "vecs": .bvecs / .fvecs (each vector prefixed by an int32 dimension)
//  * ground truth: big-ann-benchmarks .ibin-style (uint32 nq, uint32 k, int32 ids[nq*k],
//    float dists[nq*k]) and TEXMEX .ivecs (ids only)
#pragma once

#include <optional>

#include "fusion/common.h"

namespace fusion {

// Infers dtype from the file extension. Returns nullopt for unknown extensions.
std::optional<DType> DTypeFromExtension(const std::string& path);

// Read-only memory-mapped vector file. Works for files much larger than RAM.
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

  const uint8_t* Get(uint64_t i) const { return base_ + i * stride_; }
  template <class T>
  const T* Row(uint64_t i) const {
    return reinterpret_cast<const T*>(Get(i));
  }

 private:
  VectorFile() = default;
  std::string path_;
  DType dtype_ = DType::kFloat;
  uint64_t n_ = 0;
  uint32_t dim_ = 0;
  size_t vec_bytes_ = 0;
  size_t stride_ = 0;
  const uint8_t* base_ = nullptr;
  void* map_ = nullptr;
  size_t map_len_ = 0;
};

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
