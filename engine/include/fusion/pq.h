// Product quantization (8-bit codes). The compressed vectors are the tier that lives in
// GPU memory (HBM); the codebook is shared by all vectors.
#pragma once

#include <vector>

#include "fusion/common.h"
#include "fusion/distance.h"
#include "fusion/hd_common.h"

namespace fusion {

class PQCodebook {
 public:
  PQCodebook() = default;
  PQCodebook(uint32_t dim, uint32_t m);

  uint32_t dim() const { return dim_; }
  uint32_t m() const { return m_; }        // sub-spaces = code bytes per vector
  uint32_t dsub() const { return dsub_; }  // dimensions per sub-space
  // m x 256 x dsub floats.
  const float* centroids() const { return centroids_.data(); }
  float* mutable_centroids() { return centroids_.data(); }
  const float* SubCentroids(uint32_t mi) const {
    return centroids_.data() + static_cast<size_t>(mi) * kPQKsub * dsub_;
  }

  // Distance table for a query: lut[m * 256 + j] = ||q_m - c_{m,j}||^2.
  void ComputeLUT(const float* q, float* lut) const;

  template <class T>
  void Encode(const T* x, uint8_t* code) const {
    float buf[512];
    for (uint32_t mi = 0; mi < m_; ++mi) {
      const T* xs = x + static_cast<size_t>(mi) * dsub_;
      for (uint32_t t = 0; t < dsub_; ++t) buf[t] = static_cast<float>(xs[t]);
      const float* c = SubCentroids(mi);
      uint32_t best = 0;
      float best_d = L2Sqr(buf, c, dsub_);
      for (uint32_t j = 1; j < kPQKsub; ++j) {
        float dd = L2Sqr(buf, c + static_cast<size_t>(j) * dsub_, dsub_);
        if (dd < best_d) {
          best_d = dd;
          best = j;
        }
      }
      code[mi] = static_cast<uint8_t>(best);
    }
  }

  void Decode(const uint8_t* code, float* x) const;

  void Save(const std::string& path) const;
  static PQCodebook Load(const std::string& path);

 private:
  uint32_t dim_ = 0, m_ = 0, dsub_ = 0;
  std::vector<float> centroids_;
};

struct PQTrainParams {
  uint32_t m = 0;
  uint32_t iters = 20;
  uint64_t seed = 7;
};

// Trains one 256-centroid k-means per sub-space on float samples x (n x dim).
PQCodebook TrainPQ(const float* x, size_t n, uint32_t dim, const PQTrainParams& p);

// Default number of sub-spaces: dim/8 (8 dims per byte) when possible, else dim/4, else dim/2.
// SIFT (128) -> 16 bytes, DEEP (96) -> 12 bytes, SPACEV (100) -> 25 bytes.
uint32_t DefaultPQM(uint32_t dim);

// PQ codes file: uint64 n, uint32 m, uint32 reserved, then n*m bytes.
void SavePQCodes(const std::string& path, const uint8_t* codes, uint64_t n, uint32_t m);
std::vector<uint8_t> LoadPQCodes(const std::string& path, uint64_t* n, uint32_t* m);

}  // namespace fusion
