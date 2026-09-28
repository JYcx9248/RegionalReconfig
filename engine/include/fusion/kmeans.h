// Lloyd's k-means on float data, used for PQ codebooks and for each split of the
// hierarchical balanced clustering.
#pragma once

#include <vector>

#include "fusion/common.h"
#include "fusion/distance.h"

namespace fusion {

struct KMeansParams {
  uint32_t k = 256;
  uint32_t iters = 20;
  uint64_t seed = 1234;
  // Size-balancing penalty. With balance > 0 the assignment cost of point x to centroid c is
  //   ||x - c||^2 + balance * mean_sq_err * |C_c| / (n / k)
  // where |C_c| is the size of cluster c in the previous iteration. It pushes points away
  // from over-full clusters (in the spirit of SPTAG's balanced k-means). 0 = plain k-means.
  float balance = 0.f;
};

// Trains centroids on x (n x d, row-major). Returns k' x d centroids where k' = min(k, n).
// Uses OpenMP internally (runs serially when called from inside a parallel region).
std::vector<float> KMeansTrain(const float* x, size_t n, size_t d, const KMeansParams& p);

// Index of the nearest centroid (plain L2, no balance penalty).
template <class T>
inline uint32_t NearestCentroid(const T* x, const float* centroids, uint32_t k, size_t d,
                                float* dist_out = nullptr) {
  uint32_t best = 0;
  float best_d = L2SqrMixed(x, centroids, d);
  for (uint32_t c = 1; c < k; ++c) {
    float dd = L2SqrMixed(x, centroids + static_cast<size_t>(c) * d, d);
    if (dd < best_d) {
      best_d = dd;
      best = c;
    }
  }
  if (dist_out) *dist_out = best_d;
  return best;
}

}  // namespace fusion
