// Small functions shared verbatim by the CPU backend and the CUDA kernels, so that the
// logic the GPU runs is the same logic the CPU tests exercise.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define FUSION_HD __host__ __device__ __forceinline__
#else
#define FUSION_HD inline
#endif

namespace fusion {

// Number of centroids per PQ sub-space (8-bit codes).
constexpr uint32_t kPQKsub = 256;
// Marks an empty slot in the candidate-deduplication hash tables.
constexpr uint32_t kEmptySlot = 0xFFFFFFFFu;

// 32-bit integer mixer (lowbias32). Spreads consecutive vector IDs over the table.
FUSION_HD uint32_t HashId(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

// Asymmetric distance: sum over sub-spaces of the precomputed query-to-centroid distances.
// lut is laid out as lut[m * 256 + centroid].
FUSION_HD float AdcDistance(const float* lut, const uint8_t* code, uint32_t m) {
  float s = 0.f;
  for (uint32_t i = 0; i < m; ++i) s += lut[i * kPQKsub + code[i]];
  return s;
}

}  // namespace fusion
