// Squared L2 distances for the supported element types.
//
// Integer types accumulate in int32 (exact for dim <= 33000), so distances on uint8/int8
// data are exact integers stored in a float. The loops are written so that GCC/Clang
// auto-vectorize them at -O3 (float uses an OpenMP SIMD reduction).
#pragma once

#include <cstddef>
#include <cstdint>

namespace fusion {

inline float L2Sqr(const uint8_t* a, const uint8_t* b, size_t d) {
  int32_t s = 0;
  for (size_t i = 0; i < d; ++i) {
    int32_t t = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
    s += t * t;
  }
  return static_cast<float>(s);
}

inline float L2Sqr(const int8_t* a, const int8_t* b, size_t d) {
  int32_t s = 0;
  for (size_t i = 0; i < d; ++i) {
    int32_t t = static_cast<int32_t>(a[i]) - static_cast<int32_t>(b[i]);
    s += t * t;
  }
  return static_cast<float>(s);
}

inline float L2Sqr(const float* a, const float* b, size_t d) {
  float s = 0.f;
#pragma omp simd reduction(+ : s)
  for (size_t i = 0; i < d; ++i) {
    float t = a[i] - b[i];
    s += t * t;
  }
  return s;
}

// Distance between a typed vector and a float vector (k-means, PQ).
template <class T>
inline float L2SqrMixed(const T* a, const float* b, size_t d) {
  float s = 0.f;
#pragma omp simd reduction(+ : s)
  for (size_t i = 0; i < d; ++i) {
    float t = static_cast<float>(a[i]) - b[i];
    s += t * t;
  }
  return s;
}

template <class T>
inline void ToFloat(const T* src, float* dst, size_t d) {
  for (size_t i = 0; i < d; ++i) dst[i] = static_cast<float>(src[i]);
}

}  // namespace fusion
