// hnswlib distance spaces over the native element types, so the navigation graph can store
// head vectors as uint8/int8 (4x smaller than float for SIFT/SPACEV).
#pragma once

#include <memory>

#include "fusion/common.h"
#include "fusion/distance.h"
#include "hnswlib/hnswlib.h"

namespace fusion {

template <class T>
class L2SpaceT : public hnswlib::SpaceInterface<float> {
 public:
  explicit L2SpaceT(size_t dim) : dim_(dim) {}
  size_t get_data_size() override { return dim_ * sizeof(T); }
  hnswlib::DISTFUNC<float> get_dist_func() override { return &Dist; }
  void* get_dist_func_param() override { return &dim_; }

 private:
  static float Dist(const void* a, const void* b, const void* param) {
    return L2Sqr(static_cast<const T*>(a), static_cast<const T*>(b),
                 *static_cast<const size_t*>(param));
  }
  size_t dim_;
};

inline std::unique_ptr<hnswlib::SpaceInterface<float>> MakeL2Space(DType t, size_t dim) {
  return DispatchDType(t, [&](auto tag) -> std::unique_ptr<hnswlib::SpaceInterface<float>> {
    using T = decltype(tag);
    return std::make_unique<L2SpaceT<T>>(dim);
  });
}

using HnswIndex = hnswlib::HierarchicalNSW<float>;

}  // namespace fusion
