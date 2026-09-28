#include "fusion/pq.h"

#include <cerrno>
#include <cstring>

#include "fusion/kmeans.h"

namespace fusion {

PQCodebook::PQCodebook(uint32_t dim, uint32_t m) : dim_(dim), m_(m) {
  FUSION_CHECK(m > 0 && dim % m == 0, "PQ: dim %u is not divisible by m=%u", dim, m);
  dsub_ = dim / m;
  FUSION_CHECK(dsub_ <= 512, "PQ: sub-space dimension %u too large (max 512)", dsub_);
  centroids_.assign(static_cast<size_t>(m) * kPQKsub * dsub_, 0.f);
}

void PQCodebook::ComputeLUT(const float* q, float* lut) const {
  for (uint32_t mi = 0; mi < m_; ++mi) {
    const float* qs = q + static_cast<size_t>(mi) * dsub_;
    const float* c = SubCentroids(mi);
    for (uint32_t j = 0; j < kPQKsub; ++j)
      lut[mi * kPQKsub + j] = L2Sqr(qs, c + static_cast<size_t>(j) * dsub_, dsub_);
  }
}

void PQCodebook::Decode(const uint8_t* code, float* x) const {
  for (uint32_t mi = 0; mi < m_; ++mi) {
    const float* c = SubCentroids(mi) + static_cast<size_t>(code[mi]) * dsub_;
    std::memcpy(x + static_cast<size_t>(mi) * dsub_, c, dsub_ * sizeof(float));
  }
}

void PQCodebook::Save(const std::string& path) const {
  FILE* f = std::fopen(path.c_str(), "wb");
  FUSION_CHECK(f != nullptr, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  uint32_t hdr[3] = {dim_, m_, kPQKsub};
  AppendToFile(f, hdr, sizeof(hdr), path);
  WriteVector(f, centroids_, path);
  FUSION_CHECK(std::fclose(f) == 0, "close failed for %s", path.c_str());
}

PQCodebook PQCodebook::Load(const std::string& path) {
  std::vector<uint8_t> buf = ReadFile(path);
  FUSION_CHECK(buf.size() >= 12, "bad PQ codebook %s", path.c_str());
  uint32_t hdr[3];
  std::memcpy(hdr, buf.data(), sizeof(hdr));
  FUSION_CHECK(hdr[2] == kPQKsub, "PQ codebook %s has ksub=%u (expected 256)", path.c_str(),
               hdr[2]);
  PQCodebook cb(hdr[0], hdr[1]);
  size_t bytes = cb.centroids_.size() * sizeof(float);
  FUSION_CHECK(buf.size() == 12 + bytes, "truncated PQ codebook %s", path.c_str());
  std::memcpy(cb.centroids_.data(), buf.data() + 12, bytes);
  return cb;
}

PQCodebook TrainPQ(const float* x, size_t n, uint32_t dim, const PQTrainParams& p) {
  PQCodebook cb(dim, p.m);
  const uint32_t dsub = cb.dsub();
  FUSION_CHECK(n >= kPQKsub, "PQ training needs at least 256 samples (got %zu)", n);
  // Sub-spaces are independent: train them in parallel, each k-means single-threaded.
#pragma omp parallel for schedule(dynamic, 1)
  for (uint32_t mi = 0; mi < p.m; ++mi) {
    std::vector<float> sub(n * dsub);
    for (size_t i = 0; i < n; ++i)
      std::memcpy(&sub[i * dsub], x + i * dim + static_cast<size_t>(mi) * dsub,
                  dsub * sizeof(float));
    KMeansParams kp;
    kp.k = kPQKsub;
    kp.iters = p.iters;
    kp.seed = p.seed + mi * 7919;
    std::vector<float> c = KMeansTrain(sub.data(), n, dsub, kp);
    std::memcpy(cb.mutable_centroids() + static_cast<size_t>(mi) * kPQKsub * dsub, c.data(),
                c.size() * sizeof(float));
  }
  return cb;
}

uint32_t DefaultPQM(uint32_t dim) {
  if (dim % 8 == 0) return dim / 8;
  if (dim % 4 == 0) return dim / 4;
  if (dim % 2 == 0) return dim / 2;
  return dim;
}

void SavePQCodes(const std::string& path, const uint8_t* codes, uint64_t n, uint32_t m) {
  FILE* f = std::fopen(path.c_str(), "wb");
  FUSION_CHECK(f != nullptr, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  uint64_t n64 = n;
  uint32_t hdr[2] = {m, 0};
  AppendToFile(f, &n64, sizeof(n64), path);
  AppendToFile(f, hdr, sizeof(hdr), path);
  AppendToFile(f, codes, n * m, path);
  FUSION_CHECK(std::fclose(f) == 0, "close failed for %s", path.c_str());
}

std::vector<uint8_t> LoadPQCodes(const std::string& path, uint64_t* n, uint32_t* m) {
  FILE* f = std::fopen(path.c_str(), "rb");
  FUSION_CHECK(f != nullptr, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  uint64_t n64 = 0;
  uint32_t hdr[2] = {0, 0};
  bool ok = std::fread(&n64, sizeof(n64), 1, f) == 1 && std::fread(hdr, sizeof(hdr), 1, f) == 1;
  FUSION_CHECK(ok, "bad PQ codes header in %s", path.c_str());
  std::vector<uint8_t> codes(n64 * hdr[0]);
  size_t r = codes.empty() ? 0 : std::fread(codes.data(), 1, codes.size(), f);
  std::fclose(f);
  FUSION_CHECK(r == codes.size(), "truncated PQ codes %s", path.c_str());
  *n = n64;
  *m = hdr[0];
  return codes;
}

}  // namespace fusion
