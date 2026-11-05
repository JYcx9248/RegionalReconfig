#include "fusion/dataset.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <unordered_set>

namespace fusion {
namespace {

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool IsVecsFormat(const std::string& path) {
  return EndsWith(path, ".bvecs") || EndsWith(path, ".fvecs") || EndsWith(path, ".ivecs");
}

}  // namespace

std::optional<DType> DTypeFromExtension(const std::string& path) {
  if (EndsWith(path, ".u8bin") || EndsWith(path, ".bbin") || EndsWith(path, ".bvecs"))
    return DType::kUInt8;
  if (EndsWith(path, ".i8bin")) return DType::kInt8;
  if (EndsWith(path, ".fbin") || EndsWith(path, ".fvecs")) return DType::kFloat;
  return std::nullopt;
}

std::unique_ptr<VectorFile> VectorFile::Open(const std::string& path, std::optional<DType> dtype,
                                             uint64_t max_n) {
  std::unique_ptr<VectorFile> vf(new VectorFile());
  vf->path_ = path;
  if (!dtype) dtype = DTypeFromExtension(path);
  FUSION_CHECK(dtype.has_value(), "cannot infer dtype of %s; pass --dtype", path.c_str());
  vf->dtype_ = *dtype;

  vf->fd_ = ::open(path.c_str(), O_RDONLY);
  const int fd = vf->fd_;
  FUSION_CHECK(fd >= 0, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  uint64_t fsize = FileSize(path);
  const size_t esz = DTypeSize(vf->dtype_);

  if (IsVecsFormat(path)) {
    int32_t d = 0;
    FUSION_CHECK(::pread(fd, &d, 4, 0) == 4, "cannot read header of %s", path.c_str());
    FUSION_CHECK(d > 0, "bad dimension %d in %s", d, path.c_str());
    vf->dim_ = static_cast<uint32_t>(d);
    vf->vec_bytes_ = vf->dim_ * esz;
    vf->stride_ = 4 + vf->vec_bytes_;
    vf->first_ = 4;
    vf->n_ = fsize / vf->stride_;
  } else {
    uint32_t hdr[2];
    FUSION_CHECK(::pread(fd, hdr, 8, 0) == 8, "cannot read header of %s", path.c_str());
    vf->n_ = hdr[0];
    vf->dim_ = hdr[1];
    vf->vec_bytes_ = vf->dim_ * esz;
    vf->stride_ = vf->vec_bytes_;
    vf->first_ = 8;
    FUSION_CHECK(vf->dim_ > 0, "bad dimension in %s", path.c_str());
    uint64_t avail = (fsize - 8) / vf->vec_bytes_;
    if (avail < vf->n_) {
      // Common when a prefix of a huge file was downloaded with an unmodified header.
      Log("warning: %s header says %llu vectors but file holds %llu; using %llu", path.c_str(),
          (unsigned long long)vf->n_, (unsigned long long)avail, (unsigned long long)avail);
      vf->n_ = avail;
    }
  }
  if (max_n > 0 && max_n < vf->n_) vf->n_ = max_n;
  FUSION_CHECK(vf->n_ > 0, "%s contains no vectors", path.c_str());
  return vf;
}

VectorFile::~VectorFile() {
  if (fd_ >= 0) ::close(fd_);
}

void VectorFile::Read(uint64_t b, uint64_t e, uint8_t* out) const {
  FUSION_CHECK(b <= e && e <= n_, "rows [%llu, %llu) of %s, which has %llu", (unsigned long long)b,
               (unsigned long long)e, path_.c_str(), (unsigned long long)n_);
  if (stride_ == vec_bytes_) {
    ReadBytes(Offset(b), (e - b) * vec_bytes_, out);
    return;
  }
  ForSpans(e - b, [b](size_t t) { return b + t; },
           [&](size_t t, const uint8_t* row) { std::memcpy(out + t * vec_bytes_, row, vec_bytes_); });
}

void VectorFile::ReadBytes(uint64_t off, uint64_t len, uint8_t* out) const {
  for (uint64_t done = 0; done < len;) {
    const size_t want = static_cast<size_t>(std::min<uint64_t>(len - done, uint64_t{1} << 30));
    const ssize_t got = ::pread(fd_, out + done, want, static_cast<off_t>(off + done));
    if (got < 0 && errno == EINTR) continue;
    FUSION_CHECK(got > 0, "cannot read %s at byte %llu: %s", path_.c_str(),
                 (unsigned long long)(off + done), got < 0 ? std::strerror(errno) : "the file ends there");
    done += static_cast<uint64_t>(got);
  }
  DropFromPageCache(fd_, off, len);
}

VectorSet LoadVectors(const std::string& path, std::optional<DType> dtype, uint64_t max_n) {
  auto vf = VectorFile::Open(path, dtype, max_n);
  VectorSet vs;
  vs.dtype = vf->dtype();
  vs.n = vf->size();
  vs.dim = vf->dim();
  vs.data.resize(vs.n * vs.vec_bytes());
  vf->Read(0, vs.n, vs.data.data());
  return vs;
}

void SaveBin(const std::string& path, DType dtype, uint64_t n, uint32_t dim, const void* data) {
  FUSION_CHECK(n <= 0xFFFFFFFFull, "too many vectors for bin format");
  FILE* f = std::fopen(path.c_str(), "wb");
  FUSION_CHECK(f != nullptr, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  uint32_t hdr[2] = {static_cast<uint32_t>(n), dim};
  AppendToFile(f, hdr, sizeof(hdr), path);
  AppendToFile(f, data, n * dim * DTypeSize(dtype), path);
  FUSION_CHECK(std::fclose(f) == 0, "close failed for %s", path.c_str());
}

GroundTruth LoadGroundTruth(const std::string& path) {
  GroundTruth gt;
  std::vector<uint8_t> buf = ReadFile(path);
  if (EndsWith(path, ".ivecs")) {
    FUSION_CHECK(buf.size() >= 4, "empty ground truth %s", path.c_str());
    int32_t k;
    std::memcpy(&k, buf.data(), 4);
    FUSION_CHECK(k > 0, "bad k in %s", path.c_str());
    size_t row = 4 + 4 * static_cast<size_t>(k);
    gt.k = static_cast<uint32_t>(k);
    gt.nq = static_cast<uint32_t>(buf.size() / row);
    gt.ids.resize(static_cast<size_t>(gt.nq) * gt.k);
    for (uint32_t q = 0; q < gt.nq; ++q) {
      std::memcpy(&gt.ids[static_cast<size_t>(q) * gt.k], buf.data() + q * row + 4, 4 * gt.k);
    }
    return gt;
  }
  FUSION_CHECK(buf.size() >= 8, "empty ground truth %s", path.c_str());
  uint32_t hdr[2];
  std::memcpy(hdr, buf.data(), 8);
  gt.nq = hdr[0];
  gt.k = hdr[1];
  size_t cnt = static_cast<size_t>(gt.nq) * gt.k;
  FUSION_CHECK(buf.size() >= 8 + 4 * cnt, "truncated ground truth %s", path.c_str());
  gt.ids.resize(cnt);
  std::memcpy(gt.ids.data(), buf.data() + 8, 4 * cnt);
  if (buf.size() >= 8 + 8 * cnt) {
    gt.dists.resize(cnt);
    std::memcpy(gt.dists.data(), buf.data() + 8 + 4 * cnt, 4 * cnt);
  }
  return gt;
}

void SaveGroundTruth(const std::string& path, const GroundTruth& gt) {
  FILE* f = std::fopen(path.c_str(), "wb");
  FUSION_CHECK(f != nullptr, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  uint32_t hdr[2] = {gt.nq, gt.k};
  AppendToFile(f, hdr, sizeof(hdr), path);
  WriteVector(f, gt.ids, path);
  WriteVector(f, gt.dists, path);
  FUSION_CHECK(std::fclose(f) == 0, "close failed for %s", path.c_str());
}

double RecallAtK(const GroundTruth& gt, const uint32_t* result_ids, const float* result_dists,
                 uint32_t stride, uint64_t nq, uint32_t k) {
  FUSION_CHECK(gt.k >= k, "ground truth has k=%u < requested k=%u", gt.k, k);
  FUSION_CHECK(gt.nq >= nq, "ground truth has %u queries < %llu", gt.nq, (unsigned long long)nq);
  const bool use_ties = result_dists != nullptr && !gt.dists.empty();
  double total = 0;
  std::unordered_set<uint32_t> truth;
  for (uint64_t q = 0; q < nq; ++q) {
    truth.clear();
    const uint32_t* g = &gt.ids[q * gt.k];
    for (uint32_t i = 0; i < k; ++i) truth.insert(g[i]);
    // Relative tolerance for float datasets; integer distances are exact.
    float kth = use_ties ? gt.dists[q * gt.k + k - 1] : 0.f;
    float tol = use_ties ? std::max(1e-6f, std::fabs(kth) * 1e-5f) : 0.f;
    uint32_t hits = 0;
    for (uint32_t i = 0; i < k; ++i) {
      uint32_t id = result_ids[q * stride + i];
      if (id == kInvalidId) continue;
      if (truth.count(id) || (use_ties && result_dists[q * stride + i] <= kth + tol)) ++hits;
    }
    total += static_cast<double>(std::min(hits, k)) / k;
  }
  return nq ? total / static_cast<double>(nq) : 0.0;
}

}  // namespace fusion
