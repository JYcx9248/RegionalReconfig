#include <algorithm>
#include <utility>
#include <vector>

#include "fusion/filter.h"

namespace fusion {
namespace {

// Host implementation of the filtering pipeline. Deduplication uses the same hash function
// and linear probing as the CUDA kernel.
class CpuFilterWorker : public FilterWorker {
 public:
  CpuFilterWorker(const PQCodebook& cb, const uint8_t* codes, const FilterConfig& cfg)
      : cb_(cb),
        codes_(codes),
        m_(cb.m()),
        lut_(static_cast<size_t>(cb.m()) * kPQKsub),
        cand_(std::max<uint32_t>(cfg.max_candidates, 1)),
        slots_(cfg.slot_addressed ? std::max<uint32_t>(cfg.max_candidates, 1) : 0),
        table_(NextPow2(std::max<uint32_t>(2 * cfg.max_candidates, 16)), kEmptySlot),
        ids_(std::max<uint32_t>(cfg.max_topn, 1)),
        dists_(std::max<uint32_t>(cfg.max_topn, 1)) {
    pairs_.reserve(cfg.max_candidates);
  }

  uint32_t* candidate_buffer() override { return cand_.data(); }
  uint32_t* slot_buffer() override { return slots_.empty() ? nullptr : slots_.data(); }

  void BeginQuery(const float* query) override { cb_.ComputeLUT(query, lut_.data()); }

  uint32_t Filter(uint32_t n_cand, uint32_t topn, uint32_t* n_unique) override {
    FUSION_CHECK(n_cand <= cand_.size(), "too many candidates (%u > %zu)", n_cand, cand_.size());
    FUSION_CHECK(topn <= ids_.size(), "topn %u exceeds max_topn %zu", topn, ids_.size());
    const uint32_t tsize = NextPow2(std::max<uint32_t>(2 * n_cand, 16));
    const uint32_t mask = tsize - 1;
    std::fill(table_.begin(), table_.begin() + tsize, kEmptySlot);
    pairs_.clear();
    const float* lut = lut_.data();
    const uint32_t* slots = slot_buffer();
    for (uint32_t i = 0; i < n_cand; ++i) {
      const uint32_t id = cand_[i];
      uint32_t h = HashId(id) & mask;
      while (true) {
        const uint32_t cur = table_[h];
        if (cur == kEmptySlot) {
          table_[h] = id;
          const uint64_t addr = slots ? slots[i] : id;
          pairs_.emplace_back(AdcDistance(lut, codes_ + addr * m_, m_), id);
          break;
        }
        if (cur == id) break;  // duplicate from a replicated posting list
        h = (h + 1) & mask;
      }
    }
    *n_unique = static_cast<uint32_t>(pairs_.size());
    const uint32_t out = std::min<uint32_t>(topn, static_cast<uint32_t>(pairs_.size()));
    auto less = [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) {
      return a.first < b.first || (a.first == b.first && a.second < b.second);
    };
    if (out < pairs_.size()) std::nth_element(pairs_.begin(), pairs_.begin() + out, pairs_.end(), less);
    std::sort(pairs_.begin(), pairs_.begin() + out, less);
    for (uint32_t i = 0; i < out; ++i) {
      dists_[i] = pairs_[i].first;
      ids_[i] = pairs_[i].second;
    }
    return out;
  }

  const uint32_t* result_ids() const override { return ids_.data(); }
  const float* result_dists() const override { return dists_.data(); }

 private:
  const PQCodebook& cb_;
  const uint8_t* codes_;
  const uint32_t m_;
  std::vector<float> lut_;
  std::vector<uint32_t> cand_;
  std::vector<uint32_t> slots_;  // slot-addressed backends only
  std::vector<uint32_t> table_;
  std::vector<std::pair<float, uint32_t>> pairs_;
  std::vector<uint32_t> ids_;
  std::vector<float> dists_;
};

class CpuFilterBackend : public FilterBackend {
 public:
  CpuFilterBackend(const PQCodebook& cb, const uint8_t* codes, uint64_t n, const FilterConfig& cfg)
      : cb_(cb), code_bytes_(n * cb.m()), slot_addressed_(cfg.slot_addressed) {
    FUSION_CHECK(codes != nullptr || n == 0, "the CPU filter needs the host PQ codes");
    for (int i = 0; i < cfg.num_workers; ++i)
      workers_.push_back(std::make_unique<CpuFilterWorker>(cb_, codes, cfg));
  }
  const char* name() const override { return "cpu"; }
  FilterWorker* worker(int i) override { return workers_.at(static_cast<size_t>(i)).get(); }
  uint64_t code_bytes() const override { return code_bytes_; }
  void StoreCodes(const uint32_t*, uint32_t, const uint8_t*) override {
    // The workers read the slot-addressed host store directly: nothing to copy.
    FUSION_CHECK(slot_addressed_, "StoreCodes on a filter addressed by vector ID");
  }

 private:
  PQCodebook cb_;
  uint64_t code_bytes_;
  bool slot_addressed_;
  std::vector<std::unique_ptr<CpuFilterWorker>> workers_;
};

}  // namespace

std::unique_ptr<FilterBackend> CreateCpuFilter(const PQCodebook& cb, const uint8_t* codes,
                                               uint64_t n, const FilterConfig& cfg) {
  return std::make_unique<CpuFilterBackend>(cb, codes, n, cfg);
}

#ifndef FUSION_WITH_CUDA
std::unique_ptr<FilterBackend> CreateGpuFilter(const PQCodebook&, const uint8_t*, uint64_t,
                                               const FilterConfig&) {
  throw std::runtime_error(
      "this build has no CUDA support: rebuild with -DFUSION_CUDA=ON, or use --backend cpu");
}
bool GpuFilterAvailable() { return false; }
#endif

}  // namespace fusion
