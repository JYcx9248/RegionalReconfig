// Exact k-NN ground truth by brute force (for dataset prefixes, whose official ground
// truth is not available). Output is in big-ann-benchmarks format.
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <vector>

#include "cli.h"
#include "fusion/dataset.h"
#include "fusion/distance.h"

using namespace fusion;

static void Usage() {
  std::fprintf(stderr,
               "usage: fusion_gt --base FILE --queries FILE --out FILE [--max-n N] [--nq N]\n"
               "                 [--k 100] [--dtype T] [--threads T]\n");
}

int main(int argc, char** argv) {
  const std::set<std::string> flags = {"help"};
  try {
    Args a(argc, argv, flags);
    if (a.Flag("help") || !a.Has("base") || !a.Has("queries") || !a.Has("out")) {
      Usage();
      return a.Flag("help") ? 0 : 2;
    }
    std::optional<DType> dt;
    if (a.Has("dtype")) dt = ParseDType(a.Str("dtype"));
    auto base = VectorFile::Open(a.Required("base"), dt, a.U64("max-n", 0));
    VectorSet q = LoadVectors(a.Required("queries"), base->dtype(), a.U64("nq", 0));
    const uint32_t k = static_cast<uint32_t>(a.U64("k", 100));
    if (a.Has("threads")) omp_set_num_threads(static_cast<int>(a.U64("threads", 1)));
    const std::string out = a.Required("out");
    a.WarnUnused(flags);
    FUSION_CHECK(q.dim == base->dim(), "query dim %u != base dim %u", q.dim, base->dim());
    const uint64_t n = base->size();
    const uint64_t nq = q.n;
    FUSION_CHECK(k <= n, "k larger than the dataset");
    Log("ground truth: %llu queries x %llu base vectors, k=%u", (unsigned long long)nq,
        (unsigned long long)n, k);

    using Entry = std::pair<float, uint32_t>;
    std::vector<std::vector<Entry>> heaps(nq);
    for (auto& h : heaps) h.reserve(k);
    Timer t;
    DispatchDType(base->dtype(), [&](auto tag) {
      using T = decltype(tag);
      const uint32_t d = base->dim();
      constexpr uint64_t kChunk = 1 << 15;  // keep a slice of the base hot in cache
      for (uint64_t b0 = 0; b0 < n; b0 += kChunk) {
        const uint64_t b1 = std::min(n, b0 + kChunk);
#pragma omp parallel for schedule(dynamic, 16)
        for (int64_t qi = 0; qi < static_cast<int64_t>(nq); ++qi) {
          const T* qv = reinterpret_cast<const T*>(q.Get(static_cast<uint64_t>(qi)));
          auto& h = heaps[static_cast<size_t>(qi)];
          for (uint64_t i = b0; i < b1; ++i) {
            const float dd = L2Sqr(qv, base->Row<T>(i), d);
            const Entry e(dd, static_cast<uint32_t>(i));
            if (h.size() < k) {
              h.push_back(e);
              std::push_heap(h.begin(), h.end());
            } else if (e < h.front()) {
              std::pop_heap(h.begin(), h.end());
              h.back() = e;
              std::push_heap(h.begin(), h.end());
            }
          }
        }
        if ((b0 / kChunk) % 32 == 31)
          Log("  %llu / %llu base vectors scanned", (unsigned long long)b1, (unsigned long long)n);
      }
    });

    GroundTruth gt;
    gt.nq = static_cast<uint32_t>(nq);
    gt.k = k;
    gt.ids.resize(nq * k);
    gt.dists.resize(nq * k);
    for (uint64_t qi = 0; qi < nq; ++qi) {
      auto& h = heaps[qi];
      std::sort_heap(h.begin(), h.end());
      for (uint32_t i = 0; i < k; ++i) {
        gt.dists[qi * k + i] = h[i].first;
        gt.ids[qi * k + i] = h[i].second;
      }
    }
    SaveGroundTruth(out, gt);
    Log("wrote %s in %.1fs", out.c_str(), t.Sec());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
