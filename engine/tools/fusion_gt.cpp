// Exact k-NN ground truth by brute force (for dataset prefixes, whose official ground
// truth is not available). Output is in big-ann-benchmarks format.
//
// --stride S computes it for every S-th query only (row r is query r*S): a query file longer
// than a run, which repeats no query, still has ground truth all through the run, at a fraction
// of the cost of all of it.
//
// The base is read a chunk at a time and dropped from the page cache (VectorFile), so it can be
// larger than memory.
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "cli.h"
#include "fusion/dataset.h"
#include "fusion/distance.h"

using namespace fusion;

static void Usage() {
  std::fprintf(stderr,
               "usage: fusion_gt --base FILE --queries FILE --out FILE [--max-n N] [--nq N]\n"
               "                 [--stride S] [--k 100] [--dtype T] [--threads T]\n");
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
    const uint64_t stride = a.U64("stride", 1);
    FUSION_CHECK(stride >= 1, "--stride must be at least 1");
    VectorSet q;  // queries 0, stride, 2*stride, ... of the first --nq
    {
      auto qf = VectorFile::Open(a.Required("queries"), base->dtype(), a.U64("nq", 0));
      q.dtype = qf->dtype();
      q.dim = qf->dim();
      q.n = (qf->size() + stride - 1) / stride;
      q.data.resize(q.n * q.vec_bytes());
      std::vector<uint64_t> rows(q.n);
      for (uint64_t r = 0; r < q.n; ++r) rows[r] = r * stride;
      qf->Gather(rows.data(), rows.size(), q.data.data());
    }
    const uint32_t k = static_cast<uint32_t>(a.U64("k", 100));
    if (a.Has("threads")) omp_set_num_threads(static_cast<int>(a.U64("threads", 1)));
    const std::string out = a.Required("out");
    a.WarnUnused(flags);
    FUSION_CHECK(q.dim == base->dim(), "query dim %u != base dim %u", q.dim, base->dim());
    const uint64_t n = base->size();
    const uint64_t nq = q.n;
    FUSION_CHECK(k <= n, "k larger than the dataset");
    const std::string every = stride > 1 ? " (one in " + std::to_string(stride) + ")" : "";
    Log("ground truth: %llu queries%s x %llu base vectors, k=%u", (unsigned long long)nq, every.c_str(),
        (unsigned long long)n, k);

    using Entry = std::pair<float, uint32_t>;
    std::vector<std::vector<Entry>> heaps(nq);
    for (auto& h : heaps) h.reserve(k);
    Timer t;
    DispatchDType(base->dtype(), [&](auto tag) {
      using T = decltype(tag);
      const uint32_t d = base->dim();
      const size_t vb = base->vec_bytes();
      constexpr uint64_t kChunk = 1 << 15;  // keep a slice of the base hot in cache
      base->ForEachChunk([&](uint64_t r0, uint64_t r1, const uint8_t* rows) {
        for (uint64_t b0 = r0; b0 < r1; b0 += kChunk) {
          const uint64_t b1 = std::min(r1, b0 + kChunk);
#pragma omp parallel for schedule(dynamic, 16)
          for (int64_t qi = 0; qi < static_cast<int64_t>(nq); ++qi) {
            const T* qv = reinterpret_cast<const T*>(q.Get(static_cast<uint64_t>(qi)));
            auto& h = heaps[static_cast<size_t>(qi)];
            for (uint64_t i = b0; i < b1; ++i) {
              const float dd = L2Sqr(qv, reinterpret_cast<const T*>(rows + (i - r0) * vb), d);
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
          if (b1 / (32 * kChunk) > b0 / (32 * kChunk))
            Log("  %llu / %llu base vectors scanned", (unsigned long long)b1, (unsigned long long)n);
        }
      });
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
