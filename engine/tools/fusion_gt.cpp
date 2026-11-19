// Exact k-NN ground truth by brute force (for dataset prefixes, whose official ground
// truth is not available). Output is in big-ann-benchmarks format.
//
// --stride S computes it for every S-th query only (row r is query r*S): a query file longer
// than a run, which repeats no query, still has ground truth all through the run, at a fraction
// of the cost of all of it.
//
// --index DIR reads the base from an index's page file instead of a base file (vectors_<layout>.bin,
// each vector's page and slot from layout_<layout>.map), in page order: for an index whose base
// file is gone, as the queries scripts/data/sample_queries.py draws from it. The answer is the
// same: every query keeps the k smallest (distance, ID) pairs, whatever order the base comes in.
//
// The base is read a chunk at a time and dropped from the page cache (VectorFile, or here), so it
// can be larger than memory.
//
// --exclude and --insert give the ground truth of a data version after update batches (U14): the
// base without the vectors the batches deleted (files of vector IDs, as rtier-updater -deletes
// reads them: text, one per line, or the engine's uint32 array .bin) and with the ones they
// inserted (FILE@FIRST: a batch's vectors, given IDs FIRST, FIRST+1, ... in the file's order, as
// the update controller hands them out). A run's answers are scored against the version their
// epoch read (scripts/score_answers.py).
#include <fcntl.h>
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "cli.h"
#include "version_scan.h"
#include "fusion/dataset.h"
#include "fusion/distance.h"
#include "fusion/index_meta.h"
#include "fusion/layout.h"

using namespace fusion;

static void Usage() {
  std::fprintf(stderr,
               "usage: fusion_gt (--base FILE [--max-n N] [--dtype T] | --index DIR [--layout bucket])\n"
               "                 --queries FILE --out FILE [--nq N] [--stride S] [--k 100] [--threads T]\n"
               "                 [--exclude IDS[,IDS...]] [--insert FILE@FIRST_ID[,FILE@FIRST_ID...]]\n");
}

using namespace fusion::tools;

namespace {

// --insert FILE@FIRST_ID,...: each batch's inserted vectors and the ID of its first.
std::vector<InsertFile> ParseInserts(const std::string& list) {
  std::vector<InsertFile> out;
  for (const std::string& spec : SplitList(list)) {
    const size_t at = spec.rfind('@');
    FUSION_CHECK(at != std::string::npos && at > 0 && at + 1 < spec.size() &&
                     spec.find_first_not_of("0123456789", at + 1) == std::string::npos,
                 "--insert %s: FILE@FIRST_ID", spec.c_str());
    const unsigned long long first = std::stoull(spec.substr(at + 1));
    FUSION_CHECK(first < UINT32_MAX, "--insert %s: an ID beyond 32 bits", spec.c_str());
    out.push_back({spec.substr(0, at), static_cast<uint32_t>(first)});
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const std::set<std::string> flags = {"help"};
  try {
    Args a(argc, argv, flags);
    if (a.Flag("help") || a.Has("base") == a.Has("index") || !a.Has("queries") || !a.Has("out")) {
      Usage();
      return a.Flag("help") ? 0 : 2;
    }
    std::optional<DType> dt;
    if (a.Has("dtype")) dt = ParseDType(a.Str("dtype"));
    Base base = a.Has("base") ? FileBase(a.Required("base"), dt, a.U64("max-n", 0))
                              : IndexBase(a.Required("index"), a.Str("layout", "bucket"));
    if (a.Has("exclude") || a.Has("insert"))
      base = MakeVersion(std::move(base), SplitList(a.Str("exclude")), ParseInserts(a.Str("insert"))).base;
    const uint64_t stride = a.U64("stride", 1);
    FUSION_CHECK(stride >= 1, "--stride must be at least 1");
    VectorSet q;  // queries 0, stride, 2*stride, ... of the first --nq
    {
      auto qf = VectorFile::Open(a.Required("queries"), base.dtype, a.U64("nq", 0));
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
    FUSION_CHECK(q.dim == base.dim, "query dim %u != base dim %u", q.dim, base.dim);
    const uint64_t n = base.n;
    const uint64_t nq = q.n;
    FUSION_CHECK(k <= n, "k larger than the dataset");
    const std::string every = stride > 1 ? " (one in " + std::to_string(stride) + ")" : "";
    Log("ground truth: %llu queries%s x %llu base vectors, k=%u", (unsigned long long)nq, every.c_str(),
        (unsigned long long)n, k);

    using Entry = std::pair<float, uint32_t>;
    std::vector<std::vector<Entry>> heaps(nq);
    for (auto& h : heaps) h.reserve(k);
    // Every query scans a block of the base, which stays in the cores' caches meanwhile: the
    // queries read it from there, not from memory. A block that does not fit is read from memory
    // once per query. A 1024-d float base is 4 KB a vector, and 32K rows (128 MB, the block
    // before) made the scan memory-bound: 200K such vectors against 10K queries took 113 s on 20
    // threads, 40 s with this block (i7-12700H). Few queries get a larger block, so that each
    // block is still work enough for the threads that share it.
    const uint64_t block = std::max<uint64_t>(
        std::max<uint64_t>(1, (uint64_t{512} << 10) / base.vb), (uint64_t{1} << 20) / std::max<uint64_t>(nq, 1));
    uint64_t scanned = 0;
    Timer t;
    DispatchDType(base.dtype, [&](auto tag) {
      using T = decltype(tag);
      const uint32_t d = base.dim;
      const size_t vb = base.vb;
      base.scan([&](const uint8_t* rows, const uint32_t* ids, uint64_t m) {
        for (uint64_t b0 = 0; b0 < m; b0 += block) {
          const uint64_t b1 = std::min(m, b0 + block);
#pragma omp parallel for schedule(dynamic, 16)
          for (int64_t qi = 0; qi < static_cast<int64_t>(nq); ++qi) {
            const T* qv = reinterpret_cast<const T*>(q.Get(static_cast<uint64_t>(qi)));
            auto& h = heaps[static_cast<size_t>(qi)];
            for (uint64_t i = b0; i < b1; ++i) {
              const float dd = L2Sqr(qv, reinterpret_cast<const T*>(rows + i * vb), d);
              const Entry e(dd, ids[i]);
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
        }
        const uint64_t before = scanned;
        scanned += m;
        if (scanned >> 20 > before >> 20)  // about every million vectors
          Log("  %llu / %llu base vectors scanned, %.0fs", (unsigned long long)scanned, (unsigned long long)n, t.Sec());
      });
    });
    FUSION_CHECK(scanned == n, "scanned %llu base vectors of %llu", (unsigned long long)scanned, (unsigned long long)n);

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
