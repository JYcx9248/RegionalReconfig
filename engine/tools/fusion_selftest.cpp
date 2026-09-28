// Checks the filtering backends against each other and against a straightforward reference:
//   * CPU backend vs a reference (sort + unique + ADC) -- always;
//   * slot-addressed backends (rtier's node-level PQ store) vs ID-addressed -- always for CPU;
//   * GPU backend vs CPU backend -- when built with CUDA and a GPU is present.
// Runs on synthetic codes (default) and, with --index/--queries, on a real index.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

#include "cli.h"
#include "fusion/dataset.h"
#include "fusion/distance.h"
#include "fusion/filter.h"
#include "fusion/hnsw_space.h"
#include "fusion/index_meta.h"
#include "fusion/pq.h"

using namespace fusion;

namespace {

struct Result {
  std::vector<uint32_t> ids;
  std::vector<float> dists;
  uint32_t unique = 0;
};

Result RunFilter(FilterWorker* w, const float* q, const std::vector<uint32_t>& cand, uint32_t topn) {
  std::copy(cand.begin(), cand.end(), w->candidate_buffer());
  w->BeginQuery(q);
  Result r;
  uint32_t n = w->Filter(static_cast<uint32_t>(cand.size()), topn, &r.unique);
  r.ids.assign(w->result_ids(), w->result_ids() + n);
  r.dists.assign(w->result_dists(), w->result_dists() + n);
  return r;
}

// Same, for a slot-addressed backend: slot_of[id] is where the code of `id` is stored.
Result RunFilterSlots(FilterWorker* w, const float* q, const std::vector<uint32_t>& cand,
                      const std::vector<uint32_t>& slot_of, uint32_t topn) {
  uint32_t* slots = w->slot_buffer();
  for (size_t i = 0; i < cand.size(); ++i) slots[i] = slot_of[cand[i]];
  return RunFilter(w, q, cand, topn);
}

Result Reference(const PQCodebook& cb, const uint8_t* codes, const float* q,
                 std::vector<uint32_t> cand, uint32_t topn) {
  std::vector<float> lut(static_cast<size_t>(cb.m()) * kPQKsub);
  cb.ComputeLUT(q, lut.data());
  std::sort(cand.begin(), cand.end());
  cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
  std::vector<std::pair<float, uint32_t>> all;
  for (uint32_t id : cand)
    all.emplace_back(AdcDistance(lut.data(), codes + static_cast<uint64_t>(id) * cb.m(), cb.m()), id);
  std::sort(all.begin(), all.end());
  Result r;
  r.unique = static_cast<uint32_t>(all.size());
  for (size_t i = 0; i < std::min<size_t>(topn, all.size()); ++i) {
    r.dists.push_back(all[i].first);
    r.ids.push_back(all[i].second);
  }
  return r;
}

// Distances must agree rank by rank (up to float rounding); IDs may differ only among
// near-ties, so require >= 99% overlap of the ID sets and no duplicate IDs.
bool Same(const Result& a, const Result& b, const char* what, bool exact) {
  if (a.unique != b.unique || a.ids.size() != b.ids.size()) {
    std::printf("  FAIL %s: unique %u vs %u, results %zu vs %zu\n", what, a.unique, b.unique,
                a.ids.size(), b.ids.size());
    return false;
  }
  for (size_t i = 0; i < a.dists.size(); ++i) {
    float tol = exact ? 0.f : 1e-4f * std::max(1.f, std::fabs(a.dists[i]));
    if (std::fabs(a.dists[i] - b.dists[i]) > tol) {
      std::printf("  FAIL %s: rank %zu distance %.6f vs %.6f\n", what, i, a.dists[i], b.dists[i]);
      return false;
    }
  }
  std::unordered_set<uint32_t> sa(a.ids.begin(), a.ids.end()), sb(b.ids.begin(), b.ids.end());
  if (sa.size() != a.ids.size() || sb.size() != b.ids.size()) {
    std::printf("  FAIL %s: duplicate IDs in the output\n", what);
    return false;
  }
  size_t common = 0;
  for (uint32_t id : sa) common += sb.count(id);
  if (exact ? common != sa.size() : common < 0.99 * sa.size()) {
    std::printf("  FAIL %s: only %zu of %zu IDs agree\n", what, common, sa.size());
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const std::set<std::string> flags = {"help", "cpu-only"};
  try {
    Args a(argc, argv, flags);
    if (a.Flag("help")) {
      std::fprintf(stderr, "usage: fusion_selftest [--index DIR --queries FILE] [--nq 100] "
                           "[--nprobe 64] [--topn 200] [--cpu-only] [--gpu-device D]\n");
      return 0;
    }
    const bool gpu = !a.Flag("cpu-only") && GpuFilterAvailable();
    const uint32_t topn = static_cast<uint32_t>(a.U64("topn", 200));
    const int dev = static_cast<int>(a.U64("gpu-device", 0));
    std::printf("fusion_selftest: GPU backend %s\n",
                gpu ? "available -- comparing GPU vs CPU" : "not available -- CPU checks only");
    int failures = 0;

    // ---- synthetic codes: exercises shared-memory (m=16) and global (m=64) table paths ----
    for (uint32_t m : {16u, 64u}) {
      const uint32_t dim = m * 4;
      const uint64_t n = 200000;
      std::mt19937_64 rng(m);
      std::normal_distribution<float> nd;
      PQCodebook cb(dim, m);
      for (size_t i = 0; i < static_cast<size_t>(m) * kPQKsub * cb.dsub(); ++i)
        cb.mutable_centroids()[i] = nd(rng);
      std::vector<uint8_t> codes(n * m);
      for (auto& c : codes) c = static_cast<uint8_t>(rng() & 0xFF);
      // Candidates with heavy duplication, as produced by replicated posting lists.
      std::vector<uint32_t> cand;
      std::uniform_int_distribution<uint32_t> pick(0, static_cast<uint32_t>(n - 1));
      for (int i = 0; i < 6000; ++i) {
        uint32_t id = pick(rng);
        int copies = 1 + static_cast<int>(rng() % 8);
        for (int c = 0; c < copies; ++c) cand.push_back(id);
      }
      std::shuffle(cand.begin(), cand.end(), rng);
      std::vector<float> q(dim);
      for (auto& v : q) v = nd(rng);

      FilterConfig fc;
      fc.max_candidates = static_cast<uint32_t>(cand.size());
      fc.max_topn = topn;
      fc.gpu_device = dev;
      auto cpu = CreateCpuFilter(cb, codes.data(), n, fc);
      Result rc = RunFilter(cpu->worker(0), q.data(), cand, topn);
      Result ref = Reference(cb, codes.data(), q.data(), cand, topn);
      bool ok = Same(rc, ref, "cpu vs reference", true);
      // Slot addressing: the same codes stored at shuffled slots give the same answer.
      std::vector<uint32_t> slot_of(n);
      std::iota(slot_of.begin(), slot_of.end(), 0u);
      std::shuffle(slot_of.begin(), slot_of.end(), rng);
      std::vector<uint8_t> slot_codes(codes.size());
      for (uint64_t id = 0; id < n; ++id)
        std::copy_n(&codes[id * m], m, &slot_codes[static_cast<uint64_t>(slot_of[id]) * m]);
      FilterConfig fs = fc;
      fs.slot_addressed = true;
      auto cpu_slots = CreateCpuFilter(cb, slot_codes.data(), n, fs);
      ok = Same(RunFilterSlots(cpu_slots->worker(0), q.data(), cand, slot_of, topn), rc,
                "cpu slots vs cpu", true) && ok;
      if (gpu) {
        auto gs = CreateGpuFilter(cb, nullptr, n, fs);
        std::vector<uint32_t> all(n);
        std::iota(all.begin(), all.end(), 0u);
        const uint32_t half = static_cast<uint32_t>(n / 2);  // two StoreCodes calls
        gs->StoreCodes(all.data(), half, slot_codes.data());
        gs->StoreCodes(all.data() + half, static_cast<uint32_t>(n) - half, slot_codes.data());
        ok = Same(RunFilterSlots(gs->worker(0), q.data(), cand, slot_of, topn), rc,
                  "gpu slots vs cpu", false) && ok;
      }
      if (gpu) {
        auto g = CreateGpuFilter(cb, codes.data(), n, fc);
        Result rg = RunFilter(g->worker(0), q.data(), cand, topn);
        ok = Same(rg, rc, "gpu vs cpu", false) && ok;
        // Reuse of the same worker (hash table and counter must be reset between queries).
        Result rg2 = RunFilter(g->worker(0), q.data(), cand, topn);
        ok = Same(rg2, rc, "gpu vs cpu (second query)", false) && ok;
      }
      std::printf("[%s] synthetic m=%u: %zu candidates, %u unique\n", ok ? " ok " : "FAIL", m,
                  cand.size(), rc.unique);
      failures += ok ? 0 : 1;
    }

    // ---- real index ------------------------------------------------------------------------
    if (a.Has("index") && a.Has("queries")) {
      const std::string dir = a.Str("index");
      IndexMeta meta = IndexMeta::Load(JoinPath(dir, files::kMeta));
      const DType dt = ParseDType(meta.Get("dtype"));
      const uint32_t dim = static_cast<uint32_t>(meta.GetU64("dim"));
      const uint32_t nprobe = static_cast<uint32_t>(a.U64("nprobe", 64));
      VectorSet qs = LoadVectors(a.Str("queries"), dt, a.U64("nq", 100));
      PQCodebook cb = PQCodebook::Load(JoinPath(dir, files::kPQCodebook));
      uint64_t n = 0;
      uint32_t m = 0;
      std::vector<uint8_t> codes = LoadPQCodes(JoinPath(dir, files::kPQCodes), &n, &m);
      PostingLists lists = PostingLists::Load(JoinPath(dir, files::kPostings));
      auto space = MakeL2Space(dt, dim);
      HnswIndex graph(space.get(), JoinPath(dir, files::kGraph));
      graph.setEf(2 * nprobe);

      std::vector<std::vector<uint32_t>> cands(qs.n);
      size_t max_cand = 1;
      for (uint64_t i = 0; i < qs.n; ++i) {
        auto res = graph.searchKnn(qs.Get(i), nprobe);
        while (!res.empty()) {
          uint32_t c = static_cast<uint32_t>(res.top().second);
          res.pop();
          cands[i].insert(cands[i].end(), lists.List(c), lists.List(c) + lists.Size(c));
        }
        max_cand = std::max(max_cand, cands[i].size());
      }
      FilterConfig fc;
      fc.max_candidates = static_cast<uint32_t>(max_cand);
      fc.max_topn = topn;
      fc.gpu_device = dev;
      auto cpu = CreateCpuFilter(cb, codes.data(), n, fc);
      std::unique_ptr<FilterBackend> g;
      if (gpu) g = CreateGpuFilter(cb, codes.data(), n, fc);
      int bad = 0;
      std::vector<float> qf(dim);
      for (uint64_t i = 0; i < qs.n; ++i) {
        DispatchDType(dt, [&](auto tag) {
          using T = decltype(tag);
          ToFloat(reinterpret_cast<const T*>(qs.Get(i)), qf.data(), dim);
        });
        Result rc = RunFilter(cpu->worker(0), qf.data(), cands[i], topn);
        Result ref = Reference(cb, codes.data(), qf.data(), cands[i], topn);
        bool ok = Same(rc, ref, "cpu vs reference", true);
        if (g) ok = Same(RunFilter(g->worker(0), qf.data(), cands[i], topn), rc, "gpu vs cpu", false) && ok;
        bad += ok ? 0 : 1;
      }
      std::printf("[%s] index %s: %llu queries, nprobe=%u, up to %zu candidates\n",
                  bad ? "FAIL" : " ok ", dir.c_str(), (unsigned long long)qs.n, nprobe, max_cand);
      failures += bad;
    }
    a.WarnUnused(flags);
    std::printf(failures ? "SELFTEST FAILED\n" : "SELFTEST PASSED\n");
    return failures ? 1 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
