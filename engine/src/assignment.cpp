#include "fusion/assignment.h"

#include <omp.h>

#include <algorithm>
#include <atomic>

namespace fusion {

std::unique_ptr<HnswIndex> BuildNavigationGraph(hnswlib::SpaceInterface<float>* space,
                                                const uint8_t* heads, uint32_t num_heads,
                                                size_t vec_bytes, const GraphParams& p) {
  FUSION_CHECK(num_heads > 0, "no heads to index");
  auto g = std::make_unique<HnswIndex>(space, num_heads, p.m, p.ef_construction, p.seed);
  g->addPoint(heads, 0);  // the first insert sets the entry point
  std::atomic<uint64_t> done{1};
  const uint64_t step = std::max<uint64_t>(1, num_heads / 10);
#pragma omp parallel for schedule(dynamic, 64)
  for (int64_t i = 1; i < static_cast<int64_t>(num_heads); ++i) {
    g->addPoint(heads + static_cast<size_t>(i) * vec_bytes, static_cast<hnswlib::labeltype>(i));
    uint64_t c = ++done;
    if (c % step == 0) Log("  graph: %llu / %u heads inserted", (unsigned long long)c, num_heads);
  }
  return g;
}

namespace {

template <class T>
void AssignImpl(const VectorFile& data, HnswIndex* graph, const uint8_t* heads, uint32_t num_heads,
                const AssignParams& p, PostingLists* lists, std::vector<uint32_t>* primary,
                AssignStats* st) {
  const uint64_t n = data.size();
  const uint32_t r = p.replicas;
  const uint32_t d = data.dim();
  const size_t vb = data.vec_bytes();
  FUSION_CHECK(r >= 1 && r <= 64, "replicas must be in [1, 64]");
  auto head = [&](uint32_t c) { return reinterpret_cast<const T*>(heads + static_cast<size_t>(c) * vb); };

  graph->setEf(std::max<uint32_t>(p.ef, p.candidates));
  // Distances from the graph are squared L2, so square the Eq. 2 / RNG factors.
  const float closure = (1.f + p.closure_eps) * (1.f + p.closure_eps);
  const float rng2 = p.rng_factor * p.rng_factor;

  std::vector<uint32_t> assign(n * r, kInvalidId);
  primary->assign(n, kInvalidId);
  std::atomic<uint64_t> done{0};
  const uint64_t step = std::max<uint64_t>(1, n / 10);

#pragma omp parallel
  {
    std::vector<std::pair<float, uint32_t>> cand;
    cand.reserve(p.candidates);
#pragma omp for schedule(dynamic, 256)
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
      const T* x = data.Row<T>(static_cast<uint64_t>(i));
      auto res = graph->searchKnn(x, p.candidates);
      cand.clear();
      while (!res.empty()) {
        cand.emplace_back(res.top().first, static_cast<uint32_t>(res.top().second));
        res.pop();
      }
      std::reverse(cand.begin(), cand.end());  // ascending distance
      if (cand.empty()) continue;

      uint32_t* out = &assign[static_cast<uint64_t>(i) * r];
      uint32_t cnt = 0;
      const float d1 = cand[0].first;
      out[cnt++] = cand[0].second;
      (*primary)[i] = cand[0].second;
      for (size_t j = 1; j < cand.size() && cnt < r; ++j) {
        const float dj = cand[j].first;
        if (dj > closure * d1) break;  // Eq. 2 closure rule (candidates are sorted)
        if (p.rng) {
          const T* c = head(cand[j].second);
          bool skip = false;
          for (uint32_t a = 0; a < cnt && !skip; ++a) skip = rng2 * L2Sqr(c, head(out[a]), d) <= dj;
          if (skip) continue;
        }
        out[cnt++] = cand[j].second;
      }
      uint64_t c = ++done;
      if (c % step == 0) Log("  assignment: %llu / %llu vectors", (unsigned long long)c,
                             (unsigned long long)n);
    }
  }

  // Build CSR posting lists; each list is sorted by vector ID.
  std::vector<uint64_t> counts(num_heads, 0);
  uint64_t total = 0;
  for (uint64_t i = 0; i < n * r; ++i) {
    if (assign[i] != kInvalidId) {
      counts[assign[i]]++;
      ++total;
    }
  }
  lists->offsets.assign(num_heads + 1, 0);
  for (uint32_t c = 0; c < num_heads; ++c) lists->offsets[c + 1] = lists->offsets[c] + counts[c];
  lists->ids.resize(total);
  std::vector<uint64_t> pos(lists->offsets.begin(), lists->offsets.end() - 1);
  for (uint64_t i = 0; i < n; ++i) {
    for (uint32_t j = 0; j < r; ++j) {
      uint32_t c = assign[i * r + j];
      if (c == kInvalidId) break;
      lists->ids[pos[c]++] = static_cast<uint32_t>(i);
    }
  }

  if (st) {
    st->total_entries = total;
    st->avg_replicas = static_cast<double>(total) / static_cast<double>(n);
    st->min_list = static_cast<uint32_t>(*std::min_element(counts.begin(), counts.end()));
    st->max_list = static_cast<uint32_t>(*std::max_element(counts.begin(), counts.end()));
    st->mean_list = static_cast<double>(total) / num_heads;
    st->empty_lists = static_cast<uint32_t>(std::count(counts.begin(), counts.end(), 0ull));
  }
}

}  // namespace

void AssignVectors(const VectorFile& data, HnswIndex* graph, const uint8_t* heads,
                   uint32_t num_heads, const AssignParams& p, PostingLists* lists,
                   std::vector<uint32_t>* primary, AssignStats* stats) {
  DispatchDType(data.dtype(), [&](auto tag) {
    using T = decltype(tag);
    AssignImpl<T>(data, graph, heads, num_heads, p, lists, primary, stats);
  });
}

}  // namespace fusion
