#include "fusion/builder.h"

#include <algorithm>
#include <cstring>
#include <random>
#include <unordered_set>

#include "fusion/dataset.h"
#include "fusion/hnsw_space.h"
#include "fusion/index_meta.h"
#include "fusion/layout.h"
#include "fusion/pq.h"

namespace fusion {
namespace {

// s distinct indices from [0, n), sorted (Floyd's algorithm).
std::vector<uint64_t> SampleIndices(uint64_t n, uint64_t s, uint64_t seed) {
  std::vector<uint64_t> out;
  if (s >= n) {
    out.resize(n);
    for (uint64_t i = 0; i < n; ++i) out[i] = i;
    return out;
  }
  std::mt19937_64 rng(seed);
  std::unordered_set<uint64_t> chosen;
  chosen.reserve(s * 2);
  for (uint64_t j = n - s; j < n; ++j) {
    std::uniform_int_distribution<uint64_t> pick(0, j);
    uint64_t t = pick(rng);
    if (!chosen.insert(t).second) chosen.insert(j);
  }
  out.assign(chosen.begin(), chosen.end());
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<std::string> LayoutNames(const std::string& spec) {
  if (spec == "both") return {"bucket", "id"};
  FUSION_CHECK(spec == "bucket" || spec == "id", "unknown layout '%s' (bucket, id or both)",
               spec.c_str());
  return {spec};
}

}  // namespace

void BuildIndex(const BuildParams& p) {
  FUSION_CHECK(!p.out_dir.empty(), "no output directory");
  MakeDirs(p.out_dir);
  auto data = VectorFile::Open(p.base_path, p.dtype, p.max_n);
  const uint64_t n = data->size();
  const uint32_t d = data->dim();
  const DType dt = data->dtype();
  const size_t vb = data->vec_bytes();
  auto path = [&](const std::string& f) { return JoinPath(p.out_dir, f); };
  auto need = [&](const std::string& f) { return p.overwrite || !FileExists(path(f)); };
  Log("building index for %llu x %u %s vectors from %s", (unsigned long long)n, d, DTypeName(dt),
      p.base_path.c_str());

  // Refuse to mix stages built from different inputs.
  if (!p.overwrite && FileExists(path(files::kMeta))) {
    IndexMeta old = IndexMeta::Load(path(files::kMeta));
    FUSION_CHECK(old.GetU64("num_vectors") == n && old.GetU64("dim") == d,
                 "%s holds an index for different data; use --overwrite", p.out_dir.c_str());
  }
  Timer total;

  // ---- Stage 1: hierarchical balanced clustering -> heads ------------------------------
  std::vector<uint32_t> heads;
  if (need(files::kHeads) || need(files::kCentroids)) {
    Timer t;
    Log("[1/5] hierarchical balanced clustering (target ratio %.3f)",
        p.clustering.centroid_ratio);
    ClusteringStats cs;
    heads = HierarchicalBalancedClustering(*data, p.clustering, &cs);
    Log("      %u posting lists (%.2f%% of vectors), leaf size min/mean/max = %u/%.1f/%u, "
        "%u levels, %.1fs",
        cs.num_leaves, 100.0 * cs.num_leaves / static_cast<double>(n), cs.min_leaf, cs.mean_leaf,
        cs.max_leaf, cs.levels, t.Sec());
    SaveU32Array(path(files::kHeads), heads);
    std::vector<uint8_t> hv(heads.size() * vb);
    for (size_t i = 0; i < heads.size(); ++i) std::memcpy(&hv[i * vb], data->Get(heads[i]), vb);
    SaveBin(path(files::kCentroids), dt, heads.size(), d, hv.data());
  } else {
    heads = LoadU32Array(path(files::kHeads));
    Log("[1/5] clustering: reusing %zu heads", heads.size());
  }
  const uint32_t num_heads = static_cast<uint32_t>(heads.size());
  VectorSet head_vecs = LoadVectors(path(files::kCentroids), dt);
  FUSION_CHECK(head_vecs.n == num_heads, "heads.bin and centroids.bin disagree");

  // ---- Stage 2: navigation graph over the heads ----------------------------------------
  auto space = MakeL2Space(dt, d);
  std::unique_ptr<HnswIndex> graph;
  if (need(files::kGraph)) {
    Timer t;
    Log("[2/5] navigation graph over %u heads (M=%u, efConstruction=%u)", num_heads, p.graph.m,
        p.graph.ef_construction);
    graph = BuildNavigationGraph(space.get(), head_vecs.data.data(), num_heads, vb, p.graph);
    graph->saveIndex(path(files::kGraph));
    Log("      done in %.1fs", t.Sec());
  } else {
    graph = std::make_unique<HnswIndex>(space.get(), path(files::kGraph));
    Log("[2/5] graph: reusing %s", path(files::kGraph).c_str());
  }

  // ---- Stage 3: replicated assignment -> posting lists (vector IDs only) ----------------
  PostingLists lists;
  std::vector<uint32_t> primary;
  AssignStats as;
  if (need(files::kPostings) || need(files::kPrimary)) {
    Timer t;
    Log("[3/5] assigning vectors (replicas<=%u, eps=%.3f, rng=%s x%.2f)", p.assign.replicas,
        p.assign.closure_eps, p.assign.rng ? "on" : "off", p.assign.rng_factor);
    AssignVectors(*data, graph.get(), head_vecs.data.data(), num_heads, p.assign, &lists, &primary,
                  &as);
    Log("      %.2f replicas/vector, list size mean/max = %.1f/%u, %u empty lists, %.1fs",
        as.avg_replicas, as.mean_list, as.max_list, as.empty_lists, t.Sec());
    lists.Save(path(files::kPostings));
    SaveU32Array(path(files::kPrimary), primary);
  } else {
    lists = PostingLists::Load(path(files::kPostings));
    primary = LoadU32Array(path(files::kPrimary));
    as.total_entries = lists.ids.size();
    as.avg_replicas = static_cast<double>(lists.ids.size()) / static_cast<double>(n);
    Log("[3/5] assignment: reusing posting lists (%.2f replicas/vector)", as.avg_replicas);
  }
  graph.reset();  // free memory before the PQ stage

  // ---- Stage 4: product quantization -> codes for the GPU tier --------------------------
  uint32_t pq_m = p.pq_m ? p.pq_m : DefaultPQM(d);
  if (need(files::kPQCodebook) || need(files::kPQCodes)) {
    Timer t;
    std::vector<uint64_t> idx = SampleIndices(n, p.pq_train_n, p.seed + 1);
    Log("[4/5] training PQ (m=%u, %u dims/sub-space) on %zu samples", pq_m, d / pq_m, idx.size());
    std::vector<float> train(idx.size() * d);
    DispatchDType(dt, [&](auto tag) {
      using T = decltype(tag);
      for (size_t i = 0; i < idx.size(); ++i) ToFloat(data->Row<T>(idx[i]), &train[i * d], d);
    });
    PQTrainParams tp;
    tp.m = pq_m;
    tp.iters = p.pq_iters;
    tp.seed = p.seed + 2;
    PQCodebook cb = TrainPQ(train.data(), idx.size(), d, tp);
    train.clear();
    train.shrink_to_fit();
    std::vector<uint8_t> codes(n * pq_m);
    DispatchDType(dt, [&](auto tag) {
      using T = decltype(tag);
#pragma omp parallel for schedule(static, 4096)
      for (int64_t i = 0; i < static_cast<int64_t>(n); ++i)
        cb.Encode(data->Row<T>(static_cast<uint64_t>(i)), &codes[static_cast<uint64_t>(i) * pq_m]);
    });
    cb.Save(path(files::kPQCodebook));
    SavePQCodes(path(files::kPQCodes), codes.data(), n, pq_m);
    Log("      %.1f MB of codes (%u B/vector), %.1fs", codes.size() / 1e6, pq_m, t.Sec());
  } else {
    pq_m = PQCodebook::Load(path(files::kPQCodebook)).m();
    Log("[4/5] PQ: reusing codebook (m=%u)", pq_m);
  }

  // ---- Stage 5: SSD layout ----------------------------------------------------------------
  for (const std::string& name : LayoutNames(p.layout)) {
    if (!need(files::LayoutMap(name)) && !need(files::LayoutVectors(name))) {
      Log("[5/5] layout '%s': reusing", name.c_str());
      continue;
    }
    Timer t;
    LayoutStats ls;
    LayoutMap map = name == "bucket"
                        ? BuildBucketLayout(primary, num_heads, static_cast<uint32_t>(vb),
                                            kPageSize, &ls)
                        : BuildSequentialLayout(n, static_cast<uint32_t>(vb), kPageSize, &ls);
    map.Save(path(files::LayoutMap(name)));
    WriteVectorPages(*data, map, path(files::LayoutVectors(name)));
    Log("[5/5] layout '%s': %llu pages (%llu full + %llu packed), %u vectors/page, fill %.1f%%, "
        "%.1fs",
        name.c_str(), (unsigned long long)map.num_pages, (unsigned long long)ls.full_pages,
        (unsigned long long)ls.packed_pages, map.vectors_per_page, 100.0 * ls.fill_ratio,
        t.Sec());
  }

  // ---- Metadata ----------------------------------------------------------------------------
  IndexMeta meta;
  if (FileExists(path(files::kMeta))) meta = IndexMeta::Load(path(files::kMeta));
  meta.Set("format_version", 1);
  meta.Set("source", p.base_path);
  meta.Set("dtype", DTypeName(dt));
  meta.Set("dim", d);
  meta.Set("num_vectors", n);
  meta.Set("vec_bytes", vb);
  meta.Set("num_heads", num_heads);
  meta.Set("total_postings", as.total_entries);
  meta.Set("avg_replicas", as.avg_replicas);
  meta.Set("pq_m", pq_m);
  meta.Set("page_size", kPageSize);
  std::string layouts;
  for (const char* name : {"bucket", "id"}) {
    if (FileExists(path(files::LayoutMap(name))) && FileExists(path(files::LayoutVectors(name))))
      layouts += std::string(layouts.empty() ? "" : ",") + name;
  }
  meta.Set("layouts", layouts);
  if (!meta.Has("centroid_ratio") || p.overwrite) {
    meta.Set("centroid_ratio", p.clustering.centroid_ratio);
    meta.Set("replicas", p.assign.replicas);
    meta.Set("closure_eps", p.assign.closure_eps);
    meta.Set("rng", p.assign.rng ? 1 : 0);
    meta.Set("rng_factor", p.assign.rng_factor);
    meta.Set("hnsw_m", p.graph.m);
    meta.Set("hnsw_ef_construction", p.graph.ef_construction);
  }
  meta.Save(path(files::kMeta));
  Log("index written to %s in %.1fs", p.out_dir.c_str(), total.Sec());
}

}  // namespace fusion
