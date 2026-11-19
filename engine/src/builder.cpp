#include "fusion/builder.h"

#include <algorithm>
#include <filesystem>
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

// A version's rows as the vector IDs they hold (VersionParams).
std::vector<uint32_t> ToIDs(std::vector<uint32_t> rows, const std::vector<uint32_t>& ids) {
  for (uint32_t& r : rows) r = ids[r];
  return rows;
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
  const VersionParams* ver = p.version.get();
  if (ver) {
    FUSION_CHECK(ver->ids.size() == n && ver->locs.size() == n, "the version names %zu IDs and %zu locations for %llu rows",
                 ver->ids.size(), ver->locs.size(), (unsigned long long)n);
    for (uint64_t r = 1; r < n; ++r) FUSION_CHECK(ver->ids[r] > ver->ids[r - 1], "the version's IDs must ascend");
    FUSION_CHECK(n == 0 || ver->ids.back() < ver->id_end, "vector ID %u beyond the version's %llu IDs", ver->ids.back(),
                 (unsigned long long)ver->id_end);
    // The codebook stays: the nodes' PQ codes were made with it, and are kept.
    if (!FileExists(path(files::kPQCodebook)))
      std::filesystem::copy_file(JoinPath(ver->from_index, files::kPQCodebook), path(files::kPQCodebook));
    Log("a data version of %s: %llu IDs handed out", ver->from_index.c_str(), (unsigned long long)ver->id_end);
  }
  const uint64_t id_space = ver ? ver->id_end : n;  // meta num_vectors

  // Refuse to mix stages built from different inputs.
  if (!p.overwrite && FileExists(path(files::kMeta))) {
    IndexMeta old = IndexMeta::Load(path(files::kMeta));
    FUSION_CHECK(old.GetU64("num_vectors") == id_space && old.GetU64("dim") == d,
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
    heads = HierarchicalBalancedClustering(*data, p.clustering, p.memory_bytes, &cs);
    Log("      %u posting lists (%.2f%% of vectors), leaf size min/mean/max = %u/%.1f/%u, "
        "%u levels, %.1fs",
        cs.num_leaves, 100.0 * cs.num_leaves / static_cast<double>(n), cs.min_leaf, cs.mean_leaf,
        cs.max_leaf, cs.levels, t.Sec());
    SaveU32Array(path(files::kHeads), ver ? ToIDs(heads, ver->ids) : heads);
    std::vector<uint8_t> hv(heads.size() * vb);
    data->Gather(heads.data(), heads.size(), hv.data());
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
    if (ver) {  // vector IDs, and the primary list of each (none for an ID no vector of the version has)
      for (uint32_t& id : lists.ids) id = ver->ids[id];
      std::vector<uint32_t> by_id(ver->id_end, UINT32_MAX);
      for (uint64_t r = 0; r < n; ++r) by_id[ver->ids[r]] = primary[r];
      primary.swap(by_id);
    }
    lists.Save(path(files::kPostings));
    SaveU32Array(path(files::kPrimary), primary);
    SyncAndDropFromPageCache(path(files::kPostings));
  } else {
    lists = PostingLists::Load(path(files::kPostings));
    primary = LoadU32Array(path(files::kPrimary));
    as.total_entries = lists.ids.size();
    as.avg_replicas = static_cast<double>(lists.ids.size()) / static_cast<double>(n);
    Log("[3/5] assignment: reusing posting lists (%.2f replicas/vector)", as.avg_replicas);
  }
  graph.reset();  // free memory before the PQ stage
  lists = PostingLists();
  head_vecs = VectorSet();

  // ---- Stage 4: product quantization -> codes for the GPU tier --------------------------
  uint32_t pq_m = p.pq_m ? p.pq_m : DefaultPQM(d);
  if (ver && need(files::kPQCodes)) {
    // The kept codebook's codes, by vector ID: the inserted vectors' are the ones the update
    // controller's router made with it (the same encoder), and an ID no vector of the version has
    // gets zeros, which no segment names.
    Timer t;
    PQCodebook cb = PQCodebook::Load(path(files::kPQCodebook));
    pq_m = cb.m();
    std::vector<uint8_t> codes(ver->id_end * pq_m, 0);
    DispatchDType(dt, [&](auto tag) {
      using T = decltype(tag);
      data->ForEachChunk([&](uint64_t b0, uint64_t b1, const uint8_t* rows) {
#pragma omp parallel for schedule(static, 4096)
        for (int64_t i = static_cast<int64_t>(b0); i < static_cast<int64_t>(b1); ++i)
          cb.Encode(reinterpret_cast<const T*>(rows + (static_cast<uint64_t>(i) - b0) * vb),
                    &codes[static_cast<uint64_t>(ver->ids[static_cast<uint64_t>(i)]) * pq_m]);
      });
    });
    SavePQCodes(path(files::kPQCodes), codes.data(), ver->id_end, pq_m);
    SyncAndDropFromPageCache(path(files::kPQCodes));
    Log("[4/5] PQ: %s's codebook (m=%u), codes of %llu vectors by ID, %.1fs", ver->from_index.c_str(), pq_m,
        (unsigned long long)n, t.Sec());
  } else if (need(files::kPQCodebook) || need(files::kPQCodes)) {
    Timer t;
    std::vector<uint64_t> idx = SampleIndices(n, p.pq_train_n, p.seed + 1);
    Log("[4/5] training PQ (m=%u, %u dims/sub-space) on %zu samples", pq_m, d / pq_m, idx.size());
    std::vector<float> train(idx.size() * d);
    DispatchDType(dt, [&](auto tag) {
      using T = decltype(tag);
      data->ForEachRow(idx.data(), idx.size(), [&](size_t i, const uint8_t* row) {
        ToFloat(reinterpret_cast<const T*>(row), &train[i * d], d);
      });
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
      data->ForEachChunk([&](uint64_t b0, uint64_t b1, const uint8_t* rows) {
#pragma omp parallel for schedule(static, 4096)
        for (int64_t i = static_cast<int64_t>(b0); i < static_cast<int64_t>(b1); ++i)
          cb.Encode(reinterpret_cast<const T*>(rows + (static_cast<uint64_t>(i) - b0) * vb),
                    &codes[static_cast<uint64_t>(i) * pq_m]);
      });
    });
    cb.Save(path(files::kPQCodebook));
    SavePQCodes(path(files::kPQCodes), codes.data(), n, pq_m);
    SyncAndDropFromPageCache(path(files::kPQCodes));
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
    if (ver) {
      // from_index's locations, and the inserted vectors' where the update controller put them:
      // after from_index's pages, which every node's raw store holds at the same offsets.
      LayoutMap map = LayoutMap::Load(JoinPath(ver->from_index, files::LayoutMap(name)));
      FUSION_CHECK(map.vec_bytes == vb && map.size() <= ver->id_end, "%s's layout '%s' is not of this data's vectors",
                   ver->from_index.c_str(), name.c_str());
      const uint64_t own = map.size(), first_free = map.num_pages * map.vectors_per_page;
      map.page_of.resize(ver->id_end, UINT32_MAX);
      map.slot_of.resize(ver->id_end, 0);
      uint64_t added = 0;
      for (uint64_t r = 0; r < n; ++r) {
        const uint32_t id = ver->ids[r], loc = ver->locs[r];
        if (loc == VersionParams::kNoLoc) {
          FUSION_CHECK(id < own, "vector %u has no location: neither %s's nor an inserted one", id,
                       ver->from_index.c_str());
          continue;
        }
        FUSION_CHECK(id >= own && loc >= first_free, "inserted vector %u at location %u: within %s's vectors or pages",
                     id, loc, ver->from_index.c_str());
        map.page_of[id] = loc / map.vectors_per_page;
        map.slot_of[id] = static_cast<uint16_t>(loc % map.vectors_per_page);
        map.num_pages = std::max<uint64_t>(map.num_pages, map.page_of[id] + uint64_t{1});
        ++added;
      }
      map.Save(path(files::LayoutMap(name)));
      // The page file: from_index's (the inserted vectors are on the nodes, not in it).
      std::filesystem::remove(path(files::LayoutVectors(name)));
      std::filesystem::create_symlink(std::filesystem::absolute(JoinPath(ver->from_index, files::LayoutVectors(name))),
                                      path(files::LayoutVectors(name)));
      Log("[5/5] layout '%s': %s's, and %llu inserted vectors' locations (%llu pages)", name.c_str(),
          ver->from_index.c_str(), (unsigned long long)added, (unsigned long long)map.num_pages);
      continue;
    }
    Timer t;
    LayoutStats ls;
    LayoutMap map = name == "bucket"
                        ? BuildBucketLayout(primary, num_heads, static_cast<uint32_t>(vb),
                                            kPageSize, &ls)
                        : BuildSequentialLayout(n, static_cast<uint32_t>(vb), kPageSize, &ls);
    map.Save(path(files::LayoutMap(name)));
    WriteVectorPages(*data, map, path(files::LayoutVectors(name)), p.memory_bytes);
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
  meta.Set("source", ver ? "a data version of " + ver->from_index + " (version.json)" : p.base_path);
  meta.Set("dtype", DTypeName(dt));
  meta.Set("dim", d);
  meta.Set("num_vectors", id_space);
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
    meta.Set("assign_candidates", p.assign.candidates);  // inserts follow the same rule (U14)
    meta.Set("assign_ef", p.assign.ef);
    meta.Set("hnsw_m", p.graph.m);
    meta.Set("hnsw_ef_construction", p.graph.ef_construction);
  }
  meta.Save(path(files::kMeta));
  if (ver) {  // what the version is made of: the runner checks it against the batches a run committed
    const std::string rec = path("version.json");
    FILE* f = std::fopen(rec.c_str(), "w");
    FUSION_CHECK(f != nullptr, "cannot write %s", rec.c_str());
    const bool ok = std::fwrite(ver->record.data(), 1, ver->record.size(), f) == ver->record.size();
    FUSION_CHECK(std::fclose(f) == 0 && ok, "cannot write %s", rec.c_str());
  }
  Log("index written to %s in %.1fs", p.out_dir.c_str(), total.Sec());
}

}  // namespace fusion
