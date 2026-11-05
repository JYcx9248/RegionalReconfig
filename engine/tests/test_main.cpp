// Unit and small end-to-end tests (CPU backend; no GPU needed).
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "fusion/builder.h"
#include "fusion/dataset.h"
#include "fusion/distance.h"
#include "fusion/engine.h"
#include "fusion/filter.h"
#include "fusion/index_meta.h"
#include "fusion/kmeans.h"
#include "fusion/layout.h"
#include "fusion/page_reader.h"
#include "fusion/pq.h"
#include "fusion/rerank.h"

using namespace fusion;

namespace {

int g_failed_checks = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::printf("    check failed at line %d: %s\n", __LINE__, #cond);    \
      ++g_failed_checks;                                                     \
    }                                                                        \
  } while (0)

std::string TempDir() {
  char tmpl[] = "/tmp/fusion_test_XXXXXX";
  char* d = mkdtemp(tmpl);
  if (!d) throw std::runtime_error("mkdtemp failed");
  return d;
}

// ---------------------------------------------------------------------------------------------
void TestDistances() {
  std::mt19937 rng(1);
  for (size_t d : {1u, 7u, 96u, 100u, 128u}) {
    std::vector<uint8_t> a(d), b(d);
    std::vector<int8_t> c(d), e(d);
    std::vector<float> f(d), g(d);
    for (size_t i = 0; i < d; ++i) {
      a[i] = rng() & 0xFF;
      b[i] = rng() & 0xFF;
      c[i] = static_cast<int8_t>(rng() & 0xFF);
      e[i] = static_cast<int8_t>(rng() & 0xFF);
      f[i] = static_cast<float>(rng() % 1000) / 7.f;
      g[i] = static_cast<float>(rng() % 1000) / 7.f;
    }
    double ru = 0, ri = 0, rf = 0;
    for (size_t i = 0; i < d; ++i) {
      ru += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
      ri += (double(c[i]) - e[i]) * (double(c[i]) - e[i]);
      rf += (double(f[i]) - g[i]) * (double(f[i]) - g[i]);
    }
    CHECK(L2Sqr(a.data(), b.data(), d) == static_cast<float>(ru));
    CHECK(L2Sqr(c.data(), e.data(), d) == static_cast<float>(ri));
    CHECK(std::fabs(L2Sqr(f.data(), g.data(), d) - rf) <= 1e-4 * std::max(1.0, rf));
    double rm = 0;
    for (size_t i = 0; i < d; ++i) rm += (a[i] - double(g[i])) * (a[i] - double(g[i]));
    CHECK(std::fabs(L2SqrMixed(a.data(), g.data(), d) - rm) <= 1e-5 * std::max(1.0, rm));
  }
}

void TestKMeans() {
  std::mt19937 rng(2);
  std::normal_distribution<float> nd(0.f, 0.1f);
  const float centers[4][2] = {{0, 0}, {10, 0}, {0, 10}, {10, 10}};
  std::vector<float> x;
  for (int c = 0; c < 4; ++c)
    for (int i = 0; i < 250; ++i) {
      x.push_back(centers[c][0] + nd(rng));
      x.push_back(centers[c][1] + nd(rng));
    }
  KMeansParams p;
  p.k = 4;
  p.iters = 20;
  auto cent = KMeansTrain(x.data(), 1000, 2, p);
  CHECK(cent.size() == 8);
  // Every true center must have a learned centroid close to it.
  for (auto& c : centers) {
    float best = 1e9f;
    for (int j = 0; j < 4; ++j) best = std::min(best, L2Sqr(c, &cent[j * 2], 2));
    CHECK(best < 0.1f);
  }
}

void TestPQ() {
  std::mt19937 rng(3);
  std::normal_distribution<float> nd;
  const uint32_t n = 4000, d = 32, m = 8;
  std::vector<float> x(n * d);
  for (auto& v : x) v = nd(rng);
  PQTrainParams tp;
  tp.m = m;
  tp.iters = 10;
  PQCodebook cb = TrainPQ(x.data(), n, d, tp);
  std::vector<float> q(d), lut(m * kPQKsub), dec(d);
  for (auto& v : q) v = nd(rng);
  cb.ComputeLUT(q.data(), lut.data());
  double err = 0, norm = 0;
  for (uint32_t i = 0; i < 200; ++i) {
    uint8_t code[8];
    cb.Encode(&x[i * d], code);
    cb.Decode(code, dec.data());
    float adc = AdcDistance(lut.data(), code, m);
    float ref = L2Sqr(q.data(), dec.data(), d);
    CHECK(std::fabs(adc - ref) <= 1e-3f * std::max(1.f, ref));
    err += L2Sqr(&x[i * d], dec.data(), d);
    norm += L2Sqr(&x[i * d], std::vector<float>(d, 0.f).data(), d);
  }
  CHECK(err / norm < 0.5);  // quantization must beat the trivial all-zero reconstruction

  // Save / load round trip.
  std::string dir = TempDir();
  cb.Save(dir + "/cb.bin");
  PQCodebook cb2 = PQCodebook::Load(dir + "/cb.bin");
  CHECK(cb2.m() == m && cb2.dim() == d);
  CHECK(std::memcmp(cb2.centroids(), cb.centroids(), sizeof(float) * m * kPQKsub * cb.dsub()) == 0);
}

void TestLayout() {
  std::mt19937 rng(4);
  // ~10 vectors per bucket, as with centroid_ratio = 0.1.
  const uint32_t buckets = 2003, vec_bytes = 128;
  const uint64_t n = 20011;
  std::vector<uint32_t> bucket_of(n);
  for (auto& b : bucket_of) b = static_cast<uint32_t>(rng() % buckets);
  LayoutStats ls;
  LayoutMap m = BuildBucketLayout(bucket_of, buckets, vec_bytes, kPageSize, &ls);
  const uint32_t vpp = m.vectors_per_page;
  CHECK(vpp == 32);
  std::set<std::pair<uint32_t, uint16_t>> used;
  for (uint64_t v = 0; v < n; ++v) {
    CHECK(m.page_of[v] < m.num_pages);
    CHECK(m.slot_of[v] < vpp);
    CHECK(used.insert({m.page_of[v], m.slot_of[v]}).second);  // no two vectors share a slot
  }
  CHECK(m.num_pages >= CeilDiv(n, vpp));
  // Packing quality: compare with a lower bound on the optimal number of shared pages
  // (total tail volume, and tails larger than half a page cannot share a page with each
  // other). First-fit-decreasing style packing stays within 11/9 * OPT + 1.
  {
    std::vector<uint32_t> cnt(buckets, 0);
    for (uint32_t b : bucket_of) cnt[b]++;
    uint64_t volume = 0, big = 0;
    for (uint32_t c : cnt) {
      volume += c % vpp;
      big += (c % vpp) * 2 > vpp;
    }
    const uint64_t lb = std::max<uint64_t>(CeilDiv(volume, vpp), big);
    std::printf("    %llu shared pages for %llu tails (lower bound %llu), fill %.1f%%\n",
                (unsigned long long)ls.packed_pages, (unsigned long long)ls.tails,
                (unsigned long long)lb, 100 * ls.fill_ratio);
    CHECK(ls.packed_pages * 9 <= lb * 11 + 9);
  }
  // Every page holds vectors of as few buckets as possible: full pages exactly one bucket.
  std::map<uint32_t, std::set<uint32_t>> buckets_on_page;
  for (uint64_t v = 0; v < n; ++v) buckets_on_page[m.page_of[v]].insert(bucket_of[v]);
  for (uint64_t p = 0; p < ls.full_pages; ++p) CHECK(buckets_on_page[static_cast<uint32_t>(p)].size() == 1);
  // A bucket's tail is never split: each bucket spans full pages + at most one shared page.
  std::map<uint32_t, std::set<uint32_t>> shared_pages;
  for (uint64_t v = 0; v < n; ++v)
    if (m.page_of[v] >= ls.full_pages) shared_pages[bucket_of[v]].insert(m.page_of[v]);
  for (auto& kv : shared_pages) CHECK(kv.second.size() == 1);

  LayoutMap s = BuildSequentialLayout(n, vec_bytes, kPageSize, nullptr);
  CHECK(s.page_of[33] == 1 && s.slot_of[33] == 1 && s.num_pages == CeilDiv(n, 32));
}

void TestPageReaders() {
  std::string dir = TempDir();
  const std::string path = dir + "/pages.bin";
  const uint32_t pages = 64;
  std::vector<uint8_t> data(pages * kPageSize);
  for (uint32_t p = 0; p < pages; ++p)
    for (uint32_t i = 0; i < kPageSize; ++i) data[p * kPageSize + i] = static_cast<uint8_t>(p * 7 + i);
  WriteFile(path, data.data(), data.size());
  PageFile f(path, kPageSize, /*direct=*/true);
  std::vector<IoBackend> backends = {IoBackend::kPread, IoBackend::kAio, IoBackend::kAuto};
#ifdef FUSION_HAVE_LIBURING
  backends.push_back(IoBackend::kUring);
#endif
  const std::vector<uint32_t> want = {5, 17, 3, 63, 0, 5, 42, 8, 9, 10, 11, 12};
  for (IoBackend b : backends) {
    std::unique_ptr<PageReader> r;
    try {
      r = MakePageReader(f, b, 4);  // depth 4 forces multiple submission rounds
    } catch (const std::exception& e) {
      std::printf("    (skipping %s: %s)\n", IoBackendName(b), e.what());
      continue;
    }
    AlignedBuffer buf = AllocAligned(4096, want.size() * kPageSize);
    std::vector<uint8_t*> ptrs;
    for (size_t i = 0; i < want.size(); ++i) ptrs.push_back(buf.get() + i * kPageSize);
    r->Read(want.data(), ptrs.data(), static_cast<uint32_t>(want.size()));
    for (size_t i = 0; i < want.size(); ++i)
      CHECK(std::memcmp(ptrs[i], &data[want[i] * kPageSize], kPageSize) == 0);
    std::printf("    %s reader ok (%s)\n", IoBackendName(r->backend()), f.direct() ? "O_DIRECT" : "buffered");
  }
}

void TestCpuFilter() {
  std::mt19937_64 rng(5);
  const uint32_t d = 16, m = 4;
  const uint64_t n = 1000;
  PQCodebook cb(d, m);
  std::normal_distribution<float> nd;
  for (size_t i = 0; i < m * kPQKsub * cb.dsub(); ++i) cb.mutable_centroids()[i] = nd(rng);
  std::vector<uint8_t> codes(n * m);
  for (auto& c : codes) c = rng() & 0xFF;
  FilterConfig fc;
  fc.max_candidates = 500;
  fc.max_topn = 50;
  auto f = CreateCpuFilter(cb, codes.data(), n, fc);
  FilterWorker* w = f->worker(0);
  std::vector<float> q(d);
  for (auto& v : q) v = nd(rng);
  // 100 distinct IDs, each repeated 3 times.
  for (uint32_t i = 0; i < 300; ++i) w->candidate_buffer()[i] = (i % 100) * 7;
  w->BeginQuery(q.data());
  uint32_t uniq = 0;
  uint32_t cnt = w->Filter(300, 50, &uniq);
  CHECK(uniq == 100);
  CHECK(cnt == 50);
  std::vector<float> lut(m * kPQKsub);
  cb.ComputeLUT(q.data(), lut.data());
  for (uint32_t i = 0; i < cnt; ++i) {
    uint32_t id = w->result_ids()[i];
    CHECK(id % 7 == 0 && id < 700);
    CHECK(w->result_dists()[i] == AdcDistance(lut.data(), &codes[id * m], m));
    if (i) CHECK(w->result_dists()[i - 1] <= w->result_dists()[i]);
  }
  CHECK(w->Filter(0, 50, &uniq) == 0 && uniq == 0);
}

void TestRerank() {
  // Float vectors stored in ID order.
  std::string dir = TempDir();
  const uint32_t d = 32, n = 256;  // 128 B vectors, 32 per page, 8 pages
  std::vector<float> vecs(n * d);
  std::mt19937 rng(6);
  std::normal_distribution<float> nd;
  for (auto& v : vecs) v = nd(rng);
  LayoutMap layout = BuildSequentialLayout(n, d * 4, kPageSize, nullptr);
  const std::string path = dir + "/vec.bin";
  std::vector<uint8_t> file(layout.num_pages * kPageSize, 0);
  for (uint32_t v = 0; v < n; ++v)
    std::memcpy(&file[layout.Offset(v) + layout.slot_of[v] * d * 4], &vecs[v * d], d * 4);
  WriteFile(path, file.data(), file.size());
  PageFile f(path, kPageSize, true);
  auto reader = MakePageReader(f, IoBackend::kAuto, 16);
  RerankScratch scratch(n, kPageSize);

  std::vector<float> q(d);
  for (auto& v : q) v = nd(rng);
  // Candidate list: all vectors, arbitrary order (a PQ ordering would be approximate anyway).
  std::vector<uint32_t> cand(n);
  for (uint32_t i = 0; i < n; ++i) cand[i] = (i * 37) % n;
  std::vector<std::pair<float, uint32_t>> truth;
  for (uint32_t v = 0; v < n; ++v) truth.emplace_back(L2Sqr(q.data(), &vecs[v * d], d), v);
  std::sort(truth.begin(), truth.end());

  RerankParams p;
  p.k = 10;
  p.heuristic = false;
  uint32_t ids[10];
  float dists[10];
  RerankStats st;
  uint32_t cnt = HeuristicRerank<float>(q.data(), d, cand.data(), n, layout, reader.get(), &scratch,
                                        p, ids, dists, &st);
  CHECK(cnt == 10);
  for (int i = 0; i < 10; ++i) CHECK(ids[i] == truth[i].second);
  CHECK(st.batches == n / 10 + 1);
  CHECK(st.reranked == n);
  CHECK(st.pages_read == layout.num_pages);  // every page read exactly once with dedup
  CHECK(st.intra_merged + st.buffer_hits + st.pages_read == n);

  // Without I/O dedup every vector costs one read.
  RerankStats st2;
  p.io_dedup = false;
  HeuristicRerank<float>(q.data(), d, cand.data(), n, layout, reader.get(), &scratch, p, ids, dists, &st2);
  CHECK(st2.pages_read == n);
  CHECK(ids[0] == truth[0].second);

  // eps = 1 means "always stable": stop right after the first full mini-batch.
  RerankStats st3;
  p.io_dedup = true;
  p.heuristic = true;
  p.eps = 1.f;
  HeuristicRerank<float>(q.data(), d, cand.data(), n, layout, reader.get(), &scratch, p, ids, dists, &st3);
  CHECK(st3.batches == 1 && st3.reranked == 10);

  // Candidates sorted by true distance: the top-k is found in batch 1 and batch 2 changes
  // nothing, so Algorithm 1 (eps=0.1, beta=1) stops after two mini-batches.
  std::vector<uint32_t> sorted(n);
  for (uint32_t i = 0; i < n; ++i) sorted[i] = truth[i].second;
  RerankStats st4;
  p.eps = 0.1f;
  HeuristicRerank<float>(q.data(), d, sorted.data(), n, layout, reader.get(), &scratch, p, ids, dists, &st4);
  CHECK(st4.batches == 2);
  for (int i = 0; i < 10; ++i) CHECK(ids[i] == truth[i].second);
}

// Gaussian-mixture data: realistic enough for clustering to matter.
void MakeMixture(uint64_t n, uint32_t d, uint32_t clusters, uint64_t seed, std::vector<float>* out,
                 std::vector<float>* centers) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd;
  if (centers->empty()) {
    centers->resize(static_cast<size_t>(clusters) * d);
    for (auto& c : *centers) c = nd(rng) * 4.f;
  }
  out->resize(n * d);
  for (uint64_t i = 0; i < n; ++i) {
    uint32_t c = static_cast<uint32_t>(rng() % clusters);
    for (uint32_t j = 0; j < d; ++j) (*out)[i * d + j] = (*centers)[c * d + j] + nd(rng);
  }
}

void TestEndToEnd() {
  const uint64_t n = 30000, nq = 200;
  const uint32_t d = 32, k = 10;
  std::vector<float> base, queries, centers;
  MakeMixture(n, d, 60, 11, &base, &centers);
  MakeMixture(nq, d, 60, 12, &queries, &centers);
  std::string dir = TempDir();
  SaveBin(dir + "/base.fbin", DType::kFloat, n, d, base.data());

  BuildParams bp;
  bp.base_path = dir + "/base.fbin";
  bp.out_dir = dir + "/index";
  bp.pq_m = 16;  // 2 dims per sub-space: PQ ordering good enough for the heuristic
  bp.pq_train_n = 10000;
  bp.layout = "both";
  BuildIndex(bp);

  // Exact ground truth.
  GroundTruth gt;
  gt.nq = nq;
  gt.k = k;
  for (uint64_t qi = 0; qi < nq; ++qi) {
    std::vector<std::pair<float, uint32_t>> all(n);
    for (uint64_t i = 0; i < n; ++i) all[i] = {L2Sqr(&queries[qi * d], &base[i * d], d), static_cast<uint32_t>(i)};
    std::partial_sort(all.begin(), all.begin() + k, all.end());
    for (uint32_t i = 0; i < k; ++i) {
      gt.ids.push_back(all[i].second);
      gt.dists.push_back(all[i].first);
    }
  }

  for (const char* layout : {"bucket", "id"}) {
    EngineOptions eo;
    eo.index_dir = bp.out_dir;
    eo.layout = layout;
    eo.device = FilterDevice::kCpu;
    eo.num_workers = 2;
    eo.max_nprobe = 64;
    eo.max_rerank = 200;
    auto engine = Engine::Open(eo);
    for (bool heuristic : {true, false}) {
      SearchParams sp;
      sp.k = k;
      sp.nprobe = 32;
      sp.rerank = 200;
      sp.rr.heuristic = heuristic;
      engine->Configure(sp);
      std::vector<uint32_t> ids(nq * k, kInvalidId);
      std::vector<float> dists(nq * k);
      double pages = 0, batches = 0;
      for (uint64_t qi = 0; qi < nq; ++qi) {
        QueryStats st;
        engine->Search(static_cast<int>(qi % 2), &queries[qi * d], sp, &ids[qi * k], &dists[qi * k], &st);
        pages += st.rerank.pages_read;
        batches += st.rerank.batches;
        CHECK(st.unique <= st.candidates);
        CHECK(st.lists == 32);
      }
      double recall = RecallAtK(gt, ids.data(), dists.data(), k, nq, k);
      std::printf("    layout=%-6s heuristic=%d recall@10=%.4f reads/query=%.1f batches/query=%.1f\n",
                  layout, heuristic, recall, pages / nq, batches / nq);
      CHECK(recall >= (heuristic ? 0.90 : 0.95));
    }
  }
}

// The base read with pread (VectorFile): whole, in chunks, and by ID in any order, from bin and
// vecs files (a dimension before each row) and a prefix of them. Rows out of order or past the
// end are refused.
void TestVectorFile() {
  std::string dir = TempDir();
  const uint64_t n = 5000;
  const uint32_t d = 1000;  // 1 KB rows: sparse IDs leave gaps wider than a span may cross
  std::vector<uint8_t> rows(n * d);
  std::mt19937 rng(7);
  for (auto& x : rows) x = static_cast<uint8_t>(rng());
  SaveBin(dir + "/a.u8bin", DType::kUInt8, n, d, rows.data());
  {
    FILE* f = std::fopen((dir + "/a.bvecs").c_str(), "wb");
    for (uint64_t i = 0; i < n; ++i) {
      const int32_t dd = static_cast<int32_t>(d);
      std::fwrite(&dd, 4, 1, f);
      std::fwrite(&rows[i * d], 1, d, f);
    }
    std::fclose(f);
  }
  auto row_is = [&](const uint8_t* got, uint64_t id) { return std::memcmp(got, &rows[id * d], d) == 0; };
  for (const char* name : {"/a.u8bin", "/a.bvecs"}) {
    auto vf = VectorFile::Open(dir + name);
    CHECK(vf->size() == n && vf->dim() == d && vf->vec_bytes() == d);
    std::vector<uint8_t> got(n * d);
    vf->Read(0, n, got.data());
    CHECK(got == rows);
    vf->Read(17, 4321, got.data());
    CHECK(std::memcmp(got.data(), &rows[17 * d], (4321 - 17) * d) == 0);
    uint64_t next = 0;
    bool chunks = true;
    vf->ForEachChunk([&](uint64_t b0, uint64_t b1, const uint8_t* r) {
      chunks = chunks && b0 == next && b1 > b0 && std::memcmp(r, &rows[b0 * d], (b1 - b0) * d) == 0;
      next = b1;
    });
    CHECK(chunks && next == n);
    // Dense IDs in any order with repeats, then sparse ones (each its own read).
    std::vector<uint32_t> ids;
    for (int i = 0; i < 3000; ++i) ids.push_back(static_cast<uint32_t>(rng() % n));
    ids.push_back(0);
    ids.push_back(static_cast<uint32_t>(n - 1));
    ids.push_back(ids[5]);
    for (const std::vector<uint32_t>& set : {ids, std::vector<uint32_t>{3, 700, 701, 1900, 4999}}) {
      std::vector<uint8_t> g(set.size() * d);
      vf->Gather(set.data(), set.size(), g.data());
      bool same = true;
      for (size_t i = 0; i < set.size(); ++i) same = same && row_is(&g[i * d], set[i]);
      CHECK(same);
    }
    std::vector<uint64_t> sorted = {1, 1, 2, 900, 3000, 3001};
    size_t calls = 0;
    bool in_order = true;
    vf->ForEachRow(sorted.data(), sorted.size(), [&](size_t i, const uint8_t* r) {
      in_order = in_order && i == calls++ && row_is(r, sorted[i]);
    });
    CHECK(in_order && calls == sorted.size());

    auto prefix = VectorFile::Open(dir + name, std::nullopt, 100);
    CHECK(prefix->size() == 100);
    auto refused = [&](std::vector<uint32_t> bad) {
      try {
        prefix->ForEachRow(bad.data(), bad.size(), [](size_t, const uint8_t*) {});
      } catch (const std::exception&) {
        return true;
      }
      return false;
    };
    CHECK(refused({50, 40}));
    CHECK(refused({100}));
    CHECK(!refused({40, 50, 99}));
  }
}

// The clustering with too little memory for its base: nodes too big are split with their rows
// streamed, the others clustered in batches. Its heads are those of the clustering in memory,
// whatever the budget, also when k-means cannot separate the points (all equal).
void TestStreamedClustering() {
  std::string dir = TempDir();
  ClusteringParams cp;
  cp.centroid_ratio = 0.1;
  cp.branch = 3;  // children of 20000 below the root, big enough to be streamed too
  auto same_heads = [&](const std::string& path, std::vector<uint64_t> budgets) {
    auto vf = VectorFile::Open(path);
    ClusteringStats want_st;
    std::vector<uint32_t> want = HierarchicalBalancedClustering(*vf, cp, uint64_t{1} << 30, &want_st);
    std::sort(want.begin(), want.end());
    bool same = true;
    for (uint64_t rows : budgets) {
      ClusteringStats st;
      std::vector<uint32_t> heads = HierarchicalBalancedClustering(*vf, cp, rows * vf->vec_bytes(), &st);
      std::sort(heads.begin(), heads.end());
      same = same && heads == want && st.num_leaves == want_st.num_leaves &&
             st.min_leaf == want_st.min_leaf && st.max_leaf == want_st.max_leaf;
      if (heads != want)
        std::printf("    budget %llu rows: %zu heads, %zu in memory\n", (unsigned long long)rows,
                    heads.size(), want.size());
    }
    return same;
  };
  const uint64_t n = 60000;
  const uint32_t d = 16;
  std::vector<float> base, centers;
  MakeMixture(n, d, 40, 21, &base, &centers);
  SaveBin(dir + "/base.fbin", DType::kFloat, n, d, base.data());
  // The root streamed, then batches of two children, of one, of single grandchildren.
  CHECK(same_heads(dir + "/base.fbin", {45000, 25000, 1000}));
  std::vector<float> equal(25000 * 4, 1.f);
  SaveBin(dir + "/equal.fbin", DType::kFloat, 25000, 4, equal.data());
  CHECK(same_heads(dir + "/equal.fbin", {1000}));
}

// The page file in windows of a few pages, each window's vectors read by ID: byte for byte the
// page file written in one window, with every vector where the map puts it.
void TestPageFileWindows() {
  const uint64_t n = 7001;
  const uint32_t d = 40;  // 160-byte vectors, 25 per page
  std::vector<float> base, centers;
  MakeMixture(n, d, 10, 31, &base, &centers);
  std::string dir = TempDir();
  SaveBin(dir + "/base.fbin", DType::kFloat, n, d, base.data());
  auto vf = VectorFile::Open(dir + "/base.fbin");
  std::mt19937 rng(5);
  std::vector<uint32_t> bucket_of(n);
  for (auto& b : bucket_of) b = static_cast<uint32_t>(rng() % 701);
  LayoutMap map = BuildBucketLayout(bucket_of, 701, d * 4, kPageSize, nullptr);
  WriteVectorPages(*vf, map, dir + "/one", uint64_t{1} << 30);
  WriteVectorPages(*vf, map, dir + "/windows", 3 * kPageSize);
  std::vector<uint8_t> one = ReadFile(dir + "/one"), windows = ReadFile(dir + "/windows");
  CHECK(one.size() == map.num_pages * kPageSize);
  CHECK(one == windows);
  bool placed = true;
  for (uint64_t v = 0; v < n; ++v)
    placed = placed && std::memcmp(&one[map.Offset(static_cast<uint32_t>(v)) + map.slot_of[v] * d * 4],
                                   &base[v * d], d * 4) == 0;
  CHECK(placed);
}

// A build with memory for a tenth of its base gives the same index: the same heads, and from the
// same heads and graph, the same posting lists, PQ codes, layouts and page files. (Two builds
// do not share a graph: its inserts run in parallel.)
void TestStreamedBuild() {
  const uint64_t n = 30000;
  const uint32_t d = 32;
  std::vector<float> base, centers;
  MakeMixture(n, d, 60, 41, &base, &centers);
  std::string dir = TempDir();
  SaveBin(dir + "/base.fbin", DType::kFloat, n, d, base.data());
  BuildParams bp;
  bp.base_path = dir + "/base.fbin";
  bp.pq_m = 16;
  bp.pq_train_n = 5000;
  bp.layout = "both";
  bp.out_dir = dir + "/whole";
  BuildIndex(bp);
  bp.memory_bytes = n * d * sizeof(float) / 10;
  bp.out_dir = dir + "/tenth";
  BuildIndex(bp);
  auto heads_of = [&](const std::string& index) {
    std::vector<uint32_t> h = LoadU32Array(JoinPath(index, files::kHeads));
    std::sort(h.begin(), h.end());
    return h;
  };
  CHECK(heads_of(dir + "/whole") == heads_of(dir + "/tenth"));
  bp.out_dir = dir + "/reuse";
  MakeDirs(bp.out_dir);
  for (const char* f : {files::kHeads, files::kCentroids, files::kGraph}) {
    std::vector<uint8_t> b = ReadFile(JoinPath(dir + "/whole", f));
    WriteFile(JoinPath(bp.out_dir, f), b.data(), b.size());
  }
  BuildIndex(bp);
  for (const std::string& f : {std::string(files::kPostings), std::string(files::kPrimary),
                               std::string(files::kPQCodebook), std::string(files::kPQCodes),
                               files::LayoutMap("bucket"), files::LayoutVectors("bucket"),
                               files::LayoutMap("id"), files::LayoutVectors("id")}) {
    const bool same = ReadFile(JoinPath(dir + "/whole", f)) == ReadFile(JoinPath(bp.out_dir, f));
    if (!same) std::printf("    %s differs\n", f.c_str());
    CHECK(same);
  }
}

}  // namespace

int main() {
  const std::vector<std::pair<const char*, std::function<void()>>> tests = {
      {"distances", TestDistances}, {"kmeans", TestKMeans},     {"pq", TestPQ},
      {"layout", TestLayout},       {"page_readers", TestPageReaders},
      {"cpu_filter", TestCpuFilter}, {"rerank", TestRerank},    {"end_to_end", TestEndToEnd},
      {"vector_file", TestVectorFile}, {"streamed_clustering", TestStreamedClustering},
      {"page_file_windows", TestPageFileWindows}, {"streamed_build", TestStreamedBuild},
  };
  int failed = 0;
  for (const auto& t : tests) {
    const int before = g_failed_checks;
    std::printf("[ run  ] %s\n", t.first);
    std::fflush(stdout);
    bool ok = true;
    try {
      t.second();
    } catch (const std::exception& e) {
      std::printf("    exception: %s\n", e.what());
      ok = false;
    }
    ok = ok && g_failed_checks == before;
    std::printf("[ %s ] %s\n", ok ? " ok " : "FAIL", t.first);
    failed += ok ? 0 : 1;
  }
  std::printf("%s: %d of %zu tests failed\n", failed ? "FAILED" : "PASSED", failed, tests.size());
  return failed ? 1 : 0;
}
