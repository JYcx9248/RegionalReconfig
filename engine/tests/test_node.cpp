// Tests for the rtier additions: partition files, NodeEngine primitives, the wire protocol.
#include <arpa/inet.h>
#include <fcntl.h>
#include <linux/magic.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <thread>

#include "../node/server.h"
#include "fusion/builder.h"
#include "fusion/dataset.h"
#include "fusion/index_meta.h"
#include "fusion/layout.h"
#include "fusion/node_engine.h"
#include "fusion/partition.h"
#include "fusion/raw_store.h"

using namespace fusion;

namespace {

int g_failed_checks = 0;
#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::printf("    check failed at line %d: %s\n", __LINE__, #cond); \
      ++g_failed_checks;                                                  \
    }                                                                     \
  } while (0)

std::string TempDir() {
  char tmpl[] = "/tmp/rtier_test_XXXXXX";
  char* d = mkdtemp(tmpl);
  if (!d) throw std::runtime_error("mkdtemp failed");
  return d;
}

void TestCrc32() {
  const char* s = "123456789";
  CHECK(Crc32(s, 9) == 0xCBF43926u);  // standard check value of CRC-32/IEEE
}

void TestPartitionFiles() {
  PostingLists pl;
  pl.offsets = {0, 2, 2, 5, 6};  // 4 lists, list 1 empty
  pl.ids = {7, 9, 1, 2, 3, 9};
  const std::vector<uint32_t> assign = {1, 0, 1, 0};
  const std::string dir = TempDir() + "/parts";
  // 10 vectors of 1 KB, 4 per page, stored in reverse ID order: vector v at location 9 - v.
  LayoutMap layout;
  layout.vec_bytes = 1024;
  layout.vectors_per_page = 4;
  layout.num_pages = 3;
  for (uint32_t v = 0; v < 10; ++v) {
    layout.page_of.push_back((9 - v) / 4);
    layout.slot_of.push_back(static_cast<uint16_t>((9 - v) % 4));
  }
  const auto sizes = WritePartitions(pl, assign, 2, dir, layout, "test");
  CHECK(sizes.size() == 2);
  PartitionManifest m = PartitionManifest::Load(dir);
  CHECK(m.num_lists == 4 && m.num_partitions == 2 && m.max_list_len == 3);
  CHECK(m.list_part == assign);
  CHECK(m.payload == kPayloadListsLocations && m.layout == "test");
  CHECK(m.raw.vec_bytes == 1024 && m.raw.vectors_per_page == 4 && m.raw.num_pages == 3 &&
        m.raw.page_size == kPageSize && m.raw.locations() == 12);
  ListSegment s1 = ListSegment::Load(JoinPath(dir, partfiles::SegmentName(1)));
  CHECK(s1.partition == 1);
  CHECK((s1.list_ids == std::vector<uint32_t>{0, 2}));
  uint32_t len = 0;
  const uint32_t* v = s1.Find(2, &len);
  CHECK(v != nullptr && len == 3 && v[0] == 1 && v[2] == 3);
  // Every posting carries its vector's location, the same wherever the vector appears.
  CHECK((s1.ids == std::vector<uint32_t>{7, 9, 1, 2, 3}));
  CHECK((s1.locs == std::vector<uint32_t>{2, 0, 8, 7, 6}));
  CHECK(s1.Find(1, &len) == nullptr);
  ListSegment s0 = ListSegment::Load(JoinPath(dir, partfiles::SegmentName(0)));
  v = s0.Find(1, &len);
  CHECK(v != nullptr && len == 0);
  // Corruption is detected.
  std::vector<uint8_t> raw = ReadFile(JoinPath(dir, partfiles::SegmentName(0)));
  raw[20] ^= 0xFF;
  WriteFile(JoinPath(dir, "bad.seg"), raw.data(), raw.size());
  bool threw = false;
  try {
    ListSegment::Load(JoinPath(dir, "bad.seg"));
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
  CHECK((s0.ids == std::vector<uint32_t>{9} && s0.locs == std::vector<uint32_t>{0}));
  // Offsets that point outside the segment are rejected even with a valid checksum.
  ListSegment bad = s0;
  bad.offsets = {0, 5, 3};
  bad.list_ids = {1, 3};
  bad.ids = {1, 2, 3, 4, 5};
  bad.locs = {1, 2, 3, 4, 5};
  bad.Save(JoinPath(dir, "bad2.seg"));
  threw = false;
  try {
    ListSegment::Load(JoinPath(dir, "bad2.seg"));
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

// A small index shared by the engine and wire tests.
struct Fixture {
  std::string index, parts;
  std::string index_nopages;  // symlinks to index, without vectors_bucket.bin
  uint32_t dim = 16, nq = 40, num_parts = 4;
  std::vector<float> queries;
};

Fixture& GetFixture() {
  static Fixture f = [] {
    Fixture x;
    const uint64_t n = 6000;
    std::mt19937_64 rng(5);
    std::normal_distribution<float> nd;
    std::vector<float> centers(40 * x.dim), base(n * x.dim);
    for (auto& c : centers) c = nd(rng) * 4.f;
    auto sample = [&](std::vector<float>* out, uint64_t cnt) {
      out->resize(cnt * x.dim);
      for (uint64_t i = 0; i < cnt; ++i) {
        const uint64_t c = rng() % 40;
        for (uint32_t j = 0; j < x.dim; ++j) (*out)[i * x.dim + j] = centers[c * x.dim + j] + nd(rng);
      }
    };
    sample(&base, n);
    sample(&x.queries, x.nq);
    const std::string dir = TempDir();
    SaveBin(dir + "/base.fbin", DType::kFloat, n, x.dim, base.data());
    BuildParams bp;
    bp.base_path = dir + "/base.fbin";
    bp.out_dir = dir + "/index";
    bp.pq_m = 8;
    bp.pq_train_n = 4000;
    BuildIndex(bp);
    x.index = bp.out_dir;
    // Placeholder assignment for tests only: contiguous ranges of list IDs.
    PostingLists pl = PostingLists::Load(JoinPath(x.index, files::kPostings));
    std::vector<uint32_t> assign(pl.num_lists());
    for (uint32_t c = 0; c < pl.num_lists(); ++c)
      assign[c] = static_cast<uint32_t>(uint64_t(c) * x.num_parts / pl.num_lists());
    x.parts = dir + "/parts";
    WritePartitions(pl, assign, x.num_parts, x.parts,
                    LayoutMap::Load(JoinPath(x.index, files::LayoutMap("bucket"))), "bucket");
    // The same index without its page file: a node opened on it can only get raw vectors from
    // peers, which shows that nothing but bootstrapping reads the index's pages.
    x.index_nopages = dir + "/index-nopages";
    MakeDirs(x.index_nopages);
    for (const char* name : {files::kMeta, files::kHeads, files::kCentroids, files::kGraph,
                             files::kPostings, files::kPrimary, files::kPQCodebook, files::kPQCodes})
      if (FileExists(JoinPath(x.index, name)))
        CHECK(::symlink(JoinPath(x.index, name).c_str(), JoinPath(x.index_nopages, name).c_str()) == 0);
    return x;
  }();
  return f;
}

std::unique_ptr<NodeEngine> OpenNode(const Fixture& f, const std::vector<uint32_t>& parts,
                                     bool graph, const std::string& index_dir = "",
                                     uint64_t read_iops = 0) {
  NodeOptions o;
  o.index_dir = index_dir.empty() ? f.index : index_dir;
  o.partitions_dir = f.parts;
  o.num_workers = 2;
  o.max_nprobe = 64;
  o.max_rerank = 4096;
  o.direct_io = false;
  o.read_iops = read_iops;
  auto e = NodeEngine::Open(o);
  if (graph) e->LoadGraph(JoinPath(f.index, files::kGraph));
  for (uint32_t p : parts)
    e->LoadPartition(p, JoinPath(f.parts, partfiles::SegmentName(p)), PQSource::kIndex);
  return e;
}

using Result = std::vector<std::pair<float, uint32_t>>;

// In-process stand-in for the node server's peer pool: raw vectors are fetched by calling the
// named peer's RawGet directly.
void LinkPeers(NodeEngine* node, std::map<std::string, NodeEngine*> peers) {
  node->SetRawFetcher([peers](const std::string& peer, const uint32_t* locs, const uint32_t* lists,
                              size_t n, uint8_t* out) {
    auto it = peers.find(peer);
    if (it == peers.end()) throw std::runtime_error("unknown peer " + peer);
    it->second->RawGet(locs, lists, n, out);
  });
}

// For each of `ids`, the first list of partition `part` that names it (deduplicated): the lists
// a FILTER over that partition would have gathered those candidates from.
std::vector<uint32_t> ListsNaming(const Fixture& f, uint32_t part, const std::vector<uint32_t>& ids) {
  const ListSegment s = ListSegment::Load(JoinPath(f.parts, partfiles::SegmentName(part)));
  std::vector<uint32_t> out;
  for (uint32_t id : ids)
    for (size_t i = 0; i < s.list_ids.size(); ++i)
      if (std::find(s.ids.begin() + s.offsets[i], s.ids.begin() + s.offsets[i + 1], id) !=
          s.ids.begin() + s.offsets[i + 1]) {
        out.push_back(s.list_ids[i]);
        break;
      }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

// A list of partition `part` that names location `loc` (kInvalidId if none).
uint32_t ListNamingLoc(const Fixture& f, uint32_t part, uint32_t loc) {
  const ListSegment s = ListSegment::Load(JoinPath(f.parts, partfiles::SegmentName(part)));
  for (size_t i = 0; i < s.list_ids.size(); ++i)
    for (uint64_t j = s.offsets[i]; j < s.offsets[i + 1]; ++j)
      if (s.locs[j] == loc) return s.list_ids[i];
  return kInvalidId;
}

// Distinct IDs and locations named by the given partitions' segments.
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> Named(const Fixture& f,
                                                              const std::vector<uint32_t>& parts) {
  std::vector<uint32_t> ids, locs;
  for (uint32_t p : parts) {
    const ListSegment s = ListSegment::Load(JoinPath(f.parts, partfiles::SegmentName(p)));
    ids.insert(ids.end(), s.ids.begin(), s.ids.end());
    locs.insert(locs.end(), s.locs.begin(), s.locs.end());
  }
  for (auto* v : {&ids, &locs}) {
    std::sort(v->begin(), v->end());
    v->erase(std::unique(v->begin(), v->end()), v->end());
  }
  return {ids, locs};
}

void TestNodePrimitives() {
  Fixture& f = GetFixture();
  auto all = OpenNode(f, {0, 1, 2, 3}, true);
  auto left = OpenNode(f, {0, 1}, false);  // two "data nodes" holding half the partitions each
  auto right = OpenNode(f, {2, 3}, false);
  const uint32_t k = 10, nprobe = 16;
  PartitionManifest man = PartitionManifest::Load(f.parts);
  for (uint32_t qi = 0; qi < f.nq; ++qi) {
    const float* q = &f.queries[qi * f.dim];
    // Single-node oracle: exhaustive (every probed candidate re-ranked, no early stop).
    SearchParams sp;
    sp.k = k;
    sp.nprobe = nprobe;
    sp.rerank = 4096;
    sp.rr.heuristic = false;
    std::vector<uint32_t> ids(k);
    std::vector<float> d(k);
    QueryStats qs;
    const uint32_t n0 = all->SearchLocal(0, q, sp, ids.data(), d.data(), &qs);
    Result oracle;
    for (uint32_t i = 0; i < n0; ++i) oracle.emplace_back(d[i], ids[i]);

    // Same query spread over two nodes: navigate, filter per owner, re-rank per owner, merge.
    std::vector<uint32_t> lists(nprobe);
    const uint32_t nl = all->Navigate(q, nprobe, 0, lists.data());
    CHECK(nl == nprobe);
    std::map<uint32_t, std::vector<uint32_t>> by_node;  // 0 = left, 1 = right
    for (uint32_t i = 0; i < nl; ++i) by_node[man.list_part[lists[i]] < 2 ? 0 : 1].push_back(lists[i]);
    std::map<uint32_t, float> merged;  // id -> exact distance (dedup across nodes)
    for (auto& [node, ls] : by_node) {
      NodeEngine* e = node == 0 ? left.get() : right.get();
      std::vector<uint32_t> cand(4096);
      std::vector<float> pqd(4096);
      FilterStats fs;
      const uint32_t nc = e->Filter(0, q, ls.data(), static_cast<uint32_t>(ls.size()), 4096,
                                    cand.data(), pqd.data(), &fs);
      CHECK(nc == fs.unique);
      for (uint32_t i = 1; i < nc; ++i)
        CHECK(pqd[i - 1] < pqd[i] || (pqd[i - 1] == pqd[i] && cand[i - 1] < cand[i]));
      std::vector<uint32_t> rid(k);
      std::vector<float> rd(k);
      RerankStats rs;
      const uint32_t nr = e->Rerank(1, q, cand.data(), nc, k, ls.data(),
                                    static_cast<uint32_t>(ls.size()), rid.data(), rd.data(), &rs);
      for (uint32_t i = 0; i < nr; ++i) merged[rid[i]] = rd[i];
    }
    Result dist;
    for (auto& [id, dd] : merged) dist.emplace_back(dd, id);
    std::sort(dist.begin(), dist.end());
    if (dist.size() > k) dist.resize(k);
    CHECK(dist == oracle);
  }

  // Residency errors and eviction.
  std::vector<uint32_t> lists(4);
  const uint32_t nl = all->Navigate(&f.queries[0], 4, 0, lists.data());
  CHECK(nl == 4);
  bool not_resident = false, no_graph = false;
  CHECK(all->EvictPartition(man.list_part[lists[0]]));
  CHECK(!all->EvictPartition(man.list_part[lists[0]]));
  try {
    std::vector<uint32_t> ids(10);
    std::vector<float> d(10);
    FilterStats fs;
    all->Filter(0, &f.queries[0], lists.data(), nl, 10, ids.data(), d.data(), &fs);
  } catch (const NotResidentError&) {
    not_resident = true;
  }
  CHECK(not_resident);
  try {
    left->Navigate(&f.queries[0], 4, 0, lists.data());
  } catch (const NoGraphError&) {
    no_graph = true;
  }
  CHECK(no_graph);
  CHECK(all->ResidentPartitions().size() == 3);
}

void TestPQStore() {
  PQStore s(100, 2, 5);  // IDs 0..99, 2-byte codes, room for 5 codes
  auto codes_of = [](const std::vector<uint32_t>& ids) {
    std::vector<uint8_t> c;
    for (uint32_t id : ids) {
      c.push_back(static_cast<uint8_t>(id));
      c.push_back(static_cast<uint8_t>(id * 3));
    }
    return c;
  };
  auto throws_capacity = [&](const std::vector<uint32_t>& ids) {
    try {
      s.Put(ids.data(), ids.size(), codes_of(ids).data(), PQOrigin::kReceived);
    } catch (const PQCapacityError&) {
      return true;
    }
    return false;
  };
  std::vector<uint32_t> mirrored;
  s.SetMirror([&](const uint32_t* slots, uint32_t n) { mirrored.assign(slots, slots + n); });

  // New codes arrive staged; a repeated ID is installed once.
  const std::vector<uint32_t> a = {7, 3, 7, 9};
  CHECK((s.Missing(a.data(), a.size()) == std::vector<uint32_t>{3, 7, 9}));
  CHECK(s.Put(a.data(), a.size(), codes_of(a).data(), PQOrigin::kReceived) == 3);
  CHECK(mirrored.size() == 3 && std::is_sorted(mirrored.begin(), mirrored.end()));
  PQStats st = s.Stats();
  CHECK(st.resident == 3 && st.staged == 3 && st.received == 3 && st.skipped == 1);
  CHECK(s.Missing(a.data(), a.size()).empty());

  // Nothing cached: 3 new codes do not fit in the 2 free slots, and nothing changes.
  const std::vector<uint32_t> b = {1, 2, 4, 3};
  CHECK(throws_capacity(b) && s.Stats().resident == 3 && s.Missing(b.data(), b.size()).size() == 3);

  // A mirror failure undoes the Put.
  s.SetMirror([](const uint32_t*, uint32_t) { throw std::runtime_error("device copy failed"); });
  const std::vector<uint32_t> one = {50};
  bool threw = false;
  try {
    s.Put(one.data(), 1, codes_of(one).data(), PQOrigin::kReceived);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw && s.Stats().resident == 3 && s.Missing(one.data(), 1).size() == 1);
  s.SetMirror(nullptr);

  // One reference per posting makes codes live; Get copies codes out.
  const std::vector<uint32_t> postings = {3, 7, 7};
  std::vector<uint32_t> slots(3);
  s.Ref(postings.data(), postings.size(), slots.data());
  CHECK(slots[1] == slots[2] && slots[0] != slots[1]);
  st = s.Stats();
  CHECK(st.live == 2 && st.staged == 1 && st.cached == 0);
  std::vector<uint8_t> out(4);
  const std::vector<uint32_t> g = {7, 3};
  s.Get(g.data(), g.size(), out.data());
  CHECK(out[0] == 7 && out[1] == 21 && out[2] == 3 && out[3] == 9);

  // A staging that did not load: the code installed for it (9) is freed.
  CHECK(s.ReleaseStaged() == 1 && s.Stats().resident == 2 && s.Stats().staged == 0);
  const std::vector<uint32_t> nine = {9};
  threw = false;
  try {
    s.Get(nine.data(), 1, out.data());
  } catch (const PQMissingError&) {
    threw = true;
  }
  CHECK(threw);
  threw = false;
  try {
    s.Ref(nine.data(), 1, nullptr);
  } catch (const PQMissingError&) {
    threw = true;
  }
  CHECK(threw);

  // A partition moving away leaves its codes cached, not freed (static dataset).
  s.Unref(postings.data(), 2);  // 3, and one of 7's two references
  st = s.Stats();
  CHECK(st.live == 1 && st.cached == 1 && st.resident == 2);
  s.Unref(postings.data() + 2, 1);
  st = s.Stats();
  CHECK(st.live == 0 && st.cached == 2 && st.resident == 2 && st.freed == 1);
  const std::vector<uint32_t> both = {3, 7};
  CHECK(s.Missing(both.data(), both.size()).empty());

  // Cached codes make room only when needed: 3 free slots, 4 new codes -> one is dropped.
  const std::vector<uint32_t> four = {20, 21, 22, 23};
  CHECK(s.Put(four.data(), four.size(), codes_of(four).data(), PQOrigin::kReceived) == 4);
  st = s.Stats();
  CHECK(st.resident == 5 && st.staged == 4 && st.cached == 1 && st.evicted == 1);
  const auto gone = s.Missing(both.data(), both.size());
  CHECK(gone.size() == 1);
  const uint32_t kept = gone.empty() || gone[0] == 7 ? 3 : 7;

  // Live and staged codes are never dropped: 2 more codes need 2 slots, only 1 is cached.
  CHECK(throws_capacity({30, 31}) && s.Stats().resident == 5 && s.Stats().evicted == 1);

  // A staging protects the cached codes it needs: they cannot be dropped for its new codes.
  const std::vector<uint32_t> want = {kept, 40};
  CHECK((s.Reserve(want.data(), want.size()) == std::vector<uint32_t>{40}));
  st = s.Stats();
  CHECK(st.cached == 0 && st.staged == 5);
  CHECK(throws_capacity({40}) && s.Stats().resident == 5);

  // Rollback: the codes installed for the staging go, the protected one returns to the cache.
  CHECK(s.ReleaseStaged() == 4);
  st = s.Stats();
  CHECK(st.resident == 1 && st.cached == 1 && st.staged == 0);
  const std::vector<uint32_t> kept_v = {kept};
  s.Get(kept_v.data(), 1, out.data());
  CHECK(out[0] == kept && out[1] == static_cast<uint8_t>(kept * 3));

  // Freed slots are reused.
  const std::vector<uint32_t> idx = {10, 11, 12, 13};
  CHECK(s.Put(idx.data(), idx.size(), codes_of(idx).data(), PQOrigin::kIndex) == 4);
  const std::vector<uint32_t> thirteen = {13};
  s.Get(thirteen.data(), 1, out.data());
  CHECK(out[0] == 13 && out[1] == 39 && s.Stats().from_index == 4 && s.Stats().resident == 5);
}

// Moving partitions between two NodeEngines the way an agent does: PQMissing on the
// destination, PQGet on the source, PQPut, then LoadPartition(kPresent).
void TestPQMigration() {
  Fixture& f = GetFixture();
  auto seg = [&](uint32_t p) { return JoinPath(f.parts, partfiles::SegmentName(p)); };
  auto distinct = [&](const std::vector<uint32_t>& parts) {
    std::vector<uint32_t> v;
    for (uint32_t p : parts) {
      const ListSegment s = ListSegment::Load(seg(p));
      v.insert(v.end(), s.ids.begin(), s.ids.end());
    }
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
  };
  auto move_codes = [](NodeEngine* from, NodeEngine* to, const std::vector<uint32_t>& ids) {
    std::vector<uint8_t> c(ids.size() * from->pq_stats().m);
    from->PQGet(ids.data(), ids.size(), c.data());
    return to->PQPut(ids.data(), ids.size(), c.data());
  };

  auto src = OpenNode(f, {0, 1, 2}, true);  // bootstrapped from the index
  const auto d012 = distinct({0, 1, 2}), d12 = distinct({1, 2}), d1 = distinct({1});
  PQStats st = src->pq_stats();
  CHECK(st.resident == d012.size() && st.from_index == d012.size() && st.received == 0);
  size_t postings = 0;
  for (uint32_t p : {0u, 1u, 2u}) postings += ListSegment::Load(seg(p)).ids.size();
  CHECK(postings > d012.size());  // boundary replication: fewer codes than postings

  auto dst = OpenNode(f, {}, false);
  LinkPeers(dst.get(), {{"src", src.get()}});
  bool threw = false;
  try {  // raw vectors missing, and no source to fetch them from
    dst->LoadPartition(1, seg(1), PQSource::kPresent);
  } catch (const RawMissingError&) {
    threw = true;
  }
  CHECK(threw && dst->ResidentPartitions().empty() && dst->pq_stats().resident == 0);
  threw = false;
  try {
    dst->LoadPartition(1, seg(1), PQSource::kPresent, "src");
  } catch (const PQMissingError&) {
    threw = true;
  }
  CHECK(threw && dst->ResidentPartitions().empty() && dst->pq_stats().resident == 0 &&
        dst->raw_stats(true).pending == 0);

  // Two sources: the needs are split so that no code is fetched twice.
  const auto miss = dst->PQMissing({{seg(1)}, {seg(2)}});
  CHECK(miss.size() == 2 && miss[0] == d1 && miss[0].size() + miss[1].size() == d12.size());
  for (const auto& ids : miss) {
    const PQPutResult r = move_codes(src.get(), dst.get(), ids);
    CHECK(r.installed == ids.size() && r.skipped == 0);
  }
  CHECK(dst->pq_stats().staged == d12.size());
  dst->LoadPartition(1, seg(1), PQSource::kPresent, "src");
  dst->LoadPartition(2, seg(2), PQSource::kPresent, "src");
  st = dst->pq_stats();
  CHECK(st.resident == d12.size() && st.live == d12.size() && st.staged == 0 &&
        st.received == d12.size() && st.from_index == 0);

  // Another partition later: only the codes the node lacks are listed.
  const auto miss0 = dst->PQMissing({{seg(0)}});
  CHECK(miss0[0].size() == d012.size() - d12.size());
  move_codes(src.get(), dst.get(), miss0[0]);
  dst->LoadPartition(0, seg(0), PQSource::kPresent, "src");
  CHECK(dst->pq_stats().resident == d012.size());

  // Same PQ top-n as the source for every query (codes are addressed by slot on both nodes,
  // with different slots).
  PartitionManifest man = PartitionManifest::Load(f.parts);
  for (uint32_t qi = 0; qi < f.nq; ++qi) {
    const float* q = &f.queries[qi * f.dim];
    std::vector<uint32_t> lists(16), mine;
    const uint32_t nl = src->Navigate(q, 16, 0, lists.data());
    for (uint32_t i = 0; i < nl; ++i)
      if (man.list_part[lists[i]] <= 2) mine.push_back(lists[i]);
    std::vector<uint32_t> ia(200), ib(200);
    std::vector<float> da(200), db(200);
    FilterStats fa, fb;
    const uint32_t na = src->Filter(0, q, mine.data(), static_cast<uint32_t>(mine.size()), 200,
                                    ia.data(), da.data(), &fa);
    const uint32_t nb = dst->Filter(0, q, mine.data(), static_cast<uint32_t>(mine.size()), 200,
                                    ib.data(), db.data(), &fb);
    CHECK(na == nb && ia == ib && da == db);
  }

  // Evicting partition 0 keeps its codes: those only partition 0 named become cached.
  CHECK(dst->EvictPartition(0));
  st = dst->pq_stats();
  CHECK(st.resident == d012.size() && st.live == d12.size() &&
        st.cached == d012.size() - d12.size());
  // Bringing partition 0 back needs no transfer at all.
  CHECK(dst->PQMissing({{seg(0)}})[0].empty());
  dst->LoadPartition(0, seg(0), PQSource::kPresent, "src");
  st = dst->pq_stats();
  CHECK(st.live == d012.size() && st.cached == 0 && st.staged == 0 && st.received == d012.size());
  CHECK(dst->EvictPartition(0));

  // Aborted staging: the codes installed for it are freed, the cached codes it protected go
  // back to the cache.
  auto full = OpenNode(f, {0, 1, 2, 3}, false);
  const auto miss3 = dst->PQMissing({{seg(3)}});
  CHECK(!miss3[0].empty());
  const auto d3 = distinct({3});
  size_t claimed = 0;  // cached codes (named only by partition 0) that partition 3 also names
  for (uint32_t id : d3)
    if (std::binary_search(d012.begin(), d012.end(), id) &&
        !std::binary_search(d12.begin(), d12.end(), id))
      ++claimed;
  move_codes(full.get(), dst.get(), miss3[0]);
  CHECK(dst->pq_stats().staged == miss3[0].size() + claimed);
  CHECK(dst->PQReleaseStaged() == miss3[0].size());
  st = dst->pq_stats();
  CHECK(st.resident == d012.size() && st.staged == 0 && st.cached == d012.size() - d12.size());

  // A node whose budget is smaller than a partition's vectors cannot load it.
  NodeOptions o;
  o.index_dir = f.index;
  o.partitions_dir = f.parts;
  o.max_nprobe = 64;
  o.max_rerank = 4096;
  o.direct_io = false;
  o.pq_capacity = d1.size() - 1;
  auto small = NodeEngine::Open(o);
  threw = false;
  try {
    small->LoadPartition(1, seg(1), PQSource::kIndex);
  } catch (const PQCapacityError&) {
    threw = true;
  }
  CHECK(threw && small->ResidentPartitions().empty() && small->pq_stats().resident == 0);
}

// Queries race evictions, agent-style stagings (PQMissing, PQGet on another node, PQPut,
// LoadPartition) and releases on a node whose PQ budget forces cached codes out and slots to be
// reused. Every Filter that succeeds must equal the answer of a node holding everything;
// afterwards exactly the codes of the resident partitions are live.
void TestPQEvictUnderLoad() {
  Fixture& f = GetFixture();
  auto seg = [&](uint32_t p) { return JoinPath(f.parts, partfiles::SegmentName(p)); };
  auto oracle = OpenNode(f, {0, 1, 2, 3}, true);
  std::vector<std::set<uint32_t>> ids_of(f.num_parts);
  for (uint32_t p = 0; p < f.num_parts; ++p) {
    const ListSegment s = ListSegment::Load(seg(p));
    ids_of[p].insert(s.ids.begin(), s.ids.end());
  }
  std::set<uint32_t> half = ids_of[0];
  half.insert(ids_of[1].begin(), ids_of[1].end());
  NodeOptions o;
  o.index_dir = f.index;
  o.partitions_dir = f.parts;
  o.num_workers = 2;
  o.max_nprobe = 64;
  o.max_rerank = 4096;
  o.direct_io = false;
  o.pq_capacity = half.size() + 200;  // about two of the four partitions fit
  auto node = NodeEngine::Open(o);
  LinkPeers(node.get(), {{"oracle", oracle.get()}});
  PartitionManifest man = PartitionManifest::Load(f.parts);
  std::vector<std::vector<uint32_t>> qlists(f.nq);
  for (uint32_t qi = 0; qi < f.nq; ++qi) {
    qlists[qi].resize(32);
    qlists[qi].resize(oracle->Navigate(&f.queries[qi * f.dim], 32, 0, qlists[qi].data()));
  }

  std::mutex oracle_mu;  // the oracle's worker 0 is shared by the query threads
  std::atomic<bool> stop{false};
  std::atomic<long> ok{0}, bad{0}, errors{0}, stagings{0};
  auto query_thread = [&](int wi) {
    std::mt19937 r(wi * 77 + 1);
    std::vector<uint32_t> ia(4096), ib(4096);
    std::vector<float> da(4096), db(4096);
    while (!stop.load()) {
      const uint32_t qi = r() % f.nq;
      const uint32_t mask = r() & 0xF;  // lists of a random subset of partitions
      std::vector<uint32_t> ls;
      for (uint32_t c : qlists[qi])
        if (mask & (1u << man.list_part[c])) ls.push_back(c);
      if (ls.empty()) continue;
      FilterStats fa, fb;
      uint32_t na = 0, nb = 0;
      try {
        na = node->Filter(wi, &f.queries[qi * f.dim], ls.data(), static_cast<uint32_t>(ls.size()),
                          4096, ia.data(), da.data(), &fa);
      } catch (const NotResidentError&) {
        continue;
      } catch (const std::exception& e) {
        if (errors++ == 0) std::printf("    query error: %s\n", e.what());
        continue;
      }
      {
        std::lock_guard<std::mutex> l(oracle_mu);
        nb = oracle->Filter(0, &f.queries[qi * f.dim], ls.data(), static_cast<uint32_t>(ls.size()),
                            4096, ib.data(), db.data(), &fb);
      }
      const bool same = na == nb && std::equal(ia.begin(), ia.begin() + na, ib.begin()) &&
                        std::equal(da.begin(), da.begin() + na, db.begin());
      (same ? ok : bad)++;
    }
  };
  std::vector<std::thread> threads;
  for (int w = 0; w < 2; ++w) threads.emplace_back(query_thread, w);

  std::mt19937 r(12345);
  const uint32_t m = oracle->pq_stats().m;
  long absent = 0;
  for (int op = 0; op < 400; ++op) {
    const uint32_t p = r() % f.num_parts;
    const int action = r() % 10;
    try {
      if (action < 4) {
        node->EvictPartition(p);
      } else if (action < 8) {  // agent-style staging, codes installed in reverse order
        const auto miss = node->PQMissing({{seg(p)}});
        std::vector<uint32_t> ids(miss[0].rbegin(), miss[0].rend());
        std::vector<uint8_t> codes(ids.size() * m);
        oracle->PQGet(ids.data(), ids.size(), codes.data());
        const size_t h = ids.size() / 2;
        node->PQPut(ids.data(), h, codes.data());
        node->PQPut(ids.data() + h, ids.size() - h, codes.data() + h * m);
        node->LoadPartition(p, seg(p), PQSource::kPresent, "oracle");
        stagings++;
      } else if (action < 9) {
        node->LoadPartition(p, seg(p), PQSource::kIndex);
      } else {
        node->PQReleaseStaged();
      }
    } catch (const PQCapacityError&) {
      node->PQReleaseStaged();  // an aborted staging cleans up
    } catch (const PQMissingError& e) {
      // Cannot happen: the codes a staging counted as present are protected.
      if (absent++ == 0) std::printf("    %s\n", e.what());
    }
  }
  stop = true;
  for (auto& t : threads) t.join();

  node->PQReleaseStaged();
  std::set<uint32_t> want;
  for (uint32_t p : node->ResidentPartitions()) want.insert(ids_of[p].begin(), ids_of[p].end());
  const PQStats st = node->pq_stats();
  if (bad || errors || absent || st.live != want.size())
    std::printf("    %ld ok, %ld mismatches, %ld errors, %ld absent, %ld stagings; %llu live codes, "
                "want %zu\n", ok.load(), bad.load(), errors.load(), absent, stagings.load(),
                (unsigned long long)st.live, want.size());
  CHECK(bad == 0 && errors == 0 && absent == 0 && ok > 0 && stagings > 0);
  CHECK(st.live == want.size() && st.staged == 0 && st.resident == st.live + st.cached);
  CHECK(st.resident <= o.pq_capacity && st.evicted > 0);  // the budget forced cached codes out
}

// direct: installs write whole pages with O_DIRECT (reading back a page that already holds
// vectors); otherwise buffered writes of the vectors alone.
void RawStoreCase(bool direct) {
  RawLayout L;
  L.vec_bytes = 1000;  // 4 per 4 KB page, 96 bytes unused at the end of each
  L.vectors_per_page = 4;
  L.num_pages = 5;  // 20 locations
  RawStore s("", L, direct);
  if (direct && !s.pages().direct()) std::printf("    no O_DIRECT in TMPDIR: buffered installs only\n");
  auto vec = [](uint32_t loc) {
    std::vector<uint8_t> v(1000);
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<uint8_t>(loc * 7 + i);
    return v;
  };
  auto vecs = [&](const std::vector<uint32_t>& locs) {
    std::vector<uint8_t> out;
    for (uint32_t l : locs) {
      const auto v = vec(l);
      out.insert(out.end(), v.begin(), v.end());
    }
    return out;
  };
  // A batch with a repeated location and a run across a page boundary (3 | 4 5).
  const std::vector<uint32_t> a = {9, 3, 9, 4, 5, 13};
  CHECK((s.Missing(a.data(), a.size()) == std::vector<uint32_t>{3, 4, 5, 9, 13}));
  CHECK(s.Put(a.data(), a.size(), vecs(a).data(), RawOrigin::kStreamed) == 5);
  RawStats st = s.Stats();
  CHECK(st.present == 5 && st.streamed == 5 && st.skipped == 1 && st.locations == 20 &&
        st.vec_bytes == 1000);
  CHECK(s.Missing(a.data(), a.size()).empty() && s.Present(13) && !s.Present(12));
  const std::vector<uint32_t> b = {4, 6};
  CHECK(s.Put(b.data(), b.size(), vecs(b).data(), RawOrigin::kFetched) == 1);
  CHECK(s.Stats().fetched == 1 && s.Stats().present == 6);
  // Get keeps the order asked for, repeats included.
  const std::vector<uint32_t> g = {13, 4, 13, 6, 3};
  std::vector<uint8_t> out(g.size() * 1000);
  s.Get(g.data(), g.size(), out.data());
  CHECK(out == vecs(g) && s.Stats().served == 5);
  bool threw = false;
  const std::vector<uint32_t> absent = {6, 7};
  try {
    s.Get(absent.data(), absent.size(), out.data());
  } catch (const RawMissingError&) {
    threw = true;
  }
  CHECK(threw && s.Stats().served == 5);
  threw = false;
  const std::vector<uint32_t> far = {20};
  try {
    s.Missing(far.data(), 1);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  // Re-ranking reads whole pages from the same file: page 1 holds locations 4, 5 and 6.
  auto reader = MakePageReader(s.pages(), IoBackend::kPread, 4);
  AlignedBuffer page = AllocAligned(4096, 4096);
  const uint32_t one = 1;
  uint8_t* bufs[1] = {page.get()};
  reader->Read(&one, bufs, 1);
  for (uint32_t loc : {4u, 5u, 6u})
    CHECK(std::memcmp(page.get() + (loc - 4) * 1000, vec(loc).data(), 1000) == 0);
  // Installs one at a time into pages that already hold vectors keep those (a direct install
  // rewrites the whole page), and fill a page completely.
  for (uint32_t loc : {7u, 0u, 2u, 1u, 12u}) CHECK(s.Put(&loc, 1, vec(loc).data(), RawOrigin::kFetched) == 1);
  const std::vector<uint32_t> all = {0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 13};
  std::vector<uint8_t> back(all.size() * 1000);
  s.Get(all.data(), all.size(), back.data());
  CHECK(back == vecs(all) && s.Stats().present == all.size());
  const uint32_t first = 0;
  reader->Read(&first, bufs, 1);
  for (uint32_t loc = 0; loc < 4; ++loc) CHECK(std::memcmp(page.get() + loc * 1000, vec(loc).data(), 1000) == 0);
  // CountAbsent: of 3, 7, 8, 11 and 19, the last three are not here.
  std::vector<uint64_t> mask(1, 0);
  for (uint32_t loc : {3u, 7u, 8u, 11u, 19u}) mask[0] |= uint64_t{1} << loc;
  CHECK(s.CountAbsent(mask) == 3);
}

// A Get spread over many pages (600), with repeated locations, comes back complete and in order.
void RawStoreManyPages(bool direct) {
  RawLayout L;
  L.vec_bytes = 1000;
  L.vectors_per_page = 4;
  L.num_pages = 600;
  RawStore s("", L, direct);
  std::vector<uint32_t> all(L.locations());
  std::iota(all.begin(), all.end(), 0u);
  auto vec = [](uint32_t loc, uint8_t* out) {
    for (size_t i = 0; i < 1000; ++i) out[i] = static_cast<uint8_t>(loc * 13 + i);
  };
  std::vector<uint8_t> vecs(all.size() * 1000);
  for (uint32_t l : all) vec(l, vecs.data() + static_cast<size_t>(l) * 1000);
  CHECK(s.Put(all.data(), all.size(), vecs.data(), RawOrigin::kStreamed) == all.size());
  std::vector<uint32_t> want;
  for (uint32_t i = 0; i < 1500; ++i) want.push_back((i * 7919u) % L.locations());  // spread, repeats
  std::vector<uint8_t> out(want.size() * 1000), ref(1000);
  s.Get(want.data(), want.size(), out.data());
  bool same = true;
  for (size_t i = 0; i < want.size(); ++i) {
    vec(want[i], ref.data());
    same = same && std::memcmp(out.data() + i * 1000, ref.data(), 1000) == 0;
  }
  CHECK(same);
}

void TestRawStore() {
  RawStoreCase(false);
  RawStoreCase(true);
  RawStoreManyPages(false);
  RawStoreManyPages(true);
}

// Stages partitions on `to` the way an agent does after a migration: the PQ codes it lacks from
// `from`, then loads with `peer` (the name of `from`) as the source of the raw vectors.
void StageFrom(const Fixture& f, NodeEngine* to, NodeEngine* from, const std::vector<uint32_t>& parts,
               const std::string& peer) {
  std::vector<std::string> paths;
  for (uint32_t p : parts) paths.push_back(JoinPath(f.parts, partfiles::SegmentName(p)));
  const auto miss = to->PQMissing({paths});
  std::vector<uint8_t> codes(miss[0].size() * from->pq_stats().m);
  from->PQGet(miss[0].data(), miss[0].size(), codes.data());
  to->PQPut(miss[0].data(), miss[0].size(), codes.data());
  for (size_t i = 0; i < parts.size(); ++i) to->LoadPartition(parts[i], paths[i], PQSource::kPresent, peer);
}

// Filters at `at` over the query's lists of `parts`, re-ranks the top 200 at `at` and at `ref`,
// and reports whether the answers agree.
bool SameRerank(const Fixture& f, uint32_t qi, NodeEngine* at, NodeEngine* ref,
                const std::vector<uint32_t>& parts) {
  static const PartitionManifest man = PartitionManifest::Load(GetFixture().parts);
  const float* q = &f.queries[qi * f.dim];
  std::vector<uint32_t> lists(16), mine;
  const uint32_t nl = ref->Navigate(q, 16, 0, lists.data());
  for (uint32_t i = 0; i < nl; ++i)
    if (std::find(parts.begin(), parts.end(), man.list_part[lists[i]]) != parts.end())
      mine.push_back(lists[i]);
  if (mine.empty()) return true;
  std::vector<uint32_t> cand(200), ia(10), ib(10);
  std::vector<float> pqd(200), da(10), db(10);
  FilterStats fs;
  const uint32_t nc = at->Filter(0, q, mine.data(), static_cast<uint32_t>(mine.size()), 200,
                                 cand.data(), pqd.data(), &fs);
  RerankStats ra, rb;
  const uint32_t nm = static_cast<uint32_t>(mine.size());
  const uint32_t na = at->Rerank(0, q, cand.data(), nc, 10, mine.data(), nm, ia.data(), da.data(), &ra);
  const uint32_t nb = ref->Rerank(0, q, cand.data(), nc, 10, mine.data(), nm, ib.data(), db.data(), &rb);
  return na == nb && ia == ib && da == db;
}

// Raw vectors after a migration (U9): the new owner goes online without them, re-ranking fetches
// the ones it needs from the old owner on demand -- through the source of the query's lists --
// and only those; a stream (the lazy-stream baseline) can install the rest; the answers are the
// old owner's throughout. The new nodes' index has no page file at all.
void TestRawMigration() {
  Fixture& f = GetFixture();
  auto seg = [&](uint32_t p) { return JoinPath(f.parts, partfiles::SegmentName(p)); };
  auto src = OpenNode(f, {0, 1, 2, 3}, true);
  const auto locs_all = Named(f, {0, 1, 2, 3}).second;
  CHECK(src->raw_stats(true).present == locs_all.size() &&
        src->raw_stats(true).from_index == locs_all.size());

  auto dst = OpenNode(f, {}, false, f.index_nopages);
  LinkPeers(dst.get(), {{"src", src.get()}});
  StageFrom(f, dst.get(), src.get(), {2, 3}, "src");
  const auto locs23 = Named(f, {2, 3}).second;
  RawStats rs = dst->raw_stats(true);
  CHECK(rs.present == 0 && rs.pending == locs23.size() && rs.from_index == 0);

  // Bootstrapping needs the index's pages, which this index does not have.
  bool threw = false;
  try {
    dst->LoadPartition(0, seg(0), PQSource::kIndex);
  } catch (const std::exception&) {
    threw = true;
  }
  dst->PQReleaseStaged();
  CHECK(threw && dst->ResidentPartitions().size() == 2);

  // Online at once: the answers are the old owner's, the vectors fetched as needed.
  bool same = true;
  for (uint32_t qi = 0; qi < f.nq / 2; ++qi) same = SameRerank(f, qi, dst.get(), src.get(), {2, 3}) && same;
  CHECK(same);
  rs = dst->raw_stats(true);
  CHECK(rs.fetched > 0 && rs.fetches > 0 && rs.present == rs.fetched &&
        rs.pending == locs23.size() - rs.present);

  // A node that takes partition 3 from dst before dst has all its vectors: dst fetches what it
  // lacks itself from src on the way (chained), and the answers still agree.
  auto third = OpenNode(f, {}, false, f.index_nopages);
  LinkPeers(third.get(), {{"dst", dst.get()}});
  StageFrom(f, third.get(), dst.get(), {3}, "dst");
  const uint64_t dst_fetched = dst->raw_stats(true).fetched;
  same = true;
  for (uint32_t qi = 0; qi < f.nq; ++qi) same = SameRerank(f, qi, third.get(), src.get(), {3}) && same;
  CHECK(same && third->raw_stats(true).fetched > 0 && dst->raw_stats(true).fetched > dst_fetched);

  // What nothing needed stays missing (the lazy protocol stops here); a stream brings the rest:
  // exactly the vectors still missing, once, each with a list that names it.
  const auto miss = dst->RawMissing({{seg(2)}, {seg(3)}});
  rs = dst->raw_stats(true);
  CHECK(rs.pending > 0);
  CHECK(miss.size() == 2 && miss[0].locs.size() + miss[1].locs.size() == locs23.size() - rs.present &&
        miss[1].locs.size() + miss[0].locs.size() == rs.pending);
  for (size_t g = 0; g < miss.size(); ++g)
    for (size_t i = 0; i < miss[g].locs.size(); ++i)
      CHECK(ListNamingLoc(f, g == 0 ? 2 : 3, miss[g].locs[i]) != kInvalidId);
  for (const auto& m : miss) {
    std::vector<uint8_t> v(m.locs.size() * dst->vec_bytes());
    src->RawGet(m.locs.data(), m.lists.data(), m.locs.size(), v.data());
    const RawPutResult pr = dst->RawPut(m.locs.data(), m.locs.size(), v.data());
    CHECK(pr.installed == m.locs.size() && pr.skipped == 0);
  }
  rs = dst->raw_stats(true);
  CHECK(rs.present == locs23.size() && rs.pending == 0 &&
        rs.streamed == miss[0].locs.size() + miss[1].locs.size());
  CHECK(dst->RawMissing({{seg(2), seg(3)}})[0].locs.empty());
  same = true;
  for (uint32_t qi = 0; qi < f.nq; ++qi) same = SameRerank(f, qi, dst.get(), src.get(), {2, 3}) && same;
  CHECK(same && dst->raw_stats(true).fetched == rs.fetched);  // nothing left to fetch

  // The same stream pulled by the node itself (RawPull, data node to data node): exactly the
  // vectors it lacks, installed as streamed and not as fetched on demand; a second pull brings
  // nothing.
  auto pulled = OpenNode(f, {}, false, f.index_nopages);
  LinkPeers(pulled.get(), {{"src", src.get()}});
  StageFrom(f, pulled.get(), src.get(), {2, 3}, "src");
  const auto want = pulled->RawMissing({{seg(2)}, {seg(3)}});
  uint64_t got = 0;
  for (const auto& m : want) {
    const RawPutResult pr = pulled->RawPull(m.locs.data(), m.lists.data(), m.locs.size());
    CHECK(pr.installed + pr.skipped == m.locs.size());
    got += pr.installed;
  }
  RawStats ps = pulled->raw_stats(true);
  CHECK(got == locs23.size() && ps.present == got && ps.streamed == got && ps.fetched == 0 &&
        ps.fetches == 0 && ps.pending == 0);
  for (const auto& m : want) {
    const RawPutResult pr = pulled->RawPull(m.locs.data(), m.lists.data(), m.locs.size());
    CHECK(pr.installed == 0 && pr.skipped == m.locs.size());
  }
  same = true;
  for (uint32_t qi = 0; qi < f.nq; ++qi) same = SameRerank(f, qi, pulled.get(), src.get(), {2, 3}) && same;
  CHECK(same && pulled->raw_stats(true).fetched == 0);

  // Chains as RERANK's fetches do: pulling partition 3 from `third`, which holds only the vectors
  // its queries fetched, brings them all -- third fetches what it lacks from dst on the way.
  auto fourth = OpenNode(f, {}, false, f.index_nopages);
  LinkPeers(fourth.get(), {{"third", third.get()}});
  StageFrom(f, fourth.get(), third.get(), {3}, "third");
  const auto want3 = fourth->RawMissing({{seg(3)}});
  const uint64_t third_fetched = third->raw_stats(true).fetched;
  CHECK(third->raw_stats(true).pending > 0);
  const RawPutResult p3 = fourth->RawPull(want3[0].locs.data(), want3[0].lists.data(), want3[0].locs.size());
  CHECK(p3.installed == want3[0].locs.size() && fourth->RawMissing({{seg(3)}})[0].locs.empty() &&
        third->raw_stats(true).fetched > third_fetched);

  // A partition that moves away leaves its vectors (a cache, like its PQ codes): it comes back
  // with nothing to transfer, and no source is needed.
  CHECK(dst->EvictPartition(2));
  CHECK(dst->raw_stats(true).present == locs23.size() && dst->RawMissing({{seg(2)}})[0].locs.empty());
  dst->LoadPartition(2, seg(2), PQSource::kPresent);
  CHECK(dst->ResidentPartitions().size() == 2 && dst->raw_stats(true).pending == 0);

  // The whole pipeline on a node that holds every partition but no vector yet: SearchLocal
  // fetches before re-ranking and returns the single-node answer.
  auto whole = OpenNode(f, {}, true, f.index_nopages);
  LinkPeers(whole.get(), {{"src", src.get()}});
  StageFrom(f, whole.get(), src.get(), {0, 1, 2, 3}, "src");
  same = true;
  for (uint32_t qi = 0; qi < f.nq; ++qi) {
    SearchParams sp;
    sp.k = 10;
    sp.nprobe = 16;
    sp.rerank = 400;
    sp.rr.heuristic = false;
    std::vector<uint32_t> ia(10), ib(10);
    std::vector<float> da(10), db(10);
    QueryStats qa, qb;
    const float* q = &f.queries[qi * f.dim];
    const uint32_t na = whole->SearchLocal(1, q, sp, ia.data(), da.data(), &qa);
    const uint32_t nb = src->SearchLocal(1, q, sp, ib.data(), db.data(), &qb);
    same = same && na == nb && ia == ib && da == db;
  }
  CHECK(same && whole->raw_stats(true).fetched > 0 && whole->raw_stats(true).fetched == whole->raw_stats(true).present);
}

// Drops a file's pages from the page cache; false where that is not possible (tmpfs keeps a
// file in its pages).
bool EvictFromCache(const std::string& path) {
  struct statfs sf;
  if (::statfs(path.c_str(), &sf) != 0 || sf.f_type == TMPFS_MAGIC) return false;
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return false;
  const bool ok = ::fdatasync(fd) == 0 && ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) == 0;
  ::close(fd);
  return ok;
}

// The read budget (rtier_node --read-iops): reads are spaced to the rate once a burst is used
// up, a raw store charges only the reads that reach the device, and a node whose budget binds
// returns the answers of one without.
void TestReadBudget() {
  using Clock = std::chrono::steady_clock;
  auto since = [](Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); };
  {
    const double burst = 1000 * ReadBudget::kBurstSeconds;  // reads an idle budget lets through
    ReadBudget b(1000);
    auto t0 = Clock::now();
    b.Charge(static_cast<uint64_t>(burst / 2));
    CHECK(since(t0) < 0.01);
    t0 = Clock::now();
    b.Charge(static_cast<uint64_t>(burst / 2) + 200);  // 200 reads past the burst: 200 ms
    CHECK(since(t0) >= 0.19);
  }
  {
    ReadBudget b(20000);  // 12000 reads from 4 threads: (12000 - burst) / 20000 s at least
    const double least = (12000 - 20000 * ReadBudget::kBurstSeconds) / 20000;
    const auto t0 = Clock::now();
    std::vector<std::thread> ts;
    for (int i = 0; i < 4; ++i)
      ts.emplace_back([&b] {
        for (int j = 0; j < 3000; ++j) b.Charge(1);
      });
    for (auto& t : ts) t.join();
    const double s = since(t0);
    const ReadBudgetStats st = b.Stats();
    CHECK(s >= least - 0.01 && s < least + 3.0 && st.charged == 12000 && st.wait_us > 0 &&
          st.reads_per_sec == 20000);
  }
  {
    RawLayout L;
    L.vec_bytes = 1000;
    L.vectors_per_page = 4;
    L.num_pages = 4;
    const std::string path = TempDir() + "/raw.pages";
    RawStore s(path, L, true);
    ReadBudget b(1000000);
    s.SetReadBudget(&b);
    std::vector<uint8_t> v(3 * 1000, 7), out(3 * 1000);
    const std::vector<uint32_t> three = {0, 1, 2};
    CHECK(s.Put(three.data(), 2, v.data(), RawOrigin::kStreamed) == 2);  // buffered: cached
    s.Get(three.data(), 2, out.data());
    CHECK(b.Stats().charged == 0);
    if (s.pages().direct()) {
      CHECK(s.Put(&three[2], 1, v.data(), RawOrigin::kFetched) == 1);  // page 0 is read back
      CHECK(b.Stats().charged == 1);
    } else {
      std::printf("    no O_DIRECT in TMPDIR: no read-back to charge\n");
      CHECK(s.Put(&three[2], 1, v.data(), RawOrigin::kFetched) == 1);
    }
    const uint64_t before = b.Stats().charged;
    if (EvictFromCache(path)) {
      s.Get(three.data(), 3, out.data());  // one run on one page: one read
      CHECK(b.Stats().charged == before + 1 && out == v);
    } else {
      std::printf("    TMPDIR cannot drop cached pages: cache misses not checked\n");
    }
  }
  {
    Fixture& f = GetFixture();
    auto ref = OpenNode(f, {0, 1, 2, 3}, true);
    auto node = OpenNode(f, {0, 1, 2, 3}, true, "", 2000);  // a burst of 200 reads
    bool same = true;
    for (uint32_t qi = 0; qi < f.nq; ++qi) {
      SearchParams sp;
      sp.k = 10;
      sp.nprobe = 16;
      sp.rerank = 400;
      sp.rr.heuristic = false;
      std::vector<uint32_t> ia(10), ib(10);
      std::vector<float> da(10), db(10);
      QueryStats qa, qb;
      const float* q = &f.queries[qi * f.dim];
      const uint32_t na = node->SearchLocal(0, q, sp, ia.data(), da.data(), &qa);
      const uint32_t nb = ref->SearchLocal(0, q, sp, ib.data(), db.data(), &qb);
      same = same && na == nb && ia == ib && da == db;
    }
    const ReadBudgetStats rs = node->read_budget_stats();
    CHECK(same && rs.reads_per_sec == 2000 && rs.charged > 400 && rs.wait_us > 0 &&
          ref->read_budget_stats().charged == 0 && ref->read_budget_stats().reads_per_sec == 0);
  }
}

// Minimal blocking client for the wire test.
rtier::Frame Call(int fd, rtier::Op op, const rtier::BodyWriter& w, uint64_t id) {
  rtier::Frame req;
  req.type = op;
  req.req_id = id;
  req.epoch = 7;
  req.body = w.buf;
  rtier::WriteFrame(fd, req);
  rtier::Frame resp;
  if (!rtier::ReadFrame(fd, &resp)) throw std::runtime_error("server closed the connection");
  return resp;
}

int Connect(int port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
  if (fd < 0 || ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
    throw std::runtime_error("cannot connect to the test server");
  return fd;
}

void TestWire() {
  Fixture& f = GetFixture();
  auto e = OpenNode(f, {0, 1, 2, 3}, true);
  rtier::NodeServer server(e.get());
  const int port = server.ListenTcp("127.0.0.1", 0);
  std::thread serve([&] { server.Serve(); });

  const int fd = Connect(port);

  rtier::BodyWriter empty;
  rtier::Frame r = Call(fd, rtier::kPing, empty, 1);
  CHECK(r.status == rtier::kOk && r.req_id == 1 && r.epoch == 7 && (r.flags & rtier::kFlagResponse));
  r = Call(fd, rtier::kInfo, empty, 2);
  const std::string info(r.body.begin(), r.body.end());
  CHECK(info.find("\"dim\":16") != std::string::npos);
  CHECK(info.find("\"graph_loaded\":true") != std::string::npos);

  rtier::BodyWriter nav;
  nav.Put<uint32_t>(8);
  nav.Put<uint32_t>(0);
  nav.PutBytes(&f.queries[0], f.dim * sizeof(float));
  r = Call(fd, rtier::kNavigate, nav, 3);
  CHECK(r.status == rtier::kOk);
  rtier::BodyReader br(r.body.data(), r.body.size());
  const uint32_t n = br.Get<uint32_t>();
  CHECK(n == 8 && br.remaining() == 8 * 4);

  rtier::BodyWriter bad;  // query of the wrong length
  bad.Put<uint32_t>(8);
  bad.Put<uint32_t>(0);
  bad.PutBytes(&f.queries[0], 3);
  r = Call(fd, rtier::kNavigate, bad, 4);
  CHECK(r.status == rtier::kBadRequest);

  rtier::BodyWriter ev;
  ev.Put<uint32_t>(0);
  r = Call(fd, rtier::kEvictPartition, ev, 5);
  CHECK(r.status == rtier::kOk && r.body.size() == 1 && r.body[0] == 1);
  for (uint32_t p = 1; p < f.num_parts; ++p) {  // evict the rest: the codes stay, cached
    rtier::BodyWriter evp;
    evp.Put<uint32_t>(p);
    r = Call(fd, rtier::kEvictPartition, evp, 50 + p);
    CHECK(r.status == rtier::kOk && r.body.size() == 1 && r.body[0] == 1);
  }
  PQStats st = e->pq_stats();
  CHECK(st.live == 0 && st.resident > 0 && st.cached == st.resident);
  const std::string seg0 = JoinPath(f.parts, partfiles::SegmentName(0));
  std::vector<uint32_t> ids0 = ListSegment::Load(seg0).ids;
  std::sort(ids0.begin(), ids0.end());
  ids0.erase(std::unique(ids0.begin(), ids0.end()), ids0.end());
  auto load0 = [&](uint8_t pq_source, const std::string& raw_peer) {
    rtier::BodyWriter w;
    w.Put<uint32_t>(0);
    w.Put<uint8_t>(pq_source);
    w.Put<uint32_t>(static_cast<uint32_t>(raw_peer.size()));
    w.PutBytes(raw_peer.data(), raw_peer.size());
    w.PutBytes(seg0.data(), seg0.size());
    return w;
  };
  // The codes and vectors are still here: partition 0 loads again without any transfer.
  r = Call(fd, rtier::kLoadPartition, load0(0, ""), 6);
  CHECK(r.status == rtier::kOk && e->pq_stats().live == ids0.size());

  // A node without the codes or vectors (and without the index's pages): with no source for
  // the vectors the load is refused; with one, the load reports the codes absent; PQ_MISSING
  // lists them and PQ_PUT installs them (as a peer would send them); then the load succeeds.
  const std::string peer1 = "127.0.0.1:" + std::to_string(port);
  auto fresh = OpenNode(f, {}, false, f.index_nopages);
  rtier::NodeServer server2(fresh.get());
  const int port2 = server2.ListenTcp("127.0.0.1", 0);
  std::thread serve2([&] { server2.Serve(); });
  const int fd2 = Connect(port2);
  r = Call(fd2, rtier::kLoadPartition, load0(0, ""), 59);
  CHECK(r.status == rtier::kRawAbsent);
  r = Call(fd2, rtier::kLoadPartition, load0(0, peer1), 60);
  CHECK(r.status == rtier::kPQAbsent);
  rtier::BodyWriter miss;
  miss.Put<uint32_t>(1);
  miss.Put<uint32_t>(1);
  miss.Put<uint32_t>(static_cast<uint32_t>(seg0.size()));
  miss.PutBytes(seg0.data(), seg0.size());
  r = Call(fd2, rtier::kPQMissing, miss, 61);
  CHECK(r.status == rtier::kOk);
  rtier::BodyReader mr(r.body.data(), r.body.size());
  CHECK(mr.Get<uint32_t>() == 1);
  const uint32_t nmiss = mr.Get<uint32_t>();
  CHECK(nmiss == ids0.size() && mr.remaining() == nmiss * 4u);
  std::vector<uint32_t> missing(nmiss);
  std::memcpy(missing.data(), mr.Bytes(nmiss * 4u), nmiss * 4u);
  CHECK(missing == ids0);
  const uint32_t m = e->pq_stats().m;
  std::vector<uint8_t> codes(nmiss * m);
  e->PQGet(missing.data(), nmiss, codes.data());
  rtier::BodyWriter put;
  put.Put<uint32_t>(nmiss);
  put.Put<uint32_t>(m);
  put.PutBytes(missing.data(), nmiss * 4u);
  put.PutBytes(codes.data(), codes.size());
  r = Call(fd2, rtier::kPQPut, put, 62);
  CHECK(r.status == rtier::kOk && r.body.size() == 8);
  rtier::BodyReader pr(r.body.data(), r.body.size());
  CHECK(pr.Get<uint32_t>() == nmiss && pr.Get<uint32_t>() == 0);
  rtier::BodyWriter get;  // PQ_GET returns what was installed
  get.Put<uint32_t>(nmiss);
  get.PutBytes(missing.data(), nmiss * 4u);
  r = Call(fd2, rtier::kPQGet, get, 63);
  CHECK(r.status == rtier::kOk && r.body.size() == 4 + codes.size());
  CHECK(std::memcmp(r.body.data() + 4, codes.data(), codes.size()) == 0);
  r = Call(fd2, rtier::kLoadPartition, load0(0, peer1), 64);
  CHECK(r.status == rtier::kOk);
  st = fresh->pq_stats();
  CHECK(st.live == ids0.size() && st.staged == 0 && st.received == ids0.size());

  // RERANK on the new node fetches the vectors it lacks from the first node over the wire
  // (the node server's peer pool) and answers like the first node.
  const std::vector<uint32_t> first64(ids0.begin(), ids0.begin() + 64);
  const std::vector<uint32_t> lists64 = ListsNaming(f, 0, first64);
  auto rerank = [&](int conn, uint64_t id, uint32_t* fetched) {
    rtier::BodyWriter w;
    w.Put<uint32_t>(10);
    w.Put<uint32_t>(64);
    w.PutBytes(ids0.data(), 64 * 4);
    w.Put<uint32_t>(static_cast<uint32_t>(lists64.size()));
    w.PutBytes(lists64.data(), lists64.size() * 4);
    w.PutBytes(&f.queries[0], f.dim * sizeof(float));
    rtier::Frame resp = Call(conn, rtier::kRerank, w, id);
    CHECK(resp.status == rtier::kOk && resp.body.size() == 4 + 10 * 8 + 8);
    *fetched = 0;
    if (resp.body.size() >= 8) std::memcpy(fetched, resp.body.data() + resp.body.size() - 4, 4);
    resp.body.resize(resp.body.size() - 8);  // the counters
    return resp.body;
  };
  uint32_t fetched1 = 0, fetched2 = 0, fetched3 = 0;
  const auto want = rerank(fd, 70, &fetched1);
  const auto got = rerank(fd2, 71, &fetched2);
  CHECK(want == got && fetched1 == 0 && fetched2 > 0 && fetched2 <= 64);
  CHECK(fresh->raw_stats(true).fetches == 1 && e->raw_stats(true).served >= fetched2);
  CHECK(rerank(fd2, 72, &fetched3) == want && fetched3 == 0);

  // The stream: RAW_MISSING lists what is still missing, RAW_GET reads it from the first node,
  // RAW_PUT installs it.
  const std::vector<uint32_t> locs0 = Named(f, {0}).second;
  rtier::BodyWriter rmiss;
  rmiss.Put<uint32_t>(1);
  rmiss.Put<uint32_t>(1);
  rmiss.Put<uint32_t>(static_cast<uint32_t>(seg0.size()));
  rmiss.PutBytes(seg0.data(), seg0.size());
  r = Call(fd2, rtier::kRawMissing, rmiss, 73);
  CHECK(r.status == rtier::kOk);
  rtier::BodyReader rr(r.body.data(), r.body.size());
  CHECK(rr.Get<uint32_t>() == 1);
  const uint32_t nraw = rr.Get<uint32_t>();
  CHECK(nraw == locs0.size() - fetched2 && rr.remaining() == nraw * 8u);
  std::vector<uint32_t> rlocs(nraw), rlists(nraw);
  std::memcpy(rlocs.data(), rr.Bytes(nraw * 4u), nraw * 4u);
  std::memcpy(rlists.data(), rr.Bytes(nraw * 4u), nraw * 4u);
  for (uint32_t i = 0; i < nraw; i += 97) CHECK(ListNamingLoc(f, 0, rlocs[i]) != kInvalidId);
  // Both ways of streaming: RAW_PULL has this node fetch the first half itself (from the first
  // node, through its peer pool); RAW_GET from the first node and RAW_PUT bring the rest.
  const uint32_t half = nraw / 2, rest = nraw - half;
  rtier::BodyWriter rpull;
  rpull.Put<uint32_t>(half);
  rpull.PutBytes(rlocs.data(), half * 4u);
  rpull.PutBytes(rlists.data(), half * 4u);
  r = Call(fd2, rtier::kRawPull, rpull, 78);
  CHECK(r.status == rtier::kOk && r.body.size() == 8);
  rtier::BodyReader pl(r.body.data(), r.body.size());
  CHECK(pl.Get<uint32_t>() == half && pl.Get<uint32_t>() == 0);
  rtier::BodyWriter rget;
  rget.Put<uint32_t>(rest);
  rget.PutBytes(rlocs.data() + half, rest * 4u);
  rget.PutBytes(rlists.data() + half, rest * 4u);
  r = Call(fd, rtier::kRawGet, rget, 74);
  const uint32_t vb = fresh->vec_bytes();
  CHECK(r.status == rtier::kOk && r.body.size() == 4 + static_cast<size_t>(rest) * vb);
  rtier::BodyWriter rput;
  rput.Put<uint32_t>(rest);
  rput.Put<uint32_t>(vb);
  rput.PutBytes(rlocs.data() + half, rest * 4u);
  rput.PutBytes(r.body.data() + 4, static_cast<size_t>(rest) * vb);
  r = Call(fd2, rtier::kRawPut, rput, 75);
  CHECK(r.status == rtier::kOk && r.body.size() == 8);
  rtier::BodyReader pr2(r.body.data(), r.body.size());
  CHECK(pr2.Get<uint32_t>() == rest && pr2.Get<uint32_t>() == 0);
  const RawStats raw = fresh->raw_stats(true);
  CHECK(raw.present == locs0.size() && raw.pending == 0 && raw.streamed == nraw &&
        raw.fetches == 1);  // what RAW_PULL brought counts as streamed, not as fetched on demand
  r = Call(fd2, rtier::kRawPull, rpull, 79);  // nothing left to pull
  CHECK(r.status == rtier::kOk && r.body.size() == 8);
  rtier::BodyReader pl2(r.body.data(), r.body.size());
  CHECK(pl2.Get<uint32_t>() == 0 && pl2.Get<uint32_t>() == half);
  // A vector nobody told this node about cannot be had here.
  const std::vector<uint32_t> locs3 = Named(f, {3}).second;
  uint32_t other = kInvalidId;
  for (uint32_t l : locs3)
    if (!std::binary_search(locs0.begin(), locs0.end(), l)) other = l;
  rtier::BodyWriter rget2;  // its list's partition never loaded here: the list has no source
  rget2.Put<uint32_t>(1);
  rget2.Put<uint32_t>(other);
  rget2.Put<uint32_t>(ListNamingLoc(f, 3, other));
  r = Call(fd2, rtier::kRawGet, rget2, 76);
  CHECK(other != kInvalidId && r.status == rtier::kRawAbsent);
  r = Call(fd2, rtier::kRawPull, rget2, 80);  // nor pulled
  CHECK(r.status == rtier::kRawAbsent);
  // RAW_CHECK: of these, the ones not here.
  rtier::BodyWriter check;
  check.Put<uint32_t>(3);
  check.Put<uint32_t>(other);
  check.Put<uint32_t>(locs0[0]);
  check.Put<uint32_t>(other);
  r = Call(fd2, rtier::kRawCheck, check, 77);
  CHECK(r.status == rtier::kOk && r.body.size() == 8);
  rtier::BodyReader cr(r.body.data(), r.body.size());
  CHECK(cr.Get<uint32_t>() == 1 && cr.Get<uint32_t>() == other);
  r = Call(fd2, rtier::kPQRelease, empty, 65);
  CHECK(r.status == rtier::kOk && r.body.size() == 8);
  r = Call(fd2, rtier::kInfo, empty, 66);
  const std::string info2(r.body.begin(), r.body.end());
  CHECK(info2.find("\"pq\":{\"m\":8,") != std::string::npos);
  CHECK(info2.find("\"staged\":0,") != std::string::npos);
  CHECK(info2.find("\"raw\":{\"vec_bytes\":64,") != std::string::npos);
  CHECK(info2.find("\"payload\":\"lists+locations\"") != std::string::npos);
  ::close(fd2);
  server2.Stop();
  serve2.join();

  r = Call(fd, static_cast<rtier::Op>(0x7f), empty, 7);
  CHECK(r.status == rtier::kBadRequest);

  ::close(fd);
  server.Stop();
  serve.join();
}

}  // namespace

int main() {
  const std::vector<std::pair<const char*, std::function<void()>>> tests = {
      {"crc32", TestCrc32},
      {"partition_files", TestPartitionFiles},
      {"node_primitives", TestNodePrimitives},
      {"pq_store", TestPQStore},
      {"pq_migration", TestPQMigration},
      {"pq_evict_under_load", TestPQEvictUnderLoad},
      {"raw_store", TestRawStore},
      {"raw_migration", TestRawMigration},
      {"read_budget", TestReadBudget},
      {"wire", TestWire},
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
