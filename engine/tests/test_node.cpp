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
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <future>
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
  bad.holders.clear();
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
  std::vector<float> base;  // the index's vectors
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
    x.base = base;
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
                                     uint64_t read_iops = 0, int workers = 2) {
  NodeOptions o;
  o.index_dir = index_dir.empty() ? f.index : index_dir;
  o.partitions_dir = f.parts;
  o.num_workers = workers;
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

// In-process stand-in for the node server's peer pool and the agents' address book: `node` is
// node `self` and reaches each of `peers` (by node ID) by calling its RawGet directly.
void LinkPeers(NodeEngine* node, uint16_t self, std::map<uint16_t, NodeEngine*> peers) {
  std::vector<std::pair<uint16_t, std::string>> book;
  std::map<std::string, NodeEngine*> by_addr;
  for (const auto& kv : peers) {
    const std::string addr = "node-" + std::to_string(kv.first);
    book.emplace_back(kv.first, addr);
    by_addr[addr] = kv.second;
  }
  node->SetNodes(self, book);
  node->SetRawFetcher([by_addr](const std::string& peer, const uint32_t* locs, size_t n, uint8_t* out) {
    auto it = by_addr.find(peer);
    if (it == by_addr.end()) throw std::runtime_error("unknown peer " + peer);
    it->second->RawGet(locs, n, out);
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
      const uint32_t nc = e->Filter(0, q, ls.data(), nullptr, static_cast<uint32_t>(ls.size()), 4096,
                                    cand.data(), pqd.data(), &fs);
      CHECK(nc == fs.unique);
      for (uint32_t i = 1; i < nc; ++i)
        CHECK(pqd[i - 1] < pqd[i] || (pqd[i - 1] == pqd[i] && cand[i - 1] < cand[i]));
      std::vector<uint32_t> rid(k);
      std::vector<float> rd(k);
      RerankStats rs;
      const uint32_t nr = e->Rerank(1, q, cand.data(), nc, k, ls.data(), nullptr,
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
    all->Filter(0, &f.queries[0], lists.data(), nullptr, nl, 10, ids.data(), d.data(), &fs);
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

// A partition can be resident at several versions (U14): FILTER and RERANK name the version
// their query's epoch sees, loading a version replaces only that version, and an eviction drops
// every version.
void TestPartitionVersions() {
  Fixture& f = GetFixture();
  auto node = OpenNode(f, {0, 1, 2, 3}, true);  // every partition at version 0
  PartitionManifest man = PartitionManifest::Load(f.parts);
  const float* q = &f.queries[0];
  std::vector<uint32_t> lists(8);
  const uint32_t nl = node->Navigate(q, 8, 0, lists.data());
  CHECK(nl == 8);
  const uint32_t p = man.list_part[lists[0]];
  std::vector<uint32_t> mine;  // the probed lists of partition p
  for (uint32_t i = 0; i < nl; ++i)
    if (man.list_part[lists[i]] == p) mine.push_back(lists[i]);
  const uint32_t nm = static_cast<uint32_t>(mine.size());
  auto filter = [&](const std::vector<uint32_t>* versions, std::vector<uint32_t>* ids) {
    ids->assign(200, 0);
    std::vector<float> d(200);
    FilterStats fs;
    const uint32_t n = node->Filter(0, q, mine.data(), versions ? versions->data() : nullptr, nm,
                                    200, ids->data(), d.data(), &fs);
    ids->resize(n);
  };
  auto versions_of = [&](uint32_t part) {
    std::vector<uint32_t> vs;
    for (const auto& [pp, v] : node->ResidentVersions())
      if (pp == part) vs.push_back(v);
    return vs;
  };
  const std::vector<uint32_t> v0(nm, 0), v1(nm, 1);
  std::vector<uint32_t> at0, at1, newest;
  filter(&v0, &at0);
  CHECK(!at0.empty());
  bool not_resident = false;
  try {
    filter(&v1, &at1);
  } catch (const NotResidentError&) {
    not_resident = true;
  }
  CHECK(not_resident);

  // Version 1 next to version 0 (the same lists here): both serve, and no version means the
  // newest one.
  const std::string seg = JoinPath(f.parts, partfiles::SegmentName(p));
  node->LoadPartition(p, seg, PQSource::kPresent, 1);
  filter(&v1, &at1);
  filter(nullptr, &newest);
  CHECK(at1 == at0 && newest == at0);
  filter(&v0, &at0);
  CHECK(at0 == at1);
  CHECK((versions_of(p) == std::vector<uint32_t>{0, 1}));
  // Loading a version again replaces it; RERANK takes the versions too.
  node->LoadPartition(p, seg, PQSource::kPresent, 1);
  CHECK((versions_of(p) == std::vector<uint32_t>{0, 1}));
  {
    std::vector<uint32_t> out(10);
    std::vector<float> d(10);
    RerankStats rs;
    CHECK(node->Rerank(0, q, at1.data(), static_cast<uint32_t>(at1.size()), 10, mine.data(),
                       v1.data(), nm, out.data(), d.data(), &rs) > 0);
  }
  // An eviction drops every version.
  CHECK(node->EvictPartition(p));
  CHECK(versions_of(p).empty());
  const auto res = node->ResidentPartitions();
  CHECK(std::find(res.begin(), res.end(), p) == res.end());
}

// Inserted vectors (U14). ASSIGN routes a vector the way the build did (the same lists and
// the same PQ code); the ID and location spaces leave room for inserts; RAW_ADD serves inserted
// vectors from memory at once, writes them only while update writes are not paused, and holds
// the sender back when its buffer is full.
void TestInsertedVectors() {
  Fixture& f = GetFixture();
  const uint64_t n = f.base.size() / f.dim;
  const PartitionManifest man = PartitionManifest::Load(f.parts);
  auto open = [&](bool full, uint64_t buffer) {
    NodeOptions o;
    o.index_dir = f.index;
    o.partitions_dir = f.parts;
    o.num_workers = 2;
    o.max_nprobe = 64;
    o.max_rerank = 4096;
    o.direct_io = false;
    o.max_vectors = n + 64;
    o.max_pages = man.raw.num_pages + 4;
    o.update_buffer_bytes = buffer;
    auto e = NodeEngine::Open(o);
    if (full) {
      e->LoadGraph(JoinPath(f.index, files::kGraph));
      for (uint32_t p = 0; p < f.num_parts; ++p)
        e->LoadPartition(p, JoinPath(f.parts, partfiles::SegmentName(p)), PQSource::kIndex);
    }
    return e;
  };
  auto node = open(true, 256u << 20);
  CHECK(node->id_capacity() == n + 64);
  const uint64_t space0 = man.raw.locations();
  CHECK(node->raw_stats(false).locations == space0 + 4ull * man.raw.vectors_per_page);

  // ASSIGN on the index's own vectors gives back what the build gave them.
  const uint32_t r = node->assign_replicas(), m = node->pq_m(), k = 300;
  std::vector<uint32_t> lists(static_cast<size_t>(k) * r);
  std::vector<uint8_t> codes(static_cast<size_t>(k) * m);
  node->Assign(f.base.data(), k, lists.data(), codes.data());
  const PostingLists pl = PostingLists::Load(JoinPath(f.index, files::kPostings));
  std::vector<std::vector<uint32_t>> built(k);
  for (uint32_t c = 0; c < pl.num_lists(); ++c)
    for (uint32_t j = 0; j < pl.Size(c); ++j)
      if (pl.List(c)[j] < k) built[pl.List(c)[j]].push_back(c);
  uint32_t same = 0;
  for (uint32_t i = 0; i < k; ++i) {
    std::vector<uint32_t> got;
    for (uint32_t j = 0; j < r; ++j)
      if (lists[i * r + j] != kInvalidId) got.push_back(lists[i * r + j]);
    std::sort(got.begin(), got.end());
    if (got == built[i]) ++same;
  }
  std::printf("    ASSIGN: %u of %u vectors get the build's lists\n", same, k);
  CHECK(same == k);
  std::vector<uint8_t> stored(codes.size());
  {
    std::FILE* pf = std::fopen(JoinPath(f.index, files::kPQCodes).c_str(), "rb");
    CHECK(pf != nullptr);
    CHECK(std::fseek(pf, 16, SEEK_SET) == 0);
    CHECK(std::fread(stored.data(), 1, stored.size(), pf) == stored.size());
    std::fclose(pf);
  }
  CHECK(codes == stored);

  // New IDs past the index's vectors take PQ codes; one past the ID space does not.
  std::vector<uint32_t> new_ids = {static_cast<uint32_t>(n), static_cast<uint32_t>(n + 1)};
  CHECK(node->PQPut(new_ids.data(), 2, codes.data()).installed == 2);
  bool refused = false;
  try {
    const uint32_t far = static_cast<uint32_t>(n + 64);
    node->PQPut(&far, 1, codes.data());
  } catch (const std::exception&) {
    refused = true;
  }
  CHECK(refused);

  // Inserted vectors at new locations: served at once, written only once writes resume.
  const uint32_t vb = node->vec_bytes();
  std::vector<uint32_t> locs = {static_cast<uint32_t>(space0), static_cast<uint32_t>(space0 + 1),
                                static_cast<uint32_t>(space0 + 2)};
  const uint8_t* vecs = reinterpret_cast<const uint8_t*>(f.base.data());
  node->PauseUpdateWrites(true);
  CHECK(node->RawAdd(locs.data(), locs.size(), vecs).installed == 3);
  CHECK(node->RawAdd(locs.data(), 1, vecs).skipped == 1);
  std::vector<uint8_t> out(locs.size() * vb);
  node->RawGet(locs.data(), locs.size(), out.data());
  CHECK(std::memcmp(out.data(), vecs, out.size()) == 0);
  CHECK(node->RawAbsent(locs.data(), locs.size()).empty());
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  CHECK(node->fetch_stats().added_queued == 3 && node->fetch_stats().added_written == 0);
  node->PauseUpdateWrites(false);
  for (int i = 0; i < 500 && node->fetch_stats().added_written < 3; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  CHECK(node->fetch_stats().added_written == 3 && node->fetch_stats().added_queued == 0);
  CHECK(node->raw_stats(false).added == 3);
  std::fill(out.begin(), out.end(), 0);
  node->RawGet(locs.data(), locs.size(), out.data());
  CHECK(std::memcmp(out.data(), vecs, out.size()) == 0);
  refused = false;
  try {
    const uint32_t far = static_cast<uint32_t>(space0 + 4ull * man.raw.vectors_per_page);
    node->RawAdd(&far, 1, vecs);
  } catch (const std::exception&) {
    refused = true;
  }
  CHECK(refused);

  // Backpressure: with room for 4 vectors and writes paused, a fifth waits for the writer.
  auto small = open(false, 4ull * vb);
  small->PauseUpdateWrites(true);
  std::vector<uint32_t> five = {static_cast<uint32_t>(space0), static_cast<uint32_t>(space0 + 1),
                                static_cast<uint32_t>(space0 + 2), static_cast<uint32_t>(space0 + 3),
                                static_cast<uint32_t>(space0 + 4)};
  CHECK(small->RawAdd(five.data(), 4, vecs).installed == 4);
  std::atomic<bool> done{false};
  std::thread t([&] {
    small->RawAdd(&five[4], 1, vecs + 4 * vb);
    done = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  CHECK(!done);
  small->PauseUpdateWrites(false);
  t.join();
  CHECK(done);
}

// An update batch on one node (U14): StageVersions builds each partition's next version from
// the base one (inserts in, deletes out) next to it, all or nothing; the base version keeps
// serving; dropping it and forgetting the deleted vectors frees their codes; a version saved
// as a segment loads elsewhere.
void TestStageVersions() {
  Fixture& f = GetFixture();
  const uint64_t n = f.base.size() / f.dim;
  const PartitionManifest man = PartitionManifest::Load(f.parts);
  NodeOptions o;
  o.index_dir = f.index;
  o.partitions_dir = f.parts;
  o.num_workers = 2;
  o.max_nprobe = 64;
  o.max_rerank = 4096;
  o.direct_io = false;
  o.max_vectors = n + 64;
  o.max_pages = man.raw.num_pages + 4;
  o.max_list_len = man.max_list_len + 16;
  auto node = NodeEngine::Open(o);
  node->SetNodes(1, {});  // SavePartition records this node as the holder of what it holds
  node->LoadGraph(JoinPath(f.index, files::kGraph));
  for (uint32_t p = 0; p < f.num_parts; ++p)
    node->LoadPartition(p, JoinPath(f.parts, partfiles::SegmentName(p)), PQSource::kIndex);
  const float* q = &f.queries[0];
  auto search = [&](uint32_t* top) {
    SearchParams sp;
    sp.k = 10;
    sp.nprobe = 16;
    sp.rerank = 4096;
    sp.rr.heuristic = false;
    std::vector<uint32_t> ids(10);
    std::vector<float> d(10);
    QueryStats qs;
    node->SearchLocal(0, q, sp, ids.data(), d.data(), &qs);
    *top = ids[0];
    return std::vector<uint32_t>(ids.begin(), ids.end());
  };
  uint32_t top0 = 0;
  const std::vector<uint32_t> before = search(&top0);

  // The batch: the query itself as a new vector, and the old top answer deleted.
  const uint32_t r = node->assign_replicas(), m = node->pq_m();
  std::vector<uint32_t> lists(r);
  std::vector<uint8_t> code(m);
  node->Assign(q, 1, lists.data(), code.data());
  const uint32_t id = static_cast<uint32_t>(n), loc = static_cast<uint32_t>(man.raw.locations());
  std::vector<uint32_t> al, ai, ac;
  for (uint32_t c : lists)
    if (c != kInvalidId) {
      al.push_back(c);
      ai.push_back(id);
      ac.push_back(loc);
    }
  std::vector<VersionStage> parts;
  for (uint32_t p = 0; p < f.num_parts; ++p) parts.push_back({p, 0, 1});
  const uint32_t del = top0;

  // Refused, and nothing changes: the new vector's code and raw vector are not here yet.
  bool refused = false;
  try {
    node->StageVersions(parts, al.data(), ai.data(), ac.data(), al.size(), &del, 1);
  } catch (const RawMissingError&) {
    refused = true;
  }
  CHECK(refused);
  CHECK(node->RawAdd(&loc, 1, reinterpret_cast<const uint8_t*>(q)).installed == 1);
  refused = false;
  try {
    node->StageVersions(parts, al.data(), ai.data(), ac.data(), al.size(), &del, 1);
  } catch (const PQMissingError&) {
    refused = true;
  }
  CHECK(refused);
  CHECK(node->ResidentVersions().size() == f.num_parts);  // still version 0 only
  CHECK(node->PQPut(&id, 1, code.data()).installed == 1);
  const auto res = node->StageVersions(parts, al.data(), ai.data(), ac.data(), al.size(), &del, 1);
  std::vector<uint32_t> changed;
  uint64_t added = 0, removed = 0;
  for (const auto& x : res) {
    if (x.changed) changed.push_back(x.partition);
    added += x.added;
    removed += x.removed;
  }
  CHECK(!changed.empty() && added == al.size() && removed > 0);
  CHECK(node->ResidentVersions().size() == f.num_parts + changed.size());

  // Version 0 still answers as before; version 1 has the insert and not the deleted vector.
  std::vector<uint32_t> probe(16);
  const uint32_t np = node->Navigate(q, 16, 0, probe.data());
  auto filter = [&](uint32_t v, std::vector<uint32_t>* out) {
    std::vector<uint32_t> vs(np);
    for (uint32_t i = 0; i < np; ++i) {
      const uint32_t p = man.list_part[probe[i]];
      vs[i] = std::find(changed.begin(), changed.end(), p) != changed.end() ? v : 0;
    }
    out->assign(4096, 0);
    std::vector<float> d(4096);
    FilterStats fs;
    out->resize(node->Filter(0, q, probe.data(), vs.data(), np, 4096, out->data(), d.data(), &fs));
  };
  std::vector<uint32_t> c0, c1;
  filter(0, &c0);
  filter(1, &c1);
  auto has = [](const std::vector<uint32_t>& v, uint32_t x) { return std::find(v.begin(), v.end(), x) != v.end(); };
  CHECK(has(c0, del) && !has(c0, id));
  CHECK(!has(c1, del) && has(c1, id));
  uint32_t top1 = 0;
  const std::vector<uint32_t> after = search(&top1);  // the newest versions
  CHECK(top1 == id && !has(after, del) && after != before);

  // Saved and loaded elsewhere as version 1, the partition answers the same.
  const uint32_t p0 = changed[0];
  const std::string saved = JoinPath(TempDir(), "saved.seg");
  node->SavePartition(p0, 1, saved);
  const ListSegment back = ListSegment::Load(saved);
  CHECK(back.partition == p0 && std::find(back.ids.begin(), back.ids.end(), del) == back.ids.end());

  // A node without room for a list that long refuses the segment at load, before it changes
  // anything (here: one list grown past the build's longest, on a node at the build's limit).
  ListSegment longer = back;
  longer.holders.clear();
  const uint64_t grow = man.max_list_len + 1 - (longer.offsets[1] - longer.offsets[0]);
  longer.ids.insert(longer.ids.begin() + longer.offsets[1], grow, longer.ids[longer.offsets[0]]);
  longer.locs.insert(longer.locs.begin() + longer.offsets[1], grow, longer.locs[longer.offsets[0]]);
  for (size_t i = 1; i < longer.offsets.size(); ++i) longer.offsets[i] += grow;
  const std::string long_path = JoinPath(TempDir(), "longer.seg");
  longer.Save(long_path);
  NodeOptions tight = o;
  tight.max_list_len = 0;  // the manifest's
  auto small = NodeEngine::Open(tight);
  refused = false;
  try {
    small->LoadPartition(p0, long_path, PQSource::kIndex);
  } catch (const std::exception& e) {
    refused = std::string(e.what()).find("--max-list-len") != std::string::npos;
  }
  CHECK(refused && small->ResidentVersions().empty() && small->pq_stats().resident == 0);

  // Reclaim, in the controller's order. The deleted vector is marked first: version 0 still
  // names it, so its code stays for the old epoch's queries. Then version 0 of the changed
  // partitions goes, and the code goes with its last reference -- freed, never cached. Without
  // the deleted flag a code that a version names is left alone (the inserted one here).
  CHECK(node->ForgetVectors(&id, 1, false) == 0 && node->pq_stats().deleted == 0);
  const PQStats pq0 = node->pq_stats();
  CHECK(node->ForgetVectors(&del, 1, true) == 0 && node->pq_stats().deleted == 1);
  for (uint32_t p : changed) CHECK(node->DropVersion(p, 0) && !node->DropVersion(p, 0));
  CHECK(node->ResidentVersions().size() == f.num_parts);
  const PQStats pq1 = node->pq_stats();
  CHECK(pq1.deleted == 0 && pq1.cached == pq0.cached && pq1.freed == pq0.freed + 1 &&
        pq1.resident == pq0.resident - 1);
  std::vector<uint8_t> gone(m);
  refused = false;
  try {
    node->PQGet(&del, 1, gone.data());
  } catch (const PQMissingError&) {
    refused = true;
  }
  CHECK(refused);
  CHECK(node->ForgetVectors(&del, 1, true) == 0);
  uint32_t top2 = 0;
  CHECK(search(&top2) == after);

  // A posting may only name a list of a staged partition.
  std::vector<VersionStage> one = {{changed[0], 1, 2}};
  uint32_t other = kInvalidId;
  for (uint32_t c = 0; c < man.num_lists && other == kInvalidId; ++c)
    if (man.list_part[c] != changed[0]) other = c;
  refused = false;
  try {
    node->StageVersions(one, &other, &id, &loc, 1, nullptr, 0);
  } catch (const std::exception&) {
    refused = true;
  }
  CHECK(refused);
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

  // Deleted vectors (U14) never leave a cached code. An unreferenced copy goes at once; a code
  // still named is marked and goes with its last reference, or when the staging protecting it
  // ends. Without the deleted flag only unreferenced copies go.
  const std::vector<uint32_t> ten_eleven = {10, 11};
  s.Ref(ten_eleven.data(), 2, nullptr);  // 10 and 11 live; 12 and 13 staged, never named
  const std::vector<uint32_t> unused = {kept, 10, 12};
  CHECK(s.Forget(unused.data(), unused.size(), false) == 1);  // the cached one only
  st = s.Stats();
  CHECK(st.deleted == 0 && st.resident == 4 && st.cached == 0);
  CHECK(s.Forget(ten_eleven.data(), 2, true) == 0);
  CHECK(s.Stats().deleted == 2 && s.Stats().live == 2);
  s.Unref(ten_eleven.data(), 1);  // 10's last reference: freed, not cached
  st = s.Stats();
  CHECK(st.deleted == 1 && st.cached == 0 && st.live == 1 && st.resident == 3);
  const std::vector<uint32_t> eleven = {11};
  CHECK(s.Reserve(eleven.data(), 1).empty());  // a staging claims 11 ...
  s.Unref(eleven.data(), 1);                    // ... so it stays, staged, past its last reference
  st = s.Stats();
  CHECK(st.staged == 3 && st.deleted == 1 && st.resident == 3);
  CHECK(s.ReleaseStaged() == 3);  // 12 and 13 never named, and 11 deleted: none is cached
  st = s.Stats();
  CHECK(st.resident == 0 && st.cached == 0 && st.deleted == 0);
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
  LinkPeers(dst.get(), 2, {{1, src.get()}});
  bool threw = false;
  try {  // raw vectors missing, and no holder to fetch them from
    dst->LoadPartition(1, seg(1), PQSource::kPresent);
  } catch (const RawMissingError&) {
    threw = true;
  }
  CHECK(threw && dst->ResidentPartitions().empty() && dst->pq_stats().resident == 0);
  threw = false;
  try {
    dst->LoadPartition(1, seg(1), PQSource::kPresent, 0, 1);
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
  dst->LoadPartition(1, seg(1), PQSource::kPresent, 0, 1);
  dst->LoadPartition(2, seg(2), PQSource::kPresent, 0, 1);
  st = dst->pq_stats();
  CHECK(st.resident == d12.size() && st.live == d12.size() && st.staged == 0 &&
        st.received == d12.size() && st.from_index == 0);

  // Another partition later: only the codes the node lacks are listed.
  const auto miss0 = dst->PQMissing({{seg(0)}});
  CHECK(miss0[0].size() == d012.size() - d12.size());
  move_codes(src.get(), dst.get(), miss0[0]);
  dst->LoadPartition(0, seg(0), PQSource::kPresent, 0, 1);
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
    const uint32_t na = src->Filter(0, q, mine.data(), nullptr, static_cast<uint32_t>(mine.size()), 200,
                                    ia.data(), da.data(), &fa);
    const uint32_t nb = dst->Filter(0, q, mine.data(), nullptr, static_cast<uint32_t>(mine.size()), 200,
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
  dst->LoadPartition(0, seg(0), PQSource::kPresent, 0, 1);
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
  LinkPeers(node.get(), 2, {{1, oracle.get()}});
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
        na = node->Filter(wi, &f.queries[qi * f.dim], ls.data(), nullptr, static_cast<uint32_t>(ls.size()),
                          4096, ia.data(), da.data(), &fa);
      } catch (const NotResidentError&) {
        continue;
      } catch (const std::exception& e) {
        if (errors++ == 0) std::printf("    query error: %s\n", e.what());
        continue;
      }
      {
        std::lock_guard<std::mutex> l(oracle_mu);
        nb = oracle->Filter(0, &f.queries[qi * f.dim], ls.data(), nullptr, static_cast<uint32_t>(ls.size()),
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
        node->LoadPartition(p, seg(p), PQSource::kPresent, 0, 1);
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

// Stages partitions on `to` the way an agent does after a migration: `from` writes each as a
// segment file (every posting records its current holder), `to` gets the PQ codes it lacks
// from `from` and loads the files.
void StageFrom(const Fixture& f, NodeEngine* to, NodeEngine* from, const std::vector<uint32_t>& parts) {
  (void)f;
  static int seq = 0;
  std::vector<std::string> paths;
  for (uint32_t p : parts) {
    paths.push_back(JoinPath(TempDir(), "handoff-" + std::to_string(++seq) + ".seg"));
    from->SavePartition(p, 0, paths.back());
  }
  const auto miss = to->PQMissing({paths});
  std::vector<uint8_t> codes(miss[0].size() * from->pq_stats().m);
  from->PQGet(miss[0].data(), miss[0].size(), codes.data());
  to->PQPut(miss[0].data(), miss[0].size(), codes.data());
  for (size_t i = 0; i < parts.size(); ++i) to->LoadPartition(parts[i], paths[i], PQSource::kPresent);
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
  const uint32_t nc = at->Filter(0, q, mine.data(), nullptr, static_cast<uint32_t>(mine.size()), 200,
                                 cand.data(), pqd.data(), &fs);
  RerankStats ra, rb;
  const uint32_t nm = static_cast<uint32_t>(mine.size());
  const uint32_t na = at->Rerank(0, q, cand.data(), nc, 10, mine.data(), nullptr, nm, ia.data(), da.data(), &ra);
  const uint32_t nb = ref->Rerank(0, q, cand.data(), nc, 10, mine.data(), nullptr, nm, ib.data(), db.data(), &rb);
  return na == nb && ia == ib && da == db;
}

// Raw vectors after a migration (U9): the new owner goes online without them, re-ranking fetches
// the ones it needs on demand -- from the holder a posting of the query's lists records, every
// holder at once -- and only those; a stream (the lazy-stream baseline) can install the rest; the
// answers are the old owner's throughout. The new nodes' index has no page file at all.
void TestRawMigration() {
  Fixture& f = GetFixture();
  auto seg = [&](uint32_t p) { return JoinPath(f.parts, partfiles::SegmentName(p)); };
  static const PartitionManifest man = PartitionManifest::Load(f.parts);
  // Node IDs: src 1, dst 2, third 3, pulled 4, fourth 5, after 6, whole 7, slow 8, src2 9, split 10.
  auto src = OpenNode(f, {0, 1, 2, 3}, true);
  src->SetNodes(1, {});
  const auto locs_all = Named(f, {0, 1, 2, 3}).second;
  CHECK(src->raw_stats(true).present == locs_all.size() &&
        src->raw_stats(true).from_index == locs_all.size());

  auto dst = OpenNode(f, {}, false, f.index_nopages);
  LinkPeers(dst.get(), 2, {{1, src.get()}});
  StageFrom(f, dst.get(), src.get(), {2, 3});
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

  // dst hands partition 3 on before it holds all of its vectors. The segment it writes says
  // where each one is -- dst for those it fetched, src for the others -- so third fetches each
  // straight from its holder, and dst fetches nothing on its behalf: no chain.
  auto third = OpenNode(f, {}, false, f.index_nopages);
  LinkPeers(third.get(), 3, {{1, src.get()}, {2, dst.get()}});
  StageFrom(f, third.get(), dst.get(), {3});
  const uint64_t dst_fetched = dst->raw_stats(true).fetched;
  const uint64_t served0 = src->raw_stats(true).served + dst->raw_stats(true).served;
  same = true;
  for (uint32_t qi = 0; qi < f.nq; ++qi) same = SameRerank(f, qi, third.get(), src.get(), {3}) && same;
  const uint64_t third_fetched = third->raw_stats(true).fetched;
  CHECK(same && third_fetched > 0 && dst->raw_stats(true).fetched == dst_fetched &&
        src->raw_stats(true).served + dst->raw_stats(true).served - served0 == third_fetched);

  // What nothing needed stays missing (the lazy protocol stops here); a stream brings the rest:
  // exactly the vectors still missing, once, by holder (all src's here), each with a list of
  // these partitions that names it.
  const auto miss = dst->RawMissing({2, 3});
  rs = dst->raw_stats(true);
  CHECK(rs.pending > 0);
  CHECK(miss.size() == 1 && miss[0].holder == 1 && miss[0].locs.size() == locs23.size() - rs.present &&
        miss[0].locs.size() == rs.pending);
  for (size_t i = 0; i < miss[0].locs.size(); ++i) {
    const uint32_t part = man.list_part[miss[0].lists[i]];
    CHECK((part == 2 || part == 3) && ListNamingLoc(f, part, miss[0].locs[i]) != kInvalidId);
  }
  std::vector<uint8_t> v(miss[0].locs.size() * dst->vec_bytes());
  src->RawGet(miss[0].locs.data(), miss[0].locs.size(), v.data());
  const RawPutResult pr = dst->RawPut(miss[0].locs.data(), miss[0].locs.size(), v.data());
  CHECK(pr.installed == miss[0].locs.size() && pr.skipped == 0);
  rs = dst->raw_stats(true);
  CHECK(rs.present == locs23.size() && rs.pending == 0 && rs.streamed == miss[0].locs.size());
  CHECK(dst->RawMissing({2, 3}).empty());
  same = true;
  for (uint32_t qi = 0; qi < f.nq; ++qi) same = SameRerank(f, qi, dst.get(), src.get(), {2, 3}) && same;
  CHECK(same && dst->raw_stats(true).fetched == rs.fetched);  // nothing left to fetch

  // The same stream pulled by the node itself (RawPull, data node to data node): exactly the
  // vectors it lacks, installed as streamed and not as fetched on demand; a second pull brings
  // nothing.
  auto pulled = OpenNode(f, {}, false, f.index_nopages);
  LinkPeers(pulled.get(), 4, {{1, src.get()}});
  StageFrom(f, pulled.get(), src.get(), {2, 3});
  const auto want = pulled->RawMissing({2, 3});
  uint64_t got = 0;
  for (const auto& m : want) {
    const RawPutResult r = pulled->RawPull(m.locs.data(), m.lists.data(), m.locs.size());
    CHECK(r.installed + r.skipped == m.locs.size());
    got += r.installed;
  }
  RawStats ps = pulled->raw_stats(true);
  CHECK(got == locs23.size() && ps.present == got && ps.streamed == got && ps.fetched == 0 &&
        ps.fetches == 0 && ps.pending == 0);
  for (const auto& m : want) {
    const RawPutResult r = pulled->RawPull(m.locs.data(), m.lists.data(), m.locs.size());
    CHECK(r.installed == 0 && r.skipped == m.locs.size());
  }
  same = true;
  for (uint32_t qi = 0; qi < f.nq; ++qi) same = SameRerank(f, qi, pulled.get(), src.get(), {2, 3}) && same;
  CHECK(same && pulled->raw_stats(true).fetched == 0);

  // Partition 3 moves on from third (src -> dst -> third -> fourth): third's segment records
  // third, dst or src for every posting, and fourth pulls each vector from that node; third
  // fetches nothing on the way.
  auto fourth = OpenNode(f, {}, false, f.index_nopages);
  LinkPeers(fourth.get(), 5, {{1, src.get()}, {2, dst.get()}, {3, third.get()}});
  StageFrom(f, fourth.get(), third.get(), {3});
  const auto want3 = fourth->RawMissing({3});
  std::set<uint16_t> holders;
  uint64_t need3 = 0, got3 = 0;
  for (const auto& m : want3) {
    holders.insert(m.holder);
    need3 += m.locs.size();
    got3 += fourth->RawPull(m.locs.data(), m.lists.data(), m.locs.size()).installed;
  }
  CHECK(holders.count(3) && holders.count(1) && got3 == need3 && fourth->RawMissing({3}).empty() &&
        third->raw_stats(true).fetched == third_fetched);

  // Before a node leaves, the nodes whose postings name it drain it. `after` takes partition 3
  // from third without fetching anything, so its postings name third for what third holds.
  // RawHeldBy(3) lists those; once they are pulled from third, a second call only repoints the
  // postings of other lists that name the same vectors, and then nothing names third: with
  // third gone, every answer still comes out the same.
  auto after = OpenNode(f, {}, false, f.index_nopages);
  LinkPeers(after.get(), 6, {{1, src.get()}, {2, dst.get()}, {3, third.get()}});
  StageFrom(f, after.get(), third.get(), {3});
  HeldBy hb = after->RawHeldBy(3);
  CHECK(hb.repointed == 0 && hb.missing.holder == 3 && hb.missing.locs.size() == third_fetched);
  CHECK(after->RawPull(hb.missing.locs.data(), hb.missing.lists.data(), hb.missing.locs.size()).installed ==
        third_fetched);
  hb = after->RawHeldBy(3);
  CHECK(hb.missing.locs.empty());
  CHECK(after->RawHeldBy(3).repointed == 0 && after->RawHeldBy(3).missing.locs.empty());
  LinkPeers(after.get(), 6, {{1, src.get()}, {2, dst.get()}});  // third is gone
  third.reset();
  same = true;
  for (uint32_t qi = 0; qi < f.nq; ++qi) same = SameRerank(f, qi, after.get(), src.get(), {3}) && same;
  CHECK(same);

  // A partition that moves away leaves its vectors (a cache, like its PQ codes): it comes back
  // with nothing to transfer, and no holder is needed.
  CHECK(dst->EvictPartition(2));
  CHECK(dst->raw_stats(true).present == locs23.size());
  dst->LoadPartition(2, seg(2), PQSource::kPresent);
  CHECK(dst->ResidentPartitions().size() == 2 && dst->raw_stats(true).pending == 0 &&
        dst->RawMissing({2}).empty());

  // The whole pipeline on a node that holds every partition but no vector yet: SearchLocal
  // fetches before re-ranking and returns the single-node answer.
  auto whole = OpenNode(f, {}, true, f.index_nopages);
  LinkPeers(whole.get(), 7, {{1, src.get()}});
  StageFrom(f, whole.get(), src.get(), {0, 1, 2, 3});
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

  // Two queries that miss the same vectors at once: one fetches them, the other waits for that
  // fetch, and the holder serves each vector once.
  auto slow = OpenNode(f, {}, false, f.index_nopages);
  slow->SetNodes(8, {{1, "node-1"}});
  slow->SetRawFetcher([&](const std::string&, const uint32_t* locs, size_t n, uint8_t* out) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    src->RawGet(locs, n, out);
  });
  StageFrom(f, slow.get(), src.get(), {2, 3});
  const float* q = &f.queries[0];
  std::vector<uint32_t> lists(16), mine;
  const uint32_t nl = src->Navigate(q, 16, 0, lists.data());
  for (uint32_t i = 0; i < nl; ++i)
    if (man.list_part[lists[i]] >= 2) mine.push_back(lists[i]);
  std::vector<uint32_t> cand(200);
  std::vector<float> pqd(200);
  FilterStats fs;
  const uint32_t nc = slow->Filter(0, q, mine.data(), nullptr, static_cast<uint32_t>(mine.size()), 200,
                                   cand.data(), pqd.data(), &fs);
  CHECK(nc > 0);
  const uint64_t served1 = src->raw_stats(true).served;
  std::vector<uint32_t> ra(10), rb(10);
  std::vector<float> da(10), db(10);
  RerankStats sa, sb;
  std::thread other([&] {
    slow->Rerank(1, q, cand.data(), nc, 10, mine.data(), nullptr, static_cast<uint32_t>(mine.size()),
                 rb.data(), db.data(), &sb);
  });
  slow->Rerank(0, q, cand.data(), nc, 10, mine.data(), nullptr, static_cast<uint32_t>(mine.size()), ra.data(),
               da.data(), &sa);
  other.join();
  CHECK(ra == rb && da == db && slow->fetch_stats().coalesced > 0 &&
        src->raw_stats(true).served - served1 == slow->raw_stats(true).fetched);

  // A query whose missing vectors lie at two holders fetches from both at once, one request
  // each: `split` takes partition 2 from src and partition 3 from src2, as a new node takes
  // partitions from several donors. Until one query has needed both, each fetch waits (up to
  // 1 s) for one from the other holder to begin: they meet only if they run in parallel.
  auto src2 = OpenNode(f, {2, 3}, false);
  src2->SetNodes(9, {});
  auto split = OpenNode(f, {}, false, f.index_nopages);
  split->SetNodes(10, {{1, "node-1"}, {9, "node-9"}});
  std::mutex mu;
  std::condition_variable cv;
  std::set<std::string> busy, asked;
  bool probing = true, met = false;
  split->SetRawFetcher([&](const std::string& peer, const uint32_t* locs, size_t n, uint8_t* out) {
    {
      std::unique_lock<std::mutex> l(mu);
      asked.insert(peer);
      busy.insert(peer);
      cv.notify_all();
      if (probing && cv.wait_for(l, std::chrono::seconds(1), [&] { return busy.size() > 1; })) met = true;
    }
    (peer == "node-1" ? src : src2)->RawGet(locs, n, out);
    std::lock_guard<std::mutex> l(mu);
    busy.erase(peer);
  });
  StageFrom(f, split.get(), src.get(), {2});
  StageFrom(f, split.get(), src2.get(), {3});
  same = true;
  bool both = false;
  for (uint32_t qi = 0; qi < f.nq; ++qi) {
    {
      std::lock_guard<std::mutex> l(mu);
      asked.clear();
    }
    same = SameRerank(f, qi, split.get(), src.get(), {2, 3}) && same;
    std::lock_guard<std::mutex> l(mu);
    if (probing && asked.size() == 2) {
      both = true;
      probing = false;
    }
  }
  // One request per holder: more requests than fetching calls.
  const RawStats ss = split->raw_stats(true);
  CHECK(same && both && met && ss.fetched > 0 && ss.fetches > split->fetch_stats().fetch_calls);
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
    // A Get puts its reads in flight together, a window at a time: the vectors come back in the
    // order asked for, repeats included, over more pages than one window holds, and a budget
    // charges one read per page the cache lacks, however many runs the page holds (two vectors
    // apart on each page: two runs, one read). The odd pages: the kernel reads ahead at the start
    // of a file and after sequential reads, so no page asked for arrives before its read.
    RawLayout L;
    L.vec_bytes = 1000;
    L.vectors_per_page = 4;
    L.num_pages = 1000;
    const std::string path = TempDir() + "/raw-many.pages";
    RawStore s(path, L, true);
    std::vector<uint32_t> all(L.locations());
    std::vector<uint8_t> vecs(all.size() * L.vec_bytes);
    for (uint32_t loc = 0; loc < all.size(); ++loc) {
      all[loc] = loc;
      for (uint32_t k = 0; k < L.vec_bytes; ++k) vecs[loc * L.vec_bytes + k] = static_cast<uint8_t>(loc * 31 + k * 7);
    }
    CHECK(s.Put(all.data(), all.size(), vecs.data(), RawOrigin::kStreamed) == all.size());
    std::vector<uint32_t> ask;  // two vectors on each of 500 pages, every tenth asked twice
    for (uint32_t p = 0; p < 500; ++p) {
      ask.push_back((2 * p + 1) * L.vectors_per_page + p % 2);
      ask.push_back((2 * p + 1) * L.vectors_per_page + p % 2 + 2);
    }
    for (size_t i = 0; i < 500; i += 10) ask.push_back(ask[i]);
    std::shuffle(ask.begin(), ask.end(), std::mt19937(7));
    auto right = [&](const std::vector<uint8_t>& out) {
      for (size_t i = 0; i < ask.size(); ++i)
        if (std::memcmp(&out[i * L.vec_bytes], &vecs[ask[i] * L.vec_bytes], L.vec_bytes) != 0) return false;
      return true;
    };
    std::vector<uint8_t> out(ask.size() * L.vec_bytes);
    s.Get(ask.data(), ask.size(), out.data());
    CHECK(right(out));
    ReadBudget b(1000000);
    s.SetReadBudget(&b);
    if (EvictFromCache(path)) {
      std::fill(out.begin(), out.end(), 0);
      s.Get(ask.data(), ask.size(), out.data());
      CHECK(right(out) && b.Stats().charged == 500);
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
  nav.Put<uint32_t>(0xFFFFFFFFu);  // the newest graph
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
  auto load0 = [&](uint8_t pq_source, const std::string& path) {
    rtier::BodyWriter w;
    w.Put<uint32_t>(0);  // partition
    w.Put<uint32_t>(0);  // version
    w.Put<uint8_t>(pq_source);
    w.PutBytes(path.data(), path.size());
    return w;
  };
  auto set_nodes = [](uint16_t self, const std::vector<std::pair<uint16_t, std::string>>& peers) {
    rtier::BodyWriter w;
    w.Put<uint16_t>(self);
    w.Put<uint16_t>(0);
    w.Put<uint32_t>(static_cast<uint32_t>(peers.size()));
    for (const auto& kv : peers) {
      w.Put<uint16_t>(kv.first);
      w.Put<uint16_t>(0);
      w.Put<uint32_t>(static_cast<uint32_t>(kv.second.size()));
      w.PutBytes(kv.second.data(), kv.second.size());
    }
    return w;
  };
  // The codes and vectors are still here: partition 0 loads again without any transfer.
  r = Call(fd, rtier::kLoadPartition, load0(0, seg0), 6);
  CHECK(r.status == rtier::kOk && e->pq_stats().live == ids0.size());

  // The hand-off: this node is node 1 (SET_NODES) and writes partition 0 as a segment that
  // records, for every posting, the node holding its vector -- itself (SAVE_PARTITION).
  r = Call(fd, rtier::kSetNodes, set_nodes(1, {}), 58);
  CHECK(r.status == rtier::kOk);
  const std::string saved0 = JoinPath(TempDir(), "wire-part0.seg");
  rtier::BodyWriter save;
  save.Put<uint32_t>(0);
  save.Put<uint32_t>(0);
  save.PutBytes(saved0.data(), saved0.size());
  r = Call(fd, rtier::kSavePartition, save, 57);
  const ListSegment s0 = ListSegment::Load(saved0);
  CHECK(r.status == rtier::kOk && s0.holders.size() == s0.ids.size() &&
        std::all_of(s0.holders.begin(), s0.holders.end(), [](uint16_t h) { return h == 1; }));

  // A node without the codes or vectors (and without the index's pages): the build's segment,
  // which records no holder, is refused (the vectors could not be fetched); the saved one
  // reports the codes absent; PQ_MISSING lists them and PQ_PUT installs them (as a peer would
  // send them); then the load succeeds.
  const std::string peer1 = "127.0.0.1:" + std::to_string(port);
  auto fresh = OpenNode(f, {}, false, f.index_nopages);
  rtier::NodeServer server2(fresh.get());
  const int port2 = server2.ListenTcp("127.0.0.1", 0);
  std::thread serve2([&] { server2.Serve(); });
  const int fd2 = Connect(port2);
  r = Call(fd2, rtier::kSetNodes, set_nodes(2, {{1, peer1}}), 56);
  CHECK(r.status == rtier::kOk);
  r = Call(fd2, rtier::kLoadPartition, load0(0, seg0), 59);
  CHECK(r.status == rtier::kRawAbsent);
  r = Call(fd2, rtier::kLoadPartition, load0(0, saved0), 60);
  CHECK(r.status == rtier::kPQAbsent);
  rtier::BodyWriter miss;
  miss.Put<uint32_t>(1);
  miss.Put<uint32_t>(1);
  miss.Put<uint32_t>(static_cast<uint32_t>(saved0.size()));
  miss.PutBytes(saved0.data(), saved0.size());
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
  r = Call(fd2, rtier::kLoadPartition, load0(0, saved0), 64);
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
    const std::vector<uint32_t> versions64(lists64.size(), 0);
    w.PutBytes(versions64.data(), versions64.size() * 4);
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

  // The stream: RAW_MISSING lists what is still missing, by holder (node 1 here), RAW_GET
  // reads it from the first node, RAW_PUT installs it.
  const std::vector<uint32_t> locs0 = Named(f, {0}).second;
  rtier::BodyWriter rmiss;
  rmiss.Put<uint32_t>(1);
  rmiss.Put<uint32_t>(0);  // partition 0
  r = Call(fd2, rtier::kRawMissing, rmiss, 73);
  CHECK(r.status == rtier::kOk);
  rtier::BodyReader rr(r.body.data(), r.body.size());
  CHECK(rr.Get<uint32_t>() == 1 && rr.Get<uint16_t>() == 1);
  rr.Get<uint16_t>();
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
  rtier::BodyWriter rget2;  // RAW_GET serves only what is here: there is no chain to follow
  rget2.Put<uint32_t>(1);
  rget2.Put<uint32_t>(other);
  r = Call(fd2, rtier::kRawGet, rget2, 76);
  CHECK(other != kInvalidId && r.status == rtier::kRawAbsent);
  rtier::BodyWriter rpull2;  // nor pulled: its list's partition is not here
  rpull2.Put<uint32_t>(1);
  rpull2.Put<uint32_t>(other);
  rpull2.Put<uint32_t>(ListNamingLoc(f, 3, other));
  r = Call(fd2, rtier::kRawPull, rpull2, 80);
  CHECK(r.status == rtier::kNotResident);
  // RAW_HELD_BY before node 1 leaves: every posting naming it has its vector here now.
  rtier::BodyWriter held;
  held.Put<uint16_t>(1);
  held.Put<uint16_t>(0);
  r = Call(fd2, rtier::kRawHeldBy, held, 81);
  CHECK(r.status == rtier::kOk && r.body.size() == 12);
  rtier::BodyReader hr(r.body.data(), r.body.size());
  CHECK(hr.Get<uint64_t>() > 0 && hr.Get<uint32_t>() == 0);
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
  // RAW_TRAFFIC: what node 1 served (the RERANK's fetch, the pull, the RAW_GET) all went to
  // node 2 (fetched, pulled, put), and the refused RAW_GET moved nothing.
  auto traffic = [](int conn, uint64_t id) {
    rtier::BodyWriter none;
    rtier::Frame t = Call(conn, rtier::kRawTraffic, none, id);
    CHECK(t.status == rtier::kOk && t.body.size() == 16);
    rtier::BodyReader tr(t.body.data(), t.body.size());
    const uint64_t sent = tr.Get<uint64_t>();
    return std::make_pair(sent, tr.Get<uint64_t>());
  };
  const uint64_t moved = e->raw_stats(false).served * vb;
  CHECK(moved >= static_cast<uint64_t>(fetched2 + nraw) * vb);
  CHECK(traffic(fd, 82) == std::make_pair(moved, uint64_t{0}));
  CHECK(traffic(fd2, 83) == std::make_pair(uint64_t{0}, moved));
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

// A RERANK fetches the raw vectors it lacks before it takes an engine worker: on a node with one
// worker, a FILTER completes while a RERANK waits at the holder. Holding the worker, the RERANK
// would keep it until the holder answered.
void TestFetchHoldsNoWorker() {
  Fixture& f = GetFixture();
  static const PartitionManifest man = PartitionManifest::Load(f.parts);
  auto src = OpenNode(f, {0, 1, 2, 3}, true);
  src->SetNodes(1, {});
  auto dst = OpenNode(f, {}, false, f.index_nopages, 0, 1);
  dst->SetNodes(2, {{1, "node-1"}});
  StageFrom(f, dst.get(), src.get(), {2, 3});
  // A query's lists in the moved partitions, its candidates from dst's FILTER and the old
  // owner's answer, before the server shares dst's only worker.
  const float* q = &f.queries[0];
  std::vector<uint32_t> lists(16), mine;
  const uint32_t nl = src->Navigate(q, 16, 0, lists.data());
  for (uint32_t i = 0; i < nl; ++i)
    if (man.list_part[lists[i]] >= 2) mine.push_back(lists[i]);
  const uint32_t nm = static_cast<uint32_t>(mine.size());
  const std::vector<uint32_t> versions(nm, 0);
  std::vector<uint32_t> cand(200), want_ids(10);
  std::vector<float> pqd(200), want_d(10);
  FilterStats fs;
  const uint32_t nc = dst->Filter(0, q, mine.data(), nullptr, nm, 200, cand.data(), pqd.data(), &fs);
  RerankStats rs;
  const uint32_t nw =
      src->Rerank(0, q, cand.data(), nc, 10, mine.data(), nullptr, nm, want_ids.data(), want_d.data(), &rs);
  CHECK(nm > 0 && nc > 0 && nw == 10);

  rtier::NodeServer server(dst.get());
  // The holder answers only when let go. (This fetcher replaces the server's.)
  std::mutex mu;
  std::condition_variable cv;
  bool waiting = false, go = false;
  dst->SetRawFetcher([&](const std::string&, const uint32_t* locs, size_t n, uint8_t* out) {
    {
      std::unique_lock<std::mutex> l(mu);
      waiting = true;
      cv.notify_all();
      cv.wait(l, [&] { return go; });
    }
    src->RawGet(locs, n, out);
  });
  const int port = server.ListenTcp("127.0.0.1", 0);
  std::thread serve([&] { server.Serve(); });
  const int a = Connect(port), b = Connect(port);

  rtier::BodyWriter rr;
  rr.Put<uint32_t>(10);
  rr.Put<uint32_t>(nc);
  rr.PutBytes(cand.data(), nc * 4);
  rr.Put<uint32_t>(nm);
  rr.PutBytes(mine.data(), nm * 4);
  rr.PutBytes(versions.data(), nm * 4);
  rr.PutBytes(q, f.dim * sizeof(float));
  rtier::Frame reranked;
  std::thread first([&] { reranked = Call(a, rtier::kRerank, rr, 1); });
  {
    std::unique_lock<std::mutex> l(mu);
    CHECK(cv.wait_for(l, std::chrono::seconds(30), [&] { return waiting; }));
  }
  rtier::BodyWriter fr;
  fr.Put<uint32_t>(200);
  fr.Put<uint32_t>(nm);
  fr.PutBytes(mine.data(), nm * 4);
  fr.PutBytes(versions.data(), nm * 4);
  fr.PutBytes(q, f.dim * sizeof(float));
  std::promise<rtier::Frame> filtered;
  std::future<rtier::Frame> done = filtered.get_future();
  std::thread second([&] { filtered.set_value(Call(b, rtier::kFilter, fr, 2)); });
  const bool worker_free = done.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
  {
    std::lock_guard<std::mutex> l(mu);
    go = true;
  }
  cv.notify_all();
  first.join();
  second.join();
  CHECK(worker_free && done.get().status == rtier::kOk);

  // The RERANK answers as the old owner does, and counts what it fetched before the worker.
  CHECK(reranked.status == rtier::kOk);
  rtier::BodyReader br(reranked.body.data(), reranked.body.size());
  const uint32_t cnt = br.Get<uint32_t>();
  std::vector<uint32_t> got_ids(cnt);
  std::vector<float> got_d(cnt);
  std::memcpy(got_ids.data(), br.Bytes(cnt * 4), cnt * 4);
  std::memcpy(got_d.data(), br.Bytes(cnt * 4), cnt * 4);
  br.Get<uint32_t>();  // pages read
  const uint32_t fetched = br.Get<uint32_t>();
  CHECK(cnt == nw && got_ids == want_ids && got_d == want_d && fetched > 0 &&
        fetched == dst->raw_stats(true).fetched);
  ::close(a);
  ::close(b);
  server.Stop();
  serve.join();
}

// The navigation graph's lists for every query at `graph`.
std::vector<std::vector<uint32_t>> NavigateAll(const Fixture& f, NodeEngine& e, uint32_t graph) {
  std::vector<std::vector<uint32_t>> out;
  for (uint32_t q = 0; q < f.nq; ++q) {
    std::vector<uint32_t> lists(16);
    const uint32_t n = e.Navigate(&f.queries[q * f.dim], 16, 40, lists.data(), graph);
    lists.resize(n);
    out.push_back(lists);
  }
  return out;
}

template <class F>
bool Throws(F&& fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

// Graph patches (U14 major epochs): the router edits its own graph and exports a patch; an entry
// stages it next to its graph -- its own version still searches the graph alone, the next one
// searches through the patch, exactly as the router's edited graph -- and merges it in place
// while queries search through it.
void TestGraphPatches() {
  Fixture& f = GetFixture();
  auto router = OpenNode(f, {}, true);
  auto entry = OpenNode(f, {}, true);
  auto later = OpenNode(f, {}, true);  // gets the patch as a file
  const uint32_t nl = router->num_lists();
  const GraphState g0 = router->graph_state();
  CHECK(g0.loaded && g0.version == 0 && g0.patch == 0 && g0.count == nl && g0.deleted == 0);
  CHECK(g0.capacity >= nl + nl / 5);  // the default headroom, 20%

  const auto before = NavigateAll(f, *entry, 0);
  // Three new heads at queries (so that queries find them) and two heads queries probe first.
  const std::vector<uint32_t> labels = {nl, nl + 1, nl + 2};
  const std::vector<float> vecs(f.queries.begin(), f.queries.begin() + 3 * f.dim);
  std::vector<uint32_t> dead = {before[5][0], before[7][0]};
  if (dead[0] == dead[1]) dead[1] = before[7][1];
  const std::vector<uint8_t> patch = router->EditGraph(1, labels.data(), vecs.data(), 3, dead.data(), 2);
  const GraphState g1 = router->graph_state();
  CHECK(g1.version == 1 && g1.count == nl + 3 && g1.deleted == 2 && g1.patch == 0);
  const auto want = NavigateAll(f, *router, kNewestGraph);
  bool fresh = false;
  for (const auto& ls : want)
    for (uint32_t c : ls) {
      fresh = fresh || c >= nl;
      CHECK(c != dead[0] && c != dead[1]);
    }
  CHECK(fresh);
  CHECK(want != before);

  // Staged: the graph's version searches the graph alone, the patch's version through it.
  entry->StageGraphPatch(patch.data(), patch.size());
  CHECK(entry->graph_state().version == 0 && entry->graph_state().patch == 1);
  CHECK(NavigateAll(f, *entry, 0) == before);
  CHECK(NavigateAll(f, *entry, 1) == want);
  CHECK(NavigateAll(f, *entry, kNewestGraph) == want);
  CHECK(Throws([&] { NavigateAll(f, *entry, 2); }));
  const std::string path = JoinPath(TempDir(), "patch-1.bin");
  entry->SaveGraphPatch(1, path);

  // Merged in place while queries search through the patch: none sees anything else.
  std::atomic<bool> stop{false};
  std::atomic<int> wrong{0}, rounds{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < 4; ++t)
    ts.emplace_back([&] {
      while (!stop) {
        if (NavigateAll(f, *entry, 1) != want) ++wrong;
        ++rounds;
      }
    });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  entry->MergeGraphPatch(1);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  stop = true;
  for (auto& t : ts) t.join();
  CHECK(wrong == 0 && rounds > 4);
  const GraphState ge = entry->graph_state();
  CHECK(ge.version == 1 && ge.patch == 0 && ge.count == nl + 3 && ge.deleted == 2);
  CHECK(NavigateAll(f, *entry, 1) == want);
  CHECK(Throws([&] { NavigateAll(f, *entry, 0); }));  // the old version is gone
  entry->MergeGraphPatch(1);  // merged already: nothing to do

  // A later entry: the build's graph, then the patch from the file.
  const std::vector<uint8_t> file = ReadFile(path);
  later->StageGraphPatch(file.data(), file.size());
  later->MergeGraphPatch(1);
  CHECK(NavigateAll(f, *later, 1) == want);

  // Refused, with nothing changed: a patch for another version, a damaged one.
  CHECK(Throws([&] { later->StageGraphPatch(patch.data(), patch.size()); }));
  std::vector<uint8_t> broken = patch;
  broken[broken.size() / 2] ^= 1;
  auto other = OpenNode(f, {}, true);
  CHECK(Throws([&] { other->StageGraphPatch(broken.data(), broken.size()); }));
  CHECK(other->graph_state().patch == 0);

  // An edit beyond the headroom, or one deleting a deleted head, changes nothing.
  std::vector<uint32_t> many(g1.capacity - g1.count + 1);
  for (size_t i = 0; i < many.size(); ++i) many[i] = nl + 3 + static_cast<uint32_t>(i);
  const std::vector<float> mv(many.size() * f.dim, 0.5f);
  CHECK(Throws([&] { router->EditGraph(2, many.data(), mv.data(), static_cast<uint32_t>(many.size()), nullptr, 0); }));
  CHECK(Throws([&] { router->EditGraph(2, nullptr, nullptr, 0, dead.data(), 1); }));
  CHECK(Throws([&] { router->EditGraph(3, nullptr, nullptr, 0, &before[9][0], 1); }));  // not the next version
  CHECK(router->graph_state().count == g1.count && router->graph_state().version == 1);

  // A second edit on top of the first; a patch staged and dropped.
  const std::vector<uint32_t> l2 = {nl + 3};
  const std::vector<float> v2(f.queries.begin() + 10 * f.dim, f.queries.begin() + 11 * f.dim);
  const std::vector<uint8_t> p2 = router->EditGraph(2, l2.data(), v2.data(), 1, nullptr, 0);
  entry->StageGraphPatch(p2.data(), p2.size());
  CHECK(entry->DropGraphPatch(2) && !entry->DropGraphPatch(2) && entry->graph_state().patch == 0);
  entry->StageGraphPatch(p2.data(), p2.size());
  entry->MergeGraphPatch(2);
  CHECK(NavigateAll(f, *entry, 2) == NavigateAll(f, *router, kNewestGraph));

  // The router's decisions, on the PQ codes of a list's members.
  auto owner = OpenNode(f, {0}, false);
  owner->SetNodes(1, {});
  const ListSegment s0 = ListSegment::Load(JoinPath(f.parts, partfiles::SegmentName(0)));
  size_t longest = 0;
  for (size_t i = 1; i < s0.list_ids.size(); ++i)
    if (s0.offsets[i + 1] - s0.offsets[i] > s0.offsets[longest + 1] - s0.offsets[longest]) longest = i;
  const ListPostings lp = owner->GetList(s0.list_ids[longest], 0);
  const uint32_t n = static_cast<uint32_t>(lp.ids.size());
  CHECK(n >= 4);
  const uint32_t m = owner->pq_m();
  std::vector<uint8_t> codes(static_cast<size_t>(n) * m);
  owner->PQGet(lp.ids.data(), n, codes.data());
  std::vector<uint8_t> side(n), side2(n);
  uint32_t heads[2], heads2[2];
  router->SplitCodes(codes.data(), n, side.data(), heads);
  router->SplitCodes(codes.data(), n, side2.data(), heads2);
  const uint32_t ones = static_cast<uint32_t>(std::count(side.begin(), side.end(), 1));
  CHECK(ones == n / 2 && side == side2 && heads[0] == heads2[0] && heads[1] == heads2[1]);
  CHECK(heads[0] < n && heads[1] < n && side[heads[0]] == 0 && side[heads[1]] == 1);
  std::vector<uint32_t> nearest(n);
  router->NearestHeads(codes.data(), n, 32, nearest.data());
  for (uint32_t c : nearest) CHECK(c < nl + 4 && c != dead[0] && c != dead[1]);
}

// List edits (U14 major epochs): a list split inside its partition, a list merged into a list
// another node owns -- its postings name the node that holds their vectors, fetched from there
// when a query needs them -- and a later owner that learns the new lists from the segment.
void TestListEdits() {
  Fixture& f = GetFixture();
  auto x = OpenNode(f, {0, 1}, true);
  auto y = OpenNode(f, {2, 3}, true);
  LinkPeers(x.get(), 1, {{2, y.get()}});
  LinkPeers(y.get(), 2, {{1, x.get()}});
  const uint32_t nl = x->num_lists();
  const uint32_t m = x->pq_m();
  auto longest_of = [&](uint32_t part) {
    const ListSegment s = ListSegment::Load(JoinPath(f.parts, partfiles::SegmentName(part)));
    size_t best = 0;
    for (size_t i = 1; i < s.list_ids.size(); ++i)
      if (s.offsets[i + 1] - s.offsets[i] > s.offsets[best + 1] - s.offsets[best]) best = i;
    return s.list_ids[best];
  };
  auto shortest_of = [&](uint32_t part) {
    const ListSegment s = ListSegment::Load(JoinPath(f.parts, partfiles::SegmentName(part)));
    size_t best = s.list_ids.size();
    for (size_t i = 0; i < s.list_ids.size(); ++i)
      if (s.offsets[i + 1] > s.offsets[i] &&
          (best == s.list_ids.size() || s.offsets[i + 1] - s.offsets[i] < s.offsets[best + 1] - s.offsets[best]))
        best = i;
    return s.list_ids[best];
  };
  auto filter = [&](NodeEngine& e, uint32_t qi, const std::vector<uint32_t>& lists, const std::vector<uint32_t>& vers) {
    std::vector<uint32_t> ids(4096);
    std::vector<float> d(4096);
    FilterStats st;
    const uint32_t k = e.Filter(0, &f.queries[qi * f.dim], lists.data(), vers.data(),
                                static_cast<uint32_t>(lists.size()), 4096, ids.data(), d.data(), &st);
    ids.resize(k);
    return ids;
  };

  // Split L (partition 0) into nl and nl + 1, inside the partition: every posting stays held here.
  const uint32_t L = longest_of(0);
  const ListPostings lp = x->GetList(L, 0);
  const uint32_t n = static_cast<uint32_t>(lp.ids.size());
  for (uint16_t h : lp.holders) CHECK(h == 1);
  std::vector<uint8_t> codes(static_cast<size_t>(n) * m), side(n);
  x->PQGet(lp.ids.data(), n, codes.data());
  uint32_t heads[2];
  x->SplitCodes(codes.data(), n, side.data(), heads);
  std::vector<uint32_t> lists(n);
  for (uint32_t i = 0; i < n; ++i) lists[i] = nl + side[i];
  const std::vector<uint32_t> created = {nl, nl + 1}, created_parts = {0, 0};
  VersionEdits e;
  e.lists = lists.data();
  e.ids = lp.ids.data();
  e.locs = lp.locs.data();
  e.holders = lp.holders.data();
  e.nadds = n;
  e.created = created.data();
  e.created_parts = created_parts.data();
  e.ncreated = 2;
  e.removed = &L;
  e.nremoved = 1;
  const auto res = x->StageVersions({{0, 0, 1}}, e);
  CHECK(res.size() == 1 && res[0].changed && res[0].added == n && res[0].removed == n);
  std::map<uint32_t, uint32_t> lens(res[0].lengths.begin(), res[0].lengths.end());
  const uint32_t ones = static_cast<uint32_t>(std::count(side.begin(), side.end(), 1));
  CHECK(lens.size() == 2 && lens[nl] == n - ones && lens[nl + 1] == ones);
  // The children name exactly L's postings: the same FILTER answer, at version 1 against 0.
  for (uint32_t qi = 0; qi < 5; ++qi) CHECK(filter(*x, qi, {nl, nl + 1}, {1, 1}) == filter(*x, qi, {L}, {0}));
  CHECK(Throws([&] { x->GetList(L, 1); }));
  CHECK(x->GetList(nl + 1, 1).ids.size() == ones);

  // A later owner gets partition 0 at version 1 and learns the new lists from its segment.
  auto z = OpenNode(f, {}, false);
  LinkPeers(z.get(), 3, {{1, x.get()}});
  const std::string seg = JoinPath(TempDir(), "p0v1.seg");
  x->SavePartition(0, 1, seg);
  const auto miss = z->PQMissing({{seg}});
  std::vector<uint8_t> mc(miss[0].size() * m);
  x->PQGet(miss[0].data(), miss[0].size(), mc.data());
  z->PQPut(miss[0].data(), miss[0].size(), mc.data());
  z->LoadPartition(0, seg, PQSource::kPresent, 1);
  for (uint32_t qi = 0; qi < 5; ++qi) CHECK(filter(*z, qi, {nl, nl + 1}, {1, 1}) == filter(*x, qi, {nl, nl + 1}, {1, 1}));
  for (uint16_t h : z->GetList(nl, 1).holders) CHECK(h == 1);  // z holds none of them

  // Merge S (partition 1, on x) into N (partition 2, on y): its postings move to another owner,
  // with their codes, and name x as their vectors' holder.
  const uint32_t S = shortest_of(1), N = shortest_of(2);
  const ListPostings sp = x->GetList(S, 0);
  const uint32_t ns = static_cast<uint32_t>(sp.ids.size());
  std::vector<uint8_t> sc(static_cast<size_t>(ns) * m);
  x->PQGet(sp.ids.data(), ns, sc.data());
  y->PQPut(sp.ids.data(), ns, sc.data());
  const std::vector<uint32_t> to_n(ns, N);
  VersionEdits me;
  me.lists = to_n.data();
  me.ids = sp.ids.data();
  me.locs = sp.locs.data();
  me.holders = sp.holders.data();
  me.nadds = ns;
  const auto ry = y->StageVersions({{2, 0, 1}}, me);
  CHECK(ry.size() == 1 && ry[0].changed && ry[0].added > 0 && ry[0].lengths.size() == 1 && ry[0].lengths[0].first == N);
  VersionEdits rm;
  rm.removed = &S;
  rm.nremoved = 1;
  const auto rx = x->StageVersions({{1, 0, 1}}, rm);
  CHECK(rx.size() == 1 && rx[0].changed && rx[0].removed == ns);
  // A vector of S that y lacks: re-ranked on y through N at version 1, fetched from x in one hop.
  uint32_t v = ns;
  for (uint32_t i = 0; i < ns && v == ns; ++i)
    if (!y->RawAbsent(&sp.locs[i], 1).empty()) v = i;
  CHECK(v < ns);
  if (v < ns) {
    const float* q = &f.queries[0];
    uint32_t oy = 0, ox = 0;
    float dy = 0, dx = 0;
    RerankStats sty, stx;
    const uint32_t vers1 = 1, vers0 = 0;
    CHECK(y->Rerank(0, q, &sp.ids[v], 1, 1, &N, &vers1, 1, &oy, &dy, &sty) == 1);
    CHECK(x->Rerank(0, q, &sp.ids[v], 1, 1, &S, &vers0, 1, &ox, &dx, &stx) == 1);
    CHECK(oy == ox && dy == dx && sty.fetched == 1);
    CHECK(y->RawAbsent(&sp.locs[v], 1).empty());
  }

  // Refused, with nothing changed.
  const auto vy = y->ResidentVersions();
  const auto vx = x->ResidentVersions();
  VersionEdits again;  // a list that exists already
  again.created = created.data();
  again.created_parts = created_parts.data();
  again.ncreated = 1;
  CHECK(Throws([&] { x->StageVersions({{0, 1, 2}}, again); }));
  VersionEdits to_gone;  // a posting for a list the base version no longer has
  to_gone.lists = &S;
  to_gone.ids = &sp.ids[0];
  to_gone.locs = &sp.locs[0];
  to_gone.nadds = 1;
  CHECK(Throws([&] { x->StageVersions({{1, 1, 2}}, to_gone); }));
  VersionEdits nowhere;  // a vector y lacks, with no holder
  const uint32_t N3 = longest_of(3);
  uint32_t absent = ns;
  for (uint32_t i = 0; i < ns && absent == ns; ++i)
    if (!y->RawAbsent(&sp.locs[i], 1).empty()) absent = i;
  if (absent < ns) {
    nowhere.lists = &N3;
    nowhere.ids = &sp.ids[absent];
    nowhere.locs = &sp.locs[absent];
    nowhere.nadds = 1;
    bool raw_missing = false;
    try {
      y->StageVersions({{3, 0, 1}}, nowhere);
    } catch (const RawMissingError&) {
      raw_missing = true;
    }
    CHECK(raw_missing);
  }
  VersionEdits beyond;  // a list beyond the room for lists
  const uint32_t far = x->graph_state().capacity;
  beyond.created = &far;
  beyond.created_parts = created_parts.data();
  beyond.ncreated = 1;
  CHECK(Throws([&] { x->StageVersions({{0, 1, 2}}, beyond); }));
  CHECK(y->ResidentVersions() == vy && x->ResidentVersions() == vx);
}

// A second generation of the fixture's index (U14 full rebuild): re-clustered with another seed
// into 3 partitions, with the first generation's codebook, codes and page layout -- what
// fusion_build reuses when those files are already in its output directory.
struct Generation {
  std::string index, parts;
};

const Generation& GetGeneration2() {
  static Generation g = [] {
    const Fixture& f = GetFixture();
    const std::string dir = f.index.substr(0, f.index.rfind('/'));
    Generation x;
    x.index = dir + "/index-g2";
    MakeDirs(x.index);
    for (const std::string& name : {std::string(files::kPQCodebook), std::string(files::kPQCodes),
                                    files::LayoutMap("bucket"), files::LayoutVectors("bucket")})
      CHECK(::symlink(JoinPath(f.index, name).c_str(), JoinPath(x.index, name).c_str()) == 0);
    BuildParams bp;
    bp.base_path = dir + "/base.fbin";
    bp.out_dir = x.index;
    bp.pq_m = 8;
    bp.pq_train_n = 4000;
    bp.seed = 7;
    bp.clustering.seed = 7;
    BuildIndex(bp);
    PostingLists pl = PostingLists::Load(JoinPath(x.index, files::kPostings));
    std::vector<uint32_t> assign(pl.num_lists());
    for (uint32_t c = 0; c < pl.num_lists(); ++c) assign[c] = static_cast<uint32_t>(uint64_t(c) * 3 / pl.num_lists());
    x.parts = dir + "/parts-g2";
    WritePartitions(pl, assign, 3, x.parts, LayoutMap::Load(JoinPath(x.index, files::LayoutMap("bucket"))), "bucket");
    return x;
  }();
  return g;
}

std::unique_ptr<NodeEngine> OpenOn(const std::string& index, const std::string& parts) {
  NodeOptions o;
  o.index_dir = index;
  o.partitions_dir = parts;
  o.num_workers = 2;
  o.max_nprobe = 64;
  o.max_rerank = 4096;
  o.max_list_len = 4096;  // room for either generation's lists
  o.direct_io = false;
  return NodeEngine::Open(o);
}

// Full rebuild (U14): two nodes that served the first generation move to the second without a
// restart. They keep their PQ codes and raw vectors -- the new partitions' codes come from the
// generation's index, a vector that only the other node holds is fetched from it, as after a
// migration -- and answer as a node built on the second generation does.
void TestGenerations() {
  const Fixture& f = GetFixture();
  const Generation& g2 = GetGeneration2();
  const PartitionManifest man2 = PartitionManifest::Load(g2.parts);
  CHECK(man2.num_partitions == 3);

  // The second generation's oracle: one node built on it.
  auto oracle = OpenOn(g2.index, g2.parts);
  oracle->LoadGraph(JoinPath(g2.index, files::kGraph));
  for (uint32_t p = 0; p < 3; ++p)
    oracle->LoadPartition(p, JoinPath(g2.parts, partfiles::SegmentName(p)), PQSource::kIndex);

  // Generation 0 on two nodes: a holds partitions 0-1, b 2-3, each with their raw vectors.
  auto a = OpenOn(f.index, f.parts), b = OpenOn(f.index, f.parts);
  for (uint32_t p = 0; p < 4; ++p)
    (p < 2 ? a : b)->LoadPartition(p, JoinPath(f.parts, partfiles::SegmentName(p)), PQSource::kIndex);
  a->LoadGraph(JoinPath(f.index, files::kGraph));
  LinkPeers(a.get(), 1, {{2, b.get()}});
  LinkPeers(b.get(), 2, {{1, a.get()}});
  // The bitmap: what the node holds.
  const std::vector<uint64_t> bits_a = a->RawBitmap(), bits_b = b->RawBitmap();
  auto popcount = [](const std::vector<uint64_t>& v) {
    uint64_t n = 0;
    for (uint64_t w : v) n += static_cast<uint64_t>(__builtin_popcountll(w));
    return n;
  };
  CHECK(popcount(bits_a) == a->raw_stats(true).present && popcount(bits_b) == b->raw_stats(true).present);
  auto has = [](const std::vector<uint64_t>& v, uint32_t loc) { return (v[loc >> 6] >> (loc & 63)) & 1; };
  for (uint32_t loc : Named(f, {0, 1}).second) CHECK(has(bits_a, loc));

  // Refused while a partition is resident; refused for another codebook. Nothing changes.
  CHECK(Throws([&] { a->SwitchGeneration(1, g2.index, g2.parts); }));
  CHECK(a->generation() == 0 && a->num_partitions() == 4 && a->has_graph());
  for (uint32_t p = 0; p < 4; ++p) (p < 2 ? a : b)->EvictPartition(p);
  const std::string other = TempDir() + "/index-other";
  MakeDirs(other);
  for (const char* name : {files::kMeta, files::kGraph, files::kPQCodes})
    CHECK(::symlink(JoinPath(g2.index, name).c_str(), JoinPath(other, name).c_str()) == 0);
  std::vector<uint8_t> cb = ReadFile(JoinPath(g2.index, files::kPQCodebook));
  cb[cb.size() - 1] ^= 0x40;  // one centroid differs
  WriteFile(JoinPath(other, files::kPQCodebook), cb.data(), cb.size());
  CHECK(Throws([&] { a->SwitchGeneration(1, other, g2.parts); }));
  CHECK(a->generation() == 0 && a->num_partitions() == 4 && a->has_graph());

  // The switch: the graph goes, the codes and raw vectors stay.
  const uint64_t codes_a = a->pq_stats().resident, present_a = a->raw_stats(true).present;
  a->SwitchGeneration(1, g2.index, g2.parts);
  b->SwitchGeneration(1, g2.index, g2.parts);
  CHECK(a->generation() == 1 && a->num_partitions() == 3 && a->num_lists() == man2.num_lists && !a->has_graph());
  CHECK(a->pq_stats().resident == codes_a && a->raw_stats(true).present == present_a);
  a->LoadGraph(JoinPath(g2.index, files::kGraph));

  // The second generation's partitions: 0-1 on a, 2 on b. A posting whose vector its owner
  // lacks names the other node, which holds it: one hop.
  const std::string staged = TempDir();
  auto stage = [&](NodeEngine* owner, const std::vector<uint64_t>& mine, uint16_t other_id, uint32_t p) {
    ListSegment s = ListSegment::Load(JoinPath(g2.parts, partfiles::SegmentName(p)));
    s.holders.assign(s.ids.size(), kNoHolder);
    for (size_t j = 0; j < s.ids.size(); ++j)
      if (!has(mine, s.locs[j])) s.holders[j] = other_id;
    const std::string path = JoinPath(staged, partfiles::SegmentName(p));
    s.Save(path);
    owner->LoadPartition(p, path, PQSource::kGenIndex);
  };
  // Without holders, a vector that is not here is refused before anything changes.
  CHECK(Throws([&] { b->LoadPartition(2, JoinPath(g2.parts, partfiles::SegmentName(2)), PQSource::kGenIndex); }));
  CHECK(b->ResidentPartitions().empty());
  stage(a.get(), bits_a, 2, 0);
  stage(a.get(), bits_a, 2, 1);
  stage(b.get(), bits_b, 1, 2);
  CHECK(a->ResidentPartitions() == std::vector<uint32_t>({0, 1}) && b->ResidentPartitions() == std::vector<uint32_t>({2}));

  // Every answer is the second generation's: navigate on a, filter and re-rank at each owner.
  const uint32_t k = 10, nprobe = 16;
  bool same = true;
  for (uint32_t qi = 0; qi < f.nq; ++qi) {
    const float* q = &f.queries[qi * f.dim];
    SearchParams sp;
    sp.k = k;
    sp.nprobe = nprobe;
    sp.rerank = 4096;
    sp.rr.heuristic = false;
    std::vector<uint32_t> ids(k);
    std::vector<float> d(k);
    QueryStats qs;
    const uint32_t n0 = oracle->SearchLocal(0, q, sp, ids.data(), d.data(), &qs);
    Result want;
    for (uint32_t i = 0; i < n0; ++i) want.emplace_back(d[i], ids[i]);
    std::vector<uint32_t> lists(nprobe);
    const uint32_t nl = a->Navigate(q, nprobe, 0, lists.data());
    std::map<int, std::vector<uint32_t>> by_node;
    for (uint32_t i = 0; i < nl; ++i) by_node[man2.list_part[lists[i]] < 2 ? 0 : 1].push_back(lists[i]);
    std::map<uint32_t, float> merged;
    for (auto& [node, ls] : by_node) {
      NodeEngine* e = node == 0 ? a.get() : b.get();
      std::vector<uint32_t> cand(4096), rid(k);
      std::vector<float> pqd(4096), rd(k);
      FilterStats fs;
      const uint32_t nc = e->Filter(0, q, ls.data(), nullptr, static_cast<uint32_t>(ls.size()), 4096,
                                    cand.data(), pqd.data(), &fs);
      RerankStats rs;
      const uint32_t nr = e->Rerank(1, q, cand.data(), nc, k, ls.data(), nullptr, static_cast<uint32_t>(ls.size()),
                                    rid.data(), rd.data(), &rs);
      for (uint32_t i = 0; i < nr; ++i) merged[rid[i]] = rd[i];
    }
    Result got;
    for (const auto& [id, dist] : merged) got.emplace_back(dist, id);
    std::sort(got.begin(), got.end());
    if (got.size() > k) got.resize(k);
    same = same && got == want;
  }
  CHECK(same);
  // Some vectors came from the other node, and only on demand.
  CHECK(a->raw_stats(true).fetched + b->raw_stats(true).fetched > 0);
}

}  // namespace

int main() {
  const std::vector<std::pair<const char*, std::function<void()>>> tests = {
      {"crc32", TestCrc32},
      {"partition_files", TestPartitionFiles},
      {"node_primitives", TestNodePrimitives},
      {"partition_versions", TestPartitionVersions},
      {"inserted_vectors", TestInsertedVectors},
      {"stage_versions", TestStageVersions},
      {"pq_store", TestPQStore},
      {"pq_migration", TestPQMigration},
      {"pq_evict_under_load", TestPQEvictUnderLoad},
      {"raw_store", TestRawStore},
      {"raw_migration", TestRawMigration},
      {"read_budget", TestReadBudget},
      {"graph_patches", TestGraphPatches},
      {"list_edits", TestListEdits},
      {"generations", TestGenerations},
      {"wire", TestWire},
      {"fetch_holds_no_worker", TestFetchHoldsNoWorker},
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
