#include "fusion/node_engine.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <iterator>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "fusion/dataset.h"
#include "fusion/distance.h"
#include "fusion/filter.h"
#include "fusion/hnsw_space.h"
#include "fusion/index_meta.h"
#include "fusion/layout.h"
#include "fusion/partition.h"
#include "fusion/pq.h"

namespace fusion {
namespace {

constexpr uint64_t kPQFileHeader = 16;  // pq_codes.bin: u64 n, u32 m, u32 reserved (fusion/pq.h)

// A partition's posting lists as held by a node, with the PQ slot of every posting (resolved
// once at load time: slots do not move while referenced).
struct ResidentSegment {
  ListSegment seg;
  std::vector<uint32_t> slots;                    // parallel to seg.ids
  mutable const ResidentSegment* next = nullptr;  // Reclaimer queue
};

// Drops the PQ references of released segments and deletes them, off the query path: the last
// holder of an evicted segment -- often a query that pinned it -- only queues it (no
// allocation), and a background thread walks its postings. Flush waits for the queue.
class Reclaimer {
 public:
  explicit Reclaimer(PQStore* store) : store_(store), thread_([this] { Run(); }) {}
  Reclaimer(const Reclaimer&) = delete;
  Reclaimer& operator=(const Reclaimer&) = delete;
  ~Reclaimer() {  // handles whatever is still queued
    {
      std::lock_guard<std::mutex> l(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
  }

  void Push(const ResidentSegment* r) {
    {
      std::lock_guard<std::mutex> l(mu_);
      r->next = head_;
      head_ = r;
      ++queued_;
    }
    cv_.notify_all();
  }

  // Returns once every segment queued before the call has been handled.
  void Flush() {
    std::unique_lock<std::mutex> l(mu_);
    const uint64_t target = queued_;
    done_.wait(l, [&] { return freed_ >= target; });
  }

 private:
  void Run() {
    std::unique_lock<std::mutex> l(mu_);
    for (;;) {
      cv_.wait(l, [&] { return stop_ || head_ != nullptr; });
      if (head_ == nullptr) return;  // stopping, and nothing left
      const ResidentSegment* list = head_;
      head_ = nullptr;
      l.unlock();
      uint64_t n = 0;
      while (list != nullptr) {
        const ResidentSegment* next = list->next;
        store_->Unref(list->seg.ids.data(), list->seg.ids.size());
        delete list;
        list = next;
        ++n;
      }
      l.lock();
      freed_ += n;
      done_.notify_all();
    }
  }

  PQStore* store_;
  std::mutex mu_;
  std::condition_variable cv_, done_;
  const ResidentSegment* head_ = nullptr;
  uint64_t queued_ = 0, freed_ = 0;
  bool stop_ = false;
  std::thread thread_;  // last member: starts once the others are initialized
};

// Read-only file for positioned reads (thread-safe).
class ReadOnlyFile {
 public:
  ReadOnlyFile() = default;
  ReadOnlyFile(const ReadOnlyFile&) = delete;
  ReadOnlyFile& operator=(const ReadOnlyFile&) = delete;
  ~ReadOnlyFile() {
    if (fd_ >= 0) ::close(fd_);
  }
  void Open(const std::string& path) {
    path_ = path;
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    FUSION_CHECK(fd_ >= 0, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  }
  uint64_t Size() const {
    struct stat sb;
    FUSION_CHECK(::fstat(fd_, &sb) == 0, "stat %s: %s", path_.c_str(), std::strerror(errno));
    return static_cast<uint64_t>(sb.st_size);
  }
  void ReadAt(void* buf, size_t n, uint64_t off) const {
    size_t got = 0;
    while (got < n) {
      const ssize_t r = ::pread(fd_, static_cast<char*>(buf) + got, n - got,
                                static_cast<off_t>(off + got));
      if (r < 0 && errno == EINTR) continue;
      FUSION_CHECK(r > 0, "read %s at %llu: %s", path_.c_str(),
                   static_cast<unsigned long long>(off + got),
                   r == 0 ? "unexpected end of file" : std::strerror(errno));
      got += static_cast<size_t>(r);
    }
  }

 private:
  std::string path_;
  int fd_ = -1;
};

template <class T>
class NodeEngineImpl : public NodeEngine {
 public:
  NodeEngineImpl(const NodeOptions& o, const IndexMeta& meta, PartitionManifest man)
      : opts_(o), man_(std::move(man)) {
    auto path = [&](const std::string& f) { return JoinPath(o.index_dir, f); };
    dim_ = static_cast<uint32_t>(meta.GetU64("dim"));
    n_ = meta.GetU64("num_vectors");
    FUSION_CHECK(man_.payload == kPayloadListsLocations,
                 "the partition directory %s has payload '%s': its posting lists carry no raw-vector "
                 "locations (rerun rtier_segment)",
                 o.partitions_dir.c_str(), man_.payload.c_str());
    FUSION_CHECK(o.layout.empty() || o.layout == man_.layout,
                 "the partitions' locations refer to the '%s' layout, not '%s'",
                 man_.layout.c_str(), o.layout.c_str());
    const std::string layouts = meta.GetOr("layouts", "");
    FUSION_CHECK(("," + layouts + ",").find("," + man_.layout + ",") != std::string::npos,
                 "index has no '%s' layout (available: %s)", man_.layout.c_str(), layouts.c_str());
    FUSION_CHECK(man_.raw.vec_bytes == dim_ * DTypeSize(DTypeOf<T>()),
                 "the partitions' page geometry has %u-byte vectors, the index %zu-byte ones",
                 man_.raw.vec_bytes, dim_ * DTypeSize(DTypeOf<T>()));
    FUSION_CHECK(o.num_workers >= 1, "need at least one worker");
    FUSION_CHECK(o.max_nprobe >= 1 && o.max_rerank >= 1, "max_nprobe and max_rerank must be >= 1");
    space_ = MakeL2Space(DTypeOf<T>(), dim_);
    segs_.resize(man_.num_partitions);
    list_src_.assign(man_.num_lists, 0);
    index_pages_path_ = path(files::LayoutVectors(man_.layout));
    const uint64_t max_cand =
        static_cast<uint64_t>(std::min<uint32_t>(o.max_nprobe, man_.num_lists)) *
        std::max<uint32_t>(man_.max_list_len, 1);
    FUSION_CHECK(max_cand < 0xFFFFFFFFull, "max_nprobe too large");
    max_candidates_ = static_cast<uint32_t>(max_cand);

    // PQ codes: node-level store, filled as partitions arrive (fusion/pq_store.h). The index's
    // pq_codes.bin is only read to bootstrap (PQSource::kIndex).
    PQCodebook cb = PQCodebook::Load(path(files::kPQCodebook));
    pq_file_.Open(path(files::kPQCodes));
    uint64_t hdr[2] = {0, 0};  // u64 n, then u32 m + u32 reserved
    pq_file_.ReadAt(hdr, sizeof(hdr), 0);
    const uint32_t code_m = static_cast<uint32_t>(hdr[1] & 0xFFFFFFFFu);
    FUSION_CHECK(hdr[0] == n_ && code_m == cb.m(), "PQ codes do not match the index");
    FUSION_CHECK(pq_file_.Size() >= kPQFileHeader + n_ * code_m, "PQ code file is truncated");
    const uint64_t cap = o.pq_capacity ? o.pq_capacity : n_;
    FUSION_CHECK(cap <= n_, "PQ capacity %llu exceeds the index's %llu vectors",
                 static_cast<unsigned long long>(cap), static_cast<unsigned long long>(n_));
    store_ = std::make_unique<PQStore>(n_, cb.m(), cap);
    reclaimer_ = std::make_unique<Reclaimer>(store_.get());

    FilterConfig fc;
    fc.num_workers = o.num_workers;
    fc.max_candidates = max_candidates_;
    fc.max_topn = o.max_rerank;
    fc.gpu_device = o.gpu_device;
    fc.slot_addressed = true;
    if (o.device == FilterDevice::kGpu) {
      filter_ = CreateGpuFilter(cb, nullptr, cap, fc);
    } else {
      filter_ = CreateCpuFilter(cb, store_->codes(), cap, fc);
    }
    FilterBackend* fb = filter_.get();
    const uint8_t* host_codes = store_->codes();
    store_->SetMirror([fb, host_codes](const uint32_t* slots, uint32_t n) {
      fb->StoreCodes(slots, n, host_codes);
    });

    // Raw vectors: node-level too, a sparse copy of the page file filled as partitions arrive
    // (fusion/raw_store.h). The index's page file is only read to bootstrap.
    raw_ = std::make_unique<RawStore>(o.raw_file, man_.raw, o.direct_io);
    if (o.read_iops) {
      budget_ = std::make_unique<ReadBudget>(o.read_iops);
      raw_->SetReadBudget(budget_.get());
    }
    for (int i = 0; i < o.num_workers; ++i) {
      auto w = std::make_unique<Worker>();
      w->reader = MakePageReader(raw_->pages(), o.io, o.io_depth);
      if (budget_) w->reader = ChargedPageReader(std::move(w->reader), budget_.get());
      w->scratch = std::make_unique<RerankScratch>(o.max_rerank, man_.raw.page_size);
      w->qf.resize(dim_);
      w->locs.resize(o.max_rerank);
      workers_.push_back(std::move(w));
    }
    writer_ = std::thread([this] { WriterLoop(); });
  }

  ~NodeEngineImpl() override {
    {
      std::lock_guard<std::mutex> l(cache_mu_);
      stop_writer_ = true;
    }
    writer_cv_.notify_all();
    if (writer_.joinable()) writer_.join();  // installs what is still queued first
  }

  // ------------------------------------------------------------------ staging
  void LoadGraph(const std::string& path) override {
    auto g = std::make_shared<HnswIndex>(space_.get(), path);
    FUSION_CHECK(g->getCurrentElementCount() == man_.num_lists,
                 "graph has %zu heads but the partition manifest has %u lists",
                 static_cast<size_t>(g->getCurrentElementCount()), man_.num_lists);
    g->setEf(1);  // the search width is passed per call (see NavigateImpl)
    std::lock_guard<std::mutex> l(graph_mu_);
    graph_ = std::move(g);
  }

  bool has_graph() const override {
    std::lock_guard<std::mutex> l(graph_mu_);
    return graph_ != nullptr;
  }

  void LoadPartition(uint32_t p, const std::string& seg_path, PQSource pq,
                     const std::string& raw_peer) override {
    FUSION_CHECK(p < man_.num_partitions, "partition %u out of range (%u partitions)", p,
                 man_.num_partitions);
    auto rs = std::make_unique<ResidentSegment>();
    rs->seg = LoadSegment(seg_path);
    FUSION_CHECK(rs->seg.partition == p, "%s holds partition %u, not %u", seg_path.c_str(),
                 rs->seg.partition, p);
    const std::vector<uint32_t>& ids = rs->seg.ids;
    const std::vector<uint32_t>& locs = rs->seg.locs;
    std::vector<uint32_t> raw_miss;
    if (pq == PQSource::kIndex) {
      FUSION_CHECK(raw_peer.empty(), "a bootstrap load takes its raw vectors from the index, not %s",
                   raw_peer.c_str());
      FillFromIndex(ids);
      FillRawFromIndex(locs);
    } else {
      raw_miss = raw_->Missing(locs.data(), locs.size());
      if (!raw_miss.empty() && raw_peer.empty())
        throw RawMissingError(StrFormat("partition %u: %zu raw vectors are not on this node and no "
                                        "source was given", p, raw_miss.size()));
    }
    rs->slots.resize(ids.size());
    // PQMissingError: nothing changed. The locations are recorded with the codes.
    store_->Ref(ids.data(), ids.size(), rs->slots.data(), locs.data());
    SetListSources(rs->seg.list_ids, raw_miss.empty() ? std::string() : raw_peer);
    // The references are dropped (by the reclaimer) once the last holder -- the table below,
    // or a query that pinned the segment -- lets go of it.
    Reclaimer* reclaimer = reclaimer_.get();
    std::shared_ptr<const ResidentSegment> seg(
        rs.release(), [reclaimer](const ResidentSegment* r) { reclaimer->Push(r); });
    std::shared_ptr<const ResidentSegment> old;
    {
      std::lock_guard<std::mutex> l(seg_mu_);
      old = std::move(segs_[p]);
      segs_[p] = std::move(seg);
    }
    resident_version_.fetch_add(1);
  }

  bool EvictPartition(uint32_t p) override {
    FUSION_CHECK(p < man_.num_partitions, "partition %u out of range", p);
    std::shared_ptr<const ResidentSegment> old;
    {
      std::lock_guard<std::mutex> l(seg_mu_);
      old.swap(segs_[p]);
    }
    const bool was = old != nullptr;
    // Unless a request still holds the partition, its references are dropped (its codes are
    // cached) before we return. The source of its lists stays: a peer that got the partition
    // from here may still ask for its vectors.
    old.reset();
    reclaimer_->Flush();
    resident_version_.fetch_add(1);
    return was;
  }

  std::vector<uint32_t> ResidentPartitions() const override {
    std::lock_guard<std::mutex> l(seg_mu_);
    std::vector<uint32_t> r;
    for (uint32_t p = 0; p < segs_.size(); ++p)
      if (segs_[p]) r.push_back(p);
    return r;
  }

  // ---------------------------------------------------------------- PQ codes
  std::vector<std::vector<uint32_t>> PQMissing(
      const std::vector<std::vector<std::string>>& groups) override {
    std::vector<std::vector<uint32_t>> out(groups.size());
    std::vector<uint32_t> taken;  // sorted: listed for an earlier group
    for (size_t g = 0; g < groups.size(); ++g) {
      std::vector<uint32_t> ids;
      for (const std::string& path : groups[g]) {
        const ListSegment s = LoadSegment(path);
        ids.insert(ids.end(), s.ids.begin(), s.ids.end());
      }
      const std::vector<uint32_t> miss = store_->Reserve(ids.data(), ids.size());
      std::vector<uint32_t> fresh;
      std::set_difference(miss.begin(), miss.end(), taken.begin(), taken.end(),
                          std::back_inserter(fresh));
      std::vector<uint32_t> merged;
      merged.reserve(taken.size() + fresh.size());
      std::merge(taken.begin(), taken.end(), fresh.begin(), fresh.end(), std::back_inserter(merged));
      taken.swap(merged);
      out[g] = std::move(fresh);
    }
    return out;
  }

  void PQGet(const uint32_t* ids, size_t n, uint8_t* out) override { store_->Get(ids, n, out); }

  PQPutResult PQPut(const uint32_t* ids, size_t n, const uint8_t* codes) override {
    PQPutResult r;
    r.installed = store_->Put(ids, n, codes, PQOrigin::kReceived);
    r.skipped = n - r.installed;
    return r;
  }

  size_t PQReleaseStaged() override {
    reclaimer_->Flush();  // settle pending reference drops first
    return store_->ReleaseStaged();
  }
  PQStats pq_stats() const override { return store_->Stats(); }

  // ------------------------------------------------------------- raw vectors
  std::vector<RawRefs> RawMissing(const std::vector<std::vector<std::string>>& groups) override {
    std::vector<RawRefs> out(groups.size());
    std::vector<uint32_t> taken;  // sorted: listed for an earlier group
    for (size_t g = 0; g < groups.size(); ++g) {
      std::vector<std::pair<uint32_t, uint32_t>> refs;  // (location, a list naming it)
      for (const std::string& path : groups[g]) {
        const ListSegment seg = LoadSegment(path);
        for (size_t i = 0; i < seg.list_ids.size(); ++i)
          for (uint64_t j = seg.offsets[i]; j < seg.offsets[i + 1]; ++j)
            if (!raw_->Present(seg.locs[j])) refs.emplace_back(seg.locs[j], seg.list_ids[i]);
      }
      std::sort(refs.begin(), refs.end());
      {
        // What queries fetched is not listed again: it is in memory, on its way to the SSD.
        std::lock_guard<std::mutex> l(cache_mu_);
        refs.erase(std::remove_if(refs.begin(), refs.end(),
                                  [&](const std::pair<uint32_t, uint32_t>& r) { return cache_.count(r.first) != 0; }),
                   refs.end());
      }
      RawRefs& o = out[g];
      for (size_t i = 0; i < refs.size(); ++i) {
        if (i > 0 && refs[i].first == refs[i - 1].first) continue;  // one list per location
        if (std::binary_search(taken.begin(), taken.end(), refs[i].first)) continue;
        o.locs.push_back(refs[i].first);
        o.lists.push_back(refs[i].second);
      }
      std::vector<uint32_t> merged;
      merged.reserve(taken.size() + o.locs.size());
      std::merge(taken.begin(), taken.end(), o.locs.begin(), o.locs.end(), std::back_inserter(merged));
      taken.swap(merged);
    }
    return out;
  }

  std::vector<uint32_t> RawAbsent(const uint32_t* locs, size_t n) const override {
    std::vector<uint32_t> miss = raw_->Missing(locs, n);
    std::lock_guard<std::mutex> l(cache_mu_);
    miss.erase(std::remove_if(miss.begin(), miss.end(), [&](uint32_t loc) { return cache_.count(loc) != 0; }),
               miss.end());
    return miss;
  }

  void RawGet(const uint32_t* locs, const uint32_t* lists, size_t n, uint8_t* out) override {
    EnsureRawVia(locs, lists, n);  // a vector this node lacks itself comes through it (chains)
    const uint32_t vb = raw_->layout().vec_bytes;
    std::vector<uint32_t> disk;
    std::vector<size_t> at;
    size_t from_cache = 0;
    {
      std::lock_guard<std::mutex> l(cache_mu_);
      for (size_t i = 0; i < n; ++i) {
        if (raw_->Present(locs[i])) {
          disk.push_back(locs[i]);
          at.push_back(i);
          continue;
        }
        const auto it = cache_.find(locs[i]);
        if (it == cache_.end())
          throw RawMissingError(StrFormat("raw vector at location %u is not on this node", locs[i]));
        std::memcpy(out + i * vb, it->second.data(), vb);
        ++from_cache;
      }
    }
    if (from_cache) raw_->CountServed(from_cache);
    if (disk.size() == n) {
      raw_->Get(locs, n, out);
      return;
    }
    std::vector<uint8_t> buf(disk.size() * vb);
    if (!disk.empty()) raw_->Get(disk.data(), disk.size(), buf.data());
    for (size_t j = 0; j < disk.size(); ++j) std::memcpy(out + at[j] * vb, buf.data() + j * vb, vb);
  }

  RawPutResult RawPut(const uint32_t* locs, size_t n, const uint8_t* vecs) override {
    RawPutResult r;
    r.installed = raw_->Put(locs, n, vecs, RawOrigin::kStreamed);
    r.skipped = n - r.installed;
    return r;
  }

  // Without settle, a snapshot that waits for nothing: INFO is polled for metrics while the
  // writer may be busy. With settle, first waits for the writer to install what was fetched
  // before the call, so that (with no query fetching) every fetched vector counts as present.
  RawStats raw_stats(bool settle) const override {
    if (settle) DrainWrites();
    RawStats st = raw_->Stats();
    CountCachedAndPending(&st);
    return st;
  }

  FetchStats fetch_stats() const override {
    FetchStats f;
    {
      std::lock_guard<std::mutex> l(cache_mu_);
      f.cached = cache_.size();
      f.queued = queued_;
      f.queued_peak = queued_peak_;
      f.write_batches = written_batches_;
    }
    f.present = raw_->Stats().present;
    auto get = [](const std::atomic<uint64_t>& a) { return a.load(std::memory_order_relaxed); };
    f.fetched = get(fc_.fetched);
    f.fetch_calls = get(fc_.fetch_calls);
    f.fetch_us = get(fc_.fetch_us);
    f.written = get(fc_.written);
    f.write_us = get(fc_.write_us);
    f.sync_installed = get(fc_.sync_installed);
    f.sync_us = get(fc_.sync_us);
    f.drain_calls = get(fc_.drain_calls);
    f.drain_waiting = get(fc_.drain_waiting);
    f.drain_us = get(fc_.drain_us);
    f.rerank_disk = get(fc_.rerank_disk);
    f.rerank_mem = get(fc_.rerank_mem);
    f.pending_recounts = get(fc_.pending_recounts);
    f.pending_us = get(fc_.pending_us);
    return f;
  }

  ReadBudgetStats read_budget_stats() const override {
    return budget_ ? budget_->Stats() : ReadBudgetStats{};
  }

  void SetRawFetcher(RawFetcher fetcher) override {
    std::lock_guard<std::mutex> l(raw_mu_);
    fetcher_ = std::move(fetcher);
  }

  // --------------------------------------------------------------- primitives
  uint32_t Navigate(const void* query, uint32_t nprobe, uint32_t ef, uint32_t* lists) override {
    return NavigateImpl(static_cast<const T*>(query), nprobe, ef, lists);
  }

  uint32_t Filter(int wi, const void* query, const uint32_t* lists, uint32_t nlists, uint32_t topn,
                  uint32_t* ids, float* dists, FilterStats* st) override {
    FUSION_CHECK(topn >= 1 && topn <= opts_.max_rerank, "topn %u outside [1, %u]", topn,
                 opts_.max_rerank);
    Worker& w = *workers_.at(static_cast<size_t>(wi));
    FilterWorker* fw = filter_->worker(wi);
    ToFloat(static_cast<const T*>(query), w.qf.data(), dim_);
    fw->BeginQuery(w.qf.data());
    PinScope pins{w};
    const uint32_t nc = Gather(w, lists, nlists, fw->candidate_buffer(), fw->slot_buffer());
    *st = FilterStats();
    st->lists = nlists;
    st->gathered = nc;
    const uint32_t nt = fw->Filter(nc, topn, &st->unique);
    std::memcpy(ids, fw->result_ids(), nt * sizeof(uint32_t));
    std::memcpy(dists, fw->result_dists(), nt * sizeof(float));
    return nt;
  }

  uint32_t Rerank(int wi, const void* query, const uint32_t* ids, uint32_t n, uint32_t k,
                  const uint32_t* lists, uint32_t nlists, uint32_t* out_ids, float* out_dists,
                  RerankStats* st) override {
    FUSION_CHECK(n <= opts_.max_rerank, "rerank input %u exceeds max_rerank %u", n,
                 opts_.max_rerank);
    FUSION_CHECK(nlists <= opts_.max_nprobe, "%u lists exceed max_nprobe %u", nlists,
                 opts_.max_nprobe);
    FUSION_CHECK(k >= 1, "k must be >= 1");
    for (uint32_t i = 0; i < n; ++i)
      FUSION_CHECK(ids[i] < n_, "vector ID %u out of range", ids[i]);
    Worker& w = *workers_.at(static_cast<size_t>(wi));
    RerankParams rp;
    rp.k = k;
    rp.heuristic = false;  // fixed n: every given candidate is re-ranked
    rp.batch = std::max<uint32_t>(std::min<uint32_t>(n, opts_.io_depth), 1);
    *st = RerankStats();
    if (n == 0) return 0;
    // A candidate comes from a FILTER of this node, so a partition loaded here names it and the
    // PQ store knows its location.
    uint32_t* locs = w.locs.data();
    if (store_->Locate(ids, n, locs) != 0) {
      uint32_t i = 0;
      while (locs[i] != PQStore::kNoLoc) ++i;
      throw NotResidentError(
          StrFormat("vector %u is not named by any partition loaded on this node", ids[i]));
    }
    Timer t;
    st->fetched = static_cast<uint32_t>(EnsureRawInLists(locs, n, lists, nlists));
    st->fetch_us = t.Us();
    // Candidates whose vector is on the SSD are re-ranked from it; the others were fetched and
    // are re-ranked from memory, before (or while) the writer installs them. Fixed n and the
    // (distance, ID) order make the merged top-k the one a single pass would keep.
    const T* q = static_cast<const T*>(query);
    const uint32_t vb = raw_->layout().vec_bytes;
    w.disk_ids.clear();
    w.disk_locs.clear();
    w.mem_ids.clear();
    w.mem_vecs.clear();
    {
      std::lock_guard<std::mutex> l(cache_mu_);
      for (uint32_t i = 0; i < n; ++i) {
        if (raw_->Present(locs[i])) {
          w.disk_ids.push_back(ids[i]);
          w.disk_locs.push_back(locs[i]);
          continue;
        }
        const auto it = cache_.find(locs[i]);
        if (it == cache_.end())
          throw RawMissingError(StrFormat("raw vector %u (location %u) is neither here nor fetched", ids[i], locs[i]));
        w.mem_ids.push_back(ids[i]);
        w.mem_vecs.insert(w.mem_vecs.end(), it->second.begin(), it->second.end());
      }
    }
    const uint32_t nd = static_cast<uint32_t>(w.disk_ids.size());
    fc_.rerank_disk.fetch_add(nd, std::memory_order_relaxed);
    fc_.rerank_mem.fetch_add(w.mem_ids.size(), std::memory_order_relaxed);
    const uint32_t cnt = nd == 0 ? 0 : HeuristicRerank<T>(q, dim_, w.disk_ids.data(), w.disk_locs.data(), nd,
                                                          raw_->layout(), w.reader.get(), w.scratch.get(), rp,
                                                          out_ids, out_dists, st);
    if (w.mem_ids.empty()) return cnt;
    w.merged.clear();
    for (uint32_t i = 0; i < cnt; ++i) w.merged.emplace_back(out_dists[i], out_ids[i]);
    for (size_t i = 0; i < w.mem_ids.size(); ++i)
      w.merged.emplace_back(L2Sqr(q, reinterpret_cast<const T*>(w.mem_vecs.data() + i * vb), dim_), w.mem_ids[i]);
    st->reranked += static_cast<uint32_t>(w.mem_ids.size());
    const size_t keep = std::min<size_t>(k, w.merged.size());
    std::partial_sort(w.merged.begin(), w.merged.begin() + keep, w.merged.end());
    for (size_t i = 0; i < keep; ++i) {
      out_dists[i] = w.merged[i].first;
      out_ids[i] = w.merged[i].second;
    }
    return static_cast<uint32_t>(keep);
  }

  uint32_t SearchLocal(int wi, const void* query, const SearchParams& p, uint32_t* ids,
                       float* dists, QueryStats* st) override {
    FUSION_CHECK(p.nprobe >= 1 && p.nprobe <= opts_.max_nprobe, "nprobe %u outside [1, %u]",
                 p.nprobe, opts_.max_nprobe);
    FUSION_CHECK(p.rerank >= 1 && p.rerank <= opts_.max_rerank, "rerank %u outside [1, %u]",
                 p.rerank, opts_.max_rerank);
    Timer total;
    Worker& w = *workers_.at(static_cast<size_t>(wi));
    FilterWorker* fw = filter_->worker(wi);
    const T* q = static_cast<const T*>(query);
    ToFloat(q, w.qf.data(), dim_);
    fw->BeginQuery(w.qf.data());

    Timer t;
    w.lists.resize(p.nprobe);
    const uint32_t nl = NavigateImpl(q, p.nprobe, p.graph_ef, w.lists.data());
    st->graph_us = t.Us();
    st->lists = nl;

    uint32_t nt = 0;
    {
      PinScope pins{w};  // the codes being scored cannot be freed meanwhile
      t.Reset();
      const uint32_t nc = Gather(w, w.lists.data(), nl, fw->candidate_buffer(), fw->slot_buffer());
      st->candidates = nc;
      st->gather_us = t.Us();

      t.Reset();
      uint32_t n_unique = 0;
      nt = fw->Filter(nc, p.rerank, &n_unique);
      st->unique = n_unique;
      st->topn = nt;
      st->filter_us = t.Us();
      // Still pinned: the candidates' codes, and with them their locations, are held.
      FUSION_CHECK(store_->Locate(fw->result_ids(), nt, w.locs.data()) == 0,
                   "a filtered candidate has no known raw-vector location");
    }

    t.Reset();
    RerankParams rp = p.rr;
    rp.k = p.k;
    st->rerank = RerankStats();
    st->rerank.fetched = static_cast<uint32_t>(EnsureRawInLists(w.locs.data(), nt, w.lists.data(), nl));
    // Re-ranked from the SSD below (heuristic order): a candidate fetched by this query or an
    // earlier one may still be in memory only.
    for (uint32_t i = 0; i < nt; ++i)
      if (!raw_->Present(w.locs[i])) {
        DrainWrites();
        break;
      }
    const uint32_t cnt = HeuristicRerank<T>(q, dim_, fw->result_ids(), w.locs.data(), nt,
                                            raw_->layout(), w.reader.get(), w.scratch.get(), rp,
                                            ids, dists, &st->rerank);
    st->rerank_us = t.Us();
    st->total_us = total.Us();
    return cnt;
  }

  DType dtype() const override { return DTypeOf<T>(); }
  uint32_t dim() const override { return dim_; }
  uint64_t size() const override { return n_; }
  uint32_t num_lists() const override { return man_.num_lists; }
  uint32_t num_partitions() const override { return man_.num_partitions; }
  int num_workers() const override { return static_cast<int>(workers_.size()); }
  const NodeOptions& options() const override { return opts_; }
  const char* filter_name() const override { return filter_->name(); }
  const std::string& payload() const override { return man_.payload; }
  uint32_t vec_bytes() const override { return man_.raw.vec_bytes; }

  std::string Describe() const override {
    const double mb = 1e6;
    uint64_t seg_bytes = 0;
    size_t resident = 0;
    {
      std::lock_guard<std::mutex> l(seg_mu_);
      for (const auto& s : segs_) {
        if (!s) continue;
        ++resident;
        seg_bytes += s->seg.bytes();
      }
    }
    const PQStats pq = store_->Stats();
    std::string s;
    s += StrFormat("  vectors          : %llu x %u (%s)\n", (unsigned long long)n_, dim_,
                   DTypeName(DTypeOf<T>()));
    s += StrFormat("  partitions       : %zu of %u resident, %.1f MB of posting lists "
                   "(%u lists in total, payload %s)\n",
                   resident, man_.num_partitions, seg_bytes / mb, man_.num_lists,
                   man_.payload.c_str());
    s += StrFormat("  navigation graph : %s\n", has_graph() ? "loaded" : "not loaded");
    s += StrFormat("  %s tier          : PQ codes of %llu vectors (%.1f MB; %llu live, %llu cached), "
                   "one per vector, budget %llu codes (%.1f MB)\n",
                   std::strcmp(filter_->name(), "gpu") == 0 ? "GPU " : "host",
                   (unsigned long long)pq.resident, pq.resident * pq.m / mb,
                   (unsigned long long)pq.live, (unsigned long long)pq.cached,
                   (unsigned long long)pq.capacity, filter_->code_bytes() / mb);
    const RawStats raw = raw_stats(false);
    s += StrFormat("  SSD tier         : raw vectors of %llu of %llu locations (%.1f MB, %llu pending), "
                   "a sparse copy of the '%s' page file in %s (%s I/O)\n",
                   (unsigned long long)raw.present, (unsigned long long)raw.locations,
                   raw.present * raw.vec_bytes / mb, (unsigned long long)raw.pending,
                   man_.layout.c_str(), raw_->path().c_str(),
                   raw_->pages().direct() ? "direct" : "buffered");
    if (budget_)
      s += StrFormat("  read budget      : %llu device reads/s\n", (unsigned long long)opts_.read_iops);
    s += StrFormat("  workers          : %d, max candidates/request %u\n", num_workers(),
                   max_candidates_);
    return s;
  }

 private:
  struct Worker {
    std::unique_ptr<PageReader> reader;
    std::unique_ptr<RerankScratch> scratch;
    std::vector<float> qf;
    std::vector<uint32_t> lists;
    std::vector<uint32_t> locs;  // raw-vector locations of the candidates being re-ranked
    std::vector<std::shared_ptr<const ResidentSegment>> held;  // pinned by Gather
    // Re-ranking split between the SSD and the fetch cache (see Rerank).
    std::vector<uint32_t> disk_ids, disk_locs, mem_ids;
    std::vector<uint8_t> mem_vecs;
    std::vector<std::pair<float, uint32_t>> merged;
  };

  // Unpins the segments a request gathered from when it finishes, normally or not.
  struct PinScope {
    Worker& w;
    ~PinScope() { w.held.clear(); }
  };

  // Loads a segment file and checks that its lists belong to its partition.
  ListSegment LoadSegment(const std::string& path) const {
    ListSegment s = ListSegment::Load(path);
    FUSION_CHECK(s.partition < man_.num_partitions, "%s: partition %u out of range", path.c_str(),
                 s.partition);
    for (uint32_t c : s.list_ids)
      FUSION_CHECK(c < man_.num_lists && man_.list_part[c] == s.partition,
                   "%s: list %u does not belong to partition %u", path.c_str(), c, s.partition);
    return s;
  }

  // Bootstrap: installs the codes of `ids` this node lacks from the index's pq_codes.bin (the
  // ones it has are protected meanwhile, so making room never drops them).
  void FillFromIndex(const std::vector<uint32_t>& ids) {
    const std::vector<uint32_t> miss = store_->Reserve(ids.data(), ids.size());
    if (miss.empty()) return;
    const uint32_t m = store_->m();
    std::vector<uint8_t> buf(miss.size() * m);
    for (size_t i = 0; i < miss.size();) {  // one read per run of consecutive IDs
      size_t j = i + 1;
      while (j < miss.size() && miss[j] == miss[j - 1] + 1) ++j;
      pq_file_.ReadAt(buf.data() + i * m, (j - i) * m,
                      kPQFileHeader + static_cast<uint64_t>(miss[i]) * m);
      i = j;
    }
    store_->Put(miss.data(), miss.size(), buf.data(), PQOrigin::kIndex);
  }

  // Bootstrap: copies the raw vectors at `locs` this node lacks from the index's page file, one
  // page read for all the vectors a page holds.
  void FillRawFromIndex(const std::vector<uint32_t>& locs) {
    const std::vector<uint32_t> miss = raw_->Missing(locs.data(), locs.size());
    if (miss.empty()) return;
    const RawLayout& L = raw_->layout();
    ReadOnlyFile& f = IndexPages();
    constexpr size_t kBatch = size_t{1} << 16;
    std::vector<uint8_t> page(L.page_size), vecs;
    uint32_t cur = kInvalidId;
    for (size_t i = 0; i < miss.size(); i += kBatch) {
      const size_t n = std::min(kBatch, miss.size() - i);
      vecs.resize(n * L.vec_bytes);
      for (size_t j = 0; j < n; ++j) {
        const uint32_t loc = miss[i + j];
        if (L.page(loc) != cur) {
          cur = L.page(loc);
          f.ReadAt(page.data(), L.page_size, static_cast<uint64_t>(cur) * L.page_size);
        }
        std::memcpy(vecs.data() + j * L.vec_bytes,
                    page.data() + static_cast<size_t>(L.slot(loc)) * L.vec_bytes, L.vec_bytes);
      }
      raw_->Put(miss.data() + i, n, vecs.data(), RawOrigin::kIndex);
    }
  }

  // The index's page file, opened at the first bootstrap load: a node that only receives
  // partitions from peers never reads it (and need not have it).
  ReadOnlyFile& IndexPages() {
    std::lock_guard<std::mutex> l(index_pages_mu_);
    if (!index_pages_) {
      auto f = std::make_unique<ReadOnlyFile>();
      f->Open(index_pages_path_);
      const RawLayout& L = raw_->layout();
      FUSION_CHECK(f->Size() >= L.num_pages * L.page_size, "vector page file %s is truncated",
                   index_pages_path_.c_str());
      index_pages_ = std::move(f);
    }
    return *index_pages_;
  }

  // Makes `peer` the raw-vector source of `lists` ("": none, their vectors are all here).
  void SetListSources(const std::vector<uint32_t>& lists, const std::string& peer) {
    std::lock_guard<std::mutex> l(raw_mu_);
    uint16_t src = 0;
    if (!peer.empty()) {
      size_t i = 0;
      while (i < sources_.size() && sources_[i] != peer) ++i;
      if (i == sources_.size()) {
        FUSION_CHECK(sources_.size() < 0xFFFF, "too many raw-vector sources");
        sources_.push_back(peer);
      }
      src = static_cast<uint16_t>(i + 1);
    }
    for (uint32_t c : lists) list_src_[c] = src;
  }

  // Makes the raw vectors at locs[0..n) present, the missing ones fetched through the query's
  // lists: the lists a query probed name all its candidates, so one of them names each missing
  // vector and knows its source. Only the missing ones are scanned for, and only then.
  size_t EnsureRawInLists(const uint32_t* locs, size_t n, const uint32_t* lists, uint32_t nlists) {
    const std::vector<uint32_t> miss = RawAbsent(locs, n);  // sorted, unique; cached is not missing
    if (miss.empty()) return 0;
    std::vector<uint32_t> via(miss.size(), kInvalidId);
    std::vector<std::shared_ptr<const ResidentSegment>> segs(nlists);
    {
      std::lock_guard<std::mutex> l(seg_mu_);
      for (uint32_t i = 0; i < nlists; ++i) {
        FUSION_CHECK(lists[i] < man_.num_lists, "list %u out of range", lists[i]);
        segs[i] = segs_[man_.list_part[lists[i]]];
      }
    }
    size_t left = miss.size();
    for (uint32_t i = 0; i < nlists && left; ++i) {
      if (!segs[i]) continue;
      uint32_t len = 0;
      const uint32_t* v = segs[i]->seg.Find(lists[i], &len);
      if (!v) continue;
      const uint32_t* ls = segs[i]->seg.locs.data() + (v - segs[i]->seg.ids.data());
      for (uint32_t j = 0; j < len && left; ++j) {
        const auto it = std::lower_bound(miss.begin(), miss.end(), ls[j]);
        if (it != miss.end() && *it == ls[j] && via[it - miss.begin()] == kInvalidId) {
          via[it - miss.begin()] = lists[i];
          --left;
        }
      }
    }
    return FetchRaw(miss, via);
  }

  // The same for a peer's request, which names the list of every location (RAW_GET).
  size_t EnsureRawVia(const uint32_t* locs, const uint32_t* lists, size_t n) {
    std::vector<std::pair<uint32_t, uint32_t>> refs;
    {
      std::lock_guard<std::mutex> l(cache_mu_);
      for (size_t i = 0; i < n; ++i)
        if (!raw_->Present(locs[i]) && !cache_.count(locs[i])) refs.emplace_back(locs[i], lists[i]);
    }
    if (refs.empty()) return 0;
    std::sort(refs.begin(), refs.end());
    std::vector<uint32_t> miss, via;
    for (size_t i = 0; i < refs.size(); ++i) {
      if (i > 0 && refs[i].first == refs[i - 1].first) continue;
      FUSION_CHECK(refs[i].second < man_.num_lists, "list %u out of range", refs[i].second);
      miss.push_back(refs[i].first);
      via.push_back(refs[i].second);
    }
    return FetchRaw(miss, via);
  }

  // Fetches miss[i] from the source of list via[i] (kInvalidId: no list named it), one request
  // per source; returns how many were installed. RawMissingError if one has no source or its
  // fetch fails. Two queries that miss the same vector may both fetch it; the second write is
  // a no-op.
  size_t FetchRaw(const std::vector<uint32_t>& miss, const std::vector<uint32_t>& via) {
    std::vector<std::vector<uint32_t>> locs_by, lists_by;
    std::vector<std::string> peers;
    RawFetcher fetch;
    {
      std::lock_guard<std::mutex> l(raw_mu_);
      locs_by.resize(sources_.size());
      lists_by.resize(sources_.size());
      size_t unknown = 0;
      uint32_t first = 0;
      for (size_t i = 0; i < miss.size(); ++i) {
        const uint16_t src = via[i] == kInvalidId ? 0 : list_src_[via[i]];
        if (src) {
          locs_by[src - 1].push_back(miss[i]);
          lists_by[src - 1].push_back(via[i]);
        } else if (!raw_->Present(miss[i]) && !Cached(miss[i]) && unknown++ == 0) {  // it may have arrived
          first = miss[i];
        }
      }
      if (unknown)
        throw RawMissingError(StrFormat("%zu raw vectors are not on this node and no list naming "
                                        "them has a source (first: location %u)", unknown, first));
      peers = sources_;
      fetch = fetcher_;
    }
    size_t fetched = 0;
    std::vector<uint8_t> buf;
    Timer t;
    struct Count {  // also when a fetch throws
      FetchCounters& fc;
      Timer& t;
      ~Count() {
        fc.fetch_calls.fetch_add(1, std::memory_order_relaxed);
        fc.fetch_us.fetch_add(static_cast<uint64_t>(t.Us()), std::memory_order_relaxed);
      }
    } count{fc_, t};
    for (size_t src = 0; src < locs_by.size(); ++src) {
      const std::vector<uint32_t>& ls = locs_by[src];
      if (ls.empty()) continue;
      if (!fetch) throw RawMissingError("raw vectors are missing and this node has no fetcher");
      buf.resize(ls.size() * raw_->layout().vec_bytes);
      try {
        fetch(peers[src], ls.data(), lists_by[src].data(), ls.size(), buf.data());
      } catch (const std::exception& e) {
        throw RawMissingError(StrFormat("fetching %zu raw vectors from %s: %s", ls.size(),
                                        peers[src].c_str(), e.what()));
      }
      raw_->CountFetch();
      fetched += Cache(ls.data(), ls.size(), buf.data());
    }
    return fetched;
  }

  bool Cached(uint32_t loc) const {
    std::lock_guard<std::mutex> l(cache_mu_);
    return cache_.count(loc) != 0;
  }

  // Keeps fetched vectors for the queries and queues them for the SSD; returns how many were
  // new. Past kMaxQueuedBytes of queued vectors the fetching query installs them itself:
  // backpressure, so that a slow disk cannot grow the cache without bound.
  size_t Cache(const uint32_t* locs, size_t n, const uint8_t* vecs) {
    static constexpr size_t kMaxQueuedBytes = 64u << 20;
    const uint32_t vb = raw_->layout().vec_bytes;
    std::vector<uint32_t> fresh;
    bool full = false;
    {
      std::lock_guard<std::mutex> l(cache_mu_);
      full = (queued_ + n) * vb > kMaxQueuedBytes;
      if (!full) {
        for (size_t i = 0; i < n; ++i) {
          if (raw_->Present(locs[i]) || cache_.count(locs[i])) continue;  // fetched twice: once is enough
          cache_.emplace(locs[i], std::vector<uint8_t>(vecs + i * vb, vecs + (i + 1) * vb));
          fresh.push_back(locs[i]);
        }
        if (!fresh.empty()) {
          queued_ += fresh.size();
          queued_peak_ = std::max(queued_peak_, queued_);
          ++queued_batches_;
          write_queue_.push_back(fresh);
        }
      }
    }
    if (full) {
      Timer t;
      const size_t installed = raw_->Put(locs, n, vecs, RawOrigin::kFetched);
      fc_.sync_installed.fetch_add(installed, std::memory_order_relaxed);
      fc_.sync_us.fetch_add(static_cast<uint64_t>(t.Us()), std::memory_order_relaxed);
      fc_.fetched.fetch_add(installed, std::memory_order_relaxed);
      return installed;
    }
    if (!fresh.empty()) writer_cv_.notify_one();
    fc_.fetched.fetch_add(fresh.size(), std::memory_order_relaxed);
    return fresh.size();
  }

  // The writer: installs queued vectors on the SSD, then drops them from the cache. It takes
  // everything queued (up to kWriteRound vectors) at once, so that the vectors of one page that
  // different queries fetched go out in one page write.
  void WriterLoop() {
    static constexpr size_t kWriteRound = size_t{1} << 16;
    const uint32_t vb = raw_->layout().vec_bytes;
    std::vector<uint32_t> locs;
    std::vector<uint8_t> vecs;
    std::unique_lock<std::mutex> l(cache_mu_);
    for (;;) {
      writer_cv_.wait(l, [&] { return stop_writer_ || !write_queue_.empty(); });
      if (write_queue_.empty()) return;  // stopping, and nothing left to install
      locs.clear();
      uint64_t batches = 0;
      while (!write_queue_.empty() &&
             (locs.empty() || locs.size() + write_queue_.front().size() <= kWriteRound)) {
        locs.insert(locs.end(), write_queue_.front().begin(), write_queue_.front().end());
        write_queue_.pop_front();
        ++batches;
      }
      vecs.resize(locs.size() * vb);
      for (size_t i = 0; i < locs.size(); ++i) std::memcpy(vecs.data() + i * vb, cache_.at(locs[i]).data(), vb);
      l.unlock();
      bool ok = true;
      try {
        Timer t;
        const size_t installed = raw_->Put(locs.data(), locs.size(), vecs.data(), RawOrigin::kFetched);
        fc_.written.fetch_add(installed, std::memory_order_relaxed);
        fc_.write_us.fetch_add(static_cast<uint64_t>(t.Us()), std::memory_order_relaxed);
      } catch (const std::exception& e) {
        ok = false;  // they stay in the cache, still served from memory
        Log("raw-vector writer: %s", e.what());
      }
      l.lock();
      if (ok)
        for (uint32_t loc : locs) cache_.erase(loc);
      queued_ -= locs.size();
      written_batches_ += batches;
      drain_cv_.notify_all();
    }
  }

  // Waits until the vectors queued before the call are on the SSD, not for an empty queue:
  // queries may keep fetching meanwhile. Only for callers that need them there (a settled
  // raw_stats, SearchLocal); INFO, which the metrics poll, does not wait.
  void DrainWrites() const {
    Timer t;
    fc_.drain_calls.fetch_add(1, std::memory_order_relaxed);
    fc_.drain_waiting.fetch_add(1, std::memory_order_relaxed);
    {
      std::unique_lock<std::mutex> l(cache_mu_);
      const uint64_t mine = queued_batches_;
      drain_cv_.wait(l, [&] { return written_batches_ >= mine; });
    }
    fc_.drain_waiting.fetch_sub(1, std::memory_order_relaxed);
    fc_.drain_us.fetch_add(static_cast<uint64_t>(t.Us()), std::memory_order_relaxed);
  }

  // cached: fetched vectors not on the SSD yet (one the stream installed first is counted as
  // present only, so that every cached one is later installed as fetched). pending: distinct
  // locations named by resident partitions whose vector is neither here nor fetched -- what the
  // lazy protocol leaves with the old owners until a query needs it. The locations the resident
  // partitions name are collected again only after a load or an eviction (a scan of every
  // resident posting); the count is a word-wise pass over them and the presence bits.
  void CountCachedAndPending(RawStats* st) const {
    std::lock_guard<std::mutex> l(pending_mu_);
    const uint64_t v = resident_version_.load();
    if (v != named_version_) {
      Timer t;
      std::vector<std::shared_ptr<const ResidentSegment>> segs;
      {
        std::lock_guard<std::mutex> sl(seg_mu_);
        for (const auto& seg : segs_)
          if (seg) segs.push_back(seg);
      }
      named_.assign(CeilDiv(raw_->layout().locations(), 64), 0);
      for (const auto& seg : segs)
        for (uint32_t loc : seg->seg.locs) named_[loc >> 6] |= uint64_t{1} << (loc & 63);
      named_version_ = v;
      fc_.pending_recounts.fetch_add(1, std::memory_order_relaxed);
      fc_.pending_us.fetch_add(static_cast<uint64_t>(t.Us()), std::memory_order_relaxed);
    }
    uint64_t pending = raw_->CountAbsent(named_), cached = 0;
    std::lock_guard<std::mutex> cl(cache_mu_);
    for (const auto& kv : cache_) {
      if (raw_->Present(kv.first)) continue;
      ++cached;
      if (named_[kv.first >> 6] >> (kv.first & 63) & 1) --pending;
    }
    st->cached = cached;
    st->pending = pending;
  }

  uint32_t NavigateImpl(const T* q, uint32_t nprobe, uint32_t ef, uint32_t* lists) const {
    FUSION_CHECK(nprobe >= 1 && nprobe <= opts_.max_nprobe, "nprobe %u outside [1, %u]", nprobe,
                 opts_.max_nprobe);
    std::shared_ptr<HnswIndex> g;
    {
      std::lock_guard<std::mutex> l(graph_mu_);
      g = graph_;
    }
    if (!g) throw NoGraphError("navigation graph is not loaded on this node");
    // hnswlib searches with width max(ef_, k) and keeps the k best; ef_ is 1 here, so asking
    // for k' = max(ef, nprobe) and keeping the nprobe best equals setEf(ef) + searchKnn(nprobe)
    // (what fusion::Engine does) without mutating shared state.
    const uint32_t width = std::max(ef ? ef : 2 * nprobe, nprobe);
    auto res = g->searchKnn(q, width);
    while (res.size() > nprobe) res.pop();
    const uint32_t n = static_cast<uint32_t>(res.size());
    for (uint32_t i = n; i-- > 0;) {  // max-heap: farthest first -> fill from the back
      lists[i] = static_cast<uint32_t>(res.top().second);
      res.pop();
    }
    return n;
  }

  // Copies the vector IDs of `lists` and the PQ slots of their codes into out_ids/out_slots.
  // Throws NotResidentError if a list's partition is not resident. The segments stay pinned in
  // w.held (the caller's PinScope releases them) so that their codes cannot be freed while the
  // filter reads them.
  uint32_t Gather(Worker& w, const uint32_t* lists, uint32_t nlists, uint32_t* out_ids,
                  uint32_t* out_slots) {
    FUSION_CHECK(nlists <= opts_.max_nprobe, "%u lists exceed max_nprobe %u", nlists,
                 opts_.max_nprobe);
    w.held.clear();
    w.held.resize(nlists);
    {
      std::lock_guard<std::mutex> l(seg_mu_);
      for (uint32_t i = 0; i < nlists; ++i) {
        const uint32_t c = lists[i];
        if (c >= man_.num_lists) throw std::runtime_error(StrFormat("list %u out of range", c));
        w.held[i] = segs_[man_.list_part[c]];
      }
    }
    uint64_t nc = 0;
    for (uint32_t i = 0; i < nlists; ++i) {
      const uint32_t c = lists[i];
      if (!w.held[i]) {
        throw NotResidentError(
            StrFormat("list %u: partition %u is not resident", c, man_.list_part[c]));
      }
      const ResidentSegment& rs = *w.held[i];
      uint32_t len = 0;
      const uint32_t* v = rs.seg.Find(c, &len);
      FUSION_CHECK(v != nullptr, "list %u missing from its segment", c);
      FUSION_CHECK(nc + len <= max_candidates_, "gathered IDs exceed max candidates %u",
                   max_candidates_);
      const size_t off = static_cast<size_t>(v - rs.seg.ids.data());
      std::memcpy(out_ids + nc, v, len * sizeof(uint32_t));
      std::memcpy(out_slots + nc, rs.slots.data() + off, len * sizeof(uint32_t));
      nc += len;
    }
    return static_cast<uint32_t>(nc);
  }

  // Destruction runs bottom-up: workers (pins) and resident segments queue their segments on
  // the reclaimer, which frees them while the store is still there; the store's mirror uses
  // the filter, which outlives it.
  NodeOptions opts_;
  PartitionManifest man_;
  uint32_t dim_ = 0;
  uint64_t n_ = 0;
  uint32_t max_candidates_ = 0;
  std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
  mutable std::mutex graph_mu_;
  std::shared_ptr<HnswIndex> graph_;
  ReadOnlyFile pq_file_;
  std::unique_ptr<FilterBackend> filter_;
  std::unique_ptr<PQStore> store_;
  std::unique_ptr<Reclaimer> reclaimer_;
  mutable std::mutex seg_mu_;
  std::vector<std::shared_ptr<const ResidentSegment>> segs_;
  std::unique_ptr<ReadBudget> budget_;  // before raw_ and workers_, which charge it
  std::unique_ptr<RawStore> raw_;
  std::string index_pages_path_;
  std::mutex index_pages_mu_;
  std::unique_ptr<ReadOnlyFile> index_pages_;  // bootstrap only (IndexPages)
  // Where to fetch a raw vector this node lacks: the source of any posting list that names it
  // (1 + an index into sources_, 0 for none), set when the list's partition loads from a
  // peer. Lists overlap, so no per-vector table is needed, and a list keeps its source after
  // its partition leaves, for peers that got the partition from here.
  mutable std::mutex raw_mu_;
  std::vector<uint16_t> list_src_;
  std::vector<std::string> sources_;
  std::atomic<uint64_t> resident_version_{1};  // bumped by loads and evictions (PendingCount)
  // Raw vectors fetched on demand that are not on the SSD yet. Queries use them from here and
  // the writer installs them in the background: the query that fetched a vector does not wait
  // for the write, nor for reading it back (a page read right after its vectors were written
  // also pays their write-back). A vector leaves the cache only once it is present, under
  // cache_mu_, so a lookup under the lock finds it in one place or the other.
  mutable std::mutex cache_mu_;
  mutable std::condition_variable writer_cv_;  // queue not empty, or stopping
  mutable std::condition_variable drain_cv_;   // batches written (DrainWrites)
  std::unordered_map<uint32_t, std::vector<uint8_t>> cache_;
  std::deque<std::vector<uint32_t>> write_queue_;  // batches of cached locations to install
  size_t queued_ = 0;                              // locations in write_queue_
  size_t queued_peak_ = 0;
  uint64_t queued_batches_ = 0, written_batches_ = 0;  // tickets: DrainWrites waits for its own
  struct FetchCounters {  // fetch_stats
    std::atomic<uint64_t> fetched{0}, fetch_calls{0}, fetch_us{0}, written{0}, write_us{0},
        sync_installed{0}, sync_us{0}, drain_calls{0}, drain_waiting{0}, drain_us{0},
        rerank_disk{0}, rerank_mem{0}, pending_recounts{0}, pending_us{0};
  };
  mutable FetchCounters fc_;
  bool stop_writer_ = false;
  std::thread writer_;
  mutable std::mutex pending_mu_;
  mutable uint64_t named_version_ = 0;
  mutable std::vector<uint64_t> named_;  // locations the resident partitions name (a bit each)
  RawFetcher fetcher_;
  std::vector<std::unique_ptr<Worker>> workers_;
};

}  // namespace

std::unique_ptr<NodeEngine> NodeEngine::Open(const NodeOptions& opts) {
  IndexMeta meta = IndexMeta::Load(JoinPath(opts.index_dir, files::kMeta));
  PartitionManifest man = PartitionManifest::Load(opts.partitions_dir);
  const DType dt = ParseDType(meta.Get("dtype"));
  return DispatchDType(dt, [&](auto tag) -> std::unique_ptr<NodeEngine> {
    using T = decltype(tag);
    return std::make_unique<NodeEngineImpl<T>>(opts, meta, man);
  });
}

}  // namespace fusion
