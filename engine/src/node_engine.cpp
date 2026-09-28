#include "fusion/node_engine.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstring>
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
    for (int i = 0; i < o.num_workers; ++i) {
      auto w = std::make_unique<Worker>();
      w->reader = MakePageReader(raw_->pages(), o.io, o.io_depth);
      w->scratch = std::make_unique<RerankScratch>(o.max_rerank, man_.raw.page_size);
      w->qf.resize(dim_);
      w->locs.resize(o.max_rerank);
      workers_.push_back(std::move(w));
    }
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
    if (!raw_miss.empty()) NotePending(raw_miss, raw_peer);
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
    // cached) before we return.
    old.reset();
    reclaimer_->Flush();
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
  std::vector<std::vector<uint32_t>> RawMissing(
      const std::vector<std::vector<std::string>>& groups) override {
    std::vector<std::vector<uint32_t>> out(groups.size());
    std::vector<uint32_t> taken;  // sorted: listed for an earlier group
    for (size_t g = 0; g < groups.size(); ++g) {
      std::vector<uint32_t> locs;
      for (const std::string& path : groups[g]) {
        const ListSegment s = LoadSegment(path);
        locs.insert(locs.end(), s.locs.begin(), s.locs.end());
      }
      const std::vector<uint32_t> miss = raw_->Missing(locs.data(), locs.size());
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

  std::vector<uint32_t> RawAbsent(const uint32_t* locs, size_t n) const override {
    return raw_->Missing(locs, n);
  }

  void RawGet(const uint32_t* locs, size_t n, uint8_t* out) override {
    EnsureRaw(locs, n);  // a vector this node is itself still waiting for comes through it
    raw_->Get(locs, n, out);
  }

  RawPutResult RawPut(const uint32_t* locs, size_t n, const uint8_t* vecs) override {
    RawPutResult r;
    r.installed = raw_->Put(locs, n, vecs, RawOrigin::kStreamed);
    r.skipped = n - r.installed;
    ForgetPending(locs, n);
    return r;
  }

  RawStats raw_stats() const override {
    RawStats st = raw_->Stats();
    std::lock_guard<std::mutex> l(raw_mu_);
    st.pending = pending_.size();
    return st;
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
                  uint32_t* out_ids, float* out_dists, RerankStats* st) override {
    FUSION_CHECK(n <= opts_.max_rerank, "rerank input %u exceeds max_rerank %u", n,
                 opts_.max_rerank);
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
    st->fetched = static_cast<uint32_t>(EnsureRaw(locs, n));
    st->fetch_us = t.Us();
    return HeuristicRerank<T>(static_cast<const T*>(query), dim_, ids, locs, n, raw_->layout(),
                              w.reader.get(), w.scratch.get(), rp, out_ids, out_dists, st);
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
    st->rerank.fetched = static_cast<uint32_t>(EnsureRaw(w.locs.data(), nt));
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
    const RawStats raw = raw_stats();
    s += StrFormat("  SSD tier         : raw vectors of %llu of %llu locations (%.1f MB, %llu pending), "
                   "a sparse copy of the '%s' page file in %s (%s I/O)\n",
                   (unsigned long long)raw.present, (unsigned long long)raw.locations,
                   raw.present * raw.vec_bytes / mb, (unsigned long long)raw.pending,
                   man_.layout.c_str(), raw_->path().c_str(),
                   raw_->pages().direct() ? "direct" : "buffered");
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

  // Records where the raw vectors at `locs` (not on this node) are to be fetched from.
  void NotePending(const std::vector<uint32_t>& locs, const std::string& peer) {
    std::lock_guard<std::mutex> l(raw_mu_);
    size_t src = 0;
    while (src < sources_.size() && sources_[src] != peer) ++src;
    if (src == sources_.size()) {
      FUSION_CHECK(sources_.size() < 0xFFFF, "too many raw-vector sources");
      sources_.push_back(peer);
    }
    // Checked under the lock that RawPut and EnsureRaw take to forget arrived vectors, after
    // their bits are set: a vector cannot arrive in between and stay listed.
    for (uint32_t loc : locs)
      if (!raw_->Present(loc)) pending_[loc] = static_cast<uint16_t>(src);  // latest source wins
  }

  void ForgetPending(const uint32_t* locs, size_t n) {
    std::lock_guard<std::mutex> l(raw_mu_);
    if (pending_.empty()) return;
    for (size_t i = 0; i < n; ++i)
      if (raw_->Present(locs[i])) pending_.erase(locs[i]);
  }

  // Makes the raw vectors at locs[0..n) present, fetching the missing ones from their sources
  // (one request per source); returns how many were fetched. RawMissingError if one has no
  // known source or its fetch fails. Two queries that miss the same vector may both fetch it;
  // the second write is a no-op.
  size_t EnsureRaw(const uint32_t* locs, size_t n) {
    const std::vector<uint32_t> miss = raw_->Missing(locs, n);
    if (miss.empty()) return 0;
    std::vector<std::vector<uint32_t>> by_src;
    std::vector<std::string> peers;
    RawFetcher fetch;
    {
      std::lock_guard<std::mutex> l(raw_mu_);
      by_src.resize(sources_.size());
      size_t unknown = 0;
      uint32_t first = 0;
      for (uint32_t loc : miss) {
        const auto it = pending_.find(loc);
        if (it != pending_.end()) {
          by_src[it->second].push_back(loc);
        } else if (!raw_->Present(loc) && unknown++ == 0) {  // it may have arrived meanwhile
          first = loc;
        }
      }
      if (unknown)
        throw RawMissingError(StrFormat("%zu raw vectors are not on this node and no source is "
                                        "known for them (first: location %u)", unknown, first));
      peers = sources_;
      fetch = fetcher_;
    }
    size_t fetched = 0;
    std::vector<uint8_t> buf;
    for (size_t src = 0; src < by_src.size(); ++src) {
      const std::vector<uint32_t>& ls = by_src[src];
      if (ls.empty()) continue;
      if (!fetch) throw RawMissingError("raw vectors are missing and this node has no fetcher");
      buf.resize(ls.size() * raw_->layout().vec_bytes);
      try {
        fetch(peers[src], ls.data(), ls.size(), buf.data());
      } catch (const std::exception& e) {
        throw RawMissingError(StrFormat("fetching %zu raw vectors from %s: %s", ls.size(),
                                        peers[src].c_str(), e.what()));
      }
      raw_->CountFetch();
      fetched += raw_->Put(ls.data(), ls.size(), buf.data(), RawOrigin::kFetched);
      ForgetPending(ls.data(), ls.size());
    }
    return fetched;
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
  std::unique_ptr<RawStore> raw_;
  std::string index_pages_path_;
  std::mutex index_pages_mu_;
  std::unique_ptr<ReadOnlyFile> index_pages_;  // bootstrap only (IndexPages)
  // Raw vectors named by loaded partitions that have not arrived yet: location -> index into
  // sources_ (the data node to fetch it from). Empty in the steady state.
  mutable std::mutex raw_mu_;
  std::unordered_map<uint32_t, uint16_t> pending_;
  std::vector<std::string> sources_;
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
