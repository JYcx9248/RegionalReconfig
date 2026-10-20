#include "fusion/node_engine.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "fusion/assignment.h"
#include "fusion/dataset.h"
#include "fusion/distance.h"
#include "fusion/filter.h"
#include "fusion/graph_patch.h"
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
// A posting's holder in memory: this node holds the vector. A segment written for a hand-off
// records this node's ID there instead.
constexpr uint16_t kSelfHolder = 0xFFFF;

struct ResidentSegment {
  ListSegment seg;                                // seg.holders moves into `holders` at load
  std::vector<uint32_t> slots;                    // parallel to seg.ids
  // Per posting, the node holding its raw vector (kSelfHolder, a peer's ID, or kNoHolder). The
  // key (list, vector) has one holder at a time: a fetch through the posting makes it this node.
  // The only part of a shared segment that changes, hence atomic. A posting is made to say
  // kSelfHolder only once its vector is here (present, or in the fetch cache), and release /
  // acquire carry that: a thread that reads kSelfHolder finds the vector.
  std::unique_ptr<std::atomic<uint16_t>[]> holders;
  uint32_t version = 0;                           // of the partition (see NodeEngine)
  mutable const ResidentSegment* next = nullptr;  // Reclaimer queue

  uint16_t Holder(size_t j) const { return holders[j].load(std::memory_order_acquire); }
  void SetHolder(size_t j, uint16_t h) const { holders[j].store(h, std::memory_order_release); }
  // The posting of `loc` in `list`, or SIZE_MAX.
  size_t Find(uint32_t list, uint32_t loc) const {
    uint32_t len = 0;
    const uint32_t* v = seg.Find(list, &len);
    if (!v) return SIZE_MAX;
    const size_t at = static_cast<size_t>(v - seg.ids.data());
    for (size_t j = at; j < at + len; ++j)
      if (seg.locs[j] == loc) return j;
    return SIZE_MAX;
  }
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
  NodeEngineImpl(const NodeOptions& o, const IndexMeta& meta, PartitionManifest manifest)
      : opts_(o), man_(std::make_shared<const PartitionManifest>(std::move(manifest))) {
    const PartitionManifest& man = *man_;
    auto path = [&](const std::string& f) { return JoinPath(o.index_dir, f); };
    dim_ = static_cast<uint32_t>(meta.GetU64("dim"));
    n_ = meta.GetU64("num_vectors");
    ids_ = std::max<uint64_t>(n_, o.max_vectors);
    assign_ = AssignOf(meta);
    FUSION_CHECK(man.payload == kPayloadListsLocations,
                 "the partition directory %s has payload '%s': its posting lists carry no raw-vector "
                 "locations (rerun rtier_segment)",
                 o.partitions_dir.c_str(), man.payload.c_str());
    FUSION_CHECK(o.layout.empty() || o.layout == man.layout,
                 "the partitions' locations refer to the '%s' layout, not '%s'",
                 man.layout.c_str(), o.layout.c_str());
    const std::string layouts = meta.GetOr("layouts", "");
    FUSION_CHECK(("," + layouts + ",").find("," + man.layout + ",") != std::string::npos,
                 "index has no '%s' layout (available: %s)", man.layout.c_str(), layouts.c_str());
    FUSION_CHECK(man.raw.vec_bytes == dim_ * DTypeSize(DTypeOf<T>()),
                 "the partitions' page geometry has %u-byte vectors, the index %zu-byte ones",
                 man.raw.vec_bytes, dim_ * DTypeSize(DTypeOf<T>()));
    FUSION_CHECK(o.num_workers >= 1, "need at least one worker");
    FUSION_CHECK(o.max_nprobe >= 1 && o.max_rerank >= 1, "max_nprobe and max_rerank must be >= 1");
    geom_ = man.raw;
    layout_name_ = man.layout;
    payload_ = man.payload;
    space_ = MakeL2Space(DTypeOf<T>(), dim_);
    segs_.resize(man.num_partitions);
    // Lists: the build's, then room for the ones major epochs add (U14), each registered as it
    // arrives -- created by StageVersions, or in a segment loaded from a peer.
    PublishLists(MakeLists(man));
    index_pages_path_ = path(files::LayoutVectors(man.layout));
    // Inserts make lists longer (U14): o.max_list_len leaves room for them.
    max_list_len_ = std::max<uint32_t>(std::max(o.max_list_len, man.max_list_len), 1);
    const uint64_t max_cand =
        static_cast<uint64_t>(std::min<uint32_t>(o.max_nprobe, man.num_lists)) * max_list_len_;
    FUSION_CHECK(max_cand < 0xFFFFFFFFull, "max_nprobe too large");
    max_candidates_ = static_cast<uint32_t>(max_cand);

    // PQ codes: node-level store, filled as partitions arrive (fusion/pq_store.h). The index's
    // pq_codes.bin is only read to bootstrap (PQSource::kIndex) and, after a full rebuild, for
    // the codes a generation's partitions need (PQSource::kGenIndex).
    PQCodebook cb = PQCodebook::Load(path(files::kPQCodebook));
    pq_file_ = OpenPQFile(path(files::kPQCodes), n_, cb.m());
    const uint64_t cap = o.pq_capacity ? o.pq_capacity : ids_;
    FUSION_CHECK(cap <= ids_, "PQ capacity %llu exceeds the %llu vector IDs",
                 static_cast<unsigned long long>(cap), static_cast<unsigned long long>(ids_));
    store_ = std::make_unique<PQStore>(ids_, cb.m(), cap);
    codebook_ = std::make_unique<PQCodebook>(cb);
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
    RawLayout raw_layout = man.raw;  // with room for the locations of inserted vectors
    raw_layout.num_pages = std::max<uint64_t>(raw_layout.num_pages, o.max_pages);
    raw_ = std::make_unique<RawStore>(o.raw_file, raw_layout, o.direct_io);
    if (o.read_iops) {
      budget_ = std::make_unique<ReadBudget>(o.read_iops);
      raw_->SetReadBudget(budget_.get());
    }
    for (int i = 0; i < o.num_workers; ++i) {
      auto w = std::make_unique<Worker>();
      w->reader = MakePageReader(raw_->pages(), o.io, o.io_depth);
      if (budget_) w->reader = ChargedPageReader(std::move(w->reader), budget_.get());
      w->scratch = std::make_unique<RerankScratch>(o.max_rerank, man.raw.page_size);
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
    room_cv_.notify_all();
    if (writer_.joinable()) writer_.join();  // installs what is still queued first
  }

  // ------------------------------------------------------------------ staging
  // The build's graph (version 0), with room for the heads major epochs add.
  void LoadGraph(const std::string& path) override {
    const uint32_t num_lists = Man()->num_lists;
    auto g = std::make_shared<HnswIndex>(space_.get(), path, false, Lists()->capacity);
    FUSION_CHECK(g->getCurrentElementCount() == num_lists,
                 "graph has %zu heads but the partition manifest has %u lists",
                 static_cast<size_t>(g->getCurrentElementCount()), num_lists);
    g->setEf(1);  // the search width is passed per call (see NavigateImpl)
    std::lock_guard<std::mutex> l(graph_mu_);
    graph_ = std::move(g);
    graph_version_ = 0;
    patch_.reset();
    graph_busy_ = false;
  }

  bool has_graph() const override {
    std::lock_guard<std::mutex> l(graph_mu_);
    return graph_ != nullptr;
  }

  GraphState graph_state() const override {
    GraphState st;
    std::lock_guard<std::mutex> l(graph_mu_);
    if (!graph_) return st;
    st.loaded = true;
    st.version = graph_version_;
    st.patch = patch_ ? patch_->patch.version : 0;
    st.count = graph_->cur_element_count;
    st.capacity = graph_->max_elements_;
    st.deleted = graph_->num_deleted_;
    return st;
  }

  // ----------------------------------------------------- generations (U14 full rebuild)
  void SwitchGeneration(uint32_t generation, const std::string& index_dir,
                        const std::string& partitions_dir) override {
    // Everything the generation brings is read and checked before anything changes.
    const IndexMeta meta = IndexMeta::Load(JoinPath(index_dir, files::kMeta));
    FUSION_CHECK(meta.GetU64("dim") == dim_ && ParseDType(meta.Get("dtype")) == DTypeOf<T>(),
                 "generation %u: %s holds %s vectors of %llu dimensions, this node %s ones of %u",
                 generation, index_dir.c_str(), meta.Get("dtype").c_str(),
                 static_cast<unsigned long long>(meta.GetU64("dim")), DTypeName(DTypeOf<T>()), dim_);
    const uint64_t n = meta.GetU64("num_vectors");
    FUSION_CHECK(n <= ids_, "generation %u has %llu vectors, this node has room for %llu IDs", generation,
                 static_cast<unsigned long long>(n), static_cast<unsigned long long>(ids_));
    // One codebook for every generation: the PQ codes this node holds stay valid.
    const PQCodebook cb = PQCodebook::Load(JoinPath(index_dir, files::kPQCodebook));
    const size_t cb_floats = static_cast<size_t>(cb.m()) * kPQKsub * cb.dsub();
    FUSION_CHECK(cb.m() == codebook_->m() && cb.dim() == codebook_->dim() &&
                     std::memcmp(cb.centroids(), codebook_->centroids(), cb_floats * sizeof(float)) == 0,
                 "generation %u: %s has another PQ codebook (a full rebuild keeps the codebook)", generation,
                 index_dir.c_str());
    auto man = std::make_shared<const PartitionManifest>(PartitionManifest::Load(partitions_dir));
    FUSION_CHECK(man->payload == payload_ && man->layout == layout_name_ &&
                     man->raw.page_size == geom_.page_size && man->raw.vec_bytes == geom_.vec_bytes &&
                     man->raw.vectors_per_page == geom_.vectors_per_page &&
                     man->raw.num_pages <= raw_->layout().num_pages,
                 "generation %u: %s refers to another page file than this node's (a full rebuild keeps "
                 "the vectors' locations)", generation, partitions_dir.c_str());
    // A list longer than max_list_len_ is refused when its partition loads, as any other.
    FUSION_CHECK(static_cast<uint64_t>(std::min<uint32_t>(opts_.max_nprobe, man->num_lists)) * max_list_len_ <=
                     max_candidates_,
                 "generation %u: %u lists of up to %u postings exceed the filter's candidate buffers", generation,
                 man->num_lists, max_list_len_);
    std::shared_ptr<ReadOnlyFile> pq = OpenPQFile(JoinPath(index_dir, files::kPQCodes), n, cb.m());
    std::unique_ptr<ListMap> lm = MakeLists(*man);
    std::shared_ptr<HnswIndex> old_graph;
    {
      std::lock_guard<std::mutex> sl(seg_mu_);
      for (uint32_t p = 0; p < segs_.size(); ++p)
        FUSION_CHECK(segs_[p].empty(), "generation %u: partition %u is still resident (evict every partition first)",
                     generation, p);
      std::lock_guard<std::mutex> gl(graph_mu_);
      FUSION_CHECK(!graph_busy_, "the graph is being changed");
      old_graph = std::move(graph_);  // searches still running keep their own reference
      patch_.reset();
      graph_version_ = 0;
      segs_.assign(man->num_partitions, {});
      std::lock_guard<std::mutex> l(gen_mu_);
      man_ = man;
      pq_file_ = pq;
      n_ = n;
      assign_ = AssignOf(meta);
      lists_.store(lm.get(), std::memory_order_release);
      list_maps_.push_back(std::move(lm));
      generation_.store(generation);
    }
    {
      std::lock_guard<std::mutex> l(index_pages_mu_);
      index_pages_path_ = JoinPath(index_dir, files::LayoutVectors(layout_name_));
      index_pages_.reset();
    }
    resident_version_.fetch_add(1);
  }

  std::vector<uint64_t> RawBitmap() const override {
    std::lock_guard<std::mutex> l(cache_mu_);  // the writer moves a vector to the SSD under it
    std::vector<uint64_t> bits = raw_->Bitmap();
    for (const auto& kv : cache_) bits[kv.first >> 6] |= uint64_t{1} << (kv.first & 63);
    return bits;
  }

  // ------------------------------------------------------- graph patches (U14)
  void StageGraphPatch(const uint8_t* data, size_t n) override {
    GraphPatch p = GraphPatch::Parse(data, n);
    std::shared_ptr<HnswIndex> g;
    uint32_t v = 0;
    {
      std::lock_guard<std::mutex> l(graph_mu_);
      if (!graph_) throw NoGraphError("navigation graph is not loaded on this node");
      FUSION_CHECK(!graph_busy_, "the graph is being changed");
      g = graph_;
      v = graph_version_;
    }
    FUSION_CHECK(p.base_version == v && p.version == v + 1,
                 "graph patch from version %u to %u, the graph is at version %u", p.base_version, p.version, v);
    auto sp = std::make_shared<const StagedPatch>(StagePatch(*g, std::move(p)));
    std::lock_guard<std::mutex> l(graph_mu_);
    FUSION_CHECK(graph_ == g && graph_version_ == v && !graph_busy_, "the graph changed while a patch was staged");
    patch_ = std::move(sp);
  }

  void MergeGraphPatch(uint32_t version) override {
    std::shared_ptr<HnswIndex> g;
    std::shared_ptr<const StagedPatch> sp;
    {
      std::lock_guard<std::mutex> l(graph_mu_);
      if (!graph_) throw NoGraphError("navigation graph is not loaded on this node");
      if (!patch_ || patch_->patch.version != version) {
        if (graph_version_ == version) return;  // merged already
        throw NotResidentError(StrFormat("graph patch %u is not staged (the graph is at version %u)", version,
                                         graph_version_));
      }
      FUSION_CHECK(!graph_busy_, "the graph is being changed");
      graph_busy_ = true;  // searches of the version before the patch are refused from here on
      g = graph_;
      sp = patch_;
    }
    WaitPlainSearches();
    // Searches through the patch go on: they read the elements written here from the patch. If
    // the merge fails half way the graph stays busy (no search of it alone) until a LoadGraph.
    ApplyPatch(*g, *sp);
    std::lock_guard<std::mutex> l(graph_mu_);
    graph_version_ = version;
    patch_.reset();
    graph_busy_ = false;
  }

  bool DropGraphPatch(uint32_t version) override {
    std::lock_guard<std::mutex> l(graph_mu_);
    if (!patch_ || patch_->patch.version != version) return false;
    patch_.reset();
    return true;
  }

  void SaveGraphPatch(uint32_t version, const std::string& path) const override {
    std::shared_ptr<const StagedPatch> sp;
    {
      std::lock_guard<std::mutex> l(graph_mu_);
      sp = patch_;
    }
    if (!sp || sp->patch.version != version)
      throw NotResidentError(StrFormat("graph patch %u is not staged", version));
    const std::vector<uint8_t> b = sp->patch.Serialize();
    WriteFile(path, b.data(), b.size());
  }

  std::vector<uint8_t> EditGraph(uint32_t version, const uint32_t* labels, const void* vecs, uint32_t nnew,
                                 const uint32_t* deleted, uint32_t ndeleted) override {
    std::shared_ptr<HnswIndex> g;
    uint32_t base = 0;
    {
      std::lock_guard<std::mutex> l(graph_mu_);
      if (!graph_) throw NoGraphError("navigation graph is not loaded on this node");
      FUSION_CHECK(!patch_ && !graph_busy_, "the graph has a staged patch or is being changed");
      FUSION_CHECK(version == graph_version_ + 1, "an edit to version %u of a graph at version %u", version,
                   graph_version_);
      g = graph_;
      base = graph_version_;
    }
    // Refused before anything changes.
    FUSION_CHECK(g->cur_element_count + nnew <= g->max_elements_,
                 "the graph has room for %zu heads, it holds %zu and the edit adds %u (rtier_node --graph-headroom)",
                 g->max_elements_, static_cast<size_t>(g->cur_element_count), nnew);
    std::vector<hnswlib::tableint> dead(ndeleted);
    {
      std::lock_guard<std::mutex> l(g->label_lookup_lock);
      std::unordered_set<uint32_t> seen;
      for (uint32_t i = 0; i < nnew; ++i)
        FUSION_CHECK(g->label_lookup_.count(labels[i]) == 0 && seen.insert(labels[i]).second,
                     "head %u exists already", labels[i]);
      for (uint32_t i = 0; i < ndeleted; ++i) {
        const auto it = g->label_lookup_.find(deleted[i]);
        FUSION_CHECK(it != g->label_lookup_.end(), "head %u is not in the graph", deleted[i]);
        FUSION_CHECK(!g->isMarkedDeleted(it->second) && seen.insert(deleted[i]).second,
                     "head %u is deleted already", deleted[i]);
        dead[i] = it->second;
      }
    }
    {
      std::lock_guard<std::mutex> l(graph_mu_);
      FUSION_CHECK(graph_ == g && !patch_ && !graph_busy_, "the graph changed meanwhile");
      graph_busy_ = true;
    }
    WaitPlainSearches();
    // The elements an insert changes: the new one, and the neighbors it links to at every level
    // (hnswlib rewrites exactly those lists, in mutuallyConnectNewElement). If an insert fails
    // half way the graph stays busy (no search) until a LoadGraph.
    std::vector<uint32_t> changed;
    const T* x = static_cast<const T*>(vecs);
    for (uint32_t i = 0; i < nnew; ++i) {
      g->addPoint(x + static_cast<size_t>(i) * dim_, labels[i]);
      hnswlib::tableint id;
      {
        std::lock_guard<std::mutex> l(g->label_lookup_lock);
        id = g->label_lookup_.at(labels[i]);
      }
      changed.push_back(id);
      for (int level = 0; level <= g->element_levels_[id]; ++level) {
        hnswlib::linklistsizeint* ll = level == 0 ? g->get_linklist0(id) : g->get_linklist(id, level);
        const unsigned short k = g->getListCount(ll);
        for (unsigned short j = 0; j < k; ++j) changed.push_back(ll[1 + j]);
      }
    }
    for (hnswlib::tableint id : dead) {
      g->markDeletedInternal(id);
      changed.push_back(id);
    }
    GraphPatch p = CapturePatch(*g, std::move(changed), base, version);
    {
      std::lock_guard<std::mutex> l(graph_mu_);
      graph_version_ = version;
      graph_busy_ = false;
    }
    return p.Serialize();
  }

  // Seeds: the member farthest from the mean, then the member farthest from it (ties: the first
  // found). Then a 2-means whose assignment splits the members in halves -- ordered by how much
  // closer they are to one center than to the other -- until it stops changing.
  void SplitCodes(const uint8_t* codes, uint32_t n, uint8_t* side, uint32_t heads[2]) const override {
    FUSION_CHECK(n >= 2, "a list of %u postings cannot split", n);
    const uint32_t m = codebook_->m();
    std::vector<float> x(static_cast<size_t>(n) * dim_);
    for (uint32_t i = 0; i < n; ++i)
      codebook_->Decode(codes + static_cast<size_t>(i) * m, x.data() + static_cast<size_t>(i) * dim_);
    auto row = [&](uint32_t i) { return x.data() + static_cast<size_t>(i) * dim_; };
    auto d2 = [&](const float* a, const float* b) { return L2Sqr(a, b, dim_); };
    std::vector<float> c0(dim_, 0.f), c1(dim_, 0.f);
    for (uint32_t i = 0; i < n; ++i)
      for (uint32_t d = 0; d < dim_; ++d) c0[d] += row(i)[d] / static_cast<float>(n);
    auto farthest = [&](const float* from, uint32_t skip) {
      uint32_t best = n;
      float bd = -1.f;
      for (uint32_t i = 0; i < n; ++i) {
        if (i == skip) continue;
        const float d = d2(row(i), from);
        if (d > bd) {
          bd = d;
          best = i;
        }
      }
      return best;
    };
    const uint32_t a = farthest(c0.data(), n);
    const uint32_t b = farthest(row(a), a);
    c0.assign(row(a), row(a) + dim_);
    c1.assign(row(b), row(b) + dim_);
    std::vector<uint8_t> cur(n, 2);
    std::vector<std::pair<float, uint32_t>> order(n);
    const uint32_t half = (n + 1) / 2;
    std::vector<float>* cs[2] = {&c0, &c1};
    for (int it = 0; it < 32; ++it) {
      for (uint32_t i = 0; i < n; ++i) order[i] = {d2(row(i), c0.data()) - d2(row(i), c1.data()), i};
      std::sort(order.begin(), order.end());
      bool same = true;
      for (uint32_t k = 0; k < n; ++k) {
        const uint8_t s = k < half ? 0 : 1;
        same = same && cur[order[k].second] == s;
        cur[order[k].second] = s;
      }
      for (int s = 0; s < 2; ++s) {
        std::fill(cs[s]->begin(), cs[s]->end(), 0.f);
        const float cnt = static_cast<float>(s == 0 ? half : n - half);
        for (uint32_t i = 0; i < n; ++i)
          if (cur[i] == s)
            for (uint32_t d = 0; d < dim_; ++d) (*cs[s])[d] += row(i)[d] / cnt;
      }
      if (same) break;
    }
    for (int s = 0; s < 2; ++s) {
      float bd = 0.f;
      heads[s] = n;
      for (uint32_t i = 0; i < n; ++i) {
        if (cur[i] != s) continue;
        const float d = d2(row(i), cs[s]->data());
        if (heads[s] == n || d < bd) {
          bd = d;
          heads[s] = i;
        }
      }
    }
    std::copy(cur.begin(), cur.end(), side);
  }

  void NearestHeads(const uint8_t* codes, uint32_t n, uint32_t ef, uint32_t* lists) const override {
    const GraphRef r = Graph(kNewestGraph);
    const uint32_t m = codebook_->m();
    std::vector<float> xf(dim_);
    std::vector<T> q(dim_);
    for (uint32_t i = 0; i < n; ++i) {
      codebook_->Decode(codes + static_cast<size_t>(i) * m, xf.data());
      for (uint32_t d = 0; d < dim_; ++d) q[d] = FromFloat(xf[d]);
      auto res = r.Search(q.data(), std::max<uint32_t>(ef, 1));
      FUSION_CHECK(!res.empty(), "the graph has no head left");
      while (res.size() > 1) res.pop();
      lists[i] = static_cast<uint32_t>(res.top().second);
    }
  }

  void LoadPartition(uint32_t p, const std::string& seg_path, PQSource pq, uint32_t version,
                     uint16_t fallback_holder) override {
    const uint32_t num_parts = Man()->num_partitions;
    FUSION_CHECK(p < num_parts, "partition %u out of range (%u partitions)", p, num_parts);
    auto rs = std::make_unique<ResidentSegment>();
    rs->seg = LoadSegment(seg_path);
    FUSION_CHECK(rs->seg.partition == p, "%s holds partition %u, not %u", seg_path.c_str(),
                 rs->seg.partition, p);
    // A version that update batches made can hold lists longer than the build's (U14): refuse
    // it here, before the flip, rather than fail the queries that gather it.
    for (size_t li = 0; li + 1 < rs->seg.offsets.size(); ++li)
      FUSION_CHECK(rs->seg.offsets[li + 1] - rs->seg.offsets[li] <= max_list_len_,
                   "%s: list %u holds %llu postings, more than the %u a request may gather (--max-list-len)",
                   seg_path.c_str(), rs->seg.list_ids[li],
                   static_cast<unsigned long long>(rs->seg.offsets[li + 1] - rs->seg.offsets[li]), max_list_len_);
    const std::vector<uint32_t>& ids = rs->seg.ids;
    const std::vector<uint32_t>& locs = rs->seg.locs;
    rs->holders.reset(new std::atomic<uint16_t>[ids.size()]);
    if (pq == PQSource::kIndex) {
      FillFromIndex(ids);
      FillRawFromIndex(locs);
      for (size_t j = 0; j < ids.size(); ++j) rs->SetHolder(j, kSelfHolder);
    } else {
      // A generation's partitions after a full rebuild (U14): the codes the node lacks come from
      // the generation's pq_codes.bin (one codebook: the same codes), the raw vectors from the
      // holders the postings record, as after a migration.
      if (pq == PQSource::kGenIndex) FillFromIndex(ids);
      // Each key's holder: this node if the vector is here, else the one the posting records.
      // A missing vector with no holder could never be fetched: refused before anything changes.
      const std::vector<uint32_t> absent = RawAbsent(locs.data(), locs.size());
      const uint16_t me = self_id_.load();
      size_t unknown = 0;
      uint32_t first = 0;
      for (size_t j = 0; j < ids.size(); ++j) {
        uint16_t h = rs->seg.holders.empty() ? kNoHolder : rs->seg.holders[j];
        if (h == kNoHolder) h = fallback_holder;
        if (!std::binary_search(absent.begin(), absent.end(), locs[j])) {
          h = kSelfHolder;
        } else if (h == kNoHolder || h == me) {
          if (unknown++ == 0) first = locs[j];
        }
        rs->SetHolder(j, h);
      }
      if (unknown)
        throw RawMissingError(StrFormat("partition %u: %zu postings name raw vectors that are not on "
                                        "this node and record no other holder (first: location %u)",
                                        p, unknown, first));
    }
    std::vector<uint16_t>().swap(rs->seg.holders);
    rs->slots.resize(ids.size());
    // PQMissingError: nothing changed. The locations are recorded with the codes.
    store_->Ref(ids.data(), ids.size(), rs->slots.data(), locs.data());
    // The references are dropped (by the reclaimer) once the last holder -- the table below,
    // or a query that pinned the segment -- lets go of it.
    rs->version = version;
    Reclaimer* reclaimer = reclaimer_.get();
    std::shared_ptr<const ResidentSegment> seg(
        rs.release(), [reclaimer](const ResidentSegment* r) { reclaimer->Push(r); });
    std::shared_ptr<const ResidentSegment> old;
    {
      std::lock_guard<std::mutex> l(seg_mu_);
      FUSION_CHECK(p < segs_.size(), "partition %u out of range: the generation changed", p);
      auto& vs = segs_[p];  // ascending versions
      auto it = std::find_if(vs.begin(), vs.end(),
                             [&](const auto& r) { return r->version >= version; });
      if (it != vs.end() && (*it)->version == version) {
        old = std::move(*it);
        *it = std::move(seg);
      } else {
        vs.insert(it, std::move(seg));
      }
    }
    resident_version_.fetch_add(1);
  }

  bool EvictPartition(uint32_t p) override {
    std::vector<std::shared_ptr<const ResidentSegment>> old;
    {
      std::lock_guard<std::mutex> l(seg_mu_);
      FUSION_CHECK(p < segs_.size(), "partition %u out of range", p);
      old.swap(segs_[p]);
    }
    const bool was = !old.empty();
    // Unless a request still holds the partition, its references are dropped (its codes are
    // cached) before we return. The source of its lists stays: a peer that got the partition
    // from here may still ask for its vectors.
    old.clear();
    reclaimer_->Flush();
    resident_version_.fetch_add(1);
    return was;
  }

  std::vector<uint32_t> ResidentPartitions() const override {
    std::lock_guard<std::mutex> l(seg_mu_);
    std::vector<uint32_t> r;
    for (uint32_t p = 0; p < segs_.size(); ++p)
      if (!segs_[p].empty()) r.push_back(p);
    return r;
  }

  std::vector<std::pair<uint32_t, uint32_t>> ResidentVersions() const override {
    std::lock_guard<std::mutex> l(seg_mu_);
    std::vector<std::pair<uint32_t, uint32_t>> r;
    for (uint32_t p = 0; p < segs_.size(); ++p)
      for (const auto& s : segs_[p]) r.emplace_back(p, s->version);
    return r;
  }

  // ------------------------------------------------------------ update batches (U14)
  std::vector<StageResult> StageVersions(const std::vector<VersionStage>& parts, const VersionEdits& e) override {
    std::unordered_map<uint32_t, size_t> which;  // partition -> index in parts
    for (size_t i = 0; i < parts.size(); ++i) {
      const VersionStage& vs = parts[i];
      FUSION_CHECK(vs.partition < Man()->num_partitions, "partition %u out of range", vs.partition);
      FUSION_CHECK(vs.next != vs.base, "partition %u: the next version must differ from the base", vs.partition);
      FUSION_CHECK(which.emplace(vs.partition, i).second, "partition %u is listed twice", vs.partition);
    }
    auto staged = [&](uint32_t part, const char* what, uint32_t list) {
      const auto it = which.find(part);
      FUSION_CHECK(it != which.end(), "%s list %u of partition %u, which is not staged", what, list, part);
      return it->second;
    };
    // The lists a major epoch creates and removes, by index in parts.
    std::vector<std::vector<uint32_t>> created(parts.size()), removed(parts.size());
    std::unordered_map<uint32_t, size_t> created_in;  // list -> index in parts
    std::unordered_set<uint32_t> gone;               // the removed lists
    for (size_t c = 0; c < e.ncreated; ++c) {
      const uint32_t list = e.created[c], part = e.created_parts[c];
      const ListMap* lm = Lists();
      FUSION_CHECK(list < lm->capacity, "list %u is beyond the room for lists (%u; rtier_node --graph-headroom)",
                   list, lm->capacity);
      const uint32_t known = lm->part[list].load(std::memory_order_acquire);
      FUSION_CHECK(known == kInvalidId || known == part, "list %u is in partition %u, not %u", list, known, part);
      const size_t i = staged(part, "the batch creates", list);
      FUSION_CHECK(created_in.emplace(list, i).second, "list %u is created twice", list);
      created[i].push_back(list);
    }
    for (size_t r = 0; r < e.nremoved; ++r) {
      const uint32_t list = e.removed[r];
      const size_t i = staged(PartOf(list), "the batch removes", list);
      FUSION_CHECK(gone.insert(list).second && created_in.count(list) == 0,
                   "list %u is removed twice, or created and removed", list);
      removed[i].push_back(list);
    }
    const uint64_t space = raw_->layout().locations();
    const uint16_t me = self_id_.load();
    // The postings to add, by partition and list.
    struct Post {
      uint32_t id, loc;
      uint16_t holder;
    };
    std::vector<std::map<uint32_t, std::vector<Post>>> adds(parts.size());
    for (size_t a = 0; a < e.nadds; ++a) {
      const uint32_t list = e.lists[a];
      FUSION_CHECK(e.ids[a] < ids_, "vector ID %u out of range", e.ids[a]);
      FUSION_CHECK(e.locs[a] < space, "location %u out of range", e.locs[a]);
      size_t i;
      const auto ci = created_in.find(list);
      if (ci != created_in.end()) {
        i = ci->second;
      } else {
        i = staged(PartOf(list), "a posting names", list);
        FUSION_CHECK(gone.count(list) == 0, "a posting names list %u, which the batch removes", list);
      }
      // The posting's holder: this node if the raw vector is here (an insert was RAW_ADDed, a
      // posting moves within this node), else the node the posting names (it moves between
      // owners: the vector is fetched from there when a query needs it).
      uint16_t h = e.holders ? e.holders[a] : kNoHolder;
      if (Held(e.locs[a])) {
        h = kSelfHolder;
      } else if (!Fetchable(h) || h == me) {
        throw RawMissingError(StrFormat("vector %u at location %u is not on this node and its posting names no "
                                        "other holder (RAW_ADD it first)", e.ids[a], e.locs[a]));
      }
      adds[i][list].push_back({e.ids[a], e.locs[a], h});
    }
    std::vector<uint32_t> del(e.deleted, e.deleted + e.ndeleted);
    std::sort(del.begin(), del.end());
    del.erase(std::unique(del.begin(), del.end()), del.end());
    auto is_deleted = [&](uint32_t id) { return std::binary_search(del.begin(), del.end(), id); };

    // Build every next version first; install only if all could be built. A built segment
    // holds PQ references from the moment it exists, and drops them (through the reclaimer)
    // if it is discarded.
    std::vector<StageResult> out(parts.size());
    std::vector<std::shared_ptr<const ResidentSegment>> built(parts.size());
    Reclaimer* reclaimer = reclaimer_.get();
    for (size_t i = 0; i < parts.size(); ++i) {
      const VersionStage& vs = parts[i];
      out[i].partition = vs.partition;
      std::shared_ptr<const ResidentSegment> base;
      {
        std::lock_guard<std::mutex> l(seg_mu_);
        base = SegmentLocked(vs.partition, &vs.base);
      }
      if (!base)
        throw NotResidentError(StrFormat("partition %u is not resident at version %u", vs.partition, vs.base));
      const ListSegment& b = base->seg;
      auto in_base = [&](uint32_t list) { return std::binary_search(b.list_ids.begin(), b.list_ids.end(), list); };
      for (uint32_t list : removed[i])
        FUSION_CHECK(in_base(list), "list %u is not in partition %u's segment", list, vs.partition);
      for (uint32_t list : created[i])
        FUSION_CHECK(!in_base(list), "list %u is in partition %u's segment already", list, vs.partition);
      for (const auto& kv : adds[i])
        FUSION_CHECK(in_base(kv.first) || created_in.count(kv.first) != 0,
                     "list %u is not in partition %u's segment", kv.first, vs.partition);
      // The next version's lists: the base's without the removed ones, with the created ones.
      std::vector<uint32_t> next_lists;
      for (uint32_t list : b.list_ids)
        if (gone.count(list) == 0) next_lists.push_back(list);
      next_lists.insert(next_lists.end(), created[i].begin(), created[i].end());
      std::sort(next_lists.begin(), next_lists.end());
      for (uint32_t list : removed[i]) {
        uint32_t len = 0;
        b.Find(list, &len);
        out[i].removed += len;
      }
      auto rs = std::make_unique<ResidentSegment>();
      ListSegment& seg = rs->seg;
      seg.partition = vs.partition;
      seg.list_ids = next_lists;
      seg.offsets.assign(1, 0);
      // A kept posting keeps its holder; an added one is held here, or by the node it names.
      std::vector<Post> post;
      std::vector<uint16_t> holders;
      for (uint32_t list : next_lists) {
        post.clear();
        uint32_t base_len = 0;
        const uint32_t* bv = in_base(list) ? b.Find(list, &base_len) : nullptr;
        if (bv) {
          const size_t at = static_cast<size_t>(bv - b.ids.data());
          for (size_t j = at; j < at + base_len; ++j) {
            if (is_deleted(b.ids[j])) {
              ++out[i].removed;
              continue;
            }
            post.push_back({b.ids[j], b.locs[j], base->Holder(j)});
          }
        }
        const auto it = adds[i].find(list);
        if (it != adds[i].end()) {
          const size_t kept = post.size();
          for (const Post& x : it->second)
            if (!is_deleted(x.id)) post.push_back(x);
          std::stable_sort(post.begin(), post.end(), [](const Post& x, const Post& y) { return x.id < y.id; });
          post.erase(std::unique(post.begin(), post.end(), [](const Post& x, const Post& y) { return x.id == y.id; }),
                     post.end());
          out[i].added += post.size() - std::min(post.size(), kept);
        }
        FUSION_CHECK(post.size() <= max_list_len_,
                     "list %u would hold %zu postings, more than the %u a request may gather (--max-list-len)",
                     list, post.size(), max_list_len_);
        for (const Post& pl : post) {
          seg.ids.push_back(pl.id);
          seg.locs.push_back(pl.loc);
          holders.push_back(pl.holder);
        }
        seg.offsets.push_back(seg.ids.size());
        if (!bv || post.size() != base_len) out[i].lengths.emplace_back(list, static_cast<uint32_t>(post.size()));
      }
      out[i].changed = out[i].added + out[i].removed > 0 || !created[i].empty() || !removed[i].empty();
      if (!out[i].changed) {
        out[i].lengths.clear();
        continue;
      }
      rs->holders.reset(new std::atomic<uint16_t>[holders.size()]);
      for (size_t j = 0; j < holders.size(); ++j) rs->SetHolder(j, holders[j]);
      rs->version = vs.next;
      rs->slots.resize(seg.ids.size());
      // PQMissingError: this segment holds no reference yet; the ones built before drop theirs.
      store_->Ref(seg.ids.data(), seg.ids.size(), rs->slots.data(), seg.locs.data());
      built[i].reset(rs.release(), [reclaimer](const ResidentSegment* r) { reclaimer->Push(r); });
    }
    // Every version could be built: the lists the batch creates belong to their partitions now.
    for (const auto& kv : created_in) RegisterList(kv.first, parts[kv.second].partition);
    std::vector<std::shared_ptr<const ResidentSegment>> replaced;
    {
      std::lock_guard<std::mutex> l(seg_mu_);
      for (size_t i = 0; i < parts.size(); ++i) {
        if (!built[i]) continue;
        auto& vs = segs_[parts[i].partition];
        const uint32_t v = parts[i].next;
        auto it = std::find_if(vs.begin(), vs.end(), [&](const auto& r) { return r->version >= v; });
        if (it != vs.end() && (*it)->version == v) {
          replaced.push_back(std::move(*it));
          *it = std::move(built[i]);
        } else {
          vs.insert(it, std::move(built[i]));
        }
      }
    }
    resident_version_.fetch_add(1);
    return out;
  }

  ListPostings GetList(uint32_t list, uint32_t version) const override {
    const uint32_t p = PartOf(list);
    std::shared_ptr<const ResidentSegment> s;
    {
      std::lock_guard<std::mutex> l(seg_mu_);
      s = SegmentLocked(p, &version);
    }
    if (!s) throw NotResidentError(StrFormat("partition %u is not resident at version %u", p, version));
    uint32_t len = 0;
    const uint32_t* v = s->seg.Find(list, &len);
    if (!v) throw NotResidentError(StrFormat("list %u is not in partition %u at version %u", list, p, version));
    const size_t at = static_cast<size_t>(v - s->seg.ids.data());
    const uint16_t me = self_id_.load();
    ListPostings out;
    for (size_t j = at; j < at + len; ++j) {
      out.ids.push_back(s->seg.ids[j]);
      out.locs.push_back(s->seg.locs[j]);
      const uint16_t h = s->Holder(j);
      FUSION_CHECK(h != kSelfHolder || me != kNoHolder,
                   "list %u: this node holds vectors of it but does not know its own node ID (SetNodes)", list);
      out.holders.push_back(h == kSelfHolder ? me : h);
    }
    return out;
  }

  bool DropVersion(uint32_t p, uint32_t version) override {
    std::shared_ptr<const ResidentSegment> old;
    {
      std::lock_guard<std::mutex> l(seg_mu_);
      FUSION_CHECK(p < segs_.size(), "partition %u out of range", p);
      auto& vs = segs_[p];
      auto it = std::find_if(vs.begin(), vs.end(), [&](const auto& r) { return r->version == version; });
      if (it != vs.end()) {
        old = std::move(*it);
        vs.erase(it);
      }
    }
    const bool was = old != nullptr;
    // Its references go now, unless a query still holds it; the codes of deleted vectors that
    // ForgetVectors marked go with them.
    old.reset();
    reclaimer_->Flush();
    resident_version_.fetch_add(1);
    return was;
  }

  size_t ForgetVectors(const uint32_t* ids, size_t n, bool deleted) override {
    reclaimer_->Flush();
    return store_->Forget(ids, n, deleted);
  }

  void SavePartition(uint32_t p, uint32_t version, const std::string& path) const override {
    std::shared_ptr<const ResidentSegment> seg;
    {
      std::lock_guard<std::mutex> l(seg_mu_);
      FUSION_CHECK(p < segs_.size(), "partition %u out of range", p);
      seg = SegmentLocked(p, &version);
    }
    if (!seg) throw NotResidentError(StrFormat("partition %u is not resident at version %u", p, version));
    // The next owner gets every key's current holder: this node, by its ID, for the vectors it
    // holds; the recorded holder for the others.
    const uint16_t me = self_id_.load();
    ListSegment out = seg->seg;
    out.holders.resize(out.ids.size());
    for (size_t j = 0; j < out.ids.size(); ++j) {
      const uint16_t h = seg->Holder(j);
      FUSION_CHECK(h != kSelfHolder || me != kNoHolder,
                   "partition %u: this node holds vectors of it but does not know its own node ID (SetNodes)", p);
      out.holders[j] = h == kSelfHolder ? me : h;
    }
    out.Save(path);
  }

  void SetNodes(uint16_t self, const std::vector<std::pair<uint16_t, std::string>>& peers) override {
    FUSION_CHECK(self != kSelfHolder, "node ID %u is reserved", self);
    if (self != kNoHolder) {
      uint16_t was = kNoHolder;
      FUSION_CHECK(self_id_.compare_exchange_strong(was, self) || was == self,
                   "this node is node %u already, not %u", was, self);
    }
    std::lock_guard<std::mutex> l(raw_mu_);
    for (const auto& kv : peers) {
      FUSION_CHECK(kv.first != kNoHolder && kv.first != kSelfHolder, "node ID %u is reserved", kv.first);
      book_[kv.first] = kv.second;
    }
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
  std::vector<RawRefs> RawMissing(const std::vector<uint32_t>& partitions) override {
    std::vector<std::shared_ptr<const ResidentSegment>> segs;
    {
      std::lock_guard<std::mutex> l(seg_mu_);
      for (uint32_t p : partitions) {
        FUSION_CHECK(p < segs_.size(), "partition %u out of range", p);
        std::shared_ptr<const ResidentSegment> s = SegmentLocked(p, nullptr);
        if (!s) throw NotResidentError(StrFormat("partition %u is not resident", p));
        segs.push_back(std::move(s));
      }
    }
    struct Ref {
      uint32_t loc, list;
      uint16_t holder;
    };
    std::vector<Ref> refs;  // every posting whose vector is not here (fetched ones are)
    for (const auto& s : segs) {
      const std::vector<uint32_t> absent = RawAbsent(s->seg.locs.data(), s->seg.locs.size());
      if (absent.empty()) continue;
      for (size_t i = 0; i < s->seg.list_ids.size(); ++i)
        for (uint64_t j = s->seg.offsets[i]; j < s->seg.offsets[i + 1]; ++j)
          if (std::binary_search(absent.begin(), absent.end(), s->seg.locs[j]))
            refs.push_back({s->seg.locs[j], s->seg.list_ids[i], s->Holder(j)});
    }
    std::stable_sort(refs.begin(), refs.end(), [](const Ref& a, const Ref& b) { return a.loc < b.loc; });
    // Each vector once, from the first posting that records a holder. A posting that says the
    // vector is here: a query fetched it since RawAbsent looked.
    std::map<uint16_t, RawRefs> by;
    size_t unknown = 0;
    uint32_t first = 0;
    for (size_t i = 0; i < refs.size();) {
      size_t e = i;  // [i, e): the postings of one location
      while (e < refs.size() && refs[e].loc == refs[i].loc) ++e;
      size_t k = e;
      bool here = false;
      for (size_t r = i; r < e; ++r) {
        here = here || refs[r].holder == kSelfHolder;
        if (k == e && Fetchable(refs[r].holder)) k = r;
      }
      if (!here && k < e) {
        RawRefs& g = by[refs[k].holder];
        g.locs.push_back(refs[k].loc);
        g.lists.push_back(refs[k].list);
      } else if (!here && unknown++ == 0) {
        first = refs[i].loc;
      }
      i = e;
    }
    if (unknown)
      throw RawMissingError(StrFormat("%zu raw vectors are not on this node and no posting naming "
                                      "them records a holder (first: location %u)", unknown, first));
    std::vector<RawRefs> out;
    for (auto& kv : by) {
      kv.second.holder = kv.first;
      out.push_back(std::move(kv.second));
    }
    return out;
  }

  HeldBy RawHeldBy(uint16_t holder) override {
    FUSION_CHECK(Fetchable(holder), "node ID %u is reserved", holder);
    std::vector<std::shared_ptr<const ResidentSegment>> segs;
    {
      std::lock_guard<std::mutex> l(seg_mu_);
      for (const auto& vs : segs_) segs.insert(segs.end(), vs.begin(), vs.end());
    }
    HeldBy out;
    out.missing.holder = holder;
    std::vector<std::pair<uint32_t, uint32_t>> missing;  // (location, list)
    for (const auto& s : segs) {
      std::vector<size_t> named;
      for (size_t j = 0; j < s->seg.ids.size(); ++j)
        if (s->Holder(j) == holder) named.push_back(j);
      if (named.empty()) continue;
      std::vector<uint32_t> locs(named.size());
      for (size_t k = 0; k < named.size(); ++k) locs[k] = s->seg.locs[named[k]];
      const std::vector<uint32_t> absent = RawAbsent(locs.data(), locs.size());
      for (size_t j : named) {
        if (!std::binary_search(absent.begin(), absent.end(), s->seg.locs[j])) {
          s->SetHolder(j, kSelfHolder);  // the vector is here: the key's holder is this node
          ++out.repointed;
          continue;
        }
        const size_t li = static_cast<size_t>(
            std::upper_bound(s->seg.offsets.begin(), s->seg.offsets.end(), j) - s->seg.offsets.begin() - 1);
        missing.emplace_back(s->seg.locs[j], s->seg.list_ids[li]);
      }
    }
    std::sort(missing.begin(), missing.end());
    for (size_t i = 0; i < missing.size(); ++i) {
      if (i > 0 && missing[i].first == missing[i - 1].first) continue;
      out.missing.locs.push_back(missing[i].first);
      out.missing.lists.push_back(missing[i].second);
    }
    return out;
  }

  std::vector<uint32_t> RawAbsent(const uint32_t* locs, size_t n) const override {
    std::vector<uint32_t> miss = raw_->Missing(locs, n);
    std::lock_guard<std::mutex> l(cache_mu_);
    // Present is looked at again: a vector the writer installed since the first look has left
    // the cache.
    miss.erase(std::remove_if(miss.begin(), miss.end(), [&](uint32_t loc) { return HeldLocked(loc); }),
               miss.end());
    return miss;
  }

  void RawGet(const uint32_t* locs, size_t n, uint8_t* out) override {
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

  // Inserted vectors (U14) go where fetched ones go -- the cache, served from memory until the
  // writer installs them -- but into their own queue, which the writer takes last.
  RawPutResult RawAdd(const uint32_t* locs, size_t n, const uint8_t* vecs) override {
    const uint32_t vb = raw_->layout().vec_bytes;
    const uint64_t space = raw_->layout().locations();
    for (size_t i = 0; i < n; ++i)
      FUSION_CHECK(locs[i] < space, "location %u is outside the page file's %llu locations", locs[i],
                   static_cast<unsigned long long>(space));
    FUSION_CHECK(n * vb <= opts_.update_buffer_bytes,
                 "%zu inserted vectors do not fit in the update buffer (%llu bytes)", n,
                 static_cast<unsigned long long>(opts_.update_buffer_bytes));
    std::vector<uint32_t> fresh;
    Timer t;
    {
      std::unique_lock<std::mutex> l(cache_mu_);
      room_cv_.wait(l, [&] {
        return stop_writer_ || (added_queued_ + n) * vb <= opts_.update_buffer_bytes;
      });
      for (size_t i = 0; i < n; ++i) {
        if (HeldLocked(locs[i])) continue;
        cache_.emplace(locs[i], std::vector<uint8_t>(vecs + i * vb, vecs + (i + 1) * vb));
        fresh.push_back(locs[i]);
      }
      if (!fresh.empty()) {
        added_queued_ += fresh.size();
        update_queue_.push_back(fresh);
      }
    }
    fc_.added_wait_us.fetch_add(static_cast<uint64_t>(t.Us()), std::memory_order_relaxed);
    fc_.added.fetch_add(fresh.size(), std::memory_order_relaxed);
    if (!fresh.empty()) writer_cv_.notify_one();
    RawPutResult r;
    r.installed = fresh.size();
    r.skipped = n - r.installed;
    return r;
  }

  void PauseUpdateWrites(bool paused) override {
    {
      std::lock_guard<std::mutex> l(cache_mu_);
      update_paused_ = paused;
    }
    writer_cv_.notify_all();
  }

  // The build's rule (AssignVectors in src/assignment.cpp) for one vector at a time. The build
  // searched with setEf(max(ef, candidates)) and searchKnn(x, candidates); the graph here keeps
  // ef 1 (see NavigateImpl), so asking for max(ef, candidates) and keeping the best candidates
  // runs the same search.
  void Assign(const void* vecs, uint32_t n, uint32_t* lists, uint8_t* codes) override {
    const GraphRef g = Graph(kNewestGraph);
    const AssignParams ap = AssignNow();
    FUSION_CHECK(!ap.rng, "the index was built with RNG pruning, which Assign does not reproduce");
    const uint32_t r = ap.replicas;
    const float closure = (1.f + ap.closure_eps) * (1.f + ap.closure_eps);
    const uint32_t width = std::max(ap.ef, ap.candidates);
    const T* x = static_cast<const T*>(vecs);
    std::vector<std::pair<float, uint32_t>> cand;
    for (uint32_t i = 0; i < n; ++i) {
      const T* v = x + static_cast<size_t>(i) * dim_;
      auto res = g.Search(v, width);
      while (res.size() > ap.candidates) res.pop();
      cand.clear();
      while (!res.empty()) {
        cand.emplace_back(res.top().first, static_cast<uint32_t>(res.top().second));
        res.pop();
      }
      std::reverse(cand.begin(), cand.end());  // ascending distance
      uint32_t* out = lists + static_cast<size_t>(i) * r;
      std::fill(out, out + r, kInvalidId);
      if (!cand.empty()) {
        uint32_t cnt = 0;
        const float d1 = cand[0].first;
        out[cnt++] = cand[0].second;
        for (size_t j = 1; j < cand.size() && cnt < r; ++j) {
          if (cand[j].first > closure * d1) break;  // Eq. 2 closure rule (candidates are sorted)
          out[cnt++] = cand[j].second;
        }
      }
      codebook_->Encode(v, codes + static_cast<size_t>(i) * codebook_->m());
    }
  }

  uint32_t assign_replicas() const override { return AssignNow().replicas; }
  uint32_t pq_m() const override { return codebook_->m(); }
  uint64_t id_capacity() const override { return ids_; }

  // Fetched as RERANK fetches (from the postings' holders, the same peer requests), but written
  // straight to the SSD as a stream's batch: nothing waits for these vectors in memory, and a
  // copy of a whole partition must not pass through the fetch cache and its writer.
  RawPutResult RawPull(const uint32_t* locs, const uint32_t* lists, size_t n) override {
    const std::vector<uint32_t> absent = RawAbsent(locs, n);
    std::map<uint16_t, std::vector<Posting>> by;  // the postings to pull through, by holder
    std::map<uint32_t, std::shared_ptr<const ResidentSegment>> segs;  // by partition: the newest
    for (size_t i = 0; i < n; ++i) {
      if (!std::binary_search(absent.begin(), absent.end(), locs[i])) continue;
      const uint32_t p = PartOf(lists[i]);
      auto it = segs.find(p);
      if (it == segs.end()) {
        std::lock_guard<std::mutex> l(seg_mu_);
        it = segs.emplace(p, SegmentLocked(p, nullptr)).first;
      }
      if (!it->second) throw NotResidentError(StrFormat("partition %u is not resident", p));
      const size_t j = it->second->Find(lists[i], locs[i]);
      FUSION_CHECK(j != SIZE_MAX, "list %u does not name location %u here", lists[i], locs[i]);
      const uint16_t h = it->second->Holder(j);
      if (h != kSelfHolder) by[h].push_back({it->second, j});  // here: a query fetched it meanwhile
    }
    RawPutResult r;
    std::vector<uint8_t> buf;
    for (auto& kv : by) {
      std::vector<uint32_t> ls;
      for (const Posting& q : kv.second) ls.push_back(q.seg->seg.locs[q.j]);
      std::sort(ls.begin(), ls.end());
      ls.erase(std::unique(ls.begin(), ls.end()), ls.end());
      FetchFrom(kv.first, ls, &buf);
      r.installed += raw_->Put(ls.data(), ls.size(), buf.data(), RawOrigin::kStreamed);
      for (const Posting& q : kv.second) q.seg->SetHolder(q.j, kSelfHolder);
    }
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
      f.added_queued = added_queued_;
    }
    f.present = raw_->Stats().present;
    auto get = [](const std::atomic<uint64_t>& a) { return a.load(std::memory_order_relaxed); };
    f.fetched = get(fc_.fetched);
    f.fetch_calls = get(fc_.fetch_calls);
    f.coalesced = get(fc_.coalesced);
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
    f.added = get(fc_.added);
    f.added_written = get(fc_.added_written);
    f.added_wait_us = get(fc_.added_wait_us);
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
  uint32_t Navigate(const void* query, uint32_t nprobe, uint32_t ef, uint32_t* lists, uint32_t graph) override {
    return NavigateImpl(static_cast<const T*>(query), nprobe, ef, graph, lists);
  }

  uint32_t Filter(int wi, const void* query, const uint32_t* lists, const uint32_t* versions,
                  uint32_t nlists, uint32_t topn, uint32_t* ids, float* dists,
                  FilterStats* st) override {
    FUSION_CHECK(topn >= 1 && topn <= opts_.max_rerank, "topn %u outside [1, %u]", topn,
                 opts_.max_rerank);
    Worker& w = *workers_.at(static_cast<size_t>(wi));
    FilterWorker* fw = filter_->worker(wi);
    ToFloat(static_cast<const T*>(query), w.qf.data(), dim_);
    fw->BeginQuery(w.qf.data());
    PinScope pins{w};
    const uint32_t nc =
        Gather(w, lists, versions, nlists, fw->candidate_buffer(), fw->slot_buffer());
    *st = FilterStats();
    st->lists = nlists;
    st->gathered = nc;
    const uint32_t nt = fw->Filter(nc, topn, &st->unique);
    std::memcpy(ids, fw->result_ids(), nt * sizeof(uint32_t));
    std::memcpy(dists, fw->result_dists(), nt * sizeof(float));
    return nt;
  }

  uint32_t Rerank(int wi, const void* query, const uint32_t* ids, uint32_t n, uint32_t k,
                  const uint32_t* lists, const uint32_t* versions, uint32_t nlists,
                  uint32_t* out_ids, float* out_dists, RerankStats* st) override {
    FUSION_CHECK(n <= opts_.max_rerank, "rerank input %u exceeds max_rerank %u", n,
                 opts_.max_rerank);
    FUSION_CHECK(nlists <= opts_.max_nprobe, "%u lists exceed max_nprobe %u", nlists,
                 opts_.max_nprobe);
    FUSION_CHECK(k >= 1, "k must be >= 1");
    for (uint32_t i = 0; i < n; ++i)
      FUSION_CHECK(ids[i] < ids_, "vector ID %u out of range", ids[i]);
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
    st->fetched = static_cast<uint32_t>(EnsureRawInLists(locs, n, lists, versions, nlists));
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
    const uint32_t nl = NavigateImpl(q, p.nprobe, p.graph_ef, kNewestGraph, w.lists.data());
    st->graph_us = t.Us();
    st->lists = nl;

    uint32_t nt = 0;
    {
      PinScope pins{w};  // the codes being scored cannot be freed meanwhile
      t.Reset();
      const uint32_t nc =
          Gather(w, w.lists.data(), nullptr, nl, fw->candidate_buffer(), fw->slot_buffer());
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
    st->rerank.fetched =
        static_cast<uint32_t>(EnsureRawInLists(w.locs.data(), nt, w.lists.data(), nullptr, nl));
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
  uint32_t num_lists() const override { return Man()->num_lists; }
  uint32_t num_partitions() const override { return Man()->num_partitions; }
  uint32_t generation() const override { return generation_.load(); }
  int num_workers() const override { return static_cast<int>(workers_.size()); }
  const NodeOptions& options() const override { return opts_; }
  const char* filter_name() const override { return filter_->name(); }
  const std::string& payload() const override { return payload_; }
  uint32_t vec_bytes() const override { return geom_.vec_bytes; }

  std::string Describe() const override {
    const double mb = 1e6;
    uint64_t seg_bytes = 0;
    size_t resident = 0;
    {
      std::lock_guard<std::mutex> l(seg_mu_);
      for (const auto& vs : segs_) {
        if (vs.empty()) continue;
        ++resident;
        for (const auto& s : vs) seg_bytes += s->seg.bytes();
      }
    }
    const PQStats pq = store_->Stats();
    const auto man = Man();
    std::string s;
    s += StrFormat("  vectors          : %llu x %u (%s)\n", (unsigned long long)n_.load(), dim_,
                   DTypeName(DTypeOf<T>()));
    s += StrFormat("  partitions       : %zu of %u resident, %.1f MB of posting lists "
                   "(%u lists in total, payload %s, generation %u)\n",
                   resident, man->num_partitions, seg_bytes / mb, man->num_lists,
                   man->payload.c_str(), generation_.load());
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
                   layout_name_.c_str(), raw_->path().c_str(),
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
    const auto man = Man();
    const ListMap* lm = Lists();
    FUSION_CHECK(s.partition < man->num_partitions, "%s: partition %u out of range", path.c_str(),
                 s.partition);
    for (uint32_t c : s.list_ids)
      FUSION_CHECK(c < lm->capacity && (c >= man->num_lists || man->list_part[c] == s.partition),
                   "%s: list %u does not belong to partition %u", path.c_str(), c, s.partition);
    // A list a major epoch created (U14) is learned from the segment that holds it.
    for (uint32_t c : s.list_ids)
      if (c >= man->num_lists) RegisterList(c, s.partition);
    return s;
  }

  // Bootstrap (and a generation's partitions after a full rebuild): installs the codes of `ids`
  // this node lacks from the index's pq_codes.bin (the ones it has are protected meanwhile, so
  // making room never drops them). The file holds the index's vectors only: an inserted
  // vector's code must be here already.
  void FillFromIndex(const std::vector<uint32_t>& all) {
    std::shared_ptr<ReadOnlyFile> file;
    uint64_t n = 0;  // the vectors the file holds: read with it, a generation switch replaces both
    {
      std::lock_guard<std::mutex> l(gen_mu_);
      file = pq_file_;
      n = n_.load();
    }
    std::vector<uint32_t> ids;
    ids.reserve(all.size());
    for (uint32_t id : all)
      if (id < n) ids.push_back(id);
    const std::vector<uint32_t> miss = store_->Reserve(ids.data(), ids.size());
    if (miss.empty()) return;
    const uint32_t m = store_->m();
    std::vector<uint8_t> buf(miss.size() * m);
    for (size_t i = 0; i < miss.size();) {  // one read per run of consecutive IDs
      size_t j = i + 1;
      while (j < miss.size() && miss[j] == miss[j - 1] + 1) ++j;
      file->ReadAt(buf.data() + i * m, (j - i) * m,
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
      const RawLayout L = Man()->raw;  // the index's pages: the local store may have room for more
      FUSION_CHECK(f->Size() >= L.num_pages * L.page_size, "vector page file %s is truncated",
                   index_pages_path_.c_str());
      index_pages_ = std::move(f);
    }
    return *index_pages_;
  }

  // A posting in a resident segment (index into its arrays).
  struct Posting {
    std::shared_ptr<const ResidentSegment> seg;
    size_t j = SIZE_MAX;
  };

  // A holder a vector can be fetched from: a peer's ID.
  static bool Fetchable(uint16_t holder) { return holder != kNoHolder && holder != kSelfHolder; }

  // Room for heads and lists: the build's, and graph_headroom more (U14 major epochs).
  uint32_t HeadCapacity(uint32_t num_lists) const {
    const double extra = std::ceil(static_cast<double>(num_lists) * std::max(0.0, opts_.graph_headroom));
    const uint64_t cap = static_cast<uint64_t>(num_lists) + static_cast<uint64_t>(extra);
    FUSION_CHECK(cap < kInvalidId, "graph headroom %.2f is too large", opts_.graph_headroom);
    return static_cast<uint32_t>(cap);
  }

  // What one search reads: the graph alone, or the graph through the staged patch. A search of
  // the graph alone is counted while it runs, so that a merge or an edit in place can wait for
  // it; a search through the patch reads the elements those write from the patch.
  struct GraphRef {
    std::shared_ptr<HnswIndex> g;
    std::shared_ptr<const StagedPatch> patch;
    std::atomic<int>* plain = nullptr;
    GraphRef() = default;
    GraphRef(const GraphRef&) = delete;
    GraphRef& operator=(const GraphRef&) = delete;
    GraphRef(GraphRef&& o) noexcept : g(std::move(o.g)), patch(std::move(o.patch)), plain(o.plain) { o.plain = nullptr; }
    ~GraphRef() {
      if (plain) plain->fetch_sub(1, std::memory_order_release);
    }
    std::priority_queue<std::pair<float, hnswlib::labeltype>> Search(const void* q, size_t k) const {
      return patch ? SearchPatched(*g, *patch, q, k) : g->searchKnn(q, k);
    }
  };

  // The graph at `version` (kNewestGraph: the staged patch's version if there is one, else the
  // graph's). NotResidentError if this node does not have it.
  GraphRef Graph(uint32_t version) const {
    GraphRef r;
    std::lock_guard<std::mutex> l(graph_mu_);
    if (!graph_) throw NoGraphError("navigation graph is not loaded on this node");
    if (version == kNewestGraph) version = patch_ ? patch_->patch.version : graph_version_;
    if (patch_ && version == patch_->patch.version) {
      r.g = graph_;
      r.patch = patch_;
      return r;
    }
    if (version == graph_version_ && !graph_busy_) {
      r.g = graph_;
      plain_searches_.fetch_add(1, std::memory_order_acquire);
      r.plain = &plain_searches_;
      return r;
    }
    const std::string staged = patch_ ? StrFormat(", a patch to version %u is staged", patch_->patch.version) : "";
    throw NotResidentError(StrFormat("graph version %u is not here (the graph is at version %u%s)", version,
                                     graph_version_, staged.c_str()));
  }

  // Waits for the searches of the graph alone that started before it became busy.
  void WaitPlainSearches() const {
    while (plain_searches_.load(std::memory_order_acquire) != 0)
      std::this_thread::sleep_for(std::chrono::microseconds(50));
  }

  // A decoded PQ vector as a query of the graph's element type.
  static T FromFloat(float v) {
    if constexpr (std::is_floating_point<T>::value) {
      return v;
    } else {
      const float lo = static_cast<float>(std::numeric_limits<T>::min());
      const float hi = static_cast<float>(std::numeric_limits<T>::max());
      return static_cast<T>(std::min(hi, std::max(lo, std::nearbyint(v))));
    }
  }

  // The partition of `list`; NotResidentError if this node does not know the list.
  uint32_t PartOf(uint32_t list) const {
    const ListMap* lm = Lists();
    if (list >= lm->capacity) throw NotResidentError(StrFormat("list %u out of range (%u)", list, lm->capacity));
    const uint32_t p = lm->part[list].load(std::memory_order_acquire);
    if (p == kInvalidId) throw NotResidentError(StrFormat("list %u is not known on this node", list));
    return p;
  }

  // Records that `list`, created by a major epoch, belongs to partition `part`.
  void RegisterList(uint32_t list, uint32_t part) const {
    const ListMap* lm = Lists();
    FUSION_CHECK(list < lm->capacity, "list %u is beyond the room for lists (%u; rtier_node --graph-headroom)",
                 list, lm->capacity);
    FUSION_CHECK(part < Man()->num_partitions, "partition %u out of range", part);
    uint32_t was = kInvalidId;
    if (!lm->part[list].compare_exchange_strong(was, part, std::memory_order_acq_rel) && was != part)
      throw std::invalid_argument(StrFormat("list %u is in partition %u, not %u", list, was, part));
  }

  // List -> partition (kInvalidId: unknown here), read without locks: the generation's lists,
  // then up to `capacity` the ones major epochs add. A full rebuild publishes another map; the
  // ones it replaced stay until the engine goes, since a request may still be reading one.
  struct ListMap {
    uint32_t capacity = 0;
    std::unique_ptr<std::atomic<uint32_t>[]> part;
  };
  const ListMap* Lists() const { return lists_.load(std::memory_order_acquire); }
  std::unique_ptr<ListMap> MakeLists(const PartitionManifest& man) const {
    auto lm = std::make_unique<ListMap>();
    lm->capacity = HeadCapacity(man.num_lists);
    lm->part.reset(new std::atomic<uint32_t>[lm->capacity]);
    for (uint32_t c = 0; c < lm->capacity; ++c)
      lm->part[c].store(c < man.num_lists ? man.list_part[c] : kInvalidId, std::memory_order_relaxed);
    return lm;
  }
  void PublishLists(std::unique_ptr<ListMap> lm) {
    std::lock_guard<std::mutex> l(gen_mu_);
    lists_.store(lm.get(), std::memory_order_release);
    list_maps_.push_back(std::move(lm));
  }
  std::shared_ptr<const PartitionManifest> Man() const {
    std::lock_guard<std::mutex> l(gen_mu_);
    return man_;
  }
  AssignParams AssignNow() const {
    std::lock_guard<std::mutex> l(gen_mu_);
    return assign_;
  }
  // The build's assignment rule (U14): inserted vectors are routed by it, so its parameters come
  // from the index; the two the build did not record are its defaults (fusion_build --assign-*).
  static AssignParams AssignOf(const IndexMeta& meta) {
    AssignParams a;
    a.replicas = static_cast<uint32_t>(std::stoul(meta.GetOr("replicas", "8")));
    a.closure_eps = std::stof(meta.GetOr("closure_eps", "10"));
    a.rng = meta.GetOr("rng", "0") != "0";
    a.candidates = static_cast<uint32_t>(std::stoul(meta.GetOr("assign_candidates", "64")));
    a.ef = static_cast<uint32_t>(std::stoul(meta.GetOr("assign_ef", "128")));
    return a;
  }
  // An index's pq_codes.bin, checked against its vector count and the codebook.
  static std::shared_ptr<ReadOnlyFile> OpenPQFile(const std::string& path, uint64_t n, uint32_t m) {
    auto f = std::make_shared<ReadOnlyFile>();
    f->Open(path);
    uint64_t hdr[2] = {0, 0};  // u64 n, then u32 m + u32 reserved
    f->ReadAt(hdr, sizeof(hdr), 0);
    const uint32_t code_m = static_cast<uint32_t>(hdr[1] & 0xFFFFFFFFu);
    FUSION_CHECK(hdr[0] == n && code_m == m, "PQ codes do not match the index (%s)", path.c_str());
    FUSION_CHECK(f->Size() >= kPQFileHeader + n * code_m, "PQ code file is truncated (%s)", path.c_str());
    return f;
  }

  // Fetches the vectors at locs (sorted, unique) from the data node `holder` into *buf.
  // RawMissingError if the node does not know where that holder is, or the fetch fails.
  void FetchFrom(uint16_t holder, const std::vector<uint32_t>& locs, std::vector<uint8_t>* buf) {
    std::string addr;
    RawFetcher fetch;
    {
      std::lock_guard<std::mutex> l(raw_mu_);
      const auto it = book_.find(holder);
      if (it != book_.end()) addr = it->second;
      fetch = fetcher_;
    }
    if (!Fetchable(holder))
      throw RawMissingError(StrFormat("%zu raw vectors are not on this node and their postings record "
                                      "no holder (first: location %u)", locs.size(), locs[0]));
    if (addr.empty())
      throw RawMissingError(StrFormat("%zu raw vectors are held by node %u, whose address this node "
                                      "does not know (first: location %u)", locs.size(), holder, locs[0]));
    if (!fetch) throw RawMissingError("raw vectors are missing and this node has no fetcher");
    buf->resize(locs.size() * raw_->layout().vec_bytes);
    try {
      fetch(addr, locs.data(), locs.size(), buf->data());
    } catch (const std::exception& e) {
      throw RawMissingError(StrFormat("fetching %zu raw vectors from node %u (%s): %s", locs.size(),
                                      holder, addr.c_str(), e.what()));
    }
  }

  // Makes the raw vectors at locs[0..n) present, the missing ones fetched through the query's
  // lists: the lists a query probed name all its candidates, so one of them names each missing
  // vector and records its holder. Only the missing ones are scanned for, and only then.
  size_t EnsureRawInLists(const uint32_t* locs, size_t n, const uint32_t* lists,
                          const uint32_t* versions, uint32_t nlists) {
    const std::vector<uint32_t> miss = RawAbsent(locs, n);  // sorted, unique; cached is not missing
    if (miss.empty()) return 0;
    std::vector<Posting> via(miss.size());
    std::vector<std::shared_ptr<const ResidentSegment>> segs(nlists);
    {
      std::lock_guard<std::mutex> l(seg_mu_);
      for (uint32_t i = 0; i < nlists; ++i) {
        segs[i] = SegmentLocked(PartOf(lists[i]), versions ? &versions[i] : nullptr);
      }
    }
    size_t left = miss.size();
    for (uint32_t i = 0; i < nlists && left; ++i) {
      if (!segs[i]) continue;
      uint32_t len = 0;
      const uint32_t* v = segs[i]->seg.Find(lists[i], &len);
      if (!v) continue;
      const size_t at = static_cast<size_t>(v - segs[i]->seg.ids.data());
      for (size_t j = at; j < at + len && left; ++j) {
        const auto it = std::lower_bound(miss.begin(), miss.end(), segs[i]->seg.locs[j]);
        if (it != miss.end() && *it == segs[i]->seg.locs[j] && !via[it - miss.begin()].seg) {
          via[it - miss.begin()] = {segs[i], j};
          --left;
        }
      }
    }
    return FetchForQuery(miss, via);
  }

  // A fetch in flight: queries that miss the same vector meanwhile wait for it instead of
  // fetching it again.
  struct InFlight {
    bool done = false;
    bool ok = false;
  };

  // Fetches miss[i] from the holder that posting via[i] records, one request per holder, and
  // waits for those another query is fetching already; every posting used then names this node.
  // Returns how many vectors this call installed. RawMissingError if one has no holder or a
  // fetch fails (its own, or one it waited for).
  size_t FetchForQuery(const std::vector<uint32_t>& miss, const std::vector<Posting>& via) {
    Timer t;
    struct Count {  // also when a fetch throws
      FetchCounters& fc;
      Timer& t;
      ~Count() {
        fc.fetch_calls.fetch_add(1, std::memory_order_relaxed);
        fc.fetch_us.fetch_add(static_cast<uint64_t>(t.Us()), std::memory_order_relaxed);
      }
    } count{fc_, t};
    std::vector<size_t> mine, here;  // indices into miss: to fetch; arrived meanwhile
    std::vector<std::pair<size_t, std::shared_ptr<InFlight>>> waits;
    std::map<uint32_t, std::shared_ptr<InFlight>> claimed;  // by location
    {
      std::lock_guard<std::mutex> l(cache_mu_);
      for (size_t i = 0; i < miss.size(); ++i) {
        if (HeldLocked(miss[i])) {
          here.push_back(i);
          continue;
        }
        const auto it = inflight_.find(miss[i]);
        if (it != inflight_.end()) {
          waits.emplace_back(i, it->second);
          continue;
        }
        auto f = std::make_shared<InFlight>();
        inflight_.emplace(miss[i], f);
        claimed.emplace(miss[i], f);
        mine.push_back(i);
      }
    }
    if (!waits.empty()) fc_.coalesced.fetch_add(waits.size(), std::memory_order_relaxed);
    // However this call ends, its claims end with it, so that no waiter blocks for good.
    auto settle = [&](const std::vector<uint32_t>& locs, bool ok) {
      std::lock_guard<std::mutex> l(cache_mu_);
      for (uint32_t loc : locs) {
        const auto it = claimed.find(loc);
        if (it == claimed.end()) continue;
        it->second->done = true;
        it->second->ok = ok;
        inflight_.erase(loc);
        claimed.erase(it);
      }
      inflight_cv_.notify_all();
    };
    struct Unclaim {
      std::function<void()> f;
      ~Unclaim() { f(); }
    } unclaim{[&] {
      std::vector<uint32_t> rest;
      for (const auto& kv : claimed) rest.push_back(kv.first);
      if (!rest.empty()) settle(rest, false);
    }};
    std::map<uint16_t, std::vector<size_t>> by;  // what this call fetches, by holder
    // Claimed, but the posting says here: a stream or a drain (RawPull) brought the vector since
    // the claim. It is here, and its claim ends well.
    std::vector<uint32_t> arrived;
    size_t unknown = 0;
    uint32_t first = 0;
    bool listed = false;  // whether a list of the query names the first one
    for (size_t i : mine) {
      const uint16_t h = via[i].seg ? via[i].seg->Holder(via[i].j) : kNoHolder;
      if (Fetchable(h)) {
        by[h].push_back(i);
      } else if (h == kSelfHolder) {
        arrived.push_back(miss[i]);
      } else if (unknown++ == 0) {
        first = miss[i];
        listed = via[i].seg != nullptr;
      }
    }
    if (unknown)
      throw RawMissingError(StrFormat("%zu raw vectors are not on this node and cannot be fetched (first: "
                                      "location %u, %s)", unknown, first,
                                      listed ? "its posting records no holder" : "no list of the query names it"));
    if (!arrived.empty()) settle(arrived, true);
    size_t fetched = 0;
    std::vector<uint8_t> buf;
    for (const auto& kv : by) {
      std::vector<uint32_t> ls;
      for (size_t i : kv.second) ls.push_back(miss[i]);  // ascending, as miss
      FetchFrom(kv.first, ls, &buf);
      raw_->CountFetch();
      fetched += Cache(ls.data(), ls.size(), buf.data());
      for (size_t i : kv.second) via[i].seg->SetHolder(via[i].j, kSelfHolder);
      settle(ls, true);
    }
    if (!waits.empty()) {
      std::unique_lock<std::mutex> l(cache_mu_);
      inflight_cv_.wait(l, [&] {
        for (const auto& w : waits)
          if (!w.second->done) return false;
        return true;
      });
      for (const auto& w : waits)
        if (!w.second->ok)
          throw RawMissingError(StrFormat("raw vector at location %u: the fetch this query waited for "
                                          "failed", miss[w.first]));
    }
    for (const auto& w : waits)
      if (via[w.first].seg) via[w.first].seg->SetHolder(via[w.first].j, kSelfHolder);
    for (size_t i : here)
      if (via[i].seg) via[i].seg->SetHolder(via[i].j, kSelfHolder);
    return fetched;
  }

  // Whether the raw vector at loc is here: on the SSD, or fetched and in the cache. Both under
  // cache_mu_: the writer marks a vector present before it drops it from the cache, so the two
  // looks cannot both miss one on its way from memory to the SSD.
  bool HeldLocked(uint32_t loc) const { return raw_->Present(loc) || cache_.count(loc) != 0; }
  bool Held(uint32_t loc) const {
    std::lock_guard<std::mutex> l(cache_mu_);
    return HeldLocked(loc);
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
          if (HeldLocked(locs[i])) continue;  // fetched twice: once is enough
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
      // Fetched vectors first: the queries that fetched them hold memory until they are written.
      // Inserted ones (U14) only when no fetched one waits, within update_write_bps and not
      // while paused -- except when stopping, which writes everything.
      auto updates_due = [&] {
        return !update_queue_.empty() &&
               (stop_writer_ ||
                (!update_paused_ && std::chrono::steady_clock::now() >= next_update_write_));
      };
      while (!stop_writer_ && write_queue_.empty() && !updates_due()) {
        if (!update_queue_.empty() && !update_paused_)
          writer_cv_.wait_until(l, next_update_write_);
        else
          writer_cv_.wait(l);
      }
      std::deque<std::vector<uint32_t>>* q = nullptr;
      if (!write_queue_.empty())
        q = &write_queue_;
      else if (updates_due())
        q = &update_queue_;
      else
        return;  // stopping, and nothing left to install
      const bool added = q == &update_queue_;
      locs.clear();
      uint64_t batches = 0;
      while (!q->empty() && (locs.empty() || locs.size() + q->front().size() <= kWriteRound)) {
        locs.insert(locs.end(), q->front().begin(), q->front().end());
        q->pop_front();
        ++batches;
      }
      vecs.resize(locs.size() * vb);
      for (size_t i = 0; i < locs.size(); ++i) std::memcpy(vecs.data() + i * vb, cache_.at(locs[i]).data(), vb);
      l.unlock();
      bool ok = true;
      try {
        Timer t;
        const size_t installed =
            raw_->Put(locs.data(), locs.size(), vecs.data(), added ? RawOrigin::kAdded : RawOrigin::kFetched);
        (added ? fc_.added_written : fc_.written).fetch_add(installed, std::memory_order_relaxed);
        fc_.write_us.fetch_add(static_cast<uint64_t>(t.Us()), std::memory_order_relaxed);
      } catch (const std::exception& e) {
        ok = false;  // they stay in the cache, still served from memory
        Log("raw-vector writer: %s", e.what());
      }
      l.lock();
      if (ok)
        for (uint32_t loc : locs) cache_.erase(loc);
      if (added) {
        added_queued_ -= locs.size();
        if (opts_.update_write_bps) {
          const double us = 1e6 * static_cast<double>(locs.size() * vb) / opts_.update_write_bps;
          next_update_write_ = std::chrono::steady_clock::now() +
                               std::chrono::microseconds(static_cast<int64_t>(us));
        }
        room_cv_.notify_all();
      } else {
        queued_ -= locs.size();
        written_batches_ += batches;
        drain_cv_.notify_all();
      }
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
        for (const auto& vs : segs_)
          segs.insert(segs.end(), vs.begin(), vs.end());
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

  uint32_t NavigateImpl(const T* q, uint32_t nprobe, uint32_t ef, uint32_t graph, uint32_t* lists) const {
    FUSION_CHECK(nprobe >= 1 && nprobe <= opts_.max_nprobe, "nprobe %u outside [1, %u]", nprobe,
                 opts_.max_nprobe);
    const GraphRef g = Graph(graph);
    // hnswlib searches with width max(ef_, k) and keeps the k best; ef_ is 1 here, so asking
    // for k' = max(ef, nprobe) and keeping the nprobe best equals setEf(ef) + searchKnn(nprobe)
    // (what fusion::Engine does) without mutating shared state.
    const uint32_t width = std::max(ef ? ef : 2 * nprobe, nprobe);
    auto res = g.Search(q, width);
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
  uint32_t Gather(Worker& w, const uint32_t* lists, const uint32_t* versions, uint32_t nlists,
                  uint32_t* out_ids, uint32_t* out_slots) {
    FUSION_CHECK(nlists <= opts_.max_nprobe, "%u lists exceed max_nprobe %u", nlists,
                 opts_.max_nprobe);
    w.held.clear();
    w.held.resize(nlists);
    {
      std::lock_guard<std::mutex> l(seg_mu_);
      for (uint32_t i = 0; i < nlists; ++i) {
        w.held[i] = SegmentLocked(PartOf(lists[i]), versions ? &versions[i] : nullptr);
      }
    }
    uint64_t nc = 0;
    for (uint32_t i = 0; i < nlists; ++i) {
      const uint32_t c = lists[i];
      if (!w.held[i]) {
        if (versions)
          throw NotResidentError(StrFormat("list %u: partition %u is not resident at version %u", c,
                                           PartOf(c), versions[i]));
        throw NotResidentError(StrFormat("list %u: partition %u is not resident", c, PartOf(c)));
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
  // The generation's partition manifest and what comes with it (a full rebuild replaces them,
  // under gen_mu_, which is taken last whenever it nests): the index's vector count, its PQ code
  // file, the assignment rule. The page geometry and the payload never change.
  mutable std::mutex gen_mu_;
  std::shared_ptr<const PartitionManifest> man_;
  std::atomic<uint32_t> generation_{0};
  RawLayout geom_;
  std::string layout_name_, payload_;
  uint32_t dim_ = 0;
  std::atomic<uint64_t> n_{0};  // the index's vectors
  uint64_t ids_ = 0;            // the vector-ID space: n_ and room for inserted vectors
  uint32_t max_list_len_ = 0;  // longest list a request may gather (room for inserts)
  AssignParams assign_;  // the build's, for Assign
  std::unique_ptr<PQCodebook> codebook_;
  uint32_t max_candidates_ = 0;
  std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
  // The navigation graph, at graph_version_, and the patch to the next version if one is staged
  // (U14 major epochs). graph_busy_: being patched or edited in place -- no search of the graph
  // alone may start, and the ones running are waited for (plain_searches_).
  mutable std::mutex graph_mu_;
  std::shared_ptr<HnswIndex> graph_;
  uint32_t graph_version_ = 0;
  std::shared_ptr<const StagedPatch> patch_;
  bool graph_busy_ = false;
  mutable std::atomic<int> plain_searches_{0};
  std::atomic<const ListMap*> lists_{nullptr};
  std::vector<std::unique_ptr<ListMap>> list_maps_;  // under gen_mu_: every map published
  std::shared_ptr<ReadOnlyFile> pq_file_;            // under gen_mu_
  std::unique_ptr<FilterBackend> filter_;
  std::unique_ptr<PQStore> store_;
  std::unique_ptr<Reclaimer> reclaimer_;
  mutable std::mutex seg_mu_;
  // Per partition, its resident versions, ascending (guarded by seg_mu_).
  std::vector<std::vector<std::shared_ptr<const ResidentSegment>>> segs_;

  // Partition p at *version, or at its newest version if version is null; null if not
  // resident. seg_mu_ must be held.
  std::shared_ptr<const ResidentSegment> SegmentLocked(uint32_t p, const uint32_t* version) const {
    if (p >= segs_.size()) return nullptr;
    const auto& vs = segs_[p];
    if (vs.empty()) return nullptr;
    if (!version) return vs.back();
    for (const auto& s : vs)
      if (s->version == *version) return s;
    return nullptr;
  }
  std::unique_ptr<ReadBudget> budget_;  // before raw_ and workers_, which charge it
  std::unique_ptr<RawStore> raw_;
  std::string index_pages_path_;
  std::mutex index_pages_mu_;
  std::unique_ptr<ReadOnlyFile> index_pages_;  // bootstrap only (IndexPages)
  // Where the holders that postings record are: this node's ID and its peers' addresses
  // (SetNodes), and how to fetch from them.
  mutable std::mutex raw_mu_;
  std::atomic<uint16_t> self_id_{kNoHolder};
  std::unordered_map<uint16_t, std::string> book_;
  // Fetches in flight, by location (under cache_mu_): a query that misses a vector another one
  // is fetching waits for that fetch.
  std::unordered_map<uint32_t, std::shared_ptr<InFlight>> inflight_;
  std::condition_variable inflight_cv_;
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
  // Inserted vectors (U14): cached like fetched ones, queued apart and written last.
  mutable std::condition_variable room_cv_;  // inserted vectors written (RawAdd backpressure)
  std::deque<std::vector<uint32_t>> update_queue_;
  size_t added_queued_ = 0;  // locations in update_queue_
  bool update_paused_ = false;
  std::chrono::steady_clock::time_point next_update_write_{};  // update_write_bps
  struct FetchCounters {  // fetch_stats
    std::atomic<uint64_t> fetched{0}, fetch_calls{0}, coalesced{0}, fetch_us{0}, written{0}, write_us{0},
        sync_installed{0}, sync_us{0}, drain_calls{0}, drain_waiting{0}, drain_us{0},
        rerank_disk{0}, rerank_mem{0}, pending_recounts{0}, pending_us{0}, added{0},
        added_written{0}, added_wait_us{0};
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
