#include "fusion/engine.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <vector>

#include "fusion/dataset.h"
#include "fusion/distance.h"
#include "fusion/filter.h"
#include "fusion/hnsw_space.h"
#include "fusion/index_meta.h"
#include "fusion/layout.h"
#include "fusion/pq.h"

namespace fusion {
namespace {

template <class T>
class EngineImpl : public Engine {
 public:
  explicit EngineImpl(const EngineOptions& o, const IndexMeta& meta) : opts_(o) {
    auto path = [&](const std::string& f) { return JoinPath(o.index_dir, f); };
    dim_ = static_cast<uint32_t>(meta.GetU64("dim"));
    n_ = meta.GetU64("num_vectors");
    const std::string layouts = meta.GetOr("layouts", "");
    FUSION_CHECK(("," + layouts + ",").find("," + o.layout + ",") != std::string::npos,
                 "index has no '%s' layout (available: %s)", o.layout.c_str(), layouts.c_str());
    FUSION_CHECK(o.num_workers >= 1, "need at least one worker");

    // Host-memory tier: navigation graph, posting lists (IDs only), page mapping table.
    space_ = MakeL2Space(DTypeOf<T>(), dim_);
    graph_ = std::make_unique<HnswIndex>(space_.get(), path(files::kGraph));
    lists_ = PostingLists::Load(path(files::kPostings));
    layout_ = LayoutMap::Load(path(files::LayoutMap(o.layout)));
    FUSION_CHECK(layout_.size() == n_, "layout map size mismatch");
    FUSION_CHECK(graph_->getCurrentElementCount() == lists_.num_lists(),
                 "graph and posting lists disagree");

    // Upper bound on gathered IDs: the max_nprobe longest lists.
    std::vector<uint32_t> sizes(lists_.num_lists());
    for (uint32_t c = 0; c < lists_.num_lists(); ++c) sizes[c] = lists_.Size(c);
    const uint32_t m = std::min<uint32_t>(o.max_nprobe, lists_.num_lists());
    std::partial_sort(sizes.begin(), sizes.begin() + m, sizes.end(), std::greater<uint32_t>());
    uint64_t max_cand = 0;
    for (uint32_t i = 0; i < m; ++i) max_cand += sizes[i];
    FUSION_CHECK(max_cand < 0xFFFFFFFFull, "max_nprobe too large");
    max_candidates_ = static_cast<uint32_t>(std::max<uint64_t>(max_cand, 1));

    // GPU (or host) tier: PQ codes.
    PQCodebook cb = PQCodebook::Load(path(files::kPQCodebook));
    uint64_t code_n = 0;
    uint32_t code_m = 0;
    codes_ = LoadPQCodes(path(files::kPQCodes), &code_n, &code_m);
    FUSION_CHECK(code_n == n_ && code_m == cb.m(), "PQ codes do not match the index");
    FilterConfig fc;
    fc.num_workers = o.num_workers;
    fc.max_candidates = max_candidates_;
    fc.max_topn = std::max<uint32_t>(o.max_rerank, 1);
    fc.gpu_device = o.gpu_device;
    if (o.device == FilterDevice::kGpu) {
      filter_ = CreateGpuFilter(cb, codes_.data(), n_, fc);
      codes_.clear();
      codes_.shrink_to_fit();  // the device holds the codes now
    } else {
      filter_ = CreateCpuFilter(cb, codes_.data(), n_, fc);
    }
    pq_m_ = cb.m();

    // SSD tier.
    page_file_ = std::make_unique<PageFile>(path(files::LayoutVectors(o.layout)),
                                            layout_.page_size, o.direct_io);
    FUSION_CHECK(page_file_->num_pages() >= layout_.num_pages, "vector page file is truncated");
    for (int i = 0; i < o.num_workers; ++i) {
      Worker w;
      w.reader = MakePageReader(*page_file_, o.io, o.io_depth);
      w.scratch = std::make_unique<RerankScratch>(fc.max_topn, layout_.page_size);
      w.qf.resize(dim_);
      workers_.push_back(std::move(w));
    }
  }

  void Configure(const SearchParams& p) override {
    graph_->setEf(p.graph_ef ? std::max(p.graph_ef, p.nprobe) : 2 * p.nprobe);
  }

  uint32_t Search(int wi, const void* query, const SearchParams& p, uint32_t* ids, float* dists,
                  QueryStats* st) override {
    FUSION_CHECK(p.nprobe >= 1 && p.nprobe <= opts_.max_nprobe, "nprobe %u outside [1, %u]",
                 p.nprobe, opts_.max_nprobe);
    FUSION_CHECK(p.rerank >= 1 && p.rerank <= opts_.max_rerank, "rerank %u outside [1, %u]",
                 p.rerank, opts_.max_rerank);
    Timer total;
    Worker& w = workers_.at(static_cast<size_t>(wi));
    FilterWorker* fw = filter_->worker(wi);
    const T* q = static_cast<const T*>(query);

    // (1) distance table, asynchronous on the GPU.
    ToFloat(q, w.qf.data(), dim_);
    fw->BeginQuery(w.qf.data());

    // (2) top-m posting lists from the navigation graph.
    Timer t;
    auto res = graph_->searchKnn(q, p.nprobe);
    st->graph_us = t.Us();

    // (3) collect vector IDs of those lists.
    t.Reset();
    uint32_t* cand = fw->candidate_buffer();
    uint32_t nc = 0;
    st->lists = static_cast<uint32_t>(res.size());
    while (!res.empty()) {
      const uint32_t c = static_cast<uint32_t>(res.top().second);
      res.pop();
      const uint32_t len = lists_.Size(c);
      std::memcpy(cand + nc, lists_.List(c), len * sizeof(uint32_t));
      nc += len;
    }
    st->candidates = nc;
    st->gather_us = t.Us();

    // (4-7) dedup + PQ distances + top-n.
    t.Reset();
    uint32_t n_unique = 0;
    const uint32_t nt = fw->Filter(nc, p.rerank, &n_unique);
    st->unique = n_unique;
    st->topn = nt;
    st->filter_us = t.Us();

    // (8) heuristic re-ranking against raw vectors on SSD.
    t.Reset();
    RerankParams rp = p.rr;
    rp.k = p.k;
    st->rerank = RerankStats();
    const uint32_t cnt = HeuristicRerank<T>(q, dim_, fw->result_ids(), nt, layout_,
                                            w.reader.get(), w.scratch.get(), rp, ids, dists,
                                            &st->rerank);
    st->rerank_us = t.Us();
    st->total_us = total.Us();
    return cnt;
  }

  DType dtype() const override { return DTypeOf<T>(); }
  uint32_t dim() const override { return dim_; }
  uint64_t size() const override { return n_; }
  int num_workers() const override { return static_cast<int>(workers_.size()); }
  const char* filter_name() const override { return filter_->name(); }
  IoBackend io_backend() const override { return workers_.front().reader->backend(); }

  std::string Describe() const override {
    const double mb = 1e6;
    const uint64_t graph_bytes =
        static_cast<uint64_t>(graph_->max_elements_) * graph_->size_data_per_element_;
    const uint64_t list_bytes = lists_.ids.size() * 4 + lists_.offsets.size() * 8;
    const uint64_t map_bytes = layout_.size() * 6;
    std::string s;
    s += StrFormat("  vectors          : %llu x %u (%s)\n", (unsigned long long)n_, dim_,
                   DTypeName(DTypeOf<T>()));
    s += StrFormat("  host memory tier : graph %.1f MB (%llu heads) + ID lists %.1f MB "
                   "(%.2f IDs/vector) + page map %.1f MB\n",
                   graph_bytes / mb, (unsigned long long)lists_.num_lists(), list_bytes / mb,
                   static_cast<double>(lists_.ids.size()) / static_cast<double>(n_),
                   map_bytes / mb);
    s += StrFormat("  %s tier          : PQ codes %.1f MB (%u B/vector)\n",
                   std::strcmp(filter_->name(), "gpu") == 0 ? "GPU " : "host",
                   filter_->code_bytes() / mb, pq_m_);
    s += StrFormat("  SSD tier         : %llu pages x %u B = %.1f MB ('%s' layout, %u vectors/page, "
                   "%s I/O, %s)\n",
                   (unsigned long long)layout_.num_pages, layout_.page_size,
                   layout_.num_pages * static_cast<double>(layout_.page_size) / mb,
                   opts_.layout.c_str(), layout_.vectors_per_page,
                   page_file_->direct() ? "direct" : "buffered", IoBackendName(io_backend()));
    s += StrFormat("  workers          : %d, max candidates/query %u\n", num_workers(),
                   max_candidates_);
    return s;
  }

 private:
  struct Worker {
    std::unique_ptr<PageReader> reader;
    std::unique_ptr<RerankScratch> scratch;
    std::vector<float> qf;
  };

  EngineOptions opts_;
  uint32_t dim_ = 0;
  uint64_t n_ = 0;
  uint32_t pq_m_ = 0;
  uint32_t max_candidates_ = 0;
  std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
  std::unique_ptr<HnswIndex> graph_;
  PostingLists lists_;
  LayoutMap layout_;
  std::vector<uint8_t> codes_;
  std::unique_ptr<FilterBackend> filter_;
  std::unique_ptr<PageFile> page_file_;
  std::vector<Worker> workers_;
};

}  // namespace

std::unique_ptr<Engine> Engine::Open(const EngineOptions& opts) {
  IndexMeta meta = IndexMeta::Load(JoinPath(opts.index_dir, files::kMeta));
  const DType dt = ParseDType(meta.Get("dtype"));
  return DispatchDType(dt, [&](auto tag) -> std::unique_ptr<Engine> {
    using T = decltype(tag);
    return std::make_unique<EngineImpl<T>>(opts, meta);
  });
}

}  // namespace fusion
