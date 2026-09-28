#include "fusion/rerank.h"

#include <algorithm>

#include "fusion/distance.h"
#include "fusion/hd_common.h"

namespace fusion {

RerankScratch::RerankScratch(uint32_t max_candidates, uint32_t page_size)
    : capacity_(std::max<uint32_t>(max_candidates, 1)),
      page_size_(page_size),
      pages_(AllocAligned(4096, static_cast<size_t>(capacity_) * page_size)),
      buf_batch_(capacity_, 0) {
  const uint32_t tsize = NextPow2(std::max<uint32_t>(2 * capacity_, 16));
  keys_.assign(tsize, 0);
  vals_.assign(tsize, 0);
  gens_.assign(tsize, 0);
  mask_ = tsize - 1;
  req_pages.reserve(capacity_);
  req_bufs.reserve(capacity_);
}

void RerankScratch::NewQuery() {
  next_buf_ = 0;
  if (++gen_ == 0) {  // wrapped: clear stale generations
    std::fill(gens_.begin(), gens_.end(), 0);
    gen_ = 1;
  }
}

bool RerankScratch::Find(uint32_t page, uint32_t* buf) const {
  for (uint32_t h = HashId(page) & mask_;; h = (h + 1) & mask_) {
    if (gens_[h] != gen_) return false;
    if (keys_[h] == page) {
      *buf = vals_[h];
      return true;
    }
  }
}

uint32_t RerankScratch::Allocate(uint32_t page, uint32_t batch_no, bool track) {
  FUSION_CHECK(next_buf_ < capacity_, "re-ranking buffer overflow");
  const uint32_t b = next_buf_++;
  buf_batch_[b] = batch_no;
  if (track) {
    uint32_t h = HashId(page) & mask_;
    while (gens_[h] == gen_) h = (h + 1) & mask_;
    gens_[h] = gen_;
    keys_[h] = page;
    vals_[h] = b;
  }
  return b;
}

namespace {

// where(j) is candidate j's (page, slot) in the file `reader` reads.
template <class T, class Where>
uint32_t RerankImpl(const T* query, uint32_t dim, const uint32_t* cand, uint32_t n_cand,
                    uint32_t vb, Where where, PageReader* reader, RerankScratch* s,
                    const RerankParams& p, uint32_t* out_ids, float* out_dists, RerankStats* st) {
  const uint32_t k = std::max<uint32_t>(p.k, 1);
  const uint32_t bsz = p.batch ? p.batch : k;
  s->NewQuery();
  auto& heap = s->heap;  // max-heap on (distance, id): heap.front() is the current k-th best
  heap.clear();
  s->cand_buf.resize(bsz);
  uint32_t stable = 0;
  uint32_t batch_no = 0;

  for (uint32_t i = 0; i < n_cand; i += bsz, ++batch_no) {
    const uint32_t end = std::min(n_cand, i + bsz);

    // S_{n-1}: IDs in the heap before this mini-batch.
    s->prev.clear();
    for (const auto& e : heap) s->prev.push_back(e.second);

    // Plan the mini-batch's reads.
    s->req_pages.clear();
    s->req_bufs.clear();
    for (uint32_t j = i; j < end; ++j) {
      const uint32_t page = where(j).first;
      uint32_t b;
      if (p.io_dedup && s->Find(page, &b)) {
        if (s->loaded_in_batch(b) == batch_no) {
          st->intra_merged++;
        } else {
          st->buffer_hits++;
        }
      } else {
        b = s->Allocate(page, batch_no, p.io_dedup);
        s->req_pages.push_back(page);
        s->req_bufs.push_back(s->Buffer(b));
      }
      s->cand_buf[j - i] = b;
    }
    Timer io_timer;
    if (!s->req_pages.empty())
      reader->Read(s->req_pages.data(), s->req_bufs.data(), static_cast<uint32_t>(s->req_pages.size()));
    st->io_us += io_timer.Us();
    st->pages_read += static_cast<uint32_t>(s->req_pages.size());

    // Exact distances; keep the k best in the max-heap. Candidates are ordered by (distance,
    // id), so the kept set does not depend on the order candidates arrive in (rtier compares
    // results computed on different nodes and needs this to be deterministic).
    Timer cpu_timer;
    for (uint32_t j = i; j < end; ++j) {
      const uint32_t id = cand[j];
      const T* v = reinterpret_cast<const T*>(s->Buffer(s->cand_buf[j - i]) +
                                              static_cast<size_t>(where(j).second) * vb);
      const float d = L2Sqr(query, v, dim);
      if (heap.size() < k) {
        heap.emplace_back(d, id);
        std::push_heap(heap.begin(), heap.end());
      } else if (std::make_pair(d, id) < heap.front()) {
        std::pop_heap(heap.begin(), heap.end());
        heap.back() = {d, id};
        std::push_heap(heap.begin(), heap.end());
      }
    }
    st->compute_us += cpu_timer.Us();
    st->reranked += end - i;
    st->batches++;

    // The stop rule only applies once the heap holds k results (with batch = k, the paper's
    // setting, this is always true after the first mini-batch).
    if (p.heuristic && heap.size() >= k) {
      uint32_t changed = 0;
      for (const auto& e : heap) {
        if (std::find(s->prev.begin(), s->prev.end(), e.second) == s->prev.end()) ++changed;
      }
      const float delta = static_cast<float>(changed) / static_cast<float>(k);
      if (delta <= p.eps) {
        if (++stable >= p.beta) break;
      } else {
        stable = 0;
      }
    }
  }

  std::sort_heap(heap.begin(), heap.end());  // ascending
  for (size_t i = 0; i < heap.size(); ++i) {
    out_dists[i] = heap[i].first;
    out_ids[i] = heap[i].second;
  }
  return static_cast<uint32_t>(heap.size());
}

}  // namespace

template <class T>
uint32_t HeuristicRerank(const T* query, uint32_t dim, const uint32_t* cand, uint32_t n_cand,
                         const LayoutMap& layout, PageReader* reader, RerankScratch* s,
                         const RerankParams& p, uint32_t* out_ids, float* out_dists,
                         RerankStats* st) {
  auto where = [&](uint32_t j) {
    return std::make_pair(layout.page_of[cand[j]], static_cast<uint32_t>(layout.slot_of[cand[j]]));
  };
  return RerankImpl(query, dim, cand, n_cand, layout.vec_bytes, where, reader, s, p, out_ids,
                    out_dists, st);
}

template <class T>
uint32_t HeuristicRerank(const T* query, uint32_t dim, const uint32_t* cand, const uint32_t* locs,
                         uint32_t n_cand, const RawLayout& layout, PageReader* reader,
                         RerankScratch* s, const RerankParams& p, uint32_t* out_ids,
                         float* out_dists, RerankStats* st) {
  auto where = [&](uint32_t j) { return std::make_pair(layout.page(locs[j]), layout.slot(locs[j])); };
  return RerankImpl(query, dim, cand, n_cand, layout.vec_bytes, where, reader, s, p, out_ids,
                    out_dists, st);
}

#define FUSION_INSTANTIATE_RERANK(T)                                                          \
  template uint32_t HeuristicRerank<T>(const T*, uint32_t, const uint32_t*, uint32_t,          \
                                       const LayoutMap&, PageReader*, RerankScratch*,           \
                                       const RerankParams&, uint32_t*, float*, RerankStats*);   \
  template uint32_t HeuristicRerank<T>(const T*, uint32_t, const uint32_t*, const uint32_t*,   \
                                       uint32_t, const RawLayout&, PageReader*, RerankScratch*, \
                                       const RerankParams&, uint32_t*, float*, RerankStats*);
FUSION_INSTANTIATE_RERANK(uint8_t)
FUSION_INSTANTIATE_RERANK(int8_t)
FUSION_INSTANTIATE_RERANK(float)
#undef FUSION_INSTANTIATE_RERANK

}  // namespace fusion
