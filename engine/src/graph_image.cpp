#include "fusion/graph_image.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <type_traits>

#include "fusion/common.h"

namespace fusion {
namespace {

using hnswlib::labeltype;
using hnswlib::linklistsizeint;
using hnswlib::tableint;

// saveIndex's header: these fields, in this order, each as writeBinaryPOD writes it (its bytes).
constexpr size_t kHeaderBytes =
    sizeof(HnswIndex::offsetLevel0_) + sizeof(HnswIndex::max_elements_) + sizeof(size_t) /* cur_element_count */ +
    sizeof(HnswIndex::size_data_per_element_) + sizeof(HnswIndex::label_offset_) + sizeof(HnswIndex::offsetData_) +
    sizeof(HnswIndex::maxlevel_) + sizeof(HnswIndex::enterpoint_node_) + sizeof(HnswIndex::maxM_) +
    sizeof(HnswIndex::maxM0_) + sizeof(HnswIndex::M_) + sizeof(HnswIndex::mult_) +
    sizeof(HnswIndex::ef_construction_);
// writeBinaryPOD writes the element count's atomic as it lies in memory: its value.
static_assert(sizeof(std::atomic<size_t>) == sizeof(size_t), "hnswlib's element count is not a plain size_t");

template <typename F>
void Put(std::vector<uint8_t>* out, const F& v) {
  static_assert(std::is_trivially_copyable<F>::value, "a header field is copied as bytes");
  const auto* p = reinterpret_cast<const uint8_t*>(&v);
  out->insert(out->end(), p, p + sizeof(F));
}

template <typename F>
F Get(const uint8_t*& p) {
  F v;
  std::memcpy(&v, p, sizeof(F));
  p += sizeof(F);
  return v;
}

std::vector<uint8_t> Header(const HnswIndex& g) {
  std::vector<uint8_t> h;
  h.reserve(kHeaderBytes);
  Put(&h, g.offsetLevel0_);
  Put(&h, g.max_elements_);
  Put(&h, static_cast<size_t>(g.cur_element_count));
  Put(&h, g.size_data_per_element_);
  Put(&h, g.label_offset_);
  Put(&h, g.offsetData_);
  Put(&h, g.maxlevel_);
  Put(&h, g.enterpoint_node_);
  Put(&h, g.maxM_);
  Put(&h, g.maxM0_);
  Put(&h, g.M_);
  Put(&h, g.mult_);
  Put(&h, g.ef_construction_);
  return h;
}

// An element's upper-level link lists, as saveIndex sizes them.
uint32_t LinkBytes(const HnswIndex& g, size_t i) {
  return g.element_levels_[i] > 0 ? static_cast<uint32_t>(g.size_links_per_element_ * g.element_levels_[i]) : 0;
}

}  // namespace

GraphImage::GraphImage(const HnswIndex& g) : header_(Header(g)) {
  const size_t count = g.cur_element_count;
  level0_ = static_cast<uint64_t>(count) * g.size_data_per_element_;
  link_at_.resize(count + 1);
  uint64_t at = header_.size() + level0_;
  for (size_t i = 0; i < count; ++i) {
    link_at_[i] = at;
    at += sizeof(uint32_t) + LinkBytes(g, i);
  }
  link_at_[count] = at;
  size_ = at;
}

void GraphImage::Read(const HnswIndex& g, uint64_t off, size_t n, uint8_t* out) const {
  FUSION_CHECK(off <= size_ && n <= size_ - off, "graph image: bytes %llu to %llu of %llu",
               static_cast<unsigned long long>(off), static_cast<unsigned long long>(off + n),
               static_cast<unsigned long long>(size_));
  auto take = [&](const void* src, size_t k) {
    std::memcpy(out, src, k);
    out += k;
    off += k;
    n -= k;
  };
  if (n > 0 && off < header_.size()) take(header_.data() + off, std::min<uint64_t>(n, header_.size() - off));
  const uint64_t links = header_.size() + level0_;
  if (n > 0 && off < links) take(g.data_level0_memory_ + (off - header_.size()), std::min<uint64_t>(n, links - off));
  if (n == 0) return;
  // The link records: an element's u32 size, then its upper-level lists.
  size_t i = static_cast<size_t>(std::upper_bound(link_at_.begin(), link_at_.end(), off) - link_at_.begin()) - 1;
  while (n > 0) {
    const uint64_t rel = off - link_at_[i];
    const uint32_t len = static_cast<uint32_t>(link_at_[i + 1] - link_at_[i] - sizeof(uint32_t));
    if (rel < sizeof(uint32_t)) {
      take(reinterpret_cast<const uint8_t*>(&len) + rel, std::min<uint64_t>(n, sizeof(uint32_t) - rel));
    } else {
      const uint64_t at = rel - sizeof(uint32_t);
      take(g.linkLists_[i] + at, std::min<uint64_t>(n, len - at));
    }
    if (off == link_at_[i + 1]) ++i;
  }
}

GraphLoader::GraphLoader(hnswlib::SpaceInterface<float>* space, size_t max_elements)
    : space_(space), room_(max_elements) {
  header_.reserve(kHeaderBytes);
}

void GraphLoader::Begin() {
  const uint8_t* p = header_.data();
  auto g = std::make_unique<HnswIndex>(space_);
  g->offsetLevel0_ = Get<size_t>(p);
  const size_t own_room = Get<size_t>(p);
  count_ = Get<size_t>(p);
  g->size_data_per_element_ = Get<size_t>(p);
  g->label_offset_ = Get<size_t>(p);
  g->offsetData_ = Get<size_t>(p);
  g->maxlevel_ = Get<int>(p);
  g->enterpoint_node_ = Get<tableint>(p);
  g->maxM_ = Get<size_t>(p);
  g->maxM0_ = Get<size_t>(p);
  g->M_ = Get<size_t>(p);
  g->mult_ = Get<double>(p);
  g->ef_construction_ = Get<size_t>(p);

  // From here on as loadIndex, which also takes the graph's own room when it holds more.
  const size_t room = room_ < count_ ? own_room : room_;
  g->max_elements_ = room;
  g->data_size_ = space_->get_data_size();
  g->fstdistfunc_ = space_->get_dist_func();
  g->dist_func_param_ = space_->get_dist_func_param();
  g->size_links_per_element_ = g->maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
  g->size_links_level0_ = g->maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
  // loadIndex trusts the file; the bytes of a peer are checked to be a graph over this node's
  // vectors before anything is allocated for them.
  FUSION_CHECK(count_ <= room && g->offsetLevel0_ == 0 && g->offsetData_ == g->size_links_level0_ &&
                   g->label_offset_ == g->offsetData_ + g->data_size_ &&
                   g->size_data_per_element_ == g->label_offset_ + sizeof(labeltype) && g->mult_ > 0,
               "graph image: %zu elements of %zu bytes (vectors of %zu) do not make a graph over this node's "
               "vectors of %zu bytes, with room for %zu",
               count_, g->size_data_per_element_, g->label_offset_ - g->offsetData_, g->data_size_, room);
  g->data_level0_memory_ = static_cast<char*>(std::malloc(room * g->size_data_per_element_));
  if (g->data_level0_memory_ == nullptr) throw std::runtime_error("graph image: no memory for the level-0 records");
  std::vector<std::mutex>(room).swap(g->link_list_locks_);
  std::vector<std::mutex>(HnswIndex::MAX_LABEL_OPERATION_LOCKS).swap(g->label_op_locks_);
  g->visited_list_pool_.reset(new hnswlib::VisitedListPool(1, room));
  g->linkLists_ = static_cast<char**>(std::malloc(sizeof(void*) * room));
  if (g->linkLists_ == nullptr) throw std::runtime_error("graph image: no memory for the link lists");
  g->element_levels_ = std::vector<int>(room);
  g->revSize_ = 1.0 / g->mult_;
  g->ef_ = 10;
  // The count grows with the link records that are in, so that a graph given up half way frees
  // exactly the lists it holds (hnswlib's clear()).
  g->cur_element_count = 0;
  g_ = std::move(g);
  level0_ = static_cast<uint64_t>(count_) * g_->size_data_per_element_;
  state_ = count_ == 0 ? State::kDone : State::kLevel0;
}

void GraphLoader::EndElement() {
  const auto id = static_cast<tableint>(elem_);
  g_->label_lookup_[g_->getExternalLabel(id)] = id;
  if (++elem_ < count_) {
    state_ = State::kLinkSize;
    return;
  }
  for (size_t i = 0; i < count_; ++i)
    if (g_->isMarkedDeleted(static_cast<tableint>(i))) g_->num_deleted_ += 1;
  state_ = State::kDone;
}

void GraphLoader::Append(const uint8_t* p, size_t n) {
  received_ += n;
  while (n > 0) {
    size_t k = 0;
    switch (state_) {
      case State::kHeader:
        k = std::min(n, kHeaderBytes - header_.size());
        header_.insert(header_.end(), p, p + k);
        if (header_.size() == kHeaderBytes) Begin();
        break;
      case State::kLevel0:
        k = static_cast<size_t>(std::min<uint64_t>(n, level0_ - level0_in_));
        std::memcpy(g_->data_level0_memory_ + level0_in_, p, k);
        level0_in_ += k;
        if (level0_in_ == level0_) state_ = State::kLinkSize;
        break;
      case State::kLinkSize:
        k = std::min(n, sizeof(size_) - size_in_);
        std::memcpy(size_ + size_in_, p, k);
        size_in_ += k;
        if (size_in_ == sizeof(size_)) {
          size_in_ = 0;
          std::memcpy(&links_, size_, sizeof(links_));
          const size_t level = links_ / g_->size_links_per_element_;
          FUSION_CHECK(links_ % g_->size_links_per_element_ == 0 && level <= static_cast<size_t>(g_->maxlevel_),
                       "graph image: element %zu has %u bytes of upper-level links (%zu per level, top level %d)",
                       elem_, links_, g_->size_links_per_element_, g_->maxlevel_);
          if (links_ == 0) {
            g_->linkLists_[elem_] = nullptr;
            g_->element_levels_[elem_] = 0;
            g_->cur_element_count = elem_ + 1;
            EndElement();
          } else {
            char* b = static_cast<char*>(std::malloc(links_));
            if (b == nullptr) throw std::runtime_error("graph image: no memory for a link list");
            g_->linkLists_[elem_] = b;
            g_->element_levels_[elem_] = static_cast<int>(level);
            g_->cur_element_count = elem_ + 1;
            links_in_ = 0;
            state_ = State::kLinks;
          }
        }
        break;
      case State::kLinks:
        k = std::min<size_t>(n, links_ - links_in_);
        std::memcpy(g_->linkLists_[elem_] + links_in_, p, k);
        links_in_ += static_cast<uint32_t>(k);
        if (links_in_ == links_) EndElement();
        break;
      case State::kDone:
        throw std::runtime_error(StrFormat("graph image: %zu bytes past its end", n));
    }
    p += k;
    n -= k;
  }
}

std::unique_ptr<HnswIndex> GraphLoader::Take() {
  FUSION_CHECK(state_ == State::kDone && g_ != nullptr, "graph image: %llu bytes in, the graph is not complete",
               static_cast<unsigned long long>(received_));
  return std::move(g_);
}

}  // namespace fusion
