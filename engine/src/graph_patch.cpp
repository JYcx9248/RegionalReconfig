#include "fusion/graph_patch.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>

#include "fusion/common.h"
#include "fusion/partition.h"  // Crc32

namespace fusion {
namespace {

constexpr uint32_t kPatchMagic = 0x50475452;  // "RTGP"
constexpr uint32_t kPatchFormat = 1;

using hnswlib::labeltype;
using hnswlib::tableint;
using TopQueue = std::priority_queue<std::pair<float, tableint>, std::vector<std::pair<float, tableint>>,
                                     HnswIndex::CompareByFirst>;

// The graph as a search sees it with the patch applied: a changed element's bytes come from the
// patch, every other element's from the graph. The accessors mirror hnswlib's.
struct View {
  const HnswIndex& g;
  const StagedPatch& sp;

  const char* Level0(tableint id) const {
    if (sp.Changed(id)) return reinterpret_cast<const char*>(sp.patch.level0.data()) + sp.patch.Find(id) * sp.patch.data_bytes;
    return g.data_level0_memory_ + static_cast<size_t>(id) * g.size_data_per_element_;
  }
  const int* Links0(tableint id) const { return reinterpret_cast<const int*>(Level0(id) + g.offsetLevel0_); }
  const unsigned int* Links(tableint id, int level) const {
    if (sp.Changed(id)) {
      const size_t i = sp.patch.Find(id);
      return reinterpret_cast<const unsigned int*>(sp.patch.upper.data() + sp.patch.upper_at[i] +
                                                   static_cast<size_t>(level - 1) * sp.patch.links_bytes);
    }
    return reinterpret_cast<const unsigned int*>(g.linkLists_[id] + static_cast<size_t>(level - 1) * g.size_links_per_element_);
  }
  const char* Data(tableint id) const { return Level0(id) + g.offsetData_; }
  labeltype Label(tableint id) const {
    labeltype l;
    std::memcpy(&l, Level0(id) + g.label_offset_, sizeof(l));
    return l;
  }
  bool Deleted(tableint id) const {
    return *(reinterpret_cast<const unsigned char*>(Links0(id)) + 2) & HnswIndex::DELETE_MARK;
  }
  float Dist(const void* q, tableint id) const { return g.fstdistfunc_(q, Data(id), g.dist_func_param_); }
};

// hnswlib's searchBaseLayerST without a filter or a stop condition.
template <bool bare_bone_search>
TopQueue BaseLayer(const View& v, tableint ep_id, const void* q, size_t ef) {
  const HnswIndex& g = v.g;
  hnswlib::VisitedList* vl = g.visited_list_pool_->getFreeVisitedList();
  hnswlib::vl_type* visited_array = vl->mass;
  const hnswlib::vl_type visited_array_tag = vl->curV;

  TopQueue top_candidates;
  TopQueue candidate_set;
  float lowerBound;
  if (bare_bone_search || !v.Deleted(ep_id)) {
    const float dist = v.Dist(q, ep_id);
    lowerBound = dist;
    top_candidates.emplace(dist, ep_id);
    candidate_set.emplace(-dist, ep_id);
  } else {
    lowerBound = std::numeric_limits<float>::max();
    candidate_set.emplace(-lowerBound, ep_id);
  }
  visited_array[ep_id] = visited_array_tag;

  while (!candidate_set.empty()) {
    const std::pair<float, tableint> current_node_pair = candidate_set.top();
    const float candidate_dist = -current_node_pair.first;
    const bool flag_stop_search = bare_bone_search
                                      ? candidate_dist > lowerBound
                                      : candidate_dist > lowerBound && top_candidates.size() == ef;
    if (flag_stop_search) break;
    candidate_set.pop();

    const tableint current_node_id = current_node_pair.second;
    const int* data = v.Links0(current_node_id);
    const size_t size = *reinterpret_cast<const unsigned short int*>(data);
    for (size_t j = 1; j <= size; j++) {
      const int candidate_id = *(data + j);
      if (!(visited_array[candidate_id] == visited_array_tag)) {
        visited_array[candidate_id] = visited_array_tag;
        const float dist = v.Dist(q, candidate_id);
        if (top_candidates.size() < ef || lowerBound > dist) {
          candidate_set.emplace(-dist, candidate_id);
          if (bare_bone_search || !v.Deleted(candidate_id)) top_candidates.emplace(dist, candidate_id);
          while (top_candidates.size() > ef) top_candidates.pop();
          if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
        }
      }
    }
  }
  g.visited_list_pool_->releaseVisitedList(vl);
  return top_candidates;
}

void PutU32(std::vector<uint8_t>* b, uint32_t v) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
  b->insert(b->end(), p, p + 4);
}
void PutU64(std::vector<uint8_t>* b, uint64_t v) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
  b->insert(b->end(), p, p + 8);
}

class Reader {
 public:
  Reader(const uint8_t* p, size_t n) : p_(p), n_(n) {}
  template <class T>
  T Get() {
    T v;
    std::memcpy(&v, Take(sizeof(T)), sizeof(T));
    return v;
  }
  const uint8_t* Take(size_t n) {
    FUSION_CHECK(n <= n_ - off_, "graph patch truncated");
    const uint8_t* r = p_ + off_;
    off_ += n;
    return r;
  }
  size_t off() const { return off_; }

 private:
  const uint8_t* p_;
  size_t n_;
  size_t off_ = 0;
};

}  // namespace

size_t GraphPatch::Find(uint32_t id) const {
  const auto it = std::lower_bound(ids.begin(), ids.end(), id);
  return it != ids.end() && *it == id ? static_cast<size_t>(it - ids.begin()) : SIZE_MAX;
}

std::vector<uint8_t> GraphPatch::Serialize() const {
  std::vector<uint8_t> b;
  b.reserve(64 + ids.size() * 8 + level0.size() + upper.size());
  PutU32(&b, kPatchMagic);
  PutU32(&b, kPatchFormat);
  PutU32(&b, base_version);
  PutU32(&b, version);
  PutU64(&b, count);
  PutU64(&b, num_deleted);
  PutU32(&b, entry);
  PutU32(&b, static_cast<uint32_t>(maxlevel));
  PutU64(&b, data_bytes);
  PutU64(&b, links_bytes);
  PutU32(&b, static_cast<uint32_t>(ids.size()));
  for (uint32_t id : ids) PutU32(&b, id);
  for (int32_t l : levels) PutU32(&b, static_cast<uint32_t>(l));
  b.insert(b.end(), level0.begin(), level0.end());
  b.insert(b.end(), upper.begin(), upper.end());
  PutU32(&b, Crc32(b.data(), b.size()));
  return b;
}

GraphPatch GraphPatch::Parse(const uint8_t* data, size_t n) {
  FUSION_CHECK(n >= 8, "graph patch truncated");
  uint32_t crc;
  std::memcpy(&crc, data + n - 4, 4);
  FUSION_CHECK(Crc32(data, n - 4) == crc, "graph patch: CRC mismatch");
  Reader r(data, n - 4);
  FUSION_CHECK(r.Get<uint32_t>() == kPatchMagic, "not a graph patch");
  const uint32_t format = r.Get<uint32_t>();
  FUSION_CHECK(format == kPatchFormat, "graph patch format %u, this node reads %u", format, kPatchFormat);
  GraphPatch p;
  p.base_version = r.Get<uint32_t>();
  p.version = r.Get<uint32_t>();
  p.count = r.Get<uint64_t>();
  p.num_deleted = r.Get<uint64_t>();
  p.entry = r.Get<uint32_t>();
  p.maxlevel = static_cast<int32_t>(r.Get<uint32_t>());
  p.data_bytes = r.Get<uint64_t>();
  p.links_bytes = r.Get<uint64_t>();
  const uint32_t k = r.Get<uint32_t>();
  p.ids.resize(k);
  p.levels.resize(k);
  for (auto& id : p.ids) id = r.Get<uint32_t>();
  for (auto& l : p.levels) l = static_cast<int32_t>(r.Get<uint32_t>());
  FUSION_CHECK(std::is_sorted(p.ids.begin(), p.ids.end()) &&
                   std::adjacent_find(p.ids.begin(), p.ids.end()) == p.ids.end(),
               "graph patch: element IDs not ascending");
  FUSION_CHECK(p.data_bytes > 0 && p.links_bytes > 0, "graph patch: empty element layout");
  p.upper_at.assign(1, 0);
  for (int32_t l : p.levels) {
    FUSION_CHECK(l >= 0 && l <= 64, "graph patch: level %d out of range", l);
    p.upper_at.push_back(p.upper_at.back() + static_cast<uint64_t>(l) * p.links_bytes);
  }
  const uint8_t* l0 = r.Take(static_cast<size_t>(k) * p.data_bytes);
  p.level0.assign(l0, l0 + static_cast<size_t>(k) * p.data_bytes);
  const uint8_t* up = r.Take(p.upper_at.back());
  p.upper.assign(up, up + p.upper_at.back());
  FUSION_CHECK(r.off() == n - 4, "graph patch: %zu trailing bytes", n - 4 - r.off());
  return p;
}

GraphPatch CapturePatch(const HnswIndex& g, std::vector<uint32_t> ids, uint32_t base_version,
                        uint32_t version) {
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  GraphPatch p;
  p.base_version = base_version;
  p.version = version;
  p.count = g.cur_element_count;
  p.num_deleted = g.num_deleted_;
  p.entry = g.enterpoint_node_;
  p.maxlevel = g.maxlevel_;
  p.data_bytes = g.size_data_per_element_;
  p.links_bytes = g.size_links_per_element_;
  p.upper_at.assign(1, 0);
  p.level0.resize(ids.size() * p.data_bytes);
  for (size_t i = 0; i < ids.size(); ++i) {
    const uint32_t id = ids[i];
    FUSION_CHECK(id < p.count, "element %u is not in the graph (%llu elements)", id,
                 static_cast<unsigned long long>(p.count));
    const int level = g.element_levels_[id];
    p.levels.push_back(level);
    std::memcpy(p.level0.data() + i * p.data_bytes, g.data_level0_memory_ + static_cast<size_t>(id) * p.data_bytes,
                p.data_bytes);
    const size_t up = static_cast<size_t>(level) * p.links_bytes;
    if (up > 0) {
      const uint8_t* src = reinterpret_cast<const uint8_t*>(g.linkLists_[id]);
      p.upper.insert(p.upper.end(), src, src + up);
    }
    p.upper_at.push_back(p.upper.size());
  }
  p.ids = std::move(ids);
  return p;
}

StagedPatch StagePatch(const HnswIndex& g, GraphPatch p) {
  FUSION_CHECK(p.data_bytes == g.size_data_per_element_ && p.links_bytes == g.size_links_per_element_,
               "graph patch for elements of %llu/%llu bytes, the graph's are %zu/%zu",
               static_cast<unsigned long long>(p.data_bytes), static_cast<unsigned long long>(p.links_bytes),
               g.size_data_per_element_, g.size_links_per_element_);
  const uint64_t cur = g.cur_element_count;
  FUSION_CHECK(p.count >= cur, "graph patch leaves %llu elements, the graph has %llu",
               static_cast<unsigned long long>(p.count), static_cast<unsigned long long>(cur));
  FUSION_CHECK(p.count <= g.max_elements_,
               "graph patch needs room for %llu elements, the graph was loaded with room for %zu "
               "(rtier_node --graph-headroom)",
               static_cast<unsigned long long>(p.count), g.max_elements_);
  FUSION_CHECK(p.entry < p.count && p.maxlevel >= 0, "graph patch: entry point %u of %llu elements", p.entry,
               static_cast<unsigned long long>(p.count));
  FUSION_CHECK(p.num_deleted <= p.count, "graph patch: more deleted elements than elements");
  uint64_t fresh = 0;
  for (size_t i = 0; i < p.ids.size(); ++i) {
    const uint32_t id = p.ids[i];
    FUSION_CHECK(id < p.count, "graph patch names element %u of %llu", id, static_cast<unsigned long long>(p.count));
    if (id >= cur) {
      ++fresh;
    } else {
      FUSION_CHECK(g.element_levels_[id] == p.levels[i], "graph patch: element %u at level %d, the graph's is at %d",
                   id, p.levels[i], g.element_levels_[id]);
    }
  }
  FUSION_CHECK(fresh == p.count - cur, "graph patch adds %llu elements but holds %llu of them",
               static_cast<unsigned long long>(p.count - cur), static_cast<unsigned long long>(fresh));
  StagedPatch sp;
  sp.changed.assign((g.max_elements_ + 63) / 64, 0);
  for (uint32_t id : p.ids) sp.changed[id >> 6] |= uint64_t{1} << (id & 63);
  sp.patch = std::move(p);
  return sp;
}

std::priority_queue<std::pair<float, labeltype>> SearchPatched(const HnswIndex& g, const StagedPatch& sp,
                                                              const void* q, size_t k) {
  const GraphPatch& p = sp.patch;
  const View v{g, sp};
  std::priority_queue<std::pair<float, labeltype>> result;
  if (p.count == 0) return result;

  tableint currObj = p.entry;
  float curdist = v.Dist(q, currObj);
  for (int level = p.maxlevel; level > 0; level--) {
    bool changed = true;
    while (changed) {
      changed = false;
      const unsigned int* data = v.Links(currObj, level);
      const int size = *reinterpret_cast<const unsigned short int*>(data);
      const tableint* datal = reinterpret_cast<const tableint*>(data + 1);
      for (int i = 0; i < size; i++) {
        const tableint cand = datal[i];
        if (cand > g.max_elements_) throw std::runtime_error("cand error");
        const float d = v.Dist(q, cand);
        if (d < curdist) {
          curdist = d;
          currObj = cand;
          changed = true;
        }
      }
    }
  }
  const size_t ef = std::max(g.ef_, k);
  TopQueue top = p.num_deleted == 0 ? BaseLayer<true>(v, currObj, q, ef) : BaseLayer<false>(v, currObj, q, ef);
  while (top.size() > k) top.pop();
  while (!top.empty()) {
    const std::pair<float, tableint> rez = top.top();
    result.push(std::pair<float, labeltype>(rez.first, v.Label(rez.second)));
    top.pop();
  }
  return result;
}

void ApplyPatch(HnswIndex& g, const StagedPatch& sp) {
  const GraphPatch& p = sp.patch;
  const uint64_t cur = g.cur_element_count;
  for (size_t i = 0; i < p.ids.size(); ++i) {
    const uint32_t id = p.ids[i];
    const int level = p.levels[i];
    std::memcpy(g.data_level0_memory_ + static_cast<size_t>(id) * g.size_data_per_element_,
                p.level0.data() + i * p.data_bytes, p.data_bytes);
    const size_t up = static_cast<size_t>(level) * p.links_bytes;
    if (id >= cur) {
      // A new element: its upper lists get their own buffer, as hnswlib's addPoint allocates them.
      if (level > 0) {
        char* buf = static_cast<char*>(std::malloc(up + 1));
        FUSION_CHECK(buf != nullptr, "out of memory for the upper links of element %u", id);
        std::memcpy(buf, p.upper.data() + p.upper_at[i], up);
        g.linkLists_[id] = buf;
      } else {
        g.linkLists_[id] = nullptr;
      }
      g.element_levels_[id] = level;
      labeltype label;
      std::memcpy(&label, p.level0.data() + i * p.data_bytes + g.label_offset_, sizeof(label));
      std::lock_guard<std::mutex> l(g.label_lookup_lock);
      g.label_lookup_[label] = id;
    } else if (up > 0) {
      std::memcpy(g.linkLists_[id], p.upper.data() + p.upper_at[i], up);
    }
  }
  g.enterpoint_node_ = p.entry;
  g.maxlevel_ = p.maxlevel;
  g.num_deleted_ = p.num_deleted;
  g.cur_element_count = p.count;
}

}  // namespace fusion
