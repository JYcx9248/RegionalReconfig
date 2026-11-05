#include "fusion/clustering.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <unordered_set>

#include "fusion/kmeans.h"

namespace fusion {
namespace {

struct Node {
  uint64_t b, e;      // range in the permutation array
  bool leaf = false;  // a leaf as it is: named by a split whose rows were streamed
};

// Nodes this big are split one at a time, in parallel inside (their k-means too); smaller ones
// in parallel, each alone. Only nodes this big are split with their rows streamed, so a node gets
// the same arithmetic, and the same children, whether its rows are streamed or in memory.
constexpr uint64_t kBigNode = 20000;

template <class T>
class HierClusterer {
 public:
  HierClusterer(const VectorFile& data, const ClusteringParams& p, uint64_t memory_bytes)
      : data_(data), p_(p), d_(data.dim()), vb_(data.vec_bytes()),
        batch_rows_(std::max<uint64_t>(1, memory_bytes / data.vec_bytes())) {
    FUSION_CHECK(p.centroid_ratio > 0 && p.centroid_ratio <= 1, "centroid ratio must be in (0,1]");
    target_ = std::max(1.0, 1.0 / p.centroid_ratio);
    leaf_max_ = std::max<uint64_t>(1, static_cast<uint64_t>(std::floor(target_ * p.leaf_slack)));
    perm_.resize(data.size());
    std::iota(perm_.begin(), perm_.end(), 0u);
  }

  std::vector<uint32_t> Run(ClusteringStats* st) {
    std::vector<uint32_t> heads;
    std::vector<uint32_t> leaf_sizes;

    // A node whose rows do not fit in memory is split with its rows streamed from the file, level
    // by level, until every node fits. Its rows are read in file order: the root's members are in
    // ID order, and a split keeps each child's in its parent's order.
    std::vector<Node> big{{0, perm_.size()}}, fits;
    uint32_t streamed = 0;
    while (!big.empty()) {
      std::vector<Node> next;
      size_t split = 0;
      for (const Node& nd : big) {
        const uint64_t n = nd.e - nd.b;
        if (nd.leaf || n <= batch_rows_ || n < kBigNode) {
          fits.push_back(nd);
          continue;
        }
        Output out;
        Split(nd, &out, /*streamed=*/true);
        next.insert(next.end(), out.children.begin(), out.children.end());
        ++split;
      }
      if (split > 0) {
        ++streamed;
        Log("  clustering level %u: %zu nodes split, their rows streamed", streamed, split);
      }
      big.swap(next);
    }

    // The other nodes in batches that fit in memory together, each batch's rows read in one pass,
    // by ID. One batch when the whole base fits: the original in-memory clustering.
    uint32_t levels = 0;
    for (size_t i = 0, batch = 1; i < fits.size(); ++batch) {
      size_t j = i + 1;
      uint64_t rows = fits[i].e - fits[i].b;
      while (j < fits.size() && rows + (fits[j].e - fits[j].b) <= batch_rows_) {
        rows += fits[j].e - fits[j].b;
        ++j;
      }
      if (streamed > 0)
        Log("  clustering batch %zu: %zu nodes, %llu vectors", batch, j - i, (unsigned long long)rows);
      levels = std::max(levels, Batch(fits, i, j, rows, streamed, &heads, &leaf_sizes));
      i = j;
    }

    if (st) {
      st->num_leaves = static_cast<uint32_t>(heads.size());
      st->levels = streamed + levels;
      st->min_leaf = *std::min_element(leaf_sizes.begin(), leaf_sizes.end());
      st->max_leaf = *std::max_element(leaf_sizes.begin(), leaf_sizes.end());
      st->mean_leaf = static_cast<double>(perm_.size()) / static_cast<double>(heads.size());
    }
    return heads;
  }

 private:
  struct Output {
    std::vector<Node> children;
    std::vector<uint32_t> heads;
    std::vector<uint32_t> leaf_sizes;
  };

  // Clusters nodes[i, j) to their leaves with their rows in memory. While it runs, rows_ holds
  // the batch's rows in ID order, and perm_ positions in rows_ instead of IDs. That map keeps
  // order, so every node sees its members in the same order, and splits the same way, as with
  // IDs. The heads and perm_ get IDs back after.
  uint32_t Batch(const std::vector<Node>& nodes, size_t i, size_t j, uint64_t rows,
                 uint32_t first_level, std::vector<uint32_t>* heads,
                 std::vector<uint32_t>* leaf_sizes) {
    std::vector<uint32_t> ids;
    ids.reserve(rows);
    for (size_t k = i; k < j; ++k)
      ids.insert(ids.end(), perm_.begin() + static_cast<std::ptrdiff_t>(nodes[k].b),
                 perm_.begin() + static_cast<std::ptrdiff_t>(nodes[k].e));
    std::sort(ids.begin(), ids.end());
    std::vector<uint8_t> buf(ids.size() * vb_);
    data_.Gather(ids.data(), ids.size(), buf.data());
    rows_ = buf.data();
    std::vector<uint64_t> start(j - i + 1, 0);  // where each node starts in the batch
    for (size_t k = i; k < j; ++k) start[k - i + 1] = start[k - i] + (nodes[k].e - nodes[k].b);
    auto remap = [&](auto f) {
#pragma omp parallel for schedule(static) if (rows >= kBigNode)
      for (int64_t t = 0; t < static_cast<int64_t>(rows); ++t) {
        const auto k = std::upper_bound(start.begin(), start.end(), static_cast<uint64_t>(t)) -
                       start.begin() - 1;
        const uint64_t q = nodes[i + static_cast<size_t>(k)].b + (static_cast<uint64_t>(t) - start[k]);
        perm_[q] = f(perm_[q]);
      }
    };
    remap([&](uint32_t id) {
      return static_cast<uint32_t>(std::lower_bound(ids.begin(), ids.end(), id) - ids.begin());
    });
    const size_t first_head = heads->size();
    const uint32_t levels = Levels({nodes.begin() + static_cast<std::ptrdiff_t>(i),
                                    nodes.begin() + static_cast<std::ptrdiff_t>(j)},
                                   first_level, heads, leaf_sizes);
    for (size_t h = first_head; h < heads->size(); ++h) (*heads)[h] = ids[(*heads)[h]];
    remap([&](uint32_t x) { return ids[x]; });
    rows_ = nullptr;
    return levels;
  }

  // Splits the frontier level by level until every node is a leaf (the original loop).
  uint32_t Levels(std::vector<Node> frontier, uint32_t first_level, std::vector<uint32_t>* heads,
                  std::vector<uint32_t>* leaf_sizes) {
    uint32_t levels = 0;
    while (!frontier.empty()) {
      ++levels;
      std::vector<Node> next, small;
      for (const Node& nd : frontier) {
        if (nd.e - nd.b >= kBigNode) {
          Output out;
          Split(nd, &out, /*streamed=*/false);
          Collect(out, &next, heads, leaf_sizes);
        } else {
          small.push_back(nd);
        }
      }
      std::vector<Output> outs(static_cast<size_t>(omp_get_max_threads()));
#pragma omp parallel for schedule(dynamic, 1)
      for (size_t i = 0; i < small.size(); ++i)
        Split(small[i], &outs[omp_get_thread_num()], /*streamed=*/false);
      for (Output& o : outs) Collect(o, &next, heads, leaf_sizes);
      Log("  clustering level %u: %zu nodes split, %zu leaves so far", first_level + levels,
          frontier.size(), heads->size());
      frontier.swap(next);
    }
    return levels;
  }

  static void Collect(Output& o, std::vector<Node>* next, std::vector<uint32_t>* heads,
                      std::vector<uint32_t>* sizes) {
    next->insert(next->end(), o.children.begin(), o.children.end());
    heads->insert(heads->end(), o.heads.begin(), o.heads.end());
    sizes->insert(sizes->end(), o.leaf_sizes.begin(), o.leaf_sizes.end());
    o.children.clear();
    o.heads.clear();
    o.leaf_sizes.clear();
  }

  // A row of the batch in memory, by its position in rows_.
  const T* Row(uint32_t x) const {
    return reinterpret_cast<const T*>(rows_ + static_cast<size_t>(x) * vb_);
  }

  void AddLeaf(uint64_t b, uint64_t e, Output* out) {
    out->heads.push_back(MakeLeafHead(b, e));
    out->leaf_sizes.push_back(static_cast<uint32_t>(e - b));
  }

  // The head is the member closest to the leaf mean.
  uint32_t MakeLeafHead(uint64_t b, uint64_t e) const {
    const uint64_t n = e - b;
    if (n == 1) return perm_[b];
    std::vector<double> sum(d_, 0.0);
    for (uint64_t i = b; i < e; ++i) {
      const T* x = Row(perm_[i]);
      for (uint32_t j = 0; j < d_; ++j) sum[j] += static_cast<double>(x[j]);
    }
    std::vector<float> mean(d_);
    for (uint32_t j = 0; j < d_; ++j) mean[j] = static_cast<float>(sum[j] / static_cast<double>(n));
    uint64_t best = b;
    float best_d = L2SqrMixed(Row(perm_[b]), mean.data(), d_);
    for (uint64_t i = b + 1; i < e; ++i) {
      float dd = L2SqrMixed(Row(perm_[i]), mean.data(), d_);
      if (dd < best_d) {
        best_d = dd;
        best = i;
      }
    }
    return perm_[best];
  }

  // Labels a streamed node's members a chunk at a time, their rows read in file order.
  void LabelStreamed(const Node& nd, const float* cent, uint32_t kk, uint32_t* label) const {
    const uint64_t n = nd.e - nd.b, chunk = data_.chunk_rows();
    std::vector<uint8_t> buf(std::min(n, chunk) * vb_);
    for (uint64_t i0 = 0; i0 < n; i0 += chunk) {
      const uint64_t i1 = std::min(n, i0 + chunk);
      data_.ForEachRow(&perm_[nd.b + i0], i1 - i0, [&](size_t t, const uint8_t* row) {
        std::memcpy(&buf[t * vb_], row, vb_);
      });
#pragma omp parallel for schedule(static)
      for (int64_t i = static_cast<int64_t>(i0); i < static_cast<int64_t>(i1); ++i)
        label[i] = NearestCentroid(reinterpret_cast<const T*>(&buf[(static_cast<uint64_t>(i) - i0) * vb_]),
                                   cent, kk, d_);
    }
  }

  // Splits a node by k-means on a sample of its members. A streamed split reads its rows from
  // the file and cannot make leaves (MakeLeafHead needs the rows in memory): it names them, and
  // the batch that gets them makes them.
  void Split(const Node& nd, Output* out, bool streamed) {
    const uint64_t n = nd.e - nd.b;
    if (nd.leaf || n <= leaf_max_) {
      AddLeaf(nd.b, nd.e, out);
      return;
    }
    uint32_t k = static_cast<uint32_t>(std::llround(static_cast<double>(n) / target_));
    k = std::max<uint32_t>(2, std::min<uint32_t>(k, p_.branch));

    // Training sample (Floyd's algorithm for distinct positions): its members in the order their
    // rows go into it.
    uint64_t s = std::max<uint64_t>(static_cast<uint64_t>(k) * p_.sample_per_centroid, 1024);
    s = std::min<uint64_t>({s, n, static_cast<uint64_t>(p_.max_sample)});
    std::mt19937_64 rng(p_.seed ^ (nd.b * 0x9E3779B97F4A7C15ull) ^ (n << 17));
    std::vector<uint32_t> picked(s);
    if (s == n) {
      for (uint64_t i = 0; i < n; ++i) picked[i] = perm_[nd.b + i];
    } else {
      std::unordered_set<uint64_t> chosen;
      chosen.reserve(s * 2);
      for (uint64_t j = n - s; j < n; ++j) {
        std::uniform_int_distribution<uint64_t> pick(0, j);
        uint64_t t = pick(rng);
        if (!chosen.insert(t).second) chosen.insert(j);
      }
      uint64_t i = 0;
      for (uint64_t pos : chosen) picked[i++] = perm_[nd.b + pos];
    }
    std::vector<float> sample(s * d_);
    if (streamed) {
      std::vector<uint8_t> rows(s * vb_);
      data_.Gather(picked.data(), s, rows.data());
      for (uint64_t i = 0; i < s; ++i)
        ToFloat(reinterpret_cast<const T*>(&rows[i * vb_]), &sample[i * d_], d_);
    } else {
      for (uint64_t i = 0; i < s; ++i) ToFloat(Row(picked[i]), &sample[i * d_], d_);
    }

    KMeansParams kp;
    kp.k = k;
    kp.iters = p_.kmeans_iters;
    kp.seed = rng();
    kp.balance = p_.balance;
    std::vector<float> cent = KMeansTrain(sample.data(), s, d_, kp);
    const uint32_t kk = static_cast<uint32_t>(cent.size() / d_);

    std::vector<uint32_t> label(n);
    if (streamed) {
      LabelStreamed(nd, cent.data(), kk, label.data());
    } else {
#pragma omp parallel for schedule(static) if (n >= kBigNode)
      for (uint64_t i = 0; i < n; ++i)
        label[i] = NearestCentroid(Row(perm_[nd.b + i]), cent.data(), kk, d_);
    }
    std::vector<uint64_t> cnt(kk, 0);
    for (uint32_t l : label) cnt[l]++;
    MergeTinyChildren(cent, kk, &label, &cnt);

    auto leaf = [&](uint64_t b, uint64_t e) {
      if (streamed) {
        out->children.push_back({b, e, true});
      } else {
        AddLeaf(b, e, out);
      }
    };
    if (*std::max_element(cnt.begin(), cnt.end()) == n) {
      // k-means could not separate the points (e.g. many duplicates): cut into chunks.
      const uint64_t cs = static_cast<uint64_t>(std::ceil(target_));
      for (uint64_t b = nd.b; b < nd.e; b += cs) leaf(b, std::min(nd.e, b + cs));
      return;
    }

    // Reorder the node's range by child label (counting sort).
    std::vector<uint64_t> start(kk + 1, 0);
    for (uint32_t c = 0; c < kk; ++c) start[c + 1] = start[c] + cnt[c];
    std::vector<uint64_t> pos(start.begin(), start.end() - 1);
    std::vector<uint32_t> tmp(n);
    for (uint64_t i = 0; i < n; ++i) tmp[pos[label[i]]++] = perm_[nd.b + i];
    std::copy(tmp.begin(), tmp.end(), perm_.begin() + static_cast<std::ptrdiff_t>(nd.b));

    for (uint32_t c = 0; c < kk; ++c) {
      if (cnt[c] == 0) continue;
      Node ch{nd.b + start[c], nd.b + start[c] + cnt[c]};
      if (cnt[c] <= leaf_max_) {
        leaf(ch.b, ch.e);
      } else {
        out->children.push_back(ch);
      }
    }
  }

  // Folds children with fewer than target/2 members into their nearest sibling when the
  // merged cluster still fits in a leaf. Keeps the number of lists close to ratio * N.
  void MergeTinyChildren(const std::vector<float>& cent, uint32_t kk, std::vector<uint32_t>* label,
                         std::vector<uint64_t>* cnt) const {
    const uint64_t tiny = static_cast<uint64_t>(target_ / 2);
    if (tiny == 0 || kk < 2) return;
    std::vector<uint32_t> order(kk), remap(kk);
    std::iota(order.begin(), order.end(), 0u);
    std::iota(remap.begin(), remap.end(), 0u);
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return (*cnt)[a] < (*cnt)[b]; });
    bool changed = false;
    for (uint32_t c : order) {
      if ((*cnt)[c] == 0 || (*cnt)[c] >= tiny) continue;
      uint32_t best = kk;
      float best_d = 0;
      for (uint32_t c2 = 0; c2 < kk; ++c2) {
        if (c2 == c || (*cnt)[c2] == 0 || (*cnt)[c] + (*cnt)[c2] > leaf_max_) continue;
        float dd = L2Sqr(&cent[static_cast<size_t>(c) * d_], &cent[static_cast<size_t>(c2) * d_], d_);
        if (best == kk || dd < best_d) {
          best = c2;
          best_d = dd;
        }
      }
      if (best == kk) continue;
      remap[c] = best;
      (*cnt)[best] += (*cnt)[c];
      (*cnt)[c] = 0;
      changed = true;
    }
    if (!changed) return;
    for (uint32_t c = 0; c < kk; ++c) {
      uint32_t r = c;
      while (remap[r] != r) r = remap[r];
      remap[c] = r;
    }
    for (uint32_t& l : *label) l = remap[l];
  }

  const VectorFile& data_;
  const ClusteringParams p_;
  const uint32_t d_;
  const size_t vb_;
  const uint64_t batch_rows_;  // rows held in memory at once
  double target_ = 10;
  uint64_t leaf_max_ = 15;
  std::vector<uint32_t> perm_;
  const uint8_t* rows_ = nullptr;  // the batch's rows (Batch)
};

}  // namespace

std::vector<uint32_t> HierarchicalBalancedClustering(const VectorFile& data,
                                                     const ClusteringParams& p,
                                                     uint64_t memory_bytes,
                                                     ClusteringStats* stats) {
  FUSION_CHECK(data.size() <= 0xFFFFFFFEull, "at most 2^32-2 vectors are supported");
  return DispatchDType(data.dtype(), [&](auto tag) {
    using T = decltype(tag);
    HierClusterer<T> hc(data, p, memory_bytes);
    return hc.Run(stats);
  });
}

}  // namespace fusion
