#include "fusion/clustering.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <unordered_set>

#include "fusion/kmeans.h"

namespace fusion {
namespace {

struct Node {
  uint64_t b, e;  // range in the permutation array
};

template <class T>
class HierClusterer {
 public:
  HierClusterer(const VectorFile& data, const ClusteringParams& p)
      : data_(data), p_(p), d_(data.dim()) {
    FUSION_CHECK(p.centroid_ratio > 0 && p.centroid_ratio <= 1, "centroid ratio must be in (0,1]");
    target_ = std::max(1.0, 1.0 / p.centroid_ratio);
    leaf_max_ = std::max<uint64_t>(1, static_cast<uint64_t>(std::floor(target_ * p.leaf_slack)));
    perm_.resize(data.size());
    std::iota(perm_.begin(), perm_.end(), 0u);
  }

  std::vector<uint32_t> Run(ClusteringStats* st) {
    std::vector<Node> frontier{{0, perm_.size()}};
    std::vector<uint32_t> heads;
    std::vector<uint32_t> leaf_sizes;
    uint32_t levels = 0;
    constexpr uint64_t kBigNode = 20000;  // split big nodes one at a time, in parallel inside

    while (!frontier.empty()) {
      ++levels;
      std::vector<Node> next, small;
      for (const Node& nd : frontier) {
        if (nd.e - nd.b >= kBigNode) {
          Output out;
          Split(nd, &out);
          Collect(out, &next, &heads, &leaf_sizes);
        } else {
          small.push_back(nd);
        }
      }
      std::vector<Output> outs(static_cast<size_t>(omp_get_max_threads()));
#pragma omp parallel for schedule(dynamic, 1)
      for (size_t i = 0; i < small.size(); ++i) Split(small[i], &outs[omp_get_thread_num()]);
      for (Output& o : outs) Collect(o, &next, &heads, &leaf_sizes);
      Log("  clustering level %u: %zu nodes split, %zu leaves so far", levels, frontier.size(),
          heads.size());
      frontier.swap(next);
    }

    if (st) {
      st->num_leaves = static_cast<uint32_t>(heads.size());
      st->levels = levels;
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

  static void Collect(Output& o, std::vector<Node>* next, std::vector<uint32_t>* heads,
                      std::vector<uint32_t>* sizes) {
    next->insert(next->end(), o.children.begin(), o.children.end());
    heads->insert(heads->end(), o.heads.begin(), o.heads.end());
    sizes->insert(sizes->end(), o.leaf_sizes.begin(), o.leaf_sizes.end());
    o.children.clear();
    o.heads.clear();
    o.leaf_sizes.clear();
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
      const T* x = data_.Row<T>(perm_[i]);
      for (uint32_t j = 0; j < d_; ++j) sum[j] += static_cast<double>(x[j]);
    }
    std::vector<float> mean(d_);
    for (uint32_t j = 0; j < d_; ++j) mean[j] = static_cast<float>(sum[j] / static_cast<double>(n));
    uint64_t best = b;
    float best_d = L2SqrMixed(data_.Row<T>(perm_[b]), mean.data(), d_);
    for (uint64_t i = b + 1; i < e; ++i) {
      float dd = L2SqrMixed(data_.Row<T>(perm_[i]), mean.data(), d_);
      if (dd < best_d) {
        best_d = dd;
        best = i;
      }
    }
    return perm_[best];
  }

  void Split(const Node& nd, Output* out) {
    const uint64_t n = nd.e - nd.b;
    if (n <= leaf_max_) {
      AddLeaf(nd.b, nd.e, out);
      return;
    }
    uint32_t k = static_cast<uint32_t>(std::llround(static_cast<double>(n) / target_));
    k = std::max<uint32_t>(2, std::min<uint32_t>(k, p_.branch));

    // Training sample (Floyd's algorithm for distinct positions).
    uint64_t s = std::max<uint64_t>(static_cast<uint64_t>(k) * p_.sample_per_centroid, 1024);
    s = std::min<uint64_t>({s, n, static_cast<uint64_t>(p_.max_sample)});
    std::mt19937_64 rng(p_.seed ^ (nd.b * 0x9E3779B97F4A7C15ull) ^ (n << 17));
    std::vector<float> sample(s * d_);
    if (s == n) {
      for (uint64_t i = 0; i < n; ++i) ToFloat(data_.Row<T>(perm_[nd.b + i]), &sample[i * d_], d_);
    } else {
      std::unordered_set<uint64_t> chosen;
      chosen.reserve(s * 2);
      for (uint64_t j = n - s; j < n; ++j) {
        std::uniform_int_distribution<uint64_t> pick(0, j);
        uint64_t t = pick(rng);
        if (!chosen.insert(t).second) chosen.insert(j);
      }
      uint64_t i = 0;
      for (uint64_t pos : chosen) ToFloat(data_.Row<T>(perm_[nd.b + pos]), &sample[(i++) * d_], d_);
    }

    KMeansParams kp;
    kp.k = k;
    kp.iters = p_.kmeans_iters;
    kp.seed = rng();
    kp.balance = p_.balance;
    std::vector<float> cent = KMeansTrain(sample.data(), s, d_, kp);
    const uint32_t kk = static_cast<uint32_t>(cent.size() / d_);

    std::vector<uint32_t> label(n);
#pragma omp parallel for schedule(static) if (n >= 20000)
    for (uint64_t i = 0; i < n; ++i)
      label[i] = NearestCentroid(data_.Row<T>(perm_[nd.b + i]), cent.data(), kk, d_);
    std::vector<uint64_t> cnt(kk, 0);
    for (uint32_t l : label) cnt[l]++;
    MergeTinyChildren(cent, kk, &label, &cnt);

    if (*std::max_element(cnt.begin(), cnt.end()) == n) {
      // k-means could not separate the points (e.g. many duplicates): cut into chunks.
      const uint64_t cs = static_cast<uint64_t>(std::ceil(target_));
      for (uint64_t b = nd.b; b < nd.e; b += cs) AddLeaf(b, std::min(nd.e, b + cs), out);
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
        AddLeaf(ch.b, ch.e, out);
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
  double target_ = 10;
  uint64_t leaf_max_ = 15;
  std::vector<uint32_t> perm_;
};

}  // namespace

std::vector<uint32_t> HierarchicalBalancedClustering(const VectorFile& data,
                                                     const ClusteringParams& p,
                                                     ClusteringStats* stats) {
  FUSION_CHECK(data.size() <= 0xFFFFFFFEull, "at most 2^32-2 vectors are supported");
  return DispatchDType(data.dtype(), [&](auto tag) {
    using T = decltype(tag);
    HierClusterer<T> hc(data, p);
    return hc.Run(stats);
  });
}

}  // namespace fusion
