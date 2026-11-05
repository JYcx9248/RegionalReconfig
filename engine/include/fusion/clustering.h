// Hierarchical balanced clustering (paper Section 3.1, following SPANN): recursively split
// the dataset with (balance-penalized) k-means until every cluster holds about
// 1/centroid_ratio vectors. Each leaf is represented by its "head": the member closest to
// the leaf mean. Heads are real data vectors, so the navigation graph can store them in the
// dataset's native type.
#pragma once

#include <vector>

#include "fusion/dataset.h"

namespace fusion {

struct ClusteringParams {
  double centroid_ratio = 0.1;  // number of posting lists ~= ratio * N (paper: ~10%)
  uint32_t branch = 32;         // max children per split
  uint32_t kmeans_iters = 10;
  float balance = 0.5f;         // k-means size-balance penalty (0 = plain k-means)
  uint32_t sample_per_centroid = 64;
  uint32_t max_sample = 65536;  // training sample cap per split
  float leaf_slack = 1.5f;      // nodes with <= slack * target members become leaves
  uint64_t seed = 42;
};

struct ClusteringStats {
  uint32_t num_leaves = 0;
  uint32_t levels = 0;
  uint32_t min_leaf = 0, max_leaf = 0;
  double mean_leaf = 0;
};

// Returns the data IDs of the heads (one per leaf). Holds about `memory_bytes` of base vectors at
// once, plus 12 bytes per vector. A node whose rows do not fit is split with its rows streamed
// from the file. The other nodes are clustered in batches that fit, each batch's rows read in one
// pass. A node's rows spread over the whole file, so each streamed split and each batch reads
// about all of it: the bigger the budget, the fewer passes. The heads do not depend on
// `memory_bytes`, but their order does.
std::vector<uint32_t> HierarchicalBalancedClustering(const VectorFile& data,
                                                     const ClusteringParams& p,
                                                     uint64_t memory_bytes,
                                                     ClusteringStats* stats);

}  // namespace fusion
