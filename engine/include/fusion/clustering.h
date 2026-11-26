// Hierarchical balanced clustering (paper Section 3.1, following SPANN): recursively split
// the dataset with (balance-penalized) k-means until every cluster holds about
// 1/centroid_ratio vectors. Each leaf is represented by its "head": the member closest to
// the leaf mean. Heads are real data vectors, so the navigation graph can store them in the
// dataset's native type.
//
// And the flat k-means of Quake's index (U14: an update strategy runs on its own paper's index),
// whose heads are the clusters' means instead.
#pragma once

#include <string>
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
  // "balanced": the hierarchical clustering above; "kmeans": FlatKMeans, its centroids the heads.
  std::string method = "balanced";
  uint32_t nlist = 0;           // kmeans: lists (0: round(sqrt(N)), as Quake's evaluation sizes them)
  uint32_t flat_iters = 5;      // kmeans: Lloyd iterations (Quake's niter)
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

// Quake's build (its clustering.cpp, kmeans_cpu: Faiss's Clustering with niter): p.nlist (or
// round(sqrt(N))) centroids, trained on a random sample of 256 vectors per centroid (Faiss's
// max_points_per_centroid) -- fewer if memory_bytes cannot hold them as floats -- seeded with
// distinct sample points and run for p.flat_iters iterations; an empty cluster takes half of a
// big one (Faiss's split). Returns nlist x dim floats: no member stands for a cluster, its mean
// does.
std::vector<float> FlatKMeans(const VectorFile& data, const ClusteringParams& p, uint64_t memory_bytes,
                              ClusteringStats* stats);

// The number of lists FlatKMeans makes of n vectors.
uint32_t FlatKMeansLists(const ClusteringParams& p, uint64_t n);

}  // namespace fusion
