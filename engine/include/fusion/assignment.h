// Navigation graph over the heads and SPANN-style replicated assignment of every vector to
// posting lists (paper Section 3.1, Eq. 2).
#pragma once

#include <vector>

#include "fusion/dataset.h"
#include "fusion/hnsw_space.h"
#include "fusion/index_meta.h"

namespace fusion {

struct GraphParams {
  uint32_t m = 32;                 // hnswlib M (level-0 degree is 2*M = 64, as in the paper)
  uint32_t ef_construction = 200;
  uint64_t seed = 100;
};

// Builds the in-memory navigation graph over head vectors (K x vec_bytes, native dtype).
std::unique_ptr<HnswIndex> BuildNavigationGraph(hnswlib::SpaceInterface<float>* space,
                                                const uint8_t* heads, uint32_t num_heads,
                                                size_t vec_bytes, const GraphParams& p);

// Defaults follow the paper: Eq. 2 with at most 8 replicas, using SPANN's published closure
// factor (eps1 = 10), which lets nearly every vector reach the 8-replica cap -- consistent with
// the paper's statement that replication expands the posting lists about 8x. SPANN's optional
// RNG pruning ("representative replication") is available but off by default because the
// FusionANNS paper does not mention it; it cuts replication to ~2x on our test data.
struct AssignParams {
  uint32_t replicas = 8;      // max posting lists per vector (paper: 8)
  float closure_eps = 10.f;   // Eq. 2: v -> C_i iff Dist(v,C_i) <= (1+eps) * Dist(v,C_1)
  bool rng = false;           // SPANN's RNG rule: skip C_i if it is closer to an already
  float rng_factor = 1.0f;    //   chosen head than to v (rng_factor * Dist(C_i,C_j) <= Dist(v,C_i))
  uint32_t candidates = 64;   // nearest heads examined per vector
  uint32_t ef = 128;          // graph search width during assignment
};

struct AssignStats {
  double avg_replicas = 0;
  uint64_t total_entries = 0;
  uint32_t min_list = 0, max_list = 0;
  double mean_list = 0;
  uint32_t empty_lists = 0;
};

// Searches every vector's nearest heads on the graph and assigns it to up to `replicas`
// posting lists. Also returns each vector's primary (nearest) head, which drives the SSD
// layout. `heads` holds the head vectors (K x vec_bytes).
void AssignVectors(const VectorFile& data, HnswIndex* graph, const uint8_t* heads,
                   uint32_t num_heads, const AssignParams& p, PostingLists* lists,
                   std::vector<uint32_t>* primary, AssignStats* stats);

}  // namespace fusion
