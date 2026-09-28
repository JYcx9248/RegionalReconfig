// Offline construction of the multi-tiered index (paper Section 3.1, Figure 6).
#pragma once

#include <optional>
#include <string>

#include "fusion/assignment.h"
#include "fusion/clustering.h"
#include "fusion/common.h"

namespace fusion {

struct BuildParams {
  std::string base_path;
  std::optional<DType> dtype;  // default: from the file extension
  uint64_t max_n = 0;          // 0 = whole file
  std::string out_dir;

  ClusteringParams clustering;
  GraphParams graph;
  AssignParams assign;

  uint32_t pq_m = 0;            // 0 = DefaultPQM(dim)
  uint32_t pq_train_n = 100000; // training sample size
  uint32_t pq_iters = 20;

  std::string layout = "bucket";  // "bucket", "id" or "both"
  uint64_t seed = 42;
  bool overwrite = false;         // rebuild stages whose outputs already exist
};

// Runs every build stage. Stages whose outputs already exist are skipped unless
// `overwrite` is set, so an interrupted build can be resumed.
void BuildIndex(const BuildParams& p);

}  // namespace fusion
