// Offline construction of the multi-tiered index (paper Section 3.1, Figure 6).
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "fusion/assignment.h"
#include "fusion/clustering.h"
#include "fusion/common.h"

namespace fusion {

// The next generation of an index over a data version after update batches (U14 full rebuild):
// re-clustered over the version's vectors, it keeps what the nodes hold of the generation they
// serve -- every vector's ID and raw-vector location, and the PQ codebook -- so that their codes
// and raw vectors stay valid. base_path then holds the version's vectors, one row each; ids[r] is
// row r's vector ID, ascending; locs[r] its raw-vector location if it was inserted, kNoLoc if
// from_index's layout has it. postings.bin, heads.bin and primary.bin name vector IDs, and
// pq_codes.bin and the layout maps are indexed by vector ID up to id_end. The page files are
// from_index's: they hold its vectors only, and a node never reads the inserted ones from them
// (it holds them, or fetches them from a holder), so no node can bootstrap on this generation.
struct VersionParams {
  static constexpr uint32_t kNoLoc = 0xFFFFFFFFu;
  std::string from_index;     // built from data: its page files hold every vector of it
  std::vector<uint32_t> ids;  // per row of base_path
  std::vector<uint32_t> locs;
  uint64_t id_end = 0;        // meta num_vectors: the IDs handed out
  std::string record;         // version.json: what the version is made of, for the runner's check
};

struct BuildParams {
  std::string base_path;
  std::shared_ptr<const VersionParams> version;  // null: an index of base_path's vectors, ID = row
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
  // Base vectors held in memory at once: the clustering's batches, the page file's windows. The
  // base is read in passes, never mapped, so it can be larger than memory. The index does not
  // depend on it, except for the order of its heads.
  uint64_t memory_bytes = uint64_t{16} << 30;
  bool overwrite = false;         // rebuild stages whose outputs already exist
};

// Runs every build stage. Stages whose outputs already exist are skipped unless
// `overwrite` is set, so an interrupted build can be resumed.
void BuildIndex(const BuildParams& p);

}  // namespace fusion
