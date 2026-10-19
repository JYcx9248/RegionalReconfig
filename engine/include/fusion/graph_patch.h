// Patches to the navigation graph (U14 major epochs).
//
// A major epoch splits and merges posting lists: their heads change, so the graph over the
// heads changes too. Every entry holds the whole graph and holds it once -- no node keeps two
// versions of it -- so the new version is not loaded next to the old one. Instead, the update
// controller's router applies the head changes to its own copy (the authoritative one: new
// heads inserted, removed ones marked deleted) and exports the bytes that changed: a patch. An
// entry stages the patch next to its graph without touching the graph. Until the patch is
// merged, queries of the epoch before it search the graph alone, and queries of the epoch after
// it search the graph through the patch: an element the patch changed is read from the patch,
// every other one from the graph, so the search is the one on the patched graph, step by step
// (SearchPatched). After the grace period of the epoch before it, the patch is merged into the
// graph in place (ApplyPatch): the bytes it writes are those of the changed elements, which no
// running search reads from the graph -- searches of the epoch after it read them from the
// patch -- so the merge needs no pause.
//
// A patch holds, for every changed element, its whole level-0 record (links, vector, label, the
// deleted mark) and its upper-level link lists, plus the graph's globals after the change
// (element count, entry point, top level, deleted count). New elements take the internal IDs
// after the graph's last one, inside the capacity left at load time (graph headroom).
#pragma once

#include <cstddef>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "fusion/hnsw_space.h"

namespace fusion {

struct GraphPatch {
  uint32_t base_version = 0;  // the graph version it applies to
  uint32_t version = 0;       // the version it makes (base_version + 1)
  uint64_t count = 0;         // elements after it
  uint64_t num_deleted = 0;   // elements marked deleted after it
  uint32_t entry = 0;         // entry point after it
  int32_t maxlevel = 0;       // top level after it
  uint64_t data_bytes = 0;    // one level-0 record (the graph's size_data_per_element_)
  uint64_t links_bytes = 0;   // one upper-level link list (size_links_per_element_)
  std::vector<uint32_t> ids;       // the changed elements' internal IDs, ascending
  std::vector<int32_t> levels;     // their top levels
  std::vector<uint8_t> level0;     // their level-0 records, data_bytes each
  std::vector<uint64_t> upper_at;  // ids.size() + 1 offsets into upper
  std::vector<uint8_t> upper;      // their upper-level link lists, levels[i] * links_bytes each

  // Index of `id` in ids, or SIZE_MAX.
  size_t Find(uint32_t id) const;
  // The patch file / wire format, with a CRC-32.
  std::vector<uint8_t> Serialize() const;
  static GraphPatch Parse(const uint8_t* data, size_t n);
};

// The current state of the elements `ids` of g (and g's globals): the patch that turns a graph
// at base_version, equal to g except at those elements, into g.
GraphPatch CapturePatch(const HnswIndex& g, std::vector<uint32_t> ids, uint32_t base_version,
                        uint32_t version);

// A patch staged next to a graph: membership of the changed elements, one bit per element of
// the graph's capacity.
struct StagedPatch {
  GraphPatch patch;
  std::vector<uint64_t> changed;
  bool Changed(uint32_t id) const { return (changed[id >> 6] >> (id & 63)) & 1; }
};

// Checks that p applies to g -- the same element layout, every new element inside g's capacity
// and in the patch, existing elements at their levels -- and indexes it.
StagedPatch StagePatch(const HnswIndex& g, GraphPatch p);

// hnswlib's searchKnn(q, k) on the graph g would become with the patch applied, without
// changing g: the same traversal, step by step, so the same result.
std::priority_queue<std::pair<float, hnswlib::labeltype>> SearchPatched(const HnswIndex& g,
                                                                       const StagedPatch& sp,
                                                                       const void* q, size_t k);

// Writes the patch into g in place. Safe while searches through the same patch run: it writes
// only the changed elements, which those read from the patch, and the globals last. No search
// of g alone may run meanwhile.
void ApplyPatch(HnswIndex& g, const StagedPatch& sp);

}  // namespace fusion
