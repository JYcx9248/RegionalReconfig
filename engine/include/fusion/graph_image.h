// The navigation graph's bytes in hnswlib's index format, from and to memory (U16).
//
// A new entry needs the graph that the entries hold in memory, and needs it only in memory:
// queries search it there (NAVIGATE). So a graph transfer moves the bytes of hnswlib's index
// format -- the vendored v0.8.0's saveIndex / loadIndex, kept unmodified -- from one node's
// memory to another's, and neither end writes them to a file. The source answers for any range
// of the format from its loaded graph (GraphImage), without serializing it first, so it holds no
// second copy. The destination builds its graph from the bytes in order, as they arrive
// (GraphLoader), so it holds no second copy either. A graph changed by major epochs (U14) is sent
// at its merged version: the patches it took are in its bytes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "fusion/hnsw_space.h"

namespace fusion {

// The layout of a loaded graph's image: what saveIndex would write for it, byte for byte.
// Building it reads each element's level, not its bytes; Read copies the bytes from the graph.
class GraphImage {
 public:
  explicit GraphImage(const HnswIndex& g);
  uint64_t size() const { return size_; }
  // Bytes [off, off + n) of g's image. g must be the graph the layout was made of, unchanged
  // since (the caller pins it against in-place changes).
  void Read(const HnswIndex& g, uint64_t off, size_t n, uint8_t* out) const;

 private:
  std::vector<uint8_t> header_;
  uint64_t level0_ = 0;            // the level-0 records' bytes, after the header
  std::vector<uint64_t> link_at_;  // where element i's link record starts; [count] = size_
  uint64_t size_ = 0;
};

// A graph built from the bytes of its image, as hnswlib's loadIndex builds it from the file,
// with room for max_elements (more if the graph holds more, as loadIndex does).
class GraphLoader {
 public:
  GraphLoader(hnswlib::SpaceInterface<float>* space, size_t max_elements);
  // The next n bytes of the image.
  void Append(const uint8_t* p, size_t n);
  uint64_t received() const { return received_; }
  bool complete() const { return state_ == State::kDone; }
  // The graph, once complete.
  std::unique_ptr<HnswIndex> Take();

 private:
  enum class State { kHeader, kLevel0, kLinkSize, kLinks, kDone };
  void Begin();          // the header is in: allocate as loadIndex does
  void EndElement();     // element elem_'s link record is in
  hnswlib::SpaceInterface<float>* space_;
  size_t room_;
  std::unique_ptr<HnswIndex> g_;
  State state_ = State::kHeader;
  std::vector<uint8_t> header_;
  size_t count_ = 0;               // elements in the image
  uint64_t level0_ = 0, level0_in_ = 0;
  size_t elem_ = 0;                // the element whose link record comes next
  uint8_t size_[4] = {0, 0, 0, 0};
  size_t size_in_ = 0;
  uint32_t links_ = 0, links_in_ = 0;
  uint64_t received_ = 0;
};

}  // namespace fusion
