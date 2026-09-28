// Raw vectors held by one rtier data node (its SSD tier), as a sparse subset of the index's page
// file -- the decided part of design question U1.
//
// Every vector has a canonical location in the page file (fusion/layout.h, RawLayout), carried
// by the posting lists next to its ID (fusion/partition.h). A node keeps the vectors it holds in
// a local file laid out like the whole page file -- same offsets, sparse on disk -- and one
// presence bit per location. A page may also have room for vectors of lists the node does not
// own (the bucket layout packs list tails together); those slots stay empty and are never read.
//
// Vectors arrive three ways: copied from the index's page file when a partition is bootstrapped
// (the initial deployment), streamed by the agent after a migration (RAW_PUT), or fetched on
// demand from the old owner when a query needs one before the stream brought it (NodeEngine,
// U9). Nothing is dropped: the vectors of partitions that move away stay as a cache, as PQ codes
// do (the dataset is static; U14), so a partition that comes back moves no vector, and a node
// that gave a partition away can still serve its vectors to the new owner.
//
// Writes go through a buffered descriptor; re-ranking reads whole pages through pages() (with
// O_DIRECT when the file system supports it). A presence bit is set only after the write of its
// vector returned, and a direct read first writes back the dirty page-cache range it covers, so
// a reader that saw the bit reads the vector. Readers ignore the slots whose bit is clear.
#pragma once

#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "fusion/common.h"
#include "fusion/layout.h"
#include "fusion/page_reader.h"

namespace fusion {

// A raw vector the operation needs is not on this node (and could not be fetched).
class RawMissingError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Where installed vectors came from (statistics only).
enum class RawOrigin { kIndex, kStreamed, kFetched };

struct RawStats {
  uint32_t vec_bytes = 0;
  uint64_t locations = 0;   // the page file's address space: num_pages x vectors_per_page
  uint64_t present = 0;     // vectors held now
  uint64_t pending = 0;     // named by loaded partitions, not here yet (NodeEngine)
  uint64_t from_index = 0;  // cumulative: copied from the index's page file (bootstrap)
  uint64_t streamed = 0;    // cumulative: installed by the agent's stream (RAW_PUT)
  uint64_t fetched = 0;     // cumulative: fetched on demand from a peer
  uint64_t fetches = 0;     // cumulative: on-demand fetch round trips
  uint64_t skipped = 0;     // cumulative: offered but already here
  uint64_t served = 0;      // cumulative: copied out for peers (RAW_GET)
};

class RawStore {
 public:
  // path: the local file, created or truncated (presence is not persisted: a node starts
  // empty); "" = an anonymous temporary file. direct: O_DIRECT page reads when possible.
  RawStore(const std::string& path, const RawLayout& layout, bool direct);
  ~RawStore();
  RawStore(const RawStore&) = delete;
  RawStore& operator=(const RawStore&) = delete;

  const RawLayout& layout() const { return layout_; }
  // The local file, for page readers (fusion/page_reader.h). Only present slots may be used.
  const PageFile& pages() const { return *pages_; }
  const std::string& path() const { return path_; }

  bool Present(uint32_t loc) const {
    return (bits_[loc >> 6].load(std::memory_order_acquire) >> (loc & 63)) & 1;
  }
  // Locations among locs[0..n) that are not here: sorted, without duplicates.
  std::vector<uint32_t> Missing(const uint32_t* locs, size_t n) const;
  // Writes n vectors (vec_bytes each, in the order of locs) and marks them present; locations
  // already here are skipped. Returns the number installed.
  size_t Put(const uint32_t* locs, size_t n, const uint8_t* vecs, RawOrigin origin);
  // Copies n vectors out (vec_bytes each, in the order of locs). Throws RawMissingError, having
  // copied nothing, if one is not here.
  void Get(const uint32_t* locs, size_t n, uint8_t* out);
  void CountFetch() { fetches_.fetch_add(1, std::memory_order_relaxed); }

  RawStats Stats() const;

 private:
  void CheckRange(const uint32_t* locs, size_t n) const;

  const RawLayout layout_;
  std::string path_;
  int fd_ = -1;  // buffered, read-write
  std::unique_ptr<PageFile> pages_;
  std::unique_ptr<std::atomic<uint64_t>[]> bits_;  // one bit per location
  std::atomic<uint64_t> present_{0}, from_index_{0}, streamed_{0}, fetched_{0}, fetches_{0},
      skipped_{0}, served_{0};
};

}  // namespace fusion
