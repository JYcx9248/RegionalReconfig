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
// Re-ranking reads whole pages through pages(), with O_DIRECT when the file system supports it.
// Installs run one at a time. Fetched vectors -- a few scattered vectors per page, installed by
// the engine's background writer while queries read the same file -- are written as whole pages
// with O_DIRECT: a page that already holds vectors is read back first (batched), so its present
// slots are written back unchanged; the others start empty. Buffered writes of single vectors
// would go through the page cache: a partial write of a page that is not cached reads it first,
// synchronously and under the file's inode lock, which the direct reads of the same file then
// wait for, and it leaves dirty pages that a direct read must write back first -- a writer that
// slows every query. Bulk installs (bootstrap, streams: sorted batches that fill a page over
// several requests) are buffered writes of the vectors alone, which the page cache combines per
// page. A presence bit is set only after the write of its vector completed (and a direct read
// writes back the dirty range it covers first), so a reader that saw the bit reads the vector;
// a reader of a page being rewritten gets its present slots either way. Readers ignore the slots
// whose bit is clear. Without O_DIRECT (e.g. tmpfs) every install is buffered.
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
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
  uint64_t present = 0;     // vectors on the SSD now
  uint64_t cached = 0;      // fetched, in memory until the writer installs them (NodeEngine)
  uint64_t pending = 0;     // named by resident partitions, neither here nor fetched (NodeEngine;
                            // with the lazy protocol, what queries have not needed so far)
  uint64_t from_index = 0;  // cumulative: copied from the index's page file (bootstrap)
  uint64_t streamed = 0;    // cumulative: installed by the agent's stream (RAW_PUT)
  uint64_t fetched = 0;     // cumulative: installed after an on-demand fetch from a peer (the
                            // ones still cached are not counted yet)
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
  // already here are skipped. Returns the number installed. Serialized with other Puts.
  size_t Put(const uint32_t* locs, size_t n, const uint8_t* vecs, RawOrigin origin);
  // Locations set in `mask` (one bit per location, 64 per word, as many words as there are
  // locations / 64 rounded up) whose vector is not here.
  uint64_t CountAbsent(const std::vector<uint64_t>& mask) const;
  // Copies n vectors out (vec_bytes each, in the order of locs). Throws RawMissingError, having
  // copied nothing, if one is not here.
  void Get(const uint32_t* locs, size_t n, uint8_t* out);
  void CountFetch() { fetches_.fetch_add(1, std::memory_order_relaxed); }
  // Vectors a node served to peers from memory (fetched, not on the SSD yet), counted like Get's.
  void CountServed(size_t n) { served_.fetch_add(n, std::memory_order_relaxed); }

  RawStats Stats() const;

 private:
  void CheckRange(const uint32_t* locs, size_t n) const;
  void PutPages(const uint32_t* locs, const uint8_t* vecs, const std::vector<size_t>& order);
  void PutBuffered(const uint32_t* locs, const uint8_t* vecs, const std::vector<size_t>& order);

  const RawLayout layout_;
  std::string path_;
  int fd_ = -1;   // buffered, read-write (Get; installs without O_DIRECT)
  int dfd_ = -1;  // O_DIRECT, read-write (installs), -1 if unsupported
  std::unique_ptr<PageFile> pages_;
  std::mutex put_mu_;  // one installer at a time: a page is rewritten whole
  class DirectIo;
  std::unique_ptr<DirectIo> io_;  // under put_mu_
  uint8_t* chunk_ = nullptr;      // under put_mu_: kChunkPages page-aligned pages
  std::unique_ptr<std::atomic<uint64_t>[]> bits_;  // one bit per location
  std::atomic<uint64_t> present_{0}, from_index_{0}, streamed_{0}, fetched_{0}, fetches_{0},
      skipped_{0}, served_{0};
};

}  // namespace fusion
