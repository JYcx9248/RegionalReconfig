// A read cache of raw-vector pages in CPU memory, in front of a data node's raw store
// (fusion/raw_store.h), which it treats as a black box: every read of the store's pages looks
// here first -- re-ranking's (RERANK, SEARCH_LOCAL: CachedPageReader) and those served to peers
// (RAW_GET: a query's fetch, a stream's or a drain's pull) -- and a page read from the store is
// kept. A hit costs no device read, so the read budget never sees it (fusion/read_budget.h).
//
// Bounded in bytes (NodeOptions::raw_cache_bytes; rtier_node --raw-cache-share, 5% of the memory
// the node may use by default), whole pages, CLOCK eviction. A page enters with its reference bit
// clear and gets it on a hit, so pages read once -- a stream's, or a worst-case query's -- are
// the first to go, and the pages queries come back to stay.
//
// Installs change pages: RawStore::Put drops the pages it wrote here after its writes return and
// before it sets their presence bits (Invalidate). A read that started before such a write could
// still put back what it read, so a reader takes a ticket (Version) before it reads a page from
// the store and keeps the page only if no write of it ended since (Insert): a reader that saw a
// vector's bit never gets a copy of its page from before that vector was written.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "fusion/page_reader.h"
#include "fusion/read_budget.h"

namespace fusion {

struct RawCacheStats {
  uint64_t capacity = 0;                    // bytes it may hold; 0: off
  uint64_t bytes = 0;                       // now: the pages it holds
  uint64_t hits = 0, misses = 0;            // cumulative: the node's own reads (re-ranking), in pages
  uint64_t peer_hits = 0, peer_misses = 0;  // cumulative: reads served to peers (RAW_GET), in pages
  uint64_t evicted = 0;                     // cumulative: pages evicted to make room
  uint64_t dropped = 0;                     // cumulative: pages dropped because an install wrote them
};

class RawCache {
 public:
  // bytes: what it may hold (rounded down to whole pages; less than a page: off).
  RawCache(uint64_t bytes, uint32_t page_size);
  RawCache(const RawCache&) = delete;
  RawCache& operator=(const RawCache&) = delete;

  bool enabled() const { return frames_ > 0; }
  uint32_t page_size() const { return page_size_; }
  // Copies page p into out (page_size bytes) if it is held; counts a hit or a miss for class c.
  bool Lookup(uint32_t page, uint8_t* out, ReadClass c);
  // The ticket of a reader about to read page p from the store.
  uint64_t Version(uint32_t page) const;
  // Keeps page p (page_size bytes at data), read with that ticket, unless a write of p ended
  // since the ticket was taken.
  void Insert(uint32_t page, const uint8_t* data, uint64_t version);
  // A write of page p ended: drop it, and void the tickets taken before.
  void Invalidate(uint32_t page);
  RawCacheStats Stats() const;

 private:
  struct Shard {
    std::mutex mu;
    std::unordered_map<uint32_t, uint32_t> index;  // page -> frame
    std::vector<uint32_t> page_of;                 // frame -> page
    std::vector<uint8_t> ref;                      // frame -> reference bit
    std::vector<uint32_t> free;                    // frames an install emptied
    std::unique_ptr<uint8_t[]> data;               // cap frames, allocated at the first insert
    uint32_t cap = 0, used = 0, hand = 0;          // frames: at most, handed out so far, the clock
  };
  Shard& ShardOf(uint32_t page);
  std::atomic<uint64_t>& VersionOf(uint32_t page) const;

  const uint32_t page_size_;
  uint64_t frames_ = 0;
  uint32_t shard_bits_ = 0;
  std::vector<std::unique_ptr<Shard>> shards_;
  // Write versions, striped by page: a write of any page of a stripe voids the stripe's tickets,
  // which costs a reader at most a page it does not keep.
  static constexpr uint32_t kVersionBits = 16;
  std::unique_ptr<std::atomic<uint64_t>[]> versions_;
  std::atomic<uint64_t> held_{0}, hits_{0}, misses_{0}, peer_hits_{0}, peer_misses_{0}, evicted_{0}, dropped_{0};
};

// A page reader that serves what the cache holds and reads the rest through `reader`, keeping
// what it read (the node's own reads: re-ranking). One per worker thread, as `reader`.
std::unique_ptr<PageReader> CachedPageReader(std::unique_ptr<PageReader> reader, RawCache* cache);

}  // namespace fusion
