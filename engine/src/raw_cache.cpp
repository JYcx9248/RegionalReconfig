#include "fusion/raw_cache.h"

#include <algorithm>
#include <cstring>

namespace fusion {
namespace {

// Fibonacci hashing: pages next to each other (a list's vectors) spread over shards and stripes.
uint64_t Mix(uint32_t page) { return static_cast<uint64_t>(page) * 0x9E3779B97F4A7C15ull; }

class Cached final : public PageReader {
 public:
  Cached(std::unique_ptr<PageReader> r, RawCache* c) : r_(std::move(r)), c_(c) {}
  void Read(const uint32_t* pages, uint8_t* const* bufs, uint32_t count) override {
    pages_.clear();
    bufs_.clear();
    tickets_.clear();
    for (uint32_t i = 0; i < count; ++i) {
      if (c_->Lookup(pages[i], bufs[i], ReadClass::kOwn)) continue;
      pages_.push_back(pages[i]);
      bufs_.push_back(bufs[i]);
      tickets_.push_back(c_->Version(pages[i]));  // before the read
    }
    if (pages_.empty()) return;
    r_->Read(pages_.data(), bufs_.data(), static_cast<uint32_t>(pages_.size()));
    for (size_t j = 0; j < pages_.size(); ++j) c_->Insert(pages_[j], bufs_[j], tickets_[j]);
  }
  IoBackend backend() const override { return r_->backend(); }

 private:
  std::unique_ptr<PageReader> r_;
  RawCache* c_;
  std::vector<uint32_t> pages_;
  std::vector<uint8_t*> bufs_;
  std::vector<uint64_t> tickets_;
};

}  // namespace

RawCache::RawCache(uint64_t bytes, uint32_t page_size) : page_size_(page_size) {
  frames_ = page_size ? bytes / page_size : 0;
  if (!frames_) return;
  // Up to 64 shards, each with its own lock, as long as each gets at least 64 frames (a small
  // cache keeps one clock).
  while (shard_bits_ < 6 && (frames_ >> (shard_bits_ + 1)) >= 64) ++shard_bits_;
  const uint64_t n = uint64_t{1} << shard_bits_;
  for (uint64_t s = 0; s < n; ++s) {
    auto sh = std::make_unique<Shard>();
    sh->cap = static_cast<uint32_t>(frames_ / n + (s < frames_ % n ? 1 : 0));
    shards_.push_back(std::move(sh));
  }
  versions_ = std::make_unique<std::atomic<uint64_t>[]>(size_t{1} << kVersionBits);
  for (size_t i = 0; i < (size_t{1} << kVersionBits); ++i) versions_[i].store(0, std::memory_order_relaxed);
}

RawCache::Shard& RawCache::ShardOf(uint32_t page) {
  return *shards_[shard_bits_ ? Mix(page) >> (64 - shard_bits_) : 0];
}

std::atomic<uint64_t>& RawCache::VersionOf(uint32_t page) const {
  return versions_[(Mix(page) >> 16) & ((uint64_t{1} << kVersionBits) - 1)];
}

bool RawCache::Lookup(uint32_t page, uint8_t* out, ReadClass c) {
  if (!frames_) return false;
  const bool peer = c == ReadClass::kPeer;
  Shard& sh = ShardOf(page);
  {
    std::lock_guard<std::mutex> l(sh.mu);
    const auto it = sh.index.find(page);
    if (it != sh.index.end()) {
      sh.ref[it->second] = 1;
      std::memcpy(out, sh.data.get() + static_cast<size_t>(it->second) * page_size_, page_size_);
      (peer ? peer_hits_ : hits_).fetch_add(1, std::memory_order_relaxed);
      return true;
    }
  }
  (peer ? peer_misses_ : misses_).fetch_add(1, std::memory_order_relaxed);
  return false;
}

uint64_t RawCache::Version(uint32_t page) const {
  return frames_ ? VersionOf(page).load(std::memory_order_acquire) : 0;
}

void RawCache::Insert(uint32_t page, const uint8_t* data, uint64_t version) {
  if (!frames_) return;
  Shard& sh = ShardOf(page);
  std::lock_guard<std::mutex> l(sh.mu);
  // Under the shard's lock, as Invalidate bumps it: a write that ended after the ticket voids it.
  if (VersionOf(page).load(std::memory_order_relaxed) != version) return;
  uint32_t f;
  const auto it = sh.index.find(page);
  if (it != sh.index.end()) {
    f = it->second;  // another reader kept it meanwhile: the same bytes
  } else {
    if (!sh.data) sh.data.reset(new uint8_t[static_cast<size_t>(sh.cap) * page_size_]);
    if (!sh.free.empty()) {
      f = sh.free.back();
      sh.free.pop_back();
      held_.fetch_add(1, std::memory_order_relaxed);
    } else if (sh.used < sh.cap) {
      f = sh.used++;
      sh.page_of.push_back(page);
      sh.ref.push_back(0);
      held_.fetch_add(1, std::memory_order_relaxed);
    } else {
      // CLOCK: the first frame whose bit is clear, clearing the bits passed over. A frame an
      // install emptied is on the free list, never here.
      for (;;) {
        f = sh.hand;
        sh.hand = (sh.hand + 1) % sh.used;
        if (!sh.ref[f]) break;
        sh.ref[f] = 0;
      }
      sh.index.erase(sh.page_of[f]);
      evicted_.fetch_add(1, std::memory_order_relaxed);
    }
    sh.page_of[f] = page;
    sh.ref[f] = 0;
    sh.index.emplace(page, f);
  }
  std::memcpy(sh.data.get() + static_cast<size_t>(f) * page_size_, data, page_size_);
}

void RawCache::Invalidate(uint32_t page) {
  if (!frames_) return;
  Shard& sh = ShardOf(page);
  std::lock_guard<std::mutex> l(sh.mu);
  VersionOf(page).fetch_add(1, std::memory_order_release);
  const auto it = sh.index.find(page);
  if (it == sh.index.end()) return;
  const uint32_t f = it->second;
  sh.index.erase(it);
  sh.ref[f] = 0;
  sh.page_of[f] = UINT32_MAX;
  sh.free.push_back(f);
  held_.fetch_sub(1, std::memory_order_relaxed);
  dropped_.fetch_add(1, std::memory_order_relaxed);
}

RawCacheStats RawCache::Stats() const {
  RawCacheStats s;
  auto get = [](const std::atomic<uint64_t>& a) { return a.load(std::memory_order_relaxed); };
  s.capacity = frames_ * page_size_;
  s.bytes = get(held_) * page_size_;
  s.hits = get(hits_);
  s.misses = get(misses_);
  s.peer_hits = get(peer_hits_);
  s.peer_misses = get(peer_misses_);
  s.evicted = get(evicted_);
  s.dropped = get(dropped_);
  return s;
}

std::unique_ptr<PageReader> CachedPageReader(std::unique_ptr<PageReader> reader, RawCache* cache) {
  if (!cache || !cache->enabled()) return reader;
  return std::make_unique<Cached>(std::move(reader), cache);
}

}  // namespace fusion
