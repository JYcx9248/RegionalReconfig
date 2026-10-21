// The reply of a streamed listing (PQ_MISSING, RAW_MISSING, RAW_HELD_BY; wire.h).
#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "wire.h"

namespace rtier {

// A listing's entries go out in frames of about max (kStreamBatch). A frame goes when it is full
// -- written by the listing's own thread, which waits for the write, so the requester's reading
// paces the listing -- or when its first entry has waited kStreamLingerMs, written by the
// streamer's thread, so that a slow listing (a large segment read from disk) still feeds the
// requester; the last one is the reply itself (Last). Without a writer (Handle called directly)
// everything goes in the last one. Batch holds the entries: size(), Encode(BodyWriter*, bool
// last), Clear().
template <class Batch>
class Streamer {
 public:
  using Write = std::function<void(const Frame&)>;
  Streamer(const Frame& req, Write write, size_t max, Batch batch = Batch())
      : type_(req.type), req_id_(req.req_id), epoch_(req.epoch), write_(std::move(write)), max_(max),
        batch_(std::move(batch)) {
    if (write_) timer_ = std::thread([this] { Linger(); });
  }
  ~Streamer() { Stop(); }
  Streamer(const Streamer&) = delete;
  Streamer& operator=(const Streamer&) = delete;

  // How many entries a listing hands over at a time, so that no frame grows far past max.
  size_t piece() const { return std::max<size_t>(1, max_ / 4); }

  // add(batch) puts entries into the batch. A write that failed (the requester went away)
  // throws here, which ends the listing.
  template <class F>
  void Add(F&& add) {
    std::lock_guard<std::mutex> l(mu_);
    if (failed_) std::rethrow_exception(failed_);
    if (batch_.size() == 0) {
      first_ = std::chrono::steady_clock::now();
      cv_.notify_one();
    }
    add(batch_);
    if (write_ && batch_.size() >= max_) WriteLocked();
  }

  // The body of the last frame: what is left, after finish(batch) (the listing's totals).
  template <class F>
  std::vector<uint8_t> Last(F&& finish) {
    Stop();
    if (failed_) std::rethrow_exception(failed_);
    finish(batch_);
    BodyWriter w;
    batch_.Encode(&w, true);
    return std::move(w.buf);
  }
  std::vector<uint8_t> Last() {
    return Last([](Batch&) {});
  }

 private:
  void WriteLocked() {
    Frame f;
    f.type = type_;
    f.flags = kFlagResponse | kFlagMore;
    f.req_id = req_id_;
    f.epoch = epoch_;
    BodyWriter w;
    batch_.Encode(&w, false);
    batch_.Clear();
    f.body = std::move(w.buf);
    write_(f);
  }

  void Linger() {
    const auto linger = std::chrono::milliseconds(kStreamLingerMs);
    std::unique_lock<std::mutex> l(mu_);
    while (!stop_) {
      if (failed_ || batch_.size() == 0) {
        cv_.wait_for(l, linger);
        continue;
      }
      if (std::chrono::steady_clock::now() < first_ + linger) {
        cv_.wait_until(l, first_ + linger);
        continue;
      }
      try {
        WriteLocked();
      } catch (...) {
        failed_ = std::current_exception();
      }
    }
  }

  void Stop() {
    {
      std::lock_guard<std::mutex> l(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    if (timer_.joinable()) timer_.join();
  }

  const uint8_t type_;
  const uint64_t req_id_, epoch_;
  const Write write_;
  const size_t max_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::exception_ptr failed_;
  std::chrono::steady_clock::time_point first_;
  Batch batch_;
  std::thread timer_;
};

}  // namespace rtier
