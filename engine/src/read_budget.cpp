#include "fusion/read_budget.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace fusion {
namespace {

double NowNs() {
  return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now().time_since_epoch())
                                 .count());
}

class Charged final : public PageReader {
 public:
  Charged(std::unique_ptr<PageReader> r, ReadBudget* b) : r_(std::move(r)), b_(b) {}
  void Read(const uint32_t* pages, uint8_t* const* bufs, uint32_t count) override {
    const ReadBudget::Admission a = b_->Admit(count);
    r_->Read(pages, bufs, count);
  }
  IoBackend backend() const override { return r_->backend(); }

 private:
  std::unique_ptr<PageReader> r_;
  ReadBudget* b_;
};

}  // namespace

ReadBudget::ReadBudget(uint64_t reads_per_sec, bool peers_first, uint32_t depth)
    : rate_(reads_per_sec),
      interval_ns_(reads_per_sec ? 1e9 / static_cast<double>(reads_per_sec) : 0.0),
      burst_ns_(kBurstSeconds * 1e9),
      peers_first_(peers_first),
      depth_(peers_first ? depth : 0) {
  FUSION_CHECK(!peers_first || reads_per_sec || depth, "peers first needs a rate or a depth to order reads in");
}

ReadBudget::Admission::~Admission() {
  if (b_ && held_) b_->Release(held_);
}

ReadBudget::Admission ReadBudget::Admit(uint64_t reads, ReadClass c) {
  uint32_t held = 0;
  if (depth_ && reads) {
    // A request larger than the depth takes all of it: it is at the device alone, at its own
    // queue depth, as it would be with nothing else to read.
    held = static_cast<uint32_t>(std::min<uint64_t>(reads, depth_));
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> l(order_mu_);
    if (c == ReadClass::kPeer) {
      ++peers_waiting_;
      order_cv_.wait(l, [&] { return in_flight_ + held <= depth_; });
      --peers_waiting_;
    } else {
      order_cv_.wait(l, [&] { return peers_waiting_ == 0 && in_flight_ + held <= depth_; });
    }
    in_flight_ += held;
    l.unlock();
    wait_ns_.fetch_add(static_cast<uint64_t>(
                           std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()),
                       std::memory_order_relaxed);
  }
  Charge(reads, c);
  return Admission(this, held);
}

void ReadBudget::Release(uint32_t held) {
  {
    std::lock_guard<std::mutex> l(order_mu_);
    in_flight_ -= held;
  }
  order_cv_.notify_all();
}

void ReadBudget::Charge(uint64_t reads, ReadClass c) {
  if (reads == 0) return;
  charged_.fetch_add(reads, std::memory_order_relaxed);
  if (c == ReadClass::kPeer) peer_.fetch_add(reads, std::memory_order_relaxed);
  if (!rate_) return;
  const double now = NowNs();
  const double cost = static_cast<double>(reads) * interval_ns_;
  double release;
  {
    std::lock_guard<std::mutex> l(mu_);
    tat_ns_ = std::max(tat_ns_, now) + cost;  // every read: the own ones after it wait for it
    release = tat_ns_ - burst_ns_;
    if (peers_first_ && c == ReadClass::kPeer) {
      peer_tat_ns_ = std::max(peer_tat_ns_, now) + cost;
      release = peer_tat_ns_ - burst_ns_;
    }
  }
  if (release > now) {
    const auto wait = std::chrono::nanoseconds(static_cast<int64_t>(release - now));
    std::this_thread::sleep_for(wait);
    wait_ns_.fetch_add(static_cast<uint64_t>(wait.count()), std::memory_order_relaxed);
  }
}

ReadBudgetStats ReadBudget::Stats() const {
  ReadBudgetStats s;
  s.reads_per_sec = rate_;
  s.depth = depth_;
  s.charged = charged_.load(std::memory_order_relaxed);
  s.peer = peer_.load(std::memory_order_relaxed);
  s.wait_us = wait_ns_.load(std::memory_order_relaxed) / 1000;
  return s;
}

std::unique_ptr<PageReader> ChargedPageReader(std::unique_ptr<PageReader> reader, ReadBudget* budget) {
  return std::make_unique<Charged>(std::move(reader), budget);
}

}  // namespace fusion
