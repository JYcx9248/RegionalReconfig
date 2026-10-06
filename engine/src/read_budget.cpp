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
    b_->Charge(count);
    r_->Read(pages, bufs, count);
  }
  IoBackend backend() const override { return r_->backend(); }

 private:
  std::unique_ptr<PageReader> r_;
  ReadBudget* b_;
};

}  // namespace

ReadBudget::ReadBudget(uint64_t reads_per_sec)
    : rate_(reads_per_sec),
      interval_ns_(1e9 / static_cast<double>(reads_per_sec)),
      burst_ns_(kBurstSeconds * 1e9) {
  FUSION_CHECK(reads_per_sec > 0, "a read budget needs a positive rate");
}

void ReadBudget::Charge(uint64_t reads) {
  if (reads == 0) return;
  const double now = NowNs();
  double release;
  {
    std::lock_guard<std::mutex> l(mu_);
    tat_ns_ = std::max(tat_ns_, now) + static_cast<double>(reads) * interval_ns_;
    release = tat_ns_ - burst_ns_;
  }
  charged_.fetch_add(reads, std::memory_order_relaxed);
  if (release > now) {
    const auto wait = std::chrono::nanoseconds(static_cast<int64_t>(release - now));
    std::this_thread::sleep_for(wait);
    wait_ns_.fetch_add(static_cast<uint64_t>(wait.count()), std::memory_order_relaxed);
  }
}

ReadBudgetStats ReadBudget::Stats() const {
  ReadBudgetStats s;
  s.reads_per_sec = rate_;
  s.charged = charged_.load(std::memory_order_relaxed);
  s.wait_us = wait_ns_.load(std::memory_order_relaxed) / 1000;
  return s;
}

std::unique_ptr<PageReader> ChargedPageReader(std::unique_ptr<PageReader> reader, ReadBudget* budget) {
  return std::make_unique<Charged>(std::move(reader), budget);
}

}  // namespace fusion
