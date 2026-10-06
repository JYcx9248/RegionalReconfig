// A data node's device-read budget: at most so many reads per second (rtier_node --read-iops).
//
// Several data nodes on one machine share its SSD, so a scale-out adds no read capacity unless
// each node gets its own share. With root, run_local.py gives each node a cgroup with an io.max
// read-IOPS limit; without root the io controller is not delegated (systemd hands unprivileged
// users cpu, memory and pids only), and this budget stands in for it inside the node. It charges
// what reaches the device, as io.max does: every page re-ranking reads (O_DIRECT, unless the node
// runs with --no-direct), the pages the background writer reads back before rewriting them, and
// the RAW_GET reads the page cache does not serve. Not charged: bootstrap (before the load),
// segment and graph files (the agent's reads), and the reads a buffered install may cause --
// streams fill pages that are holes, which cost no read.
//
// A caller blocks until its reads fit the budget, in arrival order (generic cell rate algorithm:
// reads are spaced 1/rate apart, and an idle spell earns at most kBurstSeconds of budget, so a
// node that was idle does not get a long free run). The burst is what io.max allows: the kernel
// throttles in slices and lets a slice's worth of I/O through at once -- 100 ms on a disk that
// reports itself rotational, as the virtual disk of the reference runs does (WSL2), 20 ms on
// SSDs. With 20 ms the same load waited for the budget in bursts that io.max let through
// (BIGANN-10M, 2 nodes at 62% of 40K reads/s: p99 latency 30-60 ms instead of 7-10 ms).
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

#include "fusion/page_reader.h"

namespace fusion {

struct ReadBudgetStats {
  uint64_t reads_per_sec = 0;  // 0: no budget (nothing is charged)
  uint64_t charged = 0;        // cumulative: reads charged
  uint64_t wait_us = 0;        // cumulative: time callers waited for the budget
};

class ReadBudget {
 public:
  static constexpr double kBurstSeconds = 0.1;

  explicit ReadBudget(uint64_t reads_per_sec);
  // Blocks until `reads` more reads fit the budget.
  void Charge(uint64_t reads);
  ReadBudgetStats Stats() const;

 private:
  const uint64_t rate_;
  const double interval_ns_;  // 1 s / rate
  const double burst_ns_;
  std::mutex mu_;
  double tat_ns_ = 0;  // when the reads charged so far are paid for, on the steady clock
  std::atomic<uint64_t> charged_{0}, wait_ns_{0};
};

// A page reader that charges every page it reads to the budget before reading it.
std::unique_ptr<PageReader> ChargedPageReader(std::unique_ptr<PageReader> reader, ReadBudget* budget);

}  // namespace fusion
