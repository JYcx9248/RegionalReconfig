// A data node's device reads (rtier_node): counted always, ordered with --peer-reads-first, and
// limited to so many a second only with --read-iops.
//
// What counts is what reaches the device: every page re-ranking reads (O_DIRECT, unless the node
// runs with --no-direct), the pages the background writer reads back before rewriting them, and
// the RAW_GET reads the page cache does not serve. Not counted: bootstrap (before the load),
// segment and graph files (the agent's reads), the reads a buffered install may cause -- streams
// fill pages that are holes, which cost no read -- and the kernel's read-ahead, which a stream's
// sorted batches get (engine/README.md, known limits). The node's stats report the reads ("reads":
// charged, and peer: those served to other nodes) whether or not anything limits them.
//
// The rate (--read-iops) is for several data nodes on one machine, which share its SSD: a scale-out
// adds no read capacity unless each node gets its own share. With root, run_local.py gives each
// node a cgroup with an io.max read-IOPS limit; without root the io controller is not delegated
// (systemd hands unprivileged users cpu, memory and pids only), and the rate stands in for it
// inside the node. A caller blocks until its reads fit, in arrival order (generic cell rate
// algorithm: reads are spaced 1/rate apart, and an idle spell earns at most kBurstSeconds of
// budget, so a node that was idle does not get a long free run). The burst is what io.max allows:
// the kernel throttles in slices and lets a slice's worth of I/O through at once -- 100 ms on a
// disk that reports itself rotational, as the virtual disk of the reference runs does (WSL2),
// 20 ms on SSDs. With 20 ms the same load waited for the budget in bursts that io.max let through
// (BIGANN-10M, 2 nodes at 62% of 40K reads/s: p99 latency 30-60 ms instead of 7-10 ms). Since
// 2026-11-12 no experiment sets a rate: a cluster's nodes have a device each.
//
// Peers first (rtier_node --peer-reads-first) is for a baseline: copy-then-flip whose stream
// takes the device ahead of the queries, the reads RAW_GET serves other nodes (a stream's pulls,
// a fetch) before the node's own (re-ranking, the writer's read-backs). With no rate, the order is
// all there is to set: at most `depth` reads (--io-depth) are at the device at once, and a peer's
// reads go in ahead of every own read still waiting, so the queries get what the transfer leaves
// of the device. depth is about the device's queue (32 for a SATA SSD's NCQ, more on NVMe): below
// it the device idles, far above it the own reads queue in the kernel ahead of the peers' again.
// Own reads already admitted keep their turn, so for at most depth reads the device sees both. A
// request of more reads than depth takes the whole depth. With a rate too, the peer reads are also
// spaced only after the peer reads charged before them. Off by default: every read in arrival
// order, none waiting for another.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>

#include "fusion/page_reader.h"

namespace fusion {

// Whose reads a charge is: the node's own, or a peer's (served by RAW_GET).
enum class ReadClass { kOwn, kPeer };

struct ReadBudgetStats {
  uint64_t reads_per_sec = 0;  // the rate; 0: no limit
  uint32_t depth = 0;          // peers first: reads at the device at once; 0: no order
  uint64_t charged = 0;        // cumulative: device reads, both classes
  uint64_t peer = 0;           // cumulative: those served to other nodes (RAW_GET)
  uint64_t wait_us = 0;        // cumulative: time callers waited, for the rate or their turn
};

class ReadBudget {
 public:
  static constexpr double kBurstSeconds = 0.1;

  // reads_per_sec 0: no rate. peers_first with depth > 0: the order above.
  explicit ReadBudget(uint64_t reads_per_sec, bool peers_first = false, uint32_t depth = 0);

  // Reads at the device: from Admit until it goes out of scope (the reads done).
  class Admission {
   public:
    Admission(Admission&& o) noexcept : b_(o.b_), held_(o.held_) { o.b_ = nullptr; }
    Admission(const Admission&) = delete;
    Admission& operator=(const Admission&) = delete;
    Admission& operator=(Admission&&) = delete;
    ~Admission();

   private:
    friend class ReadBudget;
    Admission(ReadBudget* b, uint32_t held) : b_(b), held_(held) {}
    ReadBudget* b_;
    uint32_t held_;  // places at the device this admission holds
  };

  // Waits for the reads' turn (peers first), then for the rate; counts them. Hold the result
  // while the reads run.
  Admission Admit(uint64_t reads, ReadClass c = ReadClass::kOwn);
  // The rate alone, and the count: for reads that need no turn.
  void Charge(uint64_t reads, ReadClass c = ReadClass::kOwn);
  ReadBudgetStats Stats() const;
  uint64_t rate() const { return rate_; }

 private:
  void Release(uint32_t held);

  const uint64_t rate_;
  const double interval_ns_;  // 1 s / rate
  const double burst_ns_;
  const bool peers_first_;
  const uint32_t depth_;  // 0: no order
  std::mutex mu_;
  double tat_ns_ = 0;       // when the reads charged so far are paid for, on the steady clock
  double peer_tat_ns_ = 0;  // peers first: when the peer reads charged so far are paid for
  std::mutex order_mu_;
  std::condition_variable order_cv_;
  uint32_t in_flight_ = 0;      // places held at the device
  uint32_t peers_waiting_ = 0;  // peer admissions waiting for their turn
  std::atomic<uint64_t> charged_{0}, peer_{0}, wait_ns_{0};
};

// A page reader that admits every batch it reads (its turn, the rate, the count) before reading.
std::unique_ptr<PageReader> ChargedPageReader(std::unique_ptr<PageReader> reader, ReadBudget* budget);

}  // namespace fusion
