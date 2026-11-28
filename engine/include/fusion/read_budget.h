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
// Two orders are for copy-then-flip's baselines, whose stream reads the old owners' devices while
// their queries do. Peers first (rtier_node --peer-reads-first): the stream takes the device ahead
// of the queries, the reads RAW_GET serves other nodes (a stream's pulls, a fetch) before the
// node's own (re-ranking, the writer's read-backs). Own first (--own-reads-first): the other way
// round, the stream gets what the queries leave of the device -- on an old owner whose queries
// keep it busy, nothing until they let up, so the flip waits for the load to fall. With no rate,
// the order is all there is to set: at most `depth` reads (--io-depth) are at the device at once,
// and the first class's reads go in ahead of every read of the other class still waiting. depth
// is about the device's queue (32 for a SATA SSD's NCQ, more on NVMe): below it the device idles,
// far above it the waiting reads queue in the kernel, in arrival order again. Reads already
// admitted keep their turn, so for at most depth reads the device sees both classes. A request of
// more reads than depth takes the whole depth; with own first a peer's request is cut into
// batches of peer_batch() reads (RawStore::Get), each admitted alone, so that the queries wait for
// at most a batch of the stream's. With a rate too, the first class's reads are also spaced only
// after those of its own charged before them. Off by default: every read in arrival order, none
// waiting for another.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>

#include "fusion/page_reader.h"

namespace fusion {

// Whose reads a charge is: the node's own, or a peer's (served by RAW_GET).
enum class ReadClass { kOwn, kPeer };
// Which class goes first, if any (the order above).
enum class ReadOrder { kArrival, kPeersFirst, kOwnFirst };

struct ReadBudgetStats {
  uint64_t reads_per_sec = 0;  // the rate; 0: no limit
  uint32_t depth = 0;          // an order: reads at the device at once; 0: none
  uint64_t charged = 0;        // cumulative: device reads, both classes
  uint64_t peer = 0;           // cumulative: those served to other nodes (RAW_GET)
  uint64_t wait_us = 0;        // cumulative: time callers waited, for the rate or their turn
};

class ReadBudget {
 public:
  static constexpr double kBurstSeconds = 0.1;

  // reads_per_sec 0: no rate. An order with depth > 0: the order above.
  explicit ReadBudget(uint64_t reads_per_sec, ReadOrder order = ReadOrder::kArrival, uint32_t depth = 0);

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

  // Waits for the reads' turn (an order), then for the rate; counts them. Hold the result while
  // the reads run.
  Admission Admit(uint64_t reads, ReadClass c = ReadClass::kOwn);
  // The rate alone, and the count: for reads that need no turn.
  void Charge(uint64_t reads, ReadClass c = ReadClass::kOwn);
  ReadBudgetStats Stats() const;
  uint64_t rate() const { return rate_; }
  // The most reads a peer's request admits at once: with own first, a few, so that the queries
  // never wait long behind the stream; otherwise as many as it has.
  uint32_t peer_batch() const {
    return order_ == ReadOrder::kOwnFirst && depth_ ? std::max<uint32_t>(1, depth_ / 8) : UINT32_MAX;
  }

 private:
  void Release(uint32_t held);

  const uint64_t rate_;
  const double interval_ns_;  // 1 s / rate
  const double burst_ns_;
  const ReadOrder order_;
  const ReadClass first_;  // with an order: the class that goes first
  const uint32_t depth_;   // 0: no order
  std::mutex mu_;
  double tat_ns_ = 0;        // when the reads charged so far are paid for, on the steady clock
  double first_tat_ns_ = 0;  // an order: when the first class's reads charged so far are paid for
  std::mutex order_mu_;
  std::condition_variable order_cv_;
  uint32_t in_flight_ = 0;      // places held at the device
  uint32_t first_waiting_ = 0;  // admissions of the first class waiting for their turn
  std::atomic<uint64_t> charged_{0}, peer_{0}, wait_ns_{0};
};

// A page reader that admits every batch it reads (its turn, the rate, the count) before reading.
std::unique_ptr<PageReader> ChargedPageReader(std::unique_ptr<PageReader> reader, ReadBudget* budget);

}  // namespace fusion
