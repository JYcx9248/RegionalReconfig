// rtier data-node service: serves fusion::NodeEngine over the rtier wire protocol.
//
// One thread per connection; each connection handles one request at a time (clients keep a
// pool of connections for concurrency). Requests that touch the engine's per-worker state
// (FILTER, RERANK, SEARCH_LOCAL) borrow one of the engine's workers for their duration -- a
// RERANK only once it has the raw vectors it lacks, fetched on its connection's thread, so that
// a new node's workers do not wait on the network while its FILTERs queue for them.
// Serve() returns only after every connection thread has finished, so the engine may be
// destroyed right after it.
//
// The server also gives the engine its way to fetch raw vectors on demand (PeerPool): a data
// node asks the holder a posting records directly, data node to data node, when a RERANK needs a
// vector it does not have (U9). RAW_GET takes no engine worker, so two nodes fetching from each
// other cannot deadlock on workers.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "fusion/node_engine.h"
#include "wire.h"

namespace rtier {

// Connections to other data nodes, for RAW_GET. Idle connections are kept per address; a
// request takes one, or opens one, for its duration. Thread-safe.
class PeerPool {
 public:
  explicit PeerPool(int timeout_ms = 30000) : timeout_ms_(timeout_ms) {}
  ~PeerPool();
  PeerPool(const PeerPool&) = delete;
  PeerPool& operator=(const PeerPool&) = delete;

  // Fetches n raw vectors (vec_bytes each, in the order of locs) from the data node at addr
  // ("host:port") into out. Throws std::runtime_error on failure.
  void RawGet(const std::string& addr, const uint32_t* locs, size_t n, uint32_t vec_bytes, uint8_t* out);

 private:
  int Take(const std::string& addr);
  void Give(const std::string& addr, int fd);

  // Idle connections kept per peer: a burst of fetches opens more, closed again as they come
  // back -- each one kept is a descriptor here and a thread at the peer.
  static constexpr size_t kMaxIdlePerPeer = 16;

  const int timeout_ms_;
  std::mutex mu_;
  std::map<std::string, std::vector<int>> idle_;
  std::atomic<uint64_t> next_id_{1};
};

class NodeServer {
 public:
  explicit NodeServer(fusion::NodeEngine* engine);
  ~NodeServer();

  // Binds a listening socket. Port 0 picks a free port; the bound port is returned.
  int ListenTcp(const std::string& host, int port);
  void ListenUnix(const std::string& path);

  // Accepts connections on every listener until Stop(). Blocks.
  void Serve();
  void Stop();

  // Handles one request frame (exposed for tests). The frames of a streamed reply but the last
  // go to write as they fill (wire.h); without it, all of the reply is in the returned frame.
  Frame Handle(const Frame& req, const std::function<void(const Frame&)>& write = {});

  // Entries per frame of a streamed reply (kStreamBatch; tests stream a small index in several).
  void SetStreamBatch(size_t entries) { stream_batch_ = std::max<size_t>(1, entries); }

  // One line of the node's stats log (rtier_node --stats-file): per-operation counters, the
  // fetch path's (fusion::FetchStats), the read budget's and the raw-vector bytes sent and
  // received (RAW_TRAFFIC), without INFO's raw-vector totals, which wait for the writer and
  // rescan the resident postings.
  std::string StatsJson() const;

 private:
  struct OpStats {
    std::atomic<uint64_t> count{0}, errors{0}, busy_us{0};
    std::atomic<uint64_t> wait_us{0};  // of busy_us, waiting for an engine worker
  };

  void AcceptLoop(int listen_fd);
  void ServeConnection(int fd, uint64_t id);
  int AcquireWorker(OpStats* st);
  void ReleaseWorker(int w);
  std::string InfoJson(bool settle) const;
  std::string OpsJson() const;
  std::string GraphJson() const;
  std::string FetchJson() const;

  fusion::NodeEngine* engine_;
  std::atomic<bool> stop_{false};
  std::vector<int> listen_fds_;
  std::vector<std::string> unix_paths_;
  std::mutex pool_mu_;
  std::condition_variable pool_cv_;
  std::vector<int> free_workers_;
  std::atomic<uint64_t> max_epoch_seen_{0};
  std::atomic<uint64_t> raw_sent_{0}, raw_received_{0};  // RAW_TRAFFIC
  std::atomic<size_t> stream_batch_{kStreamBatch};
  std::mutex conn_mu_;
  std::set<int> conn_fds_;                          // open connections (shut down by Stop)
  std::map<uint64_t, std::thread> conn_threads_;   // by connection; joined by Serve
  std::vector<uint64_t> conn_ended_;               // whose thread has ended, to join
  uint64_t next_conn_ = 0;
  OpStats stats_[256];
  PeerPool peers_;  // the engine's raw-vector fetcher (installed by the constructor)
};

}  // namespace rtier
