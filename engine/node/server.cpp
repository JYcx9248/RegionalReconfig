#include "server.h"

#include "stream.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <chrono>
#include <exception>
#include <functional>
#include <thread>

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "the rtier wire format is little endian");

namespace rtier {

// ------------------------------------------------------------------------------ frame I/O
namespace {

bool ReadFull(int fd, void* buf, size_t n, bool eof_ok_at_start) {
  size_t got = 0;
  while (got < n) {
    const ssize_t r = ::recv(fd, static_cast<char*>(buf) + got, n - got, 0);
    if (r == 0) {
      if (got == 0 && eof_ok_at_start) return false;
      throw std::runtime_error("connection closed in the middle of a frame");
    }
    if (r < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("recv: ") + std::strerror(errno));
    }
    got += static_cast<size_t>(r);
  }
  return true;
}

void WriteFull(int fd, const void* buf, size_t n) {
  size_t sent = 0;
  while (sent < n) {
    const ssize_t r = ::send(fd, static_cast<const char*>(buf) + sent, n - sent, MSG_NOSIGNAL);
    if (r < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("send: ") + std::strerror(errno));
    }
    sent += static_cast<size_t>(r);
  }
}

}  // namespace

bool ReadFrame(int fd, Frame* f) {
  uint8_t head[8];
  if (!ReadFull(fd, head, sizeof(head), true)) return false;
  uint32_t magic, len;
  std::memcpy(&magic, head, 4);
  std::memcpy(&len, head + 4, 4);
  if (magic != kMagicStart) throw std::runtime_error("bad frame magic");
  if (len < 24 || len > kMaxFrameBytes) throw std::runtime_error("bad frame length");
  std::vector<uint8_t> rest(len);
  ReadFull(fd, rest.data(), len, false);
  uint32_t end;
  std::memcpy(&end, rest.data() + len - 4, 4);
  if (end != kMagicEnd) throw std::runtime_error("bad frame trailer");
  f->type = rest[0];
  f->flags = rest[1];
  std::memcpy(&f->status, rest.data() + 2, 2);
  std::memcpy(&f->req_id, rest.data() + 4, 8);
  std::memcpy(&f->epoch, rest.data() + 12, 8);
  f->body.assign(rest.begin() + 20, rest.end() - 4);
  return true;
}

void WriteFrame(int fd, const Frame& f) {
  const uint32_t len = static_cast<uint32_t>(20 + f.body.size() + 4);
  if (len > kMaxFrameBytes) throw std::runtime_error("frame too large");
  std::vector<uint8_t> buf(8 + len);
  uint8_t* p = buf.data();
  std::memcpy(p, &kMagicStart, 4);
  std::memcpy(p + 4, &len, 4);
  p[8] = f.type;
  p[9] = f.flags;
  std::memcpy(p + 10, &f.status, 2);
  std::memcpy(p + 12, &f.req_id, 8);
  std::memcpy(p + 20, &f.epoch, 8);
  if (!f.body.empty()) std::memcpy(p + 28, f.body.data(), f.body.size());
  std::memcpy(p + 28 + f.body.size(), &kMagicEnd, 4);
  WriteFull(fd, buf.data(), buf.size());
}

// ------------------------------------------------------------------------------ server
namespace {

const char* OpName(uint8_t op) {
  switch (op) {
    case kPing: return "ping";
    case kInfo: return "info";
    case kLoadGraph: return "load_graph";
    case kLoadPartition: return "load_partition";
    case kEvictPartition: return "evict_partition";
    case kPQMissing: return "pq_missing";
    case kPQGet: return "pq_get";
    case kPQPut: return "pq_put";
    case kPQRelease: return "pq_release";
    case kRawMissing: return "raw_missing";
    case kRawGet: return "raw_get";
    case kRawPut: return "raw_put";
    case kRawCheck: return "raw_check";
    case kRawPull: return "raw_pull";
    case kRawAdd: return "raw_add";
    case kPauseUpdates: return "pause_updates";
    case kNavigate: return "navigate";
    case kFilter: return "filter";
    case kRerank: return "rerank";
    case kSearchLocal: return "search_local";
    case kAssign: return "assign";
    case kSplitList: return "split_list";
    case kNearestHeads: return "nearest_heads";
    case kEditGraph: return "edit_graph";
    case kStageVersions: return "stage_versions";
    case kDropVersion: return "drop_version";
    case kForget: return "forget";
    case kSavePartition: return "save_partition";
    case kSetNodes: return "set_nodes";
    case kRawHeldBy: return "raw_held_by";
    case kListGet: return "list_get";
    case kPatchStage: return "patch_stage";
    case kPatchLoad: return "patch_load";
    case kPatchMerge: return "patch_merge";
    case kPatchDrop: return "patch_drop";
    case kPatchSave: return "patch_save";
    case kSwitchGeneration: return "switch_generation";
    case kRawBitmap: return "raw_bitmap";
    case kRawTraffic: return "raw_traffic";
    default: return nullptr;
  }
}

// Copies n elements out of the (unaligned) body.
template <class T>
std::vector<T> TakeArray(BodyReader* r, size_t n) {
  std::vector<T> v(n);
  if (n) std::memcpy(v.data(), r->Bytes(n * sizeof(T)), n * sizeof(T));
  return v;
}

// The query vector is the rest of the body; copied into 8-byte aligned storage.
std::vector<uint64_t> TakeQuery(BodyReader* r, const fusion::NodeEngine& e) {
  const size_t qb = static_cast<size_t>(e.dim()) * fusion::DTypeSize(e.dtype());
  if (r->remaining() != qb)
    throw std::invalid_argument(fusion::StrFormat(
        "query has %zu bytes, expected %zu (dim %u, %s)", r->remaining(), qb, e.dim(),
        fusion::DTypeName(e.dtype())));
  std::vector<uint64_t> q((qb + 7) / 8);
  std::memcpy(q.data(), r->Bytes(qb), qb);
  return q;
}

void Require(bool cond, const std::string& msg) {
  if (!cond) throw std::invalid_argument(msg);
}

std::string TakeStr(BodyReader* r) {
  const uint32_t n = r->Get<uint32_t>();
  const uint8_t* p = r->Bytes(n);
  return std::string(reinterpret_cast<const char*>(p), n);
}

// PQ_MISSING / RAW_MISSING: u32 groups, per group u32 paths, str path[paths]; nothing after.
std::vector<std::vector<std::string>> TakeGroups(BodyReader* r) {
  const uint32_t ngroups = r->Get<uint32_t>();
  Require(ngroups <= 1024, "too many groups");
  std::vector<std::vector<std::string>> groups(ngroups);
  for (auto& g : groups) {
    const uint32_t npaths = r->Get<uint32_t>();
    Require(npaths <= r->remaining() / 4, "bad path count");
    for (uint32_t i = 0; i < npaths; ++i) g.push_back(TakeStr(r));
  }
  Require(r->remaining() == 0, "trailing bytes after the groups");
  return groups;
}

// PQ_MISSING's frames: u32 runs, (u32 group, u32 n, u32 ids[n])[runs] -- the IDs in the order
// they were found, a run per group.
struct IdBatch {
  std::vector<std::pair<uint32_t, std::vector<uint32_t>>> runs;
  size_t n = 0;

  size_t size() const { return n; }
  void Add(uint32_t group, const uint32_t* ids, size_t k) {
    if (runs.empty() || runs.back().first != group) runs.emplace_back(group, std::vector<uint32_t>());
    runs.back().second.insert(runs.back().second.end(), ids, ids + k);
    n += k;
  }
  void Encode(BodyWriter* w, bool) const {
    w->Put<uint32_t>(static_cast<uint32_t>(runs.size()));
    for (const auto& [group, ids] : runs) {
      w->Put<uint32_t>(group);
      w->Put<uint32_t>(static_cast<uint32_t>(ids.size()));
      w->PutBytes(ids.data(), ids.size() * 4);
    }
  }
  void Clear() {
    runs.clear();
    n = 0;
  }
};

// RAW_MISSING's frames: u32 groups, (u16 holder, u16 pad, u32 n, u32 locs[n], u32 lists[n])
// [groups]. RAW_HELD_BY's (held_by): u64 repointed (the last frame's; 0 before), u32 n,
// u32 locs[n], u32 lists[n], all of the one holder.
struct RawBatch {
  bool held_by = false;
  uint64_t repointed = 0;
  std::map<uint16_t, fusion::RawRefs> by;
  size_t n = 0;

  size_t size() const { return n; }
  void Add(const fusion::RawRef* refs, size_t k) {
    for (size_t i = 0; i < k; ++i) {
      fusion::RawRefs& g = by[refs[i].holder];
      g.locs.push_back(refs[i].loc);
      g.lists.push_back(refs[i].list);
    }
    n += k;
  }
  void Encode(BodyWriter* w, bool last) const {
    if (held_by) {
      Require(by.size() <= 1, "RAW_HELD_BY lists one holder");
      w->Put<uint64_t>(last ? repointed : 0);
      w->Put<uint32_t>(static_cast<uint32_t>(n));
      for (const auto& [holder, g] : by) {
        w->PutBytes(g.locs.data(), g.locs.size() * 4);
        w->PutBytes(g.lists.data(), g.lists.size() * 4);
      }
      return;
    }
    w->Put<uint32_t>(static_cast<uint32_t>(by.size()));
    for (const auto& [holder, g] : by) {
      w->Put<uint16_t>(holder);
      w->Put<uint16_t>(0);
      w->Put<uint32_t>(static_cast<uint32_t>(g.locs.size()));
      w->PutBytes(g.locs.data(), g.locs.size() * 4);
      w->PutBytes(g.lists.data(), g.lists.size() * 4);
    }
  }
  void Clear() {
    by.clear();
    n = 0;
  }
};

}  // namespace

// ------------------------------------------------------------------------------ peers
PeerPool::~PeerPool() {
  for (auto& [addr, fds] : idle_)
    for (int fd : fds) ::close(fd);
}

int PeerPool::Take(const std::string& addr) {
  {
    std::lock_guard<std::mutex> l(mu_);
    auto it = idle_.find(addr);
    if (it != idle_.end() && !it->second.empty()) {
      const int fd = it->second.back();
      it->second.pop_back();
      return fd;
    }
  }
  const size_t colon = addr.rfind(':');
  if (colon == std::string::npos) throw std::runtime_error("peer address " + addr + " has no port");
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  const int gai = ::getaddrinfo(addr.substr(0, colon).c_str(), addr.substr(colon + 1).c_str(), &hints, &res);
  if (gai != 0) throw std::runtime_error("resolve " + addr + ": " + ::gai_strerror(gai));
  int fd = -1;
  std::string err = "no address";
  for (addrinfo* ai = res; ai != nullptr && fd < 0; ai = ai->ai_next) {
    fd = ::socket(ai->ai_family, ai->ai_socktype | SOCK_CLOEXEC, ai->ai_protocol);
    if (fd < 0) {
      err = std::strerror(errno);
      continue;
    }
    if (::connect(fd, ai->ai_addr, ai->ai_addrlen) != 0) {
      err = std::strerror(errno);
      ::close(fd);
      fd = -1;
    }
  }
  ::freeaddrinfo(res);
  if (fd < 0) throw std::runtime_error("connect to " + addr + ": " + err);
  const int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  // A peer that stops answering fails the fetch instead of hanging the query.
  timeval tv{timeout_ms_ / 1000, (timeout_ms_ % 1000) * 1000};
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
  return fd;
}

void PeerPool::Give(const std::string& addr, int fd) {
  {
    std::lock_guard<std::mutex> l(mu_);
    auto& idle = idle_[addr];
    if (idle.size() < kMaxIdlePerPeer) {
      idle.push_back(fd);
      return;
    }
  }
  ::close(fd);
}

void PeerPool::RawGet(const std::string& addr, const uint32_t* locs, size_t n, uint32_t vec_bytes,
                      uint8_t* out) {
  for (size_t done = 0; done < n;) {  // frames stay far below kMaxFrameBytes
    const size_t batch = std::min<size_t>(
        {n - done, kMaxRawBatch, std::max<size_t>(1, (kMaxFrameBytes / 4) / vec_bytes)});
    Frame req;
    req.type = kRawGet;
    req.req_id = next_id_.fetch_add(1);
    BodyWriter w;
    w.Put<uint32_t>(static_cast<uint32_t>(batch));
    w.PutBytes(locs + done, batch * 4);
    req.body = std::move(w.buf);
    const int fd = Take(addr);
    Frame resp;
    try {
      WriteFrame(fd, req);
      if (!ReadFrame(fd, &resp)) throw std::runtime_error("connection closed");
    } catch (const std::exception& e) {
      ::close(fd);
      throw std::runtime_error("RAW_GET to " + addr + ": " + e.what());
    }
    if (resp.req_id != req.req_id || resp.type != kRawGet || !(resp.flags & kFlagResponse)) {
      ::close(fd);
      throw std::runtime_error("RAW_GET to " + addr + ": mismatched response");
    }
    Give(addr, fd);
    if (resp.status != kOk)
      throw std::runtime_error(fusion::StrFormat("RAW_GET to %s: status %u: %s", addr.c_str(),
                                                 resp.status, std::string(resp.body.begin(), resp.body.end()).c_str()));
    BodyReader r(resp.body.data(), resp.body.size());
    const uint32_t vb = r.Get<uint32_t>();
    if (vb != vec_bytes || r.remaining() != batch * vb)
      throw std::runtime_error(fusion::StrFormat("RAW_GET to %s: %zu bytes of %u-byte vectors for "
                                                 "%zu locations", addr.c_str(), r.remaining(), vb, batch));
    std::memcpy(out + done * vec_bytes, r.Bytes(batch * vb), batch * vb);
    done += batch;
  }
}

// ------------------------------------------------------------------------------ server
NodeServer::NodeServer(fusion::NodeEngine* engine) : engine_(engine) {
  for (int i = engine->num_workers() - 1; i >= 0; --i) free_workers_.push_back(i);
  const uint32_t vb = engine->vec_bytes();
  engine->SetRawFetcher([this, vb](const std::string& peer, const uint32_t* locs, size_t n, uint8_t* out) {
    peers_.RawGet(peer, locs, n, vb, out);
    raw_received_.fetch_add(static_cast<uint64_t>(n) * vb, std::memory_order_relaxed);
  });
}

NodeServer::~NodeServer() {
  Stop();
  engine_->SetRawFetcher(nullptr);  // peers_ goes away with the server
}

int NodeServer::ListenTcp(const std::string& host, int port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) throw std::runtime_error(std::string("socket: ") + std::strerror(errno));
  const int one = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1)
    throw std::runtime_error("bad listen address " + host);
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 || ::listen(fd, 512) < 0) {
    const std::string err = std::strerror(errno);
    ::close(fd);
    throw std::runtime_error(fusion::StrFormat("listen on %s:%d: %s", host.c_str(), port, err.c_str()));
  }
  socklen_t alen = sizeof(addr);
  ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &alen);
  listen_fds_.push_back(fd);
  return ntohs(addr.sin_port);
}

void NodeServer::ListenUnix(const std::string& path) {
  const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) throw std::runtime_error(std::string("socket: ") + std::strerror(errno));
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  if (path.size() >= sizeof(addr.sun_path)) throw std::runtime_error("unix socket path too long");
  std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
  ::unlink(path.c_str());
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 || ::listen(fd, 512) < 0) {
    const std::string err = std::strerror(errno);
    ::close(fd);
    throw std::runtime_error("listen on " + path + ": " + err);
  }
  listen_fds_.push_back(fd);
  unix_paths_.push_back(path);
}

void NodeServer::Serve() {
  std::vector<std::thread> threads;
  for (int fd : listen_fds_) threads.emplace_back([this, fd] { AcceptLoop(fd); });
  for (auto& t : threads) t.join();
  // Listeners are closed and Stop() shut every open connection down: wait for the connection
  // threads, which may still be finishing a request on the engine.
  std::map<uint64_t, std::thread> conns;
  {
    std::lock_guard<std::mutex> l(conn_mu_);
    conns.swap(conn_threads_);
  }
  for (auto& kv : conns) kv.second.join();
}

void NodeServer::Stop() {
  if (stop_.exchange(true)) return;
  for (int fd : listen_fds_) ::shutdown(fd, SHUT_RDWR);
  for (const auto& p : unix_paths_) ::unlink(p.c_str());
  std::lock_guard<std::mutex> l(conn_mu_);
  for (int fd : conn_fds_) ::shutdown(fd, SHUT_RDWR);  // unblocks recv; the thread closes fd
}

void NodeServer::AcceptLoop(int listen_fd) {
  while (!stop_) {
    const int fd = ::accept(listen_fd, nullptr, nullptr);
    if (fd < 0) {
      if (stop_ || errno == EBADF || errno == EINVAL) break;  // listener shut down
      // Anything else is not the listener's end: a connection's own network error, which accept
      // passes on (ECONNABORTED, EPROTO, ...), or descriptors or memory running out for a while.
      if (errno != EINTR) std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    const int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));  // fails harmlessly on AF_UNIX
    std::lock_guard<std::mutex> l(conn_mu_);
    if (stop_) {
      ::close(fd);
      break;
    }
    // The threads of connections that ended: their last step is to say so, under this lock.
    for (uint64_t id : conn_ended_) {
      const auto it = conn_threads_.find(id);
      if (it == conn_threads_.end()) continue;
      it->second.join();
      conn_threads_.erase(it);
    }
    conn_ended_.clear();
    const uint64_t id = next_conn_++;
    try {
      conn_threads_.emplace(id, std::thread([this, fd, id] { ServeConnection(fd, id); }));
      conn_fds_.insert(fd);
    } catch (const std::exception& e) {  // no thread to be had: this connection is refused
      fusion::Log("connection refused: %s", e.what());
      ::close(fd);
    }
  }
  ::close(listen_fd);
}

void NodeServer::ServeConnection(int fd, uint64_t id) {
  try {
    Frame req;
    const std::function<void(const Frame&)> write = [fd](const Frame& f) { WriteFrame(fd, f); };
    while (!stop_ && ReadFrame(fd, &req)) WriteFrame(fd, Handle(req, write));
  } catch (const std::exception& e) {
    if (!stop_) fusion::Log("connection closed: %s", e.what());
  }
  std::lock_guard<std::mutex> l(conn_mu_);
  conn_fds_.erase(fd);
  ::close(fd);
  conn_ended_.push_back(id);
}

int NodeServer::AcquireWorker(OpStats* st) {
  fusion::Timer t;
  std::unique_lock<std::mutex> l(pool_mu_);
  pool_cv_.wait(l, [&] { return !free_workers_.empty(); });
  const int w = free_workers_.back();
  free_workers_.pop_back();
  st->wait_us += static_cast<uint64_t>(t.Us());
  return w;
}

void NodeServer::ReleaseWorker(int w) {
  {
    std::lock_guard<std::mutex> l(pool_mu_);
    free_workers_.push_back(w);
  }
  pool_cv_.notify_one();
}

Frame NodeServer::Handle(const Frame& req, const std::function<void(const Frame&)>& write) {
  Frame resp;
  resp.type = req.type;
  resp.flags = kFlagResponse;
  resp.req_id = req.req_id;
  resp.epoch = req.epoch;
  uint64_t seen = max_epoch_seen_.load();
  while (req.epoch > seen && !max_epoch_seen_.compare_exchange_weak(seen, req.epoch)) {
  }
  OpStats& st = stats_[req.type];
  st.count++;
  fusion::Timer timer;
  int worker = -1;
  struct Guard {
    NodeServer* s;
    int* w;
    ~Guard() {
      if (*w >= 0) s->ReleaseWorker(*w);
    }
  } guard{this, &worker};

  try {
    fusion::NodeEngine& e = *engine_;
    const auto& o = e.options();
    BodyReader r(req.body.data(), req.body.size());
    BodyWriter w;
    switch (req.type) {
      case kPing:
        break;
      case kInfo: {
        const bool settle = r.remaining() >= 1 && (r.Get<uint8_t>() & kInfoSettle) != 0;
        const std::string s = InfoJson(settle);
        w.PutBytes(s.data(), s.size());
        break;
      }
      case kLoadGraph:
        e.LoadGraph(r.Rest());
        break;
      case kLoadPartition: {
        const uint32_t p = r.Get<uint32_t>();
        const uint32_t version = r.Get<uint32_t>();
        const uint8_t src = r.Get<uint8_t>();
        Require(src <= 2, "unknown PQ source");
        e.LoadPartition(p, r.Rest(),
                        src == 1   ? fusion::PQSource::kIndex
                        : src == 2 ? fusion::PQSource::kGenIndex
                                   : fusion::PQSource::kPresent,
                        version);
        break;
      }
      case kSwitchGeneration: {
        const uint32_t gen = r.Get<uint32_t>();
        const std::string index_dir = TakeStr(&r);
        const std::string partitions_dir = TakeStr(&r);
        Require(r.remaining() == 0, "trailing bytes in SWITCH_GENERATION");
        e.SwitchGeneration(gen, index_dir, partitions_dir);
        break;
      }
      case kRawBitmap: {
        const std::vector<uint64_t> bits = e.RawBitmap();
        w.Put<uint64_t>(e.raw_stats(false).locations);
        w.PutBytes(bits.data(), bits.size() * sizeof(uint64_t));
        break;
      }
      case kEvictPartition:
        w.Put<uint8_t>(e.EvictPartition(r.Get<uint32_t>()) ? 1 : 0);
        break;
      case kPQMissing: {
        const auto groups = TakeGroups(&r);
        Streamer<IdBatch> stream(req, write, stream_batch_);
        e.PQMissingStream(groups, [&](size_t g, const uint32_t* ids, size_t n) {
          for (size_t i = 0; i < n; i += stream.piece()) {
            const size_t k = std::min(stream.piece(), n - i);
            stream.Add([&](IdBatch& b) { b.Add(static_cast<uint32_t>(g), ids + i, k); });
          }
        });
        const std::vector<uint8_t> last = stream.Last();
        w.PutBytes(last.data(), last.size());
        break;
      }
      case kPQGet: {
        const uint32_t n = r.Get<uint32_t>();
        const uint32_t m = e.pq_stats().m;
        Require(n <= kMaxPQBatch && static_cast<uint64_t>(n) * m <= kMaxFrameBytes / 2,
                "PQ_GET batch too large");
        const auto ids = TakeArray<uint32_t>(&r, n);
        Require(r.remaining() == 0, "trailing bytes in PQ_GET");
        std::vector<uint8_t> codes(static_cast<size_t>(n) * m);
        e.PQGet(ids.data(), n, codes.data());
        w.Put<uint32_t>(m);
        w.PutBytes(codes.data(), codes.size());
        break;
      }
      case kPQPut: {
        const uint32_t n = r.Get<uint32_t>();
        const uint32_t m = r.Get<uint32_t>();
        Require(m == e.pq_stats().m, "PQ code size does not match this node's index");
        Require(n <= kMaxPQBatch && static_cast<uint64_t>(n) * m <= kMaxFrameBytes / 2,
                "PQ_PUT batch too large");
        const auto ids = TakeArray<uint32_t>(&r, n);
        Require(r.remaining() == static_cast<size_t>(n) * m, "PQ_PUT: codes do not match the IDs");
        const fusion::PQPutResult res = e.PQPut(ids.data(), n, r.Bytes(static_cast<size_t>(n) * m));
        w.Put<uint32_t>(static_cast<uint32_t>(res.installed));
        w.Put<uint32_t>(static_cast<uint32_t>(res.skipped));
        break;
      }
      case kPQRelease:
        w.Put<uint64_t>(e.PQReleaseStaged());
        break;
      case kRawMissing: {
        const uint32_t np = r.Get<uint32_t>();
        Require(static_cast<uint64_t>(np) * 4 == r.remaining(), "RAW_MISSING: partitions do not match");
        const auto parts = TakeArray<uint32_t>(&r, np);
        Streamer<RawBatch> stream(req, write, stream_batch_);
        e.RawMissingStream(parts, [&](const std::vector<fusion::RawRef>& refs) {
          for (size_t i = 0; i < refs.size(); i += stream.piece()) {
            const size_t k = std::min(stream.piece(), refs.size() - i);
            stream.Add([&](RawBatch& b) { b.Add(refs.data() + i, k); });
          }
        });
        const std::vector<uint8_t> last = stream.Last();
        w.PutBytes(last.data(), last.size());
        break;
      }
      case kRawHeldBy: {
        const uint16_t holder = r.Get<uint16_t>();
        r.Get<uint16_t>();
        Require(r.remaining() == 0, "trailing bytes in RAW_HELD_BY");
        RawBatch held;
        held.held_by = true;
        Streamer<RawBatch> stream(req, write, stream_batch_, std::move(held));
        const uint64_t repointed = e.RawHeldByStream(holder, [&](const std::vector<fusion::RawRef>& refs) {
          for (size_t i = 0; i < refs.size(); i += stream.piece()) {
            const size_t k = std::min(stream.piece(), refs.size() - i);
            stream.Add([&](RawBatch& b) { b.Add(refs.data() + i, k); });
          }
        });
        const std::vector<uint8_t> last = stream.Last([&](RawBatch& b) { b.repointed = repointed; });
        w.PutBytes(last.data(), last.size());
        break;
      }
      case kSetNodes: {
        const uint16_t self = r.Get<uint16_t>();
        r.Get<uint16_t>();
        const uint32_t n = r.Get<uint32_t>();
        std::vector<std::pair<uint16_t, std::string>> peers;
        for (uint32_t i = 0; i < n; ++i) {
          const uint16_t id = r.Get<uint16_t>();
          r.Get<uint16_t>();
          peers.emplace_back(id, TakeStr(&r));
        }
        Require(r.remaining() == 0, "trailing bytes in SET_NODES");
        e.SetNodes(self, peers);
        break;
      }
      case kRawGet: {
        const uint32_t n = r.Get<uint32_t>();
        const uint32_t vb = e.vec_bytes();
        Require(n <= kMaxRawBatch && static_cast<uint64_t>(n) * vb <= kMaxFrameBytes / 2,
                "RAW_GET batch too large");
        const auto locs = TakeArray<uint32_t>(&r, n);
        Require(r.remaining() == 0, "trailing bytes in RAW_GET");
        std::vector<uint8_t> vecs(static_cast<size_t>(n) * vb);
        e.RawGet(locs.data(), n, vecs.data());
        raw_sent_.fetch_add(vecs.size(), std::memory_order_relaxed);
        w.Put<uint32_t>(vb);
        w.PutBytes(vecs.data(), vecs.size());
        break;
      }
      case kRawCheck: {
        const uint32_t n = r.Get<uint32_t>();
        Require(n <= kMaxRawBatch, "RAW_CHECK batch too large");
        const auto locs = TakeArray<uint32_t>(&r, n);
        Require(r.remaining() == 0, "trailing bytes in RAW_CHECK");
        const auto absent = e.RawAbsent(locs.data(), n);
        w.Put<uint32_t>(static_cast<uint32_t>(absent.size()));
        w.PutBytes(absent.data(), absent.size() * 4);
        break;
      }
      case kRawPut: {
        const uint32_t n = r.Get<uint32_t>();
        const uint32_t vb = r.Get<uint32_t>();
        Require(vb == e.vec_bytes(), "vector size does not match this node's index");
        Require(n <= kMaxRawBatch && static_cast<uint64_t>(n) * vb <= kMaxFrameBytes / 2,
                "RAW_PUT batch too large");
        const auto locs = TakeArray<uint32_t>(&r, n);
        Require(r.remaining() == static_cast<size_t>(n) * vb, "RAW_PUT: vectors do not match the locations");
        const fusion::RawPutResult res = e.RawPut(locs.data(), n, r.Bytes(static_cast<size_t>(n) * vb));
        raw_received_.fetch_add(static_cast<uint64_t>(n) * vb, std::memory_order_relaxed);
        w.Put<uint32_t>(static_cast<uint32_t>(res.installed));
        w.Put<uint32_t>(static_cast<uint32_t>(res.skipped));
        break;
      }
      case kRawAdd: {
        const uint32_t n = r.Get<uint32_t>();
        const uint32_t vb = r.Get<uint32_t>();
        Require(vb == e.vec_bytes(), "vector size does not match this node's index");
        Require(n <= kMaxRawBatch && static_cast<uint64_t>(n) * vb <= kMaxFrameBytes / 2,
                "RAW_ADD batch too large");
        const auto locs = TakeArray<uint32_t>(&r, n);
        Require(r.remaining() == static_cast<size_t>(n) * vb, "RAW_ADD: vectors do not match the locations");
        const fusion::RawPutResult res = e.RawAdd(locs.data(), n, r.Bytes(static_cast<size_t>(n) * vb));
        w.Put<uint32_t>(static_cast<uint32_t>(res.installed));
        w.Put<uint32_t>(static_cast<uint32_t>(res.skipped));
        break;
      }
      case kPauseUpdates:
        e.PauseUpdateWrites(r.Get<uint8_t>() != 0);
        break;
      case kRawTraffic:
        Require(r.remaining() == 0, "trailing bytes in RAW_TRAFFIC");
        w.Put<uint64_t>(raw_sent_.load(std::memory_order_relaxed));
        w.Put<uint64_t>(raw_received_.load(std::memory_order_relaxed));
        break;
      case kRawPull: {
        const uint32_t n = r.Get<uint32_t>();
        Require(n <= kMaxRawBatch && static_cast<uint64_t>(n) * e.vec_bytes() <= kMaxFrameBytes / 2,
                "RAW_PULL batch too large");  // what it asks a source for must fit a RAW_GET reply
        const auto locs = TakeArray<uint32_t>(&r, n);
        const auto lists = TakeArray<uint32_t>(&r, n);
        Require(r.remaining() == 0, "trailing bytes in RAW_PULL");
        const fusion::RawPutResult res = e.RawPull(locs.data(), lists.data(), n);
        w.Put<uint32_t>(static_cast<uint32_t>(res.installed));
        w.Put<uint32_t>(static_cast<uint32_t>(res.skipped));
        break;
      }
      case kNavigate: {
        const uint32_t nprobe = r.Get<uint32_t>();
        const uint32_t ef = r.Get<uint32_t>();
        const uint32_t graph = r.Get<uint32_t>();
        Require(nprobe >= 1 && nprobe <= o.max_nprobe, "nprobe out of range");
        const auto q = TakeQuery(&r, e);
        std::vector<uint32_t> lists(nprobe);
        const uint32_t n = e.Navigate(q.data(), nprobe, ef, lists.data(), graph);
        w.Put(n);
        w.PutBytes(lists.data(), n * 4);
        break;
      }
      case kFilter: {
        const uint32_t topn = r.Get<uint32_t>();
        const uint32_t nlists = r.Get<uint32_t>();
        Require(topn >= 1 && topn <= o.max_rerank, "topn out of range");
        Require(nlists <= o.max_nprobe, "too many lists");
        const auto lists = TakeArray<uint32_t>(&r, nlists);
        const auto versions = TakeArray<uint32_t>(&r, nlists);
        const auto q = TakeQuery(&r, e);
        std::vector<uint32_t> ids(topn);
        std::vector<float> dists(topn);
        fusion::FilterStats fs;
        worker = AcquireWorker(&st);
        const uint32_t n = e.Filter(worker, q.data(), lists.data(), versions.data(), nlists, topn,
                                    ids.data(), dists.data(), &fs);
        w.Put(n);
        w.PutBytes(ids.data(), n * 4);
        w.PutBytes(dists.data(), n * 4);
        w.Put(fs.gathered);
        w.Put(fs.unique);
        break;
      }
      case kRerank: {
        const uint32_t k = r.Get<uint32_t>();
        const uint32_t n = r.Get<uint32_t>();
        Require(k >= 1 && k <= o.max_rerank, "k out of range");
        Require(n <= o.max_rerank, "too many rerank candidates");
        const auto ids = TakeArray<uint32_t>(&r, n);
        const uint32_t nlists = r.Get<uint32_t>();
        Require(nlists <= o.max_nprobe, "too many lists");
        const auto lists = TakeArray<uint32_t>(&r, nlists);
        const auto versions = TakeArray<uint32_t>(&r, nlists);
        const auto q = TakeQuery(&r, e);
        std::vector<uint32_t> out_ids(k);
        std::vector<float> out_d(k);
        fusion::RerankStats rs;
        // The vectors first, holding no worker: Rerank then finds them here.
        double fetch_us = 0;
        const uint32_t fetched = e.FetchForRerank(ids.data(), n, lists.data(), versions.data(), nlists, &fetch_us);
        worker = AcquireWorker(&st);
        const uint32_t cnt = e.Rerank(worker, q.data(), ids.data(), n, k, lists.data(),
                                      versions.data(), nlists, out_ids.data(), out_d.data(), &rs);
        rs.fetched += fetched;
        rs.fetch_us += fetch_us;
        w.Put(cnt);
        w.PutBytes(out_ids.data(), cnt * 4);
        w.PutBytes(out_d.data(), cnt * 4);
        w.Put(rs.pages_read);
        w.Put(rs.fetched);
        break;
      }
      case kAssign: {
        const uint32_t n = r.Get<uint32_t>();
        const size_t qb = static_cast<size_t>(e.dim()) * fusion::DTypeSize(e.dtype());
        const uint32_t rep = e.assign_replicas(), m = e.pq_m();
        Require(n >= 1 && static_cast<uint64_t>(n) * (qb + 4 * rep + m) <= kMaxFrameBytes / 2,
                "ASSIGN batch too large");
        Require(r.remaining() == static_cast<size_t>(n) * qb, "ASSIGN: body is not n vectors");
        std::vector<uint64_t> vecs((static_cast<size_t>(n) * qb + 7) / 8);
        std::memcpy(vecs.data(), r.Bytes(static_cast<size_t>(n) * qb), static_cast<size_t>(n) * qb);
        std::vector<uint32_t> lists(static_cast<size_t>(n) * rep);
        std::vector<uint8_t> codes(static_cast<size_t>(n) * m);
        e.Assign(vecs.data(), n, lists.data(), codes.data());
        w.Put<uint32_t>(rep);
        w.Put<uint32_t>(m);
        w.PutBytes(lists.data(), lists.size() * 4);
        w.PutBytes(codes.data(), codes.size());
        break;
      }
      case kStageVersions: {
        const uint32_t np = r.Get<uint32_t>();
        Require(np <= e.num_partitions(), "STAGE_VERSIONS: too many partitions");
        std::vector<fusion::VersionStage> parts(np);
        for (auto& vs : parts) {
          vs.partition = r.Get<uint32_t>();
          vs.base = r.Get<uint32_t>();
          vs.next = r.Get<uint32_t>();
        }
        const uint32_t na = r.Get<uint32_t>();
        Require(static_cast<uint64_t>(na) * 14 <= r.remaining(), "STAGE_VERSIONS: postings truncated");
        const auto lists = TakeArray<uint32_t>(&r, na);
        const auto ids = TakeArray<uint32_t>(&r, na);
        const auto locs = TakeArray<uint32_t>(&r, na);
        const auto holders = TakeArray<uint16_t>(&r, na);
        const uint32_t nd = r.Get<uint32_t>();
        Require(static_cast<uint64_t>(nd) * 4 <= r.remaining(), "STAGE_VERSIONS: deleted IDs truncated");
        const auto del = TakeArray<uint32_t>(&r, nd);
        const uint32_t nc = r.Get<uint32_t>();
        Require(static_cast<uint64_t>(nc) * 8 <= r.remaining(), "STAGE_VERSIONS: created lists truncated");
        std::vector<uint32_t> created(nc), created_parts(nc);
        for (uint32_t i = 0; i < nc; ++i) {
          created[i] = r.Get<uint32_t>();
          created_parts[i] = r.Get<uint32_t>();
        }
        const uint32_t nr = r.Get<uint32_t>();
        Require(static_cast<uint64_t>(nr) * 4 == r.remaining(), "STAGE_VERSIONS: removed lists do not match");
        const auto removed = TakeArray<uint32_t>(&r, nr);
        fusion::VersionEdits ed;
        ed.lists = lists.data();
        ed.ids = ids.data();
        ed.locs = locs.data();
        ed.holders = holders.data();
        ed.nadds = na;
        ed.deleted = del.data();
        ed.ndeleted = nd;
        ed.created = created.data();
        ed.created_parts = created_parts.data();
        ed.ncreated = nc;
        ed.removed = removed.data();
        ed.nremoved = nr;
        const auto res = e.StageVersions(parts, ed);
        w.Put<uint32_t>(static_cast<uint32_t>(res.size()));
        for (const auto& x : res) {
          w.Put<uint32_t>(x.partition);
          w.Put<uint8_t>(x.changed ? 1 : 0);
          w.Put<uint8_t>(0);
          w.Put<uint8_t>(0);
          w.Put<uint8_t>(0);
          w.Put<uint32_t>(static_cast<uint32_t>(x.added));
          w.Put<uint32_t>(static_cast<uint32_t>(x.removed));
          w.Put<uint32_t>(static_cast<uint32_t>(x.lengths.size()));
          for (const auto& l : x.lengths) {
            w.Put<uint32_t>(l.first);
            w.Put<uint32_t>(l.second);
          }
        }
        break;
      }
      case kListGet: {
        const uint32_t list = r.Get<uint32_t>();
        const uint32_t version = r.Get<uint32_t>();
        const fusion::ListPostings lp = e.GetList(list, version);
        w.Put<uint32_t>(static_cast<uint32_t>(lp.ids.size()));
        w.PutBytes(lp.ids.data(), lp.ids.size() * 4);
        w.PutBytes(lp.locs.data(), lp.locs.size() * 4);
        w.PutBytes(lp.holders.data(), lp.holders.size() * 2);
        break;
      }
      case kPatchStage: {
        const size_t n = r.remaining();
        e.StageGraphPatch(r.Bytes(n), n);
        break;
      }
      case kPatchLoad: {
        const std::vector<uint8_t> b = fusion::ReadFile(r.Rest());
        e.StageGraphPatch(b.data(), b.size());
        break;
      }
      case kPatchMerge:
        e.MergeGraphPatch(r.Get<uint32_t>());
        break;
      case kPatchDrop:
        w.Put<uint8_t>(e.DropGraphPatch(r.Get<uint32_t>()) ? 1 : 0);
        break;
      case kPatchSave: {
        const uint32_t v = r.Get<uint32_t>();
        e.SaveGraphPatch(v, r.Rest());
        break;
      }
      case kSplitList: {
        const uint32_t n = r.Get<uint32_t>();
        const uint32_t m = r.Get<uint32_t>();
        Require(m == e.pq_m(), "SPLIT_LIST: codes of the wrong size");
        Require(static_cast<uint64_t>(n) * m == r.remaining(), "SPLIT_LIST: body is not n codes");
        std::vector<uint8_t> side(n);
        uint32_t heads[2] = {0, 0};
        e.SplitCodes(r.Bytes(static_cast<size_t>(n) * m), n, side.data(), heads);
        w.Put<uint32_t>(heads[0]);
        w.Put<uint32_t>(heads[1]);
        w.PutBytes(side.data(), side.size());
        break;
      }
      case kNearestHeads: {
        const uint32_t n = r.Get<uint32_t>();
        const uint32_t m = r.Get<uint32_t>();
        const uint32_t ef = r.Get<uint32_t>();
        Require(m == e.pq_m(), "NEAREST_HEADS: codes of the wrong size");
        Require(static_cast<uint64_t>(n) * m == r.remaining(), "NEAREST_HEADS: body is not n codes");
        std::vector<uint32_t> lists(n);
        e.NearestHeads(r.Bytes(static_cast<size_t>(n) * m), n, ef, lists.data());
        w.PutBytes(lists.data(), lists.size() * 4);
        break;
      }
      case kEditGraph: {
        const uint32_t version = r.Get<uint32_t>();
        const uint32_t nn = r.Get<uint32_t>();
        const size_t qb = static_cast<size_t>(e.dim()) * fusion::DTypeSize(e.dtype());
        Require(static_cast<uint64_t>(nn) * (4 + qb) <= r.remaining(), "EDIT_GRAPH: heads truncated");
        const auto labels = TakeArray<uint32_t>(&r, nn);
        std::vector<uint64_t> vecs((static_cast<size_t>(nn) * qb + 7) / 8);
        if (nn) std::memcpy(vecs.data(), r.Bytes(static_cast<size_t>(nn) * qb), static_cast<size_t>(nn) * qb);
        const uint32_t nd = r.Get<uint32_t>();
        Require(static_cast<uint64_t>(nd) * 4 == r.remaining(), "EDIT_GRAPH: deleted heads do not match");
        const auto del = TakeArray<uint32_t>(&r, nd);
        const std::vector<uint8_t> patch = e.EditGraph(version, labels.data(), vecs.data(), nn, del.data(), nd);
        Require(patch.size() <= kMaxFrameBytes - 64, "EDIT_GRAPH: the patch does not fit in a frame");
        w.PutBytes(patch.data(), patch.size());
        break;
      }
      case kDropVersion: {
        const uint32_t p = r.Get<uint32_t>();
        const uint32_t v = r.Get<uint32_t>();
        w.Put<uint8_t>(e.DropVersion(p, v) ? 1 : 0);
        break;
      }
      case kForget: {
        const uint32_t flags = r.Get<uint32_t>();
        const uint32_t n = r.Get<uint32_t>();
        Require(static_cast<uint64_t>(n) * 4 == r.remaining(), "FORGET: IDs do not match");
        const auto ids = TakeArray<uint32_t>(&r, n);
        w.Put<uint64_t>(e.ForgetVectors(ids.data(), n, (flags & 1) != 0));
        break;
      }
      case kSavePartition: {
        const uint32_t p = r.Get<uint32_t>();
        const uint32_t v = r.Get<uint32_t>();
        e.SavePartition(p, v, r.Rest());
        break;
      }
      case kSearchLocal: {
        fusion::SearchParams sp;
        sp.k = r.Get<uint32_t>();
        sp.nprobe = r.Get<uint32_t>();
        sp.rerank = r.Get<uint32_t>();
        sp.graph_ef = r.Get<uint32_t>();
        sp.rr.heuristic = r.Get<uint8_t>() != 0;
        r.Bytes(3);
        Require(sp.k >= 1 && sp.k <= o.max_rerank, "k out of range");
        Require(sp.nprobe >= 1 && sp.nprobe <= o.max_nprobe, "nprobe out of range");
        Require(sp.rerank >= 1 && sp.rerank <= o.max_rerank, "rerank out of range");
        const auto q = TakeQuery(&r, e);
        std::vector<uint32_t> ids(sp.k);
        std::vector<float> dists(sp.k);
        fusion::QueryStats qs;
        worker = AcquireWorker(&st);
        const uint32_t n = e.SearchLocal(worker, q.data(), sp, ids.data(), dists.data(), &qs);
        w.Put(n);
        w.PutBytes(ids.data(), n * 4);
        w.PutBytes(dists.data(), n * 4);
        break;
      }
      default:
        throw std::invalid_argument(fusion::StrFormat("unknown operation 0x%02x", req.type));
    }
    resp.body = std::move(w.buf);
  } catch (const fusion::NotResidentError& ex) {
    resp.status = kNotResident;
    resp.body.assign(ex.what(), ex.what() + std::strlen(ex.what()));
  } catch (const fusion::NoGraphError& ex) {
    resp.status = kNoGraph;
    resp.body.assign(ex.what(), ex.what() + std::strlen(ex.what()));
  } catch (const fusion::PQMissingError& ex) {
    resp.status = kPQAbsent;
    resp.body.assign(ex.what(), ex.what() + std::strlen(ex.what()));
  } catch (const fusion::PQCapacityError& ex) {
    resp.status = kPQFull;
    resp.body.assign(ex.what(), ex.what() + std::strlen(ex.what()));
  } catch (const fusion::RawMissingError& ex) {
    resp.status = kRawAbsent;
    resp.body.assign(ex.what(), ex.what() + std::strlen(ex.what()));
  } catch (const std::invalid_argument& ex) {
    resp.status = kBadRequest;
    resp.body.assign(ex.what(), ex.what() + std::strlen(ex.what()));
  } catch (const std::exception& ex) {
    resp.status = kInternal;
    resp.body.assign(ex.what(), ex.what() + std::strlen(ex.what()));
  }
  if (resp.status != kOk) st.errors++;
  st.busy_us += static_cast<uint64_t>(timer.Us());
  return resp;
}

std::string NodeServer::InfoJson(bool settle) const {
  const fusion::NodeEngine& e = *engine_;
  const auto& o = e.options();
  std::string s = "{";
  s += fusion::StrFormat(
      "\"dtype\":\"%s\",\"dim\":%u,\"num_vectors\":%llu,\"num_lists\":%u,\"num_partitions\":%u,"
      "\"max_nprobe\":%u,\"max_rerank\":%u,\"workers\":%d,\"filter\":\"%s\",\"payload\":\"%s\","
      "\"graph_loaded\":%s,\"max_epoch_seen\":%llu,\"id_capacity\":%llu,\"graph\":%s,"
      "\"generation\":%u,\"pid\":%d,",
      fusion::DTypeName(e.dtype()), e.dim(), static_cast<unsigned long long>(e.size()),
      e.num_lists(), e.num_partitions(), o.max_nprobe, o.max_rerank, e.num_workers(),
      e.filter_name(), e.payload().c_str(), e.has_graph() ? "true" : "false",
      static_cast<unsigned long long>(max_epoch_seen_.load()),
      static_cast<unsigned long long>(e.id_capacity()), GraphJson().c_str(), e.generation(),
      static_cast<int>(::getpid()));
  s += "\"resident\":[";
  const auto res = e.ResidentPartitions();
  for (size_t i = 0; i < res.size(); ++i) s += (i ? "," : "") + std::to_string(res[i]);
  s += "],\"versions\":[";
  const auto vers = e.ResidentVersions();
  for (size_t i = 0; i < vers.size(); ++i)
    s += fusion::StrFormat("%s[%u,%u]", i ? "," : "", vers[i].first, vers[i].second);
  const fusion::PQStats pq = e.pq_stats();
  auto u = [](uint64_t v) { return static_cast<unsigned long long>(v); };
  s += fusion::StrFormat(
      "],\"pq\":{\"m\":%u,\"capacity\":%llu,\"resident\":%llu,\"live\":%llu,\"staged\":%llu,"
      "\"cached\":%llu,\"received\":%llu,\"from_index\":%llu,\"skipped\":%llu,\"evicted\":%llu,"
      "\"freed\":%llu,\"served\":%llu,\"deleted\":%llu}",
      pq.m, u(pq.capacity), u(pq.resident), u(pq.live), u(pq.staged), u(pq.cached),
      u(pq.received), u(pq.from_index), u(pq.skipped), u(pq.evicted), u(pq.freed), u(pq.served),
      u(pq.deleted));
  const fusion::RawStats raw = e.raw_stats(settle);
  s += fusion::StrFormat(
      ",\"raw\":{\"vec_bytes\":%u,\"locations\":%llu,\"present\":%llu,\"cached\":%llu,"
      "\"pending\":%llu,\"from_index\":%llu,\"streamed\":%llu,\"fetched\":%llu,\"added\":%llu,"
      "\"fetches\":%llu,\"skipped\":%llu,\"served\":%llu}",
      raw.vec_bytes, u(raw.locations), u(raw.present), u(raw.cached), u(raw.pending), u(raw.from_index),
      u(raw.streamed), u(raw.fetched), u(raw.added), u(raw.fetches), u(raw.skipped), u(raw.served));
  s += ",\"fetch\":" + FetchJson() + ",\"ops\":" + OpsJson() + "}";
  return s;
}

std::string NodeServer::GraphJson() const {
  const fusion::GraphState g = engine_->graph_state();
  return fusion::StrFormat("{\"version\":%u,\"patch\":%u,\"count\":%llu,\"capacity\":%llu,\"deleted\":%llu}",
                           g.version, g.patch, static_cast<unsigned long long>(g.count),
                           static_cast<unsigned long long>(g.capacity), static_cast<unsigned long long>(g.deleted));
}

std::string NodeServer::OpsJson() const {
  std::string s = "{";
  bool first = true;
  for (int op = 0; op < 256; ++op) {
    const char* name = OpName(static_cast<uint8_t>(op));
    if (!name) continue;
    s += fusion::StrFormat("%s\"%s\":{\"count\":%llu,\"errors\":%llu,\"busy_us\":%llu,\"wait_us\":%llu}",
                           first ? "" : ",", name,
                           static_cast<unsigned long long>(stats_[op].count.load()),
                           static_cast<unsigned long long>(stats_[op].errors.load()),
                           static_cast<unsigned long long>(stats_[op].busy_us.load()),
                           static_cast<unsigned long long>(stats_[op].wait_us.load()));
    first = false;
  }
  return s + "}";
}

std::string NodeServer::FetchJson() const {
  const fusion::FetchStats f = engine_->fetch_stats();
  auto u = [](uint64_t v) { return static_cast<unsigned long long>(v); };
  return fusion::StrFormat(
      "{\"present\":%llu,\"cached\":%llu,\"queued\":%llu,\"queued_peak\":%llu,\"fetched\":%llu,"
      "\"fetch_calls\":%llu,\"coalesced\":%llu,\"fetch_us\":%llu,\"written\":%llu,\"write_batches\":%llu,"
      "\"write_us\":%llu,\"sync_installed\":%llu,\"sync_us\":%llu,\"drain_calls\":%llu,"
      "\"drain_waiting\":%llu,\"drain_us\":%llu,\"rerank_disk\":%llu,\"rerank_mem\":%llu,"
      "\"pending_recounts\":%llu,\"pending_us\":%llu,\"added\":%llu,\"added_queued\":%llu,"
      "\"added_written\":%llu,\"added_wait_us\":%llu}",
      u(f.present), u(f.cached), u(f.queued), u(f.queued_peak), u(f.fetched), u(f.fetch_calls),
      u(f.coalesced), u(f.fetch_us), u(f.written), u(f.write_batches), u(f.write_us), u(f.sync_installed),
      u(f.sync_us), u(f.drain_calls), u(f.drain_waiting), u(f.drain_us), u(f.rerank_disk),
      u(f.rerank_mem), u(f.pending_recounts), u(f.pending_us), u(f.added), u(f.added_queued),
      u(f.added_written), u(f.added_wait_us));
}

std::string NodeServer::StatsJson() const {
  const fusion::ReadBudgetStats r = engine_->read_budget_stats();
  return "{\"fetch\":" + FetchJson() + ",\"ops\":" + OpsJson() +
         fusion::StrFormat(",\"reads\":{\"budget\":%llu,\"charged\":%llu,\"wait_us\":%llu}"
                           ",\"raw_bytes\":{\"sent\":%llu,\"received\":%llu}}",
                           static_cast<unsigned long long>(r.reads_per_sec),
                           static_cast<unsigned long long>(r.charged),
                           static_cast<unsigned long long>(r.wait_us),
                           static_cast<unsigned long long>(raw_sent_.load(std::memory_order_relaxed)),
                           static_cast<unsigned long long>(raw_received_.load(std::memory_order_relaxed)));
}

}  // namespace rtier
