#include "server.h"

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
    case kNavigate: return "navigate";
    case kFilter: return "filter";
    case kRerank: return "rerank";
    case kSearchLocal: return "search_local";
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
  std::lock_guard<std::mutex> l(mu_);
  idle_[addr].push_back(fd);
}

void PeerPool::RawGet(const std::string& addr, const uint32_t* locs, const uint32_t* lists,
                      size_t n, uint32_t vec_bytes, uint8_t* out) {
  for (size_t done = 0; done < n;) {  // frames stay far below kMaxFrameBytes
    const size_t batch = std::min<size_t>(
        {n - done, kMaxRawBatch, std::max<size_t>(1, (kMaxFrameBytes / 4) / vec_bytes)});
    Frame req;
    req.type = kRawGet;
    req.req_id = next_id_.fetch_add(1);
    BodyWriter w;
    w.Put<uint32_t>(static_cast<uint32_t>(batch));
    w.PutBytes(locs + done, batch * 4);
    w.PutBytes(lists + done, batch * 4);
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
  engine->SetRawFetcher([this, vb](const std::string& peer, const uint32_t* locs,
                                   const uint32_t* lists, size_t n, uint8_t* out) {
    peers_.RawGet(peer, locs, lists, n, vb, out);
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
  std::vector<std::thread> conns;
  {
    std::lock_guard<std::mutex> l(conn_mu_);
    conns.swap(conn_threads_);
  }
  for (auto& t : conns) t.join();
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
      if (errno == EINTR) continue;
      break;  // listener shut down
    }
    const int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));  // fails harmlessly on AF_UNIX
    std::lock_guard<std::mutex> l(conn_mu_);
    if (stop_) {
      ::close(fd);
      break;
    }
    conn_fds_.insert(fd);
    conn_threads_.emplace_back([this, fd] { ServeConnection(fd); });
  }
  ::close(listen_fd);
}

void NodeServer::ServeConnection(int fd) {
  try {
    Frame req;
    while (!stop_ && ReadFrame(fd, &req)) WriteFrame(fd, Handle(req));
  } catch (const std::exception& e) {
    if (!stop_) fusion::Log("connection closed: %s", e.what());
  }
  std::lock_guard<std::mutex> l(conn_mu_);
  conn_fds_.erase(fd);
  ::close(fd);
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

Frame NodeServer::Handle(const Frame& req) {
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
        const std::string s = InfoJson();
        w.PutBytes(s.data(), s.size());
        break;
      }
      case kLoadGraph:
        e.LoadGraph(r.Rest());
        break;
      case kLoadPartition: {
        const uint32_t p = r.Get<uint32_t>();
        const uint8_t src = r.Get<uint8_t>();
        Require(src <= 1, "unknown PQ source");
        const std::string raw_peer = TakeStr(&r);
        e.LoadPartition(p, r.Rest(), src == 1 ? fusion::PQSource::kIndex : fusion::PQSource::kPresent,
                        raw_peer);
        break;
      }
      case kEvictPartition:
        w.Put<uint8_t>(e.EvictPartition(r.Get<uint32_t>()) ? 1 : 0);
        break;
      case kPQMissing: {
        const auto groups = TakeGroups(&r);
        const uint32_t ngroups = static_cast<uint32_t>(groups.size());
        const auto miss = e.PQMissing(groups);
        uint64_t bytes = 4;
        for (const auto& m : miss) bytes += 4 + 4 * static_cast<uint64_t>(m.size());
        Require(bytes + 64 <= kMaxFrameBytes,
                "the missing IDs do not fit in one frame; stage fewer partitions per call");
        w.Put<uint32_t>(ngroups);
        for (const auto& m : miss) {
          w.Put<uint32_t>(static_cast<uint32_t>(m.size()));
          w.PutBytes(m.data(), m.size() * 4);
        }
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
        const auto groups = TakeGroups(&r);
        const auto miss = e.RawMissing(groups);
        uint64_t bytes = 4;
        for (const auto& m : miss) bytes += 4 + 8 * static_cast<uint64_t>(m.locs.size());
        Require(bytes + 64 <= kMaxFrameBytes,
                "the missing locations do not fit in one frame; ask for fewer partitions per call");
        w.Put<uint32_t>(static_cast<uint32_t>(groups.size()));
        for (const auto& m : miss) {
          w.Put<uint32_t>(static_cast<uint32_t>(m.locs.size()));
          w.PutBytes(m.locs.data(), m.locs.size() * 4);
          w.PutBytes(m.lists.data(), m.lists.size() * 4);
        }
        break;
      }
      case kRawGet: {
        const uint32_t n = r.Get<uint32_t>();
        const uint32_t vb = e.vec_bytes();
        Require(n <= kMaxRawBatch && static_cast<uint64_t>(n) * vb <= kMaxFrameBytes / 2,
                "RAW_GET batch too large");
        const auto locs = TakeArray<uint32_t>(&r, n);
        const auto lists = TakeArray<uint32_t>(&r, n);
        Require(r.remaining() == 0, "trailing bytes in RAW_GET");
        std::vector<uint8_t> vecs(static_cast<size_t>(n) * vb);
        e.RawGet(locs.data(), lists.data(), n, vecs.data());
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
        w.Put<uint32_t>(static_cast<uint32_t>(res.installed));
        w.Put<uint32_t>(static_cast<uint32_t>(res.skipped));
        break;
      }
      case kNavigate: {
        const uint32_t nprobe = r.Get<uint32_t>();
        const uint32_t ef = r.Get<uint32_t>();
        Require(nprobe >= 1 && nprobe <= o.max_nprobe, "nprobe out of range");
        const auto q = TakeQuery(&r, e);
        std::vector<uint32_t> lists(nprobe);
        const uint32_t n = e.Navigate(q.data(), nprobe, ef, lists.data());
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
        const auto q = TakeQuery(&r, e);
        std::vector<uint32_t> ids(topn);
        std::vector<float> dists(topn);
        fusion::FilterStats fs;
        worker = AcquireWorker(&st);
        const uint32_t n =
            e.Filter(worker, q.data(), lists.data(), nlists, topn, ids.data(), dists.data(), &fs);
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
        const auto q = TakeQuery(&r, e);
        std::vector<uint32_t> out_ids(k);
        std::vector<float> out_d(k);
        fusion::RerankStats rs;
        worker = AcquireWorker(&st);
        const uint32_t cnt = e.Rerank(worker, q.data(), ids.data(), n, k, lists.data(), nlists,
                                      out_ids.data(), out_d.data(), &rs);
        w.Put(cnt);
        w.PutBytes(out_ids.data(), cnt * 4);
        w.PutBytes(out_d.data(), cnt * 4);
        w.Put(rs.pages_read);
        w.Put(rs.fetched);
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

std::string NodeServer::InfoJson() const {
  const fusion::NodeEngine& e = *engine_;
  const auto& o = e.options();
  std::string s = "{";
  s += fusion::StrFormat(
      "\"dtype\":\"%s\",\"dim\":%u,\"num_vectors\":%llu,\"num_lists\":%u,\"num_partitions\":%u,"
      "\"max_nprobe\":%u,\"max_rerank\":%u,\"workers\":%d,\"filter\":\"%s\",\"payload\":\"%s\","
      "\"graph_loaded\":%s,\"max_epoch_seen\":%llu,",
      fusion::DTypeName(e.dtype()), e.dim(), static_cast<unsigned long long>(e.size()),
      e.num_lists(), e.num_partitions(), o.max_nprobe, o.max_rerank, e.num_workers(),
      e.filter_name(), e.payload().c_str(), e.has_graph() ? "true" : "false",
      static_cast<unsigned long long>(max_epoch_seen_.load()));
  s += "\"resident\":[";
  const auto res = e.ResidentPartitions();
  for (size_t i = 0; i < res.size(); ++i) s += (i ? "," : "") + std::to_string(res[i]);
  const fusion::PQStats pq = e.pq_stats();
  auto u = [](uint64_t v) { return static_cast<unsigned long long>(v); };
  s += fusion::StrFormat(
      "],\"pq\":{\"m\":%u,\"capacity\":%llu,\"resident\":%llu,\"live\":%llu,\"staged\":%llu,"
      "\"cached\":%llu,\"received\":%llu,\"from_index\":%llu,\"skipped\":%llu,\"evicted\":%llu,"
      "\"freed\":%llu,\"served\":%llu}",
      pq.m, u(pq.capacity), u(pq.resident), u(pq.live), u(pq.staged), u(pq.cached),
      u(pq.received), u(pq.from_index), u(pq.skipped), u(pq.evicted), u(pq.freed), u(pq.served));
  const fusion::RawStats raw = e.raw_stats();
  s += fusion::StrFormat(
      ",\"raw\":{\"vec_bytes\":%u,\"locations\":%llu,\"present\":%llu,\"pending\":%llu,"
      "\"from_index\":%llu,\"streamed\":%llu,\"fetched\":%llu,\"fetches\":%llu,"
      "\"skipped\":%llu,\"served\":%llu}",
      raw.vec_bytes, u(raw.locations), u(raw.present), u(raw.pending), u(raw.from_index),
      u(raw.streamed), u(raw.fetched), u(raw.fetches), u(raw.skipped), u(raw.served));
  s += ",\"fetch\":" + FetchJson() + ",\"ops\":" + OpsJson() + "}";
  return s;
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
      "\"fetch_calls\":%llu,\"fetch_us\":%llu,\"written\":%llu,\"write_batches\":%llu,"
      "\"write_us\":%llu,\"sync_installed\":%llu,\"sync_us\":%llu,\"drain_calls\":%llu,"
      "\"drain_waiting\":%llu,\"drain_us\":%llu,\"rerank_disk\":%llu,\"rerank_mem\":%llu,"
      "\"pending_recounts\":%llu,\"pending_us\":%llu}",
      u(f.present), u(f.cached), u(f.queued), u(f.queued_peak), u(f.fetched), u(f.fetch_calls),
      u(f.fetch_us), u(f.written), u(f.write_batches), u(f.write_us), u(f.sync_installed),
      u(f.sync_us), u(f.drain_calls), u(f.drain_waiting), u(f.drain_us), u(f.rerank_disk),
      u(f.rerank_mem), u(f.pending_recounts), u(f.pending_us));
}

std::string NodeServer::StatsJson() const {
  return "{\"fetch\":" + FetchJson() + ",\"ops\":" + OpsJson() + "}";
}

}  // namespace rtier
