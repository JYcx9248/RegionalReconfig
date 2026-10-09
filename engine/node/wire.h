// rtier wire protocol: length-prefixed frames over TCP or Unix sockets.
//
// The framing follows Koala's state-comm TCP API (MAGIC_START | length | type | body |
// MAGIC_END), extended with a request ID (so responses can be matched), an epoch (the placement
// epoch the sender routed with) and a status field. Everything is little endian, so vectors
// and ID arrays are copied without conversion. The Go side is internal/frame.
//
//   off  size  field
//   0    4     magic_start   0x31465452 ("RTF1")
//   4    4     length        bytes after this field: 20 + body + 4
//   8    1     type          operation (Op)
//   9    1     flags         bit 0: response
//   10   2     status        0 = OK (responses only; see Status)
//   12   8     req_id        echoed in the response
//   20   8     epoch         placement epoch of the sender (0 = none)
//   28   ...   body
//   end  4     magic_end     0x444E4546 ("FEND")
//
// Bodies of the data-node operations (u32 = uint32 little endian; "query" = dim elements of
// the index dtype; "str" = u32 length + bytes; error responses carry a UTF-8 message instead):
//   PING            -                                         -> -
//   INFO            [u8 flags]                                -> JSON text
//                   flags bit 0 (kInfoSettle): wait until the raw vectors fetched before the
//                   request are on the SSD (tests); without it INFO waits for nothing
//   LOAD_GRAPH      path                                      -> -
//   LOAD_PARTITION  u32 partition, u8 pq_source, str raw_peer, -> -
//                   path
//                   pq_source 0: every PQ code must already be on the node (PQ_PUT); raw
//                                vectors not here are fetched from raw_peer (a data node,
//                                "host:port", the source of the partition's lists) when a
//                                query needs them
//                             1: bootstrap: copy the missing codes and raw vectors from the index
//   EVICT_PARTITION u32 partition                             -> u8 was_resident
//   PQ_MISSING      u32 groups, per group: u32 paths,         -> u32 groups, per group:
//                   str path[paths] (segment files)              u32 n, u32 ids[n]
//                   (the codes already here are protected for the staging)
//   PQ_GET          u32 n, u32 ids[n]                         -> u32 m, u8 codes[n * m]
//   PQ_PUT          u32 n, u32 m, u32 ids[n], u8 codes[n * m] -> u32 installed, u32 skipped
//   PQ_RELEASE      -                                         -> u64 freed (a staging that did
//                                                                not load: its new codes)
//   RAW_MISSING     u32 groups, per group: u32 paths,         -> u32 groups, per group:
//                   str path[paths] (segment files)              u32 n, u32 locs[n], u32 lists[n]
//                                                                (a list naming each location)
//   RAW_GET         u32 n, u32 locs[n], u32 lists[n]          -> u32 vec_bytes, u8 vecs[n * vb]
//                   (lists[i] names locs[i]; a vector not here is fetched from that list's
//                   source on this node first: chained migrations)
//   RAW_PUT         u32 n, u32 vec_bytes, u32 locs[n],        -> u32 installed, u32 skipped
//                   u8 vecs[n * vec_bytes]
//   RAW_CHECK       u32 n, u32 locs[n]                        -> u32 m, u32 locs[m] (those not
//                                                                here, sorted, unique)
//   RAW_PULL        u32 n, u32 locs[n], u32 lists[n]          -> u32 installed, u32 skipped
//                   (the node fetches what it lacks from the lists' sources with RAW_GET and
//                   installs it: a stream that does not pass through the agents)
//   NAVIGATE        u32 nprobe, u32 ef, query                 -> u32 n, u32 lists[n]
//   FILTER          u32 topn, u32 nlists, u32 lists[], query  -> u32 n, u32 ids[n], f32 dists[n],
//                                                                u32 gathered, u32 unique
//   RERANK          u32 k, u32 n, u32 ids[n], u32 nlists,     -> u32 n, u32 ids[n], f32 dists[n],
//                   u32 lists[nlists], query                     u32 pages_read, u32 fetched
//                   (lists: the query's lists on this node; a missing raw vector is fetched
//                   from the source of one of them that names it)
//   SEARCH_LOCAL    u32 k, u32 nprobe, u32 rerank, u32 ef,    -> u32 n, u32 ids[n], f32 dists[n]
//                   u8 heuristic, u8[3] pad, query
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace rtier {

constexpr uint32_t kMagicStart = 0x31465452;  // "RTF1"
constexpr uint32_t kMagicEnd = 0x444E4546;    // "FEND"
constexpr uint32_t kHeaderBytes = 28;         // through epoch
constexpr uint32_t kMaxFrameBytes = 256u << 20;
constexpr uint8_t kFlagResponse = 1;
constexpr uint8_t kInfoSettle = 1;  // INFO flags
constexpr uint32_t kMaxPQBatch = 1u << 22;  // IDs per PQ_GET / PQ_PUT
constexpr uint32_t kMaxRawBatch = 1u << 20;  // locations per RAW_GET / RAW_PUT / RAW_PULL (also bytes / 2)

enum Op : uint8_t {
  kPing = 0x01,
  kInfo = 0x02,
  kLoadGraph = 0x10,
  kLoadPartition = 0x11,
  kEvictPartition = 0x12,
  kPQMissing = 0x13,
  kPQGet = 0x14,
  kPQPut = 0x15,
  kPQRelease = 0x16,
  kRawMissing = 0x17,
  kRawGet = 0x18,
  kRawPut = 0x19,
  kRawCheck = 0x1A,
  kRawPull = 0x1B,
  kNavigate = 0x20,
  kFilter = 0x21,
  kRerank = 0x22,
  kSearchLocal = 0x23,
};

enum Status : uint16_t {
  kOk = 0,
  kBadRequest = 1,
  kNotResident = 2,
  kNoGraph = 3,
  kInternal = 4,
  kUndecided = 5,  // hit a TODO(design) placeholder
  // 6 is UNAVAILABLE on the agents' query port (internal/query/wire.go)
  kPQAbsent = 7,  // a PQ code the operation needs is not on this node
  kPQFull = 8,    // the node's PQ budget (--pq-capacity) is exhausted
  kRawAbsent = 9, // a raw vector the operation needs is not on this node and was not fetched
};

struct Frame {
  uint8_t type = 0;
  uint8_t flags = 0;
  uint16_t status = 0;
  uint64_t req_id = 0;
  uint64_t epoch = 0;
  std::vector<uint8_t> body;
};

// Sequential reader over a body; throws std::invalid_argument when it runs out.
class BodyReader {
 public:
  BodyReader(const uint8_t* p, size_t n) : p_(p), n_(n) {}
  template <class T>
  T Get() {
    T v;
    Need(sizeof(T));
    std::memcpy(&v, p_ + off_, sizeof(T));
    off_ += sizeof(T);
    return v;
  }
  const uint8_t* Bytes(size_t n) {
    Need(n);
    const uint8_t* r = p_ + off_;
    off_ += n;
    return r;
  }
  std::string Rest() {
    std::string s(reinterpret_cast<const char*>(p_ + off_), n_ - off_);
    off_ = n_;
    return s;
  }
  size_t remaining() const { return n_ - off_; }

 private:
  void Need(size_t n) const {
    if (off_ + n > n_) throw std::invalid_argument("request body too short");
  }
  const uint8_t* p_;
  size_t n_;
  size_t off_ = 0;
};

class BodyWriter {
 public:
  template <class T>
  void Put(T v) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
    buf.insert(buf.end(), b, b + sizeof(T));
  }
  void PutBytes(const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    buf.insert(buf.end(), b, b + n);
  }
  std::vector<uint8_t> buf;
};

// Blocking frame I/O on a socket. ReadFrame returns false on a clean EOF before a frame starts;
// it throws std::runtime_error on malformed frames or I/O errors.
bool ReadFrame(int fd, Frame* f);
void WriteFrame(int fd, const Frame& f);

}  // namespace rtier
