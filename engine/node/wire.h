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
//   LOAD_PARTITION  u32 partition, u32 version, u8 pq_source, -> -
//                   path
//                   loads (or replaces) that version of the partition; its other versions stay
//                   pq_source 0: every PQ code must already be on the node (PQ_PUT); raw
//                                vectors not here are fetched, when a query needs them, from
//                                the holder each posting records (a segment saved at a
//                                hand-off records one for every posting)
//                             1: bootstrap: copy the missing codes and raw vectors from the index
//                             2: a generation's partition after a full rebuild (U14): copy the
//                                missing codes from the generation's index; raw vectors as 0
//   SWITCH_GENERATION u32 generation, str index_dir,         -> -
//                   str partitions_dir
//                   (full rebuild, U14: the node, holding no partition, moves to that generation
//                   of the index; its graph is dropped, its PQ codes and raw vectors stay)
//   RAW_BITMAP      -                                         -> u64 locations, u64 words[]
//                   (the raw vectors the node holds, one bit per location)
//   EVICT_PARTITION u32 partition                             -> u8 was_resident (every
//                                                                version is evicted)
//   PQ_MISSING      u32 groups, per group: u32 paths,         -> u32 groups, per group:
//                   str path[paths] (segment files)              u32 n, u32 ids[n]
//                   (the codes already here are protected for the staging)
//   PQ_GET          u32 n, u32 ids[n]                         -> u32 m, u8 codes[n * m]
//   PQ_PUT          u32 n, u32 m, u32 ids[n], u8 codes[n * m] -> u32 installed, u32 skipped
//   PQ_RELEASE      -                                         -> u64 freed (a staging that did
//                                                                not load: its new codes)
//   RAW_MISSING     u32 n, u32 partitions[n] (resident)       -> u32 groups, per group: u16 holder,
//                                                                u16 pad, u32 n, u32 locs[n],
//                                                                u32 lists[n] (a list naming each)
//                   (the vectors their newest versions name that are not here, by holder)
//   RAW_HELD_BY     u16 holder, u16 pad                       -> u64 repointed, u32 n, u32 locs[n],
//                                                                u32 lists[n]
//                   (before that node leaves: postings naming it whose vector is here now name
//                   this node; the others are returned, to be pulled with RAW_PULL)
//   RAW_GET         u32 n, u32 locs[n]                        -> u32 vec_bytes, u8 vecs[n * vb]
//                   (from this node's own store: a peer asks the holder, there is no chain)
//   RAW_PUT         u32 n, u32 vec_bytes, u32 locs[n],        -> u32 installed, u32 skipped
//                   u8 vecs[n * vec_bytes]
//   RAW_CHECK       u32 n, u32 locs[n]                        -> u32 m, u32 locs[m] (those not
//                                                                here, sorted, unique)
//   RAW_PULL        u32 n, u32 locs[n], u32 lists[n]          -> u32 installed, u32 skipped
//                   (the node fetches what it lacks from the holders its postings in those
//                   lists record, with RAW_GET, and installs it: a stream that does not pass
//                   through the agents; those postings then name this node)
//   RAW_TRAFFIC     -                                         -> u64 sent, u64 received
//                   (raw-vector bytes since startup: served by RAW_GET, and brought here --
//                   fetched from peers, on demand or by RAW_PULL, or put with RAW_PUT; an
//                   agent holds the navigation graph back while they move, U16)
//   SET_NODES       u16 self, u16 pad, u32 n,                 -> -
//                   (u16 id, u16 pad, str addr)[n]
//                   (this node's cluster ID, 0 = unchanged, and other data nodes' addresses:
//                   where the holders that postings record are)
//   RAW_ADD         u32 n, u32 vec_bytes, u32 locs[n],        -> u32 installed, u32 skipped
//                   u8 vecs[n * vec_bytes]
//                   (inserted vectors, U14: served from memory at once, written to the SSD in
//                   the background after fetched ones; blocks while the update buffer is full)
//   PAUSE_UPDATES   u8 paused                                 -> -
//                   (holds back, or resumes, writing inserted vectors)
//   NAVIGATE        u32 nprobe, u32 ef, u32 graph, query      -> u32 n, u32 lists[n]
//                   (graph: the graph version of the query's epoch, 0xFFFFFFFF = the newest
//                   here; NOT_RESIDENT if this node does not have it)
//   FILTER          u32 topn, u32 nlists, u32 lists[nlists],  -> u32 n, u32 ids[n], f32 dists[n],
//                   u32 versions[nlists], query                  u32 gathered, u32 unique
//                   (versions[i]: the version of lists[i]'s partition in the query's epoch;
//                   NOT_RESIDENT if that version is not here)
//   RERANK          u32 k, u32 n, u32 ids[n], u32 nlists,     -> u32 n, u32 ids[n], f32 dists[n],
//                   u32 lists[nlists], u32 versions[nlists],     u32 pages_read, u32 fetched
//                   query
//                   (lists: the query's lists on this node, at their versions; a missing raw
//                   vector is fetched from the holder that a posting among them records)
//   SEARCH_LOCAL    u32 k, u32 nprobe, u32 rerank, u32 ef,    -> u32 n, u32 ids[n], f32 dists[n]
//                   u8 heuristic, u8[3] pad, query
//   ASSIGN          u32 n, vectors[n]                         -> u32 r, u32 m, u32 lists[n * r],
//                                                                u8 codes[n * m]
//                   (the lists the index's build would give each vector, padded with
//                   0xFFFFFFFF, and its PQ code: the routing of inserted vectors, U14)
//   STAGE_VERSIONS  u32 nparts, (u32 partition, u32 base,     -> u32 nparts, (u32 partition,
//                   u32 next)[nparts], u32 nadds,                u8 changed, u8[3] pad,
//                   u32 lists[nadds], u32 ids[nadds],            u32 added, u32 removed,
//                   u32 locs[nadds], u16 holders[nadds],         u32 nlens, (u32 list,
//                   u32 ndel, u32 deleted[ndel], u32 ncreated,   u32 postings)[nlens])[nparts]
//                   (u32 list, u32 partition)[ncreated],
//                   u32 nremoved, u32 removed[nremoved]
//                   (an update batch, U14: each partition's next version = base - the removed
//                   lists + the created ones + the postings naming its lists - the deleted IDs;
//                   all or nothing. holders[i]: the node holding the vector if it is not here,
//                   0 = it must be here. The reply names the lists whose length changed.)
//   LIST_GET        u32 list, u32 version                     -> u32 n, u32 ids[n], u32 locs[n],
//                                                                u16 holders[n]
//                   (a posting list of a resident version, every holder by node ID)
//   PATCH_STAGE     patch                                     -> -
//                   (a graph patch, fusion/graph_patch.h, staged next to the graph: NAVIGATE of
//                   its version searches through it, of the graph's own version without it)
//   PATCH_LOAD      path                                      -> - (the same, from a file)
//   PATCH_MERGE     u32 version                               -> - (merged into the graph in place,
//                                                                after the old version's grace)
//   PATCH_DROP      u32 version                               -> u8 was_staged
//   PATCH_SAVE      u32 version, path                         -> - (the staged patch as a file)
//   SPLIT_LIST      u32 n, u32 m, u8 codes[n * m]             -> u32 head0, u32 head1, u8 side[n]
//                   (the router: a balanced 2-means of PQ codes; head s = the member closest to
//                   side s's mean)
//   NEAREST_HEADS   u32 n, u32 m, u32 ef, u8 codes[n * m]     -> u32 lists[n]
//                   (the router: the nearest live head of each decoded code)
//   EDIT_GRAPH      u32 version, u32 nnew, u32 labels[nnew],  -> patch (version - 1 -> version)
//                   vectors[nnew], u32 ndel, u32 deleted[ndel]
//                   (the router: heads inserted and marked deleted in its own graph)
//   DROP_VERSION    u32 partition, u32 version                -> u8 was_resident
//   FORGET          u32 flags, u32 n, u32 ids[n]              -> u64 freed (PQ codes no version
//                                                                names; flags bit 0: the vectors
//                                                                are deleted, and the codes a
//                                                                version still names are freed
//                                                                with their last reference)
//   SAVE_PARTITION  u32 partition, u32 version, path          -> - (that version as a segment file)
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
  kRawAdd = 0x1C,
  kPauseUpdates = 0x1D,
  kNavigate = 0x20,
  kFilter = 0x21,
  kRerank = 0x22,
  kSearchLocal = 0x23,
  kAssign = 0x24,
  kSplitList = 0x25,
  kNearestHeads = 0x26,
  kEditGraph = 0x27,
  kStageVersions = 0x30,
  kDropVersion = 0x31,
  kForget = 0x32,
  kSavePartition = 0x33,
  kSetNodes = 0x34,
  kRawHeldBy = 0x35,
  kListGet = 0x36,
  kPatchStage = 0x37,
  kPatchLoad = 0x38,
  kPatchMerge = 0x39,
  kPatchDrop = 0x3A,
  kPatchSave = 0x3B,
  kSwitchGeneration = 0x3C,
  kRawBitmap = 0x3D,
  kRawTraffic = 0x3E,
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
