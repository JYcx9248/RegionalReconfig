#include "fusion/partition.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>

namespace fusion {

namespace partfiles {
std::string SegmentName(uint32_t partition) { return StrFormat("part-%05u.seg", partition); }
}  // namespace partfiles

namespace {

constexpr uint32_t kSegMagic = 0x47535452;  // "RTSG" little endian
constexpr uint32_t kSegVersion = 2;  // 1: IDs only (payload lists-only)

struct CrcTable {
  uint32_t t[256];
  CrcTable() {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      t[i] = c;
    }
  }
};

}  // namespace

// IEEE CRC-32 (same as zlib and Go's hash/crc32.ChecksumIEEE).
uint32_t Crc32(const void* data, size_t bytes, uint32_t crc) {
  static const CrcTable table;
  const uint8_t* p = static_cast<const uint8_t*>(data);
  crc = ~crc;
  for (size_t i = 0; i < bytes; ++i) crc = table.t[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
  return ~crc;
}

PartitionManifest PartitionManifest::Load(const std::string& dir) {
  IndexMeta meta = IndexMeta::Load(JoinPath(dir, partfiles::kManifest));
  PartitionManifest m;
  m.num_lists = static_cast<uint32_t>(meta.GetU64("num_lists"));
  m.num_partitions = static_cast<uint32_t>(meta.GetU64("num_partitions"));
  m.max_list_len = static_cast<uint32_t>(meta.GetU64("max_list_len"));
  m.payload = meta.GetOr("payload", kPayloadListsOnly);
  if (m.payload == kPayloadListsLocations) {
    m.layout = meta.Get("layout");
    m.raw.page_size = static_cast<uint32_t>(meta.GetU64("page_size"));
    m.raw.vec_bytes = static_cast<uint32_t>(meta.GetU64("vec_bytes"));
    m.raw.vectors_per_page = static_cast<uint32_t>(meta.GetU64("vectors_per_page"));
    m.raw.num_pages = meta.GetU64("num_pages");
    FUSION_CHECK(m.raw.vec_bytes > 0 && m.raw.vectors_per_page >= 1 &&
                     static_cast<uint64_t>(m.raw.vec_bytes) * m.raw.vectors_per_page <= m.raw.page_size,
                 "partition manifest in %s: bad page geometry", dir.c_str());
    FUSION_CHECK(m.raw.locations() < kInvalidId, "partition manifest in %s: too many locations",
                 dir.c_str());
  }
  m.list_part = LoadU32Array(JoinPath(dir, partfiles::kListPart));
  FUSION_CHECK(m.list_part.size() == m.num_lists, "list_part.bin has %zu entries, expected %u",
               m.list_part.size(), m.num_lists);
  for (uint32_t p : m.list_part)
    FUSION_CHECK(p < m.num_partitions, "list_part.bin names partition %u >= %u", p,
                 m.num_partitions);
  return m;
}

void PartitionManifest::Save(const std::string& dir) const {
  MakeDirs(dir);
  IndexMeta meta;
  meta.Set("num_lists", num_lists);
  meta.Set("num_partitions", num_partitions);
  meta.Set("max_list_len", max_list_len);
  meta.Set("payload", payload);
  meta.Set("layout", layout);
  meta.Set("page_size", raw.page_size);
  meta.Set("vec_bytes", raw.vec_bytes);
  meta.Set("vectors_per_page", raw.vectors_per_page);
  meta.Set("num_pages", raw.num_pages);
  meta.Save(JoinPath(dir, partfiles::kManifest));
  SaveU32Array(JoinPath(dir, partfiles::kListPart), list_part);
}

const uint32_t* ListSegment::Find(uint32_t list, uint32_t* len) const {
  auto it = std::lower_bound(list_ids.begin(), list_ids.end(), list);
  if (it == list_ids.end() || *it != list) return nullptr;
  const size_t i = static_cast<size_t>(it - list_ids.begin());
  *len = static_cast<uint32_t>(offsets[i + 1] - offsets[i]);
  return ids.data() + offsets[i];
}

void ListSegment::Save(const std::string& path) const {
  FUSION_CHECK(offsets.size() == list_ids.size() + 1, "segment offsets/list mismatch");
  FUSION_CHECK(locs.size() == ids.size(), "segment has %zu locations for %zu IDs", locs.size(),
               ids.size());
  std::vector<uint8_t> buf;
  auto put = [&](const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    buf.insert(buf.end(), b, b + n);
  };
  const uint32_t hdr[4] = {kSegMagic, kSegVersion, partition,
                           static_cast<uint32_t>(list_ids.size())};
  put(hdr, sizeof(hdr));
  put(list_ids.data(), list_ids.size() * 4);
  put(offsets.data(), offsets.size() * 8);
  put(ids.data(), ids.size() * 4);
  put(locs.data(), locs.size() * 4);
  const uint32_t crc = Crc32(buf.data(), buf.size());
  put(&crc, 4);
  // Write to a temporary name and rename, so a reader never sees a half-written segment.
  const std::string tmp = path + ".tmp";
  WriteFile(tmp, buf.data(), buf.size());
  FUSION_CHECK(std::rename(tmp.c_str(), path.c_str()) == 0, "rename %s: %s", tmp.c_str(),
               std::strerror(errno));
}

ListSegment ListSegment::Load(const std::string& path) {
  std::vector<uint8_t> buf = ReadFile(path);
  FUSION_CHECK(buf.size() >= 16 + 8 + 4, "segment %s is truncated", path.c_str());
  uint32_t crc_stored;
  std::memcpy(&crc_stored, buf.data() + buf.size() - 4, 4);
  FUSION_CHECK(Crc32(buf.data(), buf.size() - 4) == crc_stored, "segment %s: checksum mismatch",
               path.c_str());
  uint32_t hdr[4];
  std::memcpy(hdr, buf.data(), sizeof(hdr));
  FUSION_CHECK(hdr[0] == kSegMagic, "%s is not a segment file", path.c_str());
  FUSION_CHECK(hdr[1] == kSegVersion,
               "segment %s has version %u, expected %u (version 1 has no raw-vector locations: "
               "rerun rtier_segment)",
               path.c_str(), hdr[1], kSegVersion);
  ListSegment s;
  s.partition = hdr[2];
  const size_t nl = hdr[3];
  size_t off = 16;
  auto need = [&](size_t n) {
    FUSION_CHECK(off + n + 4 <= buf.size(), "segment %s is truncated", path.c_str());
  };
  need(nl * 4);
  s.list_ids.resize(nl);
  std::memcpy(s.list_ids.data(), buf.data() + off, nl * 4);
  off += nl * 4;
  need((nl + 1) * 8);
  s.offsets.resize(nl + 1);
  std::memcpy(s.offsets.data(), buf.data() + off, (nl + 1) * 8);
  off += (nl + 1) * 8;
  FUSION_CHECK(s.offsets[0] == 0, "segment %s: offsets do not start at 0", path.c_str());
  for (size_t i = 0; i < nl; ++i)
    FUSION_CHECK(s.offsets[i] <= s.offsets[i + 1], "segment %s: offsets not monotonic", path.c_str());
  const uint64_t nid = s.offsets.back();
  FUSION_CHECK(nid <= (buf.size() - off) / 8, "segment %s is truncated", path.c_str());
  need(nid * 8);
  s.ids.resize(nid);
  std::memcpy(s.ids.data(), buf.data() + off, nid * 4);
  off += nid * 4;
  s.locs.resize(nid);
  std::memcpy(s.locs.data(), buf.data() + off, nid * 4);
  off += nid * 4;
  FUSION_CHECK(off + 4 == buf.size(), "segment %s has trailing bytes", path.c_str());
  FUSION_CHECK(std::is_sorted(s.list_ids.begin(), s.list_ids.end()),
               "segment %s: list IDs not sorted", path.c_str());
  return s;
}

std::vector<uint64_t> WritePartitions(const PostingLists& lists,
                                      const std::vector<uint32_t>& list_part,
                                      uint32_t num_partitions, const std::string& out_dir,
                                      const LayoutMap& layout, const std::string& layout_name) {
  const uint32_t nl = lists.num_lists();
  FUSION_CHECK(list_part.size() == nl, "assignment has %zu entries but the index has %u lists",
               list_part.size(), nl);
  FUSION_CHECK(num_partitions >= 1, "need at least one partition");
  // Locations must fit in 32 bits (next to the 32-bit ID), with kInvalidId left free.
  FUSION_CHECK(layout.raw().locations() < kInvalidId,
               "layout '%s' has %llu locations, more than a 32-bit location can address",
               layout_name.c_str(), static_cast<unsigned long long>(layout.raw().locations()));
  PartitionManifest m;
  m.num_lists = nl;
  m.num_partitions = num_partitions;
  m.list_part = list_part;
  m.layout = layout_name;
  m.raw = layout.raw();
  for (uint32_t c = 0; c < nl; ++c) {
    FUSION_CHECK(list_part[c] < num_partitions, "list %u assigned to partition %u >= %u", c,
                 list_part[c], num_partitions);
    m.max_list_len = std::max(m.max_list_len, lists.Size(c));
  }
  MakeDirs(out_dir);

  std::vector<ListSegment> segs(num_partitions);
  for (uint32_t p = 0; p < num_partitions; ++p) {
    segs[p].partition = p;
    segs[p].offsets.push_back(0);
  }
  for (uint32_t c = 0; c < nl; ++c) {  // ascending list IDs -> sorted segments
    ListSegment& s = segs[list_part[c]];
    s.list_ids.push_back(c);
    for (const uint32_t* v = lists.List(c); v != lists.List(c) + lists.Size(c); ++v) {
      FUSION_CHECK(*v < layout.size() && layout.page_of[*v] != kInvalidId,
                   "list %u names vector %u, which layout '%s' does not place", c, *v,
                   layout_name.c_str());
      s.ids.push_back(*v);
      s.locs.push_back(static_cast<uint32_t>(layout.Loc(*v)));
    }
    s.offsets.push_back(s.ids.size());
  }
  std::vector<uint64_t> sizes(num_partitions);
  for (uint32_t p = 0; p < num_partitions; ++p) {
    segs[p].Save(JoinPath(out_dir, partfiles::SegmentName(p)));
    sizes[p] = segs[p].bytes();
  }
  m.Save(out_dir);  // manifest last: its presence marks a complete directory
  return sizes;
}

}  // namespace fusion
