#include "fusion/layout.h"

#include <algorithm>
#include <cerrno>
#include <cstring>

namespace fusion {

void LayoutMap::Save(const std::string& path) const {
  FILE* f = std::fopen(path.c_str(), "wb");
  FUSION_CHECK(f != nullptr, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  uint32_t h32[4] = {page_size, vec_bytes, vectors_per_page, 0};
  uint64_t h64[2] = {size(), num_pages};
  AppendToFile(f, h32, sizeof(h32), path);
  AppendToFile(f, h64, sizeof(h64), path);
  WriteVector(f, page_of, path);
  WriteVector(f, slot_of, path);
  FUSION_CHECK(std::fclose(f) == 0, "close failed for %s", path.c_str());
}

LayoutMap LayoutMap::Load(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  FUSION_CHECK(f != nullptr, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  uint32_t h32[4];
  uint64_t h64[2];
  bool ok = std::fread(h32, sizeof(h32), 1, f) == 1 && std::fread(h64, sizeof(h64), 1, f) == 1;
  FUSION_CHECK(ok, "bad layout header in %s", path.c_str());
  LayoutMap m;
  m.page_size = h32[0];
  m.vec_bytes = h32[1];
  m.vectors_per_page = h32[2];
  m.num_pages = h64[1];
  m.page_of.resize(h64[0]);
  m.slot_of.resize(h64[0]);
  ok = std::fread(m.page_of.data(), sizeof(uint32_t), m.page_of.size(), f) == m.page_of.size() &&
       std::fread(m.slot_of.data(), sizeof(uint16_t), m.slot_of.size(), f) == m.slot_of.size();
  std::fclose(f);
  FUSION_CHECK(ok, "truncated layout map %s", path.c_str());
  return m;
}

namespace {
LayoutMap NewMap(uint64_t n, uint32_t vec_bytes, uint32_t page_size) {
  FUSION_CHECK(vec_bytes > 0 && vec_bytes <= page_size,
               "vector size %u B exceeds the %u B page (multi-page vectors are not supported)",
               vec_bytes, page_size);
  LayoutMap m;
  m.page_size = page_size;
  m.vec_bytes = vec_bytes;
  m.vectors_per_page = page_size / vec_bytes;
  FUSION_CHECK(m.vectors_per_page <= 65535, "too many vectors per page");
  m.page_of.assign(n, kInvalidId);
  m.slot_of.assign(n, 0);
  return m;
}
}  // namespace

LayoutMap BuildBucketLayout(const std::vector<uint32_t>& bucket_of, uint32_t num_buckets,
                            uint32_t vec_bytes, uint32_t page_size, LayoutStats* stats) {
  const uint64_t n = bucket_of.size();
  LayoutMap m = NewMap(n, vec_bytes, page_size);
  const uint32_t vpp = m.vectors_per_page;

  // Bucket membership as CSR (members sorted by vector ID within a bucket).
  std::vector<uint64_t> start(static_cast<size_t>(num_buckets) + 1, 0);
  for (uint64_t v = 0; v < n; ++v) {
    FUSION_CHECK(bucket_of[v] < num_buckets, "vector %llu has invalid bucket",
                 (unsigned long long)v);
    start[bucket_of[v] + 1]++;
  }
  for (uint32_t b = 0; b < num_buckets; ++b) start[b + 1] += start[b];
  std::vector<uint32_t> members(n);
  {
    std::vector<uint64_t> pos(start.begin(), start.end() - 1);
    for (uint64_t v = 0; v < n; ++v) members[pos[bucket_of[v]]++] = static_cast<uint32_t>(v);
  }

  struct Tail {
    uint64_t first;  // index into members
    uint32_t size;
  };
  std::vector<Tail> tails;
  std::vector<std::vector<uint32_t>> by_size(vpp);  // tail indices grouped by tail size
  uint64_t page = 0;
  for (uint32_t b = 0; b < num_buckets; ++b) {
    const uint64_t s = start[b + 1] - start[b];
    const uint64_t full = s / vpp;
    const uint32_t rem = static_cast<uint32_t>(s % vpp);
    for (uint64_t f = 0; f < full; ++f, ++page) {
      for (uint32_t slot = 0; slot < vpp; ++slot) {
        uint32_t v = members[start[b] + f * vpp + slot];
        m.page_of[v] = static_cast<uint32_t>(page);
        m.slot_of[v] = static_cast<uint16_t>(slot);
      }
    }
    if (rem > 0) {
      by_size[rem].push_back(static_cast<uint32_t>(tails.size()));
      tails.push_back({start[b] + full * vpp, rem});
    }
  }
  const uint64_t full_pages = page;

  // Largest-first best-fit packing of the tails.
  std::vector<size_t> next(vpp, 0);
  uint64_t remaining = tails.size();
  while (remaining > 0) {
    uint32_t cap = vpp, slot = 0;
    uint32_t s = vpp - 1;
    while (s >= 1 && cap > 0) {
      if (s <= cap && next[s] < by_size[s].size()) {
        const Tail& t = tails[by_size[s][next[s]++]];
        for (uint32_t i = 0; i < t.size; ++i) {
          uint32_t v = members[t.first + i];
          m.page_of[v] = static_cast<uint32_t>(page);
          m.slot_of[v] = static_cast<uint16_t>(slot + i);
        }
        slot += t.size;
        cap -= t.size;
        --remaining;
        s = std::min(s, cap);
      } else {
        --s;
      }
    }
    ++page;
  }
  FUSION_CHECK(page <= 0xFFFFFFFFull, "too many pages");
  m.num_pages = page;

  if (stats) {
    stats->full_pages = full_pages;
    stats->packed_pages = page - full_pages;
    stats->tails = tails.size();
    stats->fill_ratio = page ? static_cast<double>(n) / (static_cast<double>(page) * vpp) : 0;
  }
  return m;
}

LayoutMap BuildSequentialLayout(uint64_t n, uint32_t vec_bytes, uint32_t page_size,
                                LayoutStats* stats) {
  LayoutMap m = NewMap(n, vec_bytes, page_size);
  const uint32_t vpp = m.vectors_per_page;
  for (uint64_t v = 0; v < n; ++v) {
    m.page_of[v] = static_cast<uint32_t>(v / vpp);
    m.slot_of[v] = static_cast<uint16_t>(v % vpp);
  }
  m.num_pages = CeilDiv(n, vpp);
  if (stats) {
    stats->full_pages = n / vpp;
    stats->packed_pages = m.num_pages - stats->full_pages;
    stats->tails = 0;
    stats->fill_ratio = static_cast<double>(n) / (static_cast<double>(m.num_pages) * vpp);
  }
  return m;
}

void WriteVectorPages(const VectorFile& data, const LayoutMap& map, const std::string& path) {
  FUSION_CHECK(map.vec_bytes == data.vec_bytes(), "layout/data vector size mismatch");
  FUSION_CHECK(map.size() == data.size(), "layout/data size mismatch");
  const uint32_t vpp = map.vectors_per_page;
  const uint32_t ps = map.page_size;
  std::vector<uint32_t> inv(map.num_pages * vpp, kInvalidId);
  for (uint64_t v = 0; v < map.size(); ++v) {
    uint64_t idx = static_cast<uint64_t>(map.page_of[v]) * vpp + map.slot_of[v];
    FUSION_CHECK(inv[idx] == kInvalidId, "two vectors mapped to page %u slot %u", map.page_of[v],
                 map.slot_of[v]);
    inv[idx] = static_cast<uint32_t>(v);
  }

  FILE* f = std::fopen(path.c_str(), "wb");
  FUSION_CHECK(f != nullptr, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  constexpr uint64_t kChunkPages = 4096;  // 16 MB per write
  std::vector<uint8_t> buf(kChunkPages * ps);
  for (uint64_t p0 = 0; p0 < map.num_pages; p0 += kChunkPages) {
    const uint64_t cnt = std::min(kChunkPages, map.num_pages - p0);
#pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < static_cast<int64_t>(cnt); ++j) {
      uint8_t* pg = buf.data() + static_cast<uint64_t>(j) * ps;
      std::memset(pg, 0, ps);
      for (uint32_t slot = 0; slot < vpp; ++slot) {
        uint32_t v = inv[(p0 + static_cast<uint64_t>(j)) * vpp + slot];
        if (v != kInvalidId) std::memcpy(pg + slot * map.vec_bytes, data.Get(v), map.vec_bytes);
      }
    }
    AppendToFile(f, buf.data(), cnt * ps, path);
  }
  FUSION_CHECK(std::fclose(f) == 0, "close failed for %s", path.c_str());
}

}  // namespace fusion
