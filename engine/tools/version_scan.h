// The base vectors of an index, and the data versions update batches make of them (U14), as the
// tools scan them: fusion_gt (the ground truth of a version), fusion_build (a next generation
// over one). A scan gives the vectors a chunk at a time, each with its vector ID.
#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "fusion/common.h"
#include "fusion/dataset.h"
#include "fusion/index_meta.h"
#include "fusion/layout.h"

namespace fusion::tools {

// fn(rows, ids, m) gets m vectors back to back and the ID of each.
using Sink = std::function<void(const uint8_t* rows, const uint32_t* ids, uint64_t m)>;

struct Base {
  DType dtype = DType::kFloat;
  uint32_t dim = 0;
  uint64_t n = 0;  // the vectors a scan gives
  size_t vb = 0;
  std::function<void(const Sink&)> scan;
};

// A base file: vector ID = row.
inline Base FileBase(const std::string& path, std::optional<DType> dt, uint64_t max_n) {
  std::shared_ptr<VectorFile> f = VectorFile::Open(path, dt, max_n);
  Base b;
  b.dtype = f->dtype();
  b.dim = f->dim();
  b.n = f->size();
  b.vb = f->vec_bytes();
  b.scan = [f](const Sink& fn) {
    std::vector<uint32_t> ids;
    f->ForEachChunk([&](uint64_t r0, uint64_t r1, const uint8_t* rows) {
      ids.resize(r1 - r0);
      std::iota(ids.begin(), ids.end(), static_cast<uint32_t>(r0));
      fn(rows, ids.data(), r1 - r0);
    });
  };
  return b;
}

inline void ReadAll(int fd, uint8_t* p, uint64_t n, uint64_t off, const std::string& path) {
  for (uint64_t done = 0; done < n;) {
    const ssize_t r = ::pread(fd, p + done, n - done, static_cast<off_t>(off + done));
    FUSION_CHECK(r > 0 || (r < 0 && errno == EINTR), "cannot read %s at %llu: %s", path.c_str(),
                 static_cast<unsigned long long>(off + done), r == 0 ? "end of file" : std::strerror(errno));
    if (r > 0) done += static_cast<uint64_t>(r);
  }
}

// An index's page file, through its layout map, in page order: for an index whose base file is
// gone. Every vector has one location, and a location holds at most one vector.
inline Base IndexBase(const std::string& dir, const std::string& layout) {
  const IndexMeta meta = IndexMeta::Load(dir + "/" + files::kMeta);
  auto map = std::make_shared<LayoutMap>(LayoutMap::Load(dir + "/" + files::LayoutMap(layout)));
  Base b;
  b.dtype = ParseDType(meta.Get("dtype"));
  b.dim = static_cast<uint32_t>(std::stoul(meta.Get("dim")));
  b.n = map->size();
  b.vb = map->vec_bytes;
  FUSION_CHECK(b.vb == b.dim * DTypeSize(b.dtype), "layout '%s' has %zu-byte vectors, meta %u x %s",
               layout.c_str(), b.vb, b.dim, DTypeName(b.dtype));
  const std::string pages = dir + "/" + files::LayoutVectors(layout);
  b.scan = [map, pages](const Sink& fn) {
    const uint64_t vpp = map->vectors_per_page, ps = map->page_size, vb = map->vec_bytes;
    std::vector<uint32_t> id_at(map->num_pages * vpp, UINT32_MAX);
    for (uint64_t id = 0; id < map->size(); ++id) id_at[map->Loc(static_cast<uint32_t>(id))] = static_cast<uint32_t>(id);
    const int fd = ::open(pages.c_str(), O_RDONLY | O_CLOEXEC);
    FUSION_CHECK(fd >= 0, "cannot open %s: %s", pages.c_str(), std::strerror(errno));
    const uint64_t per = std::max<uint64_t>(1, (uint64_t{256} << 20) / ps);  // pages per read
    std::vector<uint8_t> buf(per * ps), rows(per * vpp * vb);
    std::vector<uint32_t> ids;
    for (uint64_t p0 = 0; p0 < map->num_pages; p0 += per) {
      const uint64_t cnt = std::min(per, map->num_pages - p0);
      ReadAll(fd, buf.data(), cnt * ps, p0 * ps, pages);
      DropFromPageCache(fd, p0 * ps, cnt * ps);
      ids.clear();
      for (uint64_t p = 0; p < cnt; ++p)
        for (uint64_t s = 0; s < vpp; ++s) {
          const uint32_t id = id_at[(p0 + p) * vpp + s];
          if (id == UINT32_MAX) continue;  // a page's free slot
          std::memcpy(&rows[ids.size() * vb], &buf[p * ps + s * vb], vb);
          ids.push_back(id);
        }
      fn(rows.data(), ids.data(), ids.size());
    }
    ::close(fd);
  };
  return b;
}

// Vector IDs a file names: the engine's uint32 array (.bin) or text, one per line, # comments --
// what rtier-updater -deletes reads.
inline std::vector<uint32_t> ReadIDs(const std::string& path) {
  if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".bin") == 0) return LoadU32Array(path);
  FILE* f = std::fopen(path.c_str(), "r");
  FUSION_CHECK(f != nullptr, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  std::vector<uint32_t> ids;
  char line[256];
  for (uint64_t no = 1; std::fgets(line, sizeof(line), f); ++no) {
    char* p = line;
    while (*p == ' ' || *p == '\t') ++p;
    if (*p == '\0' || *p == '\n' || *p == '#') continue;
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(p, &end, 10);
    while (end && (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n')) ++end;
    if (errno != 0 || end == p || (end && *end != '\0') || v > UINT32_MAX) {
      std::fclose(f);
      throw std::runtime_error(StrFormat("%s:%llu: not a vector ID", path.c_str(), static_cast<unsigned long long>(no)));
    }
    ids.push_back(static_cast<uint32_t>(v));
  }
  std::fclose(f);
  return ids;
}

inline std::vector<std::string> SplitList(const std::string& s) {
  std::vector<std::string> out;
  for (size_t i = 0; i <= s.size();) {
    const size_t j = std::min(s.find(',', i), s.size());
    if (j > i) out.push_back(s.substr(i, j - i));
    i = j + 1;
  }
  return out;
}

// A batch's inserted vectors (U14): its file, and the ID the update controller gave the first
// (the others follow, in the file's order).
struct InsertFile {
  std::string path;
  uint32_t first = 0;
};

// A data version after update batches: the base's vectors without the excluded IDs, then each
// insert file's. base.scan gives them; Kept says which IDs it has.
struct Version {
  Base base;
  uint64_t base_n = 0;  // the base's IDs: [0, base_n)
  uint64_t end = 0;     // the IDs handed out: [0, end)
  struct Inserted {
    std::shared_ptr<VectorFile> f;
    uint32_t first;
  };
  std::vector<Inserted> ins;
  std::shared_ptr<std::vector<uint64_t>> gone;  // excluded IDs, a bit each

  // A vector of the base or of an insert file has the ID (a failed batch's IDs have none).
  bool Has(uint64_t id) const {
    if (id < base_n) return true;
    for (const Inserted& x : ins)
      if (id >= x.first && id < x.first + x.f->size()) return true;
    return false;
  }
  bool Kept(uint64_t id) const { return Has(id) && !(((*gone)[id >> 6] >> (id & 63)) & 1); }
};

inline Version MakeVersion(Base base, const std::vector<std::string>& exclude, const std::vector<InsertFile>& inserts) {
  Version v;
  v.base_n = v.end = base.n;
  uint64_t inserted = 0;
  for (const InsertFile& x : inserts) {
    std::shared_ptr<VectorFile> f = VectorFile::Open(x.path);
    FUSION_CHECK(f->dtype() == base.dtype && f->dim() == base.dim, "%s holds %u-d %s vectors, the base %u-d %s",
                 x.path.c_str(), f->dim(), DTypeName(f->dtype()), base.dim, DTypeName(base.dtype));
    // A batch's IDs follow the base's and the earlier batches' (a failed batch's are skipped).
    FUSION_CHECK(x.first >= v.end && x.first + f->size() <= UINT32_MAX, "%s: IDs from %u overlap the ones before",
                 x.path.c_str(), x.first);
    v.end = x.first + f->size();
    inserted += f->size();
    v.ins.push_back({f, x.first});
  }
  v.gone = std::make_shared<std::vector<uint64_t>>((v.end + 63) / 64, 0);
  uint64_t named = 0;  // vectors of the version excluded
  for (const std::string& path : exclude)
    for (uint32_t id : ReadIDs(path)) {
      if (id >= v.end) continue;  // no vector has it yet
      uint64_t& w = (*v.gone)[id >> 6];
      const uint64_t bit = uint64_t{1} << (id & 63);
      if (!(w & bit) && v.Has(id)) ++named;
      w |= bit;
    }
  Log("data version: %llu base vectors and %llu inserted (%zu files), %llu of them excluded",
      static_cast<unsigned long long>(base.n), static_cast<unsigned long long>(inserted), v.ins.size(),
      static_cast<unsigned long long>(named));
  auto scan = base.scan;
  auto ins = v.ins;
  auto gone = v.gone;
  const size_t vb = base.vb;
  base.scan = [scan, ins, gone, vb](const Sink& fn) {
    std::vector<uint8_t> rows;
    std::vector<uint32_t> ids;
    auto kept = [&](const uint8_t* r, const uint32_t* id, uint64_t m) {
      rows.resize(m * vb);
      ids.clear();
      for (uint64_t i = 0; i < m; ++i) {
        if (((*gone)[id[i] >> 6] >> (id[i] & 63)) & 1) continue;
        std::memcpy(&rows[ids.size() * vb], r + i * vb, vb);
        ids.push_back(id[i]);
      }
      fn(rows.data(), ids.data(), ids.size());
    };
    scan(kept);
    std::vector<uint32_t> fid;
    for (const Version::Inserted& x : ins)
      x.f->ForEachChunk([&](uint64_t r0, uint64_t r1, const uint8_t* r) {
        fid.resize(r1 - r0);
        std::iota(fid.begin(), fid.end(), x.first + static_cast<uint32_t>(r0));
        kept(r, fid.data(), r1 - r0);
      });
  };
  base.n = base.n + inserted - named;  // what the scan gives
  v.base = std::move(base);
  return v;
}

}  // namespace fusion::tools
