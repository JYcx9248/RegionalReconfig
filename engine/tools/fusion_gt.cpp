// Exact k-NN ground truth by brute force (for dataset prefixes, whose official ground
// truth is not available). Output is in big-ann-benchmarks format.
//
// --stride S computes it for every S-th query only (row r is query r*S): a query file longer
// than a run, which repeats no query, still has ground truth all through the run, at a fraction
// of the cost of all of it.
//
// --index DIR reads the base from an index's page file instead of a base file (vectors_<layout>.bin,
// each vector's page and slot from layout_<layout>.map), in page order: for an index whose base
// file is gone, as the queries scripts/data/sample_queries.py draws from it. The answer is the
// same: every query keeps the k smallest (distance, ID) pairs, whatever order the base comes in.
//
// The base is read a chunk at a time and dropped from the page cache (VectorFile, or here), so it
// can be larger than memory.
//
// --exclude and --insert give the ground truth of a data version after update batches (U14): the
// base without the vectors the batches deleted (files of vector IDs, as rtier-updater -deletes
// reads them: text, one per line, or the engine's uint32 array .bin) and with the ones they
// inserted (FILE@FIRST: a batch's vectors, given IDs FIRST, FIRST+1, ... in the file's order, as
// the update controller hands them out). A run's answers are scored against the version their
// epoch read (scripts/score_answers.py).
#include <fcntl.h>
#include <omp.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "cli.h"
#include "fusion/dataset.h"
#include "fusion/distance.h"
#include "fusion/index_meta.h"
#include "fusion/layout.h"

using namespace fusion;

static void Usage() {
  std::fprintf(stderr,
               "usage: fusion_gt (--base FILE [--max-n N] [--dtype T] | --index DIR [--layout bucket])\n"
               "                 --queries FILE --out FILE [--nq N] [--stride S] [--k 100] [--threads T]\n"
               "                 [--exclude IDS[,IDS...]] [--insert FILE@FIRST_ID[,FILE@FIRST_ID...]]\n");
}

namespace {

// The base as a scan: fn(rows, ids, m) gets m vectors back to back and the ID of each.
using Sink = std::function<void(const uint8_t* rows, const uint32_t* ids, uint64_t m)>;

struct Base {
  DType dtype = DType::kFloat;
  uint32_t dim = 0;
  uint64_t n = 0;
  size_t vb = 0;
  std::function<void(const Sink&)> scan;
};

Base FileBase(const std::string& path, std::optional<DType> dt, uint64_t max_n) {
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

void ReadAll(int fd, uint8_t* p, uint64_t n, uint64_t off, const std::string& path) {
  for (uint64_t done = 0; done < n;) {
    const ssize_t r = ::pread(fd, p + done, n - done, static_cast<off_t>(off + done));
    FUSION_CHECK(r > 0 || (r < 0 && errno == EINTR), "cannot read %s at %llu: %s", path.c_str(),
                 static_cast<unsigned long long>(off + done), r == 0 ? "end of file" : std::strerror(errno));
    if (r > 0) done += static_cast<uint64_t>(r);
  }
}

Base IndexBase(const std::string& dir, const std::string& layout) {
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
    // Each location's vector: every vector has one, and a location holds at most one.
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

// Vector IDs a file names: the engine's uint32 array (.bin) or text, one per line, # comments.
std::vector<uint32_t> ReadIDs(const std::string& path) {
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

std::vector<std::string> SplitList(const std::string& s) {
  std::vector<std::string> out;
  for (size_t i = 0; i <= s.size();) {
    const size_t j = std::min(s.find(',', i), s.size());
    if (j > i) out.push_back(s.substr(i, j - i));
    i = j + 1;
  }
  return out;
}

// The base of a data version (--exclude, --insert): the base's vectors and then each insert
// file's, without the excluded IDs.
Base VersionBase(Base base, const std::vector<std::string>& exclude, const std::vector<std::string>& insert) {
  struct Insert {
    std::shared_ptr<VectorFile> f;
    uint32_t first;
  };
  std::vector<Insert> ins;
  uint64_t end = base.n, inserted = 0;  // IDs [0, end) are handed out: the base's, then the batches'
  for (const std::string& spec : insert) {
    const size_t at = spec.rfind('@');
    FUSION_CHECK(at != std::string::npos && at > 0 && at + 1 < spec.size() &&
                     spec.find_first_not_of("0123456789", at + 1) == std::string::npos,
                 "--insert %s: FILE@FIRST_ID", spec.c_str());
    const uint64_t first = std::stoull(spec.substr(at + 1));
    std::shared_ptr<VectorFile> f = VectorFile::Open(spec.substr(0, at));
    FUSION_CHECK(f->dtype() == base.dtype && f->dim() == base.dim, "%s holds %u-d %s vectors, the base %u-d %s",
                 spec.c_str(), f->dim(), DTypeName(f->dtype()), base.dim, DTypeName(base.dtype));
    // A batch's IDs follow the base's and the earlier batches' (a failed batch's are skipped).
    FUSION_CHECK(first >= end && first + f->size() <= UINT32_MAX, "--insert %s: IDs from %llu overlap the ones before",
                 spec.c_str(), static_cast<unsigned long long>(first));
    end = first + f->size();
    inserted += f->size();
    ins.push_back({f, static_cast<uint32_t>(first)});
  }
  // Whether a vector of this version has the ID: the base's, or one an insert file gives.
  auto has = [&](uint64_t id) {
    if (id < base.n) return true;
    for (const Insert& x : ins)
      if (id >= x.first && id < x.first + x.f->size()) return true;
    return false;
  };
  auto gone = std::make_shared<std::vector<uint64_t>>((end + 63) / 64, 0);
  uint64_t named = 0;  // vectors of this version excluded
  for (const std::string& path : exclude)
    for (uint32_t id : ReadIDs(path)) {
      if (id >= end) continue;  // no vector has it yet
      uint64_t& w = (*gone)[id >> 6];
      const uint64_t bit = uint64_t{1} << (id & 63);
      if (!(w & bit) && has(id)) ++named;
      w |= bit;
    }
  Log("data version: %llu base vectors and %llu inserted (%zu files), %llu of them excluded",
      static_cast<unsigned long long>(base.n), static_cast<unsigned long long>(inserted), ins.size(),
      static_cast<unsigned long long>(named));
  auto scan = base.scan;
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
    for (const Insert& x : ins)
      x.f->ForEachChunk([&](uint64_t r0, uint64_t r1, const uint8_t* r) {
        fid.resize(r1 - r0);
        std::iota(fid.begin(), fid.end(), x.first + static_cast<uint32_t>(r0));
        kept(r, fid.data(), r1 - r0);
      });
  };
  base.n = base.n + inserted - named;  // what the scan gives
  return base;
}

}  // namespace

int main(int argc, char** argv) {
  const std::set<std::string> flags = {"help"};
  try {
    Args a(argc, argv, flags);
    if (a.Flag("help") || a.Has("base") == a.Has("index") || !a.Has("queries") || !a.Has("out")) {
      Usage();
      return a.Flag("help") ? 0 : 2;
    }
    std::optional<DType> dt;
    if (a.Has("dtype")) dt = ParseDType(a.Str("dtype"));
    Base base = a.Has("base") ? FileBase(a.Required("base"), dt, a.U64("max-n", 0))
                              : IndexBase(a.Required("index"), a.Str("layout", "bucket"));
    if (a.Has("exclude") || a.Has("insert"))
      base = VersionBase(std::move(base), SplitList(a.Str("exclude")), SplitList(a.Str("insert")));
    const uint64_t stride = a.U64("stride", 1);
    FUSION_CHECK(stride >= 1, "--stride must be at least 1");
    VectorSet q;  // queries 0, stride, 2*stride, ... of the first --nq
    {
      auto qf = VectorFile::Open(a.Required("queries"), base.dtype, a.U64("nq", 0));
      q.dtype = qf->dtype();
      q.dim = qf->dim();
      q.n = (qf->size() + stride - 1) / stride;
      q.data.resize(q.n * q.vec_bytes());
      std::vector<uint64_t> rows(q.n);
      for (uint64_t r = 0; r < q.n; ++r) rows[r] = r * stride;
      qf->Gather(rows.data(), rows.size(), q.data.data());
    }
    const uint32_t k = static_cast<uint32_t>(a.U64("k", 100));
    if (a.Has("threads")) omp_set_num_threads(static_cast<int>(a.U64("threads", 1)));
    const std::string out = a.Required("out");
    a.WarnUnused(flags);
    FUSION_CHECK(q.dim == base.dim, "query dim %u != base dim %u", q.dim, base.dim);
    const uint64_t n = base.n;
    const uint64_t nq = q.n;
    FUSION_CHECK(k <= n, "k larger than the dataset");
    const std::string every = stride > 1 ? " (one in " + std::to_string(stride) + ")" : "";
    Log("ground truth: %llu queries%s x %llu base vectors, k=%u", (unsigned long long)nq, every.c_str(),
        (unsigned long long)n, k);

    using Entry = std::pair<float, uint32_t>;
    std::vector<std::vector<Entry>> heaps(nq);
    for (auto& h : heaps) h.reserve(k);
    // Every query scans a block of the base, which stays in the cores' caches meanwhile: the
    // queries read it from there, not from memory. A block that does not fit is read from memory
    // once per query. A 1024-d float base is 4 KB a vector, and 32K rows (128 MB, the block
    // before) made the scan memory-bound: 200K such vectors against 10K queries took 113 s on 20
    // threads, 40 s with this block (i7-12700H). Few queries get a larger block, so that each
    // block is still work enough for the threads that share it.
    const uint64_t block = std::max<uint64_t>(
        std::max<uint64_t>(1, (uint64_t{512} << 10) / base.vb), (uint64_t{1} << 20) / std::max<uint64_t>(nq, 1));
    uint64_t scanned = 0;
    Timer t;
    DispatchDType(base.dtype, [&](auto tag) {
      using T = decltype(tag);
      const uint32_t d = base.dim;
      const size_t vb = base.vb;
      base.scan([&](const uint8_t* rows, const uint32_t* ids, uint64_t m) {
        for (uint64_t b0 = 0; b0 < m; b0 += block) {
          const uint64_t b1 = std::min(m, b0 + block);
#pragma omp parallel for schedule(dynamic, 16)
          for (int64_t qi = 0; qi < static_cast<int64_t>(nq); ++qi) {
            const T* qv = reinterpret_cast<const T*>(q.Get(static_cast<uint64_t>(qi)));
            auto& h = heaps[static_cast<size_t>(qi)];
            for (uint64_t i = b0; i < b1; ++i) {
              const float dd = L2Sqr(qv, reinterpret_cast<const T*>(rows + i * vb), d);
              const Entry e(dd, ids[i]);
              if (h.size() < k) {
                h.push_back(e);
                std::push_heap(h.begin(), h.end());
              } else if (e < h.front()) {
                std::pop_heap(h.begin(), h.end());
                h.back() = e;
                std::push_heap(h.begin(), h.end());
              }
            }
          }
        }
        const uint64_t before = scanned;
        scanned += m;
        if (scanned >> 20 > before >> 20)  // about every million vectors
          Log("  %llu / %llu base vectors scanned, %.0fs", (unsigned long long)scanned, (unsigned long long)n, t.Sec());
      });
    });
    FUSION_CHECK(scanned == n, "scanned %llu base vectors of %llu", (unsigned long long)scanned, (unsigned long long)n);

    GroundTruth gt;
    gt.nq = static_cast<uint32_t>(nq);
    gt.k = k;
    gt.ids.resize(nq * k);
    gt.dists.resize(nq * k);
    for (uint64_t qi = 0; qi < nq; ++qi) {
      auto& h = heaps[qi];
      std::sort_heap(h.begin(), h.end());
      for (uint32_t i = 0; i < k; ++i) {
        gt.dists[qi * k + i] = h[i].first;
        gt.ids[qi * k + i] = h[i].second;
      }
    }
    SaveGroundTruth(out, gt);
    Log("wrote %s in %.1fs", out.c_str(), t.Sec());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
