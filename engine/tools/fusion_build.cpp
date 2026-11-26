// Builds a FusionANNS multi-tiered index.
#include <fcntl.h>
#include <omp.h>
#include <unistd.h>

#include <cstdio>
#include <filesystem>

#include "cli.h"
#include "fusion/builder.h"
#include "fusion/index_meta.h"
#include "fusion/partition.h"
#include "version_scan.h"

using namespace fusion;

static void Usage() {
  std::fprintf(stderr, R"(usage: fusion_build --base FILE --out DIR [options]

input
  --base FILE             base vectors (.u8bin .i8bin .fbin .bvecs .fvecs)
  --dtype T               uint8 | int8 | float (default: from extension)
  --max-n N               use only the first N vectors

clustering (posting lists)
  --clustering C          balanced | kmeans                   (default balanced, the paper's)
  --centroid-ratio R      posting lists ~= R * N              (default 0.1, as in the paper)
  --branch B              max children per k-means split      (default 32)
  --balance F             k-means size-balance penalty        (default 0.5)
  kmeans is Quake's index (OSDI '25; U14, internal/update/quake.go): a flat k-means whose means
  are the heads, and every vector in one list, its nearest head's (every head scanned, in the
  build and for inserts): one replica, no RNG pruning
  --nlist N               lists                               (default round(sqrt(N)), Quake's)
  --kmeans-iters I        Lloyd iterations                    (default 5, Quake's niter)

navigation graph (HNSW over the heads)
  --hnsw-m M              graph degree parameter              (default 32 -> 64 on level 0)
  --hnsw-efc EF           construction search width           (default 200)

replicated assignment (Eq. 2)
  --replicas R            max posting lists per vector        (default 8)
  --closure-eps E         Eq. 2 epsilon                       (default 10, SPANN's eps1)
  --rng                   enable SPANN's RNG pruning of replicas (off by default)
  --rng-factor F          RNG rule factor                     (default 1.0)
  --assign-candidates C   nearest heads examined per vector   (default 64)

product quantization
  --pq-m M                bytes per vector (sub-spaces)       (default dim/8, or dim/4)
  --pq-train N            training sample size                (default 100000)

SSD layout
  --layout L              bucket | id | both                  (default bucket)

memory
  --memory-gb G           base vectors held in memory at once     (default 16): the
                          clustering's batches and the page file's windows; the base is
                          read in passes, never mapped, and dropped from the page cache

a data version after update batches (U14 full rebuild): the next generation
  --from-index DIR        an index built from data (not a version): the next generation keeps its
                          vectors' IDs and raw-vector locations and its PQ codebook, and takes the
                          build's settings from its meta.txt unless given here. Its vectors come
                          from --base FILE, or else from its page file
  --exclude IDS[,...]     files of vector IDs the batches deleted (text or uint32 array .bin)
  --insert F@ID@LOC[,...] each batch's inserted vectors (F), with the ID and the raw-vector location
                          the update controller gave the first (updates.json's ids and locations)
  The version's vectors wait in --out's version-rows file, in ID order, while the build runs.
  version.json records what the version is made of; run_local.py checks it before a rebuild.

misc
  --threads T             OpenMP threads (default: all cores)
  --seed S
  --overwrite             rebuild stages whose outputs exist
)");
}

namespace {

using namespace fusion::tools;

std::string Hex(uint32_t v) { return StrFormat("%08x", v); }

uint32_t FileCrc32(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  FUSION_CHECK(fd >= 0, "cannot open %s", path.c_str());
  std::vector<uint8_t> buf(uint64_t{1} << 22);
  uint32_t crc = 0;
  for (uint64_t off = 0;;) {
    const ssize_t r = ::pread(fd, buf.data(), buf.size(), static_cast<off_t>(off));
    if (r < 0 && errno == EINTR) continue;
    FUSION_CHECK(r >= 0, "cannot read %s", path.c_str());
    if (r == 0) break;
    crc = Crc32(buf.data(), static_cast<size_t>(r), crc);
    off += static_cast<uint64_t>(r);
  }
  ::close(fd);
  return crc;
}

std::string JsonString(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out + "\"";
}

// --from-index: writes the version's vectors in ID order to rows_path and fills p.version.
void PrepareVersion(const Args& a, const std::string& from, const std::string& rows_path, BuildParams* p) {
  FUSION_CHECK(!std::filesystem::exists(JoinPath(from, "version.json")),
               "%s is itself a data version: build from the index the versions started from, with every batch since",
               from.c_str());
  const IndexMeta fm = IndexMeta::Load(JoinPath(from, files::kMeta));
  const uint64_t from_n = fm.GetU64("num_vectors");
  std::string layout = fm.Has("layouts") ? fm.Get("layouts") : "bucket";
  layout = layout.substr(0, layout.find(','));
  Base base = a.Has("base") ? FileBase(a.Required("base"), p->dtype, from_n) : IndexBase(from, layout);
  FUSION_CHECK(base.n == from_n, "%llu base vectors, %s has %llu", (unsigned long long)base.n, from.c_str(),
               (unsigned long long)from_n);
  std::vector<InsertFile> inserts;
  std::vector<uint32_t> first_loc;
  for (const std::string& spec : SplitList(a.Str("insert"))) {
    const size_t at2 = spec.rfind('@'), at1 = at2 == std::string::npos || at2 == 0 ? at2 : spec.rfind('@', at2 - 1);
    FUSION_CHECK(at1 != std::string::npos && at1 > 0 && at2 + 1 < spec.size() &&
                     spec.find_first_not_of("0123456789", at1 + 1) == at2 &&
                     spec.find_first_not_of("0123456789", at2 + 1) == std::string::npos,
                 "--insert %s: FILE@FIRST_ID@FIRST_LOC", spec.c_str());
    const unsigned long long id = std::stoull(spec.substr(at1 + 1, at2 - at1 - 1)), loc = std::stoull(spec.substr(at2 + 1));
    FUSION_CHECK(id < UINT32_MAX && loc < UINT32_MAX, "--insert %s: beyond 32 bits", spec.c_str());
    inserts.push_back({spec.substr(0, at1), static_cast<uint32_t>(id)});
    first_loc.push_back(static_cast<uint32_t>(loc));
  }
  const std::vector<std::string> exclude = SplitList(a.Str("exclude"));
  const Version v = MakeVersion(base, exclude, inserts);

  auto ver = std::make_shared<VersionParams>();
  ver->from_index = std::filesystem::absolute(from).string();
  ver->id_end = v.end;
  std::vector<uint32_t> row_of(v.end, UINT32_MAX);
  for (uint64_t id = 0; id < v.end; ++id) {
    if (!v.Kept(id)) continue;
    uint32_t loc = VersionParams::kNoLoc;
    for (size_t k = 0; k < v.ins.size() && id >= v.base_n; ++k)
      if (id >= v.ins[k].first && id < v.ins[k].first + v.ins[k].f->size())
        loc = first_loc[k] + static_cast<uint32_t>(id - v.ins[k].first);
    row_of[id] = static_cast<uint32_t>(ver->ids.size());
    ver->ids.push_back(static_cast<uint32_t>(id));
    ver->locs.push_back(loc);
  }
  // The rows file: the version's vectors in ID order, each written where its row is.
  const int fd = ::open(rows_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  FUSION_CHECK(fd >= 0, "cannot create %s", rows_path.c_str());
  const uint32_t hdr[2] = {static_cast<uint32_t>(ver->ids.size()), base.dim};
  auto put = [&](const void* data, size_t bytes, uint64_t off) {
    const auto* q = static_cast<const uint8_t*>(data);
    for (size_t done = 0; done < bytes;) {
      const ssize_t r = ::pwrite(fd, q + done, bytes - done, static_cast<off_t>(off + done));
      if (r < 0 && errno == EINTR) continue;
      FUSION_CHECK(r > 0, "cannot write %s", rows_path.c_str());
      done += static_cast<size_t>(r);
    }
  };
  put(hdr, sizeof(hdr), 0);
  const uint64_t vb = base.vb;
  uint64_t written = 0;
  v.base.scan([&](const uint8_t* rows, const uint32_t* ids, uint64_t m) {
    for (uint64_t i = 0; i < m;) {  // runs of consecutive rows, one write each
      uint64_t j = i + 1;
      while (j < m && row_of[ids[j]] == row_of[ids[j - 1]] + 1) ++j;
      put(rows + i * vb, (j - i) * vb, sizeof(hdr) + static_cast<uint64_t>(row_of[ids[i]]) * vb);
      written += j - i;
      i = j;
    }
  });
  FUSION_CHECK(::close(fd) == 0 && written == ver->ids.size(), "%s: %llu of %zu rows written", rows_path.c_str(),
               (unsigned long long)written, ver->ids.size());

  std::string rec = "{\n  \"from_index\": " + JsonString(ver->from_index) + ",\n  \"id_end\": " +
                    std::to_string(v.end) + ",\n  \"vectors\": " + std::to_string(ver->ids.size()) +
                    ",\n  \"exclude\": [";
  for (size_t k = 0; k < exclude.size(); ++k)
    rec += std::string(k ? ", " : "") + "{\"path\": " + JsonString(std::filesystem::absolute(exclude[k]).string()) +
           ", \"crc32\": \"" + Hex(FileCrc32(exclude[k])) + "\"}";
  rec += "],\n  \"insert\": [";
  for (size_t k = 0; k < inserts.size(); ++k)
    rec += std::string(k ? ", " : "") + "{\"path\": " + JsonString(std::filesystem::absolute(inserts[k].path).string()) +
           ", \"crc32\": \"" + Hex(FileCrc32(inserts[k].path)) + "\", \"first_id\": " +
           std::to_string(inserts[k].first) + ", \"first_loc\": " + std::to_string(first_loc[k]) +
           ", \"count\": " + std::to_string(v.ins[k].f->size()) + "}";
  rec += "]\n}\n";
  ver->record = rec;
  p->version = ver;
  p->base_path = rows_path;
  // The build's settings: from_index's, unless given.
  auto num = [&](const char* flag, const char* key, double def) {
    return a.Has(flag) ? a.F64(flag, def) : fm.Has(key) ? std::stod(fm.Get(key)) : def;
  };
  p->clustering.centroid_ratio = num("centroid-ratio", "centroid_ratio", 0.1);
  p->clustering.method = a.Has("clustering") ? a.Str("clustering") : fm.GetOr("clustering", "balanced");
  if (p->clustering.method == "kmeans") {
    // The version's own list count unless given: Quake sizes an index by its initial data.
    p->clustering.nlist = static_cast<uint32_t>(a.U64("nlist", 0));
    p->clustering.flat_iters = static_cast<uint32_t>(num("kmeans-iters", "kmeans_iters", 5));
  }
  p->assign.replicas = static_cast<uint32_t>(num("replicas", "replicas", 8));
  p->assign.closure_eps = static_cast<float>(num("closure-eps", "closure_eps", 10.0));
  p->assign.rng_factor = static_cast<float>(num("rng-factor", "rng_factor", 1.0));
  p->assign.rng = a.Flag("rng") || (fm.Has("rng") && fm.Get("rng") == "1");
  p->assign.candidates = static_cast<uint32_t>(num("assign-candidates", "assign_candidates", 64));
  if (fm.Has("assign_ef")) p->assign.ef = static_cast<uint32_t>(std::stoul(fm.Get("assign_ef")));
  p->graph.m = static_cast<uint32_t>(num("hnsw-m", "hnsw_m", 32));
  p->graph.ef_construction = static_cast<uint32_t>(num("hnsw-efc", "hnsw_ef_construction", 200));
  if (p->clustering.method == "kmeans" && !a.Has("replicas") && !a.Flag("rng")) {
    p->assign.replicas = 1;  // from_index's, a FusionANNS one, would replicate
    p->assign.rng = false;
  }
}

std::string RowsExt(DType dt) {
  switch (dt) {
    case DType::kUInt8: return ".u8bin";
    case DType::kInt8: return ".i8bin";
    default: return ".fbin";
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::set<std::string> flags = {"rng", "overwrite", "help"};
  try {
    Args a(argc, argv, flags);
    if (a.Flag("help") || !(a.Has("base") || a.Has("from-index")) || !a.Has("out") ||
        ((a.Has("exclude") || a.Has("insert")) && !a.Has("from-index"))) {
      Usage();
      return a.Flag("help") ? 0 : 2;
    }
    BuildParams p;
    p.base_path = a.Str("base");
    p.out_dir = a.Required("out");
    if (a.Has("dtype")) p.dtype = ParseDType(a.Str("dtype"));
    p.max_n = a.U64("max-n", 0);
    p.seed = a.U64("seed", 42);

    p.clustering.centroid_ratio = a.F64("centroid-ratio", 0.1);
    p.clustering.branch = static_cast<uint32_t>(a.U64("branch", 32));
    p.clustering.balance = static_cast<float>(a.F64("balance", 0.5));
    p.clustering.seed = p.seed;

    p.graph.m = static_cast<uint32_t>(a.U64("hnsw-m", 32));
    p.graph.ef_construction = static_cast<uint32_t>(a.U64("hnsw-efc", 200));

    p.assign.replicas = static_cast<uint32_t>(a.U64("replicas", 8));
    p.assign.closure_eps = static_cast<float>(a.F64("closure-eps", 10.0));
    p.assign.rng_factor = static_cast<float>(a.F64("rng-factor", 1.0));
    p.assign.rng = a.Flag("rng");
    p.assign.candidates = static_cast<uint32_t>(a.U64("assign-candidates", 64));

    p.clustering.method = a.Str("clustering", "balanced");
    p.clustering.nlist = static_cast<uint32_t>(a.U64("nlist", 0));
    p.clustering.flat_iters = static_cast<uint32_t>(a.U64("kmeans-iters", 5));
    if (p.clustering.method == "kmeans" && !a.Has("replicas")) p.assign.replicas = 1;

    p.pq_m = static_cast<uint32_t>(a.U64("pq-m", 0));
    p.pq_train_n = static_cast<uint32_t>(a.U64("pq-train", 100000));
    p.layout = a.Str("layout", "bucket");
    const double memory_gb = a.F64("memory-gb", 16.0);
    FUSION_CHECK(memory_gb > 0, "--memory-gb must be positive");
    p.memory_bytes = static_cast<uint64_t>(memory_gb * static_cast<double>(uint64_t{1} << 30));
    p.overwrite = a.Flag("overwrite");
    if (a.Has("threads")) omp_set_num_threads(static_cast<int>(a.U64("threads", 1)));
    std::string rows;  // a data version's rows file, while the build runs
    if (a.Has("from-index")) {
      std::filesystem::create_directories(p.out_dir);
      const IndexMeta fm = IndexMeta::Load(JoinPath(a.Required("from-index"), files::kMeta));
      rows = JoinPath(p.out_dir, "version-rows" + RowsExt(ParseDType(fm.Get("dtype"))));
      PrepareVersion(a, a.Required("from-index"), rows, &p);
      p.dtype.reset();  // the rows file's extension says it
      p.max_n = 0;
    }
    FUSION_CHECK(p.clustering.method == "balanced" || p.clustering.method == "kmeans",
                 "--clustering %s: balanced or kmeans", p.clustering.method.c_str());
    if (p.clustering.method == "kmeans") {
      // Quake's assignment: one list per vector, its nearest head's.
      FUSION_CHECK(p.assign.replicas == 1 && !p.assign.rng,
                   "--clustering kmeans assigns every vector to one list: no --replicas but 1, no --rng");
      p.assign.exact = true;
    }
    a.WarnUnused(flags);

    BuildIndex(p);
    if (!rows.empty()) std::filesystem::remove(rows);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
