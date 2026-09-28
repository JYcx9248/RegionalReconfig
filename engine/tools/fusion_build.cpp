// Builds a FusionANNS multi-tiered index.
#include <omp.h>

#include <cstdio>

#include "cli.h"
#include "fusion/builder.h"

using namespace fusion;

static void Usage() {
  std::fprintf(stderr, R"(usage: fusion_build --base FILE --out DIR [options]

input
  --base FILE             base vectors (.u8bin .i8bin .fbin .bvecs .fvecs)
  --dtype T               uint8 | int8 | float (default: from extension)
  --max-n N               use only the first N vectors

clustering (posting lists)
  --centroid-ratio R      posting lists ~= R * N              (default 0.1, as in the paper)
  --branch B              max children per k-means split      (default 32)
  --balance F             k-means size-balance penalty        (default 0.5)

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

misc
  --threads T             OpenMP threads (default: all cores)
  --seed S
  --overwrite             rebuild stages whose outputs exist
)");
}

int main(int argc, char** argv) {
  const std::set<std::string> flags = {"rng", "overwrite", "help"};
  try {
    Args a(argc, argv, flags);
    if (a.Flag("help") || !a.Has("base") || !a.Has("out")) {
      Usage();
      return a.Flag("help") ? 0 : 2;
    }
    BuildParams p;
    p.base_path = a.Required("base");
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

    p.pq_m = static_cast<uint32_t>(a.U64("pq-m", 0));
    p.pq_train_n = static_cast<uint32_t>(a.U64("pq-train", 100000));
    p.layout = a.Str("layout", "bucket");
    p.overwrite = a.Flag("overwrite");
    if (a.Has("threads")) omp_set_num_threads(static_cast<int>(a.U64("threads", 1)));
    a.WarnUnused(flags);

    BuildIndex(p);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
