// Runs queries against a FusionANNS index and reports recall, throughput, latency and the
// per-query work of every stage. Sweeps nprobe (m), rerank (n) and thread counts.
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

#include "cli.h"
#include "fusion/dataset.h"
#include "fusion/engine.h"
#include "fusion/filter.h"

using namespace fusion;

static void Usage() {
  std::fprintf(stderr, R"(usage: fusion_search --index DIR --queries FILE [--gt FILE] [options]

workload
  --nq N                  number of queries (default: all)
  --k K                   neighbors per query (default 10)
  --warmup N              untimed warm-up queries (default 100)

search parameters (comma-separated lists are swept)
  --nprobe LIST           m: posting lists from the navigation graph (default 64)
  --rerank LIST           n: PQ candidates passed to re-ranking (default 100)
  --threads LIST          concurrent query threads (default 1)
  --ef N                  graph search width (default 2 * nprobe)

FusionANNS techniques (for ablations)
  --backend gpu|cpu       PQ filtering device (default: gpu if available)
  --layout bucket|id      SSD layout (default bucket)
  --no-heuristic          re-rank all n candidates (no early termination)
  --eps E --beta B        heuristic re-ranking thresholds (default 0.1, 1)
  --batch S               mini-batch size (default k)
  --no-io-dedup           one SSD read per vector, no page reuse

I/O
  --io auto|uring|aio|pread   (default auto)
  --io-depth D            max in-flight reads per thread (default 64)
  --no-direct             buffered I/O instead of O_DIRECT

output
  --csv FILE              append one line per configuration
  --gpu-device D
)");
}

struct Summary {
  double recall = -1, qps = 0, mean_ms = 0, p50_ms = 0, p99_ms = 0;
  double lists = 0, cand = 0, uniq = 0, topn = 0, batches = 0, reranked = 0, pages = 0,
         merged = 0, hits = 0;
  double graph_us = 0, gather_us = 0, filter_us = 0, rerank_us = 0, io_us = 0;
};

int main(int argc, char** argv) {
  const std::set<std::string> flags = {"no-heuristic", "no-io-dedup", "no-direct", "help"};
  try {
    Args a(argc, argv, flags);
    if (a.Flag("help") || !a.Has("index") || !a.Has("queries")) {
      Usage();
      return a.Flag("help") ? 0 : 2;
    }
    const std::vector<uint64_t> nprobes = a.List("nprobe", "64");
    const std::vector<uint64_t> reranks = a.List("rerank", "100");
    const std::vector<uint64_t> threads = a.List("threads", "1");
    FUSION_CHECK(!nprobes.empty() && !reranks.empty() && !threads.empty(), "empty sweep list");

    EngineOptions eo;
    eo.index_dir = a.Required("index");
    eo.layout = a.Str("layout", "bucket");
    const std::string backend = a.Str("backend", GpuFilterAvailable() ? "gpu" : "cpu");
    FUSION_CHECK(backend == "gpu" || backend == "cpu", "--backend must be gpu or cpu");
    eo.device = backend == "gpu" ? FilterDevice::kGpu : FilterDevice::kCpu;
    eo.num_workers = static_cast<int>(*std::max_element(threads.begin(), threads.end()));
    eo.max_nprobe = static_cast<uint32_t>(*std::max_element(nprobes.begin(), nprobes.end()));
    eo.max_rerank = static_cast<uint32_t>(*std::max_element(reranks.begin(), reranks.end()));
    eo.io = ParseIoBackend(a.Str("io", "auto"));
    eo.io_depth = static_cast<uint32_t>(a.U64("io-depth", 64));
    eo.direct_io = !a.Flag("no-direct");
    eo.gpu_device = static_cast<int>(a.U64("gpu-device", 0));

    SearchParams base;
    base.k = static_cast<uint32_t>(a.U64("k", 10));
    base.graph_ef = static_cast<uint32_t>(a.U64("ef", 0));
    base.rr.heuristic = !a.Flag("no-heuristic");
    base.rr.io_dedup = !a.Flag("no-io-dedup");
    base.rr.eps = static_cast<float>(a.F64("eps", 0.1));
    base.rr.beta = static_cast<uint32_t>(a.U64("beta", 1));
    base.rr.batch = static_cast<uint32_t>(a.U64("batch", 0));
    const uint64_t warmup_n = a.U64("warmup", 100);
    const std::string csv = a.Str("csv", "");
    const std::string gt_path = a.Str("gt", "");
    const uint64_t nq_limit = a.U64("nq", 0);
    const std::string query_path = a.Required("queries");
    a.WarnUnused(flags);

    Timer load;
    auto engine = Engine::Open(eo);
    Log("engine ready in %.1fs (%s filter)", load.Sec(), engine->filter_name());
    std::fprintf(stderr, "%s", engine->Describe().c_str());

    VectorSet q = LoadVectors(query_path, engine->dtype(), nq_limit);
    FUSION_CHECK(q.dim == engine->dim(), "query dim %u != index dim %u", q.dim, engine->dim());
    const uint64_t nq = q.n;
    GroundTruth gt;
    if (!gt_path.empty()) {
      gt = LoadGroundTruth(gt_path);
      FUSION_CHECK(gt.nq >= nq, "ground truth covers %u queries, need %llu", gt.nq,
                   (unsigned long long)nq);
    }
    const uint32_t k = base.k;
    std::vector<uint32_t> ids(nq * k);
    std::vector<float> dists(nq * k);
    std::vector<QueryStats> stats(nq);

    FILE* csvf = nullptr;
    if (!csv.empty()) {
      bool fresh = !FileExists(csv);
      csvf = std::fopen(csv.c_str(), "a");
      FUSION_CHECK(csvf != nullptr, "cannot open %s", csv.c_str());
      if (fresh)
        std::fprintf(csvf, "backend,layout,heuristic,io_dedup,eps,beta,batch,k,nprobe,rerank,"
                           "threads,recall,qps,mean_ms,p50_ms,p99_ms,lists,candidates,unique,"
                           "topn,batches,reranked,pages_read,intra_merged,buffer_hits,graph_us,"
                           "gather_us,filter_us,rerank_us,io_us\n");
    }

    std::printf("%6s %6s %4s %9s %9s %8s %8s %8s %7s %7s %7s %8s %6s %6s %6s %8s %8s %8s %8s\n",
                "nprobe", "rerank", "thr", "recall", "QPS", "mean_ms", "p50_ms", "p99_ms", "cand",
                "unique", "batches", "reranked", "reads", "merged", "bufhit", "graph_us",
                "gathr_us", "filtr_us", "rrank_us");

    for (uint64_t np : nprobes) {
      for (uint64_t rr : reranks) {
        SearchParams sp = base;
        sp.nprobe = static_cast<uint32_t>(np);
        sp.rerank = static_cast<uint32_t>(rr);
        engine->Configure(sp);

        // Warm-up (untimed): touches the graph, lists and GPU kernels.
        {
          std::vector<uint32_t> wi(k);
          std::vector<float> wd(k);
          QueryStats ws;
          for (uint64_t i = 0; i < std::min(warmup_n, nq); ++i)
            engine->Search(0, q.Get(i), sp, wi.data(), wd.data(), &ws);
        }

        for (uint64_t nt : threads) {
          std::fill(ids.begin(), ids.end(), kInvalidId);
          std::fill(dists.begin(), dists.end(), 0.f);
          std::atomic<uint64_t> next{0};
          std::vector<std::thread> pool;
          Timer wall;
          for (uint64_t t = 0; t < nt; ++t) {
            pool.emplace_back([&, t] {
              uint64_t i;
              while ((i = next++) < nq) {
                engine->Search(static_cast<int>(t), q.Get(i), sp, &ids[i * k], &dists[i * k],
                               &stats[i]);
              }
            });
          }
          for (auto& th : pool) th.join();
          const double sec = wall.Sec();

          Summary s;
          s.qps = nq / sec;
          if (!gt_path.empty()) s.recall = RecallAtK(gt, ids.data(), dists.data(), k, nq, k);
          std::vector<double> lat(nq);
          for (uint64_t i = 0; i < nq; ++i) {
            const QueryStats& st = stats[i];
            lat[i] = st.total_us / 1000.0;
            s.lists += st.lists;
            s.cand += st.candidates;
            s.uniq += st.unique;
            s.topn += st.topn;
            s.batches += st.rerank.batches;
            s.reranked += st.rerank.reranked;
            s.pages += st.rerank.pages_read;
            s.merged += st.rerank.intra_merged;
            s.hits += st.rerank.buffer_hits;
            s.graph_us += st.graph_us;
            s.gather_us += st.gather_us;
            s.filter_us += st.filter_us;
            s.rerank_us += st.rerank_us;
            s.io_us += st.rerank.io_us;
          }
          for (double* v : {&s.lists, &s.cand, &s.uniq, &s.topn, &s.batches, &s.reranked,
                            &s.pages, &s.merged, &s.hits, &s.graph_us, &s.gather_us,
                            &s.filter_us, &s.rerank_us, &s.io_us})
            *v /= static_cast<double>(nq);
          std::sort(lat.begin(), lat.end());
          for (double l : lat) s.mean_ms += l;
          s.mean_ms /= static_cast<double>(nq);
          s.p50_ms = lat[nq / 2];
          s.p99_ms = lat[std::min<uint64_t>(nq - 1, static_cast<uint64_t>(nq * 0.99))];

          char recall[32];
          if (s.recall >= 0) std::snprintf(recall, sizeof(recall), "%.4f", s.recall);
          else std::snprintf(recall, sizeof(recall), "-");
          std::printf("%6llu %6llu %4llu %9s %9.0f %8.3f %8.3f %8.3f %7.0f %7.0f %7.2f %8.1f "
                      "%6.1f %6.1f %6.1f %8.1f %8.1f %8.1f %8.1f\n",
                      (unsigned long long)np, (unsigned long long)rr, (unsigned long long)nt,
                      recall, s.qps, s.mean_ms, s.p50_ms, s.p99_ms, s.cand, s.uniq, s.batches,
                      s.reranked, s.pages, s.merged, s.hits, s.graph_us, s.gather_us, s.filter_us,
                      s.rerank_us);
          std::fflush(stdout);
          if (csvf) {
            std::fprintf(csvf,
                         "%s,%s,%d,%d,%.3f,%u,%u,%u,%llu,%llu,%llu,%.5f,%.1f,%.4f,%.4f,%.4f,%.2f,"
                         "%.1f,%.1f,%.1f,%.3f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                         engine->filter_name(), eo.layout.c_str(), sp.rr.heuristic ? 1 : 0,
                         sp.rr.io_dedup ? 1 : 0, sp.rr.eps, sp.rr.beta,
                         sp.rr.batch ? sp.rr.batch : k, k, (unsigned long long)np,
                         (unsigned long long)rr, (unsigned long long)nt, s.recall, s.qps,
                         s.mean_ms, s.p50_ms, s.p99_ms, s.lists, s.cand, s.uniq, s.topn,
                         s.batches, s.reranked, s.pages, s.merged, s.hits, s.graph_us,
                         s.gather_us, s.filter_us, s.rerank_us, s.io_us);
            std::fflush(csvf);
          }
        }
      }
    }
    if (csvf) std::fclose(csvf);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
