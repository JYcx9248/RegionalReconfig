// rtier data-node service: fusion::NodeEngine behind the rtier wire protocol.
//
// Prints "READY tcp=<port>" on stdout once it accepts connections (the Go agent and the tests
// wait for that line). SIGINT/SIGTERM stop it.
#include <signal.h>

#include <cstdio>
#include <thread>

#include "../node/server.h"
#include "cli.h"
#include "fusion/node_engine.h"
#include "fusion/partition.h"

using namespace fusion;

static void Usage() {
  std::fprintf(stderr, R"(usage: rtier_node --index DIR --partitions DIR [options]

  --index DIR            index built by fusion_build
  --partitions DIR       partition directory written by rtier_segment
  --listen HOST:PORT     TCP address (default 127.0.0.1:0 = pick a free port)
  --unix PATH            also listen on a Unix socket
  --graph PATH           load the navigation graph at startup ("index" = <index>/graph.hnsw)
  --load LIST            load these partitions at startup, e.g. "0,1,2" or "all"
  --backend B            cpu | gpu (default cpu)
  --workers N            engine workers = concurrent FILTER/RERANK requests (default 2)
  --max-nprobe M         largest nprobe / lists per request (default 256)
  --max-rerank N         largest top-n / rerank input per request (default 1000)
  --pq-capacity N        PQ codes this node may hold, i.e. its GPU-memory budget in codes
                         (default: every vector of the index)
  --raw-file PATH        this node's raw vectors: a sparse file laid out like the index's page
                         file, created empty (default: an anonymous temporary file)
  --layout L             bucket | id (default: the layout the partitions were written for)
  --io B                 auto | uring | aio | pread (default auto)
  --io-depth D           (default 64)
  --no-direct            buffered instead of O_DIRECT reads
  --gpu G                GPU device (default 0)
)");
}

int main(int argc, char** argv) {
  const std::set<std::string> flags = {"no-direct", "help"};
  try {
    Args a(argc, argv, flags);
    if (a.Flag("help") || !a.Has("index") || !a.Has("partitions")) {
      Usage();
      return a.Flag("help") ? 0 : 2;
    }
    NodeOptions o;
    o.index_dir = a.Required("index");
    o.partitions_dir = a.Required("partitions");
    o.layout = a.Str("layout", "");
    o.raw_file = a.Str("raw-file", "");
    const std::string backend = a.Str("backend", "cpu");
    FUSION_CHECK(backend == "cpu" || backend == "gpu", "unknown backend %s", backend.c_str());
    o.device = backend == "gpu" ? FilterDevice::kGpu : FilterDevice::kCpu;
    o.num_workers = static_cast<int>(a.U64("workers", 2));
    o.max_nprobe = static_cast<uint32_t>(a.U64("max-nprobe", 256));
    o.max_rerank = static_cast<uint32_t>(a.U64("max-rerank", 1000));
    o.pq_capacity = a.U64("pq-capacity", 0);
    o.io = ParseIoBackend(a.Str("io", "auto"));
    o.io_depth = static_cast<uint32_t>(a.U64("io-depth", 64));
    o.direct_io = !a.Flag("no-direct");
    o.gpu_device = static_cast<int>(a.U64("gpu", 0));
    const std::string listen = a.Str("listen", "127.0.0.1:0");
    const std::string unix_path = a.Str("unix", "");
    const std::string graph = a.Str("graph", "");
    const std::string load = a.Str("load", "");
    a.WarnUnused(flags);

    // Block the stop signals in every thread; one thread waits for them.
    sigset_t sigs;
    sigemptyset(&sigs);
    sigaddset(&sigs, SIGINT);
    sigaddset(&sigs, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &sigs, nullptr);

    auto engine = NodeEngine::Open(o);
    if (!graph.empty())
      engine->LoadGraph(graph == "index" ? JoinPath(o.index_dir, files::kGraph) : graph);
    if (!load.empty()) {
      std::vector<uint64_t> parts;
      if (load == "all") {
        for (uint32_t p = 0; p < engine->num_partitions(); ++p) parts.push_back(p);
      } else {
        parts = ParseUintList(load);
      }
      for (uint64_t p : parts) {  // bootstrap: PQ codes and raw vectors come from the index
        engine->LoadPartition(static_cast<uint32_t>(p),
                              JoinPath(o.partitions_dir, partfiles::SegmentName(static_cast<uint32_t>(p))),
                              PQSource::kIndex);
      }
    }
    std::fprintf(stderr, "%s", engine->Describe().c_str());

    rtier::NodeServer server(engine.get());
    const size_t colon = listen.rfind(':');
    FUSION_CHECK(colon != std::string::npos, "--listen wants HOST:PORT");
    const int port = server.ListenTcp(listen.substr(0, colon), std::stoi(listen.substr(colon + 1)));
    if (!unix_path.empty()) server.ListenUnix(unix_path);

    std::thread stopper([&] {
      int sig = 0;
      sigwait(&sigs, &sig);
      server.Stop();
    });
    stopper.detach();

    std::printf("READY tcp=%d\n", port);
    std::fflush(stdout);
    server.Serve();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
