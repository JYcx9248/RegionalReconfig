// gp_kaminpar: the partitioning step of GP -- "Unleashing Graph Partitioning for Large-Scale
// Nearest Neighbor Search" (Gottesbueren, Dhulipala, Jayaram, Lacki; PVLDB 18, VLDB '25) -- done
// the way the paper's code does it (github.com/larsgottesbueren/gp-ann, src/partitioning.cpp:
// PartitionAdjListGraph -> ParallelSymmetrizeAndConvertToCSR -> PartitionGraphWithKaMinPar).
//
//   gp_kaminpar GRAPH.bin K EPSILON THREADS OUT.bin [WEIGHTS.bin]
//
// GRAPH.bin is the directed kNN graph: u64 n, u64 m, u64 xadj[n + 1], u32 adj[m]. The graph is
// symmetrized as the paper's code symmetrizes it: every node lists its own neighbors, then the
// nodes that list it, in increasing order, so a pair of mutual neighbors becomes two parallel
// edges (the cut counts it twice); no edge weights. KaMinPar runs with its default context and
// the given imbalance (the paper: epsilon = 0.05). OUT.bin is u32 block[n]. The paper's nodes
// weigh the same; WEIGHTS.bin (u64 n, u32 weight[n]), a departure for rtier, gives each node a
// weight, which the blocks then balance instead of their node counts: a list head weighing its
// list's postings balances the partitions' postings.
//
// Build it against KaMinPar at the commit the paper's code pins (see CMakeLists.txt here).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <kaminpar-shm/kaminpar.h>

namespace {

void Fail(const char* what) {
  std::fprintf(stderr, "gp_kaminpar: %s\n", what);
  std::exit(1);
}

template <class T>
void Read(std::FILE* f, T* p, size_t n) {
  if (n != 0 && std::fread(p, sizeof(T), n, f) != n) Fail("short read");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6 && argc != 7) Fail("usage: gp_kaminpar GRAPH.bin K EPSILON THREADS OUT.bin [WEIGHTS.bin]");
  const auto k = static_cast<kaminpar::shm::BlockID>(std::strtoul(argv[2], nullptr, 10));
  const double epsilon = std::strtod(argv[3], nullptr);
  const int threads = std::atoi(argv[4]);
  if (k < 2 || threads < 1) Fail("K must be >= 2 and THREADS >= 1");

  std::FILE* in = std::fopen(argv[1], "rb");
  if (in == nullptr) Fail("cannot open GRAPH.bin");
  uint64_t n = 0, m = 0;
  Read(in, &n, 1);
  Read(in, &m, 1);
  std::vector<uint64_t> xadj(n + 1);
  std::vector<uint32_t> adj(m);
  Read(in, xadj.data(), xadj.size());
  Read(in, adj.data(), adj.size());
  std::fclose(in);
  if (xadj[0] != 0 || xadj[n] != m) Fail("bad xadj");
  std::vector<kaminpar::shm::NodeWeight> vwgt;
  if (argc == 7) {
    std::FILE* wf = std::fopen(argv[6], "rb");
    if (wf == nullptr) Fail("cannot open WEIGHTS.bin");
    uint64_t wn = 0;
    Read(wf, &wn, 1);
    if (wn != n) Fail("WEIGHTS.bin has another node count");
    std::vector<uint32_t> w(n);
    Read(wf, w.data(), n);
    std::fclose(wf);
    vwgt.assign(w.begin(), w.end());
    for (auto& x : vwgt)
      if (x < 1) x = 1;  // an empty list still has its head
  }

  // Reverse lists, sources in increasing order (a counting sort, as parlay::group_by_index).
  std::vector<uint64_t> rdeg(n + 1, 0);
  for (uint64_t e = 0; e < m; ++e) {
    if (adj[e] >= n) Fail("neighbor out of range");
    ++rdeg[adj[e] + 1];
  }
  for (uint64_t v = 0; v < n; ++v) rdeg[v + 1] += rdeg[v];
  std::vector<uint32_t> rev(m);
  {
    std::vector<uint64_t> pos(rdeg.begin(), rdeg.end() - 1);
    for (uint64_t u = 0; u < n; ++u)
      for (uint64_t e = xadj[u]; e < xadj[u + 1]; ++e) rev[pos[adj[e]]++] = static_cast<uint32_t>(u);
  }

  std::vector<kaminpar::shm::EdgeID> sxadj(n + 1, 0);
  std::vector<kaminpar::shm::NodeID> sadj(2 * m);
  for (uint64_t u = 0; u < n; ++u) {
    uint64_t j = sxadj[u];
    for (uint64_t e = xadj[u]; e < xadj[u + 1]; ++e) sadj[j++] = adj[e];
    for (uint64_t e = rdeg[u]; e < rdeg[u + 1]; ++e) sadj[j++] = rev[e];
    sxadj[u + 1] = j;
  }

  auto ctx = kaminpar::shm::create_default_context();
  ctx.partition.epsilon = epsilon;
  kaminpar::KaMinPar shm(threads, ctx);
  shm.take_graph(static_cast<kaminpar::shm::NodeID>(n), sxadj.data(), sadj.data(),
                 vwgt.empty() ? nullptr : vwgt.data(), nullptr);
  std::vector<kaminpar::shm::BlockID> part(n);
  const auto cut = shm.compute_partition(k, part.data());

  std::vector<uint64_t> size(k, 0);
  uint64_t total = 0;
  for (uint64_t v = 0; v < n; ++v) {
    const uint64_t w = vwgt.empty() ? 1 : static_cast<uint64_t>(vwgt[v]);
    size[part[v]] += w;
    total += w;
  }
  uint64_t largest = 0;
  for (auto s : size) largest = largest > s ? largest : s;
  std::printf("gp_kaminpar: n %llu, %llu directed edges, k %u, epsilon %.3f%s: cut %lld of %llu "
              "symmetrized edge slots, largest block %llu (%.3f x average)\n",
              static_cast<unsigned long long>(n), static_cast<unsigned long long>(m), k, epsilon,
              vwgt.empty() ? "" : ", weighted", static_cast<long long>(cut),
              static_cast<unsigned long long>(2 * m), static_cast<unsigned long long>(largest),
              double(largest) * k / double(total));

  std::FILE* out = std::fopen(argv[5], "wb");
  if (out == nullptr) Fail("cannot open OUT.bin");
  std::vector<uint32_t> blocks(part.begin(), part.end());
  if (std::fwrite(blocks.data(), sizeof(uint32_t), n, out) != n) Fail("short write");
  std::fclose(out);
  return 0;
}
