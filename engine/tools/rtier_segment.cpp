// Splits an index's posting lists into partitions (rtier).
//
// The list -> partition assignment is an input: choosing it is an open design question
// (TODO(design) U2: which lists belong together, and how many partitions). This tool only
// applies a given assignment. For tests, scripts/testing/make_range_assignment.py writes a
// placeholder assignment.
//
// Every posting is written with its vector's location in the index's page file (the layout map
// of --layout), so data nodes can hold any subset of the pages at the same offsets (U1).
#include <algorithm>
#include <cstdio>

#include "cli.h"
#include "fusion/index_meta.h"
#include "fusion/partition.h"

using namespace fusion;

static void Usage() {
  std::fprintf(stderr, R"(usage: rtier_segment --index DIR --assign FILE --out DIR [--layout L]

  --index DIR     index built by fusion_build (reads postings.bin and layout_<L>.map)
  --assign FILE   uint32 partition per list (uint64 count header, as in heads.bin)
  --out DIR       partition directory to write (manifest.txt, list_part.bin, part-*.seg)
  --layout L      page layout the raw-vector locations refer to: bucket | id (default bucket)
)");
}

int main(int argc, char** argv) {
  const std::set<std::string> flags = {"help"};
  try {
    Args a(argc, argv, flags);
    if (a.Flag("help") || !a.Has("index") || !a.Has("assign") || !a.Has("out")) {
      Usage();
      return a.Flag("help") ? 0 : 2;
    }
    const std::string index = a.Required("index");
    const std::string out = a.Required("out");
    const std::vector<uint32_t> assign = LoadU32Array(a.Required("assign"));
    const std::string layout_name = a.Str("layout", "bucket");
    a.WarnUnused(flags);

    PostingLists lists = PostingLists::Load(JoinPath(index, files::kPostings));
    const LayoutMap layout = LayoutMap::Load(JoinPath(index, files::LayoutMap(layout_name)));
    uint32_t np = 0;
    for (uint32_t p : assign) np = std::max(np, p + 1);
    const auto sizes = WritePartitions(lists, assign, np, out, layout, layout_name);

    std::vector<uint32_t> nlists(np, 0);
    for (uint32_t p : assign) nlists[p]++;
    const auto [mn, mx] = std::minmax_element(sizes.begin(), sizes.end());
    uint64_t total = 0;
    for (uint64_t s : sizes) total += s;
    std::printf("wrote %u partitions to %s: %u lists, %.2f MB of posting lists with locations in "
                "the '%s' layout (per partition: min %.1f KB, max %.1f KB)\n",
                np, out.c_str(), lists.num_lists(), total / 1e6, layout_name.c_str(), *mn / 1e3,
                *mx / 1e3);
    for (uint32_t p = 0; p < np; ++p) {
      if (nlists[p] == 0) std::printf("warning: partition %u is empty\n", p);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
