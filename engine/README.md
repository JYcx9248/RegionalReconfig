# FusionANNS reimplementation

A from-the-paper reimplementation of

> Bing Tian et al. **Towards High-throughput and Low-latency Billion-scale Vector Search via
> CPU/GPU Collaborative Filtering and Re-ranking.** FAST '25.

The authors have not released code, so everything here is written from the paper's
description. Where the paper leaves details open, the choice made is listed under
[Departures from the paper](#departures-from-the-paper).

**Status**

- The CPU pipeline is covered by unit tests and end-to-end runs, and is clean under
  AddressSanitizer and UBSan.
- The GPU backend has run on an RTX 3060 Laptop (sm_86, CUDA 12.6, under WSL2) and on A100s
  (native Linux): `fusion_selftest` matches the CPU backend on synthetic codes, on 1000
  BIGANN-1M queries and on a real 1024-dimensional index (identical candidates), `fusion_search`
  gives the same recall on both backends, and the end-to-end tests pass with the cluster's nodes
  on the GPU (`RTIER_E2E_BACKEND=gpu`). On a new GPU, run `fusion_selftest` first.
- On the A100 (a 1M-vector Wikipedia index, nprobe 64, n 200) the GPU filter scales with
  concurrent queries: 164 us per query at 1 thread, 175 us at 16 (the CPU filter 646 -> 810 us).
  It is a small part of a query there -- re-ranking's 200 page reads take 2.6-3.7 ms, navigation
  1.1-1.9 ms -- so the GPU backend adds 15-20% QPS on one node. Under WSL2 it did not scale (0.38
  ms at 1 thread, 3.4 ms at 8): each query issues its own small transfers and kernels on its own
  stream with no lock on the path, and the GPU paravirtualization serialized them.
- Real data so far: BIGANN prefixes (1M, 10M, 20M) and Wikipedia chunks embedded at 1024
  dimensions (15.5M float32 vectors, one per page); `scripts/download_subset.py` also fetches
  SPACEV1B and DEEP1B prefixes.

## What is implemented

| Paper | Code |
|---|---|
| §3.1 Hierarchical balanced clustering, posting lists ≈ 10% of N | `src/clustering.cpp` |
| §3.1 Boundary replication (Eq. 2), at most 8 lists per vector | `src/assignment.cpp` |
| §3.1 Navigation graph over list centroids (host memory) | `src/assignment.cpp` (hnswlib) |
| §3.1 Posting lists that store vector IDs only (host memory) | `PostingLists` in `include/fusion/index_meta.h` |
| §3.1 PQ codes of all vectors pinned in GPU memory | `src/pq.cpp`, `src/gpu_filter.cu` |
| §3.2 CPU/GPU collaborative filtering (Fig. 7, steps 1–8) | `src/engine.cpp`, `src/gpu_filter.cu`, `src/cpu_filter.cpp` |
| §3.3 Heuristic re-ranking (Algorithm 1) | `src/rerank.cpp` |
| §3.4 Similarity-aware SSD layout (buckets + page packing) | `src/layout.cpp` |
| §3.4 Intra- and inter-mini-batch I/O dedup, direct I/O | `src/rerank.cpp`, `src/page_reader.cpp` |
| §4 Contention-free GPU memory pool, per-query streams, parallel ID dedup | `src/gpu_filter.cu` |
| §5 Benchmarks and ablations (Figs. 9–12) | `tools/fusion_search.cpp`, `scripts/run_bigann.sh` |

Tiers at query time:

| Tier | Contents |
|---|---|
| Host memory | Navigation graph over list heads, vector-ID posting lists, vector → (page, slot) map (rtier's data nodes: none -- the posting lists carry the locations) |
| GPU memory | PQ codes of all vectors (host memory with `--backend cpu`) |
| SSD | Raw vectors in 4 KB pages, read with `O_DIRECT` through io_uring, Linux AIO or pread |

## rtier additions

This copy of the engine is part of rtier (see the repository root). Additions:

| File | What |
|---|---|
| `include/fusion/partition.h`, `src/partition.cpp` | partition directory: manifest (with the page geometry), list -> partition map, per-partition posting-list segments (version 3: each posting is a vector ID, its raw vector's canonical location and the node that holds the vector; version 2 still loads) with CRC-32 |
| `include/fusion/node_engine.h`, `src/node_engine.cpp` | `NodeEngine`: posting lists held per partition (load/evict while queries run), each partition possibly at several versions (`Filter` / `Rerank` name the version their epoch sees), update batches (U14: `StageVersions` builds a partition's next version next to the current one -- inserts appended, deleted IDs dropped, all or nothing; `DropVersion`; `ForgetVectors`; `SavePartition` writes a version as a segment file for a move), room for inserted vectors (`--max-vectors`, `--max-pages`) and longer lists (`--max-list-len`; a segment with a longer list is refused at load), major epochs (U14: `StageVersions` also creates and removes lists, and adds postings that name another holder; `GetList`; lists created later are learned from staging or from the segments that hold them; the router's `EditGraph`, `SplitCodes` and `NearestHeads`; graph patches staged, searched through, merged, saved; `Navigate` of a graph version; room for heads and lists, `--graph-headroom`), full rebuilds (U14: `SwitchGeneration` moves a node that holds no partition to another generation of the index -- the generation's manifest, list registry, PQ code file and assignment rule replace the current ones, refused unless the codebook and page geometry match; `RawBitmap`; `PQSource::kGenIndex` loads a generation's partition with the PQ codes from its index and the raw vectors from the holders), `Assign` (routes a vector by the build's own rule and gives its PQ code) and `RawAdd` (inserted vectors served from memory, written by the background writer after fetched ones, pausable), node-level PQ codes and raw vectors (fetched on demand, after a migration, from the holder each posting records), navigation graph loadable after startup, primitives `Navigate` / `Filter` / `Rerank` / `SearchLocal` |
| `include/fusion/graph_patch.h`, `src/graph_patch.cpp` | graph patches (U14 major epochs): the bytes a graph edit changed (whole level-0 records and upper link lists of the changed elements, and the graph's globals), a search through a patch staged next to the graph that is step by step hnswlib's search of the patched graph, and the merge in place |
| `include/fusion/pq_store.h`, `src/pq_store.cpp` | node-level PQ store: one code per vector, reference-counted by resident postings, capacity = the node's PQ budget (`Forget` frees deleted vectors' codes, and marks the ones a version still names so that they are freed with it, never cached); also where a vector's raw-vector location is looked up |
| `include/fusion/raw_store.h`, `src/raw_store.cpp` | node-level raw vectors: a sparse local copy of the page file at the canonical offsets, one presence bit per location |
| `include/fusion/layout.h` (`RawLayout`), `src/rerank.cpp` | page geometry and locations; re-ranking by location (the layout-map variant stays for `fusion::Engine`) |
| `include/fusion/read_budget.h`, `src/read_budget.cpp` | a data node's device-read budget (`rtier_node --read-iops`), for several nodes on one SSD where no cgroup can limit each one's read IOPS: charges re-ranking's page reads, the fetch writer's read-backs and the `RAW_GET` reads the page cache misses (probed with `RWF_NOWAIT`) |
| `node/wire.h`, `node/server.{h,cpp}` | rtier wire protocol server (same framing as the Go side, `internal/frame`), and `PeerPool`, the connections a data node uses to fetch raw vectors from another (`RAW_GET`); INFO also reports the node's generation and process ID (its agent samples the process) |
| `tools/rtier_node.cpp` | data-node service; prints `READY tcp=<port>` when listening; `--raw-file` for its raw vectors; `--stats-file` appends a JSON line of counters every `--stats-ms` (CLOCK_MONOTONIC); `--read-iops` sets the read budget |
| `tools/rtier_segment.cpp` | applies a list -> partition assignment (choosing it is open question U2); writes every posting's location in `--layout` |
| `tests/test_node.cpp` | partition files, primitives (two-node split == single-node answer), partition versions, inserted vectors (`Assign` == the build's lists and codes, new IDs and locations, paused writes, backpressure), update batches (next version next to the current one, all or nothing, saved, dropped, deleted codes freed with their last version, never cached), PQ store (with deleted codes), PQ migration between two nodes, raw store, raw-vector migration (on demand from the holder each posting records, from two holders at once, coalesced, streamed, a leaving node drained, on nodes whose index has no page file), read budget (pacing, what is charged, same answers under a binding budget), wire protocol, graph patches (the patch's version searched through it == the router's edited graph, the graph's version alone; merged in place under queries; a later entry from the file; refused patches and edits change nothing; the router's split and nearest heads), list edits (a split inside the partition == the parent's FILTER answer, a merge into another owner's list re-ranked there by a one-hop fetch, new lists learned from a segment, refused edits change nothing), generations (two nodes switch to a second generation built with the first one's codebook and layout -- refused while a partition is resident or for another codebook -- keep their PQ codes and raw vectors, load the new partitions with holders, and answer as a node built on the second generation, fetching from each other) |

Behavior changes to the original engine, both for deterministic results across nodes:

- `HeuristicRerank` keeps the k best by (distance, id); before, a candidate tying the current
  k-th distance was dropped, so the kept set depended on candidate order.
- The GPU filter sorts one 64-bit key (distance bits << 32 | id) with `cub::DeviceRadixSort::SortKeys`
  instead of `SortPairs` on distances, so ties break by ID exactly like the CPU backend
  (`fusion_selftest` checks it).

PQ codes are node-level (`include/fusion/pq_store.h`, `src/pq_store.cpp`; the decided part of
U1): a data node holds one code per vector named by its resident posting lists, in slots of a
buffer sized to its budget (`--pq-capacity`, the HBM budget). Postings hold references, so a
code shared by several resident lists or partitions is stored once. A code is live while a
resident posting names it, staged while a staging needs it, and cached after its partitions
moved away: vectors are immutable, so it stays valid and is kept until a later `PQ_PUT` needs
the room (then cached codes are dropped in slot order; live and staged ones never are). A
deleted vector's code is freed with its last version instead (`Forget`, U14).
Codes arrive through `PQ_MISSING` / `PQ_PUT` (a peer's `PQ_GET`) or, when bootstrapping, from
the index's `pq_codes.bin`. Both filter backends address codes by slot in this mode
(`FilterConfig::slot_addressed`) and still order results by (distance, id); `fusion_selftest`
checks slot addressing against ID addressing.

Raw vectors are node-level too (`include/fusion/raw_store.h`, `src/raw_store.cpp`; the other
decided part of U1). Every vector has a canonical location in the index's page file,
`page * vectors_per_page + slot` (`RawLayout` in `include/fusion/layout.h`), and `rtier_segment`
writes it into the segment next to every vector ID. A data node keeps the vectors it holds in its
own sparse file laid out like the whole page file (`--raw-file`; an anonymous temporary file by
default) with a presence bit per location, and finds a candidate's location through the PQ store,
which records it with the code when a partition loads -- so no node has a vector -> page map, and
the index's page file is read only to bootstrap (`PQSource::kIndex`, `--load`). Pages may have
empty slots (the bucket layout packs tails of other lists into them); only present slots are
read. After a migration a partition loads without its raw vectors (U9). Every posting records the
holder of its vector -- the node that stores it, a `u16` node ID (segment version 3; 0 = none) --
so each key (list, vector) has one current holder: the owner if it has the vector, else the node
the posting names. In memory a posting that says "here" holds `kSelfHolder`, and only the owner's
copy of a list is valid: `SAVE_PARTITION` writes the node's own ID there for the next owner
(every move saves the segment), and `LOAD_PARTITION` makes each posting whose vector is here say
"here" and refuses a segment that names no other holder for a vector this node lacks. Data nodes
know each other by node ID (`SET_NODES`, from the agents). `Rerank` / `SearchLocal` fetch a
vector a query needs before reading pages, when the query first needs it and only then: RERANK
requests carry the query's lists on the node, the engine finds each missing candidate in one of
them, fetches it from the holder that posting records (`RAW_GET`, through the server's
`PeerPool`; one hop, since a holder has the vector; one request per holder, every holder at
once) and makes that posting say "here" -- the one key it fetched for; other lists that name the vector keep a holder that still has it. Queries
that miss a vector another one is fetching wait for that fetch (`coalesced` in the fetch stats).
A posting is made to say "here" only once its vector is present or in the fetch cache, and every
presence check looks at both under the cache's lock, since the writer marks a vector present
before it drops it from the cache. A fetched vector is re-ranked from memory: it stays in a
node-level fetch cache, served to queries and peers from there, until a background writer has
installed it. The writer takes everything queued at once and writes whole pages with `O_DIRECT`,
reading back first only the pages that already hold vectors (batched, Linux AIO), so its installs
never touch the page cache: buffered writes of single vectors made the kernel read each uncached
page synchronously under the file's inode lock, which the queries' direct reads of the same file
then waited for, and left dirty pages that a direct read had to write back first (with a slow
writer a lazy scale-out ran below capacity for minutes after its warm-up;
`results/bigann10m-spatial-scaleout/lazy-diag`). Bulk installs -- bootstrap and streams, sorted
batches that fill a page over several requests -- stay buffered: the page cache combines them per
page, where whole-page writes made a 64 KB stream request take 3.5-4.6 ms and bootstrap three
times longer. The lazy-stream baseline also streams the rest in: `RAW_MISSING` lists each missing
vector once, grouped by the holder a posting records, and `RAW_PULL` has the data node fetch them
from those holders and install them as a stream's batch (or the agents carry them: the holder's
`RAW_GET` over the bulk channel, `RAW_PUT`, with `RAW_CHECK` before each batch dropping what
queries fetched meanwhile). A node that leaves is drained the same way first: `RAW_HELD_BY` makes
the postings naming it whose vector is here say "here" and lists the others, which `RAW_PULL`
brings. A pull and a query may fetch the same vector at once (the second install is a no-op);
either one that finds the posting already saying "here" takes the vector as arrived. Nothing is
dropped: the vectors of partitions that move away stay, like cached PQ codes, so the old owner
can serve the peers whose postings name it, and a partition that comes back needs no transfer.

Known limits of the PQ path:
- `PQ_MISSING` answers in one frame (256 MiB, about 67M IDs). At billion scale a staging has to
  ask for fewer partitions at a time, or the reply has to be paginated (not done yet).
- The GPU backend keeps a host copy of the node's codes (capacity × m bytes of DRAM, used to
  serve peers and to stage copies). Reading codes back from the device would save the copy:
  whether to keep it is U15.
- `PQStore::slot_of_` is a dense `int32` array indexed by vector ID, so it is sized by the whole
  dataset rather than by what the node holds: 4 MB at 1M vectors, 400 MB at 100M, 4 GB at 1B.
  That is the one place the engine breaks the rule that no per-node structure should be sized by
  the dataset, and it exists only because posting lists overlap (a vector lands in r ≈ 7.3 of
  them), so a node that owns several partitions has to answer "do I already hold this code, and
  in which slot". It is touched only on the staging path (the `PQ_MISSING` set difference and
  `PQ_PUT`), never by a query, so it costs DRAM and not latency. The replacement is a
  vector-ID-sorted `(id, slot)` array: binary search for a single lookup and a merge for the set
  difference, which is what `PQ_MISSING` computes anyway. It costs 8 B per resident vector, so it
  wins below 50% residency -- at the ~40% a billion-vector eight-node cluster would see it saves
  under 1 GB, at the 20% a locality-aware U2 could reach it saves about 2.4 GB, and under random
  grouping (~62%) it loses. Worth doing once U2 is settled or the dataset passes ~100M vectors;
  confined to `include/fusion/pq_store.h` and `src/pq_store.cpp`, since `slot_of_` is private and
  reached only through the "is it here" and "which slot" entry points. A paged two-level array
  does not help: vector IDs do not correlate with partitions, so a node's vectors are spread
  almost uniformly over the ID space and every page gets touched.
- Released segments drop their references on a background reclaimer thread, not on the query
  that let go of them last.
- `PQ_MISSING` protects the codes it counts as present, so the load that follows finds them;
  `PQ_ABSENT` would only mean a bug, and the agent then lists and pulls again (up to three
  rounds).
- `PQ_RELEASE` (rollback of a staging) is node-wide: it frees every code installed for a staging
  and not yet used, and returns protected codes to the cache. The agent stops a staging still
  running before a rollback releases.

Known limits of the raw-vector path:
- Holders are kept per posting (2 B per posting, in memory and in the segment: with r ≈ 7.3
  postings per vector, 15 MB per million vectors over the cluster, 25% on top of the 8 B of ID
  and location). A missing vector is found by scanning the postings of the request's lists, only
  when something is missing.
- `raw.pending` (named by resident partitions, neither here nor fetched): the locations the
  resident partitions name are collected into a bitmap after a load or an eviction (one pass
  over the resident postings); INFO counts with a word-wise pass over it and the presence bits.
  INFO waits for nothing: fetched vectors still in memory count as `cached`, and the settle flag
  (tests) first waits for the writer.
- A re-ranking that must fetch holds its engine worker for the round trip -- the slowest one
  when its misses lie at several holders, since those requests run at once (a short-lived thread
  for each holder but the first) -- and so does one that waits for another query's fetch of the
  same vector. With the lazy protocol this lasts for as
  long as queries meet vectors that have not moved. Pulls (stream, drain) do not join the
  queries' fetches: a vector both need at once comes twice.
- A peer's `RAW_GET` puts its device reads in flight together (2026-10-20): a read-ahead hint
  (`POSIX_FADV_WILLNEED`) for each page the cache lacks, then the buffered reads, at most 256 per
  request at a time. 200 vectors on random uncached pages take about 3 ms, against 29-38 ms read
  one after another (this laptop's disk, with and without a read budget). What this changes end
  to end is not measured yet. The reads stay buffered on purpose: reading the pages with
  `O_DIRECT` all at once instead (tried 2026-09-30; patch kept outside the repo) made things worse
  on BIGANN-10M, 2 -> 3. The old owners were at their read-IOPS cap during the scale-out, so their
  reads queued in the cgroup throttle whatever the queue depth, and `O_DIRECT` lost the page-cache
  hits and the readahead the buffered reads got: copy-then-flip's stream took 45 s instead of
  31 s, and lazy's backlog peaked at 3.8K instead of 2.9K. A holder at its read cap gains less
  from reads in flight together, since they add no reads per second.
- The read budget charges only the reads a probe finds missing from the page cache, so the
  kernel's read-ahead reads past it uncharged. Scattered vectors get none, since the kernel reads
  random accesses as asked, but sorted, dense batches do -- a stream's -- so a stream can read
  somewhat more than its budget. How much depends on the device's `read_ahead_kb` (8 MB on this
  laptop's virtual disk).
- Installing a vector on a page that already holds vectors reads that page back first (list
  tails share pages, and the slots of other vectors must be written back unchanged): one read per
  such page, from the node's own read IOPS. The writer's rounds let the vectors of one page that
  different queries fetched share it. One installer at a time (bootstrap, stream, writer).
- The fetch cache holds at most 64 MB of vectors waiting for the writer; past that, the fetching
  query installs its vectors itself (backpressure on the query path).
- There is no disk budget: a node keeps every vector it has held (the counterpart of the PQ
  budget does not exist yet), and the local file has the page file's full logical size, sparse.
- `RAW_MISSING` answers in one frame, like `PQ_MISSING`.
- Presence is not persisted: a restarted node starts with an empty raw-vector file.

`fusion_tests`, `rtier_node_tests` and the rtier end-to-end tests (run against an ASan/UBSan
build of `rtier_node`, including a shutdown under load) are clean under the sanitizers.

## Build

You need CMake ≥ 3.18, a C++17 compiler with OpenMP, and Linux. CUDA ≥ 11 and liburing are
optional.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/fusion_tests        # unit + small end-to-end tests (CPU)
./build/fusion_selftest     # GPU vs CPU filtering on synthetic codes
```

| Option | Default | Effect |
|---|---|---|
| `-DFUSION_CUDA=AUTO\|ON\|OFF` | `AUTO` | Build the GPU backend when a CUDA compiler is found |
| `-DCMAKE_CUDA_ARCHITECTURES=…` | `native` | e.g. `70` for V100 (the paper's GPU), `80` A100, `89` L40/4090, `90` H100. CUDA 13 no longer supports `70`. |
| `-DCMAKE_CUDA_HOST_COMPILER=g++-12` | – | If nvcc rejects your default GCC |
| `-DFUSION_NATIVE=OFF` | `ON` | Drop `-march=native` when building for another machine |

## Quick start

```sh
python3 scripts/download_subset.py --dataset bigann --n 10000000 --out data/
./build/fusion_build    --base data/bigann-10M.u8bin --out idx-bigann-10M --layout both
./build/fusion_selftest --index idx-bigann-10M --queries data/bigann-query.u8bin
./build/fusion_search   --index idx-bigann-10M --queries data/bigann-query.u8bin \
                        --gt data/bigann-10M-gt.ibin \
                        --nprobe 16,32,64,128 --rerank 50,100,200 --threads 16 --csv results.csv
```

`scripts/run_bigann.sh` runs the whole flow: build, tests, data, ground truth, index,
self-test, parameter sweep, thread scaling and ablations. Set `DATASET=deep|msspacev` and
`N=...` to change the data. For prefix sizes without official ground truth, use
`fusion_gt` (brute force). Without real data, `scripts/make_synthetic.py` generates a
clustered stand-in.

## Parameters

Build (`fusion_build --help`):

| Flag | Default | Source |
|---|---|---|
| `--centroid-ratio` | 0.1 | Paper: posting lists ≈ 10% of vectors |
| `--replicas` | 8 | Paper: at most 8 lists per vector |
| `--closure-eps` | 10 | Eq. 2; value from SPANN (ε₁ = 10), so almost every vector reaches 8 lists (the paper's "8×" expansion) |
| `--rng` | off | SPANN's RNG pruning of replicas. Not mentioned in FusionANNS; cuts replication to about 2× |
| `--hnsw-m` | 32 | Level-0 degree 64, matching the paper's "top-k (typically 64)" graph neighbours |
| `--pq-m` | dim/8 (dim/4 if dim is not a multiple of 8) | Not given in the paper. With 1B codes in a 32 GB V100 it must be ≲ 30 B/vector |
| `--layout` | bucket | `id` is the unoptimized layout, for ablations; `both` builds both |

Search (`fusion_search --help`):

| Flag | Default | Notes |
|---|---|---|
| `--nprobe` (m), `--rerank` (n) | 64, 100 | The two knobs the paper tunes per dataset and recall target |
| `--eps`, `--beta`, `--batch` | 0.1, 1, k | Algorithm 1 settings from the paper. Raise `--beta` if early stopping costs too much recall |
| `--ef` | 2 × nprobe | Graph search width, specific to this implementation |
| `--backend gpu\|cpu` | gpu if available | The CPU backend is the paper's "MI (CPU)" variant |
| `--io`, `--io-depth`, `--no-direct` | auto, 64 | auto tries io_uring, then AIO, then pread |

Output columns per query: `cand` IDs gathered (with replicas), `unique` after dedup,
`batches` mini-batches re-ranked, `reranked` exact distances, `reads` SSD page reads,
`merged` reads saved within a mini-batch, `bufhit` reads saved by the per-query DRAM buffer,
and the mean time of each stage.

### Ablations (Figure 12)

| Variant | Flags |
|---|---|
| MI (CPU) | `--backend cpu --layout id --no-heuristic --no-io-dedup` |
| MI (GPU) | `--backend gpu --layout id --no-heuristic --no-io-dedup` |
| MI (GPU) + HR | `--backend gpu --layout id --no-io-dedup` |
| FusionANNS | `--backend gpu --layout bucket` |

## Departures from the paper

Some details are not in the paper; others were changed on purpose.

1. **Navigation graph.** This uses HNSW (hnswlib) over the list heads. The paper uses SPTAG's
   graph. Either one only has to return the top-m centroids.
2. **Clustering.** This uses recursive k-means with k-means++ seeding, a size-balance penalty
   and merging of tiny clusters. SPANN uses SPTAG's BKT-based head selection. Each list's head
   is the member vector closest to its mean, as in SPANN.
3. **Replication.** Eq. 2 is implemented with SPANN's ε₁ = 10 and RNG off by default (see
   Parameters). The build log prints the replication factor you actually get.
4. **Page packing.** The paper's "max-min algorithm" cites a paper on Elias-Fano index
   compression, so the method is unspecified. This uses largest-first best-fit on bucket
   tails. Tails are never split.
5. **GPU kernels.** This uses a lock-free `atomicCAS` hash table and one thread per candidate,
   with the distance table in shared memory, then a CUB radix sort. The paper describes a
   spinlock hash table and one thread per PQ dimension plus a coordinator thread.
6. **I/O.** Each mini-batch's page reads are submitted together. The paper only says
   "direct I/O".
7. **DRAM buffer.** Following Fig. 8, the buffer is per query: pages read by earlier
   mini-batches of the same query are reused. There is no cross-query page cache.
8. **Stop rule.** Algorithm 1 is checked only once the heap holds k results. With the paper's
   batch = k this changes nothing.
9. **Recall.** A result that ties the k-th ground-truth distance counts as a hit. This matters
   for integer data such as SIFT.

## Sanity results (synthetic, 1M × 128 uint8, CPU backend, 2-core VM)

These are **not** comparable to the paper's numbers. They only show that each technique moves
the metric it should.

| Variant (nprobe 64, rerank 200, PQ 32 B, 7.3 replicas/vector) | Recall@10 | SSD reads/query | QPS (2 threads) |
|---|---|---|---|
| MI (CPU): no HR, no dedup, `id` layout | 0.980 | 200 | 796 |
| + heuristic re-ranking | 0.837 | 35.5 | 2847 |
| + I/O dedup with `bucket` layout (full FusionANNS, CPU filter) | 0.837 | 20.5 | 3424 |
| I/O dedup with `id` layout (no similarity layout) | 0.837 | 35.5 | 2909 |
| `bucket` layout + dedup, heuristic off | 0.980 | 59.9 | 1582 |

With `--beta 3` recall rises to 0.917 at 6 mini-batches, versus 20 with the heuristic off.

On this synthetic data, PQ reconstruction error (≈ 38K) is larger than the 10th-neighbour
distance (≈ 25K). That noise is what makes Algorithm 1 stop early and lose recall. As a
check, the PQ implementation's error matches Faiss's `ProductQuantizer` on the same data
(38.1K vs 38.2K). Measure the recall/work trade-off on real datasets.

## Limitations

- L2 distance only; vectors of at most 4 KB (at least one per page). Updates (U14): batches of
  inserts and deletes, major epochs (splits and merges, with graph patches) and full rebuilds to
  a next generation work, but the ID and location spaces do not grow beyond the room left at
  startup (`--max-vectors`, `--max-pages`), inserted vectors' writes are not charged to the read
  budget, and deleted vectors' pages are not freed.
- One GPU per engine.
- The builder keeps N × 8 assignment IDs in RAM during the build.
- hnswlib stores a hash map entry per head, which adds several GB at 100M heads.

## Repository layout

```
include/fusion/   public headers (engine.h is the query API)
src/              library sources (gpu_filter.cu is the CUDA backend)
tools/            fusion_build, fusion_search, fusion_gt, fusion_selftest
tests/            unit and end-to-end tests
scripts/          dataset download, synthetic data, end-to-end run script
third_party/      hnswlib v0.8.0 (Apache-2.0), vendored unmodified
```

hnswlib's SSE prefetch code reads a few bytes past a link list. AddressSanitizer reports this
benign read, so add `-DNO_MANUAL_VECTORIZATION` when building with sanitizers.
