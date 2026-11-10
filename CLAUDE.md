# rtier — notes for Claude Code

rtier is a prototype of non-disruptive reconfiguration (scale-out / scale-in) for the regional
tier of a distributed, FusionANNS-style vector search system. A Go control plane (structure
reused from Koala) drives C++ data nodes built on our FusionANNS reimplementation (`engine/`).
The goal of the research: add or remove nodes under query load with exact answers throughout,
a short flip, and throughput that follows the load.

## Build and test

Linux only (the end-to-end tests and the network emulation need it). CMake ≥ 3.18, a C++17
compiler with OpenMP, Go ≥ 1.24, Python 3 (numpy for the data and testing scripts). CUDA ≥ 11 and
liburing are optional.

```sh
make CUDA=OFF            # engine in engine/build + bin/rtier-*
make test CUDA=OFF       # fusion_tests, rtier_node_tests, Go tests with -race, end-to-end tests
make e2e                 # end-to-end tests only, verbose
make fmt && make vet
RTIER_ENGINE_BIN=$PWD/engine/build go test -race ./...   # go test directly: e2e needs the engine
```

- The Makefile exports `GOTOOLCHAIN=local`: never download a Go toolchain.
- Experiments run `bin/`, which `go test` does not rebuild: `make test` builds it too, and
  `run_local.py` and `run_cluster.py` refuse to start when `bin/` or the engine's binaries are
  older than their sources (run `make`). Each results directory records the commit and the diff
  it ran (`revision.txt`, `revision.diff`).
- `-march=native` is on by default (`FUSION_NATIVE`). Rebuild the engine on each machine: a build
  copied from another CPU can die with SIGILL. Use `-DFUSION_NATIVE=OFF` for a portable build.
  `run_cluster.py` checks every host's CPU and GPUs against the build before it starts anything.
- Sanitizers: build the engine with `-fsanitize=address,undefined` into another directory and
  point `RTIER_ENGINE_BIN` at it; the end-to-end tests fail on any sanitizer report.
- The end-to-end tests start real `rtier_node` processes on a small synthetic fixture
  (8000 vectors, 8 partitions) and take about 30 s. `RTIER_E2E_TIMELINE=<dir>` makes
  `TestReconfigUnderLoad` write each protocol's per-query latency and rescale times for
  `scripts/compare_runs.py` (README, "Mean latency over time"); measure without `-race`.
  `RTIER_E2E_BACKEND=gpu` (with a CUDA build in `RTIER_ENGINE_BIN`) runs the cluster's nodes on
  the GPU backend against the CPU oracle. Nodes that join after the initial
  deployment run on a copy of the index without its page file and PQ codes
  (`indexNoBootstrap`): they can only get raw vectors and PQ codes from peers (so do the initial
  nodes of `TestBootstrapFromSnapshot`, which start on their saved raw vectors and PQ codes).
  Every node runs with `--stream-batch 100`, so each staging and drain streams in many frames.
- Configuration as in Koala: one system configuration, `config.yaml` at the repository root
  (the controller's and the agents' keys; `bin/rtier-controller -defaults` prints them all),
  which every process reads at startup, and one JSON file per experiment
  (`configs/experiment.*.json`, flat keys, complete on its own). Before each run the runner
  rewrites `config.yaml` in place -- every key at its default, then the keys the experiment
  sets by name, then the run's addresses and directories -- and the results keep a copy; an
  agent's own settings (name, data node, directory) come from its flags. The committed
  `config.yaml` holds the defaults: leave a run's rewrite out of commits. `python3
  scripts/experiment_config.py CONFIG.json` checks an experiment: every key against its schema.
  `run_local.py` runs every node on one machine; `run_cluster.py` runs them on the hosts of
  `"Hosts"`, over SSH.
  `PATH=$PWD/scripts/testing/fake_ssh:$PATH` tries `run_cluster.py` on one machine without an
  SSH server; with `FAKE_SSH_PRIVATE` (directories) and `FAKE_SSH_ROOT` each host sees its own
  copy of them (a mount namespace, root only), so the copies are made and the nodes run on them.

## Layout

| Path | What |
|---|---|
| `cmd/rtier-controller`, `cmd/rtier-agent` | control plane; per-node agent next to `rtier_node` |
| `cmd/rtier-client`, `cmd/rtier-loadgen` | status / rescale / rebuild / design list / each initial node's PQ codes (`pq-share`, offline) / a node's warmup snapshot: raw vectors (`save-raw`) and PQ codes (`save-pq`); open-loop or closed-loop load |
| `cmd/rtier-updater` | the update controller's command: one batch of inserts and deletes, or a major epoch (U14) |
| `cmd/rtier-overhead` | fan-out cost analysis: one data node emulates N owners (`-owners`, `-compare`) |
| `internal/epoch` | epoch table, CAS store, `Tracker` (pins, grace period) |
| `internal/controller` | registration, placement, reconfiguration (`reconfig.go`), update leases (`update.go`), full rebuild (`rebuild.go`); collects metrics and the nodes' diagnostics |
| `internal/agent` | staging, epoch install, entry and aggregator roles (`serve.go`), generations and the rebuild's warm-up (`generation.go`), diagnostics forwarding (`diag.go`), the graph's yield to data (`yield.go`) |
| `internal/update` | update batches (`batch.go`) and major epochs (`major.go`), with their router |
| `internal/query` | selectors (aggregator choice), strategies (`TwoPhase`), wire format |
| `internal/placement`, `internal/transfer` | placement policies, plans, the rebuild's placement (`affinity.go`); bulk transfer + token bucket |
| `internal/design` | registry of open design questions (U1–U17) |
| `engine/` | FusionANNS engine, `NodeEngine`, `rtier_node`, `rtier_segment` (see `engine/README.md`) |
| `test/e2e` | end-to-end tests |
| `config.yaml` | the system configuration (defaults committed; the runners rewrite it before each run) |
| `configs/` | experiments (JSON, one file each) |
| `scripts/` | `run_local.py` and `run_cluster.py` experiments, `experiment_config.py` (the experiment format), `plot_run.py`, `compare_runs.py` (mean latency per run, overlaid), `run_one.sh` (one run on a clean node state, on one machine or a cluster; `evict_cache.py`), `node_reads.py`, `emulation/netns.sh`, `cloudlab/` (profile and node setup), `testing/` (placeholders, the GP partitioner, `fake_ssh`) |

## Conventions

- Go uses the standard library and `gopkg.in/yaml.v3` (config.yaml, as Koala), vendored
  (`vendor/`) so that a build needs no network. The module path `rtier` is a placeholder.
- The runners (`run_local.py`, `run_cluster.py`, `run_one.sh` and what they call) use the standard
  library and PyYAML, vendored (`scripts/vendor/yaml`, pure Python): a run needs nothing but
  `python3`, never a venv (a shared server's system Python takes no packages, PEP 668). numpy
  only in the data and testing scripts.
- Open design questions live in `internal/design/design.go` (U1–U17; U4 is out of scope).
  Code that reaches an undecided question returns `design.Undecided("Un", "...")` — never
  silently pick an answer. When something gets decided, update that entry (decided part, open
  part, code location) and the matching paragraph in `README.md`.
- Test-only placeholders (e.g. the range list→partition assignment in `scripts/testing/`) are
  not design choices.
- End-to-end tests compare every answer with the single-node oracle, exactly, across
  reconfigurations. Races are reproduced deterministically with the hooks in `agent.Options`
  (`Strategy`, `Selector`, `BeforeInstall`, `BeforeGraph`, `BeforeRaw`); see
  `test/e2e/flip_race_test.go` and `TestRawVectorsFetchedOnDemand`.
- Comments explain why (the invariant, the trade-off, the U-item), not what the code does.
- Koala-derived code is marked in package comments (see `NOTICE`); engine additions are listed
  in `engine/README.md`.

## Key mechanisms

- **Epoch** = versioned routing table: partition owners, entries, aggregators, addresses; with
  U14 also each partition's version, the graph version and the index generation. A query runs
  under the one epoch its entry pinned; a forwarded aggregation carries that epoch and the lists
  grouped by owner, and the aggregator neither routes with nor checks an epoch of its own: it
  counts the work under the query's epoch (`Tracker.Pin`), so its own drain waits for it too.
- **Query**: entry NAVIGATEs → groups the lists by owner → the `owner` selector picks the
  aggregator → `TwoPhase`: FILTER (local top-n by PQ) at each owner, global top-n merge, RERANK
  of each candidate at one owner that reported it, top-k merge. Equal to the single-node answer.
- **Reconfiguration** (`copyThenFlip`, protocol `lazy` by default): pre-connect (aggregators to
  the data nodes, entries to the aggregators: `StageAggregator`) →
  new nodes pull the moved posting lists and the PQ codes they lack (the graph after them: it
  moves only while neither end stages posting lists or PQ codes, `graphYield`, and does not wait
  for raw vectors, 2026-11-08; sources round-robin over entries) → flip (CAS + install) → grace (drain the old
  epoch) → reclaim (old owners drop moved posting lists; PQ codes and raw vectors stay cached).
  Lists and graph go from the source's memory into the new node's, never through a disk, the
  agents relaying the chunks: the graph `GRAPH_GET` → `GRAPH_PUT` → `GRAPH_INSTALL`; a
  partition `SEGMENT_GET` (every posting with its current holder) → `SEGMENT_PUT` →
  `PQ_MISSING` over the received segments → `SEGMENT_INSTALL` (since 2026-11-07: before, the
  old owners first wrote the moved partitions as files, `save_segments`, 23-24 s on Amazon
  300M, and the new nodes loaded them from their disks). Only a bootstrap and a rebuild's next
  generation load segment files.
  Alongside grace and reclaim, `promoteEntries` flips every `EntryFlipInterval` to make
  entries of the new nodes whose graph has loaded, each connected to every aggregator first;
  the rescale waits for it. Raw vectors do
  not move with a scale-out: `lazy` fetches one only when a query needs it (below). Baselines:
  `lazy-stream` also streams the rest in after the flip (`StageRaw`; the rescale waits for it),
  `copy-then-flip` and `stop-and-copy` stream them before the flip (eager). A rescale to a plan
  (`rtier-client rescale -plan`) carries out any target placement. The token bucket paces only
  what goes through the agents -- posting lists and PQ codes (data class) and the graph (background);
  raw vectors move data node to data node (`RAW_GET`, `RAW_PULL`), unpaced, unless
  `RawPath: agent` (then at `RawPriority`, U8).
- **Listings stream** (`PQ_MISSING`, `RAW_MISSING`, `RAW_HELD_BY`; `engine/node/stream.h`): the
  node lists what it lacks, each entry once, in frames of about 1M entries, each sent when full,
  when its first entry has waited 100 ms, or as the last; the agent pulls what a frame names as
  it arrives, several requests in flight per source (`pullMissingPQ`, `rawPuller`).
- **PQ codes** are stored per node, once per vector, with a cache of codes of partitions that
  moved away (`engine/include/fusion/pq_store.h`). A resident partition references each code
  once; a load takes its distinct vectors' references under the store's lock and does the
  per-posting work outside it, 8 partitions at once (`loadStreams`), and RERANK's lookups take
  no lock (2026-11-07). A Put claims its new IDs with a CAS and writes the codes outside the
  lock, so a staging's PQ_PUTs install side by side; the agent keeps 4 `PULL_PQ`s in flight per
  source and 4 `PQ_PUT`s to its node (`pqStreams`, `pqPutters`, 2026-11-08). Placement
  `even-reversible-graph` (the
  default) returns partitions to former owners so a scale-in moves no PQ code. The runners give
  every node a budget (`--pq-capacity`) of the fullest node's codes at the smallest cluster size,
  plus 10% (`"PQCapacity": "auto"`, `rtier-client pq-share`; a run with a rebuild gets the
  whole ID space).
- **Raw vectors**: every posting carries its vector's canonical location in the index's page
  file and the node that holds it (segment v3); a node keeps a sparse local copy of that file
  with a presence bit per location (`engine/include/fusion/raw_store.h`) and looks locations up
  through its PQ store. After a migration, RERANK fetches a missing vector when a query first
  needs it, and only then, from the holder that a posting of the query's lists on that node
  records: one hop, data node to data node (`RAW_GET`, `PeerPool`; nodes know each other by ID,
  `SET_NODES`), and that posting then names the fetching node. A fetch brings only what its
  query needs, from every holder at once, no read-ahead: vectors move beyond that only in a
  scale-in's drain (U9). A RERANK fetches before it takes an engine worker, and holds a
  connection slot of its own at the aggregator (`nodeclient`): FILTERs wait behind it in
  neither place.
  Concurrent misses of one vector wait for one fetch, and a stream's or a drain's pull
  (`RAW_PULL`) shares it with the queries both ways (`Claims`; not the stream through the agents). Fetched vectors are re-ranked from memory
  (fetch cache) until a background writer installs them with whole-page `O_DIRECT` writes
  (`WriterLoop`, `RawStore::PutPages`); INFO never waits for it (`nodeclient.InfoSettled` does,
  for tests). A scale-in drains the leaving nodes before it returns (`drainRaw`: each remaining
  node takes over the vectors only a leaving node holds, `RAW_HELD_BY` then `RAW_PULL`), so they
  can go offline. No node drops a raw vector it has held. Only bootstrap reads the index's page
  file, and not on a warmup snapshot (below).
- **Warmup snapshots** (`"Type": "warmup"`, `run_local.py`): a run that only deploys the
  initial nodes and moves each one's raw vectors -- its raw file and presence bits (`RAW_SAVE`,
  `rtier-client save-raw`) -- and its PQ codes (`PQ_SAVE`, `rtier-client save-pq`, `pq.codes`)
  into `"SnapshotDir"`, on the node's machine (a rename: one copy on disk, even on ext4). A
  later run of the same index, partitions, placement and initial node count restores the raw
  files (`cp --reflink=auto --sparse=always`) and starts those nodes with `rtier_node
  --raw-reuse --pq-file <snapshot>/n<i>/pq.codes`, the codes read where they are, at the node's
  start: their bootstrap copies no raw vector and reads no PQ code (the agent skips
  `PQ_FROM_INDEX`), `run_cluster.py` copies neither the page file nor `pq_codes.bin` to their
  hosts, and the run checks that each got its snapshot's partitions. A snapshot is keyed by content (the
  index's `meta.txt`, the partitions' manifest, list map and segment sizes and CRCs, placement,
  initial node count), never by path or mtime: one made on the lab server serves AWS hosts that
  have the same files (copied, not rebuilt) and its `snapshot.json` and `n<i>/` under the same
  `"SnapshotDir"`; `"RequireSnapshot": true` stops a run whose initial nodes lack it. A
  snapshot made before 2026-11-07 has no PQ codes: its nodes read theirs from `pq_codes.bin`,
  all of a node's in one pass (`PQ_FROM_INDEX`); `rtier_node` opens that file only to read from
  it, so a node that joins later, its codes from peers, needs none. Nodes start one at a time, each after the previous one registered, so IDs, and with them the initial
  placement, follow the order. Stage records name files by absolute path: prepared data moves as
  is at the same paths.
- **Updates (U14)**: the update controller (`internal/update`, `rtier-updater`) takes the
  controller's lease and runs a batch as a content epoch: inserts routed by its own router
  (`ASSIGN`, the build's rule), PQ codes and raw vectors sent to the owners, the next version of
  each changed partition staged next to the current one (`STAGE_VERSIONS`), then commit = flip,
  grace, reclaim (deleted vectors' codes are freed, never cached). A major epoch splits and
  merges posting lists and ships the graph change as a patch every entry stages, searches
  through after the flip and merges after the grace period. Only with the controller's
  `AllowUpdates` (`"Updates"` in an experiment).
- **Full rebuild (U14, `rebuild.go`)**: moves the cluster to the next index generation, built
  outside with the same codebook, vector IDs and raw-vector locations. Spares take every
  partition by a lazy reconfiguration; the old nodes lose their roles and switch generations
  (`SWITCH_GENERATION`, keeping PQ codes and raw vectors); the new partitions go where their raw
  vectors are (presence bitmaps, `placement.Affinity`); the old nodes load them, replay recent
  queries to warm up, and take over in one flip; the spares are released. Manual trigger only
  (`rtier-client rebuild`, `"Reconfigurations": [{"Rebuild": ...}]`); baselines: random placement,
  no warm-up, stop-the-world.
- **Clients' entry lists (U17)**: a client (`rtier-loadgen`) reads the entries from the
  controller every 500 ms, and at once when an entry refuses a query under a newer epoch than
  its list's (the queries that meet the stale list share one read); a refusal under the list's
  own epoch (paused admission) reads nothing.
- **Diagnosing a run**: `metrics.jsonl` (the agents' counters too, e.g. `graph.yield_ms`, how
  long the graph waited for data), `stats-<node>.jsonl` (per-node counters, `rtier_node
  --stats-file`; `raw_bytes` sent and received) and `cgroups.jsonl` (CPU, disk, dirty pages,
  pressure; a host record with the whole disk's I/O and how busy each node's CPUs were; on a
  cluster one per node's host, `<node>-host`, with its CPU count; `scripts/node_reads.py`). The
  agents forward the stats logs (`rtier-agent -node-stats`) and, on a cluster, their samples
  (`SampleInterval`; `run_local.py` samples itself with `"NodeResources"`), and the
  controller stamps every record and metrics report on arrival (`"t"`, on its monotonic clock;
  the node's own as `"node_t"`): one clock with `clock.json` and the load generator, without
  synchronizing the machines' clocks (Koala's metric collector).
- **Without root** (shared servers): `"NodeResources"` pins CPUs with `taskset` and limits
  memory in user scopes (`systemd-run --user`); per-node read IOPS come from the data node's own
  read budget (`"ReadIOPSPerNode"` → `rtier_node --read-iops`, `fusion/read_budget.h`), since
  the cgroup io controller is not delegated; `run_one.sh` evicts only the run's own files
  (`scripts/evict_cache.py`).
- **Cluster runs** (`run_cluster.py`, as Koala deploys): this machine runs the stages, the
  controller and the load generator; it checks every host and leaves it clean (no earlier
  node directory; where sudo asks no password, an empty page cache, swappiness 1 and Ubuntu's
  weekly `fstrim.timer` stopped: a TRIM held a SATA SSD for 8.6 min of a run on 2026-11-09),
  copies with rsync, to the same
  paths, the binaries, `config.yaml` and what the host's nodes open (the index's meta and
  codebook and the partitions' manifest; where an initial node runs, also the graph, the
  segments, and the PQ codes and the page file unless the host's initial nodes start on a
  warmup snapshot of them; a next generation's meta, codebook, PQ codes, graph and manifest),
  checks with `ldd` that the
  binaries find their shared libraries there, and starts the nodes in order
  over SSH, each after the previous one registered. A remote process stops when its SSH session
  closes. `"NodeResources"` (except `"ReadIOPSPerNode"`) and `"Emulation"` are for one machine
  and are refused. On CloudLab, `scripts/cloudlab/`: a profile (a control node and N hosts on
  one LAN at fixed addresses, 10.10.1.1 to N+1), each node's setup (`setup_node.sh`: libraries,
  the SSD as XFS at the lab server's paths; `--gpu`: the NVIDIA driver and CUDA 12) and the
  control node's key on the hosts (`authorize_control.sh`); the `*-cloudlab-cpu-*` experiments
  run the lab's Wikipedia runs on CPU nodes (the CPU backend: a check of the workflow on
  separate machines, not the paper's results, which run on GPUs). The `*-cloudlab-gpu-*`
  experiments run on c240g5 (a P100 per node, node0 the control node): 2 → 4 (`2to4-*`) on the
  2-node warmup snapshot and 4 → 8 (`4to8-*`) on the 4-node one, each with its capacities,
  after `warmup-check-scaleout-*-cloudlab-gpu` on the small index. They run Ubuntu 24.04. The
  P100 is sm_60: configure with `-DCMAKE_CUDA_ARCHITECTURES=60`, since CMake otherwise builds
  for the GPU it finds, and node0 builds before its driver is loaded. The `amazon-300m-*`
  experiments run 4 → 8 there on 300M of the Amazon corpus's 560M documents
  (`scripts/data/subset_base.py`; at 2 nodes the PQ codes would exceed a P100), on a 4-node
  snapshot made on the control node (`amazon-300m-warmup-4`). Their `4to8-worst-*` files are
  U11's worst case there: 2M of the base's vectors as queries, drawn on the control node from
  the page file, with the layout map rebuilt from the partitions' segments
  (`scripts/data/sample_queries.py map`: the control node has no map), and ground truth for every
  1000th of the first 1M (`fusion_gt --index`). The `*-aliyun-gpu-*`
  experiments are the 2 → 4 runs on Alibaba
  Cloud (one A10 instance per data node at 192.168.1.11 on, user root, a 1 TB ESSD each); their
  control node is a CPU instance of the same Xeon, which builds the CUDA backend with
  `-DCMAKE_CUDA_ARCHITECTURES=86` (docs/wiki/Running-on-Alibaba-Cloud.md).

## Docs

- `README.md`: overview, the decided part of each open question, how to run.
- `docs/design.md`: the design doc (Chinese), exported from claude.ai at rev 123 and brought up
  to date here since (last on 2026-10-22); not synced with the claude.ai copy.
- `docs/handoff.md`: what lives only in the conversations, for a new chat to continue from
  (snapshot of 2026-10-22): how the user works, the decisions and their reasons, the state of the
  experiments, what is next; the GitHub issue diagrams; its section 5 is the case analysis of
  queries during a reconfiguration. Like everything under `docs/`, it is not in git: carry it
  over from the working tree.
- `docs/drafts/`: discussion notes and issue drafts (dataset updates, the SSD read model,
  correctness evidence, placement). `docs/server-runbook.md`: runs on the lab server.
- `engine/README.md`: engine additions and known limits.

## Current state and next steps (2026-10-22)

Done since late September:
- Raw vectors: a holder per posting instead of a source per list, one-hop fetches, a scale-in
  drains the nodes that leave (U9). Concurrent misses of one vector share a fetch.
- Dataset updates (U14): minor batches of inserts and deletes as content epochs, major epochs
  (splits and merges, graph patches), and the full rebuild to a next index generation.
- Partitioning (U2): GP on the list heads over the navigation graph, balancing list counts.
  The initial placement stays contiguous ranges (U3).
- Configuration as in Koala (2026-10-22: one `config.yaml`, rewritten by the runners from an
  experiment's JSON; the YAML experiment files of 10-19 are gone); cluster deployment
  (`run_cluster.py`); clients catch up with a stale entry list at once (U17).
- Reference runs: Wikipedia c1024 on the lab server (`results/wiki-gpu`, `docs/design.md`,
  "实验结果"); BIGANN-20M open loop on the laptop (`results/bigann20m-spatial-openloop-v2`). The
  BIGANN-10M runs' directories are gone; their numbers are in `docs/handoff.md`, section 8.
- Fixes from a review of the whole implementation (2026-10-21): the data node's accept loop,
  connection threads and read-budget charges; pre-connect in parallel; every long step under
  `StageTimeout`; a run cut short keeps its latency.csv and reconfigurations.json; plot_run
  counts answers when they came; ground truth for a stride of the queries; the graph's strict
  yield to data (U16, for now); RERANK's own connection slots; each node's PQ budget sized from
  the partitions (`"PQCapacity": "auto"`); the listings streamed; warmup snapshots of the
  initial nodes' raw vectors, and their PQ codes read in one pass (2026-10-22).

Next:
1. Issue: how queries of the old epoch are handled during a reconfiguration — cases, current
   behavior, tests. Accepted (2026-10-16): the grace period waits for every old-epoch query, so
   a slow one holds it; a stale entry list costs one refusal, after which the client reads the
   list (U17); lazy's fetches stay on the query path. Closed (2026-10-20): a stale sub-query
   (U7) needs a failure -- a client that gives up does not end the entry's pin (the agent serves
   under its own lifetime), and an aggregator pins a forward as soon as it reads it -- so it is
   out of scope with the failures (U13) and fails its query with a plain error. Draft of the
   case analysis in `docs/handoff.md`, section 5.
2. A real cluster: `run_cluster.py` first ran on real hosts on 2026-11-01, the smoke test on
   CloudLab CPU nodes (rs620: reuse, lazy 2 → 3 → 2, copy-then-flip 2 → 3, all answers exact).
   Next the GPU runs on 9 c240g5, reserved for 7 days from 2026-11-03 (`*-cloudlab-gpu-*`). On
   one c240g5 (Ubuntu 24.04, driver 570, CUDA 12.8; 2026-11-02), a build for sm_60 passes the
   end-to-end tests on the GPU backend and `fusion_selftest` on the Wikipedia index. The
   Wikipedia 4 → 8 runs (2026-11-03/04, `results/cloudlab-gpu`, figures by
   `scripts/analysis/cloudlab_gpu_figures.py`): every answer the same under both protocols;
   copy-then-flip flips after the load. Lazy restores 8 nodes' capacity within about 70 s, but
   for about 55 s after the flip its throughput falls below 4 nodes'. Two causes:
   - the fetch cache of the new owners of the hottest partitions hit its bound, 64 MB then, and
     past it each fetch waited for the S3500's writes. The bound is 2 GB since 2026-11-04;
   - each new entry wrote its 7 GB graph to the same disk. Since 2026-11-04 the graph moves from
     memory to memory instead.

   The rerun with both fixes (`results/cloudlab-gpu/4to8-v2`, 2026-11-04):
   - lazy answers 500-545 q/s for 5 s after the flip, the offered rate within 15 s;
   - its backlog peaks at about 9K and drains 32 s after the flip; latency afterwards is 14 ms;
   - from the trigger on, 13% of the queries miss 200 ms, against 74-83% before;
   - the new entries come online 69-79 s after the trigger, later than before, because the graph
     yields to the faster fetches;
   - copy-then-flip is unchanged, and every answer is the same in all runs.

   Copy-then-flip with the stream first (`--peer-reads-first`, 2026-11-05, same directory,
   `copy-then-flip-stream-first`; `ctf-stream.svg`, `ctf-budget.svg`, `4to8-stream-first.svg`):
   - its stream takes 203-210 s under the load, as long as calibrate-8's with none (209 s), and
     the flip comes 213-220 s after the trigger, against 497-526 s on one budget;
   - the queries pay for it during the copy: 371-375 q/s against 588; the backlog peaks at 94K;
   - the copy's reads only move in time: the last query is answered at 448-452 s, against 457 s
     on one budget; from the trigger to the load's end, lazy answers 216K queries, copy-then-flip
     159K on one budget and 129-133K with the stream first;
   - each pair's stream follows its new node's buffered writes: about 118 MB/s while the page
     cache takes them, 35-45 MB/s while the kernel writes back; the two runs differ by 7 s there;
   - every answer is the same in all six runs.

   U11's worst case (2026-11-05, `results/cloudlab-gpu/4to8-worst`):
   - **Setup:** 1M queries drawn from the base without repetition (`scripts/data/sample_queries.py`),
     load 550 → 1100 q/s. The reads spread 28/26/24/23% over the nodes, so C4 = 742 and
     C8 = 1466 q/s.
   - **Lazy still leads.** From the trigger to the load's end it answers 296-299K queries, against
     196K on one budget and 153-154K with the stream first. 8-12K queries wait at the load's end,
     against 112K and 154K.
   - **Without locality it pays:**
     - it fetches 27.7 GB, 64% of the copy;
     - it reaches the offered rate 105-110 s after the flip, against 3-7 s;
     - the 2 GB fetch cache fills on n5: 7.3K-12.5K installs on the query's path, and 1 s stalls;
     - the new entries come online only after the load, 314-328 s after the trigger: the graph
       yields to data strictly (U16), and data moves the whole run.
   - Every answer is the same in all six runs.
   - **Held to 570 s** (`4to8-worst-long`, lazy, two runs; `lazy-keepup.svg`): the 360 s runs'
     drain was capped by the load generator's 64 in flight per entry, with 4 entries online.
     - The backlog clears under the load at 394-412 s.
     - Then lazy answers 1100 q/s at 13.6-13.8 ms (p99 30-31 ms), as 8 nodes with every raw
       vector local do, while still fetching 13-20 MB/s.
     - By the load's end it has fetched 31.4 GB, 73% of the copy. Without locality lazy's total
       approaches the copy's; it moves it while serving.
     - `keepup-all.svg` (`--lazy-series 4to8-worst-long`) sets it against the baselines' 360 s runs.
       By 570 s copy-then-flip on one budget has moved as much (30.9-32.5 GB) and still not
       flipped.
     - The answers match the 360 s runs'.

   Amazon 300M 4 → 8 (2026-11-07, `results/cloudlab-gpu/amazon-4to8`; figures `--series
   amazon-4to8` in `results/cloudlab-gpu-figures/amazon-4to8`, `lazy-stall.svg` the new one):
   - **Capacity:** C4 564, C8 872 q/s (1.55x). The reads are skewed: the busiest node takes 1.67
     times an even share on 4 nodes and 2.19 on 8 (Wikipedia 1.37, 1.70). A query re-ranks 200
     candidates in 173 page reads. The runs: load 350 → 700 q/s at 60 s, 4 → 8 at 90 s, 360 s.
   - **Neither protocol flips during the load.** Lazy's stage_data takes 291-306 s, with or
     without load: lists and PQ codes (24.5 GB) arrive in about 100 s, then each new node loads
     its 8 partitions one at a time, about 25 s each (`PQStore::Ref`: a reference per posting,
     about 37M a partition, under the store's mutex). Lazy flips 317-320 s after the trigger.
     Copy-then-flip streams 151 GB after the same staging and flips 1661-1768 s after it.
   - **After lazy's flip the answers stop for about 75 s**, the reclaim: the old owners' `Unref`
     of whole partitions holds the PQ store's mutex, which every RERANK takes (`Locate`). Queries
     that wait over 30 s fail at the client: 153 and 190 per run, 654 in 4to8to4; none under
     copy-then-flip. The staging's `PQ_GET`s on the same mutex halve the answers for ~20 s too.
   - **A run's deployment on the snapshot takes 9.5 min:** the PQ pass about 4 (the codes are in
     the snapshot since, `results/pq-snapshot-2026-11-07`), the 16 partition loads 4.5 (the same
     `Ref`). The 4to8to4 scale-in starts only after the load: its scale-out ends at 504 s.
   - The 1,152 queries with ground truth answered in all four 4 → 8 runs have the same recall
     (0.80) in each.
   - **Fixed on 2026-11-07:** a partition moves from memory to memory, so no `save_segments`; a
     load no longer sorts the postings and references each code once per partition (3.4 s
     instead of 27 at this scale on a laptop), 8 load at once; RERANK's `Locate` takes no lock,
     so the reclaim stalls no query.

   The rerun with these fixes (2026-11-08, `results/cloudlab-gpu/amazon-4to8-v2`, commit
   f77fd51; figures `--series amazon-4to8-v2 --calibrations amazon-4to8 --before amazon-4to8`
   in `results/cloudlab-gpu-figures/amazon-4to8-v2`, `lazy-staging.svg` the new one):
   - **Lazy flips under the load,** 70-72 s after the trigger. Its stage_data takes 70-72 s:
     lists by 16-24 s, PQ codes by 54-66 s, then the 8 loads at once in 6-9 s (4.4-5.8 s of work
     each). The reclaim takes 1.2-1.5 s, nothing stalls, and no query fails in any run.
   - **Lazy answers the offered rate 13-15 s after the flip** and clears its backlog (13.5K at
     the peak) at 265-270 s. From the trigger to the load's end it answers 192K queries, against
     152K under copy-then-flip (149K under both before). Its mean latency over the run is 7 s,
     against 32 s (42 s before).
   - **The drain runs at about C8, with n4 at its read budget:** n4 serves its own queries and
     n7's fetches, and its readers wait 23 s/s. After the drain, the mean latency is 24 ms,
     against 13.9 ms on 8 warm nodes; the new nodes still fetch 12-14 MB/s.
   - **The new entries come online only after the load,** 282-292 s after the trigger. The
     graph yields to data at both ends (U16), and the fetches never stop while the load runs.
     The four old entries keep up meanwhile: navigation takes 2.5-3 ms a query and half a core
     per entry.
   - **Copy-then-flip:** stage_data takes 67-69 s, but its stream now starts under the load and
     takes longer (1587-1664 s). It flips 1657-1731 s after the trigger, as before.
   - **Deployment** on the snapshot takes 238-240 s, against 578-588 s.
   - **4to8to4 shows nothing new:**
     - its committed rates are reversed: 350 until 280 s, then 700. The Comment and the first
       run had 700, then 350;
     - the scale-out's entries came after the load, 403-414 s after the trigger;
     - the scale-in started at 504 s, after the load.
   - The 1,152 queries with ground truth answered in all eight 4 → 8 runs, old and new, have the
     same recall.

   The rerun with the PQ pipeline (2026-11-08, `results/cloudlab-gpu/amazon-4to8-v3`, ec247ac;
   4to8to4 at 917b4ad, its rates the right way round; figures `--series amazon-4to8-v3
   --calibrations amazon-4to8 --before amazon-4to8-v2`):
   - **Each new node collects its PQ codes in 13-17 s** (36-44 s before). The listing takes 6-8 s
     and no longer waits for the pulls. A new node installs up to about 100 PQ_PUTs a second
     (about 400 MB/s, 2.5 of the 4 putters busy). The sources' 64K PQ_GETs now queue on their
     store's lock: 31-49 ms a call.
   - **Lazy flips 44-45 s after the trigger** (70-72 s). The lists (16-24 s) are now the longest
     step, then the PQ codes (13-17 s) and the 8 loads (7-9 s).
   - **Lazy reaches the offered rate 5-6 s after the flip** (13-15 s). Its backlog peaks at
     10.0-10.2K (13.3-13.7K) and clears at 223-226 s (265-268 s). 45-46% of the queries miss
     200 ms from the trigger on (60%), and its mean latency is 4.1-4.2 s (6.8-7.1 s).
   - The staging still costs the queries 5-8% for a few seconds, as before.
   - **Copy-then-flip** stages in 42-45 s and still flips 1636-1718 s after the trigger: its
     stream sets the time.
   - **The new entries still come online only after the load,** 282-297 s after the trigger. In
     4to8to4 the new nodes still fetch 5-7 MB/s at 350 q/s from 280 s on, and every graph transfer
     waits until the load ends (480 s). The entries come at 497-509 s, and the scale-in starts at
     509 s, after the load (it takes 59 s, its flip at 43 s).
   - No query fails in any run. The 1,152 queries with ground truth answered in all twelve 4 → 8
     runs (v1, v2, v3) have the same recall, and so do 4to8to4's in v1 and v3 (the same rates).

   The rerun with the graph yielding to the staging only (2026-11-08, U16,
   `results/cloudlab-gpu/amazon-4to8-v4`, 3a27d05: two runs of each protocol and 4to8to4;
   figures `--series amazon-4to8-v4 --calibrations amazon-4to8 --before amazon-4to8-v3`,
   `lazy-entries.svg` and `4to8to4-before-after.svg` the new ones):
   - **The new entries come online 59-71 s after the trigger** (282-297 s), 13-25 s after the
     flip. Each graph starts as its node's staging ends and moves its 3.3 GB in 22-26 s. Lazy's
     rescale ends at 68-71 s (295-297 s); its staging and flip are as before (flip at 43-46 s).
   - **The cluster answers at the same rate:** the backlog peaks at 9.8-10.3K and clears at
     222-227 s (223-226 s), and 192K queries are answered from the trigger to the load's end, as
     in v2 and v3.
   - **The backlog waits elsewhere** (`lazy-entries.svg`):
     - The load generator sends each query, as it arrives, to the next entry (round-robin), at
       most 64 in flight at each; the rest queue at that entry.
     - The new entries start with empty queues and answer their share at once (350 q/s
       together). The initial ones work through theirs at 494-501 q/s together (848 before).
     - Of the queries sent from 149 s until the backlogs cleared, 47-51% are answered within
       200 ms (17-18%), but the slowest 1% wait 29-31 s or more (17-19 s).
     - From the trigger on, 36-37% of the queries miss 200 ms (45-46%), and the mean latency is
       4.5-5.0 s (4.6-4.8 s).
   - **4to8to4's whole cycle runs under the load.** The scale-out ends at 154 s (509 s). The
     scale-in starts at 300 s and takes 47 s, its flip at 32.5 s:
     - the lists come back from memory in 17-20 s;
     - a PQ listing takes 5 s and finds every code cached;
     - the 8 loads take about 6 s;
     - then a 14.7 s drain pulls nothing: every vector the leaving nodes fetched is still on its
       old holder.
     During its staging the mean latency is 20.1 ms (17.9 ms on 8 nodes without it). After its
     flip, on 4 nodes with every raw vector local, it is 11.4 ms, p99 22 ms, against 14.3 ms and
     32 ms on 8 nodes still fetching. One query, refused by a leaving entry, is answered on its
     retry (U17).
   - Copy-then-flip flips 1691-1746 s after the trigger (1636-1718 s). Its graphs now arrive during
     its stream, 58-70 s after the trigger, so its 4 entries come with the flip.
   - No query fails. The 1,152 queries with ground truth answered in all sixteen 4 → 8 runs
     (v1-v4) have the same recall, and so do the 1,224 of 4to8to4 in v1, v3 and v4.

   The Wikipedia runs again on the code of 2026-11-08 (2026-11-09, 9e6ae33, each experiment as
   in 4to8-v2 and 4to8-worst: `results/cloudlab-gpu/4to8-v3`, `4to8-worst-v2`,
   `4to8-worst-long-v2`; figures `--series 4to8-v3 --calibrations 4to8 --before 4to8-v2`,
   `--series 4to8-worst-v2 --calibrations 4to8-worst --before 4to8-worst --drained --lazy-series
   4to8-worst-long-v2` and `--series 4to8-worst-long-v2 --calibrations 4to8-worst --before
   4to8-worst-long`):
   - **Three runs met the hosts' weekly TRIM** (Ubuntu's `fstrim.timer`: Monday 00:00 and up to
     100 min later, each host at its own time). These were `4to8-v3/rep1/copy-then-flip` (n4's
     host for 8.6 min from the trigger on) and both `4to8-worst-long-v2` runs (n3's host from
     251 s; n2's from 196 s and n1's from 553 s).
     - A TRIM on these SATA SSDs is not queued. Each read took 54-71 µs of the disk's time
       instead of 14-18, and the node read 13-16K pages/s instead of 39K.
     - Every query that touched the node waited: 155 q/s instead of 588, and 241-246K queries
       still queued at the load's end. 9,706 and 107,166 queries were abandoned, and 232 others
       hit the client's 30 s timeout.
     - They ran again at 05:53-06:32, after every host's TRIM of the week, each initial node's
       disk at 15-16 µs a read (the first runs are in `results/cloudlab-gpu-fstrim`).
       `run_cluster.py` and `run_one.sh` stop the timer since.
   - **4 → 8, lazy:**
     - Staging takes 2.3 s (8.4-8.5 s): the lists from memory, then the PQ codes, then the 8
       loads at once. Lazy flips 2.5 s after the trigger (9.5-9.6 s).
     - Each graph starts as its node's staging ends and moves its 7 GB in 40-51 s (43-67 s). The
       new entries come online 42-53 s after the trigger (70-76 s).
     - The backlog peaks at 7.2-7.4K (8.6-8.9K) and clears at 122-124 s (132 s). From the
       trigger on, 10.3-10.7% of the queries miss 200 ms (12.8-12.9%), and the mean latency is
       0.66-0.71 s (1.12-1.17 s).
     - Then it answers 780 q/s at 13.9-14.0 ms, as 8 warm nodes do.
   - **4 → 8 → 4:**
     - The scale-out flips at 2.4 s and ends at 50.6 s (10.1 s, 78.7 s).
     - The scale-in flips 1.7 s after its trigger and ends at 2.5 s (6.6 s, 7.8 s), with no
       partition saved as a file.
     - The queries do not notice the scale-in: 12.8 ms, p99 25 ms in the 40 s after its
       trigger (22.9 ms, p99 233 ms).
   - **U11's worst case, lazy:** it flips at 2.4-2.7 s (9.9-10.2 s), and the new entries come
     online 48-55 s after the trigger (314-328 s, and 516-523 s in the 570 s runs). The data
     path still sets the pace: the initial nodes' read budgets (n1's readers wait 36 s/s after
     the flip) and the new nodes' writes, 23-33 MB/s each. So the cluster answers about as
     before:
     - in the 360 s runs the backlog peaks at 26K and 36K (34-37K). Repetition 1 clears it at
       346 s, while repetition 2 still has 10.5K at the load's end (8-12K). 297-308K queries are
       answered from the trigger to the load's end (296-299K);
     - held to 570 s, it peaks at 31K and 44K (34-37K) and clears at 374 s and 440 s
       (394-412 s). Then lazy answers 1,100 q/s at 14.2-14.4 ms, p99 30-31 ms (13.6-13.8 ms,
       p99 30-31 ms), still fetching 13 MB/s. By the load's end it has fetched 31.5 GB, 73% of the
       copy, as before. From the trigger on, 47% and 61% of the queries miss 200 ms (62-65%).

     When the new entries come online, the mean latency of the answered queries falls from about
     21 s to 14-16 s for a while, since their queues start empty (U17).
   - **The repetitions differ where a new node's fetch cache reaches its bound, and then the whole
     cluster stops for seconds** (every worst-case lazy run, old code and new; never in the
     normal 4 → 8 runs or on Amazon):
     - A new owner of hot partitions fetches 40-67 MB/s in the first 40 s after the flip. Its
       SATA SSD writes 23-33 MB/s, a little differently in each run, so its 2 GB cache fills at
       150-250 s.
     - Past the bound a query installs what it fetched itself (`Cache`). But `RawStore::Put`
       holds `put_mu_` for a whole call, and the writer installs up to 65,536 vectors (256 MB)
       a call, 10-12 s of writes. So the query waits for the writer's round, not for its own
       pages, and the cluster answers nothing for 2-4 s, until the round ends. The cache then
       refills in 6-9 s.
     - In `4to8-worst-long-v2/rep2`, n6 wrote 22.7 MB/s (26.6-32.1 in the other runs). Its
       cache hit the bound every 12 s from 153 s to 249 s, the cluster answered fewer than 100
       queries in 15 of those seconds, and queries waited 1,871 s in all to install 19,783
       vectors. In `4to8-worst-v2/rep2` the same happened on n5 (14,445 vectors).
     - Every second after a flip with fewer than 100 answers, in all eight runs, falls while a
       new node's cache is at its bound.
   - **Copy-then-flip is unchanged.** 4 → 8 flips at 503-507 s (497-526 s), and with the stream
     first at 213-223 s (213-220 s). In the worst case it flips at 621-646 s (614-627 s), and
     with the stream first at 214-223 s (214-222 s).
   - No query fails. The queries with ground truth answered in all runs have the same recall in
     each group: 257,438 in the sixteen 4 → 8 runs (4to8, v2, v3), 272,893 in 4to8to4's three,
     1,211 in the twelve worst-case runs, and 1,980 in the four 570 s runs.
3. Server experiments: copy-then-flip with the new raw stream at the other budgets and loads;
   repeats near capacity; a CCD per node before calibrating the CPU-bound regime; a full-replica
   baseline to argue for partitioning (each node's PQ budget is now its share by default).
4. Full rebuild and updates, measured: affinity against random placement, warm-up on and off,
   stop-the-world; updates from the difference of two Wikipedia snapshots. The rebuild's
   figures still need an in-place baseline that holds both generations' graph and lists on
   every node (or memory accounting in its place: `timeline.svg` draws each node's memory, and
   compare_runs.py counts a rebuild's transfers, 2026-10-28). Not implemented: the rebuild's probe-set
   trigger, a generation that includes inserted vectors, a priority between the warm-up's
   fetches and the spares'.
5. More data and scale: GP on real datasets (U2: the number of partitions), non-repeating real
   query streams (MS MARCO), BIGANN-100M, SPACEV-100M.
6. Skew mitigation, the last experiment: once the workload is final, each partition's load is
   known from it, and a custom plan built from it runs at a fixed node count (`rtier-client
   rescale -plan`). We show that any plan runs without disruption and with exact answers; the
   skew policy is not part of our design.
7. Small code items (2026-10-28: entries pre-connect to aggregators, a heap merge in `MergeTopK`,
   pulls share fetches with the queries -- done): the stream through the agents (`RawPath` agent)
   still claims nothing (`engine/README.md`, known limits).
8. Open design questions (the registry has them all). Decided on 2026-10-20: two-phase stays
   (U5), no aggregator outsourcing (U6), raw vectors kept in full (U1), the graph transfer stays
   eager (U16), nodes removed by a scale-in are shut down (U3), U7 closed with the failures, the
   pacing stays as it is (U8), the entry list stays in the epoch table (U17), no workload
   generator of our own (U11: real queries that do not repeat, and the worst case). Decided on
   2026-11-08: the graph yields to the staging's posting lists and PQ codes, not to raw vectors,
   with no rate of its own (U16: on Amazon 300M, lazy's fetches, even at 5-7 MB/s, held the new
   entries until the load ended, and 4to8to4's scale-in after them; in the rerun the entries come
   59-71 s after the trigger, and the scale-in runs under the load). Still open:
   - on the protocol: placement by per-tier bytes (U3);
   - on updates (U14, to be discussed later): the update interface (a log, LSNs, visibility;
     batch size and period are set aside); a batch over budget (scale out first?); writes
     against queries during a scale-out; staged versions left by an expired lease;
   - scale limits for billion-vector datasets, the target: single-frame graph patches (the
     listings `PQ_MISSING`, `RAW_MISSING`, `RAW_HELD_BY` stream since 2026-10-21), what
     `run_cluster.py` still copies whole (every segment to each initial node's host, and the
     page file and the PQ codes where there is no warmup snapshot), per-node
     structures sized by the dataset (`slot_of_`, the presence bitmap);
   - elsewhere: chunk storage (U12, last), the host copy of the PQ codes (U15), the number of
     partitions (U2), the PQ compression level.

## Working with the user

The user writes in Chinese or English; answer in the language of the message. Code, comments,
commit messages and GitHub issues are in English; the design doc is in Chinese. The user
commits: Claude writes a commit message and the list of files to stage (`results/<topic>/`),
and a doc change goes in the same commit as the code it describes. Changes under `docs/` are not
committed.
