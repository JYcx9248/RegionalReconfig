# rtier — notes for Claude Code

rtier is a prototype of non-disruptive reconfiguration (scale-out / scale-in) for the regional
tier of a distributed, FusionANNS-style vector search system. A Go control plane (structure
reused from Koala) drives C++ data nodes built on our FusionANNS reimplementation (`engine/`).
The goal of the research: add or remove nodes under query load with exact answers throughout,
a short flip, and throughput that follows the load.

## Build and test

Linux only (the end-to-end tests and the network emulation need it). CMake ≥ 3.18, a C++17
compiler with OpenMP, Go ≥ 1.24, Python 3 with numpy and PyYAML. CUDA ≥ 11 and liburing are
optional.

```sh
make CUDA=OFF            # engine in engine/build + bin/rtier-*
make test CUDA=OFF       # fusion_tests, rtier_node_tests, Go tests with -race, end-to-end tests
make e2e                 # end-to-end tests only, verbose
make fmt && make vet
RTIER_ENGINE_BIN=$PWD/engine/build go test -race ./...   # go test directly: e2e needs the engine
```

- The Makefile exports `GOTOOLCHAIN=local`: never download a Go toolchain.
- Experiments run `bin/`, which `go test` does not rebuild: `make test` builds it too, and
  `run_local.py` and `run_cluster.py` refuse to start when `bin/` is older than the Go sources
  (run `make`).
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
  deployment run on a copy of the index without its page file (`indexNoPages`): they can only
  get raw vectors from peers.
- Experiment configurations are YAML, a base and overrides (`configs/base/*.yaml`,
  `configs/experiment.*.yaml`); `python3 scripts/experiment_config.py CONFIG.yaml` prints one
  merged, and every key is checked against its schema. `run_local.py` runs every node on one
  machine; `run_cluster.py` runs them on the hosts of a `hosts:` section, over SSH.
  `PATH=$PWD/scripts/testing/fake_ssh:$PATH` tries `run_cluster.py` on one machine without an
  SSH server.

## Layout

| Path | What |
|---|---|
| `cmd/rtier-controller`, `cmd/rtier-agent` | control plane; per-node agent next to `rtier_node` |
| `cmd/rtier-client`, `cmd/rtier-loadgen` | status / rescale / rebuild / design list; open-loop or closed-loop load |
| `cmd/rtier-updater` | the update controller's command: one batch of inserts and deletes, or a major epoch (U14) |
| `cmd/rtier-overhead` | fan-out cost analysis: one data node emulates N owners (`-owners`, `-compare`) |
| `internal/epoch` | epoch table, CAS store, `Tracker` (pins, grace period) |
| `internal/controller` | registration, placement, reconfiguration (`reconfig.go`), update leases (`update.go`), full rebuild (`rebuild.go`); collects metrics and the nodes' diagnostics |
| `internal/agent` | staging, epoch install, entry and aggregator roles (`serve.go`), generations and the rebuild's warm-up (`generation.go`), diagnostics forwarding (`diag.go`) |
| `internal/update` | update batches (`batch.go`) and major epochs (`major.go`), with their router |
| `internal/query` | selectors (aggregator choice), strategies (`TwoPhase`), wire format |
| `internal/placement`, `internal/transfer` | placement policies, plans, the rebuild's placement (`affinity.go`); bulk transfer + token bucket |
| `internal/design` | registry of open design questions (U1–U17) |
| `engine/` | FusionANNS engine, `NodeEngine`, `rtier_node`, `rtier_segment` (see `engine/README.md`) |
| `test/e2e` | end-to-end tests |
| `configs/` | experiment configurations (YAML: bases and experiments) |
| `scripts/` | `run_local.py` and `run_cluster.py` experiments, `experiment_config.py` (the YAML format), `plot_run.py`, `compare_runs.py` (mean latency per run, overlaid), `emulation/netns.sh`, `testing/` (placeholders, the GP partitioner, `fake_ssh`) |

## Conventions

- Go uses only the standard library. The module path `rtier` is a placeholder.
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
- **Reconfiguration** (`copyThenFlip`, protocol `lazy` by default): pre-connect aggregators →
  new nodes pull segments and the PQ codes they lack (graph in the background, lower priority,
  sources round-robin over entries) → flip (CAS + install) → grace (drain the old epoch) →
  reclaim (old owners drop moved posting lists; PQ codes and raw vectors stay cached).
  Alongside grace and reclaim, `promoteEntries` flips every `entry_flip_interval` to make
  entries of the new nodes whose graph has loaded; the rescale waits for it. Raw vectors do
  not move with a scale-out: `lazy` fetches one only when a query needs it (below). Baselines:
  `lazy-stream` also streams the rest in after the flip (`StageRaw`; the rescale waits for it),
  `copy-then-flip` and `stop-and-copy` stream them before the flip (eager). A rescale to a plan
  (`rtier-client rescale -plan`) carries out any target placement. The token bucket paces only
  what goes through the agents -- segments and PQ codes (data class) and the graph (background);
  raw vectors move data node to data node (`RAW_GET`, `RAW_PULL`), unpaced, unless
  `raw_path: agent` (then at `raw_priority`, U8).
- **PQ codes** are stored per node, once per vector, with a cache of codes of partitions that
  moved away (`engine/include/fusion/pq_store.h`); placement `even-reversible-graph` (the
  default) returns partitions to former owners so a scale-in moves no PQ code.
- **Raw vectors**: every posting carries its vector's canonical location in the index's page
  file and the node that holds it (segment v3); a node keeps a sparse local copy of that file
  with a presence bit per location (`engine/include/fusion/raw_store.h`) and looks locations up
  through its PQ store. After a migration, RERANK fetches a missing vector when a query first
  needs it, and only then, from the holder that a posting of the query's lists on that node
  records: one hop, data node to data node (`RAW_GET`, `PeerPool`; nodes know each other by ID,
  `SET_NODES`), and that posting then names the fetching node. Concurrent misses of one vector
  wait for one fetch. Fetched vectors are re-ranked from memory (fetch cache) until a background
  writer installs them with whole-page `O_DIRECT` writes (`WriterLoop`, `RawStore::PutPages`);
  INFO never waits for it (`nodeclient.InfoSettled` does, for tests). A scale-in drains the
  leaving nodes before it returns (`drainRaw`: each remaining node takes over the vectors only a
  leaving node holds, `RAW_HELD_BY` then `RAW_PULL`), so they can go offline. No node drops a
  raw vector it has held. Only bootstrap reads the index's page file.
- **Updates (U14)**: the update controller (`internal/update`, `rtier-updater`) takes the
  controller's lease and runs a batch as a content epoch: inserts routed by its own router
  (`ASSIGN`, the build's rule), PQ codes and raw vectors sent to the owners, the next version of
  each changed partition staged next to the current one (`STAGE_VERSIONS`), then commit = flip,
  grace, reclaim (deleted vectors' codes are freed, never cached). A major epoch splits and
  merges posting lists and ships the graph change as a patch every entry stages, searches
  through after the flip and merges after the grace period. Only with the controller's
  `allow_updates` (`updates:` in an experiment configuration).
- **Full rebuild (U14, `rebuild.go`)**: moves the cluster to the next index generation, built
  outside with the same codebook, vector IDs and raw-vector locations. Spares take every
  partition by a lazy reconfiguration; the old nodes lose their roles and switch generations
  (`SWITCH_GENERATION`, keeping PQ codes and raw vectors); the new partitions go where their raw
  vectors are (presence bitmaps, `placement.Affinity`); the old nodes load them, replay recent
  queries to warm up, and take over in one flip; the spares are released. Manual trigger only
  (`rtier-client rebuild`, `reconfigurations: [{rebuild: ...}]`); baselines: random placement,
  no warm-up, stop-the-world.
- **Clients' entry lists (U17)**: a client (`rtier-loadgen`) reads the entries from the
  controller every 500 ms, and at once when an entry refuses a query under a newer epoch than
  its list's (the queries that meet the stale list share one read); a refusal under the list's
  own epoch (paused admission) reads nothing.
- **Diagnosing a run**: `metrics.jsonl`, `stats-<node>.jsonl` (per-node counters,
  `rtier_node --stats-file`) and `cgroups.jsonl` (CPU, disk, dirty pages, pressure; a host
  record with the whole disk's I/O and how busy each node's CPUs were). The agents forward the
  stats logs (`node_stats_file`) and, on a cluster, their samples (`sample_interval`;
  `run_local.py` samples itself with `cluster.resources`), and the controller stamps every
  record and metrics report on arrival (`"t"`, on its monotonic clock; the node's own as
  `"node_t"`): one clock with `clock.json` and the load generator, without synchronizing the
  machines' clocks (Koala's metric collector).
- **Without root** (shared servers): `cluster.resources` pins CPUs with `taskset` and limits
  memory in user scopes (`systemd-run --user`); per-node read IOPS come from the data node's own
  read budget (`read_iops_per_node` → `rtier_node --read-iops`, `fusion/read_budget.h`), since
  the cgroup io controller is not delegated; `run_one.sh` evicts only the run's own files
  (`scripts/evict_cache.py`).
- **Cluster runs** (`run_cluster.py`, as Koala deploys): this machine runs the stages, the
  controller and the load generator; it checks every host, copies the binaries, the index (its
  page file only where an initial node bootstraps), the partitions and any next generation with
  rsync to the same paths, and starts the nodes in order over SSH, each after the previous one
  registered. A remote process stops when its SSH session closes. `cluster.resources` (except
  `read_iops_per_node`) and `cluster.emulation` are for one machine and are refused.

## Docs

- `README.md`: overview, the decided part of each open question, how to run.
- `docs/design.md`: the design doc (Chinese), exported from claude.ai at rev 123 and brought up
  to date here since (last on 2026-10-20); not synced with the claude.ai copy.
- `docs/handoff.md`: a snapshot of 2026-09-27, context from the conversations that built rtier
  -- GitHub issue drafts and diagrams; its section 5 is the case analysis of queries during a
  reconfiguration.
- `docs/drafts/`: discussion notes and issue drafts (dataset updates, the SSD read model,
  correctness evidence, placement). `docs/server-runbook.md`: runs on the lab server.
- `engine/README.md`: engine additions and known limits.

## Current state and next steps (2026-10-20)

Done since late September:
- Raw vectors: a holder per posting instead of a source per list, one-hop fetches, a scale-in
  drains the nodes that leave (U9). Concurrent misses of one vector share a fetch.
- Dataset updates (U14): minor batches of inserts and deletes as content epochs, major epochs
  (splits and merges, graph patches), and the full rebuild to a next index generation.
- Partitioning (U2): GP on the list heads over the navigation graph, balancing list counts.
  The initial placement stays contiguous ranges (U3).
- YAML experiment configurations; cluster deployment (`run_cluster.py`); clients catch up
  with a stale entry list at once (U17).
- Reference runs: `results/bigann10m-spatial-scaleout-v2` (BIGANN-10M, laptop); Wikipedia
  c1024 runs on the lab server (`docs/design.md`, "实验结果").

Next:
1. Issue: how queries of the old epoch are handled during a reconfiguration — cases, current
   behavior, tests. Accepted (2026-10-16): the grace period waits for every old-epoch query, so
   a slow one holds it; a stale entry list costs one refusal, after which the client reads the
   list (U17); lazy's fetches stay on the query path. Open edge (U7, a placeholder): an entry
   whose client gave up releases its pin while its forward may still be on the way, and a
   forward that reaches its aggregator after that node's drain returned meets a reclaimed
   partition. Proposed, to be confirmed: reject it explicitly, or keep the entry's pin until the
   forward has ended. Draft of the case analysis in `docs/handoff.md`, section 5.
2. A real cluster: `run_cluster.py` has run only through `fake_ssh` on one machine. Then
   scaling curves on a multi-node GPU testbed.
3. Server experiments: copy-then-flip with the new raw stream at the other budgets and loads;
   repeats near capacity; a CCD per node before calibrating the CPU-bound regime; a full-replica
   baseline, or `--pq-capacity` for a smaller HBM, to argue for partitioning.
4. Full rebuild and updates, measured: affinity against random placement, warm-up on and off,
   stop-the-world; updates from the difference of two Wikipedia snapshots. The rebuild's
   figures still need an in-place baseline that holds both generations' graph and lists on
   every node (or memory accounting in its place), compare_runs.py's transfer breakdown for
   rebuilds, and per-node memory curves in the plots. Not implemented: the rebuild's probe-set
   trigger, a generation that includes inserted vectors, a priority between the warm-up's
   fetches and the spares'.
5. More data and scale: GP on real datasets (U2: the number of partitions), non-repeating real
   query streams (MS MARCO), BIGANN-100M, SPACEV-100M.
6. Skew mitigation, the last experiment: once the workload is final, each partition's load is
   known from it, and a custom plan built from it runs at a fixed node count (`rtier-client
   rescale -plan`). We show that any plan runs without disruption and with exact answers; the
   skew policy is not part of our design.
7. Small code items: fill the whole connection pool during pre-connect and let entries
   pre-connect to aggregators; heap merge in `MergeTopK`; raw path (see `engine/README.md`,
   known limits): pulls (stream, drain) do not join the queries' fetches.
8. Open design questions (the registry has them all):
   - on the protocol: U7 (item 1); the entry list outside the epoch table (U17); rate adaptation
     and whether to pace the drain (U8); placement by per-tier bytes, and which nodes leave or
     join a rescale by node count (U3); the graph's transfer mode, eager or stream (U16);
   - on updates (U14): the update interface (a log, LSNs, visibility; batch size and period are
     set aside); a batch over budget (scale out first?); writes against queries during a
     scale-out; freeing deleted vectors' pages; staged versions left by an expired lease;
   - elsewhere: rerank policy (U5), aggregator outsourcing (U6), chunk storage (U12), a disk
     budget for raw vectors (U1), the host copy of the PQ codes (U15), the number of partitions
     (U2), the PQ compression level.

## Working with the user

The user writes in Chinese or English; answer in the language of the message. Code, comments,
commit messages and GitHub issues are in English; the design doc is in Chinese. The user
commits: Claude writes a commit message and the list of files to stage (`results/<topic>/`),
and a doc change goes in the same commit as the code it describes. Changes under `docs/` are not
committed.
