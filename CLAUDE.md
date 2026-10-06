# rtier — notes for Claude Code

rtier is a prototype of non-disruptive reconfiguration (scale-out / scale-in) for the regional
tier of a distributed, FusionANNS-style vector search system. A Go control plane (structure
reused from Koala) drives C++ data nodes built on our FusionANNS reimplementation (`engine/`).
The goal of the research: add or remove nodes under query load with exact answers throughout,
a short flip, and throughput that follows the load.

## Build and test

Linux only (the end-to-end tests and the network emulation need it). CMake ≥ 3.18, a C++17
compiler with OpenMP, Go ≥ 1.24, Python 3 with numpy. CUDA ≥ 11 and liburing are optional.

```sh
make CUDA=OFF            # engine in engine/build + bin/rtier-*
make test CUDA=OFF       # fusion_tests, rtier_node_tests, Go tests with -race, end-to-end tests
make e2e                 # end-to-end tests only, verbose
make fmt && make vet
RTIER_ENGINE_BIN=$PWD/engine/build go test -race ./...   # go test directly: e2e needs the engine
```

- The Makefile exports `GOTOOLCHAIN=local`: never download a Go toolchain.
- Experiments run `bin/`, which `go test` does not rebuild: `make test` builds it too, and
  `run_local.py` refuses to start when `bin/` is older than the Go sources (run `make`).
- `-march=native` is on by default (`FUSION_NATIVE`). Rebuild the engine on each machine: a build
  copied from another CPU can die with SIGILL. Use `-DFUSION_NATIVE=OFF` for a portable build.
- Sanitizers: build the engine with `-fsanitize=address,undefined` into another directory and
  point `RTIER_ENGINE_BIN` at it; the end-to-end tests fail on any sanitizer report.
- The end-to-end tests start real `rtier_node` processes on a small synthetic fixture
  (8000 vectors, 8 partitions) and take about 10 s. `RTIER_E2E_TIMELINE=<dir>` makes
  `TestReconfigUnderLoad` write each protocol's per-query latency and rescale times for
  `scripts/compare_runs.py` (README, "Mean latency over time"); measure without `-race`.
  `RTIER_E2E_BACKEND=gpu` (with a CUDA build in `RTIER_ENGINE_BIN`) runs the cluster's nodes on
  the GPU backend against the CPU oracle. Nodes that join after the initial
  deployment run on a copy of the index without its page file (`indexNoPages`): they can only
  get raw vectors from peers.

## Layout

| Path | What |
|---|---|
| `cmd/rtier-controller`, `cmd/rtier-agent` | control plane; per-node agent next to `rtier_node` |
| `cmd/rtier-client`, `cmd/rtier-loadgen` | status / rescale / design list; open-loop load |
| `cmd/rtier-overhead` | fan-out cost analysis: one data node emulates N owners (`-owners`, `-compare`) |
| `internal/epoch` | epoch table, CAS store, `Tracker` (pins, grace period) |
| `internal/controller` | registration, placement, reconfiguration (`reconfig.go`) |
| `internal/agent` | staging, epoch install, entry and aggregator roles (`serve.go`) |
| `internal/query` | selectors (aggregator choice), strategies (`TwoPhase`), wire format |
| `internal/placement`, `internal/transfer` | placement policies; bulk transfer + token bucket |
| `internal/design` | registry of open design questions (U1–U17) |
| `engine/` | FusionANNS engine, `NodeEngine`, `rtier_node`, `rtier_segment` (see `engine/README.md`) |
| `test/e2e` | end-to-end tests |
| `scripts/` | `run_local.py` experiments, `plot_run.py`, `compare_runs.py` (mean latency per run, overlaid), `emulation/netns.sh`, test placeholders |

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

- **Epoch** = versioned routing table: partition owners, entries, aggregators, addresses. A
  query runs under the one epoch its entry pinned; a forwarded aggregation carries that epoch
  and the lists grouped by owner, and the aggregator neither pins nor checks an epoch of its own.
- **Query**: entry NAVIGATEs → groups the lists by owner → the `owner` selector picks the
  aggregator → `TwoPhase`: FILTER (local top-n by PQ) at each owner, global top-n merge, RERANK
  of each candidate at one owner that reported it, top-k merge. Equal to the single-node answer.
- **Reconfiguration** (`copyThenFlip`, protocol `lazy` by default): pre-connect aggregators →
  new nodes pull segments and the PQ codes they lack (graph in the background, lower priority,
  sources round-robin over entries) → flip (CAS + install) → grace (drain the old epoch) →
  reclaim (old owners drop moved posting lists; PQ codes and raw vectors stay cached).
  Alongside grace and reclaim, `promoteEntries` flips every `entry_flip_interval` to make
  entries of the new nodes whose graph has loaded; the rescale waits for it. Raw vectors do
  not move with the rescale: `lazy` fetches one only when a query needs it (below). Baselines:
  `lazy-stream` also streams the rest in after the flip (`StageRaw`, `raw_priority`; the rescale
  waits for it), `copy-then-flip` and `stop-and-copy` stream them before the flip (eager).
- **PQ codes** are stored per node, once per vector, with a cache of codes of partitions that
  moved away (`engine/include/fusion/pq_store.h`); placement `even-reversible` returns
  partitions to former owners so a scale-in moves no PQ code (and no raw vector).
- **Raw vectors**: every posting carries its vector's canonical location in the index's page
  file (segment v2); a node keeps a sparse local copy of that file with a presence bit per
  location (`engine/include/fusion/raw_store.h`) and looks locations up through its PQ store.
  After a migration, RERANK fetches a missing vector from the old owner's data node when a
  query first needs it, and only then: each posting list has a source (the old owner of its
  partition), RERANK requests carry the query's lists on that owner, and RAW_GET names the list
  of every vector so a source that lacks one fetches it from its own source (chains)
  (`EnsureRawInLists`, `FetchRaw`, `PeerPool`). Fetched vectors are re-ranked from memory
  (fetch cache) until a background writer installs them with whole-page `O_DIRECT` writes
  (`WriterLoop`, `RawStore::PutPages`); INFO never waits for it (`nodeclient.InfoSettled` does,
  for tests). What no query needs stays with the old owner, which keeps every vector it has
  held. Only bootstrap reads the index's page file.
- **Diagnosing a run**: `stats-<node>.jsonl` (per-node counters, `rtier_node --stats-file`) and,
  with `"NodeResources"`, `cgroups.jsonl` (CPU, disk, dirty pages, pressure; a `"host"` record
  with the whole disk's I/O and how busy each node's CPUs were), both on the monotonic clock of
  `clock.json`.
- **Without root** (shared servers): `"NodeResources"` pins CPUs with `taskset` and limits memory
  in user scopes (`systemd-run --user`); per-node read IOPS come from the data node's own read
  budget (`"ReadIOPSPerNode"` → `rtier_node --read-iops`, `fusion/read_budget.h`), since the
  cgroup io controller is not delegated; `run_one.sh` evicts only the run's own files
  (`scripts/evict_cache.py`).

## Docs

- `README.md`: overview, the decided part of each open question, how to run.
- `docs/design.md`: the design doc (Chinese), exported from claude.ai at rev 123 and updated
  here for the lazy raw-vector fetch; not synced with the claude.ai copy.
- `docs/handoff.md`: context from the conversations that built rtier -- GitHub issue drafts and
  diagrams, the open discussion on queries during a reconfiguration, next steps.
- `engine/README.md`: engine additions and known limits.

## Current state and next steps (late Sep 2026)

Recently done: owner-based aggregator selection (U6); one epoch per query (fixes the flip races);
new nodes become entries in batched flips as their graphs load; graph transfer at background
priority from round-robin sources (U16); no global replica of the raw vectors: canonical
locations in the segments (U1) and lazy fetch after the flip as the default protocol (U9) --
since 2026-09-28 on demand only, with sources per posting list (the full background stream is
the `lazy-stream` baseline); since 2026-09-29 fetched vectors are installed by a background
writer with whole-page direct writes (the old buffered writer held lazy below capacity for
minutes after the warm-up), INFO never waits for it, and every run writes per-node stats logs
(reference runs: `results/bigann10m-spatial-scaleout-v2`).

Next:
1. Issue: how queries of the old epoch are handled during a reconfiguration — cases, current
   behavior, tests, open edges (entry timing out while a forwarded aggregation still runs; slow
   queries holding the grace period; stale entry lists → UNAVAILABLE and retries; the warm-up
   cost of lazy fetch). Draft of the case analysis in `docs/handoff.md`, section 5.
2. Real-data pipeline (BEIR: SciFact to build it, then Quora): embed with a cosine-similarity
   model and normalize (the engine is L2 only); float32 ≤ 1024 dims (a vector must fit a 4 KB
   page); exact kNN ground truth.
3. Local experiments: data-dependent counts with `rtier-overhead` (owners per query, PQ
   redundancy, bytes and heat per partition, recall vs nprobe and n, PQ code size via
   `fusion_build --pq-m`); protocol comparisons on one machine. Nodes on one machine share its
   SSD, which caps the cluster: give each node its own CPUs and read IOPS with
   `"NodeResources"` in `run_local.py` (README) before reading anything into throughput.
   Scaling curves need a multi-node GPU testbed.
4. Small code items: fill the whole connection pool during pre-connect and let entries
   pre-connect to aggregators; heap merge in `MergeTopK`; raw path (see `engine/README.md`,
   known limits): compact pending-source map, coalescing concurrent fetches of one vector.
5. Design topics under discussion: partitioning (U2/U3), chunk storage (U12), aggregator
   outsourcing (U6), rerank policy (U5), PQ compression level, warm-up alternatives for raw
   vectors (re-rank at the old owner, stream before the flip; U9).

## Working with the user

The user writes in Chinese or English; answer in the language of the message. Code, comments,
commit messages and GitHub issues are in English; the design doc is in Chinese.
