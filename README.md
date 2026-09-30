# rtier — regional-tier reconfiguration prototype

Skeleton for non-disruptive reconfiguration of the regional tier of a distributed,
FusionANNS-style vector search system. A Go control plane (reusing Koala's structure) drives
C++ data nodes built on the FusionANNS reimplementation.

**Status:** skeleton. The settled parts of the design are implemented and covered by tests.
Open design questions (U1–U17, without U4) are placeholders: on a code path they fail with
`design decision pending [Un ...]`; the few without a code path yet (U12, U14, U15, U17) are
recorded in the registry with what the code does meanwhile. U4 (replicas of hot partitions) was
ruled out of scope -- elastic read capacity for a hot key is the serverless-instance pattern of
streaming systems, and rtier keeps one owner per partition. List them with `bin/rtier-client design`
(source: `internal/design/design.go`).

**PQ codes (decided part of U1):** node-level. A data node keeps one PQ code per vector named by
its resident posting lists, however many lists or partitions name it. When partitions move, the
destination's data node lists the codes it lacks, split by source so each is fetched once
(`PQ_MISSING`); the old owners send exactly those over the rate-limited bulk channel
(`PULL_PQ`); they are installed (`PQ_PUT`) before the partitions load, so a new node goes online
only once it holds every posting list and PQ code of its partitions and can filter at once. The
dataset is static, so the old owner keeps the codes of partitions that moved away as a cache;
they are dropped only when a later staging needs the room, and a staging protects the codes it
needs. `rtier_node --pq-capacity N` sets a node's budget (its HBM, in codes). In fact nothing is
ever dropped: a scale-out only takes partitions away and a reversible scale-in only gives back
what it took, so a node's resident set stays what it held at the initial cluster size -- which
is why the budget is dimensioned by the *smallest* cluster size, not the current one. The
end-to-end tests assert it (no eviction); it stops holding if the cluster shrinks below its
initial size or placement starts moving partitions to nodes that never held them. The initial
deployment reads codes from the index. Known limits of this path are listed in
`engine/README.md`.

**Raw vectors (the other decided part of U1):** addressed by a *canonical global offset*. The
location -- `page * vectors_per_page + slot` in the index's page file, a `u32` next to the vector
ID, 8 B per posting entry -- is written into the segments by `rtier_segment` (payload
`lists+locations`) and travels with the posting list, so it survives migrations untouched and no
node needs a global vector -> page map. Each data node keeps the vectors it holds in its own
sparse file laid out like the whole page file (`rtier_node --raw-file`), with a presence bit per
location, and finds a candidate's location through its PQ store. The same vector reached through
two lists names the same offset, so dedup is free, and the transfer unit is a single vector (list
tails share pages). Only the initial deployment reads the index's page file; the end-to-end tests
run every node that joins later on a copy of the index without it. Old owners keep the vectors of
partitions that moved away, like PQ codes. Still open: tail-page space amplification, a disk
budget, versioning under U14, and where RAG chunks live. Partition directories written before
locations existed (payload `lists-only`) are refused: rerun `rtier_segment`.

**Raw vectors after a migration (decided part of U9):** a new owner goes online with posting
lists and PQ codes only -- all it needs to filter -- and gets a raw vector only when a query first
needs it (`"protocol": "lazy"`, the default): the point of being lazy is that queries touch a
small working set of what moved, and the rest never has to. From the flip on, a RERANK that meets
a vector not there fetches it from the partition's old owner, data node to data node, and waits
for it. The data node keeps one source per posting list -- the old owner of the list's partition
-- so any list that names a missing vector says where to fetch it: lists overlap, no per-vector
table. RERANK requests carry the query's lists on that owner for this, and a peer's RAW_GET
names the list of every vector, so a node asked for one it lacks fetches it from its own source
of that list first (chains of migrations). Nothing else moves and the rescale does not wait for
raw vectors; `raw_fetched` in the rescale reply counts what queries fetched while it ran, and
each node reports `raw.fetched`, `raw.pending` (named by its partitions, not needed so far) and
`raw.present` in its metrics. That old owners stay sources rests on three invariants: no node
drops a raw vector it has held, the nodes that leave are the ones that joined last, and a
scale-in returns partitions to nodes that held them (U3); a disk budget or another leave order
would need a hand-off first. Baselines: `"lazy-stream"` also streams the rest in after the flip
(`StageRaw`, `"raw_priority": "background"` by default; the rescale waits for it and reports
`raw_vectors` / `raw_seconds`), `"copy-then-flip"` streams them before the flip, `"stop-and-copy"`
while paused.

**Placement (decided part of U3):** `"placement": "even-reversible"` (the default) sends a
partition back to a node that has owned it before when the cluster scales in, so scaling out and
back in restores the placement exactly and the scale-in moves *no* PQ code -- the returning
partitions name vectors their owner kept cached. Koala's policy (`"even"`, kept as the baseline)
does not: after 4 -> 8 -> 4 none of the partitions that moved is home again. It has no reason to
care, because its state changes with every input, so a copy left behind is stale; with a static
dataset the copy stays valid, which is what makes this worth doing.

**Graph transfer (decided part of U16):** a new node needs posting lists and PQ codes to own
partitions and answer FILTER and RERANK -- the steps that take load off the old owners -- and
the navigation graph only for the entry role, whose navigation any existing entry can do
meanwhile. So each new entry's graph source is assigned when the reconfiguration starts,
round-robin over the live entries (`graph_sources` in the rescale reply), and the pull runs at
background priority on its source (`"graph_priority": "background"`, the default): on a node
that is also sending segments or PQ codes, the graph only takes bandwidth those leave unused,
so it never delays the flip. `"data"` shares the bucket equally, as a baseline. A new node
becomes an entry as soon as its graph has loaded, not when the slowest one has: while graphs
are arriving, the controller flips every `"entry_flip_interval"` (1 s by default, about how
often clients refresh their entry list) to add all the nodes that are ready, and adds the last
ones at once when every pull has ended. These flips change only the entries, so they run
alongside the grace period of the data flip. A node whose graph fails to load stays a data
node and aggregator without holding up the others, and the rescale returns an error naming it.

**Aggregator (decided part of U6):** the entry navigates first, then works out which nodes own
the probed lists, and only then picks the aggregator among them (`"selector": "owner"`, the
default): itself if it is one of the owners; the owner if a single node holds every list -- the
query is forwarded once and both phases run where the data is, one network round trip instead of
two; itself otherwise, since forwarding would add a hop to save one of several parallel
sub-queries and pile aggregation onto the owners of hot partitions. `"local"` always aggregates at
the entry (the baseline). `"warmup"` hands the scatter/gather to a node that may aggregate but is
not an entry yet -- during a scale-out, a new node whose navigation graph is still arriving.
`rtier-overhead -owners` reports how often each case occurs for a dataset and partitioning.
A query runs under one epoch, the one its entry pinned: a forwarded query carries that epoch and
its lists grouped by owner, and the aggregator runs it as is -- it neither pins nor checks an
epoch of its own, since the entry keeps the pin until the answer is back. Epochs are installed
on the nodes in parallel, so this is what lets a joining node aggregate for an entry that
installed the new epoch before it did, and a leaving node finish the queries it was handed just
before the flip (`test/e2e/flip_race_test.go` holds an install or a query to reproduce both).

## Layout

| Path | What |
|---|---|
| `cmd/rtier-controller` | control plane: registration, placement, epochs, reconfiguration, metrics sink |
| `cmd/rtier-agent` | per-node agent next to `rtier_node`: staging, epochs, entry + aggregator roles |
| `cmd/rtier-client` | status / rescale / design / queries / single-node baseline |
| `cmd/rtier-loadgen` | open-loop load (latency measured from the scheduled send time), with a rate schedule |
| `cmd/rtier-overhead` | cost of fan-out: emulates N owners on one data node holding every partition and reports what the two-phase query adds over N = 1 (owners per query, duplicate PQ scoring, extra page reads, per-call and merge time); `-compare` sets two-phase against local re-ranking (recall with `-gt`), `-owners` shows how queries spread over owners |
| `internal/` | ctrl RPC, protocol messages, placement, epoch store + tracker, bulk transfer, query aggregation, metrics, config, ... |
| `engine/` | FusionANNS engine + `NodeEngine` (partitioned posting lists, query primitives), `rtier_node`, `rtier_segment` |
| `test/e2e` | real `rtier_node` processes; scale-out and scale-in under query load |
| `scripts/` | `run_local.py` experiment runner (capacity ladder + timed rescales), `plot_run.py` timeline, `compare_runs.py` mean latency of several runs overlaid, `emulation/netns.sh`, metrics → SQLite, Python API client |

## Build and test

Needs Linux, CMake ≥ 3.18, a C++17 compiler with OpenMP, Go ≥ 1.24 (standard library only),
Python 3 for the scripts. CUDA ≥ 11 and liburing are optional.

```sh
make                 # engine + bin/rtier-*; CUDA=AUTO builds the GPU backend when nvcc is found
make CUDA=OFF        # CPU-only engine
make test CUDA=OFF   # engine unit tests, Go tests with -race, end-to-end tests
```

For a sanitizer check, build the engine with `-fsanitize=address,undefined` and point
`RTIER_ENGINE_BIN` at it: the end-to-end tests fail if a node's log contains a sanitizer
report.

## Run on one machine

```sh
python3 engine/scripts/make_synthetic.py --n 1000000 --nq 1000 --dim 128 --dtype uint8 --out data/syn
engine/build/fusion_build --base data/syn-base.u8bin --out /tmp/rtier-run/index
# list -> partition assignment: open question U2. TEST-ONLY placeholder:
python3 scripts/testing/make_range_assignment.py /tmp/rtier-run/index 64 data/assign.bin
python3 scripts/run_local.py configs/experiment.synthetic.json results/run1
```

`configs/experiment.example.json` is the same run on BigANN (edit the paths). With
`"Emulation": {"Netns": true}` (root), each node runs in its own network namespace with its
link shaped by `tc` (`scripts/emulation/netns.sh`; delay needs the `netem` qdisc).

On one machine every node reads the same SSD, and the two-phase RERANK reads ~150 pages per
query, so without more the cluster stops at what that disk serves and a scale-out adds no
throughput (measured: 2 and 3 nodes both at ~130-140K reads/s, ~920 q/s on BIGANN-1M). With
`"NodeResources": {"CPUsPerNode": 4, "IOReadIOPSMax": 40000}` each node and its agent run in a
cgroup of their own (`systemd-run --scope`): their own CPUs and their own read IOPS on the disk
of `WorkDir`, as if each node had its machine; keep the sum below what the disk serves. The
controller and the load generator get the CPUs after the nodes'. `"MemoryMax"` and
`"IOReadBandwidthMax"` work the same way.

Queries run through the two-phase strategy (`"Strategy": "two-phase"`, the implemented half of
U5): every owner filters for its own top-n, the aggregator merges the global top-n and re-ranks
each candidate at the owner that reported it, which is exactly the single-node answer.
`test/e2e` checks that equality query by query while the cluster scales out and in, and that
a vector new to a node arrives at most once: with `lazy` only when a query needs it, with the
baselines all of them, streamed or fetched on demand.

`"Protocol"` in an experiment config picks the reconfiguration protocol (`lazy`, the default;
`lazy-stream`, `copy-then-flip` and `stop-and-copy` as baselines) and `"RawPriority"` the class of
the lazy-stream raw-vector stream; each node keeps its raw vectors in `<WorkDir>/<node>/raw.pages`.

## The smallest experiment: one node, then two

A node is added because one node no longer serves the load, so the run has to put it under a
load one node cannot serve. Capacity first, then the scale-out:

```sh
python3 engine/scripts/make_synthetic.py --n 200000 --nq 2000 --dim 64 --dtype uint8 --out data/small
engine/build/fusion_build --base data/small-base.u8bin --out /tmp/rtier-small/index
python3 scripts/testing/make_range_assignment.py /tmp/rtier-small/index 16 data/small-assign.bin
python3 scripts/run_local.py configs/experiment.small-calibrate.json results/capacity
python3 scripts/run_local.py configs/experiment.small.json results/scaleout
```

The first run offers a ladder of rates to one node and reports where it stops keeping up; set
`"Load"."Rate"` below that and `"RateSteps"` above it in `configs/experiment.small.json`, so the
load outgrows one node before the rescale. The second run writes `timeline.svg` (throughput and
p50/p99 over time, with the flips marked), `timeline.csv`, `reconfigurations.json` (when each
rescale was triggered, how long it took, what it moved) and the metrics; `scripts/plot_run.py
results/scaleout` redraws them with another window size.

`TransferRateBytesPerSec` matters more than it looks on a small dataset: at full speed the few
MB move in milliseconds and nothing is visible around the flip. The small configs pace it at
4 MB/s so the reconfiguration lasts seconds, the way it would with a real index.

What this measures is the protocol: answers stay exact, the flip is short, and throughput
follows the load once the second node is online. It does not show why ownership is partitioned
at all -- with one node, its PQ codes fit by construction. That needs a budget that binds:
`configs/experiment.small-2to3.json` starts at two nodes with `--pq-capacity` below what one
node would need for the whole dataset. Pick the number from a first run: each node reports
`pq.codes_resident` in `metrics.jsonl`, and with the range-assignment placeholder a node holds
far more than its share of the codes (boundary vectors are replicated into up to 8 lists), so a
budget much below ~90% of all codes will fail the staging on purpose.

## Mean latency over time, per protocol

`scripts/compare_runs.py` overlays the mean latency of the answered queries per window for
several runs, one line per run, with each run's reconfigurations in a strip under the time
axis. Runs that share a directory name are pooled. The end-to-end test gives the smallest
version: with `RTIER_E2E_TIMELINE` set, `TestReconfigUnderLoad` writes each protocol's
queries and rescales (2 → 3 → 2 → 3 under closed-loop load) in the formats of the load
generator and `run_local.py`. Run each protocol in its own process, so that none of them
inherits another's warm-up, and leave out the first queries, which meet a cold start:

```sh
for p in lazy lazy-stream copy-then-flip stop-and-copy; do
  RTIER_ENGINE_BIN=$PWD/engine/build RTIER_E2E_TIMELINE=$PWD/results/e2e/rep1 \
    go test -count=1 -run "TestReconfigUnderLoad\$/^$p\$" ./test/e2e/
done
python3 scripts/compare_runs.py results/e2e/rep*/lazy results/e2e/rep*/lazy-stream results/e2e/rep*/copy-then-flip \
  results/e2e/rep*/stop-and-copy --window 0.02 --skip 0.1 --out results/e2e/latency-mean
```

Leave out `-race` when measuring: the agents run inside the test process. A mean covers
answered queries only, so it compares protocols only if no query gave up. The test's clients
wait for their answers. With `run_local.py`, set `"RetryForSeconds"` in `"Load"` so that a
query refused while stop-and-copy pauses admission waits too (`rtier-loadgen -retry-for`).

## Provenance

See `NOTICE`. Koala-derived code is marked in package comments; the engine is the FusionANNS
reimplementation from the paper (FAST '25), with rtier additions listed in `engine/README.md`.
