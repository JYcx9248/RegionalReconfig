# rtier — regional-tier reconfiguration prototype

Skeleton for non-disruptive reconfiguration of the regional tier of a distributed,
FusionANNS-style vector search system. A Go control plane (reusing Koala's structure) drives
C++ data nodes built on the FusionANNS reimplementation.

**Status:** skeleton. The settled parts of the design are implemented and covered by tests.
Open design questions (U1–U17, without U4) are placeholders: on a code path they fail with
`design decision pending [Un ...]`; the few without a code path yet (U12, U15, and U17's
list outside the epoch) are recorded in the registry with what the code does meanwhile. U4
(replicas of hot partitions) is out of scope: rtier keeps one owner per partition and handles
skew by placement -- a plan reassigns partitions, at the same node count if need be (U3).
Serverless-style read capacity, how streaming systems serve a hot key, would not relieve a
disk-bound hot spot: its data is on its owner's SSD. List them with `bin/rtier-client design`
(source: `internal/design/design.go`).

**PQ codes (decided part of U1):** node-level. A data node keeps one PQ code per vector named by
its resident posting lists, however many lists or partitions name it. When partitions move, the
destination's data node lists the codes it lacks, split by source so each is fetched once
(`PQ_MISSING`); the old owners send exactly those over the rate-limited bulk channel
(`PULL_PQ`); they are installed (`PQ_PUT`) before the partitions load, so a new node goes online
only once it holds every posting list and PQ code of its partitions and can filter at once.
Vectors are immutable (an update deletes and inserts under a new ID, U14), so the old owner keeps
the codes of partitions that moved away as a cache;
they are dropped only when a later staging needs the room, and a staging protects the codes it
needs. `rtier_node --pq-capacity N` sets a node's budget (its HBM, in codes). In fact nothing is
ever dropped: a scale-out only takes partitions away and a reversible scale-in only gives back
what it took, so a node's resident set stays what it held at the initial cluster size -- which
is why the budget is dimensioned by the *smallest* cluster size, not the current one. The
end-to-end tests assert it (no eviction); it stops holding if the cluster shrinks below its
initial size or placement moves partitions to nodes that never held them, as a full rebuild's
placement or a custom plan may. Inserts add to the resident set, so with updates the budget needs
their headroom too, and a deleted vector's code is freed with its last version (U14). The
initial deployment reads codes from the index. Known limits of this path are listed in
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
partitions that moved away, like PQ codes. Raw vectors need no versions under updates: vectors are
immutable, and IDs and locations are only appended. Decided for now: raw vectors are kept in full
-- no disk budget, no eviction, and deleted vectors' pages stay -- since disk is cheap, and the
tail pages' space amplification is accepted with it. Still open: where RAG chunks live (U12).
Partition directories written before locations existed (payload `lists-only`) are refused: rerun
`rtier_segment`.

**Raw vectors after a migration (decided part of U9):** a new owner goes online with posting
lists and PQ codes only -- all it needs to filter -- and gets a raw vector only when a query first
needs it (`"protocol": "lazy"`, the default): the point of being lazy is that queries touch a
small working set of what moved, and the rest never has to. Every posting records the holder of
its vector, the node that stores it (segment version 3), so each (list, vector) has one current
holder: the list's owner if it has the vector, else the node its posting names. Only the owner's
copy of a list counts: a partition moves as its owner saves it, holders included. From the flip
on, a RERANK that meets a vector not there fetches it from the holder that one of the query's
lists on that node records -- one hop, data node to data node -- waits for that fetch only,
holding no engine worker meanwhile, nor a connection slot of the aggregator's that FILTERs use
(RERANKs have their own, `cluster.node_conns` of each kind), and makes that posting name itself;
queries that miss the same vector meanwhile wait for the one fetch. Fetched vectors are
re-ranked from memory, and a background writer installs them on the SSD with whole-page direct
writes, off the query path. Nothing else moves and a scale-out does
not wait for raw vectors; `raw_fetched` in the rescale reply counts what queries fetched while it
ran, and each node reports `raw.fetched` (installed or still in memory), `raw.pending` (named by
its partitions, not needed so far) and `raw.present` in its metrics. A scale-in does stream:
before it returns, each remaining node takes over the vectors that its postings say only a
leaving node holds (the drain; `drain_vectors` and `drain_seconds` in the reply), so the nodes
that leave can go offline. That is the only time vectors move beyond what queries ask for: a
fetch brings the query's missing vectors and nothing more -- no page or list read ahead (decided
2026-10-20) -- with one request per holder, every holder at once. A new node takes partitions
from several donors, chosen by the partition graph, so a query's misses often lie at several
holders, and the RERANK waits for the slowest request, not for their sum.
Holders stay valid because no node drops a raw vector it has held; a disk budget would need the
same hand-off first. Baselines: `"lazy-stream"` also streams the rest in after the flip
(`StageRaw`; the rescale waits for it and reports `raw_vectors` / `raw_seconds`),
`"copy-then-flip"` streams them before the flip, `"stop-and-copy"` while paused. Streams and the
drain run data node to data node by default (`RAW_PULL`), unpaced like the queries' fetches;
only through the agents (`raw_path: agent`, the default when the agents are paced) do they take a
token-bucket class: `"raw_priority"` (background by default) after the flip and for the drain,
data for the eager baselines' copy before the flip. The pacing stays as it is, for now (U8): a
fixed rate, not adapted to foreground latency, and no byte rate for the drain beyond what
`cluster.raw` and the source's read budget allow.

**Placement (decided part of U3):** `"placement": "even-reversible-graph"` (the default) keeps
partition counts even and moves partitions only from existing to new nodes on scale-out -- each
donor gives its excess -- and from leaving to remaining nodes on scale-in, where it sends every
partition it can back to a node that has owned it before. So scaling out and back in restores
the placement exactly and the scale-in moves *no* PQ code: the returning partitions name vectors
their owner kept cached. Which partitions a donor gives, and where one that cannot go home goes,
it picks with the partition graph -- navigation-graph edges between the partitions' heads, read
off the index's `graph.hnsw` when the controller starts -- so that the nodes cut as little of it
as possible and fewer queries straddle two owners. On BIGANN-10M (64 spatial partitions, nprobe
64) 2 -> 3 gives 1.71 owners per query against 1.83 when every donor gives its tail
(`"even-reversible"`, `"even-reversible-blocks"`), one node at a time to 8 gives 3.31 against
3.81, and doubling does as well as whole blocks (2 -> 4: 1.81 both; 2 -> 4 -> 8: 2.69 against
2.76); `rtier-overhead -owners -rescale 2:3,2:4 -placement even-reversible-graph -index DIR`
measures it. End to end under a per-node read budget the capacity is the same, with 2-7% less
CPU per query and 5-8% fewer PQ codes staged. Layouts reached one node at a time still drift
from a fresh even split (3.31 against 2.76 at 8 nodes): with no data exchanged between old nodes,
each node's region only shrinks, and only bigger steps avoid that. That is the price of moving no
data between old nodes, and it is accepted (a custom plan may move anything); whether the gap
closes in larger clusters, as it must when the cluster size approaches the partition count, is to
be measured. The initial placement stays contiguous ranges of partition IDs, which the spatial and
GP generators (U2) number along their bisection: close to the best at power-of-two sizes. A node
removed by a scale-in is shut down, so the highest IDs leave (the latest joiners, whose partitions
go home) and any idle node joins; a node coming back is not a case we handle.
Koala's policy (`"even"`,
kept as the baseline) is not reversible: after 4 -> 8 -> 4 none of the partitions that moved is
home again. It has no reason to care, because its state changes with every input, so a copy left
behind is stale; with immutable vectors the copy stays valid (a deleted vector's code goes with
it, U14), which is what makes this worth doing.

**Graph transfer (decided part of U16):** a new node needs posting lists and PQ codes to own
partitions and answer FILTER and RERANK -- the steps that take load off the old owners -- and
the navigation graph only for the entry role, whose navigation any existing entry can do
meanwhile. So each new entry's graph source is assigned when the reconfiguration starts,
round-robin over the live entries (`graph_sources` in the rescale reply), and the pull runs at
background priority (`"graph_priority": "background"`, the default): the graph moves only while
neither its source nor its destination moves data -- segments or PQ codes through the agent,
raw vectors through the data node (`RAW_TRAFFIC`, polled every 100 ms) -- paced or not, so it
delays neither the flip nor a query's fetch. In the worst case, queries fetching without pause,
a new node becomes an entry only once the fetching subsides; only navigation waits for it,
which the existing entries do meanwhile. This strict yield is decided for now: whether the
graph moves nothing at all while data moves, or takes the bandwidth the data leaves unused, is
settled once the experiments show what it costs. `"data"` shares the source's bucket equally,
as a baseline. A new node
becomes an entry as soon as its graph has loaded, not when the slowest one has: while graphs
are arriving, the controller flips every `"entry_flip_interval"` (1 s by default) to add all the
nodes that are ready, and adds the last ones at once when every pull has ended. These flips
change only the entries, so they run alongside the grace period of the data flip. A node whose
graph fails to load stays a data node and aggregator without holding up the others, and the
rescale returns an error naming it. The transfer stays eager: an entry without the whole graph
serves no purpose, so no node takes the entry role before its graph has loaded. With `lazy` the
graph is the only transfer in the background class. No node ever holds two copies of the graph:
a major epoch patches it in place and a full rebuild moves the roles to spares (U14). How large
the replica grows with the dataset is outside this work: an assumption, with smaller routing
state as future work (U16).

**Clients' entry lists (decided part of U17):** a client (`rtier-loadgen`) reads the entries
from the controller's epoch table every 500 ms, and at once when an entry refuses a query under a
newer epoch than its list's. That entry has left the entry set (a scale-in, a full rebuild), so
the list is stale and the retry goes to the entries of that epoch; the queries that find it stale
together share one read. A refusal carries the entry's epoch, so one under the list's own epoch
-- paused admission under stop-and-copy, or an entry that has not installed the epoch yet --
reads nothing and is retried after a pause. The list stays in the epoch table, for now: a new
node owns its partitions and aggregates from the data flip on, and only the entry role waits for
its graph, so a list kept elsewhere would change when the rescale counts as done (once every new
node is an entry), not when a node starts working.

**Aggregator (decided part of U6):** the entry navigates first, then works out which nodes own
the probed lists, and only then picks the aggregator among them (`"selector": "owner"`, the
default): itself if it is one of the owners; the owner if a single node holds every list -- the
query is forwarded once and both phases run where the data is, one network round trip instead of
two; itself otherwise, since forwarding would add a hop to save one of several parallel
sub-queries and pile aggregation onto the owners of hot partitions. `"local"` always aggregates at
the entry (the baseline). `"warmup"` hands the scatter/gather to a node that may aggregate but is
not an entry yet -- during a scale-out, a new node whose navigation graph is still arriving.
`rtier-overhead -owners` reports how often each case occurs for a dataset and partitioning.
The aggregator is not outsourced beyond this (decided for now): `"owner"` stays the default.
A query runs under one epoch, the one its entry pinned: a forwarded query carries that epoch and
its lists grouped by owner, and the aggregator runs it as is -- it neither routes with nor checks
an epoch of its own. It counts the work under the query's epoch (`Tracker.Pin`), so its own drain
waits for it, while the grace period relies on the entry, which keeps its pin until the answer is
back. Epochs are installed
on the nodes in parallel, so this is what lets a joining node aggregate for an entry that
installed the new epoch before it did, and a leaving node finish the queries it was handed just
before the flip (`test/e2e/flip_race_test.go` holds an install or a query to reproduce both).
A sub-query can meet a reclaimed partition only after a failure -- a forward still unread when
the entry's client times out -- and then fails its query with an error: failures are out of
scope (U7, U13).

**Dataset updates (decided part of U14, first version):** updates come in batches, and each
batch is a content epoch that reuses staging, flip, grace and reclaim. Vectors are immutable and
the PQ codebook is fixed: a modification deletes a vector and inserts it under a new ID, and IDs
and raw-vector locations are only ever appended, so nothing a former owner keeps goes stale. The
epoch table names a version of each partition's lists, and FILTER and RERANK name the version
their epoch sees, so a data node holds a partition's next version next to the one the current
epoch's queries read. Batches run only when the controller has `"allow_updates"` (off by
default, so a static experiment cannot have its data changed; `run_local.py`'s `updates` section turns
it on). The update controller (`internal/update`, `bin/rtier-updater`) decides a batch and the
nodes carry it out:

1. It takes the controller's lease (`BeginUpdate`): no reconfiguration and no other batch runs
   until it commits or aborts, or `"update_lease"` expires. Before granting it, the controller has
   the agents release PQ codes left staged by a staging that never ended.
2. It gives each insert an ID and a location and routes it by the build's own rule (`ASSIGN`: the
   lists the build would have chosen, and the PQ code) on its own router: an `rtier_node` with the
   index's navigation graph and codebook and nothing else, never a serving node, whose navigation
   CPU queries need. Then it checks that every owner has room.
3. It sends the codes and raw vectors to the owners' data nodes (`PQ_PUT`, `RAW_ADD`), which
   serve the vectors from memory and write them after the fetched ones, within
   `--update-write-mbps` (writes are not charged to the read budget yet).
4. It has every data node stage the next version of the partitions the batch changes
   (`STAGE_VERSIONS`): inserts appended, deleted IDs dropped. Deletes go to every node, and each
   finds its own copies.
5. `CommitUpdate` checks the staged versions, flips (all entries switch at once), waits out the
   grace period and reclaims. Every node, former owners included, frees the deleted vectors' PQ
   codes, and then the owners drop the replaced versions. A code that a replaced version still
   names is marked first and goes with it, so a deleted vector's code never becomes a cached one:
   cached codes are kept because they may be needed again, and a deleted vector never is.

A failed staging is undone before the lease is given up, and a commit that fails before the flip
drops what was staged. A partition at a later version moves with its current lists: its owner
writes them as a segment file when it moves. With `lazy`, a vector inserted into a partition
stays with the owner that added it when the partition moves on (the postings name that node as
its holder), and a scale-in that removes the node drains it first.

**Major epochs** split and merge posting lists, as a batch with its own lease
(`internal/update/major.go`). Only minor batches come from outside: the owners report the
lengths of the lists a batch changes, and `MajorDue` splits the lists over one length and merges
the ones under another (`rtier-updater -split-len`, `-merge-len`; `-split`, `-merge` by hand).
The update controller decides on PQ codes, so no raw vector is read but the new heads':

- a split is a balanced 2-means of a list's members; each child's head is the member closest to
  its mean, as the build chooses heads, and both children stay in the parent's partition;
- a merge sends each member to the list of its nearest live head, whatever its partition. A
  posting that changes owner moves as (vector ID, location, holder) with its PQ code, and its raw
  vector follows on demand from the node that holds it, as after a migration.

The graph changes without any node holding two versions of it. The update controller's router
inserts the new heads and marks the removed ones deleted (routing points, never a result), and
exports the bytes that changed: a patch (`engine/include/fusion/graph_patch.h`). Every entry
stages the patch next to its graph. Queries of the epoch after the flip search through it -- a
changed element is read from the patch, every other one from the graph, so the search is the
patched graph's, step by step -- and those of the epoch before it search the graph alone. After
the grace period every entry keeps the patch as a file, where a later entry's graph transfer takes
it, and merges it in place without a pause; the lease ends after that. The epoch table names the
graph version and the lists major epochs created; data nodes leave room for heads and lists
(`rtier_node --graph-headroom`, 20% by default).

**Full rebuild** (`internal/controller/rebuild.go`, `rtier-client rebuild`) moves the cluster to
another generation of the index: all heads, the graph, every list and partition. The next
generation is built outside the system and handed over as files (a source interface: an index
directory from `fusion_build` and a partition directory from `rtier_segment`). It keeps the
codebook, the vector IDs and the raw-vector locations: `fusion_build` reuses `pq_codebook.bin`,
`pq_codes.bin` and the page layout when they are already in its output directory. So the PQ
codes and raw vectors every node holds stay valid, and no node ever holds two generations' lists
or graphs. With N data nodes and N spares (idle nodes; `-spares`):

1. The spares take over every partition by a lazy reconfiguration. The old nodes stay members
   that only serve raw vectors (they are not drained), and lose the entry role once the spares
   have the graph.
2. The old nodes switch to the next generation (`SWITCH_GENERATION`). It drops their lists and
   graph and keeps their PQ codes and raw vectors.
3. The new partitions go where most of their raw vectors already are: the old nodes' presence
   bitmaps (`RAW_BITMAP`) weigh each (partition, node) pair, and a min-cost flow maximizes the
   total under the even policies' balance (`placement.Affinity`; `-placement random` is the
   baseline). Every posting whose vector its owner lacks names an old node that holds it, so a
   missing vector is one hop away.
4. The old nodes load the new partitions (PQ codes from the generation's index) and graph. They
   then replay the recent client queries the entries kept (`query_sample`, 1024 per entry) under
   the new generation's draft epoch, so the raw vectors those queries need are on their owners
   before any client query arrives (`-warmup`).
5. One flip puts the new generation online on the old nodes. After the grace period the spares,
   which hold only copies, move to the new generation too and are idle again.

Queries get exact answers throughout: an epoch's generation decides which lists and graph they
read. A node that joins later moves to the table's generation before it stages anything, and the
update controller's router follows the lease's generation. `-stop-the-world` is the baseline
without spares: admission is paused while every node switches where it is. Updates and other
reconfigurations wait until the rebuild ends. Not there yet: the trigger (only the manual one),
a generation that includes inserted vectors (the builder would need their IDs and locations),
and a priority between the warm-up's fetches and the spares' (only the warm-up's pace is set).

Not there yet for updates: growing the ID and location spaces, a response to a batch over budget
(it fails before anything is sent), reassigning the vectors of neighboring lists in a split
(SPFresh's LIRE), and a log to replay batches from. Deleted vectors' pages are kept (U1).
`test/e2e/update_test.go`, `test/e2e/major_test.go` and `test/e2e/rebuild_test.go` check every
answer against an oracle that applies the same decisions, across batches, a failed batch, a major
epoch, a full rebuild and rescales.

## Layout

| Path | What |
|---|---|
| `cmd/rtier-controller` | control plane: registration, placement, epochs, reconfiguration, metrics sink |
| `cmd/rtier-agent` | per-node agent next to `rtier_node`: staging, epochs, entry + aggregator roles |
| `cmd/rtier-client` | status / rescale / rebuild (U14 full rebuild) / design / queries / single-node baseline |
| `cmd/rtier-loadgen` | open-loop load (latency measured from the scheduled send time) with a rate schedule, or closed-loop users; follows the entry list (U17) |
| `cmd/rtier-updater` | the update controller: applies one batch of inserts and deletes to a running cluster (U14), routing inserts with its own router (`-router`, or `-index` to start one), and runs major epochs (`-split-len`, `-merge-len`; `-split`, `-merge`) |
| `cmd/rtier-overhead` | cost of fan-out: emulates N owners on one data node holding every partition and reports what the two-phase query adds over N = 1 (owners per query, duplicate PQ scoring, extra page reads, per-call and merge time); `-compare` sets two-phase against local re-ranking (recall with `-gt`), `-owners` shows how queries spread over owners |
| `internal/` | ctrl RPC, protocol messages, placement, epoch store + tracker, bulk transfer, query aggregation, metrics, config, ... |
| `engine/` | FusionANNS engine + `NodeEngine` (partitioned posting lists, query primitives), `rtier_node`, `rtier_segment` |
| `test/e2e` | real `rtier_node` processes under query load: scale-out and scale-in, plans, update batches, major epochs, full rebuilds, diagnostics forwarding |
| `scripts/` | `run_local.py` experiment runner (capacity ladder + timed rescales; YAML configurations of a base and overrides, `experiment_config.py`; `convert_configs.py` turns the old JSON ones into them), `run_cluster.py` the same with the nodes on other machines over SSH, `plot_run.py` timeline, `compare_runs.py` mean latency of several runs overlaid, `emulation/netns.sh`, metrics → SQLite, Python API client |

## Build and test

Needs Linux, CMake ≥ 3.18, a C++17 compiler with OpenMP, Go ≥ 1.24 (standard library only),
Python 3 with numpy and PyYAML for the scripts. CUDA ≥ 11 and liburing are optional.

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
# list -> partition assignment (U2). TEST-ONLY placeholder here: contiguous ranges. The decided
# partitioner is GP on the heads over the navigation graph (needs KaMinPar, scripts/testing/gp/):
#   make_gp_assignment.py INDEX 64 OUT --points heads --graph-source index --kaminpar PATH
# make_spatial_assignment.py (recursive bisection of the heads) is the baseline
python3 scripts/testing/make_range_assignment.py /tmp/rtier-run/index 64 data/assign.bin
python3 scripts/run_local.py configs/experiment.synthetic.yaml results/run1
```

Experiment configurations are YAML, a base and overrides: an experiment file
(`configs/experiment.*.yaml`) names the file it builds on with `base:` (`configs/base/*.yaml`)
and gives only what differs; `scripts/experiment_config.py` has every key and prints a merged
configuration. `run_local.py` makes what is missing and records what each stage was made with:
the index (`build`: fusion_build's flags), the ground truth, the assignment (`partition`: a
`file`, or `generator: gp` with `partitions: N`; `spatial` and `range` are placeholders) and
the partitions, which it makes again when the index or the assignment changes. An index built
with other flags, or a ground truth made of other vectors, queries or stride, stops the run
instead of being made again. `dataset.ground_truth_stride: S` computes the ground truth of every
S-th query only (`fusion_gt --stride`), so that a query file longer than a run, which repeats no
query, has recall measured all through it. Each results directory records the code it ran:
`revision.txt` (the commit and the files changed on top of it) and `revision.diff`; the runners
refuse Go or engine binaries older than their sources. `scripts/run_one.sh CONFIG PROTOCOL
OUT ['{...}']` runs one configuration with the protocol and any other key overridden.

`configs/experiment.example.yaml` is the same run on BigANN (edit the paths). With
`cluster.emulation: {netns: true}` (root), each node runs in its own network namespace with its
link shaped by `tc` (`scripts/emulation/netns.sh`; delay needs the `netem` qdisc).

On one machine every node reads the same SSD, and the two-phase RERANK reads ~150 pages per
query, so without more the cluster stops at what that disk serves and a scale-out adds no
throughput (measured: 2 and 3 nodes both at ~130-140K reads/s, ~920 q/s on BIGANN-1M). With
`cluster.resources: {cpus_per_node: 4, io_read_iops_max: 40000}` each node and its agent run in a
cgroup of their own (`systemd-run --scope`): their own CPUs and their own read IOPS on the disk
of `paths.work_dir`, as if each node had its machine; keep the sum below what the disk serves.
The controller and the load generator get the CPUs after the nodes'. `memory_max` and
`io_read_bandwidth_max` work the same way. Each node's and each agent's cgroup is then sampled
once a second into `cgroups.jsonl` (CPU time, reads and writes on the data disk, dirty and
writeback page cache, pressure stall times). Every run also collects each data node's stats log,
`stats-<node>.jsonl` (`rtier_node --stats-file`: operation counts and times, time spent waiting
for an engine worker, the fetch cache and its writer), which its agent forwards to the controller
and the controller stamps on arrival (`"t"`; the node's own time stays as `"node_t"`). Both are on
the monotonic clock of `clock.json`, whose `load_start` is the load's time 0.

Without root (a shared server), `cluster.resources` uses what systemd delegates to users: each
node and its agent are pinned to their CPUs with `taskset` and run in a user scope
(`systemd-run --user`) that carries `memory_max`. The io controller is not delegated, so
`io_read_iops_max` fails; `read_iops_per_node: 40000` gives each data node a read budget of its
own instead (`rtier_node --read-iops`, `engine/include/fusion/read_budget.h`): it charges
re-ranking's page reads, the pages the fetch writer reads back and the `RAW_GET` reads the page
cache misses, as io.max would, and works as root too. The kernel's read-ahead is not charged:
scattered vectors get none, a stream's sorted batches some (`engine/README.md`, known limits). Checked against io.max on BIGANN-10M
(lazy, 2 → 3 nodes at 40K reads/s each, open loop 400 → 700 q/s): the same capacity (625 q/s
on 2 nodes), flip time, latency below capacity and backlog, and the reads it charged matched the
disk's to 0.2%. Pinning does not keep other users off a
node's CPUs, so `cgroups.jsonl` also has a `"host"` record per sample: the data disk's counters
(everyone's I/O), how busy each node's CPUs were and the load average; a user scope has no
`io.stat`, so a node's I/O comes from `/proc/<pid>/io` there (bytes, no I/O counts: the read
budget's `"reads"` in `stats-<node>.jsonl` counts reads). `scripts/run_one.sh` drops only the
run's own files from the page cache without root (`scripts/evict_cache.py`) and leaves
`vm.swappiness` as it is.

Queries run through the two-phase strategy (`cluster.strategy: two-phase`, the decided answer to
U5 for now, agreed with the PI): every owner filters for its own top-n, the aggregator merges the
global top-n and re-ranks each candidate at the owner that reported it, which is exactly the
single-node answer. A one-round-trip quota per owner, whose answer can differ, is not pursued.
`test/e2e` checks that equality query by query while the cluster scales out and in, and that
a vector new to a node arrives at most once: with `lazy` only when a query needs it, with the
baselines all of them, streamed or fetched on demand.

`cluster.backend: gpu` (a CUDA build of the engine) runs the data nodes on the GPU filter backend, and
`cluster.gpus: [0, 1, 4, 5]` gives node i the i-th of those GPUs, numbered as `nvidia-smi` numbers them
(the node runs with `CUDA_DEVICE_ORDER=PCI_BUS_ID` and only its GPU visible; nodes take turns
when there are fewer GPUs than nodes). With `cluster.resources`, the `"host"` records in
`cgroups.jsonl` then also have those GPUs' utilization and memory in use, other users' jobs
included.

`cluster.protocol` in an experiment configuration picks the reconfiguration protocol (`lazy`, the
default; `lazy-stream`, `copy-then-flip` and `stop-and-copy` as baselines). `cluster.raw` sets how
a stream pulls (`streams`, `batch_bytes`, and `path`: `node`, or `agent`, the default only when
`cluster.transfer_rate_bytes_per_sec` paces the agents), and
`cluster.raw_priority` its class on the agent path. Each node keeps its raw vectors in
`<WorkDir>/<node>/raw.pages`.

## Run on a cluster

```sh
python3 scripts/run_cluster.py configs/experiment.X.yaml results/run1
```

`scripts/run_cluster.py` runs the same configuration with the data nodes on other machines, the
way Koala's runner deploys. A `hosts` section names the host of each node:

```yaml
hosts:
  nodes: [10.0.0.2, 10.0.0.3, {ssh: gpu4, addr: 10.0.0.4}]   # n1, n2, n3; a host may repeat
  root: 10.0.0.1      # where the nodes reach this machine (default: found from the route to them)
  user: chenxu        # the SSH login; ssh_options: more options for ssh
```

This machine, the root, runs the stages, the controller and the load generator. It first checks
every host over SSH: a login without a prompt, rsync, writable directories, and an engine build
that its CPU and GPUs can run. A `-march=native` build from another CPU would die with SIGILL.
Then it copies to each host, at this machine's paths, what its nodes open: `rtier_node` and
`rtier-agent`; the index's meta, PQ codebook and PQ codes (`rtier_node` checks the codes at
startup, though only the bootstrap reads them) and the partitions' manifest; where a node of the
initial deployment runs, also the page file, the graph and the segments it stages from -- a node
that joins later gets these from its peers; and of each rebuild's next generation, its meta,
codebook, PQ codes, graph and manifest (its segments come from the controller). The rest of an
index (postings, centroids, layout maps) is the build's own and stays here. rsync sends only what
changed, and a directory a host already sees at its path (this machine, a shared file system) is
not copied. The nodes start in order over SSH, each after the previous one registered, so `n1`,
`n2`, ... get IDs 1, 2, .... A remote process stops when its SSH session closes, also when the
runner dies.

The results arrive on the root as on one machine. The controller writes `metrics.jsonl`, and
each agent forwards its data node's stats log (`stats-<node>.jsonl`) and samples of its node,
itself and its machine (`cgroups.jsonl`). The controller stamps every record on arrival, as
Koala's metric collector does, so all of them share one clock -- the one of `clock.json` and the
load generator -- and the hosts' clocks need no synchronization. `cluster.gpus` applies on each
node's own machine. `cluster.resources` and `cluster.emulation` emulate several machines on one
and are refused, except `read_iops_per_node`; `run_local.py` refuses a configuration with
`hosts`. To try the runner without an SSH server, `PATH=$PWD/scripts/testing/fake_ssh:$PATH`
runs every "remote" command on this machine. The hosts then share its files and nothing is
copied; with `FAKE_SSH_PRIVATE` (directories, colon-separated) and `FAKE_SSH_ROOT`, each host
sees a copy of its own at those paths (a mount namespace, root only), so the copies are made and
the nodes run on them.

## The smallest experiment: one node, then two

A node is added because one node no longer serves the load, so the run has to put it under a
load one node cannot serve. Capacity first, then the scale-out:

```sh
python3 engine/scripts/make_synthetic.py --n 200000 --nq 2000 --dim 64 --dtype uint8 --out data/small
engine/build/fusion_build --base data/small-base.u8bin --out /tmp/rtier-small/index
python3 scripts/testing/make_range_assignment.py /tmp/rtier-small/index 16 data/small-assign.bin
python3 scripts/run_local.py configs/experiment.small-calibrate.yaml results/capacity
python3 scripts/run_local.py configs/experiment.small.yaml results/scaleout
```

The first run offers a ladder of rates to one node and reports where it stops keeping up; set
`load.rate` below that and `load.rate_steps` above it in `configs/experiment.small.yaml`, so the
load outgrows one node before the rescale. The second run writes `timeline.svg` (queries offered
and answered per second, answers counted when they came so that a stall shows, and the mean
latency by send time, with the flips marked; `--latency tail` for p50 and p99), `timeline.csv`,
`reconfigurations.json` (when each rescale was triggered, how long it took, what it moved) and
the metrics; `scripts/plot_run.py results/scaleout` redraws them with another window size. A run
cut short keeps what it measured: `reconfigurations.json` is written after each
reconfiguration, and a load generator stopped before its backlog drained (`load.drain_seconds`
after the load, 300 s by default) records the queries still unanswered as `abandoned`.

`cluster.transfer_rate_bytes_per_sec` matters more than it looks on a small dataset: at full speed the few
MB move in milliseconds and nothing is visible around the flip. The small configs pace it at
4 MB/s so the reconfiguration lasts seconds, the way it would with a real index.

What this measures is the protocol: answers stay exact, the flip is short, and throughput
follows the load once the second node is online. It does not show why ownership is partitioned
at all -- with one node, its PQ codes fit by construction. That needs a budget that binds:
`configs/experiment.small-2to3.yaml` starts at two nodes with `--pq-capacity` below what one
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
wait for their answers. With `run_local.py`, set `load.retry_for_seconds` so that a
query refused while stop-and-copy pauses admission waits too (`rtier-loadgen -retry-for`).

## Provenance

See `NOTICE`. Koala-derived code is marked in package comments; the engine is the FusionANNS
reimplementation from the paper (FAST '25), with rtier additions listed in `engine/README.md`.
