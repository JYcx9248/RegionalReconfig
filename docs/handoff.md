# rtier — handoff notes

Written on 2026-09-27 to continue the work from another Claude account. It holds what lived only
in the conversations that built rtier (Sep 18–27, 2026). Read `CLAUDE.md` first (build, layout,
conventions, key mechanisms), then this file; `docs/design.md` is the design doc (Chinese).

## 1. What to bring over

| File | What | Needed |
| --- | --- | --- |
| `rtier.tar.gz` | The whole repository: C++ engine (`engine/`, FusionANNS reimplementation + rtier additions), Go control plane, scripts, configs, tests, docs | Yes |
| `docs/design.md` | The design doc kept on claude.ai, exported at rev 123 and updated for the lazy raw-vector fetch (the claude.ai copy was not changed) | Yes: the claude.ai doc stays in the old account |
| `docs/handoff.md` | This file | Yes |
| `fusionanns.tar.gz` | The standalone FusionANNS reimplementation | No: `rtier/engine` contains it, and single-node baselines run inside rtier (`fusion_search`, `SEARCH_LOCAL`, `rtier-overhead`) |
| `rtier.zip`, `fusionanns.zip` | Older packages | No, outdated |

What does not carry over by itself: the claude.ai design doc (import `docs/design.md` into a new
doc if you want one), the chat history (condensed here), the inputs you attached in the old chats
(the FusionANNS paper, the Koala and SCDN repositories: attach them again if a session needs
them), and build outputs (`-march=native`: rebuild the engine on every machine).

## 2. Resume

```sh
tar xzf rtier.tar.gz && cd rtier
make test CUDA=OFF   # engine + Go; fusion_tests (8), rtier_node_tests (9), Go tests with -race, e2e (12)
```

Linux, CMake >= 3.18, a C++17 compiler with OpenMP, Go >= 1.24, Python 3 with numpy. The
Makefile sets `GOTOOLCHAIN=local`. Sanitizer check: build the engine with
`-fsanitize=address,undefined -DNO_MANUAL_VECTORIZATION` into another directory and point
`RTIER_ENGINE_BIN` at it (see `CLAUDE.md`). Existing partition directories from older builds are
refused (segment format 2): rerun `rtier_segment`.

## 3. Where things stand

Implemented and covered by tests (details in `CLAUDE.md`, `README.md`, `internal/design/design.go`):

- Epoch-based reconfiguration (pre-connect, stage, CAS flip, grace, reclaim); one epoch per
  query, pinned at the entry and carried by forwarded aggregations; batched entry flips as
  navigation graphs load; graph transfer at background priority from round-robin sources.
- Two-phase query (FILTER top-n at each owner, global merge, RERANK at a reporting owner):
  equal to the single-node answer; owner-based aggregator selection (U6).
- Node-level PQ codes (one per vector per node, missing ones pulled from old owners, moved-out
  codes kept as a cache) and reversible placement (a scale-in moves nothing).
- Latest change: no global replica of the raw vectors. Postings carry each vector's canonical
  location; each node keeps a sparse local copy of the page file. With the default protocol
  `lazy`, a new owner goes online with posting lists and PQ codes only; RERANK fetches missing
  vectors from the old owner on demand while the agent streams the rest in the background.
  `copy-then-flip` and `stop-and-copy` stay as eager baselines.

Last measurements (2-core sandbox, synthetic data, so trends only):

| Run | Result |
| --- | --- |
| e2e, 8000 vectors, 2 -> 3 -> 2 -> 3 nodes under load | 0 mismatches with the single-node oracle in all three protocols; lazy: 5836 queries, 0 retries, max 13.8 ms |
| Same run, lazy 2 -> 3 | 3977 vectors new to the joining node: 3935 streamed, 42 fetched on demand; scale-in and re-scale-out moved 0 |
| `run_local.py`, 30K x 64-dim uint8, 1 -> 2, 300 KB/s | 8.4K vectors fetched on demand within ~3.5 s of the flip; stream done 4.2 s after the flip (19.3K vectors); 2% sent twice with ~64 KB batches (16% with 8192-vector batches) |

## 4. GitHub issue drafts

Drafted for the project's GitHub issue (whether and where they were posted is not recorded
here). Style the user asked for: boxed ASCII in a `text` block, role box on the left, numbered
step on the right, concurrent steps as compartments of one box.

### 4.1 Query workflow (get the top k closest vectors)

Updated on 2026-09-27 to match the code: the entry (not the aggregator) groups the lists by owner
and forwards them with its epoch (one epoch per query), and right after a migration step 6 first
fetches the candidates' raw vectors that have not arrived yet from the partition's old owner.

```text
  ┌────────────┐ ┌─────────────────────────────────────────────────────┐
  │   Client   │ │ 1  Query goes to an entry node                      │
  └────────────┘ └─────┬───────────────────────────────────────────────┘
                       │
                       ▼
  ┌────────────┐ ┌─────────────────────────────────────────────────────┐
  │   Entry    │ │ 2  Search the centroid graph                 [DRAM] │
  │            │ │    → candidate posting lists                        │
  │            │ │    Group them by owner. Aggregator: itself,         │
  │            │ │    unless it owns none of the lists and another     │
  │            │ │    node owns all of them                            │
  └────────────┘ └─────┬───────────────────────────────────────────────┘
                       │  2.5  only if another node aggregates: send
                       │       the query, its lists by owner, the epoch
                       ▼
  ┌────────────┐ ┌─────────────────────────────────────────────────────┐
  │ Aggregator │ │ 3  Send each owner its lists                        │
  └────────────┘ └─────┬───────────────────────────────────────────────┘
                       │  FILTER(query, its lists, n)
                       │  to each owner, in parallel
                       ▼
  ┌────────────┐ ┌─────────────────────────────────────────────────────┐
  │   Owners   │ │ 4  Search local PQ codes of its lists     [GPU HBM] │
  │            │ │    → local top-n closest vectors (PQ distance)      │
  └────────────┘ └─────┬───────────────────────────────────────────────┘
                       │  each owner: ≤ n (ID, PQ distance)
                       ▼
  ┌────────────┐ ┌─────────────────────────────────────────────────────┐
  │ Aggregator │ │ 5  Merge, dedup vector IDs → global top-n           │
  │            │ │    Split them: each to ONE owner that reported it   │
  └────────────┘ └─────┬───────────────────────────────────────────────┘
                       │  RERANK(query, its candidates, k)
                       │  (n candidates in total)
                       ▼
  ┌────────────┐ ┌─────────────────────────────────────────────────────┐
  │   Owners   │ │ 6  Read its candidates' raw vectors           [SSD] │
  │            │ │    → exact distances → local top-k                  │
  │            │ │    (after a migration: fetch the ones not there     │
  │            │ │    yet from the partition's old owner first)        │
  └────────────┘ └─────┬───────────────────────────────────────────────┘
                       │  each owner: ≤ k (ID, exact distance)
                       ▼
  ┌────────────┐ ┌─────────────────────────────────────────────────────┐
  │ Aggregator │ │ 7  Merge → final top-k vector IDs                   │
  │            │ │    Pull related chunks from owners (U12, open)      │
  │            │ │    → client, through the entry (pin released)       │
  └────────────┘ └─────────────────────────────────────────────────────┘

  n = candidates to re-rank (tunable, e.g. 200)
  k = results (e.g. 10)
  The entry pins its current epoch when it admits the query;
  every step runs under that epoch, forwarded or not.
```

### 4.2 Reconfiguration from the new node's point of view

Updated on 2026-09-27 for the lazy raw-vector fetch: step 2's "No raw vector yet" line, step 4's
last two lines, step 7 and the footer are new; the earlier draft predates them.

```text
  ┌────────────┐ ┌─────────────────────────────────────────────────────┐
  │  New node  │ │ 0  Register with the controller → gets its NodeID   │
  │            │ │    Idle: in no epoch, no role yet                   │
  └────────────┘ └─────┬───────────────────────────────────────────────┘
                       │  rescale requested
                       ▼
  ┌────────────┐ ┌─────────────────────────────────────────────────────┐
  │ Controller │ │ 1  Pick idle node(s); the placement policy decides  │
  │            │ │    which partitions move to them                    │
  │            │ │    Draft epoch e+1 (not in force yet): the new      │
  │            │ │    node owns those partitions and may aggregate     │
  │            │ │    Graph source: an existing entry, round-robin     │
  └────────────┘ └─────┬───────────────────────────────────────────────┘
                       │  1.5  pre-connect: e+1's aggregators (new node
                       │       included) ping e+1's data nodes
                       ▼
  ┌────────────┐ ┌─────────────────────────────────────────────────────┐
  │  New node  │ │ 2  Pull its posting lists from their old owners     │
  │            │ │    List the PQ codes it lacks (each vector once),   │
  │            │ │    pull them, load the partitions                   │
  │            │ │    No raw vector yet: they come after the flip      │
  │            │ │    → tells the controller: ready for e+1            │
  ├────────────┤ ├─────────────────────────────────────────────────────┤
  │  New node  │ │ 3  Meanwhile, in the background: pull the centroid  │
  │            │ │    graph from its assigned entry (low priority),    │
  │            │ │    load it → tells the controller: ready as entry   │
  └────────────┘ └─────┬───────────────────────────────────────────────┘
                       │  every new node is ready for e+1
                       │  (its graph may still be on the way)
                       ▼
  ┌────────────┐ ┌─────────────────────────────────────────────────────┐
  │ Controller │ │ 4  Flip e → e+1: write it to the epoch store,       │
  │            │ │    install it on every node (in parallel)           │
  │            │ │    New node now answers FILTER/RERANK for its       │
  │            │ │    partitions and can aggregate; not an entry yet   │
  │            │ │    RERANK fetches the raw vectors it lacks from     │
  │            │ │    the old owner, on demand                         │
  └────────────┘ └─────┬───────────────────────────────────────────────┘
                       │  queries pinned to e keep using the old owners
                       ▼
  ┌────────────┐ ┌─────────────────────────────────────────────────────┐
  │ Old nodes  │ │ 5  Wait until no query of epoch e is running,       │
  │            │ │    then drop the moved posting lists                │
  │            │ │    (PQ codes and raw vectors stay, cached)          │
  ├────────────┤ ├─────────────────────────────────────────────────────┤
  │ Controller │ │ 6  Meanwhile, every 1 s: one flip (e+2, e+3, ...)   │
  │            │ │    makes entries of the new nodes whose graph is    │
  │            │ │    loaded, until every graph pull has ended         │
  ├────────────┤ ├─────────────────────────────────────────────────────┤
  │  New node  │ │ 7  Meanwhile: stream in the raw vectors it still    │
  │            │ │    lacks from the old owners (low priority),        │
  │            │ │    skipping those queries already fetched           │
  └────────────┘ └─────────────────────────────────────────────────────┘

  e = the epoch in force when the rescale starts.
  2 and 3 run at the same time, and so do 5, 6 and 7; the
  rescale returns once all three have finished (a node that
  leaves stays up as a raw-vector source until then).
  A new node whose graph fails stays owner + aggregator.
  Baseline copy-then-flip does step 7 before the flip instead.
```

### 4.3 Local experiments (condensed)

Local runs cannot show scaling curves (that needs a multi-node GPU testbed); they can show the
protocol works and measure data-dependent costs:

1. Data-dependent counts with `rtier-overhead`: owners per query, PQ redundancy across
   partitions, bytes and heat per partition, recall vs nprobe and n, PQ code size
   (`fusion_build --pq-m`).
2. Protocol comparison on one machine: `lazy` vs `copy-then-flip` vs `stop-and-copy` (time to
   flip, unavailable window, p99 around the flip, warm-up fetches, `raw_priority`).
3. A 1 -> 2 scale-out demo under a load one node cannot serve (capacity ladder first), each node
   pinned to its own cores (`systemd-run -p AllowedCPUs=... -p MemoryMax=...`, not wired into
   `run_local.py` yet).
4. Real data: BEIR (SciFact to build the pipeline, then Quora), embedded with a cosine-similarity
   model and normalized (the engine is L2 only), float32 <= 1024 dims (one vector per 4 KB page),
   exact kNN ground truth.

## 5. Open discussion: queries during a reconfiguration

This is where the conversation stopped: an issue on how queries behave while the cluster
reconfigures (failures excluded). Reconstructed from the code, not quoted from the chat.

Rules the behavior follows: an entry pins its current epoch when it admits a query
(`Tracker.Enter`) and keeps the pin until the answer is back; routing, the choice of aggregator
and the aggregation all use that epoch; a forwarded aggregation carries the epoch and the lists
grouped by owner, and the aggregator runs it as given (it only `Pin`s the epoch for its own
bookkeeping). The controller installs e+1 on all nodes in parallel, then waits for
`WaitDrained(e)` on every node of e, then reclaims.

| # | Situation | What happens | Test |
| --- | --- | --- | --- |
| 1 | Query admitted by an entry still on e | Answered entirely under e by e's owners, which keep every list, code and vector until the drain ends | `TestReconfigUnderLoad` (per-epoch counts) |
| 2 | Entry on e forwards to an aggregator that already installed e+1 | The aggregator runs it under e with the entry's groups | `TestLeavingNodeFinishesAdmittedQueries` (the leaving node is that aggregator) |
| 3 | Entry on e+1 forwards to a joining node that has not installed e+1 | Runs under e+1 as told | `TestJoiningNodeAggregatesBeforeInstall` |
| 4 | Leaving node with queries admitted or forwarded under e | Finished under e; the drain waits for them; afterwards it refuses client queries ("not an entry") | `TestLeavingNodeFinishesAdmittedQueries` |
| 5 | Query admitted under e+1 that probes moved lists | Served by the new owner; with `lazy`, RERANK first fetches raw vectors that have not been streamed yet from the old owner | `TestRawVectorsFetchedOnDemand` |
| 6 | Client sends to a node that is not (or no longer) an entry | UNAVAILABLE, the client retries another entry; clients refresh the entry list every 500 ms | load tests (retry counts) |
| 7 | `stop-and-copy` | Entries paused: UNAVAILABLE "admission paused" until resume | `TestReconfigUnderLoad/stop-and-copy` |
| 8 | Joining node before its graph loads | Data node and aggregator only; becomes an entry in a later entry flip | `TestEntriesJoinAsGraphsLoad` |

Conclusions reached in the discussion:

- A node already on e+1 (not new) asked to aggregate a query of e runs it under e; e's owners
  still hold the data because the drain has not ended.
- The drain criterion is the pin counters: on each node, `WaitDrained(e)` returns once a newer
  epoch is installed and no query pinned to e or older remains, and the controller waits on
  every node of e. That means "every query of e has returned at its entry", not "a query of e+1
  has been seen". The latter would need a per-sender table of recent epochs and still could not
  tell when a slow query of e finishes, so it brings nothing.
- Queries pinned to e get no speedup from the scale-out: they run on e's owners. Only queries
  admitted under e+1 use the new node, and with `lazy` the first of those pay on-demand fetches.

Open edges to write up:

- An entry that times out releases its pin while its forwarded aggregation may still be in
  flight. If the forward reaches the aggregator after that node's drain returned, the reclaim can
  evict lists under it: FILTER fails with NOT_RESIDENT. The client already got a timeout, so no
  wrong answer, only wasted work and an error.
- A slow query of e holds the grace period and the reclaim; the drain has no deadline other than
  `stage_timeout`.
- Stale entry lists cost one UNAVAILABLE and a retry per query; U17 proposes keeping the entry
  list outside the epoch table.
- Warm-up cost of `lazy`: RERANK waits for the fetch while holding an engine worker, and
  concurrent misses of one vector are not coalesced. U9's alternative (re-rank at the old owner
  during the warm-up) avoids fetches on the query path.

## 6. Next steps

1. Finish the issue in section 5 (cases, behavior, tests, open edges).
2. Real-data pipeline (section 4.3, item 4).
3. Local experiments (section 4.3).
4. Small code items: fill the whole connection pool during pre-connect and let entries
   pre-connect to aggregators; heap merge in `MergeTopK`; raw path limits listed in
   `engine/README.md` (compact pending-source map, coalescing concurrent fetches).
5. Design topics for the PI: partitioning (U2/U3), chunk storage (U12), aggregator outsourcing
   (U6), rerank policy (U5), PQ compression level, raw-vector warm-up alternatives (U9).

## 7. Working with the user

- Reply in the language of the message (Chinese or English). Code, comments, commit messages and
  GitHub issues are in English; the design doc is in Chinese.
- Go uses only the standard library; never download a Go toolchain; never work around network
  restrictions of the environment.
- Code that reaches an undecided design question returns `design.Undecided("Un", ...)`; when a
  question is decided, update its registry entry and the matching README paragraph.
- End-to-end tests compare every answer with the single-node oracle, exactly.
- When reviewing the user's drafts (issue text, step lists), say what is wrong and give the
  corrected text; diagrams follow the style in section 4.
