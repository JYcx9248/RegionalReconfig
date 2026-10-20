// Package design marks the places where the rtier design is still open.
//
// Placeholders that sit on a code path return an error wrapping ErrUndecided and name one of
// the open questions below, so a run that reaches an undecided path fails loudly instead of
// silently picking an answer. A few questions have no code path yet (U12, U13's retry and
// rollback, U15, U17's list outside the epoch); for those the entry records what the code does
// meanwhile.
// `rtier-client design` prints this list.
//
// U4 is deliberately absent: replicas of hot partitions are out of scope for rtier, which keeps
// one owner per partition and handles skew by placement (a plan reassigns partitions, U3).
// Serverless-style read capacity, how streaming systems serve a hot key, would not relieve a
// disk-bound hot spot: its data is on its owner's SSD. The number stays free so the other IDs
// do not move.
//
// One rule cuts across U1, U2, U15 and U16: no per-node structure should be sized by the whole
// dataset, because scaling out is supposed to make each node's share smaller. At a billion
// vectors on eight nodes the structures that still are cost real DRAM -- the navigation-graph
// replica (32-71 GB, U16), the host shadow of the PQ codes (6.4 GB at m=16, 51 GB at m=128,
// U15) and PQStore::slot_of_ (4 GB, see engine/README.md) -- and of those only the shadow
// shrinks as the cluster grows. The replica's size is outside the current work (U16, decided
// 2026-10-12); of the other two, U15 comes before slot_of_. (The raw-vector presence bitmap is
// one too, at one bit per location: 125 MB at a billion.)
package design

import (
	"errors"
	"fmt"
	"sort"
)

// ErrUndecided is returned by every placeholder.
var ErrUndecided = errors.New("design decision pending")

// Question is one open design question.
type Question struct {
	ID      string
	Title   string
	Options string // options discussed so far
	Where   string // where the placeholder lives
}

// Open lists the open questions, keyed by ID.
var Open = map[string]Question{
	"U1": {"U1", "Partition payload: RAG chunks (PQ codes and raw vectors are decided)",
		"Decided: PQ codes are node-level -- a node keeps one code per vector named by its " +
			"resident lists, however many lists or partitions name it; staging pulls from the " +
			"old owners only the codes the destination lacks (engine/include/fusion/pq_store.h, " +
			"internal/agent/control.go); codes of partitions that move away stay on the old node " +
			"as a cache (valid because vectors are immutable; a deleted vector's code goes with " +
			"its last version, U14) and are dropped only to make room for a later staging, which " +
			"protects the codes it needs. As the design stands nothing is ever dropped: a " +
			"scale-out only takes partitions away from a node and a reversible scale-in only " +
			"returns what it took (U3), so a node's resident set stays what it held at the " +
			"initial cluster size, which fits its budget or the deployment would not have " +
			"started -- which makes --pq-capacity a function of the SMALLEST cluster size, not " +
			"the current one. That holds while the cluster never shrinks below its initial size, " +
			"the nodes that leave are the ones that joined, and placement never hands a " +
			"partition to a node that never held it. Inserts (U14) add to the resident set, so " +
			"with updates the budget needs their headroom too, and a full rebuild's placement or " +
			"a custom plan may hand a node partitions it never held. Spilling evicted codes to " +
			"local SSD (one sequential file per partition, read back instead of pulling from a " +
			"peer) would lift the first two conditions; not implemented, and it would compete " +
			"with re-ranking for the same device. Decided and implemented too: raw vectors are " +
			"addressed by a CANONICAL GLOBAL OFFSET and each node holds a sparse subset of that " +
			"address space. The location (u32: page x vectors_per_page + slot) sits in the " +
			"posting entry next to the vector ID, 8 B per entry (segment version 2, payload " +
			"\"lists+locations\"; version 3 adds the node that holds the vector, U9), and is " +
			"inherited with the list, so it survives every migration untouched and no node needs " +
			"a global vector -> page map: a node keeps its vectors in a local sparse file laid " +
			"out like the whole page file, one presence bit per location, and finds a " +
			"candidate's location through its PQ store, which records it when a partition loads " +
			"(engine/include/fusion/raw_store.h). The transfer unit is a single vector, not a " +
			"page (bucket layout bin-packs list tails, so a page can hold vectors of several " +
			"lists). Within-node dedup falls out of the addressing: the same vector reached " +
			"through two lists names the same offset. The alternative -- keep the map and " +
			"re-address on arrival -- loses on per-node bytes once N > r (~7.3) and would be a " +
			"third dataset-sized global replica a joining node must fetch before it can re-rank. " +
			"The index's page file is read only to bootstrap the initial deployment (a joining " +
			"node runs without it); how the vectors arrive after a migration is U9. Old owners " +
			"keep the vectors of partitions that moved away, like PQ codes. Raw vectors need no " +
			"versions under U14: vectors are immutable, and IDs and locations are only appended, " +
			"never reused. Decided (2026-10-20, for now): raw vectors are kept in full -- no " +
			"disk budget and no eviction, and the pages of deleted vectors stay too, since disk " +
			"is cheap; the space amplification of tail pages shared between lists is accepted " +
			"with it. Open: where RAG chunks live (with U12, discussed last)",
		"engine/include/fusion/raw_store.h, engine/include/fusion/partition.h (locations), " +
			"internal/partitioning/payload.go (chunks)"},
	"U2": {"U2", "List -> partition assignment (GP on the navigation graph is decided) and the number of partitions",
		"the objective is r(P) = sum over partitions of |its vector set| / |their union|, the " +
			"cross-partition overlap (1 without partitioning, approaching the list-level r ~ 7.3 " +
			"under random grouping). It sets the PQ-code redundancy, the per-node residency, and " +
			"whether PQStore::slot_of_ is worth replacing (see engine/README.md); the number of " +
			"partitions follows from bytes per partition and fan-out. Decided (2026-10-19): the " +
			"initial split balances the number of posting lists per partition -- and so per node " +
			"under the even placement -- not their postings: list counts change only through a major " +
			"epoch's splits and merges, list lengths with every update batch, and those splits and " +
			"merges keep the lengths close while a full rebuild starts over. The partitioner is GP " +
			"(Gottesbueren et al., VLDB '25) on the list heads over the navigation graph's level-0 " +
			"edges -- all of them: the build's own graph, so no kNN graph is built -- with unit " +
			"weights as in the paper (blocks within 5%): make_gp_assignment.py --points heads " +
			"--graph-source index (gp/gp_kaminpar.cpp runs KaMinPar). BIGANN-1M, 64 partitions, " +
			"nprobe 64: a query touches 7.63 partitions, against 8.03 with the paper's 10-NN graph of " +
			"the heads and 8.88 for spatial bisection; owners at 8 nodes 2.96 / 3.03 / 3.15, at 16 " +
			"nodes 4.20 / 4.35 / 4.62; the graph is read in 0.2 s, the kNN graph took 65 s for 101K " +
			"heads (each head's first 10 links alone do no better than spatial: hnswlib does not keep " +
			"them by distance). KaMinPar numbers its blocks along its recursive bisection, so " +
			"consecutive partitions are neighbors and contiguous ranges suit the even placement (U3). " +
			"The cost of balancing counts: list lengths at the build are heavy-tailed (mean 72, p99 " +
			"181, max 543 -- the clustering's leaves hold 1-15 vectors, replication does the rest), " +
			"so partitions differ in postings by up to 1.26x the average and nodes by 1.04 / 1.07 / " +
			"1.15 at 4 / 8 / 16 nodes; weighting each head by its postings (--weights postings) would " +
			"bring both to 1.05 at no loss in cut quality. The fullest node's share of the vectors, " +
			"its PQ codes, stays below spatial's (0.235 against 0.252 at 8 nodes). Other generators " +
			"(test-only): contiguous ranges of list IDs (make_range_assignment.py), spatial bisection " +
			"of the heads (make_spatial_assignment.py), and the paper's own variant, GP on the " +
			"vectors over a 10-NN graph of the dataset (--points vectors: a list goes to the " +
			"plurality block of its postings, which keeps only 58% of the postings with their block). " +
			"Open: the number of partitions; GP on real datasets. Set aside (2026-10-19), since it " +
			"differs from dataset to dataset: the posting imbalance itself -- a list's length is " +
			"about how many vectors count its head among their 8 nearest (90% of the vectors reach " +
			"the cap of 8 replicas), which varies with the data's geometry (hubness), and the levers " +
			"are the build's (SPANN's RNG pruning of replicas, which Assign does not reproduce yet; a " +
			"length cap at the build; the replica count), not the partitioning",
		"internal/partitioning/partitioner.go, engine/tools/rtier_segment.cpp, scripts/testing/ " +
			"(make_range_assignment.py, make_spatial_assignment.py, make_gp_assignment.py, " +
			"gp/gp_kaminpar.cpp)"},
	"U3": {"U3", "Placement objective beyond even partition counts (reversibility and the graph choice are decided)",
		"Decided: even partition counts; a scale-out moves partitions only from existing to new " +
			"nodes, each donor giving its excess, and a scale-in returns every partition it can " +
			"to a node that has owned it before (a matching), so scaling out and back in " +
			"restores the placement and the cached PQ codes make the scale-in free (Koala's even " +
			"policy is not reversible -- after 4 -> 8 -> 4 none of the 32 moved partitions is " +
			"home again). Which partitions a donor gives, and where one that cannot go home " +
			"goes, follows the partition graph -- navigation-graph edges between the partitions' " +
			"heads, read off the index, no queries -- so that the nodes cut as little of it as " +
			"possible (placement \"even-reversible-graph\", the default; \"even-reversible\" and " +
			"\"even-reversible-blocks\" give every donor's tail). BIGANN-10M, 64 spatial " +
			"partitions, nprobe 64: 2 -> 3 gives 1.71 owners per query against 1.83 for tails, " +
			"one node at a time to 8 gives 3.31 against 3.81, doubling does as well as whole " +
			"blocks (2 -> 4: 1.81 both; 2 -> 4 -> 8: 2.69 against 2.76); end to end under a " +
			"per-node read budget the capacity is the same, with 2-7% less CPU per query and " +
			"5-8% fewer PQ codes staged (neighboring partitions share replicated vectors). " +
			"Decided (2026-10-19): the initial placement stays contiguous ranges of partition " +
			"IDs, which the spatial and GP generators number along their bisection: close to the " +
			"best at power-of-two sizes, a small gap otherwise (3 nodes: 1.82 owners per query, " +
			"against 1.71 when 2 -> 3 picks by the graph). Accepted (2026-10-12): the plans the " +
			"policies make move no data between old nodes (a scale-out old -> new only, a " +
			"scale-in leaving -> remaining only; a custom plan may move anything), so layouts " +
			"reached one node at a time nest -- each old node's region only shrinks -- and drift " +
			"from a fresh even split: 3.31 owners against 2.76 at 8 nodes (BIGANN-1M, GP: 3.56 " +
			"against 2.99, 18-23% from 4 to 8 nodes). The partition graph narrows the gap and " +
			"doubling avoids it; partitioning (U2) cannot close it. To measure (offline counts " +
			"suffice): whether the gap closes in larger clusters -- with nprobe and the " +
			"partition count fixed it must as N approaches the partition count, since a query's " +
			"owners are bounded by the partitions it probes whatever the placement, but not " +
			"necessarily when the partition count grows with N. Decided (2026-10-13): skew is " +
			"not the policy's. It is left to custom plans (rtier-client rescale -plan), a " +
			"rebalance at the same node count included, which the controller carries out " +
			"whatever the policy: we show that any plan runs without disruption and design no " +
			"skew policy, and the skew experiment runs last, on the final workload. Open: weight " +
			"by per-tier bytes (HBM is the binding tier; with partitions of unequal size donors " +
			"send unequal bytes) -- with node-level PQ codes a node's footprint depends on which " +
			"partitions it holds together (boundary replication), and staging must fit each " +
			"node's budget (--pq-capacity); Decided (2026-10-20) for which nodes leave and join: " +
			"a node removed by a scale-in is shut down, so a later scale-out brings a fresh node " +
			"and a node coming back is not a case we handle; the highest IDs leave (the latest " +
			"joiners, so partitions go home to their former owners) and any idle node joins " +
			"(lowest IDs first)",
		"internal/placement/graph.go (GraphPolicy), internal/partitioning/graph.go " +
			"(PartitionGraph), internal/placement/policy.go (EvenPolicy, ReversiblePolicy, " +
			"BlockPolicy), internal/placement/weighted.go, internal/controller/reconfig.go " +
			"(rescale)"},
	"U5": {"U5", "Global top-n under fixed-n re-ranking",
		"two-phase is implemented (every owner filters for its own top-n, the aggregator merges " +
			"the global top-n and re-ranks each candidate at the owner that reported it): " +
			"exactly the single-node answer, for a second round trip. Decided (2026-10-20, with " +
			"the PI, for now): two-phase, so every answer equals the single-node answer; a quota " +
			"of n * its share for each owner, re-ranked in one round trip (an answer that can " +
			"differ from the global top-n), is not pursued",
		"internal/query/query.go (TwoPhase implemented; ProportionalQuota placeholder)"},
	"U6": {"U6", "Outsourcing the per-query aggregator to another node",
		"Decided in the steady state: the entry chooses after navigating, among the nodes the " +
			"query has to reach -- the owners of its probed lists (selector \"owner\", the " +
			"default): itself when it owns some of them; the owner when one node holds them all " +
			"(one network round trip instead of two, and the PQ candidates never leave that " +
			"node); itself otherwise, because forwarding then adds a hop to save one of several " +
			"parallel sub-queries and piles aggregation onto the owners of hot partitions. " +
			"Decided where it applies: the only window in which the nodes that may aggregate " +
			"outnumber the nodes that may take client queries is a scale-out, while a new node's " +
			"navigation graph is still arriving. Selector \"warmup\" hands the scatter/gather to " +
			"those nodes and aggregates locally otherwise. Decided (2026-10-20, for now): the " +
			"aggregator is not outsourced -- \"owner\" stays the default, \"warmup\" stays an option " +
			"that is not the default, and a load-aware policy (selector \"outsource\") is not " +
			"pursued. Outsourcing would save little anyway: the aggregator's own work is the " +
			"fan-out and the merge, while the entry keeps the navigation, the expensive part and " +
			"the part that needs the graph (measured small on a CPU-only run: navigate 160 us of " +
			"a 445 us server-side query). Note that outsourcing and letting the new node take " +
			"queries and forward the navigation (U16's \"stream\") split the work the same way, " +
			"from opposite sides",
		"internal/query/selector.go (OwnerSelector, WarmupSelector implemented; " +
			"OutsourceSelector placeholder), internal/agent/serve.go (entry)"},
	"U7": {"U7", "Sub-queries that arrive with a stale epoch",
		"Decided (2026-10-20): not handled -- it takes a failure, out of scope with the others " +
			"(U13). An entry keeps its query's epoch pinned until its forward returns -- a " +
			"client that gives up does not end it, since the agent serves queries under its own " +
			"lifetime -- and an aggregator pins the epoch of a forwarded aggregation as soon as " +
			"it reads it, so the grace period covers every query of the epoch. A stale sub-query " +
			"needs an entry whose forward fails (the query client's 30 s timeout) before the " +
			"aggregator has read it, and that then reaches the aggregator after its drain " +
			"returned: only with a network or process failure. No wrong answer can come of it, " +
			"only wasted work and an error: a sub-query that meets a partition no longer " +
			"resident fails its query with a plain error (no placeholder). Not pursued: serving " +
			"it while the data is still resident, or forwarding it to the new owner",
		"internal/query/query.go (Aggregator.Run, NotResident), internal/agent/serve.go (entry, " +
			"forwarded), internal/query/wire.go (Client.Timeout)"},
	"U8": {"U8", "Adaptive rate control for background transfer",
		"a token bucket on a separate connection (implemented, fixed rate) with two priority " +
			"classes on the sender (implemented): data -- segments and PQ codes, what a node " +
			"needs to own partitions -- and background -- the navigation graph (graph_priority), " +
			"which only takes tokens no data sender is waiting for. With lazy (the default) the " +
			"graph is the only background transfer, while new entries join a scale-out. Only " +
			"what goes through the agents is paced: the raw-vector paths between data nodes are " +
			"not -- a query's fetch (RAW_GET, on the query path by design), a scale-in's drain " +
			"and the baselines' streams on the default data-node path (RAW_PULL) -- and only the " +
			"source's read budget bounds them. On the agent path (raw_path agent, the default " +
			"only when the agents are paced) the drain and lazy-stream's stream after the flip " +
			"take raw_priority (background by default, so the stream shares the class equally " +
			"with a graph still arriving: a baseline's tuning, not a protocol question), and the " +
			"eager baselines' copy before the flip takes the data class. Decided (2026-10-20, " +
			"for now): the mechanism stays as it is. The aim is that a reconfiguration not cut " +
			"the capacity much, and the current one is taken as enough for that: a fixed rate " +
			"(transfer_rate, set per experiment, unpaced by default) with the two classes, no " +
			"rate adapted to foreground latency, and no byte rate for the drain, whose pulls " +
			"cluster.raw bounds (batch_bytes per request, streams in flight per source, shared " +
			"with the baselines' streams) besides the source's read budget",
		"internal/transfer/adapter.go, internal/transfer/tokenbucket.go (classes), " +
			"internal/agent/control.go (rawViaNode, pullRaw), internal/controller/reconfig.go " +
			"(stageRaw, drainRaw)"},
	"U9": {"U9", "Lazy fetch of raw vectors (and chunks) after a new node is online",
		"Decided: a node goes online (the flip) only once it holds every posting list and PQ " +
			"code of its partitions, so it can filter at once; a raw vector moves only when a " +
			"query first needs it (protocol \"lazy\", the default) -- the point of being lazy is " +
			"that queries touch a small working set of what moved. Every posting records the " +
			"holder of its vector, the node that stores it (u16 node ID, segment version 3), so " +
			"each key (list, vector) has one current holder: the list's owner if it holds the " +
			"vector, else the node the posting names. Only the owner's copy of a list is valid: " +
			"a partition moves as its owner saves it from memory, holders included " +
			"(SAVE_PARTITION), and the next owner makes the postings of the vectors it holds " +
			"name itself. A RERANK that meets a vector not there fetches it from the holder that " +
			"a posting in one of the query's lists on that node records, data node to data node " +
			"(RAW_GET, one hop: a holder has the vector; data nodes know each other by node ID, " +
			"SET_NODES), and makes that one posting name itself -- other lists naming the vector " +
			"keep their holder, which still has it; queries that miss the same vector meanwhile " +
			"wait for that one fetch. The RERANK waits for the fetch only: fetched vectors are " +
			"re-ranked from memory and a background writer installs them with whole-page direct " +
			"writes. It waits holding no engine worker (2026-10-20): the data node fetches on " +
			"the request's connection thread and takes a worker only once the vectors are there, " +
			"so the node's FILTERs and other RERANKs keep the workers meanwhile. A scale-out " +
			"does not wait for raw vectors. A scale-in drains the nodes that leave first " +
			"(DrainRaw: each remaining node takes over the vectors its postings say only a " +
			"leaving node holds, RAW_HELD_BY then RAW_PULL; the rescale waits for it), so they " +
			"can go offline when it returns. Holders stay valid because no node drops a raw " +
			"vector it has held (vectors are immutable, and even a deleted vector's page is not " +
			"freed); a disk budget would need the same hand-off first. Baselines: \"lazy-stream\" " +
			"also streams the rest in after the flip (StageRaw at raw_priority; RAW_MISSING " +
			"groups each missing vector by its holder; the rescale waits for it); " +
			"\"copy-then-flip\" streams before the flip, \"stop-and-copy\" while paused. Decided " +
			"(2026-10-16): the fetch stays on the query path -- taking it off (re-ranking a " +
			"missing candidate at its holder, fetching in the background) is not a goal -- and " +
			"the queries' locality is taken as given (U11). Decided (2026-10-20): a fetch brings " +
			"only the vectors its query needs, and nothing more -- no page or list read ahead -- " +
			"with one request per holder, every holder at once. Under the graph placement a new " +
			"node takes partitions from several donors, not one contiguous range, so a query's " +
			"misses often lie at several holders, and the RERANK waits for the slowest request " +
			"rather than their sum. Vectors move beyond what queries ask for only at a scale-in, " +
			"where the drain hands over what only the leaving nodes hold (the baselines stream " +
			"by design). Open: chunks (U12)",
		"internal/controller/reconfig.go (copyThenFlip, stageRaw, drainRaw), " +
			"internal/agent/control.go (StageRaw, DrainRaw), engine/src/node_engine.cpp " +
			"(EnsureRawInLists, FetchForQuery, RawHeldBy, RawPull), engine/node/server.cpp " +
			"(PeerPool, RERANK)"},
	"U10": {"U10", "Epoch store for the atomic flip",
		"an etcd transaction (as discussed) vs. the controller-local compare-and-swap that the " +
			"single-machine prototype uses. What happens when the controller fails mid-flip is out of " +
			"scope with the other failures (U13, decided 2026-10-16)",
		"internal/epoch/store.go"},
	"U11": {"U11", "Workload model",
		"Poisson open-loop arrivals (implemented) vs. replaying SemDN-style bursty, task-coherent " +
			"query streams. Decided (2026-10-16): queries are taken to have locality, which is what " +
			"makes the lazy protocol worth it; without any, lazy still moves at most what " +
			"copy-then-flip copies, since a node fetches each vector at most once and concurrent " +
			"misses of one vector wait for one fetch (Wikipedia 2 -> 4: 3.0 GB fetched against 51.2 " +
			"GB copied). Open: a generator or a real query log with locality for the experiments (the " +
			"SemDN-style generator is a placeholder)",
		"cmd/rtier-loadgen/arrivals.go"},
	"U12": {"U12", "Vector ID -> RAG chunk ID mapping returned to the caller",
		"the tier returns vector IDs (no code path maps them yet); the chunk store and the " +
			"fetch path are not designed",
		"internal/query/query.go (Result)"},
	"U13": {"U13", "Failures during a reconfiguration (node or controller)",
		"out of scope (decided 2026-10-16): this work handles no failure -- retrying the failed " +
			"step, rolling back, or finishing without the failed node are future work. As it stands, " +
			"a failure before the flip drops the partitions already copied and keeps the old epoch; " +
			"after the flip the protocol stops and keeps every old copy (safe, wasteful)",
		"internal/controller/reconfig.go (copyThenFlip)"},
	"U14": {"U14", "Dataset updates",
		"vectors are immutable (an update deletes a vector and inserts it again under a new ID) " +
			"and the PQ codebook is fixed, so cached PQ codes and raw vectors on former owners " +
			"stay valid; what remains is maintaining posting lists and the graph. The discussion " +
			"that led here (periodic batches, each a content epoch) is in " +
			"docs/drafts/dataset-updates-discussion.md; where it differs (it rebuilt the whole " +
			"graph for a major epoch and triggered one by a health check), this entry is the " +
			"decision. In code, minor batches (inserts and deletes) and major epochs (splits and " +
			"merges). Partition versions: the epoch table names a version of each partition's " +
			"lists, a data node holds several versions of a partition side by side, and " +
			"FILTER/RERANK name the one their epoch sees. Inserts: data nodes leave room for " +
			"inserted IDs and locations (rtier_node --max-vectors, --max-pages; both only " +
			"appended) and for longer lists (--max-list-len); ASSIGN routes a vector by the " +
			"build's own rule and gives its PQ code; RAW_ADD serves inserted vectors from memory " +
			"and writes them after fetched ones, never while paused (PAUSE_UPDATES), within " +
			"--update-write-mbps -- writes are not charged to the read budget yet. Batches run " +
			"only with the controller's allow_updates (off by default: a static experiment " +
			"refuses them; an experiment configuration's updates section turns it on, leaves " +
			"room on the data nodes and requires an explicit PQ budget, since the default budget " +
			"is the whole ID space, headroom included). The update controller (internal/update, " +
			"rtier-updater) takes the controller's lease (BeginUpdate: no reconfiguration and no " +
			"other batch until the commit or the abort, or until update_lease expires; the " +
			"agents first release PQ codes left staged by a staging that never ended), routes " +
			"the inserts with its own router (an rtier_node with the graph and codebook only, " +
			"never a serving node), checks the owners' room, sends the PQ codes and raw vectors " +
			"to the owners, and has every data node stage the next version of the partitions the " +
			"batch changes (STAGE_VERSIONS: inserts appended, deleted IDs dropped; deletes are " +
			"broadcast and each node finds its copies); CommitUpdate checks the staged versions, " +
			"flips (all entries at once), waits out the grace period, and reclaims: every node " +
			"frees the deleted vectors' PQ codes (FORGET, deleted: copies nothing names go at " +
			"once, the others are marked) and then the owners drop the replaced versions, so a " +
			"deleted vector's code goes with its last version and never becomes a cached one -- " +
			"a deleted vector is no cache. A staging that fails is undone before the lease is " +
			"given up; a commit that fails before the flip drops the staged versions. A " +
			"partition at a later version moves with its current lists: its owner writes them as " +
			"a segment file for the move (SAVE_PARTITION). Major epochs " +
			"(internal/update/major.go) split and merge posting lists as a batch with its own " +
			"lease, decided on PQ codes: a split is a balanced 2-means of a list's members, each " +
			"child's head the member closest to its mean, both children in the parent's " +
			"partition; a merge sends each member to the list of its nearest live head on the " +
			"edited graph, whatever its partition -- a posting that changes owner moves as " +
			"(vector ID, location, holder) with its PQ code, its raw vector following on demand. " +
			"The update controller's router edits its graph (new heads in, removed heads marked " +
			"deleted, routing points but never a result) and exports the bytes that changed as a " +
			"patch (fusion/graph_patch.h). No node holds two versions of the graph: every entry " +
			"stages the patch next to its graph, queries of the epoch after the flip search " +
			"through it (step by step the patched graph's search), and after the grace period " +
			"every entry keeps it as a file -- a later entry's graph transfer brings the build's " +
			"graph and the patches -- and merges it in place, before the lease ends. The epoch " +
			"table names the graph version and the lists major epochs created (list IDs, like " +
			"vector IDs, are never reused); data nodes leave room for heads and lists " +
			"(rtier_node --graph-headroom, 20% by default). Trigger (decided 2026-10-16): only " +
			"minor batches come from outside; the owners report the lengths of the lists a batch " +
			"changes, and MajorDue splits the lists over one length and merges the ones under " +
			"another (rtier-updater -split-len, -merge-len; -split, -merge by hand). Full " +
			"rebuild (internal/controller/rebuild.go, decided 2026-10-19): the cluster moves to " +
			"another generation of the index -- every head, list and partition and the graph -- " +
			"built outside the system with the same codebook, vector IDs and raw-vector " +
			"locations and handed over as files (a source interface; fusion_build reuses the " +
			"codebook and page layout it finds in its output directory), so the PQ codes and raw " +
			"vectors every node holds stay valid and no node ever holds two generations' lists " +
			"or graphs. Spares (idle nodes, as many as the data nodes by default) take every " +
			"partition by a lazy reconfiguration; the old nodes stay members that only serve raw " +
			"vectors (not drained), lose the entry role once the spares have the graph, and " +
			"switch to the next generation (SWITCH_GENERATION keeps their PQ codes and raw " +
			"vectors); its partitions are placed by where their raw vectors are (presence " +
			"bitmaps, a min-cost flow under the even policies' balance: placement.Affinity), " +
			"every posting whose vector its owner lacks naming an old node that holds it (one " +
			"hop); the old nodes load the partitions (PQ codes from the generation's index) and " +
			"graph, replay the recent queries the entries kept under the draft epoch (what " +
			"re-ranking fetches stays: the warm-up), and take over in one flip; after the grace " +
			"period the spares, holding copies only, move to the next generation too and are " +
			"idle. The epoch table names its generation (and its directories); a node that joins " +
			"later and the update controller's router follow it. No update batch or " +
			"reconfiguration runs meanwhile (assumed). Baselines: random placement, no warm-up, " +
			"stop-the-world (admission paused while every node switches in place). Not " +
			"implemented: growing the ID and location spaces at an epoch boundary (agreed " +
			"2026-10-13: preallocated with headroom and grown at an epoch boundary, on the " +
			"update controller's order before a batch); the response to a batch over budget " +
			"(agreed 2026-10-13: a partition over about 1.5x the average size splits in two by a " +
			"cut of the partition graph, both halves staying on its owner, and a scale-in below " +
			"the size that fits is refused; whether to scale out first is open -- a batch over " +
			"budget now fails before anything is sent); reassigning the vectors of neighboring " +
			"lists in a split (SPFresh's LIRE, a recall matter); the full rebuild's trigger " +
			"(manual only: rtier-client rebuild; the probe-set check is design only), a " +
			"generation that includes inserted vectors (the builder would need their IDs and " +
			"locations), a priority between the warm-up's fetches and the spares' (only the " +
			"warm-up's pace is set); a graph patch over one frame (256 MB); the decision log and " +
			"replay (the pages of deleted vectors are kept: U1). An inserted vector stays with " +
			"the owner that added it when its partition moves on (the postings name that node as " +
			"its holder), and a node that leaves is drained first (U9). Open: versions staged " +
			"under a lease that expired stay on their owner until a later batch replaces them or " +
			"the partition moves",
		"internal/epoch/table.go (Versions, Graph, Lists, Generation), " +
			"engine/include/fusion/node_engine.h (partition versions, Assign, RawAdd, " +
			"StageVersions, graph patches, SwitchGeneration), " +
			"engine/include/fusion/graph_patch.h, internal/update (batch.go, major.go), " +
			"internal/controller/update.go, internal/controller/rebuild.go, " +
			"internal/placement/affinity.go, internal/controller/reconfig.go (copyThenFlip, " +
			"drainRaw), engine/include/fusion/pq_store.h (cached codes, Forget), " +
			"cmd/rtier-updater, scripts/run_local.py (updates)"},
	"U16": {"U16", "The navigation graph as a global replica: size and transfer mode",
		"every entry node needs the whole graph over the list heads, and a node that joins " +
			"starts empty -- pre-staging it on spare nodes is not allowed, because a real " +
			"deployment provisions a node at scale-out time (an experiment may pre-register " +
			"nodes, but pre-loading the graph would hide the largest transfer of the whole " +
			"reconfiguration). Measured 308 B per head at dim 32 with heads at 10.3% of the " +
			"vectors, so roughly 70 MB per million vectors and tens of GB at billion scale, " +
			"larger than the PQ codes a new node pulls. Decided (2026-10-12): the replica's size " +
			"is outside the current work -- the paper states the sizes as an assumption (7 GB " +
			"for Wikipedia's 15.5M x 1024 vectors, 32-71 GB at a billion x 128) and leaves " +
			"smaller routing state to future work: head vectors in the dataset's dtype, a lower " +
			"head ratio (fusion_build targets 10%), or replicating only the upper HNSW levels " +
			"and partitioning level 0 (which makes navigation distributed and changes recall). " +
			"The user's claim (2026-10-16): no node ever holds two copies of the graph -- a " +
			"major epoch patches it in place (U14) and a full rebuild moves the roles to spares. " +
			"Decided (2026-10-20): the transfer mode stays \"eager\" -- an entry without the whole " +
			"graph serves no purpose, so each new node takes the entry role in a flip as soon as " +
			"its graph has loaded (batched per entry_flip_interval); the \"stream\" mode (take the " +
			"role at once, forward NAVIGATE, fill in behind) is not pursued. Decided for the " +
			"transfer itself: each new entry's source is assigned when the reconfiguration " +
			"starts, round-robin over the live entries with a cursor that carries over between " +
			"reconfigurations (instead of every new node pulling from the first entry), and the " +
			"pull runs at background priority on its source (graph_priority): FILTER and RERANK " +
			"need posting lists and PQ codes, not the graph, and navigation is the smallest " +
			"share of a query that any existing entry can do meanwhile -- so the graph only gets " +
			"the bandwidth the segment and PQ transfers leave unused and never delays the flip. " +
			"Pairs with U6's warmup selector, which moves aggregation to the new node before its " +
			"graph arrives. After major epochs (U14), the transfer brings the build's graph and " +
			"then, from the same source, the patch of every later graph version, merged in order " +
			"before the node takes the role",
		"internal/controller/reconfig.go (graphSources, StageGraph, promoteEntries), " +
			"internal/transfer/tokenbucket.go (classes), engine/src/node_engine.cpp (LoadGraph)"},
	"U17": {"U17", "Where the client-facing availability list lives",
		"today the entry set is a field of the epoch table, so \"this node finished loading its " +
			"graph\" -- a local event that has nothing to do with who owns what -- costs an epoch " +
			"flip (batched: one per entry_flip_interval for the nodes ready by then, running " +
			"alongside the grace period, so no node waits for the slowest graph), and the " +
			"rescale still blocks until every graph transfer has ended. The alternative, not " +
			"pursued for now: the coordinator keeps the available list outside the epoch; a node " +
			"is added once its graph is complete, and until then it only serves work forwarded " +
			"to it (sub-queries, and aggregation under U6); a scale-in removes a node from the " +
			"list first, drains, then flips. The rule this follows: a field belongs in the epoch " +
			"table only if it changes exactly when the node set changes (ownership and the " +
			"aggregator set do; entry availability does not), because what must be " +
			"epoch-consistent is what decides where the data is and who must be drained -- a " +
			"query sent to a node that should not have it costs one UNAVAILABLE and a retry. " +
			"Decided (2026-10-19), how a client learns that its list changed: it reads the list " +
			"every 500 ms, and at once when an entry refuses a query under a newer epoch than " +
			"the list's (a refusal carries the entry's epoch). That entry has left the entry " +
			"set, so the retry goes to the entries of that epoch, and the queries that find the " +
			"list stale together share one read. A refusal under the list's epoch or an earlier " +
			"one -- paused admission, an entry that has not installed the epoch yet -- reads " +
			"nothing: a fresh list would be the same. Wherever the list lives, it needs a " +
			"version for this. Decided (2026-10-20, for now): the entry set stays in the epoch " +
			"table, and the rescale returns once every new node is an entry. Moving the list out " +
			"would not let a node work earlier: a new node owns its partitions and aggregates " +
			"from the data flip on, its graph arriving apart from its lists and PQ codes, at " +
			"background priority (U16), and it takes client queries once the graph is there -- " +
			"its lists and PQ codes have been there since the flip. What it would change is when " +
			"the rescale counts as done and how many flips a scale-out takes; the reply records " +
			"the data flip and each entry flip apart (phases, entry_flips), so an experiment can " +
			"report both. No code path for the list outside the epoch",
		"internal/epoch/table.go (Table.Entries), internal/agent/serve.go (entry), " +
			"internal/controller/reconfig.go (promoteEntries), cmd/rtier-loadgen/main.go " +
			"(entries.catchUp)"},
	"U15": {"U15", "The host copy of the PQ codes next to the GPU's",
		"with the GPU backend a node holds every code twice: PQStore keeps the authoritative " +
			"copy in host memory (capacity x m) and the filter a second one in device memory, " +
			"filled by StoreCodes as partitions load. The host copy answers a peer's PQ_GET and " +
			"carries the bookkeeping without touching the GPU; dropping it would save that DRAM " +
			"but make a migration source read its codes back over D2H, competing with its own " +
			"query kernels. No code path yet: both copies exist, and the CPU backend reads the " +
			"host one directly (no second copy)",
		"engine/include/fusion/pq_store.h (codes_), engine/src/node_engine.cpp (filter creation), " +
			"engine/src/gpu_filter.cu (d_codes_, StoreCodes)"},
}

// Undecided returns an error for a placeholder tied to open question id.
func Undecided(id, what string) error {
	q, ok := Open[id]
	if !ok {
		return fmt.Errorf("%w: %s", ErrUndecided, what)
	}
	return fmt.Errorf("%w [%s %s]: %s", ErrUndecided, q.ID, q.Title, what)
}

// IDs returns the open question IDs in order (U1, U2, ..., U10, ...).
func IDs() []string {
	ids := make([]string, 0, len(Open))
	for id := range Open {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool {
		if len(ids[i]) != len(ids[j]) {
			return len(ids[i]) < len(ids[j])
		}
		return ids[i] < ids[j]
	})
	return ids
}
