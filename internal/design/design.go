// Package design marks the places where the rtier design is still open.
//
// Placeholders that sit on a code path return an error wrapping ErrUndecided and name one of
// the open questions below, so a run that reaches an undecided path fails loudly instead of
// silently picking an answer. A few questions have no code path yet (U12, U13's retry and
// rollback, U14, U15, U17); for those the entry records what the code does meanwhile.
// `rtier-client design` prints this list.
//
// U4 is deliberately absent: replicas of hot partitions are out of scope for rtier. Elastic
// read capacity for a hot key is what streaming systems solve with serverless-style instances,
// and rtier keeps one owner per partition. The number stays free so the other IDs do not move.
//
// One rule cuts across U1, U2, U15 and U16: no per-node structure should be sized by the whole
// dataset, because scaling out is supposed to make each node's share smaller. At a billion
// vectors on eight nodes the structures that still are cost real DRAM -- the navigation-graph
// replica (32-71 GB, U16), the host shadow of the PQ codes (6.4 GB at m=16, 51 GB at m=128,
// U15) and PQStore::slot_of_ (4 GB, see engine/README.md) -- and of those only the shadow
// shrinks as the cluster grows. That ordering is the work order: U16, then U15, then slot_of_.
// (The raw-vector presence bitmap is one too, at one bit per location: 125 MB at a billion.)
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
			"resident lists, however many lists or partitions name it; staging pulls from the old " +
			"owners only the codes the destination lacks (engine/include/fusion/pq_store.h, " +
			"internal/agent/control.go); codes of partitions that move away stay on the old node " +
			"as a cache (static dataset) and are dropped only to make room for a later staging, " +
			"which protects the codes it needs. As the design stands nothing is ever dropped: a " +
			"scale-out only takes partitions away from a node and a reversible scale-in only " +
			"returns what it took (U3), so a node's resident set stays what it held at the initial " +
			"cluster size, which fits its budget or the deployment would not have started -- which " +
			"makes --pq-capacity a function of the SMALLEST cluster size, not the current one. That " +
			"holds while the cluster never shrinks below its initial size, the nodes that leave are " +
			"the ones that joined, and placement never hands a partition to a node that never held " +
			"it. Spilling evicted codes to local SSD (one sequential file per partition, read back " +
			"instead of pulling from a peer) would lift the first two conditions; not implemented, " +
			"and it would compete with re-ranking for the same device. " +
			"Decided and implemented too: raw vectors are addressed by a CANONICAL GLOBAL OFFSET and " +
			"each node holds a sparse subset of that address space. The location (u32: page x " +
			"vectors_per_page + slot) sits in the posting entry next to the vector ID, 8 B per entry " +
			"(segment version 2, payload \"lists+locations\"; version 3 adds the node that holds " +
			"the vector, U9), and is inherited with the list, so it " +
			"survives every migration untouched and no node needs a global vector -> page map: a " +
			"node keeps its vectors in a local sparse file laid out like the whole page file, one " +
			"presence bit per location, and finds a candidate's location through its PQ store, " +
			"which records it when a partition loads (engine/include/fusion/raw_store.h). The " +
			"transfer unit is a single vector, not a page (bucket layout bin-packs list tails, so a " +
			"page can hold vectors of several lists). Within-node dedup falls out of the addressing: " +
			"the same vector reached through two lists names the same offset. The alternative -- " +
			"keep the map and re-address on arrival -- loses on per-node bytes once N > r (~7.3) and " +
			"would be a third dataset-sized global replica a joining node must fetch before it can " +
			"re-rank. The index's page file is read only to bootstrap the initial deployment (a " +
			"joining node runs without it); how the vectors arrive after a migration is U9. Old " +
			"owners keep the vectors of partitions that moved away, like PQ codes. Open: the space " +
			"amplification of tail pages shared between lists; a disk budget and eviction for raw " +
			"vectors (none yet: a node keeps every vector it has held); versioning under U14; where " +
			"RAG chunks live (with U12)",
		"engine/include/fusion/raw_store.h, engine/include/fusion/partition.h (locations), " +
			"internal/partitioning/payload.go (chunks)"},
	"U2": {"U2", "List -> partition assignment and the number of partitions",
		"the objective is r(P) = sum over partitions of |its vector set| / |their union|, the " +
			"cross-partition overlap (1 without partitioning, approaching the list-level r ~ 7.3 " +
			"under random grouping). It sets the PQ-code redundancy, the per-node residency, and " +
			"whether PQStore::slot_of_ is worth replacing (see engine/README.md). Candidates: group " +
			"lists by navigation-graph locality, by centroid k-means, or by hash (the baseline); the " +
			"partition count follows from bytes per partition and fan-out. Measure r(P), recall and " +
			"per-node residency together on real datasets",
		"internal/partitioning/partitioner.go, engine/tools/rtier_segment.cpp"},
	"U3": {"U3", "Placement objective beyond even partition counts (reversibility and the graph choice are decided)",
		"Decided: even partition counts; a scale-out moves partitions only from existing to new " +
			"nodes, each donor giving its excess, and a scale-in returns every partition it can to " +
			"a node that has owned it before (a matching), so scaling out and back in restores the " +
			"placement and the cached PQ codes make the scale-in free (Koala's even policy is not " +
			"reversible -- after 4 -> 8 -> 4 none of the 32 moved partitions is home again). Which " +
			"partitions a donor gives, and where one that cannot go home goes, follows the " +
			"partition graph -- navigation-graph edges between the partitions' heads, read off the " +
			"index, no queries -- so that the nodes cut as little of it as possible (placement " +
			"\"even-reversible-graph\", the default; \"even-reversible\" and \"even-reversible-blocks\" " +
			"give every donor's tail). BIGANN-10M, 64 spatial partitions, nprobe 64: 2 -> 3 gives " +
			"1.71 owners per query against 1.83 for tails, one node at a time to 8 gives 3.31 " +
			"against 3.81, doubling does as well as whole blocks (2 -> 4: 1.81 both; 2 -> 4 -> 8: " +
			"2.69 against 2.76); end to end under a per-node read budget the capacity is the same, " +
			"with 2-7% less CPU per query and 5-8% fewer PQ codes staged (neighboring partitions " +
			"share replicated vectors). Open: the initial placement is still contiguous ranges, " +
			"poor at sizes that are not powers of two (3 nodes: 1.82 owners, 4 nodes: 1.81); " +
			"layouts reached by single-node steps nest -- with no data exchanged between old " +
			"nodes each node's region only shrinks -- and drift from a fresh even split (3.31 " +
			"against 2.76 at 8 nodes), which only bigger steps (doubling) avoid; weight by per-tier " +
			"bytes (HBM is the binding tier; with partitions of unequal size donors send unequal " +
			"bytes) and by access heat -- skew is left to custom plans (rtier-client rescale " +
			"-plan), which the controller carries out whatever the policy; with node-level PQ " +
			"codes a node's footprint depends on which partitions it holds together (boundary " +
			"replication), and staging must fit each node's budget (--pq-capacity); rebalance " +
			"without changing the node count; which nodes leave on scale-in (now: highest IDs) and " +
			"which idle nodes join (now: lowest IDs) -- the cache invariant in U1 needs the node " +
			"that left to be the one that comes back, so the join rule is part of that decision",
		"internal/placement/graph.go (GraphPolicy), internal/partitioning/graph.go (PartitionGraph), " +
			"internal/placement/policy.go (EvenPolicy, ReversiblePolicy, BlockPolicy), internal/placement/weighted.go, " +
			"internal/controller/reconfig.go (rescale)"},
	"U5": {"U5", "Global top-n under fixed-n re-ranking",
		"two-phase is implemented (every owner filters for its own top-n, the aggregator merges " +
			"the global top-n and re-ranks each candidate at the owner that reported it): exactly " +
			"the single-node answer, for a second round trip. Open: whether to pay that round " +
			"trip, or give each owner a quota of n * its share and re-rank in one, whose answer " +
			"can differ from the global top-n",
		"internal/query/query.go (TwoPhase implemented; ProportionalQuota placeholder)"},
	"U6": {"U6", "Outsourcing the per-query aggregator to another node",
		"Decided in the steady state: the entry chooses after navigating, among the nodes the " +
			"query has to reach -- the owners of its probed lists (selector \"owner\", the default): " +
			"itself when it owns some of them; the owner when one node holds them all (one network " +
			"round trip instead of two, and the PQ candidates never leave that node); itself " +
			"otherwise, because forwarding then adds a hop to save one of several parallel " +
			"sub-queries and piles aggregation onto the owners of hot partitions. " +
			"Decided where it applies: the only window in which the nodes that may aggregate " +
			"outnumber the nodes that may take client queries is a scale-out, while a new node's " +
			"navigation graph is still arriving. Selector \"warmup\" hands the scatter/gather to " +
			"those nodes and aggregates locally otherwise. Open: whether it pays for itself -- " +
			"the aggregator's own work is the fan-out and the merge, while the entry keeps the " +
			"navigation, which is the expensive part and the part that needs the graph (measured " +
			"small on a CPU-only run: navigate 160 us of a 445 us server-side query); and a " +
			"load-aware policy (selector \"outsource\") for entries that are hot for other " +
			"reasons. Note that outsourcing and letting the new node take queries and forward the " +
			"navigation (U16's \"stream\") split the work the same way, from opposite sides",
		"internal/query/selector.go (OwnerSelector, WarmupSelector implemented; OutsourceSelector " +
			"placeholder), internal/agent/serve.go (entry)"},
	"U7": {"U7", "Sub-queries that arrive with a stale epoch",
		"serve locally while the data is still resident, forward to the new owner, or reject " +
			"and let the aggregator retry",
		"internal/query/query.go (Aggregator.Run, NotResident)"},
	"U8": {"U8", "Adaptive rate control for background transfer",
		"token bucket on a separate connection (implemented, fixed rate) with two priority " +
			"classes on the sender (implemented): data -- segments and PQ codes, what a node needs " +
			"to own partitions -- and background -- the navigation graph and, after the flip, the " +
			"raw-vector stream (U9, raw_priority), which only take tokens no data sender is waiting " +
			"for. Open: how to adapt the rate to foreground latency; and within the background " +
			"class the graph (needed for the entry role) and the raw stream (which shortens the " +
			"on-demand fetches of the warm-up) now share equally -- whether one should go first, or " +
			"the two get weights",
		"internal/transfer/adapter.go, internal/transfer/tokenbucket.go (classes)"},
	"U9": {"U9", "Lazy fetch of raw vectors (and chunks) after a new node is online",
		"Decided: a node goes online (the flip) only once it holds every posting list and PQ " +
			"code of its partitions, so it can filter at once; a raw vector moves only when a " +
			"query first needs it (protocol \"lazy\", the default) -- the point of being lazy " +
			"is that queries touch a small working set of what moved. Every posting records the " +
			"holder of its vector, the node that stores it (u16 node ID, segment version 3), so " +
			"each key (list, vector) has one current holder: the list's owner if it holds the " +
			"vector, else the node the posting names. Only the owner's copy of a list is valid: a " +
			"partition moves as its owner saves it from memory, holders included (SAVE_PARTITION), " +
			"and the next owner makes the postings of the vectors it holds name itself. A RERANK " +
			"that meets a vector not there fetches it from the holder that a posting in one of the " +
			"query's lists on that node records, data node to data node (RAW_GET, one hop: a " +
			"holder has the vector; data nodes know each other by node ID, SET_NODES), and makes " +
			"that one posting name itself -- other lists naming the vector keep their holder, " +
			"which still has it; queries that miss the same vector meanwhile wait for that one " +
			"fetch. The RERANK waits for the fetch only: fetched vectors are re-ranked from memory " +
			"and a background writer installs them with whole-page direct writes. A scale-out " +
			"does not wait for raw vectors. A scale-in drains the nodes that leave first " +
			"(DrainRaw: each remaining node takes over the vectors its postings say only a leaving " +
			"node holds, RAW_HELD_BY then RAW_PULL; the rescale waits for it), so they can go " +
			"offline when it returns: lazy fetches as little as possible, it does not forbid " +
			"streaming. Holders stay valid because no node drops a raw vector it has held (static " +
			"dataset); a disk budget would need the same hand-off first. Baselines: " +
			"\"lazy-stream\" also streams the rest in after the flip (StageRaw at raw_priority; " +
			"RAW_MISSING groups each missing vector by its holder; the rescale waits for it); " +
			"\"copy-then-flip\" streams before the flip, \"stop-and-copy\" while paused. Open: " +
			"taking the fetch off the query path (re-rank the missing candidates at their holder, " +
			"which has them, and fetch the ones used in the background); fetching more per fetch; " +
			"the workload that decides the working set (U11); chunks (U12)",
		"internal/controller/reconfig.go (copyThenFlip, stageRaw, drainRaw), internal/agent/control.go " +
			"(StageRaw, DrainRaw), engine/src/node_engine.cpp (EnsureRawInLists, FetchForQuery, " +
			"RawHeldBy, RawPull), engine/node/server.cpp (PeerPool)"},
	"U10": {"U10", "Epoch store for the atomic flip",
		"an etcd transaction (as discussed) vs. the controller-local compare-and-swap that the " +
			"single-machine prototype uses; also what happens when the controller fails mid-flip",
		"internal/epoch/store.go"},
	"U11": {"U11", "Workload model",
		"Poisson open-loop arrivals (implemented) vs. replaying SemDN-style bursty, " +
			"task-coherent query streams",
		"cmd/rtier-loadgen/arrivals.go"},
	"U12": {"U12", "Vector ID -> RAG chunk ID mapping returned to the caller",
		"the tier returns vector IDs (no code path maps them yet); the chunk store and the " +
			"fetch path are not designed",
		"internal/query/query.go (Result)"},
	"U13": {"U13", "Failures during a reconfiguration (node or controller)",
		"retry the failed step, roll back, or finish without the failed node. Meanwhile: a failure " +
			"before the flip drops the partitions already copied and keeps the old epoch; after " +
			"the flip the protocol stops and keeps every old copy (safe, wasteful)",
		"internal/controller/reconfig.go (copyThenFlip)"},
	"U14": {"U14", "Dataset updates",
		"vectors are immutable (an update deletes a vector and inserts it again under a new ID) " +
			"and the PQ codebook is fixed, so cached PQ codes and raw vectors on former owners stay " +
			"valid; what remains is maintaining posting lists and the graph. The design under " +
			"discussion (periodic batches, each a content epoch) is in " +
			"docs/drafts/dataset-updates-discussion.md. In code, minor batches (inserts and deletes, " +
			"no splits). Partition versions: the epoch table names a version of each partition's " +
			"lists, a data node holds several versions of a partition side by side, and FILTER/RERANK " +
			"name the one their epoch sees. Inserts: data nodes leave room for inserted IDs and " +
			"locations (rtier_node --max-vectors, --max-pages; both only appended) and for longer " +
			"lists (--max-list-len); ASSIGN routes a vector by the build's own rule and gives its PQ " +
			"code; RAW_ADD serves inserted vectors from memory and writes them after fetched ones, " +
			"never while paused (PAUSE_UPDATES), within --update-write-mbps -- writes are not " +
			"charged to the read budget yet. Batches run only with the controller's allow_updates " +
			"(off by default: a static experiment refuses them; run_local.py's \"Updates\" turns it on, " +
			"leaves room on the data nodes and requires an explicit PQ budget, since the default " +
			"budget is the whole ID space, headroom included). The update controller " +
			"(internal/update, rtier-updater) takes the controller's lease (BeginUpdate: no " +
			"reconfiguration and no other batch until the commit or the abort, or until update_lease " +
			"expires; the agents first release PQ codes left staged by a staging that never ended), " +
			"routes the inserts with its own router (an rtier_node with the graph and codebook only, " +
			"never a serving node), checks the owners' room, sends the PQ codes and raw vectors to " +
			"the owners, and " +
			"has every data node stage the next version of the partitions the batch changes " +
			"(STAGE_VERSIONS: inserts appended, deleted IDs dropped; deletes are broadcast and each " +
			"node finds its copies); CommitUpdate checks the staged versions, flips (all entries at " +
			"once), waits out the grace period, and reclaims: every node frees the deleted vectors' " +
			"PQ codes (FORGET, deleted: copies nothing names go at once, the others are marked) and " +
			"then the owners drop the replaced versions, so a deleted vector's code goes with its " +
			"last version and never becomes a cached one -- a deleted vector is no cache. A staging " +
			"that fails is undone before the lease is given up; a commit that fails before the " +
			"flip drops the staged versions. A partition at a later version moves with its current lists: its owner writes " +
			"them as a segment file for the move (SAVE_PARTITION). Not implemented: growing the ID " +
			"and location spaces at an epoch boundary; the response to a batch over budget (splitting " +
			"partitions, scaling out first -- the batch fails before anything is sent); splits, " +
			"merges and major epochs (the graph rebuild); the health check that triggers them; the " +
			"decision log and replay; freeing the pages of deleted vectors. An inserted vector stays " +
			"with the owner that added it when its partition moves on (the postings name that node " +
			"as its holder), and a node that leaves is drained first (U9). Open: versions staged " +
			"under a lease that expired stay on their owner until a later batch replaces them or " +
			"the partition moves",
		"internal/epoch/table.go (Versions), engine/include/fusion/node_engine.h (partition versions, " +
			"Assign, RawAdd, StageVersions), internal/update (batch.go), internal/controller/update.go, " +
			"internal/controller/reconfig.go (copyThenFlip, drainRaw), engine/include/fusion/pq_store.h " +
			"(cached codes, Forget), cmd/rtier-updater, scripts/run_local.py (\"Updates\")"},
	"U16": {"U16", "The navigation graph as a global replica: size and transfer mode",
		"every entry node needs the whole graph over the list heads, and a node that joins " +
			"starts empty -- pre-staging it on spare nodes is not allowed, because a real " +
			"deployment provisions a node at scale-out time (an experiment may pre-register " +
			"nodes, but pre-loading the graph would hide the largest transfer of the whole " +
			"reconfiguration). Measured 308 B per head at dim 32 with heads at 10.3% of the " +
			"vectors, so roughly 70 MB per million vectors and tens of GB at billion scale, " +
			"larger than the PQ codes a new node pulls. Open: transfer mode -- \"eager\" (today: " +
			"fetch the whole replica during the rescale; each new node takes the entry role in a " +
			"flip as soon as its graph has loaded, batched per entry_flip_interval) " +
			"vs. \"stream\" (take the role at once, forward NAVIGATE to a node that has the graph, " +
			"fill in behind); and how to make it smaller -- head vectors in the dataset's dtype, " +
			"a lower head ratio (fusion_build targets 10%), or replicating only the upper HNSW " +
			"levels and partitioning level 0 (which makes navigation distributed and changes " +
			"recall). Only the eager mode exists in code. Decided for the transfer itself: each " +
			"new entry's source is assigned when the reconfiguration starts, round-robin over the " +
			"live entries with a cursor that carries over between reconfigurations (instead of " +
			"every new node pulling from the first entry), and the pull runs at background " +
			"priority on its source (graph_priority): FILTER and RERANK need posting lists and PQ " +
			"codes, not the graph, and navigation is the smallest share of a query that any " +
			"existing entry can do meanwhile -- so the graph only gets the bandwidth the segment " +
			"and PQ transfers leave unused and never delays the flip. Pairs with U6's warmup " +
			"selector, which moves aggregation to the new node before its graph arrives",
		"internal/controller/reconfig.go (graphSources, StageGraph, promoteEntries), " +
			"internal/transfer/tokenbucket.go (classes), engine/src/node_engine.cpp (LoadGraph)"},
	"U17": {"U17", "Where the client-facing availability list lives",
		"today the entry set is a field of the epoch table, so \"this node finished loading its " +
			"graph\" -- a local event that has nothing to do with who owns what -- costs an epoch " +
			"flip (batched: one per entry_flip_interval for the nodes ready by then, running " +
			"alongside the grace period, so no node waits for the slowest graph), and the rescale " +
			"still blocks until every graph transfer has ended. Proposed: the coordinator " +
			"keeps the available list outside the epoch; a node is added once its graph is " +
			"complete, and until then it only serves work forwarded to it (sub-queries, and " +
			"aggregation under U6); a scale-in removes a node from the list first, drains, then " +
			"flips. The rule this follows: a field belongs in the epoch table only if it changes " +
			"exactly when the node set changes (ownership and the aggregator set do; entry " +
			"availability does not), because what must be epoch-consistent is what decides where " +
			"the data is and who must be drained -- a query sent to a node that should not have " +
			"it costs one UNAVAILABLE and a retry. No code path yet",
		"internal/epoch/table.go (Table.Entries), internal/agent/serve.go (entry), " +
			"internal/controller/reconfig.go (promoteEntries)"},
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
