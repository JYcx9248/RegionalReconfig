// Package protocol defines the control-plane methods and messages exchanged over internal/ctrl
// between the controller, the node agents and rtier-client.
//
// It plays the role of Koala's rpc.proto (CoordinatorToWorker / WorkerToCoordinator messages)
// with the lazy-protocol steps renamed to the regional tier's epoch protocol:
//
//	Koala (coordinator/apiService.go rescaleLazy)      rtier (internal/controller)
//	0 prepareReconfiguration                           placement.Policy.Repartition
//	1 initializeTargetTasksLazy (TaskAssignment)       StageAggregator on new nodes
//	3 waitAllInboundPeersToConnect                     StageAggregator (pre-connect) everywhere
//	  (state that must be there first)                 StagePartitions: lists + PQ codes before the flip
//	4 notifyUpstreamTasks + InflightBarrier            store CAS + InstallEpoch everywhere
//	  (lazy state fetch during processing)             raw vectors: fetched on demand by RERANK, only
//	                                                   those queries need, from each posting's
//	                                                   holder (lazy-stream: StageRaw streams the rest
//	                                                   after the flip); nodes that leave are drained
//	                                                   first (DrainRaw)
//	5 waitAllTasksReconfigDone                         WaitDrained(old epoch) everywhere
//	6 terminateTasks                                   Evict on old owners
//	  (none)                                           StageGraph; entry flips add new nodes as graphs load
//	  (none)                                           update batches (U14): BeginUpdate, staging by the
//	                                                   update controller, CommitUpdate (flip, grace,
//	                                                   DropVersions)
package protocol

import (
	"strconv"
	"time"

	"rtier/internal/epoch"
	"rtier/internal/placement"
)

// Methods served by the agent, called by the controller.
const (
	StageAggregatorM = "StageAggregator" // connect to every data node of a (draft) table
	StagePartitionsM = "StagePartitions" // fetch segments, load them into the data node
	StageGraphM      = "StageGraph"      // fetch the navigation graph, load it
	StageRawM        = "StageRaw"        // stream the raw vectors the node's partitions lack
	DrainRawM        = "DrainRaw"        // before nodes leave: take over the raw vectors only they hold
	InstallEpochM    = "InstallEpoch"    // switch routing to a new table
	WaitDrainedM     = "WaitDrained"     // grace period: queries pinned to <= epoch finished
	EvictM           = "Evict"           // drop partitions and their files
	SavePartitionsM  = "SavePartitions"  // write versions of resident partitions as segment files (U14)
	DropVersionsM    = "DropVersions"    // drop versions an update batch replaced, and their files (U14)
	SetAdmissionM    = "SetAdmission"    // pause/resume client queries (stop-and-copy baseline)
	WaitIdleM        = "WaitIdle"        // no query in flight at all
	NodeStatusM      = "Status"
	ShutdownM        = "Shutdown"
)

// Methods served by the controller, called by agents.
const (
	RegisterM = "Register"
	MetricsN  = "Metrics" // notification: metrics.Report
)

// API methods served by the controller, called by rtier-client and the harness.
const (
	RescaleA   = "Rescale"
	StatusA    = "Status"
	WaitReadyA = "WaitReady"
	DesignA    = "Design"

	// Update batches (U14), called by the update controller (internal/update).
	BeginUpdateA  = "BeginUpdate"
	CommitUpdateA = "CommitUpdate"
	AbortUpdateA  = "AbortUpdate"
)

// Register is an agent's first message (Koala's Registration: data-plane and state-comm
// ports; here the addresses other nodes use to reach this one).
type Register struct {
	Name string         `json:"name"`
	Info epoch.NodeInfo `json:"info"` // ID is assigned by the controller
}

type RegisterReply struct {
	ID placement.NodeID `json:"id"`
}

// StageAggregatorReq pre-connects to every data node of Table.
type StageAggregatorReq struct {
	Table *epoch.Table `json:"table"`
}

// Source is where staged files come from. The raw vectors of the partitions are not: each
// posting of a segment records the node holding its vector, and the destination fetches from
// that node.
type Source struct {
	Kind       string `json:"kind"` // "peer": another agent's bulk port; "local": a directory
	Addr       string `json:"addr,omitempty"`
	Dir        string `json:"dir,omitempty"`
	Partitions []int  `json:"partitions,omitempty"`
	// Versions: the version of each of Partitions in the epoch the staging is for (absent: all
	// 0) -- the segment file to fetch and the version to load it as (U14).
	Versions []uint32 `json:"versions,omitempty"`
	File     string   `json:"file,omitempty"` // StageGraph: the file name at the source
}

// VersionOf is the version of Partitions[i].
func (s Source) VersionOf(i int) uint32 {
	if i < len(s.Versions) {
		return s.Versions[i]
	}
	return 0
}

// PartitionVersion names one version of a partition's lists (U14).
type PartitionVersion struct {
	Partition int    `json:"partition"`
	Version   uint32 `json:"version"`
}

// StagePartitionsReq asks the agent to fetch and load partitions. Nodes: the cluster's nodes,
// among them those the segments name as holders of raw vectors -- where the data node fetches
// them.
type StagePartitionsReq struct {
	Sources []Source                            `json:"sources"`
	Nodes   map[placement.NodeID]epoch.NodeInfo `json:"nodes,omitempty"`
}

// StageReply reports what a staging step moved. Bytes counts everything received (segment
// files, PQ codes, raw vectors); PQCodes/PQBytes and RawVectors/RawBytes are what was pulled
// from peers, i.e. only what the destination lacked (each vector's code and raw vector once
// per node, U1).
type StageReply struct {
	Bytes      int64 `json:"bytes"`
	PQCodes    int64 `json:"pq_codes"`
	PQBytes    int64 `json:"pq_bytes"`
	RawVectors int64 `json:"raw_vectors,omitempty"`
	RawBytes   int64 `json:"raw_bytes,omitempty"`
	// Repointed (DrainRaw): postings that named a leaving node and whose vector was here
	// already; they name this node now, nothing moved for them.
	Repointed int64   `json:"repointed,omitempty"`
	Seconds   float64 `json:"seconds"`
}

// StageRawReq asks the agent to stream in the raw vectors that the given partitions (already
// loaded) name and its data node lacks, each from the source listed with it. Priority is the
// class on the sources' buckets: "background" (the default after the flip) or "data".
type StageRawReq struct {
	Sources  []Source `json:"sources"`
	Priority string   `json:"priority,omitempty"`
}

// DrainRawReq asks the agent, before the nodes From leave, to take over the raw vectors its
// postings say only they hold: the data node pulls each from its holder (as StageRaw does, at
// Priority), and its postings then name it. Afterwards no posting on the node names a node of
// From, so they can go offline. Nodes: where they are.
type DrainRawReq struct {
	From     []placement.NodeID                  `json:"from"`
	Nodes    map[placement.NodeID]epoch.NodeInfo `json:"nodes,omitempty"`
	Priority string                              `json:"priority,omitempty"`
}

// StageGraphReq asks the agent to fetch and load the navigation graph. Priority is the
// transfer class on the source's bucket: "background" (the default the controller sends) or
// "data".
type StageGraphReq struct {
	Source   Source `json:"source"`
	Priority string `json:"priority,omitempty"`
}

type InstallEpochReq struct {
	Table *epoch.Table `json:"table"`
}

type WaitDrainedReq struct {
	Epoch uint64 `json:"epoch"`
}

type EvictReq struct {
	Partitions []int `json:"partitions"`
	// Undoing an aborted staging: stop a staging still running on the node, then free the PQ
	// codes installed for it. (Codes of evicted partitions otherwise stay cached on the node.)
	ReleaseStagedPQ bool `json:"release_staged_pq,omitempty"`
}

// SavePartitionsReq asks the agent to write these versions of resident partitions as segment
// files in its partition directory, where the nodes they move to pull them: a version an update
// batch made exists only in the data node's memory until then (U14).
type SavePartitionsReq struct {
	Versions []PartitionVersion `json:"versions"`
}

// DropVersionsReq is the reclaim of an update batch on one node (U14), after the grace period
// of the epochs that read the replaced versions. Deleted: the vectors the batch deleted; the
// node frees their PQ codes -- now if nothing names them (a former owner's cached copies), with
// the dropped versions otherwise -- so that none becomes a cached code. Then it drops Versions,
// the versions the batch replaced on this node, and their segment files.
type DropVersionsReq struct {
	Versions []PartitionVersion `json:"versions,omitempty"`
	Deleted  []uint32           `json:"deleted,omitempty"`
}

type SetAdmissionReq struct {
	Paused bool `json:"paused"`
}

// NodeStatus is an agent's view of itself.
type NodeStatus struct {
	ID          placement.NodeID `json:"id"`
	Name        string           `json:"name"`
	Epoch       uint64           `json:"epoch"`
	Ready       map[string]bool  `json:"ready"` // role -> staged
	Resident    []int            `json:"resident"`
	Paused      bool             `json:"paused"`
	GraphLoaded bool             `json:"graph_loaded"`
	// Raw vectors the data node fetched on demand since it started, and their bytes.
	RawFetched      uint64 `json:"raw_fetched"`
	RawFetchedBytes uint64 `json:"raw_fetched_bytes"`
	// Versions: every version of a partition's lists the data node holds (U14).
	Versions []PartitionVersion `json:"versions,omitempty"`
}

// RescaleReq asks for a reconfiguration in one of two forms: DataNodes, a number of data
// nodes for the controller's placement policy to spread the partitions over, or Plan, the
// placement itself. With DryRun the controller works out the placement and what would move
// (the reply's Plan, Added, Removed, Moved) and changes nothing.
type RescaleReq struct {
	DataNodes int   `json:"data_nodes,omitempty"`
	Plan      *Plan `json:"plan,omitempty"`
	DryRun    bool  `json:"dry_run,omitempty"`
}

// Plan is a reconfiguration plan: the placement a reconfiguration ends in, as the partitions
// each data node owns afterwards, keyed by node ID or by node name (a key that is a number is
// an ID). The nodes it names are the data nodes of the next epoch -- a current one it leaves
// out leaves, a registered idle node it names joins -- so one plan covers a scale-out, a
// scale-in and a rebalance at the same size. The rest follows from the current epoch as in any
// reconfiguration: which partitions move and from where (placement.Diff), the roles, and where
// raw vectors and the graph come from. FromEpoch, if set, is the epoch the plan was made for;
// the controller refuses the plan under any other, because the moves it implies depend on the
// placement it starts from.
type Plan struct {
	FromEpoch uint64           `json:"from_epoch,omitempty"`
	Owners    map[string][]int `json:"owners"`
}

// PlanOf is placement t as a plan made for epoch from, keyed by node ID.
func PlanOf(t placement.Table, from uint64) *Plan {
	p := &Plan{FromEpoch: from, Owners: map[string][]int{}}
	for id, ps := range t.ByNode() {
		p.Owners[strconv.Itoa(int(id))] = ps
	}
	return p
}

// RescaleReply summarizes a reconfiguration.
type RescaleReply struct {
	Protocol  string             `json:"protocol"`
	FromEpoch uint64             `json:"from_epoch"`
	ToEpoch   uint64             `json:"to_epoch"`
	Added     []placement.NodeID `json:"added"`
	Removed   []placement.NodeID `json:"removed"`
	Moved     []int              `json:"moved"`
	Bytes     int64              `json:"bytes"`    // segment files + PQ codes received by destinations
	PQCodes   int64              `json:"pq_codes"` // PQ codes pulled (deduplicated per destination)
	PQBytes   int64              `json:"pq_bytes"`
	// Raw vectors streamed to the destinations (deduplicated per destination): none with the
	// lazy protocol; with lazy-stream the stream runs after the flip and RawSeconds is how long
	// after the flip it ended. The vectors queries fetch on demand are not counted here (see
	// RawFetched and the data nodes' raw.fetched).
	RawVectors int64   `json:"raw_vectors"`
	RawBytes   int64   `json:"raw_bytes"`
	RawSeconds float64 `json:"raw_seconds,omitempty"`
	// Drain: when nodes leave, the raw vectors only they held, pulled by the nodes whose
	// postings named them (U9: whatever protocol, a node that leaves is never left as the
	// only holder of a vector, so it can go offline once the reconfiguration returns), and how
	// long after the flip that ended.
	DrainVectors int64   `json:"drain_vectors,omitempty"`
	DrainBytes   int64   `json:"drain_bytes,omitempty"`
	DrainSeconds float64 `json:"drain_seconds,omitempty"`
	// GraphBytes: navigation graphs copied to the new entries (in the background, see
	// EntryFlips), counted once every pull has ended.
	GraphBytes int64 `json:"graph_bytes,omitempty"`
	// RawFetched: raw vectors that queries fetched on demand while the reconfiguration ran
	// (lazy protocols), from every node's counter before and after. With lazy the fetches go
	// on after the rescale returns, for as long as queries meet vectors that have not moved.
	RawFetched      int64              `json:"raw_fetched,omitempty"`
	RawFetchedBytes int64              `json:"raw_fetched_bytes,omitempty"`
	Phases          map[string]float64 `json:"phases"` // seconds per phase
	// GraphSources: which existing entry each new entry copies the graph from, assigned when
	// the reconfiguration starts (absent: copied from the build output).
	GraphSources map[placement.NodeID]placement.NodeID `json:"graph_sources,omitempty"`
	// EntryFlips: the flips that made the new nodes entries, as their graphs loaded.
	EntryFlips []EntryFlip `json:"entry_flips,omitempty"`
	// EntryFailed: new nodes whose graph did not load (why). They are data nodes and
	// aggregators of the new epoch, but not entries.
	EntryFailed map[placement.NodeID]string `json:"entry_failed,omitempty"`
	// Plan: the placement the reconfiguration went to (with DryRun: would go to), as a plan
	// for FromEpoch that a RescaleReq can pass back, edited or not.
	Plan   *Plan `json:"plan,omitempty"`
	DryRun bool  `json:"dry_run,omitempty"`
}

// EntryFlip is one flip of a scale-out that made new nodes entries once their graph loaded.
type EntryFlip struct {
	Epoch   uint64             `json:"epoch"`
	Nodes   []placement.NodeID `json:"nodes"`
	Seconds float64            `json:"seconds"` // since the reconfiguration was triggered
}

// NodeSummary is the controller's view of one registered node.
type NodeSummary struct {
	ID    placement.NodeID `json:"id"`
	Name  string           `json:"name"`
	Info  epoch.NodeInfo   `json:"info"`
	Idle  bool             `json:"idle"` // registered but not part of the current epoch
	Alive bool             `json:"alive"`
}

// ClusterStatus is the controller's Status reply.
type ClusterStatus struct {
	Ready              bool          `json:"ready"`
	Table              *epoch.Table  `json:"table"`
	Nodes              []NodeSummary `json:"nodes"`
	ReconfigInProgress bool          `json:"reconfig_in_progress"` // a reconfiguration or an update batch
	UpdateInProgress   bool          `json:"update_in_progress,omitempty"`
	LastError          string        `json:"last_error,omitempty"`
}

// WaitReadyReq blocks until the initial deployment finished.
type WaitReadyReq struct {
	TimeoutSeconds float64 `json:"timeout_seconds"`
}

// BeginUpdateReq asks for the lease an update batch runs under (U14). Until the batch is
// committed or aborted, or the lease expires, the controller starts no reconfiguration and no
// other batch: the update controller routes the batch by the lease's epoch and stages it on
// that epoch's owners, so the placement must not change underneath it.
type BeginUpdateReq struct {
	// LeaseSeconds: how long the lease lasts without a commit or an abort (0: the controller's
	// update_lease).
	LeaseSeconds float64 `json:"lease_seconds,omitempty"`
}

// UpdateLease is the lease of one update batch.
type UpdateLease struct {
	Token string `json:"token"`
	// Table: the current epoch. The batch is staged for its placement and versions, and the
	// commit flips from it.
	Table *epoch.Table `json:"table"`
	// Nodes: every registered node. Former owners not in Table hold cached PQ codes of vectors
	// the batch deletes.
	Nodes   []NodeSummary `json:"nodes"`
	Expires time.Time     `json:"expires"`
}

// CommitUpdateReq flips to the versions an update batch staged on the owners of the lease's
// epoch: the next version of every partition the batch changed (partitions it did not change
// keep theirs). Deleted: the vectors the batch deletes, whose PQ codes the reclaim frees on
// every node, former owners included.
type CommitUpdateReq struct {
	Token     string             `json:"token"`
	FromEpoch uint64             `json:"from_epoch"`
	Versions  []PartitionVersion `json:"versions"`
	Deleted   []uint32           `json:"deleted,omitempty"`
}

// UpdateReply summarizes a committed batch. ToEpoch is FromEpoch when it changed nothing, or
// when the commit failed before the flip.
type UpdateReply struct {
	FromEpoch uint64             `json:"from_epoch"`
	ToEpoch   uint64             `json:"to_epoch"`
	Changed   []int              `json:"changed"` // partitions at a new version
	Phases    map[string]float64 `json:"phases"`  // seconds per phase
}

// AbortUpdateReq ends a lease without a flip. What the batch staged is the update
// controller's to drop.
type AbortUpdateReq struct {
	Token string `json:"token"`
}
