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
//	  (lazy state fetch during processing)             raw vectors: fetched on demand by RERANK, and
//	                                                   StageRaw streams the rest after the flip
//	5 waitAllTasksReconfigDone                         WaitDrained(old epoch) everywhere
//	6 terminateTasks                                   Evict on old owners
//	  (none)                                           StageGraph; entry flips add new nodes as graphs load
package protocol

import (
	"rtier/internal/epoch"
	"rtier/internal/placement"
)

// Methods served by the agent, called by the controller.
const (
	StageAggregatorM = "StageAggregator" // connect to every data node of a (draft) table
	StagePartitionsM = "StagePartitions" // fetch segments, load them into the data node
	StageGraphM      = "StageGraph"      // fetch the navigation graph, load it
	StageRawM        = "StageRaw"        // stream the raw vectors the node's partitions lack
	InstallEpochM    = "InstallEpoch"    // switch routing to a new table
	WaitDrainedM     = "WaitDrained"     // grace period: queries pinned to <= epoch finished
	EvictM           = "Evict"           // drop partitions and their files
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

// Source is where staged files come from.
type Source struct {
	Kind string `json:"kind"` // "peer": another agent's bulk port; "local": a directory
	Addr string `json:"addr,omitempty"`
	// NodeAddr: a peer's data node, where the destination's data node fetches the raw vectors
	// of these partitions on demand until they are streamed in (StagePartitions).
	NodeAddr   string `json:"node_addr,omitempty"`
	Dir        string `json:"dir,omitempty"`
	Partitions []int  `json:"partitions,omitempty"`
	File       string `json:"file,omitempty"` // StageGraph: the file name at the source
}

// StagePartitionsReq asks the agent to fetch and load partitions.
type StagePartitionsReq struct {
	Sources []Source `json:"sources"`
}

// StageReply reports what a staging step moved. Bytes counts everything received (segment
// files, PQ codes, raw vectors); PQCodes/PQBytes and RawVectors/RawBytes are what was pulled
// from peers, i.e. only what the destination lacked (each vector's code and raw vector once
// per node, U1).
type StageReply struct {
	Bytes      int64   `json:"bytes"`
	PQCodes    int64   `json:"pq_codes"`
	PQBytes    int64   `json:"pq_bytes"`
	RawVectors int64   `json:"raw_vectors,omitempty"`
	RawBytes   int64   `json:"raw_bytes,omitempty"`
	Seconds    float64 `json:"seconds"`
}

// StageRawReq asks the agent to stream in the raw vectors that the given partitions (already
// loaded) name and its data node lacks, each from the source listed with it. Priority is the
// class on the sources' buckets: "background" (the default after the flip) or "data".
type StageRawReq struct {
	Sources  []Source `json:"sources"`
	Priority string   `json:"priority,omitempty"`
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
}

// RescaleReq changes the number of data nodes.
type RescaleReq struct {
	DataNodes int `json:"data_nodes"`
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
	// Raw vectors streamed to the destinations (deduplicated per destination). With the lazy
	// protocol the stream runs after the flip and RawSeconds is how long after the flip it
	// ended; the vectors that queries fetched on demand before it are not counted here (see
	// the data nodes' raw.fetched).
	RawVectors int64              `json:"raw_vectors"`
	RawBytes   int64              `json:"raw_bytes"`
	RawSeconds float64            `json:"raw_seconds,omitempty"`
	Phases     map[string]float64 `json:"phases"` // seconds per phase
	// GraphSources: which existing entry each new entry copies the graph from, assigned when
	// the reconfiguration starts (absent: copied from the build output).
	GraphSources map[placement.NodeID]placement.NodeID `json:"graph_sources,omitempty"`
	// EntryFlips: the flips that made the new nodes entries, as their graphs loaded.
	EntryFlips []EntryFlip `json:"entry_flips,omitempty"`
	// EntryFailed: new nodes whose graph did not load (why). They are data nodes and
	// aggregators of the new epoch, but not entries.
	EntryFailed map[placement.NodeID]string `json:"entry_failed,omitempty"`
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
	ReconfigInProgress bool          `json:"reconfig_in_progress"`
	LastError          string        `json:"last_error,omitempty"`
}

// WaitReadyReq blocks until the initial deployment finished.
type WaitReadyReq struct {
	TimeoutSeconds float64 `json:"timeout_seconds"`
}
