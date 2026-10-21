// Package config holds the JSON configurations of the controller and the node agents
// (replacing Koala's config.yaml; JSON keeps the build free of third-party modules and is
// what the Python harness writes).
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// Duration is a time.Duration written as "5s" in JSON.
type Duration struct{ time.Duration }

func (d Duration) MarshalJSON() ([]byte, error) { return json.Marshal(d.String()) }

func (d *Duration) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		v, err := time.ParseDuration(s)
		d.Duration = v
		return err
	}
	var ns int64
	if err := json.Unmarshal(b, &ns); err != nil {
		return fmt.Errorf("config: duration must be a string like \"5s\": %s", b)
	}
	d.Duration = time.Duration(ns)
	return nil
}

// Roles a node can take. Readiness is staged in this order when a node joins:
// aggregator (needs peer connections), data (needs its partitions), entry (needs the graph).
const (
	RoleAggregator = "aggregator"
	RoleData       = "data"
	RoleEntry      = "entry"
)

// Controller configures rtier-controller.
type Controller struct {
	Listen         string   `json:"listen"`          // control plane: agents register here
	APIListen      string   `json:"api_listen"`      // rtier-client and the harness
	IndexDir       string   `json:"index_dir"`       // fusion_build output (seed of the graph)
	PartitionsDir  string   `json:"partitions_dir"`  // rtier_segment output (seed of the partitions)
	InitialNodes   int      `json:"initial_nodes"`   // data nodes of the first epoch
	InitialEntries int      `json:"initial_entries"` // entries among them; 0 = all
	NewNodeRoles   []string `json:"new_node_roles"`  // roles of nodes added by scale-out
	Placement      string   `json:"placement"`       // "even-reversible-graph" (default) | "even-reversible" | "even-reversible-blocks" | "even" | "weighted" (U3)
	Protocol       string   `json:"protocol"`        // "lazy" | "lazy-stream" | "copy-then-flip" | "stop-and-copy" (U9)
	EpochStore     string   `json:"epoch_store"`     // "memory" | "etcd" (U10)
	MetricsPath    string   `json:"metrics_path"`    // JSON-lines sink
	CallTimeout    Duration `json:"call_timeout"`    // short control calls
	StageTimeout   Duration `json:"stage_timeout"`   // data/graph staging, drains
	// GraphPriority is the class of graph transfers: "background" (default: the graph moves only
	// while neither end moves segments, PQ codes or raw vectors, U16) or "data" (it shares the
	// source's bucket with them equally -- the baseline to compare against).
	GraphPriority string `json:"graph_priority"`
	// RawPriority is the class of the raw-vector stream that follows the flip in the
	// lazy-stream baseline: "background" (default: queries fetch the vectors they need on
	// demand meanwhile, so the stream only takes bandwidth nothing else wants) or "data". The
	// lazy protocol has no stream; the eager ones stream raw vectors before the flip, as data.
	RawPriority string `json:"raw_priority"`
	// EntryFlipInterval: while new nodes are fetching the navigation graph, every interval one
	// flip makes entries of all those whose graph loaded since the previous flip (and when the
	// last pull finishes, its nodes join at once). Default 1 s: clients refresh their entry
	// list about that often, so flipping more often would not put a new entry to use sooner.
	EntryFlipInterval Duration `json:"entry_flip_interval"`
	// AllowUpdates: whether update batches (U14) may run. Off by default: an experiment on a
	// static dataset refuses BeginUpdate, so no batch can change its data, and its config says
	// that it is static.
	AllowUpdates bool `json:"allow_updates"`
	// UpdateLease: how long an update batch may hold the cluster between BeginUpdate and its
	// commit or abort (U14) before the controller gives the lease up -- an update controller that
	// died must not block reconfigurations for good. Default 10 min.
	UpdateLease Duration `json:"update_lease"`
	// StatsDir: where the nodes' diagnostics arrive -- each data node's stats log
	// (stats-<node>.jsonl) and the agents' process and host samples (cgroups.jsonl), stamped with
	// the controller's clock when they arrive. "" drops them.
	StatsDir string `json:"stats_dir"`
	// WorkDir and BulkListen: the files the controller makes for the nodes (a full rebuild's
	// segments with their holders, U14) go under WorkDir, and with BulkListen the nodes pull
	// them from there over the bulk protocol, whatever machine they run on. Without BulkListen
	// they read them from the controller's filesystem (one machine).
	WorkDir    string `json:"work_dir"`
	BulkListen string `json:"bulk_listen"`
}

// DefaultController returns the controller defaults.
func DefaultController() Controller {
	return Controller{
		Listen:        "127.0.0.1:7100",
		APIListen:     "127.0.0.1:7101",
		InitialNodes:  1,
		NewNodeRoles:  []string{RoleAggregator, RoleData, RoleEntry},
		Placement:     "even-reversible-graph",
		Protocol:      "lazy",
		EpochStore:    "memory",
		MetricsPath:   "metrics.jsonl",
		CallTimeout:   Duration{10 * time.Second},
		StageTimeout:  Duration{30 * time.Minute},
		GraphPriority: "background",
		RawPriority:   "background",

		EntryFlipInterval: Duration{time.Second},
		UpdateLease:       Duration{10 * time.Minute},
	}
}

// Agent configures rtier-agent (one per node).
type Agent struct {
	Name            string   `json:"name"`
	Controller      string   `json:"controller"`   // controller control-plane address
	QueryListen     string   `json:"query_listen"` // client queries + delegated aggregation
	BulkListen      string   `json:"bulk_listen"`  // bulk transfer server (separate connection)
	AdvertiseHost   string   `json:"advertise_host"`
	NodeAddr        string   `json:"node_addr"`  // this node's C++ rtier_node (TCP)
	NodeConns       int      `json:"node_conns"` // requests in flight per data-node client, of each kind: RERANK, and the rest
	WorkDir         string   `json:"work_dir"`   // staged segments and graph live here
	PartitionsDir   string   `json:"partitions_dir"`
	Strategy        string   `json:"strategy"` // "two-phase" (also "") | "proportional-quota" (U5's placeholder)
	Selector        string   `json:"selector"` // "owner" | "local" | "warmup" | "outsource" (U6)
	TransferRate    float64  `json:"transfer_rate_bytes_per_sec"`
	TransferAdapter string   `json:"transfer_adapter"` // "fixed" | "adaptive" (U8)
	ChunkBytes      int      `json:"chunk_bytes"`
	RawStreams      int      `json:"raw_streams"`     // connections a raw-vector stream (StageRaw) pulls from each source over at once
	RawBatchBytes   int      `json:"raw_batch_bytes"` // raw vectors per request of such a stream, in bytes
	RawPath         string   `json:"raw_path"`        // "node" | "agent" | "" (node unless TransferRate paces the agents)
	MetricsInterval Duration `json:"metrics_interval"`
	// StepTimeout: the longest a long step may take inside the agent -- staging, the graph, a raw
	// stream or drain, a grace period, a reclaim, the warm-up. It grows with the dataset and the
	// transfer rate (a billion vectors: a 32-71 GB graph per new entry), so the experiment sets it
	// with the controller's stage_timeout (cluster.stage_timeout).
	StepTimeout Duration `json:"step_timeout"`
	// QuerySample: how many recent client queries an entry keeps, for a full rebuild's warm-up to
	// replay under the next generation (U14; 0: none).
	QuerySample int `json:"query_sample"`
	// NodeStatsFile: the data node's stats log (rtier_node --stats-file), which the agent
	// forwards line by line to the controller (its stats_dir), stamped there on arrival -- one
	// clock however many machines the cluster spans. "" forwards nothing.
	NodeStatsFile string `json:"node_stats_file"`
	// SampleInterval: every interval the agent samples its data node, itself and its machine
	// (CPU, disk, memory, pressure; the node's GPU) for the controller's cgroups.jsonl. 0 = off:
	// on one machine the experiment script samples every node itself.
	SampleInterval Duration `json:"sample_interval"`
}

// DefaultAgent returns the agent defaults.
func DefaultAgent() Agent {
	return Agent{
		Controller:      "127.0.0.1:7100",
		QueryListen:     "127.0.0.1:0",
		BulkListen:      "127.0.0.1:0",
		NodeConns:       8,
		WorkDir:         "work",
		Selector:        "owner",
		TransferAdapter: "fixed",
		ChunkBytes:      1 << 20,
		RawStreams:      2,
		RawBatchBytes:   1 << 20,
		MetricsInterval: Duration{5 * time.Second},
		StepTimeout:     Duration{30 * time.Minute},
		QuerySample:     1024,
	}
}

// Load fills v (already holding defaults) from a JSON file; unknown fields are errors.
func Load(path string, v any) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("config %s: %w", path, err)
	}
	return nil
}
