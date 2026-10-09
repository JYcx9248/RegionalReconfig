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
	Placement      string   `json:"placement"`       // "even" | "even-reversible" | "weighted" (U3)
	Protocol       string   `json:"protocol"`        // "lazy" | "lazy-stream" | "copy-then-flip" | "stop-and-copy" (U9)
	EpochStore     string   `json:"epoch_store"`     // "memory" | "etcd" (U10)
	MetricsPath    string   `json:"metrics_path"`    // JSON-lines sink
	CallTimeout    Duration `json:"call_timeout"`    // short control calls
	StageTimeout   Duration `json:"stage_timeout"`   // data/graph staging, drains
	// GraphPriority is the class of graph transfers on their source's bucket: "background"
	// (default: only bandwidth the segment and PQ transfers leave unused) or "data" (share it
	// equally -- the baseline to compare against).
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
}

// DefaultController returns the controller defaults.
func DefaultController() Controller {
	return Controller{
		Listen:        "127.0.0.1:7100",
		APIListen:     "127.0.0.1:7101",
		InitialNodes:  1,
		NewNodeRoles:  []string{RoleAggregator, RoleData, RoleEntry},
		Placement:     "even-reversible",
		Protocol:      "lazy",
		EpochStore:    "memory",
		MetricsPath:   "metrics.jsonl",
		CallTimeout:   Duration{10 * time.Second},
		StageTimeout:  Duration{30 * time.Minute},
		GraphPriority: "background",
		RawPriority:   "background",

		EntryFlipInterval: Duration{time.Second},
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
	NodeConns       int      `json:"node_conns"` // connections per data-node client
	WorkDir         string   `json:"work_dir"`   // staged segments and graph live here
	PartitionsDir   string   `json:"partitions_dir"`
	Strategy        string   `json:"strategy"` // "two-phase" | "proportional-quota" (U5)
	Selector        string   `json:"selector"` // "owner" | "local" | "warmup" | "outsource" (U6)
	TransferRate    float64  `json:"transfer_rate_bytes_per_sec"`
	TransferAdapter string   `json:"transfer_adapter"` // "fixed" | "adaptive" (U8)
	ChunkBytes      int      `json:"chunk_bytes"`
	RawStreams      int      `json:"raw_streams"`     // connections a raw-vector stream (StageRaw) pulls from each source over at once
	RawBatchBytes   int      `json:"raw_batch_bytes"` // raw vectors per PULL_RAW request, in bytes
	MetricsInterval Duration `json:"metrics_interval"`
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
