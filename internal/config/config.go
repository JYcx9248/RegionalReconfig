// Package config holds config.yaml, the system configuration (Koala's config.yaml): one file
// with the settings every process of the cluster reads at startup -- the controller's and the
// agents' -- by key. A process's own identity (an agent's name, its data node, its addresses and
// directory) comes from its command line. The experiment runners rewrite the file at the
// repository root from an experiment's JSON before a run (scripts/run_local.py), and each run
// keeps a copy.
package config

import (
	"bytes"
	"fmt"
	"os"
	"reflect"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Duration is a time.Duration written as "5s" in config.yaml (a number is nanoseconds).
type Duration struct{ time.Duration }

func (d Duration) MarshalYAML() (any, error) { return d.String(), nil }

func (d *Duration) UnmarshalYAML(n *yaml.Node) error {
	if n.Kind == yaml.ScalarNode && n.Tag == "!!int" {
		var ns int64
		if err := n.Decode(&ns); err != nil {
			return err
		}
		d.Duration = time.Duration(ns)
		return nil
	}
	var s string
	if err := n.Decode(&s); err != nil {
		return fmt.Errorf("config: a duration is a string like \"5s\" (line %d)", n.Line)
	}
	v, err := time.ParseDuration(s)
	if err != nil {
		return fmt.Errorf("config: line %d: %w", n.Line, err)
	}
	d.Duration = v
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
	Listen         string   `yaml:"ControllerAddr"` // control plane: agents register here
	APIListen      string   `yaml:"APIAddr"`        // rtier-client and the harness
	IndexDir       string   `yaml:"IndexDir"`       // fusion_build output (seed of the graph)
	PartitionsDir  string   `yaml:"PartitionsDir"`  // rtier_segment output (seed of the partitions)
	InitialNodes   int      `yaml:"InitialNodes"`   // data nodes of the first epoch
	InitialEntries int      `yaml:"InitialEntries"` // entries among them; 0 = all
	NewNodeRoles   []string `yaml:"NewNodeRoles"`   // roles of nodes added by scale-out
	Placement      string   `yaml:"Placement"`      // "even-reversible-graph" (default) | "even-reversible" | "even-reversible-blocks" | "even" | "weighted" (U3)
	Protocol       string   `yaml:"Protocol"`       // "lazy" | "lazy-stream" | "copy-then-flip" | "stop-and-copy" (U9)
	EpochStore     string   `yaml:"EpochStore"`     // "memory" | "etcd" (U10)
	MetricsPath    string   `yaml:"MetricsPath"`    // JSON-lines sink
	CallTimeout    Duration `yaml:"CallTimeout"`    // short control calls
	StageTimeout   Duration `yaml:"StageTimeout"`   // data/graph staging, drains
	// GraphPriority is the class of graph transfers: "background" (default: the graph moves only
	// while neither end moves segments, PQ codes or raw vectors, U16) or "data" (it shares the
	// source's bucket with them equally -- the baseline to compare against).
	GraphPriority string `yaml:"GraphPriority"`
	// RawPriority is the class of the raw-vector stream that follows the flip in the
	// lazy-stream baseline: "background" (default: queries fetch the vectors they need on
	// demand meanwhile, so the stream only takes bandwidth nothing else wants) or "data". The
	// lazy protocol has no stream; the eager ones stream raw vectors before the flip, as data.
	RawPriority string `yaml:"RawPriority"`
	// EntryFlipInterval: while new nodes are fetching the navigation graph, every interval one
	// flip makes entries of all those whose graph loaded since the previous flip (and when the
	// last pull finishes, its nodes join at once). Default 1 s: clients refresh their entry
	// list about that often, so flipping more often would not put a new entry to use sooner.
	EntryFlipInterval Duration `yaml:"EntryFlipInterval"`
	// AllowUpdates: whether update batches (U14) may run. Off by default: an experiment on a
	// static dataset refuses BeginUpdate, so no batch can change its data, and its config says
	// that it is static.
	AllowUpdates bool `yaml:"AllowUpdates"`
	// UpdateLease: how long an update batch may hold the cluster between BeginUpdate and its
	// commit or abort (U14) before the controller gives the lease up -- an update controller that
	// died must not block reconfigurations for good. Default 10 min.
	UpdateLease Duration `yaml:"UpdateLease"`
	// StatsDir: where the nodes' diagnostics arrive -- each data node's stats log
	// (stats-<node>.jsonl) and the agents' process and host samples (cgroups.jsonl), stamped with
	// the controller's clock when they arrive. "" drops them.
	StatsDir string `yaml:"StatsDir"`
	// WorkDir and BulkListen: the files the controller makes for the nodes (a full rebuild's
	// segments with their holders, U14) go under WorkDir, and with BulkListen the nodes pull
	// them from there over the bulk protocol, whatever machine they run on. Without BulkListen
	// they read them from the controller's filesystem (one machine).
	WorkDir    string `yaml:"ControllerWorkDir"`
	BulkListen string `yaml:"ControllerBulkAddr"`
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

// Agent configures rtier-agent (one per node). The fields without a key in config.yaml
// (yaml:"-") are each agent's own and come from its command line (rtier-agent -name, -node, ...).
type Agent struct {
	Name            string   `yaml:"-"`
	Controller      string   `yaml:"ControllerAddr"` // controller control-plane address
	QueryListen     string   `yaml:"-"`              // client queries + delegated aggregation
	BulkListen      string   `yaml:"-"`              // bulk transfer server (separate connection)
	AdvertiseHost   string   `yaml:"-"`
	NodeAddr        string   `yaml:"-"`         // this node's C++ rtier_node (TCP)
	NodeConns       int      `yaml:"NodeConns"` // requests in flight per data-node client, of each kind: RERANK, and the rest
	WorkDir         string   `yaml:"-"`         // staged segments and graph live here
	PartitionsDir   string   `yaml:"PartitionsDir"`
	Strategy        string   `yaml:"Strategy"` // "two-phase" (also "") | "proportional-quota" (U5's placeholder)
	Selector        string   `yaml:"Selector"` // "owner" | "local" | "warmup" | "outsource" (U6)
	TransferRate    float64  `yaml:"TransferRateBytesPerSec"`
	TransferAdapter string   `yaml:"TransferAdapter"` // "fixed" | "adaptive" (U8)
	ChunkBytes      int      `yaml:"ChunkBytes"`
	RawStreams      int      `yaml:"RawStreams"`    // connections a raw-vector stream (StageRaw) pulls from each source over at once
	RawBatchBytes   int      `yaml:"RawBatchBytes"` // raw vectors per request of such a stream, in bytes
	RawPath         string   `yaml:"RawPath"`       // "node" | "agent" | "" (node unless TransferRate paces the agents)
	MetricsInterval Duration `yaml:"MetricsInterval"`
	// StepTimeout: the longest a long step may take inside the agent -- staging, the graph, a raw
	// stream or drain, a grace period, a reclaim, the warm-up. It grows with the dataset and the
	// transfer rate (a billion vectors: a 32-71 GB graph per new entry), so an experiment sets it;
	// one key sets it and the controller's StageTimeout.
	StepTimeout Duration `yaml:"StageTimeout"`
	// QuerySample: how many recent client queries an entry keeps, for a full rebuild's warm-up to
	// replay under the next generation (U14; 0: none).
	QuerySample int `yaml:"QuerySample"`
	// NodeStatsFile: the data node's stats log (rtier_node --stats-file), which the agent
	// forwards line by line to the controller (its StatsDir), stamped there on arrival -- one
	// clock however many machines the cluster spans. "" forwards nothing.
	NodeStatsFile string `yaml:"-"`
	// SampleInterval: every interval the agent samples its data node, itself and its machine
	// (CPU, disk, memory, pressure; the node's GPU) for the controller's cgroups.jsonl. 0 = off:
	// on one machine the experiment script samples every node itself.
	SampleInterval Duration `yaml:"SampleInterval"`
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

// Load fills v (a *Controller or an *Agent, holding its defaults) from config.yaml at path. The
// file holds both processes' keys, and each takes its own; a key neither knows is an error, so
// that a misspelled one cannot pass for a default.
func Load(path string, v any) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var keys map[string]yaml.Node
	if err := yaml.Unmarshal(b, &keys); err != nil {
		return fmt.Errorf("config %s: %w", path, err)
	}
	known := Keys()
	for k := range keys {
		if !known[k] {
			return fmt.Errorf("config %s: unknown key %q", path, k)
		}
	}
	if err := yaml.Unmarshal(b, v); err != nil {
		return fmt.Errorf("config %s: %w", path, err)
	}
	return nil
}

// Keys is every key of config.yaml: the controller's and the agents'.
func Keys() map[string]bool {
	out := map[string]bool{}
	for _, t := range []reflect.Type{reflect.TypeOf(Controller{}), reflect.TypeOf(Agent{})} {
		for i := 0; i < t.NumField(); i++ {
			if k := yamlKey(t.Field(i)); k != "" {
				out[k] = true
			}
		}
	}
	return out
}

func yamlKey(f reflect.StructField) string {
	k, _, _ := strings.Cut(f.Tag.Get("yaml"), ",")
	if k == "-" {
		return ""
	}
	return k
}

// Defaults is config.yaml with every key at its default, the controller's keys first: what the
// runners start from before they write an experiment's values. A key both processes read has
// one default.
func Defaults() ([]byte, error) {
	doc := &yaml.Node{Kind: yaml.MappingNode}
	seen := map[string]string{}
	for _, v := range []any{DefaultController(), DefaultAgent()} {
		rv := reflect.ValueOf(v)
		for i := 0; i < rv.NumField(); i++ {
			k := yamlKey(rv.Type().Field(i))
			if k == "" {
				continue
			}
			var val yaml.Node
			if err := val.Encode(rv.Field(i).Interface()); err != nil {
				return nil, fmt.Errorf("config: %s: %w", k, err)
			}
			text, err := yaml.Marshal(&val)
			if err != nil {
				return nil, err
			}
			if prev, ok := seen[k]; ok {
				if prev != string(text) {
					return nil, fmt.Errorf("config: %s has two defaults, %q and %q", k, prev, text)
				}
				continue
			}
			seen[k] = string(text)
			doc.Content = append(doc.Content, &yaml.Node{Kind: yaml.ScalarNode, Value: k}, &val)
		}
	}
	var buf bytes.Buffer
	enc := yaml.NewEncoder(&buf)
	enc.SetIndent(2)
	if err := enc.Encode(doc); err != nil {
		return nil, err
	}
	return buf.Bytes(), enc.Close()
}
