// Package controller is the global control plane of the regional tier (Koala's coordinator
// without the dataflow): node registration, placement, the epoch store, reconfiguration, the
// API for rtier-client and the harness, and the metrics sink.
package controller

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"rtier/internal/config"
	"rtier/internal/ctrl"
	"rtier/internal/design"
	"rtier/internal/epoch"
	"rtier/internal/metrics"
	"rtier/internal/partitioning"
	"rtier/internal/placement"
	"rtier/internal/protocol"
	"rtier/internal/syncflag"
	"rtier/internal/transfer"
)

// Controller is the control plane.
type Controller struct {
	cfg      config.Controller
	manifest *partitioning.Manifest
	policy   placement.Policy
	store    epoch.Store
	sink     *metrics.Sink
	nodes    *NodeManager

	// reconfigRunning: a reconfiguration runs, or an update batch holds its lease (lease). One
	// at a time: both change the epoch, and a batch is staged for one placement.
	reconfigRunning atomic.Bool
	Ready           *syncflag.SyncFlag // initial deployment finished
	Listening       *syncflag.SyncFlag // listeners bound (addresses valid)

	mu        sync.Mutex
	history   *placement.History
	graphNext int // round-robin cursor over the entries that serve the graph to new nodes
	lastErr   string
	ctrlAddr  string
	apiAddr   string
	lease     *updateLease // the update batch holding reconfigRunning, if any (update.go)
	statsMu   sync.Mutex   // appends to stats_dir
	bulkAddr  string       // where nodes pull the files the controller makes (bulk_listen)

	// BeforeGenerationFlip, when set, runs just before a full rebuild flips to the next index
	// generation, with the epoch it flips to (tests: the answers expected from it on).
	BeforeGenerationFlip func(epoch uint64)
}

// New validates cfg and loads the partition manifest.
func New(cfg config.Controller) (*Controller, error) {
	if cfg.PartitionsDir == "" || cfg.IndexDir == "" {
		return nil, fmt.Errorf("controller: index_dir and partitions_dir are required")
	}
	if cfg.InitialNodes < 1 {
		return nil, fmt.Errorf("controller: initial_nodes must be >= 1")
	}
	switch cfg.Protocol {
	case "lazy", "lazy-stream", "copy-then-flip", "stop-and-copy":
	default:
		return nil, fmt.Errorf("controller: unknown protocol %q (lazy | lazy-stream | copy-then-flip | stop-and-copy)", cfg.Protocol)
	}
	for key, v := range map[string]string{"graph_priority": cfg.GraphPriority, "raw_priority": cfg.RawPriority} {
		switch v {
		case "", "background", "data":
		default:
			return nil, fmt.Errorf("controller: unknown %s %q (background | data)", key, v)
		}
	}
	if cfg.EntryFlipInterval.Duration <= 0 {
		return nil, fmt.Errorf("controller: entry_flip_interval must be positive, got %v", cfg.EntryFlipInterval.Duration)
	}
	if cfg.UpdateLease.Duration <= 0 {
		return nil, fmt.Errorf("controller: update_lease must be positive, got %v", cfg.UpdateLease.Duration)
	}
	m, err := partitioning.LoadManifest(cfg.PartitionsDir)
	if err != nil {
		return nil, err
	}
	policy, err := newPolicy(cfg.Placement, cfg.IndexDir, m)
	if err != nil {
		return nil, err
	}
	store, err := epoch.NewStore(cfg.EpochStore)
	if err != nil {
		return nil, err
	}
	c := &Controller{
		cfg: cfg, manifest: m, policy: policy, store: store, nodes: NewNodeManager(),
		Ready: syncflag.NewSyncFlag(), Listening: syncflag.NewSyncFlag(),
	}
	if cfg.MetricsPath != "" {
		if c.sink, err = metrics.OpenSink(cfg.MetricsPath); err != nil {
			return nil, err
		}
	}
	return c, nil
}

// newPolicy makes the placement policy for an index generation's partitions. The default policy
// reads the partition graph off the index's navigation graph (U3). The file is large (one record
// per head, vector included), so the load starts now and the first reconfiguration -- which
// comes after the initial deployment -- waits for it only if it is still running.
func newPolicy(name, indexDir string, m *partitioning.Manifest) (placement.Policy, error) {
	graph := sync.OnceValues(func() (placement.Adjacency, error) {
		start := time.Now()
		g, err := partitioning.PartitionGraph(indexDir, m)
		if err != nil {
			log.Printf("controller: partition graph: %v", err)
			return nil, err
		}
		log.Printf("controller: partition graph of %d partitions loaded in %.1f s", len(g), time.Since(start).Seconds())
		return placement.Adjacency(g), nil
	})
	policy, err := placement.NewPolicy(name, graph)
	if err != nil {
		return nil, err
	}
	if _, ok := policy.(placement.GraphPolicy); ok {
		go graph()
	}
	return policy, nil
}

// Addrs returns the bound control-plane and API addresses (after Listening).
func (c *Controller) Addrs() (ctrlAddr, apiAddr string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.ctrlAddr, c.apiAddr
}

// Run serves until ctx ends.
func (c *Controller) Run(ctx context.Context) error {
	ln, err := net.Listen("tcp", c.cfg.Listen)
	if err != nil {
		return err
	}
	defer ln.Close()
	api, err := net.Listen("tcp", c.cfg.APIListen)
	if err != nil {
		return err
	}
	defer api.Close()
	if c.cfg.BulkListen != "" {
		// The files the controller makes for the nodes, served over the bulk protocol: a
		// node pulls them as it pulls a peer's segments, so no filesystem is shared.
		if c.cfg.WorkDir == "" {
			return fmt.Errorf("controller: bulk_listen needs work_dir")
		}
		if err := os.MkdirAll(c.cfg.WorkDir, 0o755); err != nil {
			return err
		}
		bln, err := net.Listen("tcp", c.cfg.BulkListen)
		if err != nil {
			return err
		}
		defer bln.Close()
		c.bulkAddr = advertised(bln.Addr().String(), ln.Addr().String())
		go (&transfer.Server{Root: c.cfg.WorkDir}).Serve(bln)
	}
	c.mu.Lock()
	c.ctrlAddr, c.apiAddr = ln.Addr().String(), api.Addr().String()
	c.mu.Unlock()
	c.Listening.Signal()
	log.Printf("controller: control plane on %s, API on %s; waiting for %d nodes",
		ln.Addr(), api.Addr(), c.cfg.InitialNodes)

	go ctrl.Serve(ln, c.setupAgentConn)
	go ctrl.Serve(api, c.setupAPIConn)
	go c.deployWhenReady(ctx)
	<-ctx.Done()
	for _, n := range c.nodes.All() {
		n.Conn.Close()
	}
	if c.sink != nil {
		c.sink.Close()
	}
	return nil
}

// advertised is addr as nodes reach it: a listener on every interface gets the host the
// control plane is reached at.
func advertised(addr, control string) string {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return addr
	}
	if ip := net.ParseIP(host); ip == nil || ip.IsUnspecified() {
		if h, _, err := net.SplitHostPort(control); err == nil {
			host = h
		}
	}
	return net.JoinHostPort(host, port)
}

func (c *Controller) setupAgentConn(conn *ctrl.Conn) {
	conn.Handle(protocol.RegisterM, func(ctx context.Context, body json.RawMessage) (any, error) {
		req, err := ctrl.Decode[protocol.Register](body)
		if err != nil {
			return nil, err
		}
		n := c.nodes.Add(req.Name, req.Info, conn)
		log.Printf("controller: node %d (%s) registered: data node %s, query %s, bulk %s",
			n.ID, n.Name, n.Info.NodeAddr, n.Info.QueryAddr, n.Info.BulkAddr)
		return protocol.RegisterReply{ID: n.ID}, nil
	})
	conn.OnNotify(protocol.MetricsN, func(body json.RawMessage) {
		rep, err := ctrl.Decode[metrics.Report](body)
		if err == nil && c.sink != nil {
			// Stamped on arrival, as Koala's metric collector does: one clock for the whole run,
			// whatever machines the nodes run on (their own clocks differ by an unknown offset).
			rep.Time = time.Now()
			c.sink.Write(rep)
		}
	})
	conn.OnNotify(protocol.NodeStatsN, func(body json.RawMessage) { c.nodeLines(body, "stats-%s.jsonl") })
	conn.OnNotify(protocol.NodeSamplesN, func(body json.RawMessage) { c.nodeLines(body, "cgroups.jsonl") })
}

// nodeLines appends the records a node forwarded to file (a pattern of the node's name) in
// stats_dir, each stamped with the controller's CLOCK_MONOTONIC on arrival ("t"; the node's own
// stamp, if any, becomes "node_t"): the clock of clock.json when the controller runs next to
// the experiment script, so the analysis scripts read every node's records on one time axis.
func (c *Controller) nodeLines(body json.RawMessage, file string) {
	if c.cfg.StatsDir == "" {
		return
	}
	req, err := ctrl.Decode[protocol.NodeLines](body)
	if err != nil || len(req.Lines) == 0 {
		return
	}
	now := metrics.Monotonic()
	name := file
	if strings.Contains(file, "%s") {
		name = fmt.Sprintf(file, req.Node)
	}
	var buf []byte
	for _, l := range req.Lines {
		var rec map[string]json.RawMessage
		if json.Unmarshal(l, &rec) != nil {
			continue
		}
		if t, ok := rec["t"]; ok {
			rec["node_t"] = t
		}
		rec["t"] = json.RawMessage(strconv.FormatFloat(now, 'f', 3, 64))
		b, err := json.Marshal(rec)
		if err != nil {
			continue
		}
		buf = append(append(buf, b...), '\n')
	}
	c.statsMu.Lock()
	defer c.statsMu.Unlock()
	f, err := os.OpenFile(filepath.Join(c.cfg.StatsDir, name), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		log.Printf("controller: node records: %v", err)
		return
	}
	f.Write(buf)
	f.Close()
}

func (c *Controller) setupAPIConn(conn *ctrl.Conn) {
	conn.Handle(protocol.RescaleA, func(ctx context.Context, body json.RawMessage) (any, error) {
		req, err := ctrl.Decode[protocol.RescaleReq](body)
		if err != nil {
			return nil, err
		}
		return c.Reconfigure(ctx, req)
	})
	conn.Handle(protocol.StatusA, func(ctx context.Context, _ json.RawMessage) (any, error) {
		return c.Status(), nil
	})
	conn.Handle(protocol.WaitReadyA, func(ctx context.Context, body json.RawMessage) (any, error) {
		req, _ := ctrl.Decode[protocol.WaitReadyReq](body)
		if req.TimeoutSeconds > 0 {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, time.Duration(req.TimeoutSeconds*float64(time.Second)))
			defer cancel()
		}
		if err := c.Ready.WaitContext(ctx); err != nil {
			return nil, fmt.Errorf("not ready (%s): %w", c.LastError(), err)
		}
		return c.Status(), nil
	})
	conn.Handle(protocol.BeginUpdateA, func(ctx context.Context, body json.RawMessage) (any, error) {
		req, err := ctrl.Decode[protocol.BeginUpdateReq](body)
		if err != nil {
			return nil, err
		}
		return c.BeginUpdate(ctx, req)
	})
	conn.Handle(protocol.CommitUpdateA, func(ctx context.Context, body json.RawMessage) (any, error) {
		req, err := ctrl.Decode[protocol.CommitUpdateReq](body)
		if err != nil {
			return nil, err
		}
		return c.CommitUpdate(ctx, req)
	})
	conn.Handle(protocol.AbortUpdateA, func(ctx context.Context, body json.RawMessage) (any, error) {
		req, err := ctrl.Decode[protocol.AbortUpdateReq](body)
		if err != nil {
			return nil, err
		}
		return struct{}{}, c.AbortUpdate(ctx, req)
	})
	conn.Handle(protocol.RebuildA, func(ctx context.Context, body json.RawMessage) (any, error) {
		req, err := ctrl.Decode[protocol.RebuildReq](body)
		if err != nil {
			return nil, err
		}
		return c.Rebuild(ctx, req)
	})
	conn.Handle(protocol.DesignA, func(ctx context.Context, _ json.RawMessage) (any, error) {
		var qs []design.Question
		for _, id := range design.IDs() {
			qs = append(qs, design.Open[id])
		}
		return qs, nil
	})
}

// Status reports the cluster state.
func (c *Controller) Status() protocol.ClusterStatus {
	t, _ := c.store.Current()
	c.mu.Lock()
	updating := c.lease != nil
	c.mu.Unlock()
	return protocol.ClusterStatus{
		Ready:              !c.Ready.IsBlocked(),
		Table:              t,
		Nodes:              c.nodes.Summaries(t),
		ReconfigInProgress: c.reconfigRunning.Load(),
		UpdateInProgress:   updating,
		LastError:          c.LastError(),
	}
}

// LastError is the last deployment or reconfiguration failure.
func (c *Controller) LastError() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lastErr
}

func (c *Controller) setErr(err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err == nil {
		c.lastErr = ""
	} else {
		c.lastErr = err.Error()
	}
}

func (c *Controller) deployWhenReady(ctx context.Context) {
	nodes, err := c.nodes.WaitFor(ctx, c.cfg.InitialNodes)
	if err != nil {
		return
	}
	if err := c.deployInitial(ctx, nodes); err != nil {
		log.Printf("controller: initial deployment failed: %v", err)
		c.setErr(err)
		return
	}
	c.Ready.Signal()
}

func (c *Controller) event(typ string, v float64) {
	if c.sink != nil {
		c.sink.Event("controller", typ, v)
	}
}

// forEach runs f on every node in parallel and returns the first error.
func forEach(nodes []*ManagedNode, f func(*ManagedNode) error) error {
	var wg sync.WaitGroup
	errs := make([]error, len(nodes))
	for i, n := range nodes {
		wg.Add(1)
		go func(i int, n *ManagedNode) {
			defer wg.Done()
			errs[i] = f(n)
		}(i, n)
	}
	wg.Wait()
	return errors.Join(errs...)
}

// managed resolves node IDs.
func (c *Controller) managed(ids []placement.NodeID) ([]*ManagedNode, error) {
	out := make([]*ManagedNode, 0, len(ids))
	for _, id := range ids {
		n, ok := c.nodes.Get(id)
		if !ok {
			return nil, fmt.Errorf("controller: unknown node %d", id)
		}
		out = append(out, n)
	}
	return out, nil
}
