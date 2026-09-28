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
)

// Controller is the control plane.
type Controller struct {
	cfg      config.Controller
	manifest *partitioning.Manifest
	policy   placement.Policy
	store    epoch.Store
	sink     *metrics.Sink
	nodes    *NodeManager

	reconfigRunning atomic.Bool
	Ready           *syncflag.SyncFlag // initial deployment finished
	Listening       *syncflag.SyncFlag // listeners bound (addresses valid)

	mu        sync.Mutex
	history   *placement.History
	graphNext int // round-robin cursor over the entries that serve the graph to new nodes
	lastErr   string
	ctrlAddr  string
	apiAddr   string
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
	case "lazy", "copy-then-flip", "stop-and-copy":
	default:
		return nil, fmt.Errorf("controller: unknown protocol %q (lazy | copy-then-flip | stop-and-copy)", cfg.Protocol)
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
	m, err := partitioning.LoadManifest(cfg.PartitionsDir)
	if err != nil {
		return nil, err
	}
	policy, err := placement.NewPolicy(cfg.Placement)
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
			c.sink.Write(rep)
		}
	})
}

func (c *Controller) setupAPIConn(conn *ctrl.Conn) {
	conn.Handle(protocol.RescaleA, func(ctx context.Context, body json.RawMessage) (any, error) {
		req, err := ctrl.Decode[protocol.RescaleReq](body)
		if err != nil {
			return nil, err
		}
		return c.Rescale(ctx, req.DataNodes)
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
	return protocol.ClusterStatus{
		Ready:              !c.Ready.IsBlocked(),
		Table:              t,
		Nodes:              c.nodes.Summaries(t),
		ReconfigInProgress: c.reconfigRunning.Load(),
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
