// Package agent is the per-node agent (Go) that sits next to the C++ data-node service.
//
// It is Koala's worker with the dataflow parts removed (worker/worker.go,
// worker/controlPlane.go): it registers with the controller, executes control commands
// (stage, install epoch, drain, evict), serves client queries as an entry node, aggregates
// queries, serves its files to peers over the bulk-transfer port, and reports metrics.
//
// Roles and their readiness (staged in this order when a node joins):
//
//	aggregator  connections to the data nodes of the next epoch      StageAggregator
//	data        its partitions' posting lists and PQ codes loaded     StagePartitions
//	entry       the navigation graph copied and loaded                StageGraph
//
// A role is active once the controller installs an epoch that lists the node in that role. The
// raw vectors of a data node's new partitions are not a readiness condition: its rtier_node
// fetches the ones a query needs from the old owner on demand, and StageRaw streams the rest in
// (after the flip with the lazy protocol, before it with copy-then-flip).
package agent

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"rtier/internal/config"
	"rtier/internal/ctrl"
	"rtier/internal/epoch"
	"rtier/internal/metrics"
	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
	"rtier/internal/placement"
	"rtier/internal/protocol"
	"rtier/internal/query"
	"rtier/internal/syncflag"
	"rtier/internal/transfer"
)

// Options holds test hooks.
type Options struct {
	// Strategy overrides cfg.Strategy. Tests inject a fake here; production leaves it nil
	// (the global top-n strategy is open question U5).
	Strategy query.Strategy
	// Selector overrides cfg.Selector (tests).
	Selector query.Selector
	// BeforeInstall, when set, runs before the agent installs an epoch. Tests hold it to make
	// this node install an epoch later than the others.
	BeforeInstall func(epoch uint64)
	// BeforeGraph, when set, runs before the agent fetches the navigation graph; an error
	// fails the staging. Tests hold it to delay the graph, or fail it.
	BeforeGraph func() error
	// BeforeRaw, when set, runs before the agent streams raw vectors in (StageRaw); an error
	// fails it. Tests hold it so that queries must fetch raw vectors on demand.
	BeforeRaw func() error
}

// Agent is one node's agent.
type Agent struct {
	cfg      config.Agent
	manifest *partitioning.Manifest
	tracker  *epoch.Tracker
	agg      *query.Aggregator
	selector query.Selector
	limiter  *transfer.TokenBucket
	bulk     *transfer.Server
	reg      *metrics.Registry
	// The local data node, twice. Queries (NAVIGATE, and FILTER/RERANK through peer) use local;
	// everything else -- serving peers' pulls, installing what arrives, loads, evictions, INFO --
	// uses localBulk, so that it never waits for one of local's connections behind a backlog of
	// queries (a raw-vector stream served through local crawled at a few MB/s under an
	// open-loop overload while the data node answered each request in under a millisecond).
	local     *nodeclient.Client
	localBulk *nodeclient.Client

	beforeInstall func(epoch uint64) // test hooks (Options)
	beforeGraph   func() error
	beforeRaw     func() error

	id         atomic.Uint32
	paused     atomic.Bool
	info       epoch.NodeInfo
	conn       *ctrl.Conn
	cancel     context.CancelFunc
	Registered *syncflag.SyncFlag // signaled once the controller assigned an ID

	mu       sync.Mutex
	ready    map[string]bool
	resident map[int]bool
	addrs    map[placement.NodeID]epoch.NodeInfo
	peers    map[string]*nodeclient.Client // data-node clients by address
	aggs     map[string]*query.Client      // remote aggregators by query address
	stagings map[*stagingRun]struct{}      // StagePartitions calls in flight
}

// stagingRun is one StagePartitions call in flight; a rollback stops it before cleaning up.
type stagingRun struct {
	cancel context.CancelFunc
	done   chan struct{}
}

// New validates the configuration and prepares an agent; Run starts it.
func New(cfg config.Agent, opts Options) (*Agent, error) {
	for name, v := range map[string]string{
		"controller": cfg.Controller, "node_addr": cfg.NodeAddr,
		"partitions_dir": cfg.PartitionsDir, "work_dir": cfg.WorkDir,
	} {
		if v == "" {
			return nil, fmt.Errorf("agent: %s is required", name)
		}
	}
	m, err := partitioning.LoadManifest(cfg.PartitionsDir)
	if err != nil {
		return nil, err
	}
	strategy := opts.Strategy
	if strategy == nil {
		if strategy, err = query.NewStrategy(cfg.Strategy); err != nil {
			// Keep running as a data node; queries fail loudly with this error.
			log.Printf("agent %s: %v; this node cannot aggregate queries", cfg.Name, err)
			strategy = failingStrategy{err}
		}
	}
	selector := opts.Selector
	if selector == nil {
		if selector, err = query.NewSelector(cfg.Selector); err != nil {
			return nil, err
		}
	}
	adapter, err := transfer.NewAdapter(cfg.TransferAdapter, cfg.TransferRate)
	if err != nil {
		return nil, err
	}
	rate, err := adapter.Rate(transfer.Observation{})
	if err != nil {
		return nil, err
	}
	a := &Agent{
		cfg:           cfg,
		manifest:      m,
		tracker:       epoch.NewTracker(),
		selector:      selector,
		beforeInstall: opts.BeforeInstall,
		beforeGraph:   opts.BeforeGraph,
		beforeRaw:     opts.BeforeRaw,
		limiter:       transfer.NewTokenBucket(rate, 0),
		reg:           metrics.NewRegistry(),
		Registered:    syncflag.NewSyncFlag(),
		ready:         map[string]bool{},
		resident:      map[int]bool{},
		addrs:         map[placement.NodeID]epoch.NodeInfo{},
		peers:         map[string]*nodeclient.Client{},
		aggs:          map[string]*query.Client{},
		stagings:      map[*stagingRun]struct{}{},
	}
	a.agg = &query.Aggregator{Nodes: a, Strategy: strategy}
	a.bulk = &transfer.Server{Root: cfg.WorkDir, Limiter: a.limiter, ChunkBytes: cfg.ChunkBytes}
	return a, nil
}

// Counter returns the current value of one of the agent's counters (e.g. "entry.forwarded").
func (a *Agent) Counter(name string) int64 { return a.reg.Counter(name).Load() }

// ID is the node ID assigned at registration.
func (a *Agent) ID() placement.NodeID { return placement.NodeID(a.id.Load()) }

// Info is how other nodes reach this node (valid after registration).
func (a *Agent) Info() epoch.NodeInfo { return a.info }

// Run starts the agent and blocks until ctx ends or the controller connection is lost.
func (a *Agent) Run(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	a.cancel = cancel
	defer cancel()
	for _, d := range []string{a.partDir(), a.graphDir()} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return err
		}
	}

	a.local = nodeclient.New(a.cfg.NodeAddr, a.cfg.NodeConns)
	defer a.local.Close()
	a.localBulk = nodeclient.New(a.cfg.NodeAddr, a.cfg.NodeConns)
	defer a.localBulk.Close()
	info, err := a.waitForNode(ctx)
	if err != nil {
		return err
	}
	if info.NumPartitions != a.manifest.NumPartitions || info.Payload != a.manifest.Payload {
		return fmt.Errorf("agent: data node %s serves %d partitions (%s), manifest says %d (%s)",
			a.cfg.NodeAddr, info.NumPartitions, info.Payload, a.manifest.NumPartitions, a.manifest.Payload)
	}

	qln, err := net.Listen("tcp", a.cfg.QueryListen)
	if err != nil {
		return err
	}
	defer qln.Close()
	bln, err := net.Listen("tcp", a.cfg.BulkListen)
	if err != nil {
		return err
	}
	defer bln.Close()
	a.info = epoch.NodeInfo{
		Name:      a.cfg.Name,
		NodeAddr:  a.cfg.NodeAddr,
		QueryAddr: a.advertise(qln.Addr()),
		BulkAddr:  a.advertise(bln.Addr()),
	}
	a.bulk.PQ = func(ctx context.Context, ids []uint32) (int, []byte, error) {
		return a.localBulk.PQGet(ctx, ids) // peers staging our partitions pull their PQ codes
	}
	a.bulk.Raw = func(ctx context.Context, locs, lists []uint32) (int, []byte, error) {
		return a.localBulk.RawGet(ctx, locs, lists) // ... and stream their raw vectors
	}
	go a.bulk.Serve(bln)
	go a.serveQueries(ctx, qln)

	conn, err := ctrl.Dial(ctx, a.cfg.Controller)
	if err != nil {
		return fmt.Errorf("agent: controller %s: %w", a.cfg.Controller, err)
	}
	a.conn = conn
	defer conn.Close()
	a.registerHandlers(conn)
	conn.Start()
	var reply protocol.RegisterReply
	if err := conn.Call(ctx, protocol.RegisterM, protocol.Register{Name: a.cfg.Name, Info: a.info}, &reply); err != nil {
		return fmt.Errorf("agent: register: %w", err)
	}
	a.id.Store(uint32(reply.ID))
	a.info.ID = reply.ID
	log.Printf("agent %s: registered as node %d (query %s, bulk %s, data node %s)",
		a.cfg.Name, reply.ID, a.info.QueryAddr, a.info.BulkAddr, a.cfg.NodeAddr)
	a.Registered.Signal()

	go a.reportMetrics(ctx)
	select {
	case <-ctx.Done():
		return nil
	case <-conn.Done():
		if ctx.Err() != nil {
			return nil
		}
		return fmt.Errorf("agent: lost the controller: %v", conn.Err())
	}
}

// Stop ends Run.
func (a *Agent) Stop() {
	if a.cancel != nil {
		a.cancel()
	}
}

func (a *Agent) waitForNode(ctx context.Context) (*nodeclient.Info, error) {
	deadline := time.Now().Add(60 * time.Second)
	for {
		cctx, cancel := context.WithTimeout(ctx, 2*time.Second)
		info, err := a.localBulk.Info(cctx)
		cancel()
		if err == nil {
			return info, nil
		}
		if ctx.Err() != nil || time.Now().After(deadline) {
			return nil, fmt.Errorf("agent: data node %s not reachable: %w", a.cfg.NodeAddr, err)
		}
		time.Sleep(200 * time.Millisecond)
	}
}

func (a *Agent) advertise(addr net.Addr) string {
	host, port, _ := net.SplitHostPort(addr.String())
	if a.cfg.AdvertiseHost != "" {
		host = a.cfg.AdvertiseHost
	} else if ip := net.ParseIP(host); ip != nil && ip.IsUnspecified() {
		host = "127.0.0.1"
	}
	return net.JoinHostPort(host, port)
}

func (a *Agent) partDir() string  { return filepath.Join(a.cfg.WorkDir, "partitions") }
func (a *Agent) graphDir() string { return filepath.Join(a.cfg.WorkDir, "graph") }

// learn records node addresses from a table.
func (a *Agent) learn(t *epoch.Table) {
	a.mu.Lock()
	defer a.mu.Unlock()
	for id, n := range t.Nodes {
		a.addrs[id] = n
	}
}

// Client implements query.Nodes: the data-node client of node id.
func (a *Agent) Client(id placement.NodeID) (*nodeclient.Client, error) {
	a.mu.Lock()
	n, ok := a.addrs[id]
	a.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("agent: unknown node %d", id)
	}
	return a.peer(n.NodeAddr), nil
}

func (a *Agent) peer(addr string) *nodeclient.Client {
	if addr == a.cfg.NodeAddr {
		return a.local
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	c, ok := a.peers[addr]
	if !ok {
		c = nodeclient.New(addr, a.cfg.NodeConns)
		a.peers[addr] = c
	}
	return c
}

func (a *Agent) aggClient(addr string) *query.Client {
	a.mu.Lock()
	defer a.mu.Unlock()
	c, ok := a.aggs[addr]
	if !ok {
		c = query.NewClient(addr, a.cfg.NodeConns)
		a.aggs[addr] = c
	}
	return c
}

func (a *Agent) setReady(role string) {
	a.mu.Lock()
	a.ready[role] = true
	a.mu.Unlock()
}

func (a *Agent) reportMetrics(ctx context.Context) {
	iv := a.cfg.MetricsInterval.Duration
	if iv <= 0 {
		iv = 5 * time.Second
	}
	a.reg.Gauge("bulk.bytes_sent", func() float64 { return float64(a.bulk.BytesSent.Load()) })
	a.reg.Gauge("epoch", func() float64 {
		if t := a.tracker.Current(); t != nil {
			return float64(t.Epoch)
		}
		return 0
	})
	a.reg.Gauge("partitions.resident", func() float64 {
		a.mu.Lock()
		defer a.mu.Unlock()
		return float64(len(a.resident))
	})
	// PQ codes held by the data node: its GPU-memory footprint, and how much of it is cache
	// kept from partitions that moved away (sampled before each report).
	var pqResident, pqCached atomic.Uint64
	a.reg.Gauge("pq.codes_resident", func() float64 { return float64(pqResident.Load()) })
	a.reg.Gauge("pq.codes_cached", func() float64 { return float64(pqCached.Load()) })
	// Raw vectors: held, still missing after a migration, and how many queries had to fetch on
	// demand -- the warm-up of a lazy reconfiguration (U9).
	var rawPresent, rawPending, rawFetched, rawStreamed atomic.Uint64
	a.reg.Gauge("raw.present", func() float64 { return float64(rawPresent.Load()) })
	a.reg.Gauge("raw.pending", func() float64 { return float64(rawPending.Load()) })
	a.reg.Gauge("raw.fetched", func() float64 { return float64(rawFetched.Load()) })
	a.reg.Gauge("raw.streamed", func() float64 { return float64(rawStreamed.Load()) })
	tk := time.NewTicker(iv)
	defer tk.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case now := <-tk.C:
			ictx, cancel := context.WithTimeout(ctx, iv)
			if info, err := a.localBulk.Info(ictx); err == nil {
				pqResident.Store(info.PQ.Resident)
				pqCached.Store(info.PQ.Cached)
				rawPresent.Store(info.Raw.Present)
				rawPending.Store(info.Raw.Pending)
				rawFetched.Store(info.Raw.FetchedOnDemand())
				rawStreamed.Store(info.Raw.Streamed)
			}
			cancel()
			rep := metrics.Report{Node: fmt.Sprintf("%s:%d", a.cfg.Name, a.ID()), Time: now, Samples: a.reg.Snapshot()}
			if err := a.conn.Notify(protocol.MetricsN, rep); err != nil && !errors.Is(err, ctrl.ErrClosed) {
				log.Printf("agent %s: metrics: %v", a.cfg.Name, err)
			}
		}
	}
}

// failingStrategy stands in when no strategy is configured (U5).
type failingStrategy struct{ err error }

func (s failingStrategy) Name() string { return "undecided" }
func (s failingStrategy) Execute(context.Context, *query.Env, *query.Request, []query.Group) ([]nodeclient.Candidate, error) {
	return nil, s.err
}
