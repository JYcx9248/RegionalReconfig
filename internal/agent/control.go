package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"rtier/internal/config"
	"rtier/internal/ctrl"
	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
	"rtier/internal/protocol"
	"rtier/internal/transfer"
)

// Longest a drain or staging step may take inside the agent.
const maxStepTime = 30 * time.Minute

func (a *Agent) registerHandlers(c *ctrl.Conn) {
	c.Handle(protocol.StageAggregatorM, a.handleStageAggregator)
	c.Handle(protocol.StagePartitionsM, a.handleStagePartitions)
	c.Handle(protocol.StageGraphM, a.handleStageGraph)
	c.Handle(protocol.StageRawM, a.handleStageRaw)
	c.Handle(protocol.InstallEpochM, a.handleInstallEpoch)
	c.Handle(protocol.WaitDrainedM, a.handleWaitDrained)
	c.Handle(protocol.EvictM, a.handleEvict)
	c.Handle(protocol.SetAdmissionM, a.handleSetAdmission)
	c.Handle(protocol.WaitIdleM, a.handleWaitIdle)
	c.Handle(protocol.NodeStatusM, a.handleStatus)
	c.Handle(protocol.ShutdownM, func(context.Context, json.RawMessage) (any, error) {
		time.AfterFunc(100*time.Millisecond, a.Stop)
		return struct{}{}, nil
	})
}

// StageAggregator: connect to (and ping) every data node of the given table, so the first
// queries after the flip do not pay connection setup (Koala: waitAllInboundPeersToConnect).
func (a *Agent) handleStageAggregator(ctx context.Context, body json.RawMessage) (any, error) {
	req, err := ctrl.Decode[protocol.StageAggregatorReq](body)
	if err != nil || req.Table == nil {
		return nil, fmt.Errorf("bad StageAggregator request: %v", err)
	}
	a.learn(req.Table)
	for _, id := range req.Table.Placement.Nodes() {
		n, ok := req.Table.Node(id)
		if !ok {
			return nil, fmt.Errorf("table has no address for node %d", id)
		}
		cctx, cancel := context.WithTimeout(ctx, 10*time.Second)
		err := a.peer(n.NodeAddr).Ping(cctx)
		cancel()
		if err != nil {
			return nil, fmt.Errorf("data node %d (%s): %w", id, n.NodeAddr, err)
		}
	}
	a.setReady(config.RoleAggregator)
	return struct{}{}, nil
}

// StagePartitions: fetch the partitions' segment files (from peers over the bulk port, or from
// a local directory for the initial deployment), bring the PQ codes they need, and load them
// into the data node.
//
// PQ codes are node-level (the decided part of U1): the data node keeps one copy per vector,
// however many of its posting lists name it. So once the segments are here, the node lists
// the codes it lacks, split by source so that each one is fetched exactly once (PQMissing);
// every source sends just those (PULL_PQ, paced by its token bucket like the files); they are
// installed (PQPut), and only then are the partitions loaded. For the initial deployment
// ("local" sources) the node reads the codes it lacks from the index instead.
//
// Raw vectors are node-level too, but a partition loads without them (U9): each is loaded with
// its source's data node as the place to fetch the vectors it lacks on demand, and StageRaw
// streams them in. The initial deployment copies them from the index instead.
func (a *Agent) handleStagePartitions(ctx context.Context, body json.RawMessage) (any, error) {
	req, err := ctrl.Decode[protocol.StagePartitionsReq](body)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, maxStepTime)
	defer cancel()
	ctx, finish := a.beginStaging(ctx)
	defer finish()
	start := time.Now()
	var (
		mu    sync.Mutex
		reply protocol.StageReply
	)
	// 1. Segment files, from all sources in parallel.
	if err := parallel(ctx, len(req.Sources), func(ctx context.Context, i int) error {
		n, err := a.fetchSegments(ctx, req.Sources[i])
		mu.Lock()
		reply.Bytes += n
		mu.Unlock()
		return err
	}); err != nil {
		return nil, err
	}
	// 2. PQ codes, 3. load.
	if err := a.stagePQAndLoad(ctx, req.Sources, &reply); err != nil {
		// Free the codes installed for partitions that did not load, and return the codes this
		// staging protected to the cache. (Every PQ_PUT has been answered by now: they are not
		// cancelled with ctx.)
		rctx, rcancel := context.WithTimeout(context.WithoutCancel(ctx), 30*time.Second)
		if freed, rerr := a.local.PQRelease(rctx); rerr != nil {
			log.Printf("agent %s: releasing staged PQ codes: %v", a.cfg.Name, rerr)
		} else if freed > 0 {
			log.Printf("agent %s: staging failed; released %d staged PQ codes", a.cfg.Name, freed)
		}
		rcancel()
		return nil, err
	}
	reply.Bytes += reply.PQBytes
	a.setReady(config.RoleData)
	a.reg.Counter("bulk.bytes_received").Add(reply.Bytes)
	a.reg.Counter("bulk.pq_codes_received").Add(reply.PQCodes)
	reply.Seconds = time.Since(start).Seconds()
	return reply, nil
}

// beginStaging registers a StagePartitions call so that stopStaging can cancel it and wait for
// it; finish unregisters it.
func (a *Agent) beginStaging(ctx context.Context) (context.Context, func()) {
	ctx, cancel := context.WithCancel(ctx)
	run := &stagingRun{cancel: cancel, done: make(chan struct{})}
	a.mu.Lock()
	a.stagings[run] = struct{}{}
	a.mu.Unlock()
	return ctx, func() {
		cancel()
		a.mu.Lock()
		delete(a.stagings, run)
		a.mu.Unlock()
		close(run.done)
	}
}

// stopStaging cancels every StagePartitions call in flight and waits until they returned (a
// staging the controller gave up on may still be running: controller timeouts do not reach
// the agent's handlers).
func (a *Agent) stopStaging(ctx context.Context) error {
	a.mu.Lock()
	runs := make([]*stagingRun, 0, len(a.stagings))
	for r := range a.stagings {
		runs = append(runs, r)
	}
	a.mu.Unlock()
	for _, r := range runs {
		r.cancel()
	}
	for _, r := range runs {
		select {
		case <-r.done:
		case <-ctx.Done():
			return fmt.Errorf("waiting for a staging to stop: %w", ctx.Err())
		}
	}
	return nil
}

// fetchSegments copies the segment files of src's partitions into the partition directory.
func (a *Agent) fetchSegments(ctx context.Context, src protocol.Source) (int64, error) {
	var names []string
	for _, p := range src.Partitions {
		files, err := partitioning.Files(a.manifest, p)
		if err != nil {
			return 0, err
		}
		names = append(names, files...)
	}
	var st transfer.Stats
	var err error
	switch src.Kind {
	case "local":
		st, err = transfer.CopyLocal(src.Dir, names, a.partDir())
	case "peer":
		remote := make([]string, len(names))
		for i, n := range names {
			remote[i] = filepath.Join("partitions", n)
		}
		st, err = transfer.Pull(ctx, src.Addr, remote, a.cfg.WorkDir, a.cfg.ChunkBytes, transfer.Data)
	default:
		err = fmt.Errorf("unknown source kind %q", src.Kind)
	}
	return st.Bytes, err
}

// maxPQRounds bounds how often stagePQAndLoad lists and pulls missing codes again.
const maxPQRounds = 3

// rawBatchBytes is the size of one request of a raw-vector stream (StageRaw).
const rawBatchBytes = 64 << 10

// stagePQAndLoad brings the PQ codes the fetched partitions need, then loads the partitions.
func (a *Agent) stagePQAndLoad(ctx context.Context, sources []protocol.Source, reply *protocol.StageReply) error {
	var pending []protocol.Source
	for _, src := range sources {
		if src.Kind != "peer" {
			for _, p := range src.Partitions { // bootstrap: the node copies what it lacks from the index
				if err := a.loadPartition(ctx, p, true, ""); err != nil {
					return err
				}
			}
			continue
		}
		if src.NodeAddr == "" {
			return fmt.Errorf("source %s has no data-node address to fetch raw vectors from", src.Addr)
		}
		pending = append(pending, src)
	}
	// A code that looked present when listed can be gone by the time its partition loads (it
	// was held only by an evicted partition that a query still used). The load then reports
	// it absent, and the codes of the partitions still to load are listed and pulled again.
	for round := 1; len(pending) > 0; round++ {
		if err := a.pullMissingPQ(ctx, pending, reply); err != nil {
			return err
		}
		var retry []protocol.Source
		for _, src := range pending {
			var again []int
			for _, p := range src.Partitions {
				if err := ctx.Err(); err != nil {
					return err
				}
				err := a.loadPartition(ctx, p, false, src.NodeAddr)
				if err != nil && nodeclient.IsPQAbsent(err) && round < maxPQRounds {
					again = append(again, p)
					continue
				}
				if err != nil {
					return err
				}
			}
			if len(again) > 0 {
				src.Partitions = again
				retry = append(retry, src)
			}
		}
		pending = retry
	}
	return nil
}

// pullMissingPQ asks the data node which PQ codes the sources' partitions need that it lacks
// (each listed under one source only) and pulls them from those sources in parallel.
func (a *Agent) pullMissingPQ(ctx context.Context, sources []protocol.Source, reply *protocol.StageReply) error {
	groups := make([][]string, len(sources))
	for i, src := range sources {
		for _, p := range src.Partitions {
			groups[i] = append(groups[i], a.segmentPath(p))
		}
	}
	missing, err := a.local.PQMissing(ctx, groups)
	if err != nil {
		return fmt.Errorf("listing the PQ codes to fetch: %w", err)
	}
	var mu sync.Mutex
	return parallel(ctx, len(sources), func(ctx context.Context, i int) error {
		st, err := transfer.PullPQ(ctx, sources[i].Addr, missing[i], 0, func(ids []uint32, m int, codes []byte) error {
			// Not cancelled with ctx: once parallel returns, every PQ_PUT has been answered, so
			// a cleanup that follows a failure sees all installed codes.
			pctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 2*time.Minute)
			defer cancel()
			_, _, err := a.local.PQPut(pctx, ids, m, codes)
			return err
		})
		mu.Lock()
		reply.PQCodes += st.Codes
		reply.PQBytes += st.Bytes
		mu.Unlock()
		if err != nil {
			return fmt.Errorf("PQ codes from %s: %w", sources[i].Addr, err)
		}
		return nil
	})
}

func (a *Agent) loadPartition(ctx context.Context, p int, fromIndex bool, rawPeer string) error {
	if err := a.local.LoadPartition(ctx, p, a.segmentPath(p), fromIndex, rawPeer); err != nil {
		return fmt.Errorf("load partition %d: %w", p, err)
	}
	a.mu.Lock()
	a.resident[p] = true
	a.mu.Unlock()
	return nil
}

func (a *Agent) segmentPath(p int) string {
	path, _ := filepath.Abs(filepath.Join(a.partDir(), partitioning.SegmentName(p)))
	return path
}

// parallel runs fn(ctx, 0..n-1) concurrently and returns the first error; the context passed
// to fn is cancelled as soon as one call fails.
func parallel(ctx context.Context, n int, fn func(ctx context.Context, i int) error) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var (
		wg    sync.WaitGroup
		mu    sync.Mutex
		first error
	)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if err := fn(ctx, i); err != nil {
				mu.Lock()
				if first == nil {
					first = err
					cancel()
				}
				mu.Unlock()
			}
		}(i)
	}
	wg.Wait()
	return first
}

// StageRaw: stream in the raw vectors that the listed partitions (loaded already) name and the
// data node lacks, each from the source listed with it -- the lazy half of a migration (U9).
// Until a vector arrives, a query that needs it has the data node fetch it from the same source
// on demand. As for PQ codes, RawMissing lists every missing vector once, under the first
// source that has it; the pulls run at the requested class on the sources' buckets (background
// by default: after the flip, the stream takes only the bandwidth nothing else needs), and
// before each batch the node is asked again what it still lacks, so that a vector a query
// fetched meanwhile is not sent again (only a batch in flight can still bring one twice).
func (a *Agent) handleStageRaw(ctx context.Context, body json.RawMessage) (any, error) {
	req, err := ctrl.Decode[protocol.StageRawReq](body)
	if err != nil {
		return nil, err
	}
	class, err := transfer.ParseClass(req.Priority)
	if err != nil {
		return nil, err
	}
	if a.beforeRaw != nil {
		if err := a.beforeRaw(); err != nil {
			return nil, err
		}
	}
	ctx, cancel := context.WithTimeout(ctx, maxStepTime)
	defer cancel()
	start := time.Now()
	groups := make([][]string, len(req.Sources))
	for i, src := range req.Sources {
		if src.Kind != "peer" {
			return nil, fmt.Errorf("StageRaw: source kind %q: raw vectors are streamed from peers", src.Kind)
		}
		for _, p := range src.Partitions {
			groups[i] = append(groups[i], a.segmentPath(p))
		}
	}
	missing, err := a.local.RawMissing(ctx, groups)
	if err != nil {
		return nil, fmt.Errorf("listing the raw vectors to fetch: %w", err)
	}
	// Requests of about rawBatchBytes: a vector that a query fetches while its batch is in
	// flight comes twice, so the batch is kept to a fraction of a second at the paced rates,
	// and still large enough to amortize the round trip and the RAW_CHECK before it.
	batch := 0
	if info, err := a.local.Info(ctx); err == nil && info.Raw.VecBytes > 0 {
		batch = max(64, rawBatchBytes/info.Raw.VecBytes)
	}
	var (
		mu    sync.Mutex
		reply protocol.StageReply
	)
	err = parallel(ctx, len(req.Sources), func(ctx context.Context, i int) error {
		still := func(locs []uint32) ([]uint32, error) { return a.local.RawAbsent(ctx, locs) }
		st, err := transfer.PullRaw(ctx, req.Sources[i].Addr, missing[i].Locs, missing[i].Lists, batch, class, still, func(locs []uint32, vb int, vecs []byte) error {
			pctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
			defer cancel()
			_, _, err := a.local.RawPut(pctx, locs, vb, vecs)
			return err
		})
		mu.Lock()
		reply.RawVectors += st.Vectors
		reply.RawBytes += st.Bytes
		mu.Unlock()
		if err != nil {
			return fmt.Errorf("raw vectors from %s: %w", req.Sources[i].Addr, err)
		}
		return nil
	})
	reply.Bytes = reply.RawBytes
	reply.Seconds = time.Since(start).Seconds()
	a.reg.Counter("bulk.bytes_received").Add(reply.RawBytes)
	a.reg.Counter("bulk.raw_vectors_received").Add(reply.RawVectors)
	if err != nil {
		return nil, err
	}
	return reply, nil
}

// StageGraph: fetch the navigation graph and load it (the entry role's prerequisite). The pull
// runs at the priority the controller asks for -- background by default, so on its source it
// only uses the bandwidth that data transfers (segments, PQ codes) leave unused.
func (a *Agent) handleStageGraph(ctx context.Context, body json.RawMessage) (any, error) {
	req, err := ctrl.Decode[protocol.StageGraphReq](body)
	if err != nil {
		return nil, err
	}
	class, err := transfer.ParseClass(req.Priority)
	if err != nil {
		return nil, err
	}
	if a.beforeGraph != nil {
		if err := a.beforeGraph(); err != nil {
			return nil, err
		}
	}
	ctx, cancel := context.WithTimeout(ctx, maxStepTime)
	defer cancel()
	const name = "graph.hnsw"
	var st transfer.Stats
	switch req.Source.Kind {
	case "local":
		st, err = transfer.CopyLocal(req.Source.Dir, []string{name}, a.graphDir())
	case "peer":
		st, err = transfer.Pull(ctx, req.Source.Addr, []string{filepath.Join("graph", name)}, a.cfg.WorkDir, a.cfg.ChunkBytes, class)
	default:
		err = fmt.Errorf("unknown source kind %q", req.Source.Kind)
	}
	if err != nil {
		return nil, err
	}
	path, _ := filepath.Abs(filepath.Join(a.graphDir(), name))
	if err := a.local.LoadGraph(ctx, path); err != nil {
		return nil, err
	}
	a.setReady(config.RoleEntry)
	a.reg.Counter("bulk.bytes_received").Add(st.Bytes)
	return protocol.StageReply{Bytes: st.Bytes, Seconds: st.Elapsed.Seconds()}, nil
}

func (a *Agent) handleInstallEpoch(ctx context.Context, body json.RawMessage) (any, error) {
	req, err := ctrl.Decode[protocol.InstallEpochReq](body)
	if err != nil || req.Table == nil {
		return nil, fmt.Errorf("bad InstallEpoch request: %v", err)
	}
	a.learn(req.Table)
	if a.beforeInstall != nil {
		a.beforeInstall(req.Table.Epoch)
	}
	if err := a.tracker.Install(req.Table); err != nil {
		return nil, err
	}
	log.Printf("agent %s: installed epoch %d", a.cfg.Name, req.Table.Epoch)
	return struct{}{}, nil
}

func (a *Agent) handleWaitDrained(ctx context.Context, body json.RawMessage) (any, error) {
	req, err := ctrl.Decode[protocol.WaitDrainedReq](body)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, maxStepTime)
	defer cancel()
	return struct{}{}, a.tracker.WaitDrained(ctx, req.Epoch)
}

// Evict: drop partitions from the data node and delete their segment files. The controller
// calls this only after the grace period, so no query can still route here for them. Their PQ
// codes and raw vectors stay on the data node as a cache (the dataset is static): a later
// reconfiguration that brings them back needs no transfer, and the new owner can still fetch
// the raw vectors from here.
func (a *Agent) handleEvict(ctx context.Context, body json.RawMessage) (any, error) {
	req, err := ctrl.Decode[protocol.EvictReq](body)
	if err != nil {
		return nil, err
	}
	if req.ReleaseStagedPQ {
		// Rollback of an aborted staging: stop a staging still running here first, so that it
		// neither loads a partition after this eviction nor installs codes after the release.
		if err := a.stopStaging(ctx); err != nil {
			return nil, err
		}
	}
	for _, p := range req.Partitions {
		if _, err := a.local.EvictPartition(ctx, p); err != nil {
			return nil, err
		}
		files, err := partitioning.Files(a.manifest, p)
		if err != nil {
			return nil, err
		}
		for _, f := range files {
			if err := os.Remove(filepath.Join(a.partDir(), f)); err != nil && !os.IsNotExist(err) {
				return nil, err
			}
		}
		a.mu.Lock()
		delete(a.resident, p)
		a.mu.Unlock()
	}
	if req.ReleaseStagedPQ {
		if _, err := a.local.PQRelease(ctx); err != nil {
			return nil, err
		}
	}
	return struct{}{}, nil
}

func (a *Agent) handleSetAdmission(ctx context.Context, body json.RawMessage) (any, error) {
	req, err := ctrl.Decode[protocol.SetAdmissionReq](body)
	if err != nil {
		return nil, err
	}
	a.paused.Store(req.Paused)
	return struct{}{}, nil
}

func (a *Agent) handleWaitIdle(ctx context.Context, _ json.RawMessage) (any, error) {
	ctx, cancel := context.WithTimeout(ctx, maxStepTime)
	defer cancel()
	return struct{}{}, a.tracker.WaitIdle(ctx)
}

func (a *Agent) handleStatus(ctx context.Context, _ json.RawMessage) (any, error) {
	st := protocol.NodeStatus{ID: a.ID(), Name: a.cfg.Name, Paused: a.paused.Load(), Ready: map[string]bool{}}
	if t := a.tracker.Current(); t != nil {
		st.Epoch = t.Epoch
	}
	a.mu.Lock()
	for r, ok := range a.ready {
		st.Ready[r] = ok
	}
	for p := range a.resident {
		st.Resident = append(st.Resident, p)
	}
	a.mu.Unlock()
	sort.Ints(st.Resident)
	if info, err := a.local.Info(ctx); err == nil {
		st.GraphLoaded = info.GraphLoaded
		st.RawFetched = info.Raw.FetchedOnDemand()
		st.RawFetchedBytes = st.RawFetched * uint64(info.Raw.VecBytes)
	}
	return st, nil
}
