package agent

import (
	"context"
	"encoding/json"
	"errors"
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
	"rtier/internal/placement"
	"rtier/internal/protocol"
	"rtier/internal/transfer"
)

// stepTimeout is the longest a long step may take inside the agent (config StepTimeout).
func (a *Agent) stepTimeout() time.Duration {
	if d := a.cfg.StepTimeout.Duration; d > 0 {
		return d
	}
	return 30 * time.Minute
}

func (a *Agent) registerHandlers(c *ctrl.Conn) {
	c.Handle(protocol.StageAggregatorM, a.handleStageAggregator)
	c.Handle(protocol.StagePartitionsM, a.handleStagePartitions)
	c.Handle(protocol.StageGraphM, a.handleStageGraph)
	c.Handle(protocol.StageRawM, a.handleStageRaw)
	c.Handle(protocol.DrainRawM, a.handleDrainRaw)
	c.Handle(protocol.InstallEpochM, a.handleInstallEpoch)
	c.Handle(protocol.WaitDrainedM, a.handleWaitDrained)
	c.Handle(protocol.EvictM, a.handleEvict)
	c.Handle(protocol.SavePartitionsM, a.handleSavePartitions)
	c.Handle(protocol.DropVersionsM, a.handleDropVersions)
	c.Handle(protocol.GraphPatchM, a.handleGraphPatch)
	c.Handle(protocol.SetAdmissionM, a.handleSetAdmission)
	c.Handle(protocol.WaitIdleM, a.handleWaitIdle)
	c.Handle(protocol.SwitchGenerationM, a.handleSwitchGeneration)
	c.Handle(protocol.RawBitmapM, a.handleRawBitmap)
	c.Handle(protocol.QuerySampleM, a.handleQuerySample)
	c.Handle(protocol.WarmupM, a.handleWarmup)
	c.Handle(protocol.NodeStatusM, a.handleStatus)
	c.Handle(protocol.ShutdownM, func(context.Context, json.RawMessage) (any, error) {
		time.AfterFunc(100*time.Millisecond, a.Stop)
		return struct{}{}, nil
	})
}

// StageAggregator: connect to every data node of the given table, so the first queries after
// the flip do not pay connection setup (Koala: waitAllInboundPeersToConnect). A node already
// connected is left alone: its queries keep the connections open, and a scale-out comes under
// load, when a ping would queue behind them in the pool. A new one gets the whole pool at once
// -- a connection for every slot of both kinds, FILTER's and RERANK's (Preconnect) -- every
// node in parallel, so the burst after the flip finds its connections open.
func (a *Agent) handleStageAggregator(ctx context.Context, body json.RawMessage) (any, error) {
	req, err := ctrl.Decode[protocol.StageAggregatorReq](body)
	if err != nil || req.Table == nil {
		return nil, fmt.Errorf("bad StageAggregator request: %v", err)
	}
	a.learn(req.Table)
	var (
		wg    sync.WaitGroup
		mu    sync.Mutex
		first error
	)
	for _, id := range req.Table.Placement.Nodes() {
		n, ok := req.Table.Node(id)
		if !ok {
			return nil, fmt.Errorf("table has no address for node %d", id)
		}
		c := a.peer(n.NodeAddr)
		if c.Open() > 0 {
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			cctx, cancel := context.WithTimeout(ctx, 10*time.Second)
			defer cancel()
			if err := c.Preconnect(cctx); err != nil {
				mu.Lock()
				if first == nil {
					first = fmt.Errorf("data node %d (%s): %w", id, n.NodeAddr, err)
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if first != nil {
		return nil, first
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
// Raw vectors are node-level too, but a partition loads without them (U9): every posting of the
// segment records the node holding its vector, where the data node fetches the ones it lacks on
// demand (the request names the cluster's nodes for that), and StageRaw streams them in. The
// initial deployment copies them from the index instead.
func (a *Agent) handleStagePartitions(ctx context.Context, body json.RawMessage) (any, error) {
	req, err := ctrl.Decode[protocol.StagePartitionsReq](body)
	if err != nil {
		return nil, err
	}
	if err := checkVersions(req.Sources); err != nil {
		return nil, err
	}
	a.learnNodes(req.Nodes)
	ctx, cancel := context.WithTimeout(ctx, a.stepTimeout())
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
		if freed, rerr := a.localBulk.PQRelease(rctx); rerr != nil {
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

// checkVersions: a source's versions, if any, name one version per partition.
func checkVersions(sources []protocol.Source) error {
	for _, src := range sources {
		if len(src.Versions) != 0 && len(src.Versions) != len(src.Partitions) {
			return fmt.Errorf("source %s: %d versions for %d partitions", src.Addr, len(src.Versions), len(src.Partitions))
		}
	}
	return nil
}

// fetchSegments copies the segment files of src's partitions, at their versions, into the
// partition directory.
func (a *Agent) fetchSegments(ctx context.Context, src protocol.Source) (int64, error) {
	var names []string
	for i, p := range src.Partitions {
		files, err := partitioning.FilesAt(a.manifest.Load(), p, src.VersionOf(i))
		if err != nil {
			return 0, err
		}
		names = append(names, files...)
	}
	var st transfer.Stats
	var err error
	switch {
	case src.Kind == "gen" && src.Addr != "":
		// From the controller's bulk port, under src.Dir there; then where the loads look.
		remote := make([]string, len(names))
		for i, n := range names {
			remote[i] = filepath.Join(src.Dir, n)
		}
		if st, err = transfer.Pull(ctx, src.Addr, remote, a.cfg.WorkDir, a.cfg.ChunkBytes, transfer.Data); err == nil {
			for i, n := range names {
				if err = os.Rename(filepath.Join(a.cfg.WorkDir, remote[i]), filepath.Join(a.partDir(), n)); err != nil {
					break
				}
			}
		}
	case src.Kind == "local" || src.Kind == "gen":
		st, err = transfer.CopyLocal(src.Dir, names, a.partDir())
	case src.Kind == "peer":
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

// A raw-vector stream (StageRaw: copy-then-flip's copy, lazy-stream's stream; DrainRaw's pulls
// too) splits each source's sorted locations into RawStreams contiguous ranges pulled
// concurrently, each with one request of about RawBatchBytes in flight. On the lab server (4 KB vectors, 2 sources,
// light load) the agent path moved 148 MB/s with one 64 KB request per source -- the stream as it
// was, ~3.3 ms a request whatever reads the sources had to spare -- 162 with one 1 MiB request,
// 272 with two and 406 with four. On BIGANN's 128-byte vectors locally more than one did not
// help: there the installs here bound the stream (RawStore::Put holds one lock, and buffered
// writes to one file take its inode lock). RawStreams stays 2 by default (config.DefaultAgent;
// decided 2026-10-20), and so does the fallback below for a setting of 0.
//
// By default the stream runs data node to data node (RawPath "node"): this node's data node
// pulls each batch from the holders its postings record, with RAW_GET as RERANK fetches. Through the
// agents (RawPath "agent", PULL_RAW) every byte also crossed both agents, which cost a source
// more CPU than the read itself (+0.34-0.46 CPU in its agent against +0.22-0.25 in its data node
// at ~80 MB/s): a handicap of the baselines that the protocol does not need. Only the agent path
// is paced by the token bucket, so a paced agent (TransferRateBytesPerSec) keeps its streams
// there unless RawPath says otherwise.
const (
	defaultRawStreams    = 2
	defaultRawBatchBytes = 1 << 20
)

// rawViaNode: whether StageRaw streams data node to data node (RAW_PULL) or through the agents.
func (a *Agent) rawViaNode() bool {
	switch a.cfg.RawPath {
	case "node":
		return true
	case "agent":
		return false
	}
	return a.cfg.TransferRate <= 0
}

// pullRawViaNode has the data node pull locs (lists[i] names locs[i]) from the holders its
// postings in those lists record, batch locations a request; Stats counts the vectors it
// installed (those it held already, fetched by queries meanwhile, are skipped by the node
// itself).
func (a *Agent) pullRawViaNode(ctx context.Context, locs, lists []uint32, batch, vecBytes int) (transfer.Stats, error) {
	start := time.Now()
	var st transfer.Stats
	for lo := 0; lo < len(locs); lo += batch {
		hi := min(lo+batch, len(locs))
		pctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
		installed, _, err := a.localBulk.RawPull(pctx, locs[lo:hi], lists[lo:hi])
		cancel()
		st.Vectors += int64(installed)
		st.Bytes += int64(installed) * int64(vecBytes)
		if err != nil {
			return st, err
		}
	}
	st.Elapsed = time.Since(start)
	return st, nil
}

// stagePQAndLoad brings the PQ codes the fetched partitions need, then loads the partitions.
func (a *Agent) stagePQAndLoad(ctx context.Context, sources []protocol.Source, reply *protocol.StageReply) error {
	var pending []protocol.Source
	for _, src := range sources {
		if src.Kind != "peer" {
			// Bootstrap: the node copies what it lacks from the index. A full rebuild's partitions
			// (U14): the PQ codes from the generation's index, the raw vectors from their holders.
			mode := nodeclient.LoadIndex
			if src.Kind == "gen" {
				mode = nodeclient.LoadGeneration
			} else {
				// The PQ codes of them all first, in one pass over the index's file: a node's codes
				// are scattered over the whole of it, and the loads read them code by code.
				paths := make([]string, len(src.Partitions))
				for i, p := range src.Partitions {
					paths[i] = a.segmentPath(p, src.VersionOf(i))
				}
				if _, err := a.localBulk.PQFromIndex(ctx, paths); err != nil {
					return fmt.Errorf("PQ codes from the index: %w", err)
				}
			}
			for i, p := range src.Partitions {
				if err := a.loadPartition(ctx, p, src.VersionOf(i), mode); err != nil {
					return err
				}
			}
			continue
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
			var againV []uint32
			for i, p := range src.Partitions {
				if err := ctx.Err(); err != nil {
					return err
				}
				err := a.loadPartition(ctx, p, src.VersionOf(i), nodeclient.LoadPeer)
				if err != nil && nodeclient.IsPQAbsent(err) && round < maxPQRounds {
					again, againV = append(again, p), append(againV, src.VersionOf(i))
					continue
				}
				if err != nil {
					return err
				}
			}
			if len(again) > 0 {
				src.Partitions, src.Versions = again, againV
				retry = append(retry, src)
			}
		}
		pending = retry
	}
	return nil
}

// pqStreams is how many PULL_PQ requests are in flight to one source at once.
const pqStreams = 2

// pullMissingPQ has the data node list the PQ codes the sources' partitions need that it lacks
// (each under one source only) and pulls them from those sources as the listing streams in, in
// requests of up to DefaultPQBatch codes, pqStreams at once per source: the first codes come
// while the rest are listed.
func (a *Agent) pullMissingPQ(ctx context.Context, sources []protocol.Source, reply *protocol.StageReply) error {
	groups := make([][]string, len(sources))
	for i, src := range sources {
		for j, p := range src.Partitions {
			groups[i] = append(groups[i], a.segmentPath(p, src.VersionOf(j)))
		}
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var (
		mu    sync.Mutex
		first error
		wg    sync.WaitGroup
	)
	queues := make([]chan []uint32, len(sources))
	for i := range sources {
		queues[i] = make(chan []uint32, pqStreams)
		for range pqStreams {
			wg.Add(1)
			go func() {
				defer wg.Done()
				pl := transfer.NewPuller(sources[i].Addr) // one connection for this worker's requests
				defer pl.Close()
				for ids := range queues[i] {
					if ctx.Err() != nil {
						continue // after a failure: only empties the queue
					}
					st, err := pl.PullPQ(ctx, ids, 0, func(ids []uint32, m int, codes []byte) error {
						// Not cancelled with ctx: once every pull has returned, every PQ_PUT has been
						// answered, so a cleanup that follows a failure sees all installed codes.
						pctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 2*time.Minute)
						defer cancel()
						_, _, err := a.localBulk.PQPut(pctx, ids, m, codes)
						return err
					})
					mu.Lock()
					reply.PQCodes += st.Codes
					reply.PQBytes += st.Bytes
					if err != nil && first == nil {
						first = fmt.Errorf("PQ codes from %s: %w", sources[i].Addr, err)
						cancel()
					}
					mu.Unlock()
				}
			}()
		}
	}
	lerr := a.localBulk.PQMissing(ctx, groups, func(g int, ids []uint32) error {
		for lo := 0; lo < len(ids); lo += transfer.DefaultPQBatch {
			select {
			case queues[g] <- ids[lo:min(lo+transfer.DefaultPQBatch, len(ids))]:
			case <-ctx.Done():
				return ctx.Err()
			}
		}
		return nil
	})
	for _, q := range queues {
		close(q)
	}
	wg.Wait()
	if first != nil {
		return first
	}
	if lerr != nil {
		return fmt.Errorf("listing the PQ codes to fetch: %w", lerr)
	}
	return nil
}

// loadPartition loads the segment file of version v of partition p as that version.
func (a *Agent) loadPartition(ctx context.Context, p int, v uint32, mode nodeclient.LoadMode) error {
	if err := a.localBulk.LoadPartition(ctx, p, v, a.segmentPath(p, v), mode); err != nil {
		return fmt.Errorf("load partition %d at version %d: %w", p, v, err)
	}
	a.mu.Lock()
	a.resident[p] = true
	a.mu.Unlock()
	return nil
}

// segmentPath is where the segment file of version v of partition p goes on this node.
func (a *Agent) segmentPath(p int, v uint32) string {
	path, _ := filepath.Abs(filepath.Join(a.partDir(), partitioning.SegmentNameV(p, v)))
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
// data node lacks -- the lazy half of a migration (U9). Until a vector arrives, a query that
// needs it has the data node fetch it from its holder on demand. The data node lists every
// missing vector once, with the node its postings say holds it, as a stream (RAW_MISSING), and
// each holder's vectors are pulled as their batches arrive (rawPuller); before each request the
// node checks again what it still lacks, so that a vector a query fetched meanwhile is not sent
// again (only the requests in flight can still bring one twice). On the agent path the pulls run
// at the requested class on the holders' buckets (background by default: after the flip, the
// stream takes only the bandwidth nothing else needs); the node path has no bucket and no class.
func (a *Agent) handleStageRaw(ctx context.Context, body json.RawMessage) (any, error) {
	req, err := ctrl.Decode[protocol.StageRawReq](body)
	if err != nil {
		return nil, err
	}
	if err := checkVersions(req.Sources); err != nil {
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
	ctx, cancel := context.WithTimeout(ctx, a.stepTimeout())
	defer cancel()
	start := time.Now()
	var parts []int
	for _, src := range req.Sources {
		if src.Kind != "peer" {
			return nil, fmt.Errorf("StageRaw: source kind %q: raw vectors are streamed from peers", src.Kind)
		}
		parts = append(parts, src.Partitions...)
	}
	p, err := a.newRawPuller(ctx, class)
	if err != nil {
		return nil, err
	}
	lerr := a.localBulk.RawMissing(p.ctx, parts, func(groups []nodeclient.RawRefs) error {
		for _, g := range groups {
			if err := p.add(g); err != nil {
				return err
			}
		}
		return nil
	})
	reply, err := p.wait()
	reply.Seconds = time.Since(start).Seconds()
	if err != nil {
		return nil, err
	}
	if lerr != nil {
		return nil, fmt.Errorf("listing the raw vectors to fetch: %w", lerr)
	}
	return reply, nil
}

// DrainRaw: before nodes leave (U9), take over the raw vectors that this node's postings say
// only they hold, so that they can go offline. Postings naming a leaving node whose vector is
// here already are made to name this node; the others are pulled from it, as StageRaw pulls.
// Then no posting here names it: a second look repoints what the pull brought (the postings of
// other lists naming the same vectors, and what a pull through the agents installed) and must
// find nothing left to pull.
func (a *Agent) handleDrainRaw(ctx context.Context, body json.RawMessage) (any, error) {
	req, err := ctrl.Decode[protocol.DrainRawReq](body)
	if err != nil {
		return nil, err
	}
	class, err := transfer.ParseClass(req.Priority)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, a.stepTimeout())
	defer cancel()
	a.learnNodes(req.Nodes)
	start := time.Now()
	var reply protocol.StageReply
	for _, from := range req.From {
		p, err := a.newRawPuller(ctx, class)
		if err != nil {
			return nil, err
		}
		repointed, lerr := a.localBulk.RawHeldBy(p.ctx, uint16(from), p.add)
		st, err := p.wait()
		reply.Repointed += int64(repointed)
		reply.RawVectors += st.RawVectors
		reply.RawBytes += st.RawBytes
		if err == nil {
			err = lerr
		}
		if err != nil {
			return nil, fmt.Errorf("draining node %d: %w", from, err)
		}
		left := 0
		again, err := a.localBulk.RawHeldBy(ctx, uint16(from), func(m nodeclient.RawRefs) error {
			left += len(m.Locs)
			return nil
		})
		if err != nil {
			return nil, fmt.Errorf("draining node %d: %w", from, err)
		}
		reply.Repointed += int64(again)
		if left > 0 {
			return nil, fmt.Errorf("draining node %d: %d raw vectors are still only there", from, left)
		}
	}
	reply.Seconds = time.Since(start).Seconds()
	return reply, nil
}

// rawPuller brings raw vectors into the data node as their listing streams in (RAW_MISSING,
// RAW_HELD_BY), in requests of about RawBatchBytes, RawStreams at once per holder: data node to
// data node (RAW_PULL: the data node asks the holders its postings record), or through the agents
// (PULL_RAW from the holder's bulk port, at class on its bucket). The listing names every vector
// once, so none comes twice -- but for one a query fetches while its request is in flight, which
// is why a request is kept to a fraction of a second at the paced rates, and still large enough
// to amortize the round trip and the check before it.
type rawPuller struct {
	a                 *Agent
	ctx               context.Context
	cancel            context.CancelFunc
	class             transfer.Class
	viaNode           bool
	batch             int // locations per request
	vecBytes, streams int

	mu     sync.Mutex
	queues map[uint16]chan nodeclient.RawRefs
	wg     sync.WaitGroup
	reply  protocol.StageReply
	first  error
}

func (a *Agent) newRawPuller(ctx context.Context, class transfer.Class) (*rawPuller, error) {
	streams, batchBytes := a.cfg.RawStreams, a.cfg.RawBatchBytes
	if streams <= 0 {
		streams = defaultRawStreams
	}
	if batchBytes <= 0 {
		batchBytes = defaultRawBatchBytes
	}
	vecBytes := 0
	if info, err := a.localBulk.Info(ctx); err == nil {
		vecBytes = info.Raw.VecBytes
	}
	batch := transfer.DefaultRawBatch
	if vecBytes > 0 {
		batch = max(64, batchBytes/vecBytes)
	}
	viaNode := a.rawViaNode()
	if viaNode && vecBytes <= 0 {
		return nil, errors.New("the data node does not report its raw-vector size")
	}
	ctx, cancel := context.WithCancel(ctx)
	return &rawPuller{a: a, ctx: ctx, cancel: cancel, class: class, viaNode: viaNode, batch: batch,
		vecBytes: vecBytes, streams: streams, queues: map[uint16]chan nodeclient.RawRefs{}}, nil
}

// add queues the vectors of g for their holder's pulls, a request at a time; it waits while
// they are behind, so the slowest holder paces the listing, and fails once a pull has failed.
func (p *rawPuller) add(g nodeclient.RawRefs) error {
	q := p.queue(g.Holder)
	for lo := 0; lo < len(g.Locs); lo += p.batch {
		hi := min(lo+p.batch, len(g.Locs))
		select {
		case q <- nodeclient.RawRefs{Holder: g.Holder, Locs: g.Locs[lo:hi], Lists: g.Lists[lo:hi]}:
		case <-p.ctx.Done():
			return p.failure()
		}
	}
	return nil
}

// queue is holder's queue, with its pulls started on first use.
func (p *rawPuller) queue(holder uint16) chan nodeclient.RawRefs {
	p.mu.Lock()
	defer p.mu.Unlock()
	q, ok := p.queues[holder]
	if !ok {
		q = make(chan nodeclient.RawRefs, p.streams)
		p.queues[holder] = q
		for range p.streams {
			p.wg.Add(1)
			go p.run(q)
		}
	}
	return q
}

func (p *rawPuller) run(q chan nodeclient.RawRefs) {
	defer p.wg.Done()
	var pl *transfer.Puller // the agent path: this worker's connection to the holder, kept across its requests
	defer func() {
		if pl != nil {
			pl.Close()
		}
	}()
	for r := range q {
		if p.ctx.Err() != nil {
			continue // after a failure: only empties the queue
		}
		var st transfer.Stats
		var err error
		switch {
		case p.viaNode:
			st, err = p.a.pullRawViaNode(p.ctx, r.Locs, r.Lists, p.batch, p.vecBytes)
		default:
			if pl == nil {
				pl, err = p.puller(r.Holder)
			}
			if err == nil {
				st, err = p.pull(pl, r)
			}
		}
		p.mu.Lock()
		p.reply.RawVectors += st.Vectors
		p.reply.RawBytes += st.Bytes
		if err != nil && p.first == nil {
			p.first = fmt.Errorf("raw vectors from node %d: %w", r.Holder, err)
			p.cancel()
		}
		p.mu.Unlock()
	}
}

// puller is a connection to holder's bulk port.
func (p *rawPuller) puller(holder uint16) (*transfer.Puller, error) {
	n, ok := p.a.nodeInfo(placement.NodeID(holder))
	if !ok {
		return nil, errors.New("this agent does not know the node's address")
	}
	return transfer.NewPuller(n.BulkAddr), nil
}

// pull brings r's vectors through the agents, on pl.
func (p *rawPuller) pull(pl *transfer.Puller, r nodeclient.RawRefs) (transfer.Stats, error) {
	ctx := p.ctx
	still := func(locs []uint32) ([]uint32, error) { return p.a.localBulk.RawAbsent(ctx, locs) }
	return pl.PullRaw(ctx, r.Locs, p.batch, p.class, still, func(locs []uint32, vb int, vecs []byte) error {
		pctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
		defer cancel()
		_, _, err := p.a.localBulk.RawPut(pctx, locs, vb, vecs)
		return err
	})
}

// failure is the first pull's error, else the context's.
func (p *rawPuller) failure() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.first != nil {
		return p.first
	}
	return p.ctx.Err()
}

// wait waits for the pulls queued so far and returns what they brought.
func (p *rawPuller) wait() (protocol.StageReply, error) {
	p.mu.Lock()
	for _, q := range p.queues {
		close(q)
	}
	p.mu.Unlock()
	p.wg.Wait()
	p.cancel()
	reply := p.reply
	reply.Bytes = reply.RawBytes
	p.a.reg.Counter("bulk.bytes_received").Add(reply.RawBytes)
	p.a.reg.Counter("bulk.raw_vectors_received").Add(reply.RawVectors)
	return reply, p.first
}

// StageGraph: fetch the navigation graph and load it (the entry role's prerequisite). The pull
// runs at the priority the controller asks for -- background by default: then it moves nothing
// while its source or this node moves data, raw vectors included (graphYield).
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
	ctx, cancel := context.WithTimeout(ctx, a.stepTimeout())
	defer cancel()
	var gate func(context.Context) error // this end's half of the yield; the source holds its own
	if class == transfer.Background {
		gate = a.yield.Wait
	}
	const name = "graph.hnsw"
	var st transfer.Stats
	switch req.Source.Kind {
	case "local":
		st, err = transfer.CopyLocal(req.Source.Dir, []string{name}, a.graphDir())
	case "peer":
		st, err = transfer.PullGated(ctx, req.Source.Addr, []string{filepath.Join("graph", name)}, a.cfg.WorkDir,
			a.cfg.ChunkBytes, class, gate)
	default:
		err = fmt.Errorf("unknown source kind %q", req.Source.Kind)
	}
	if err != nil {
		return nil, err
	}
	path, _ := filepath.Abs(filepath.Join(a.graphDir(), name))
	if err := a.localBulk.LoadGraph(ctx, path); err != nil {
		return nil, err
	}
	// The patches major epochs made since the build (U14), in order, from the same source: the
	// graph is then at the version the epoch searches.
	for v := uint32(1); v <= req.Graph; v++ {
		file := patchName(v)
		if req.Source.Kind != "peer" {
			return nil, fmt.Errorf("graph version %d: a %s graph source has no patches", req.Graph, req.Source.Kind)
		}
		pst, err := transfer.PullGated(ctx, req.Source.Addr, []string{filepath.Join("graph", file)}, a.cfg.WorkDir,
			a.cfg.ChunkBytes, class, gate)
		if err != nil {
			return nil, err
		}
		st.Bytes += pst.Bytes
		st.Elapsed += pst.Elapsed
		pp, _ := filepath.Abs(filepath.Join(a.graphDir(), file))
		if err := a.localBulk.PatchLoad(ctx, pp); err != nil {
			return nil, err
		}
		if err := a.localBulk.PatchMerge(ctx, v); err != nil {
			return nil, err
		}
	}
	a.setReady(config.RoleEntry)
	a.reg.Counter("bulk.bytes_received").Add(st.Bytes)
	return protocol.StageReply{Bytes: st.Bytes, Seconds: st.Elapsed.Seconds()}, nil
}

// patchName is the file a graph patch is kept in, in the graph directory.
func patchName(v uint32) string { return fmt.Sprintf("patch-%d.bin", v) }

// GraphPatch (U14 major epochs): once no query searches the graph without the patch -- the
// controller calls this after the grace period -- the entry writes the patch next to the graph,
// where a later entry's graph transfer takes it, and merges it in place: queries searching
// through the patch go on meanwhile. With Drop (a commit that did not flip), it unstages it.
func (a *Agent) handleGraphPatch(ctx context.Context, body json.RawMessage) (any, error) {
	req, err := ctrl.Decode[protocol.GraphPatchReq](body)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, a.stepTimeout()) // a merge in place grows with the graph
	defer cancel()
	if req.Drop {
		_, err := a.localBulk.PatchDrop(ctx, req.Version)
		return struct{}{}, err
	}
	info, err := a.localBulk.Info(ctx)
	if err != nil {
		return nil, err
	}
	if info.Graph.Version >= req.Version {
		return struct{}{}, nil // merged already (a retry): its file was written first
	}
	if err := os.MkdirAll(a.graphDir(), 0o755); err != nil {
		return nil, err
	}
	path, _ := filepath.Abs(filepath.Join(a.graphDir(), patchName(req.Version)))
	if err := a.localBulk.PatchSave(ctx, req.Version, path); err != nil {
		return nil, err
	}
	return struct{}{}, a.localBulk.PatchMerge(ctx, req.Version)
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
	ctx, cancel := context.WithTimeout(ctx, a.stepTimeout())
	defer cancel()
	return struct{}{}, a.tracker.WaitDrained(ctx, req.Epoch)
}

// Evict: drop partitions from the data node -- every version of their lists -- and delete their
// segment files. The controller calls this only after the grace period, so no query can still
// route here for them. Their PQ codes and raw vectors stay on the data node as a cache (vectors
// are immutable: an update deletes and inserts, U14): a later reconfiguration that brings them
// back needs no transfer, and the new owner can still fetch the raw vectors from here.
func (a *Agent) handleEvict(ctx context.Context, body json.RawMessage) (any, error) {
	req, err := ctrl.Decode[protocol.EvictReq](body)
	if err != nil {
		return nil, err
	}
	// An eviction drops a reference per posting (the PQ store's reclaimer): a whole partition
	// takes seconds at a billion vectors, past a request's default timeout.
	ctx, cancel := context.WithTimeout(ctx, a.stepTimeout())
	defer cancel()
	if req.ReleaseStagedPQ {
		// Rollback of an aborted staging: stop a staging still running here first, so that it
		// neither loads a partition after this eviction nor installs codes after the release.
		if err := a.stopStaging(ctx); err != nil {
			return nil, err
		}
	}
	for _, p := range req.Partitions {
		if _, err := a.localBulk.EvictPartition(ctx, p); err != nil {
			return nil, err
		}
		files, err := partitioning.Files(a.manifest.Load(), p)
		if err != nil {
			return nil, err
		}
		later, err := filepath.Glob(filepath.Join(a.partDir(), partitioning.SegmentVersionsGlob(p)))
		if err != nil {
			return nil, err
		}
		for _, f := range files {
			later = append(later, filepath.Join(a.partDir(), f))
		}
		for _, f := range later {
			if err := os.Remove(f); err != nil && !os.IsNotExist(err) {
				return nil, err
			}
		}
		a.mu.Lock()
		delete(a.resident, p)
		a.mu.Unlock()
	}
	if req.ReleaseStagedPQ {
		if _, err := a.localBulk.PQRelease(ctx); err != nil {
			return nil, err
		}
	}
	return struct{}{}, nil
}

// SavePartitions: write versions of resident partitions as segment files in the partition
// directory, where the nodes they move to pull them. A version that an update batch made lives
// only in the data node's memory until a move needs it as a file (U14), so a batch writes no
// segment; the controller asks the old owner for the file when the partition moves. The data
// node writes it (it holds the lists) and the bulk port serves it, as any other segment file.
func (a *Agent) handleSavePartitions(ctx context.Context, body json.RawMessage) (any, error) {
	req, err := ctrl.Decode[protocol.SavePartitionsReq](body)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, a.stepTimeout())
	defer cancel()
	a.saving.Add(1) // the moved lists on their way out: the graph waits (graphYield)
	defer a.saving.Add(-1)
	for _, pv := range req.Versions {
		if err := a.localBulk.SavePartition(ctx, pv.Partition, pv.Version, a.segmentPath(pv.Partition, pv.Version)); err != nil {
			return nil, fmt.Errorf("save partition %d at version %d: %w", pv.Partition, pv.Version, err)
		}
	}
	return struct{}{}, nil
}

// forgetBatch bounds the IDs of one FORGET (its frame).
const forgetBatch = 1 << 22

// DropVersions: an update batch's reclaim on this node (U14), after the grace period of the
// epochs that read the replaced versions. First the deleted vectors: their PQ codes go now
// where nothing names them, and the ones the replaced versions name are marked, so that they go
// with those versions below instead of becoming cached codes -- a deleted vector is no cache.
// Then the replaced versions go, with their segment files if a move wrote one; the partitions
// stay resident at their newer versions.
func (a *Agent) handleDropVersions(ctx context.Context, body json.RawMessage) (any, error) {
	req, err := ctrl.Decode[protocol.DropVersionsReq](body)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, a.stepTimeout()) // as an eviction: a reference per posting
	defer cancel()
	for lo := 0; lo < len(req.Deleted); lo += forgetBatch {
		if _, err := a.localBulk.ForgetVectors(ctx, req.Deleted[lo:min(lo+forgetBatch, len(req.Deleted))], true); err != nil {
			return nil, fmt.Errorf("freeing the PQ codes of deleted vectors: %w", err)
		}
	}
	for _, pv := range req.Versions {
		if _, err := a.localBulk.DropVersion(ctx, pv.Partition, pv.Version); err != nil {
			return nil, fmt.Errorf("drop partition %d at version %d: %w", pv.Partition, pv.Version, err)
		}
		if err := os.Remove(a.segmentPath(pv.Partition, pv.Version)); err != nil && !os.IsNotExist(err) {
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
	ctx, cancel := context.WithTimeout(ctx, a.stepTimeout())
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
	if info, err := a.localBulk.Info(ctx); err == nil {
		st.GraphLoaded = info.GraphLoaded
		st.Graph, st.GraphPatch = info.Graph.Version, info.Graph.Patch
		st.Generation = info.Generation
		st.RawFetched = info.Raw.FetchedOnDemand()
		st.RawFetchedBytes = st.RawFetched * uint64(info.Raw.VecBytes)
		for _, pv := range info.Versions {
			st.Versions = append(st.Versions, protocol.PartitionVersion{Partition: int(pv[0]), Version: pv[1]})
		}
	}
	return st, nil
}
