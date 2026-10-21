package agent

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"rtier/internal/ctrl"
	"rtier/internal/epoch"
	"rtier/internal/partitioning"
	"rtier/internal/protocol"
	"rtier/internal/query"
)

// Full rebuild (U14): the cluster moves to another generation of the index -- new lists,
// partitions and graph, the same codebook, vector IDs and raw-vector locations. A node runs one
// generation at a time: SwitchGeneration drops its partitions and graph (the data node keeps the
// PQ codes and raw vectors, which every generation shares), and from then on it serves the
// generation's partitions with the generation's manifest. Before the new generation goes online,
// the old nodes replay recent client queries under its draft epoch (Warmup), so that the raw
// vectors those need are already where queries will look.

// manifestFor is the manifest of t's generation, which a node with a role in t runs.
func (a *Agent) manifestFor(t *epoch.Table) (*partitioning.Manifest, error) {
	if g := a.generation.Load(); g != t.Generation {
		return nil, fmt.Errorf("%w: node %d runs index generation %d, epoch %d is generation %d",
			query.ErrUnavailable, a.ID(), g, t.Epoch, t.Generation)
	}
	return a.manifest.Load(), nil
}

// SwitchGeneration: drop every partition and the graph, with their files, and run the
// generation the request names. The controller sends it only to a node without a role, once no
// query of an epoch it had one in is running. A node at that generation already does nothing.
func (a *Agent) handleSwitchGeneration(ctx context.Context, body json.RawMessage) (any, error) {
	req, err := ctrl.Decode[protocol.GenerationReq](body)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, a.stepTimeout()) // reads the generation's manifest and PQ file
	defer cancel()
	in, err := a.localBulk.Info(ctx)
	if err != nil {
		return nil, err
	}
	if in.Generation == req.Generation && a.generation.Load() == req.Generation {
		return struct{}{}, nil
	}
	m, err := partitioning.LoadManifest(req.PartsDir)
	if err != nil {
		return nil, err
	}
	if err := a.stopStaging(ctx); err != nil {
		return nil, err
	}
	for _, p := range in.Resident {
		if _, err := a.localBulk.EvictPartition(ctx, p); err != nil {
			return nil, err
		}
	}
	if _, err := a.localBulk.PQRelease(ctx); err != nil { // codes a staging that never ended protected
		return nil, err
	}
	if err := a.localBulk.SwitchGeneration(ctx, req.Generation, req.IndexDir, req.PartsDir); err != nil {
		return nil, fmt.Errorf("switching to index generation %d: %w", req.Generation, err)
	}
	// The files peers pull from here are the generation's from now on.
	for _, d := range []string{a.partDir(), a.graphDir()} {
		if err := os.RemoveAll(d); err != nil {
			return nil, err
		}
		if err := os.MkdirAll(d, 0o755); err != nil {
			return nil, err
		}
	}
	a.manifest.Store(m)
	a.generation.Store(req.Generation)
	a.mu.Lock()
	a.resident = map[int]bool{}
	a.ready = map[string]bool{}
	a.mu.Unlock()
	log.Printf("agent %s: index generation %d (%d partitions, %d lists)", a.cfg.Name, req.Generation,
		m.NumPartitions, m.NumLists)
	return struct{}{}, nil
}

// RawBitmap: the raw vectors the data node holds, which a full rebuild places the next
// generation's partitions by.
func (a *Agent) handleRawBitmap(ctx context.Context, _ json.RawMessage) (any, error) {
	ctx, cancel := context.WithTimeout(ctx, a.stepTimeout()) // a bit per location: 125 MB at a billion
	defer cancel()
	bm, err := a.localBulk.RawBitmap(ctx)
	if err != nil {
		return nil, err
	}
	bits := make([]byte, 8*len(bm.Words))
	for i, w := range bm.Words {
		binary.LittleEndian.PutUint64(bits[8*i:], w)
	}
	return protocol.RawBitmapReply{Locations: bm.Locations, Bits: bits}, nil
}

// keep adds a client query to the sample of recent ones (query_sample), which a full rebuild's
// warm-up replays.
func (a *Agent) keep(req *query.Request) {
	n := a.cfg.QuerySample
	if n <= 0 {
		return
	}
	q := protocol.SampledQuery{Vec: append([]byte(nil), req.Vec...), K: req.Params.K, NProbe: req.Params.NProbe,
		N: req.Params.N, EF: req.Params.EF}
	a.sampleMu.Lock()
	if len(a.sample) < n {
		a.sample = append(a.sample, q)
	} else {
		a.sample[a.sampleNext] = q
	}
	a.sampleNext = (a.sampleNext + 1) % n
	a.sampleMu.Unlock()
}

// QuerySample: the recent client queries this entry kept, oldest first.
func (a *Agent) handleQuerySample(context.Context, json.RawMessage) (any, error) {
	a.sampleMu.Lock()
	defer a.sampleMu.Unlock()
	out := protocol.QuerySampleReply{Queries: make([]protocol.SampledQuery, 0, len(a.sample))}
	start := 0
	if len(a.sample) == a.cfg.QuerySample {
		start = a.sampleNext
	}
	for i := range a.sample {
		out.Queries = append(out.Queries, a.sample[(start+i)%len(a.sample)])
	}
	return out, nil
}

// Warmup: run the queries under the request's table -- the next generation's draft epoch, which
// no client routes with yet -- as an entry would, and drop the answers. Re-ranking fetches the
// raw vectors each query needs and lacks from their holders, and they stay: when the epoch goes
// online, the queries like these find them on their owners.
func (a *Agent) handleWarmup(ctx context.Context, body json.RawMessage) (any, error) {
	req, err := ctrl.Decode[protocol.WarmupReq](body)
	if err != nil || req.Table == nil {
		return nil, fmt.Errorf("bad Warmup request: %v", err)
	}
	a.learn(req.Table)
	ctx, cancel := context.WithTimeout(ctx, a.stepTimeout())
	defer cancel()
	conc := req.Concurrency
	if conc <= 0 {
		conc = 4
	}
	start := time.Now()
	var (
		failed atomic.Int64
		first  sync.Once
		firstE string
		wg     sync.WaitGroup
	)
	sem := make(chan struct{}, conc)
	for _, q := range req.Queries {
		sem <- struct{}{}
		wg.Add(1)
		go func(q protocol.SampledQuery) {
			defer wg.Done()
			defer func() { <-sem }()
			// Pinned like a forwarded aggregation: visible to this node's WaitIdle and WaitDrained.
			release := a.tracker.Pin(req.Table.Epoch)
			defer release()
			r := &query.Request{Vec: q.Vec, Params: query.Params{K: q.K, NProbe: q.NProbe, N: q.N, EF: q.EF}}
			if _, err := a.navigateAndAggregate(ctx, req.Table, r); err != nil {
				failed.Add(1)
				first.Do(func() { firstE = err.Error() })
			}
		}(q)
	}
	wg.Wait()
	return protocol.WarmupReply{Queries: len(req.Queries), Failed: int(failed.Load()), FirstError: firstE,
		Seconds: time.Since(start).Seconds()}, nil
}
