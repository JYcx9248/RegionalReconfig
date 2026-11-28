package agent

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"os"

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
// generation's partitions with the generation's manifest.

// keep adds a client query to the sample of recent ones (QuerySample, 0 by default: none), a
// window for a planner that needs one (Quake's maintenance, branch update-strategies). With none
// kept it costs a query nothing; with a sample, a lock and a copy of the vector.
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
	// The data node refuses a generation whose lists may not fit candidate buffers sized for its
	// own (--max-candidates); refused here, before its partitions are dropped.
	if in.MaxCandidates > 0 && (req.MaxCandidates == 0 || req.MaxCandidates > in.MaxCandidates) {
		return nil, fmt.Errorf("switching to index generation %d: the data node's candidate buffers hold %d "+
			"postings (--max-candidates), and the generation's lists need %d (0: not given)",
			req.Generation, in.MaxCandidates, req.MaxCandidates)
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
	if err := a.localBulk.SwitchGeneration(ctx, req.Generation, req.IndexDir, req.PartsDir, req.MaxCandidates); err != nil {
		return nil, fmt.Errorf("switching to index generation %d: %w", req.Generation, err)
	}
	// The files peers pull from here are the generation's from now on.
	if err := os.RemoveAll(a.partDir()); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(a.partDir(), 0o755); err != nil {
		return nil, err
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
