package controller

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"rtier/internal/epoch"
	"rtier/internal/partitioning"
	"rtier/internal/placement"
	"rtier/internal/protocol"
)

// Full rebuild (U14): the cluster moves to another generation of the index -- all heads, the
// graph, every list and partition -- built outside the system with the current codebook, vector
// IDs and raw-vector locations, so that the PQ codes and raw vectors every node holds stay valid.
// Queries keep running and get exact answers throughout: those of an epoch of the old generation
// read the old lists and graph, those of the new generation's epochs the new ones, and no node
// holds two generations' lists or graphs at any time ("one graph copy per node"):
//
//  1. EVACUATE  idle nodes, the spares, take over every partition by a lazy reconfiguration
//     (copyThenFlip): they pull the lists, the PQ codes and the graph from the old nodes, and
//     fetch raw vectors from them on demand. The old nodes stay members, holding raw vectors
//     only, and are not drained.
//  2. DEMOTE    once the spares are entries, a flip takes the entry and aggregator roles from the
//     old nodes; after its grace period no query uses them but for raw vectors.
//  3. SWITCH    the old nodes drop the old lists and graph and run the new generation; their PQ
//     codes and raw vectors stay, and keep being served to the spares.
//  4. PLACE     the new partitions go where most of their raw vectors are already (by the old
//     nodes' presence bitmaps, count-balanced: placement.Affinity), and every posting whose
//     vector its owner lacks names an old node that holds it: one hop.
//  5. STAGE     the old nodes load their new partitions (PQ codes from the generation's index)
//     and the new graph.
//  6. WARM-UP   they replay the queries the entries kept, under the new generation's draft
//     epoch: what re-ranking fetches stays, so the vectors those queries need are on their owners
//     before any client query arrives.
//  7. FLIP      the new generation goes online on the old nodes; the spares lose every role.
//  8. GRACE     the old generation's queries finish on the spares.
//  9. RELEASE   the spares move to the new generation too, which drops what they held, and are
//     idle again: nothing has to be drained from them, since they only hold copies.
//
// Failures (U13, out of scope): before 1's flip the old epoch stays; after it, the protocol stops
// where it is -- before 7 the spares serve the old generation, with the old nodes as holders.

// generation is an index generation the cluster can run.
type generation struct {
	id       uint32
	indexDir string
	partsDir string
	manifest *partitioning.Manifest
}

// genDirs are the directories of t's generation: the configured ones for the deployed build.
func (c *Controller) genDirs(t *epoch.Table) (index, parts string) {
	if t == nil || t.Generation == 0 {
		return c.cfg.IndexDir, c.cfg.PartitionsDir
	}
	return t.GenIndex, t.GenParts
}

// toGeneration brings nodes to t's index generation (a node that registered with the deployed
// build's configuration, or a spare released by a rebuild): a node holding no role in t drops
// what it held and runs t's lists and graph from then on. A node at that generation does nothing.
func (c *Controller) toGeneration(ctx context.Context, nodes []*ManagedNode, t *epoch.Table) error {
	index, parts := c.genDirs(t)
	req := protocol.GenerationReq{Generation: t.Generation, IndexDir: index, PartsDir: parts}
	return forEach(nodes, func(n *ManagedNode) error {
		return n.call(ctx, c.stageTimeout(), protocol.SwitchGenerationM, req, nil)
	})
}

// openGeneration checks the next generation src names against t's: the same codebook (the PQ
// codes every node holds stay valid) and the same pages (the raw vectors' locations). The data
// nodes check the rest when they switch.
func (c *Controller) openGeneration(t *epoch.Table, src protocol.GenerationSource) (*generation, error) {
	if src.Kind != "" && src.Kind != "dir" {
		return nil, fmt.Errorf("controller: generation source kind %q (dir)", src.Kind)
	}
	if src.IndexDir == "" || src.PartsDir == "" {
		return nil, errors.New("controller: a rebuild needs the next generation's index_dir and parts_dir")
	}
	g := &generation{id: t.Generation + 1}
	var err error
	if g.indexDir, err = filepath.Abs(src.IndexDir); err != nil {
		return nil, err
	}
	if g.partsDir, err = filepath.Abs(src.PartsDir); err != nil {
		return nil, err
	}
	if g.manifest, err = partitioning.LoadManifest(g.partsDir); err != nil {
		return nil, err
	}
	curIndex, curParts := c.genDirs(t)
	cur, err := partitioning.LoadManifest(curParts)
	if err != nil {
		return nil, err
	}
	if g.manifest.Payload != cur.Payload || g.manifest.VectorsPerPage != cur.VectorsPerPage {
		return nil, fmt.Errorf("controller: generation %d (%s) refers to other pages than generation %d: a full rebuild "+
			"keeps the vectors' locations", g.id, g.partsDir, t.Generation)
	}
	a, err := os.ReadFile(filepath.Join(curIndex, "pq_codebook.bin"))
	if err != nil {
		return nil, err
	}
	b, err := os.ReadFile(filepath.Join(g.indexDir, "pq_codebook.bin"))
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(a, b) {
		return nil, fmt.Errorf("controller: generation %d (%s) has another PQ codebook: a full rebuild keeps the codebook",
			g.id, g.indexDir)
	}
	return g, nil
}

// Rebuild moves the cluster to the index generation req names (a full rebuild, U14), one at a
// time with reconfigurations and update batches; like them it runs to completion even if the
// caller disconnects.
func (c *Controller) Rebuild(ctx context.Context, req protocol.RebuildReq) (*protocol.RebuildReply, error) {
	ctx = context.WithoutCancel(ctx)
	if c.Ready.IsBlocked() {
		return nil, fmt.Errorf("controller: initial deployment has not finished (%s)", c.LastError())
	}
	if !c.reconfigRunning.CompareAndSwap(false, true) {
		return nil, fmt.Errorf("controller: another reconfiguration or an update batch is in progress, retry later")
	}
	defer c.reconfigRunning.Store(false)
	reply, err := c.rebuild(ctx, req)
	c.setErr(err)
	return reply, err
}

func (c *Controller) rebuild(ctx context.Context, req protocol.RebuildReq) (*protocol.RebuildReply, error) {
	cur, err := c.store.Current()
	if err != nil {
		return nil, err
	}
	switch req.Placement {
	case "", "affinity", "random":
	default:
		return nil, fmt.Errorf("controller: rebuild placement %q (affinity | random)", req.Placement)
	}
	gen, err := c.openGeneration(cur, req.Source)
	if err != nil {
		return nil, err
	}
	old, err := c.managed(sortedKeys(cur.Nodes))
	if err != nil {
		return nil, err
	}
	reply := &protocol.RebuildReply{FromEpoch: cur.Epoch, ToEpoch: cur.Epoch, Generation: gen.id, Nodes: ids(old),
		Phases: map[string]float64{}}
	mark := time.Now()
	phase := func(name string) {
		s := time.Since(mark).Seconds()
		reply.Phases[name] = s
		c.event("rebuild."+name+"_s", s)
		mark = time.Now()
	}
	c.event("rebuild.start", float64(cur.Epoch))
	log.Printf("controller: full rebuild to index generation %d (%s), epoch %d, %d nodes", gen.id, gen.partsDir,
		cur.Epoch, len(old))
	if req.StopTheWorld {
		return c.rebuildInPlace(ctx, cur, gen, old, req, reply, phase)
	}

	n := req.Spares
	if n <= 0 {
		n = len(cur.Placement.Nodes())
	}
	idle := c.nodes.Idle(cur)
	if len(idle) < n {
		return reply, fmt.Errorf("controller: a rebuild with %d spares needs %d idle nodes, %d registered", n, n, len(idle))
	}
	spares := idle[:n]
	reply.Spares = ids(spares)
	all := append(append([]*ManagedNode(nil), old...), spares...)
	fetchedBefore, fetchedBytesBefore := c.rawFetched(ctx, all)

	// 1. EVACUATE: the spares take every partition lazily; the old nodes stay members (removed:
	// none) and keep their raw vectors.
	next, err := c.policy.Initial(cur.Placement.NumPartitions(), ids(spares))
	if err != nil {
		return reply, err
	}
	evac, err := c.copyThenFlip(ctx, cur, next, placement.Diff(cur.Placement, next), spares, nil, false, rawOnDemand)
	reply.Evacuation = evac
	if evac != nil {
		reply.ToEpoch = evac.ToEpoch
	}
	if err != nil {
		return reply, fmt.Errorf("controller: rebuild: the spares did not take over: %w", err)
	}
	phase("evacuate")

	// 2. DEMOTE: the spares are the entries and aggregators; the old nodes, members without a
	// role, only serve the raw vectors the spares' postings say they hold.
	t1, err := c.store.Current()
	if err != nil {
		return reply, err
	}
	isSpare := map[placement.NodeID]bool{}
	for _, s := range spares {
		isSpare[s.ID] = true
		if !t1.IsEntry(s.ID) {
			return reply, fmt.Errorf("controller: rebuild: spare %d is not an entry of epoch %d: its graph did not load", s.ID, t1.Epoch)
		}
	}
	t2 := t1.Clone()
	t2.Epoch = t1.Epoch + 1
	t2.Entries, t2.Aggregators = nil, nil
	for _, id := range t1.Entries {
		if isSpare[id] {
			t2.Entries = append(t2.Entries, id)
		}
	}
	for _, id := range t1.Aggregators {
		if isSpare[id] {
			t2.Aggregators = append(t2.Aggregators, id)
		}
	}
	if len(t2.Aggregators) == 0 {
		t2.Aggregators = ids(spares)
	}
	if err := c.flip(ctx, t1.Epoch, t2, all); err != nil {
		return reply, err
	}
	reply.ToEpoch = t2.Epoch
	if err := c.waitDrained(ctx, all, t1.Epoch); err != nil {
		return reply, err
	}
	phase("demote")

	// 3-6 on the old nodes, which nothing but the spares' raw-vector fetches reaches now. The
	// warm-up replays the queries every node kept as an entry: the old nodes' from before the
	// spares took over, the spares' since.
	sample, err := c.querySample(ctx, all, req.Warmup)
	if err != nil {
		return reply, err
	}
	draft, err := c.stageGeneration(ctx, t2, gen, old, sample, req, reply, phase)
	if err != nil {
		return reply, fmt.Errorf("controller: rebuild stopped before the flip to generation %d (the spares serve generation "+
			"%d): %w", gen.id, t2.Generation, err)
	}

	// 7. FLIP: the new generation goes online on the old nodes.
	if c.BeforeGenerationFlip != nil {
		c.BeforeGenerationFlip(draft.Epoch)
	}
	if err := c.flip(ctx, t2.Epoch, draft, all); err != nil {
		return reply, err
	}
	reply.ToEpoch = draft.Epoch
	phase("flip")

	// 8. GRACE: the old generation's queries finish on the spares.
	if err := c.waitDrained(ctx, all, t2.Epoch); err != nil {
		return reply, err
	}
	phase("drain")

	// 9. RELEASE: the spares hold copies only -- nothing to drain. Moving them to the new
	// generation drops the old lists and graph, and readies them for a later scale-out.
	if err := c.toGeneration(ctx, spares, draft); err != nil {
		return reply, err
	}
	if err := c.adopt(gen, draft.Placement); err != nil {
		return reply, err
	}
	phase("release")
	fetched, fetchedBytes := c.rawFetched(ctx, all)
	reply.RawFetched, reply.RawFetchedBytes = fetched-fetchedBefore, fetchedBytes-fetchedBytesBefore
	log.Printf("controller: full rebuild done: epoch %d -> %d, generation %d on %v, %d of %d vectors local, warm-up "+
		"%d queries fetched %d vectors, phases %v", reply.FromEpoch, reply.ToEpoch, gen.id, reply.Nodes, reply.Local,
		reply.Vectors, reply.WarmupQueries, reply.WarmupFetched, reply.Phases)
	return reply, nil
}

// rebuildInPlace is the baseline without spares: admission paused on every entry while all the
// nodes switch to the new generation where they are; the cluster answers nothing meanwhile.
func (c *Controller) rebuildInPlace(ctx context.Context, cur *epoch.Table, gen *generation, nodes []*ManagedNode,
	req protocol.RebuildReq, reply *protocol.RebuildReply, phase func(string)) (*protocol.RebuildReply, error) {
	fetchedBefore, fetchedBytesBefore := c.rawFetched(ctx, nodes)
	sample, err := c.querySample(ctx, nodes, req.Warmup)
	if err != nil {
		return reply, err
	}
	entries, err := c.managed(cur.Entries)
	if err != nil {
		return reply, err
	}
	admission := func(paused bool) error {
		return forEach(entries, func(n *ManagedNode) error {
			return n.call(context.Background(), c.callTimeout(), protocol.SetAdmissionM, protocol.SetAdmissionReq{Paused: paused}, nil)
		})
	}
	if err := admission(true); err != nil {
		return reply, err
	}
	resumed := false
	defer func() {
		if !resumed {
			if err := admission(false); err != nil {
				log.Printf("controller: resuming admission: %v", err)
			}
		}
	}()
	if err := forEach(nodes, func(n *ManagedNode) error {
		return n.call(ctx, c.stageTimeout(), protocol.WaitIdleM, struct{}{}, nil)
	}); err != nil {
		return reply, err
	}
	phase("pause")
	draft, err := c.stageGeneration(ctx, cur, gen, nodes, sample, req, reply, phase)
	if err != nil {
		return reply, fmt.Errorf("controller: rebuild stopped before the flip to generation %d: %w", gen.id, err)
	}
	if c.BeforeGenerationFlip != nil {
		c.BeforeGenerationFlip(draft.Epoch)
	}
	if err := c.flip(ctx, cur.Epoch, draft, nodes); err != nil {
		return reply, err
	}
	reply.ToEpoch = draft.Epoch
	// The new generation's entries are every node: all of them admit queries again.
	if entries, err = c.managed(draft.Entries); err != nil {
		return reply, err
	}
	resumed = true
	if err := admission(false); err != nil {
		return reply, err
	}
	phase("flip")
	if err := c.adopt(gen, draft.Placement); err != nil {
		return reply, err
	}
	fetched, fetchedBytes := c.rawFetched(ctx, nodes)
	reply.RawFetched, reply.RawFetchedBytes = fetched-fetchedBefore, fetchedBytes-fetchedBytesBefore
	log.Printf("controller: full rebuild (stop-the-world) done: epoch %d -> %d, generation %d, phases %v",
		reply.FromEpoch, reply.ToEpoch, gen.id, reply.Phases)
	return reply, nil
}

// stageGeneration readies the new generation on nodes, which hold no role in t (or, stopped,
// serve nothing): they switch to it (3), its partitions are placed by where their raw vectors
// are (4), they load them and its graph (5), and replay sample under its draft epoch (6). It
// returns the draft epoch, to flip to from t.
func (c *Controller) stageGeneration(ctx context.Context, t *epoch.Table, gen *generation, nodes []*ManagedNode,
	sample []protocol.SampledQuery, req protocol.RebuildReq, reply *protocol.RebuildReply, phase func(string)) (*epoch.Table, error) {

	// 3. SWITCH.
	if err := forEach(nodes, func(n *ManagedNode) error {
		return n.call(ctx, c.stageTimeout(), protocol.SwitchGenerationM,
			protocol.GenerationReq{Generation: gen.id, IndexDir: gen.indexDir, PartsDir: gen.partsDir}, nil)
	}); err != nil {
		return nil, err
	}
	phase("switch")

	// 4. PLACE. The segments with holders are the controller's to hand out: with bulk_listen the
	// nodes pull them from its work directory, wherever they run; without, they read them where
	// the controller wrote them (one machine).
	workDir := req.WorkDir
	src := protocol.Source{Kind: "gen"}
	switch {
	case c.bulkAddr != "":
		rel := filepath.Join(fmt.Sprintf("rebuild-e%d", t.Epoch), "partitions")
		workDir, src.Addr, src.Dir = filepath.Join(c.cfg.WorkDir, rel), c.bulkAddr, rel
	case workDir == "":
		workDir = fmt.Sprintf("%s.holders-e%d", gen.partsDir, t.Epoch)
		src.Dir = workDir
	default:
		src.Dir = workDir
	}
	plan, vectors, local, err := c.placeGeneration(ctx, gen, nodes, req, workDir)
	if err != nil {
		return nil, err
	}
	reply.Vectors, reply.Local = vectors, local
	draft := &epoch.Table{Epoch: t.Epoch + 1, Generation: gen.id, GenIndex: gen.indexDir, GenParts: gen.partsDir,
		Placement: plan, Nodes: map[placement.NodeID]epoch.NodeInfo{}, Entries: ids(nodes), Aggregators: ids(nodes)}
	for _, n := range nodes {
		draft.Nodes[n.ID] = n.Info
	}
	reply.Plan = protocol.PlanOf(plan, draft.Epoch)
	phase("place")

	// 5. STAGE: connections, partitions and graph, every node in parallel.
	var mu sync.Mutex
	if err := forEach(nodes, func(n *ManagedNode) error {
		if err := n.call(ctx, c.callTimeout(), protocol.StageAggregatorM, protocol.StageAggregatorReq{Table: draft}, nil); err != nil {
			return err
		}
		var st, gst protocol.StageReply
		s := src
		s.Partitions = plan.PartitionsOf(n.ID)
		if err := n.call(ctx, c.stageTimeout(), protocol.StagePartitionsM,
			protocol.StagePartitionsReq{Sources: []protocol.Source{s}, Nodes: draft.Nodes}, &st); err != nil {
			return err
		}
		if err := n.call(ctx, c.stageTimeout(), protocol.StageGraphM,
			protocol.StageGraphReq{Source: protocol.Source{Kind: "local", Dir: gen.indexDir}}, &gst); err != nil {
			return err
		}
		mu.Lock()
		reply.SegmentBytes += st.Bytes
		reply.GraphBytes += gst.Bytes
		mu.Unlock()
		return nil
	}); err != nil {
		return nil, err
	}
	phase("stage")

	// 6. WARM-UP: the sample, spread over the nodes, under the draft epoch.
	if len(sample) > 0 {
		before, beforeBytes := c.rawFetched(ctx, nodes)
		shares := make([][]protocol.SampledQuery, len(nodes))
		for i, q := range sample {
			shares[i%len(nodes)] = append(shares[i%len(nodes)], q)
		}
		var failed []string
		if err := forEach(nodes, func(n *ManagedNode) error {
			i := 0
			for i < len(nodes) && nodes[i] != n {
				i++
			}
			var wr protocol.WarmupReply
			if err := n.call(ctx, c.stageTimeout(), protocol.WarmupM, protocol.WarmupReq{Table: draft, Queries: shares[i],
				Concurrency: req.WarmupConcurrency}, &wr); err != nil {
				return err
			}
			if wr.Failed > 0 {
				mu.Lock()
				failed = append(failed, fmt.Sprintf("node %d: %d of %d queries failed (%s)", n.ID, wr.Failed, wr.Queries, wr.FirstError))
				mu.Unlock()
			}
			return nil
		}); err != nil {
			return nil, err
		}
		if len(failed) > 0 {
			sort.Strings(failed)
			return nil, fmt.Errorf("warm-up under generation %d: %v", gen.id, failed)
		}
		after, afterBytes := c.rawFetched(ctx, nodes)
		reply.WarmupQueries = len(sample)
		reply.WarmupFetched, reply.WarmupFetchedBytes = after-before, afterBytes-beforeBytes
		phase("warmup")
	}
	return draft, nil
}

// placeGeneration places gen's partitions on nodes and writes their segments into workDir with
// a holder for every posting whose vector its owner lacks: an old node that holds it. With
// "affinity" a partition goes where most of its distinct vectors are already (count-balanced);
// with "random" anywhere (a baseline). It returns the placement, and the distinct vectors of the
// partitions and how many of them their owners hold.
func (c *Controller) placeGeneration(ctx context.Context, gen *generation, nodes []*ManagedNode, req protocol.RebuildReq,
	workDir string) (placement.Table, int64, int64, error) {

	bitmaps := make([][]uint64, len(nodes))
	if err := forEach(nodes, func(n *ManagedNode) error {
		var r protocol.RawBitmapReply
		if err := n.call(ctx, c.stageTimeout(), protocol.RawBitmapM, struct{}{}, &r); err != nil {
			return err
		}
		if len(r.Bits)%8 != 0 {
			return fmt.Errorf("node %d: a raw-vector bitmap of %d bytes", n.ID, len(r.Bits))
		}
		w := make([]uint64, len(r.Bits)/8)
		for i := range w {
			w[i] = binary.LittleEndian.Uint64(r.Bits[8*i:])
		}
		for i, m := range nodes {
			if m == n {
				bitmaps[i] = w
			}
		}
		return nil
	}); err != nil {
		return placement.Table{}, 0, 0, err
	}
	has := func(i int, loc uint32) bool {
		w := int(loc >> 6)
		return w < len(bitmaps[i]) && bitmaps[i][w]>>(loc&63)&1 != 0
	}
	np := gen.manifest.NumPartitions
	segs := make([]*partitioning.Segment, np)
	weights := make([][]int64, np)
	totals := make([]int64, np)
	for p := 0; p < np; p++ {
		s, err := partitioning.ReadSegment(filepath.Join(gen.partsDir, partitioning.SegmentName(p)))
		if err != nil {
			return placement.Table{}, 0, 0, err
		}
		segs[p] = s
		locs := append([]uint32(nil), s.Locs...)
		sort.Slice(locs, func(i, j int) bool { return locs[i] < locs[j] })
		weights[p] = make([]int64, len(nodes))
		for k, loc := range locs {
			if k > 0 && loc == locs[k-1] {
				continue
			}
			totals[p]++
			for i := range nodes {
				if has(i, loc) {
					weights[p][i]++
				}
			}
		}
	}
	var plan placement.Table
	var err error
	if req.Placement == "random" {
		plan, err = placement.RandomBalanced(np, ids(nodes), req.Seed)
	} else {
		plan, err = placement.Affinity(weights, ids(nodes))
	}
	if err != nil {
		return placement.Table{}, 0, 0, err
	}
	if err := os.MkdirAll(workDir, 0o755); err != nil {
		return placement.Table{}, 0, 0, err
	}
	index := map[placement.NodeID]int{}
	for i, n := range nodes {
		index[n.ID] = i
	}
	var vectors, local int64
	var cand []int
	for p, s := range segs {
		o := index[plan.Owners[p]]
		vectors += totals[p]
		local += weights[p][o]
		s.Holders = make([]uint16, len(s.IDs))
		for j, loc := range s.Locs {
			if has(o, loc) {
				continue // the owner holds it
			}
			// Spread the fetches over the nodes that hold the vector.
			cand = cand[:0]
			for i := range nodes {
				if has(i, loc) {
					cand = append(cand, i)
				}
			}
			if len(cand) == 0 {
				return placement.Table{}, 0, 0, fmt.Errorf("partition %d: the vector at location %d (ID %d) is on no node",
					p, loc, s.IDs[j])
			}
			s.Holders[j] = uint16(nodes[cand[int(loc)%len(cand)]].ID)
		}
		if err := s.Write(filepath.Join(workDir, partitioning.SegmentName(p))); err != nil {
			return placement.Table{}, 0, 0, err
		}
	}
	return plan, vectors, local, nil
}

// querySample collects the client queries the nodes kept as entries: at most max of them (0:
// all; negative: none, no warm-up).
func (c *Controller) querySample(ctx context.Context, entries []*ManagedNode, max int) ([]protocol.SampledQuery, error) {
	if max < 0 {
		return nil, nil
	}
	samples := make([][]protocol.SampledQuery, len(entries))
	if err := forEach(entries, func(n *ManagedNode) error {
		var r protocol.QuerySampleReply
		if err := n.call(ctx, c.callTimeout(), protocol.QuerySampleM, struct{}{}, &r); err != nil {
			return err
		}
		for i, m := range entries {
			if m == n {
				samples[i] = r.Queries
			}
		}
		return nil
	}); err != nil {
		return nil, err
	}
	// The most recent ones first, taken from every entry in turn.
	var out []protocol.SampledQuery
	for k := 1; ; k++ {
		added := false
		for _, s := range samples {
			if k <= len(s) {
				out = append(out, s[len(s)-k])
				added = true
			}
		}
		if !added || (max > 0 && len(out) >= max) {
			break
		}
	}
	if max > 0 && len(out) > max {
		out = out[:max]
	}
	return out, nil
}

// waitDrained waits on nodes until no query routed with an epoch <= e is running there.
func (c *Controller) waitDrained(ctx context.Context, nodes []*ManagedNode, e uint64) error {
	return forEach(nodes, func(n *ManagedNode) error {
		return n.call(ctx, c.stageTimeout(), protocol.WaitDrainedM, protocol.WaitDrainedReq{Epoch: e}, nil)
	})
}

// adopt makes gen the controller's generation: the policy places its partitions (with its
// partition graph), and the history of who held what starts over with plan.
func (c *Controller) adopt(gen *generation, plan placement.Table) error {
	policy, err := newPolicy(c.cfg.Placement, gen.indexDir, gen.manifest)
	if err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.policy = policy
	c.manifest = gen.manifest
	c.history = placement.NewHistory(plan)
	return nil
}
