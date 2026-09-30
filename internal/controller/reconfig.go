package controller

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"rtier/internal/config"
	"rtier/internal/design"
	"rtier/internal/epoch"
	"rtier/internal/placement"
	"rtier/internal/protocol"
)

func hasRole(roles []string, r string) bool {
	for _, x := range roles {
		if x == r {
			return true
		}
	}
	return false
}

func ids(nodes []*ManagedNode) []placement.NodeID {
	out := make([]placement.NodeID, len(nodes))
	for i, n := range nodes {
		out[i] = n.ID
	}
	return out
}

func (c *Controller) stageTimeout() time.Duration { return c.cfg.StageTimeout.Duration }
func (c *Controller) callTimeout() time.Duration  { return c.cfg.CallTimeout.Duration }

// deployInitial brings up epoch 1 on the first nodes. Readiness is staged by role
// (aggregator -> data -> entry); partitions and the graph come from the build output.
func (c *Controller) deployInitial(ctx context.Context, nodes []*ManagedNode) error {
	all := ids(nodes)
	tbl, err := c.policy.Initial(c.manifest.NumPartitions, all)
	if err != nil {
		return err
	}
	entries := all
	if k := c.cfg.InitialEntries; k > 0 && k < len(all) {
		entries = all[:k]
	}
	draft := &epoch.Table{Epoch: 1, Placement: tbl, Nodes: map[placement.NodeID]epoch.NodeInfo{},
		Entries: entries, Aggregators: all}
	for _, n := range nodes {
		draft.Nodes[n.ID] = n.Info
	}
	entryNodes, _ := c.managed(entries)
	start := time.Now()

	if err := forEach(nodes, func(n *ManagedNode) error {
		return n.call(ctx, c.callTimeout(), protocol.StageAggregatorM, protocol.StageAggregatorReq{Table: draft}, nil)
	}); err != nil {
		return err
	}
	if err := forEach(nodes, func(n *ManagedNode) error {
		src := protocol.Source{Kind: "local", Dir: c.cfg.PartitionsDir, Partitions: tbl.PartitionsOf(n.ID)}
		return n.call(ctx, c.stageTimeout(), protocol.StagePartitionsM, protocol.StagePartitionsReq{Sources: []protocol.Source{src}}, nil)
	}); err != nil {
		return err
	}
	if err := forEach(entryNodes, func(n *ManagedNode) error {
		src := protocol.Source{Kind: "local", Dir: c.cfg.IndexDir}
		return n.call(ctx, c.stageTimeout(), protocol.StageGraphM, protocol.StageGraphReq{Source: src}, nil)
	}); err != nil {
		return err
	}
	if err := c.flip(ctx, 0, draft, nodes); err != nil {
		return err
	}
	c.mu.Lock()
	c.history = placement.NewHistory(tbl)
	c.mu.Unlock()
	log.Printf("controller: epoch 1 deployed on %d nodes (%d entries) in %.1fs; partitions per node %v",
		len(nodes), len(entries), time.Since(start).Seconds(), tbl.Counts())
	c.event("reconfig.deploy_s", time.Since(start).Seconds())
	return nil
}

// flip installs next in the store (compare-and-swap on prev) and then on every recipient.
// Between the two, nodes route with either epoch; both are valid because data is copied
// before the flip and reclaimed only after the grace period.
func (c *Controller) flip(ctx context.Context, prev uint64, next *epoch.Table, recipients []*ManagedNode) error {
	if err := c.store.CompareAndSwap(prev, next); err != nil {
		return err
	}
	return c.install(ctx, next, recipients)
}

// install pushes an epoch that is already in the store to the recipients.
func (c *Controller) install(ctx context.Context, next *epoch.Table, recipients []*ManagedNode) error {
	err := forEach(recipients, func(n *ManagedNode) error {
		return n.call(ctx, c.callTimeout(), protocol.InstallEpochM, protocol.InstallEpochReq{Table: next}, nil)
	})
	c.event("reconfig.epoch", float64(next.Epoch))
	return err
}

// Rescale changes the number of data nodes to target (Koala's Rescale API: one
// reconfiguration at a time). The reconfiguration runs to completion even if the caller
// disconnects: a protocol stopped half-way is worse than a late reply.
func (c *Controller) Rescale(ctx context.Context, target int) (*protocol.RescaleReply, error) {
	ctx = context.WithoutCancel(ctx)
	if c.Ready.IsBlocked() {
		return nil, fmt.Errorf("controller: initial deployment has not finished (%s)", c.LastError())
	}
	if !c.reconfigRunning.CompareAndSwap(false, true) {
		return nil, fmt.Errorf("controller: another reconfiguration is in progress, retry later")
	}
	defer c.reconfigRunning.Store(false)
	reply, err := c.rescale(ctx, target)
	c.setErr(err)
	return reply, err
}

func (c *Controller) rescale(ctx context.Context, target int) (*protocol.RescaleReply, error) {
	cur, err := c.store.Current()
	if err != nil {
		return nil, err
	}
	curIDs := cur.Placement.Nodes()
	sort.Slice(curIDs, func(i, j int) bool { return curIDs[i] < curIDs[j] })
	// Which nodes join or leave is part of U3; for now idle nodes join in ID order and the
	// highest IDs leave.
	var added, removed []*ManagedNode
	switch {
	case target > len(curIDs):
		idle := c.nodes.Idle(cur)
		if len(idle) < target-len(curIDs) {
			return nil, fmt.Errorf("controller: scale-out to %d needs %d idle nodes, %d registered",
				target, target-len(curIDs), len(idle))
		}
		added = idle[:target-len(curIDs)]
	case target < len(curIDs):
		if target < 1 {
			return nil, fmt.Errorf("controller: need at least one data node")
		}
		if removed, err = c.managed(curIDs[target:]); err != nil {
			return nil, err
		}
	default:
		// Same node count: nothing to do with the even policy (rebalancing is U3).
		return &protocol.RescaleReply{Protocol: c.cfg.Protocol, FromEpoch: cur.Epoch, ToEpoch: cur.Epoch}, nil
	}
	nextIDs := append([]placement.NodeID(nil), curIDs[:min(target, len(curIDs))]...)
	nextIDs = append(nextIDs, ids(added)...)
	// Who has owned what, for a policy that sends partitions back where their PQ codes are.
	// Only a reconfiguration writes the history, and they do not overlap.
	c.mu.Lock()
	var past placement.Past
	if c.history != nil {
		past = c.history
	}
	c.mu.Unlock()
	next, changes, err := c.policy.Repartition(cur.Placement, nextIDs, past)
	if err != nil {
		return nil, err
	}
	switch c.cfg.Protocol {
	case "lazy":
		return c.copyThenFlip(ctx, cur, next, changes, added, removed, false, rawOnDemand)
	case "lazy-stream":
		return c.copyThenFlip(ctx, cur, next, changes, added, removed, false, rawStreamAfterFlip)
	case "copy-then-flip":
		return c.copyThenFlip(ctx, cur, next, changes, added, removed, false, rawBeforeFlip)
	case "stop-and-copy":
		return c.copyThenFlip(ctx, cur, next, changes, added, removed, true, rawBeforeFlip)
	}
	return nil, fmt.Errorf("controller: unknown protocol %q", c.cfg.Protocol)
}

// rawMode says when the raw vectors of moved partitions reach their new owner (U9).
type rawMode int

const (
	// rawOnDemand (protocol "lazy", the default): only the ones queries need, fetched by RERANK
	// from the old owner when it first needs them; the rest stay with the old owner, which keeps
	// every vector it has held.
	rawOnDemand rawMode = iota
	// rawStreamAfterFlip ("lazy-stream", a baseline): on demand, and a stream brings the rest
	// after the flip.
	rawStreamAfterFlip
	// rawBeforeFlip ("copy-then-flip", "stop-and-copy"): all of them before the flip.
	rawBeforeFlip
)

// copyThenFlip is the reconfiguration protocol: "lazy" (the default) with rawOnDemand,
// "lazy-stream" with rawStreamAfterFlip, "copy-then-flip" with rawBeforeFlip, and with
// stopTheWorld too the stop-and-copy baseline (Koala's stop-and-restart):
//
//  1. PREPARE  aggregators of the next epoch pre-connect to its data nodes; destinations
//     pull the moved partitions' posting lists and the PQ codes they lack from the current
//     owners (separate bulk connections, paced by the senders' token buckets) -- what a node
//     needs to filter, so it can go online; with rawBeforeFlip, the raw vectors they lack too;
//     new entry nodes stage the graph in the background.
//  2. FLIP     compare-and-swap epoch e -> e+1 in the store, install e+1 everywhere.
//  3. GRACE    every node waits until no query pinned to epoch e is running.
//  4. RECLAIM  old owners evict the moved partitions (their PQ codes and raw vectors stay, as
//     a cache).
//  5. ENTRY    alongside 3 and 4, new nodes become entries as their graphs load: every
//     entry_flip_interval, one more flip adds those whose graph loaded since the previous
//     one (promoteEntries). A node whose graph fails to load stays a data node and aggregator.
//  6. RAW      from the flip on, a new owner's data node fetches the raw vectors a query needs
//     from the old owners on demand (every posting list knows the source of its partition),
//     and with rawOnDemand nothing else: what no query needs never moves, and the rescale
//     does not wait for any of it. That the old owners stay sources rests on invariants of
//     the design (U9): no node drops a raw vector it has held, the nodes that leave are the
//     ones that joined last, and a scale-in returns partitions to nodes that held them. With
//     rawStreamAfterFlip, alongside 3-5, the destinations also stream in the rest from the old
//     owners (raw_priority, background by default); the old owners -- removed nodes included
//     -- stay up as sources until it ends, and the rescale waits for it.
//
// Failures: before the flip, partitions already copied to destinations are dropped again and
// the old epoch stays in force. After the flip, the protocol stops and keeps every old copy
// (always safe, possibly wasteful); retrying or rolling back is open question U13.
func (c *Controller) copyThenFlip(ctx context.Context, cur *epoch.Table, next placement.Table,
	changes placement.Changes, added, removed []*ManagedNode, stopTheWorld bool, raw rawMode) (*protocol.RescaleReply, error) {

	reply := &protocol.RescaleReply{Protocol: c.cfg.Protocol, FromEpoch: cur.Epoch, ToEpoch: cur.Epoch,
		Added: ids(added), Removed: ids(removed), Moved: changes.Moved(), Phases: map[string]float64{}}
	triggered := time.Now()
	mark := triggered
	phase := func(name string) {
		s := time.Since(mark).Seconds()
		reply.Phases[name] = s
		c.event("reconfig."+name+"_s", s)
		mark = time.Now()
	}
	c.event("reconfig.start", float64(cur.Epoch))
	log.Printf("controller: reconfiguration %s: epoch %d, add %v, remove %v, move %d partitions",
		c.cfg.Protocol, cur.Epoch, reply.Added, reply.Removed, len(reply.Moved))

	// Draft of epoch e+1: the new placement; added nodes aggregate right away and become
	// entries only after their graph is loaded; removed nodes lose every role.
	draft := cur.Clone()
	draft.Epoch = cur.Epoch + 1
	draft.Placement = next
	var entryAdds []*ManagedNode
	for _, n := range added {
		draft.Nodes[n.ID] = n.Info
		if hasRole(c.cfg.NewNodeRoles, config.RoleAggregator) {
			draft.Aggregators = append(draft.Aggregators, n.ID)
		}
		if hasRole(c.cfg.NewNodeRoles, config.RoleEntry) {
			entryAdds = append(entryAdds, n)
		}
	}
	for _, n := range removed {
		delete(draft.Nodes, n.ID)
		draft.Entries = without(draft.Entries, n.ID)
		draft.Aggregators = without(draft.Aggregators, n.ID)
	}
	if len(draft.Entries) == 0 {
		return reply, fmt.Errorf("controller: the next epoch would have no entry node")
	}
	// Who serves the graph to whom is decided now, at the trigger, not when the copy starts.
	graphSrcs, graphFrom := c.graphSources(cur, entryAdds)
	if len(graphFrom) > 0 {
		reply.GraphSources = graphFrom
		log.Printf("controller: graph sources (new entry <- existing entry): %v", graphFrom)
	}
	curNodes, err := c.managed(sortedKeys(cur.Nodes))
	if err != nil {
		return reply, err
	}
	allNodes, err := c.managed(union(sortedKeys(cur.Nodes), sortedKeys(draft.Nodes)))
	if err != nil {
		return reply, err
	}
	fetchedBefore, fetchedBytesBefore := c.rawFetched(ctx, allNodes)
	plan := placement.MakePlan(changes)
	dests := make([]placement.NodeID, 0, len(plan))
	for d := range plan {
		dests = append(dests, d)
	}
	sort.Slice(dests, func(i, j int) bool { return dests[i] < dests[j] })
	destNodes, err := c.managed(dests)
	if err != nil {
		return reply, err
	}
	// Each destination pulls from the moved partitions' current owners: their bulk port for
	// files, PQ codes and raw vectors, their data node for on-demand raw-vector fetches.
	sources := map[placement.NodeID][]protocol.Source{}
	for _, d := range dests {
		for _, pull := range plan[d] {
			from, ok := c.nodes.Get(pull.From)
			if !ok {
				return reply, fmt.Errorf("unknown source node %d", pull.From)
			}
			sources[d] = append(sources[d], protocol.Source{Kind: "peer", Addr: from.Info.BulkAddr,
				NodeAddr: from.Info.NodeAddr, Partitions: pull.Partitions})
		}
	}
	bg := context.Background() // cleanup runs even if the protocol failed on a timeout

	// Before the flip, a failure drops the partitions already copied to destinations.
	flipped := false
	defer func() {
		if flipped {
			return
		}
		if err := forEach(destNodes, func(n *ManagedNode) error {
			var parts []int
			for _, pull := range plan[n.ID] {
				parts = append(parts, pull.Partitions...)
			}
			// The agent stops a staging still running there, drops what it loaded and frees
			// PQ codes installed for it.
			return n.call(bg, c.callTimeout(), protocol.EvictM,
				protocol.EvictReq{Partitions: parts, ReleaseStagedPQ: true}, nil)
		}); err != nil {
			log.Printf("controller: cleanup of staged partitions failed: %v", err)
		}
	}()

	// 1. PREPARE -- aggregator readiness: pre-connect to every data node of e+1.
	aggNodes, _ := c.managed(draft.Aggregators)
	if err := forEach(aggNodes, func(n *ManagedNode) error {
		return n.call(ctx, c.callTimeout(), protocol.StageAggregatorM, protocol.StageAggregatorReq{Table: draft}, nil)
	}); err != nil {
		return reply, err
	}
	phase("stage_aggregator")

	// Entry readiness runs in the background: the graph is large, and new nodes serve as data
	// nodes and aggregators without it. Every return waits for it, so it never overlaps the
	// next reconfiguration. Graph sources were assigned when the reconfiguration started
	// (round-robin over the entries), and the pulls run at background priority by default: on
	// a source that is also sending segments or PQ codes, the graph only gets the bandwidth
	// those leave unused, so it never delays the flip (U16). Each pull reports on its own, so a
	// node whose graph loads early does not wait for the others (promoteEntries).
	graphs := make(chan graphDone, len(entryAdds))
	var graphWG sync.WaitGroup
	var graphBytes atomic.Int64
	defer graphWG.Wait()
	for _, n := range entryAdds {
		graphWG.Add(1)
		go func(n *ManagedNode) {
			defer graphWG.Done()
			var st protocol.StageReply
			err := n.call(ctx, c.stageTimeout(), protocol.StageGraphM,
				protocol.StageGraphReq{Source: graphSrcs[n.ID], Priority: c.cfg.GraphPriority}, &st)
			graphBytes.Add(st.Bytes)
			graphs <- graphDone{n, err}
		}(n)
	}

	// Stop-and-copy baseline: pause admission everywhere and wait for in-flight queries.
	// Whatever happens next, every paused node is resumed.
	var paused []*ManagedNode
	resumed := true
	resume := func() error {
		resumed = true
		return forEach(paused, func(n *ManagedNode) error {
			return n.call(bg, c.callTimeout(), protocol.SetAdmissionM, protocol.SetAdmissionReq{Paused: false}, nil)
		})
	}
	defer func() {
		if !resumed {
			if err := resume(); err != nil {
				log.Printf("controller: resuming admission failed: %v", err)
			}
		}
	}()
	if stopTheWorld {
		if paused, err = c.managed(cur.Entries); err != nil {
			return reply, err
		}
		resumed = false
		if err := forEach(paused, func(n *ManagedNode) error {
			return n.call(ctx, c.callTimeout(), protocol.SetAdmissionM, protocol.SetAdmissionReq{Paused: true}, nil)
		}); err != nil {
			return reply, err
		}
		if err := forEach(curNodes, func(n *ManagedNode) error {
			return n.call(ctx, c.stageTimeout(), protocol.WaitIdleM, struct{}{}, nil)
		}); err != nil {
			return reply, err
		}
		phase("pause")
	}

	// 1. PREPARE -- data readiness: destinations pull segments and the PQ codes they lack from
	// current owners.
	var moved, pqCodes, pqBytes atomic.Int64
	if err := forEach(destNodes, func(n *ManagedNode) error {
		var st protocol.StageReply
		err := n.call(ctx, c.stageTimeout(), protocol.StagePartitionsM, protocol.StagePartitionsReq{Sources: sources[n.ID]}, &st)
		moved.Add(st.Bytes)
		pqCodes.Add(st.PQCodes)
		pqBytes.Add(st.PQBytes)
		return err
	}); err != nil {
		return reply, err
	}
	reply.Bytes, reply.PQCodes, reply.PQBytes = moved.Load(), pqCodes.Load(), pqBytes.Load()
	phase("stage_data")
	if raw == rawBeforeFlip { // eager: the raw vectors too, before the flip
		st, err := c.stageRaw(ctx, destNodes, sources, "data")
		reply.RawVectors, reply.RawBytes = st.RawVectors, st.RawBytes
		if err != nil {
			return reply, err
		}
		phase("stage_raw")
	}

	// 2. FLIP. Once the compare-and-swap succeeds the new epoch is authoritative; from here on
	// a failure keeps all old copies (U13).
	if err := c.store.CompareAndSwap(cur.Epoch, draft); err != nil {
		return reply, err
	}
	flipped = true
	afterFlip := func(err error) error {
		return fmt.Errorf("controller: reconfiguration stopped after flipping to epoch %d; old copies are "+
			"kept (%w): %v", draft.Epoch, design.Undecided("U13", "failure handling during reconfiguration"), err)
	}
	if err := c.install(ctx, draft, allNodes); err != nil {
		return reply, afterFlip(err)
	}
	reply.ToEpoch = draft.Epoch
	phase("flip")
	flipDone := time.Now()
	if stopTheWorld {
		if err := resume(); err != nil {
			return reply, afterFlip(err)
		}
	}

	// 6. RAW: the new owners serve their partitions already, fetching the raw vectors a query
	// needs on demand. With rawStreamAfterFlip a stream brings the rest, alongside everything
	// that follows; every return waits for it, so the old owners stay sources until it ends.
	rawDone := make(chan struct{})
	var rawErr error
	if raw == rawStreamAfterFlip {
		go func() {
			defer close(rawDone)
			var st protocol.StageReply
			st, rawErr = c.stageRaw(ctx, destNodes, sources, c.cfg.RawPriority)
			reply.RawVectors, reply.RawBytes = st.RawVectors, st.RawBytes
			reply.RawSeconds = time.Since(flipDone).Seconds()
			c.event("reconfig.raw_s", reply.RawSeconds)
		}()
	} else {
		close(rawDone)
	}
	defer func() { <-rawDone }()

	// 5. ENTRY, alongside GRACE and RECLAIM: the new nodes become entries as their graphs load.
	// Those flips change only the entries, so routing stays that of e+1: they need no grace
	// period of their own, and they do not hold up the one of e (WaitDrained(e) needs a newer
	// epoch installed and no query of e left, whichever newer epoch it is).
	var (
		entryLast  = draft
		entryFlips []protocol.EntryFlip
		entryFail  map[placement.NodeID]error
		entryErr   error
	)
	entryDone := make(chan struct{})
	go func() {
		defer close(entryDone)
		entryLast, entryFlips, entryFail, entryErr = c.promoteEntries(ctx, draft, graphs, len(entryAdds), removed, triggered)
	}()
	entries := func() {
		<-entryDone
		reply.ToEpoch, reply.EntryFlips = entryLast.Epoch, entryFlips
		reply.GraphBytes = graphBytes.Load() // every pull has reported: promoteEntries read them all
		for id, err := range entryFail {
			if reply.EntryFailed == nil {
				reply.EntryFailed = map[placement.NodeID]string{}
			}
			reply.EntryFailed[id] = err.Error()
		}
	}
	defer entries() // on every return: no flip may come after the reply

	// 3. GRACE: no query routed with epoch e is still running anywhere.
	if err := forEach(curNodes, func(n *ManagedNode) error {
		return n.call(ctx, c.stageTimeout(), protocol.WaitDrainedM, protocol.WaitDrainedReq{Epoch: cur.Epoch}, nil)
	}); err != nil {
		return reply, afterFlip(err)
	}
	phase("drain")

	// 4. RECLAIM.
	var srcIDs []placement.NodeID
	for src := range changes {
		srcIDs = append(srcIDs, src)
	}
	srcNodes, _ := c.managed(srcIDs)
	if err := forEach(srcNodes, func(n *ManagedNode) error {
		var parts []int
		for _, l := range changes[n.ID] {
			parts = append(parts, l...)
		}
		sort.Ints(parts)
		return n.call(ctx, c.callTimeout(), protocol.EvictM, protocol.EvictReq{Partitions: parts}, nil)
	}); err != nil {
		return reply, afterFlip(err)
	}
	c.mu.Lock()
	for src, m := range changes {
		for _, l := range m {
			for _, p := range l {
				c.history.Forget(p, src)
			}
		}
	}
	c.history.Record(next)
	c.mu.Unlock()
	phase("reclaim")

	entries()
	if entryErr != nil {
		return reply, afterFlip(entryErr)
	}
	<-rawDone
	if rawErr != nil {
		return reply, afterFlip(rawErr)
	}
	// What queries fetched on demand while the reconfiguration ran. Without a stream they go on
	// fetching afterwards, as they meet vectors that have not moved yet: the nodes' raw.fetched
	// counts those.
	fetched, fetchedBytes := c.rawFetched(ctx, allNodes)
	reply.RawFetched, reply.RawFetchedBytes = fetched-fetchedBefore, fetchedBytes-fetchedBytesBefore
	log.Printf("controller: reconfiguration done: epoch %d -> %d, %.1f MB moved (%d PQ codes), %d raw vectors "+
		"streamed (%.1f MB, %.2f s after the flip), phases %v, entry flips %v",
		reply.FromEpoch, reply.ToEpoch, float64(reply.Bytes)/1e6, reply.PQCodes, reply.RawVectors,
		float64(reply.RawBytes)/1e6, reply.RawSeconds, reply.Phases, reply.EntryFlips)
	if len(entryFail) > 0 {
		var errs []error
		for _, id := range sortedKeys(entryFail) {
			errs = append(errs, fmt.Errorf("node %d: %w", id, entryFail[id]))
		}
		return reply, fmt.Errorf("controller: nodes %v are data nodes and aggregators of epoch %d but not entries, "+
			"their graph did not load: %w", sortedKeys(entryFail), reply.ToEpoch, errors.Join(errs...))
	}
	return reply, nil
}

// rawFetched adds up the raw vectors the nodes' data nodes fetched on demand so far (and their
// bytes). A node that does not answer counts as zero: the numbers are for the reply only.
func (c *Controller) rawFetched(ctx context.Context, nodes []*ManagedNode) (int64, int64) {
	var n, b atomic.Int64
	forEach(nodes, func(m *ManagedNode) error {
		var st protocol.NodeStatus
		if err := m.call(ctx, c.callTimeout(), protocol.NodeStatusM, struct{}{}, &st); err == nil {
			n.Add(int64(st.RawFetched))
			b.Add(int64(st.RawFetchedBytes))
		}
		return nil
	})
	return n.Load(), b.Load()
}

// stageRaw has every destination stream in the raw vectors its new partitions lack from their
// old owners, at the given class on the owners' buckets, and adds up what moved.
func (c *Controller) stageRaw(ctx context.Context, dests []*ManagedNode, sources map[placement.NodeID][]protocol.Source,
	priority string) (protocol.StageReply, error) {
	var vectors, bytes atomic.Int64
	err := forEach(dests, func(n *ManagedNode) error {
		var st protocol.StageReply
		err := n.call(ctx, c.stageTimeout(), protocol.StageRawM, protocol.StageRawReq{Sources: sources[n.ID], Priority: priority}, &st)
		vectors.Add(st.RawVectors)
		bytes.Add(st.RawBytes)
		return err
	})
	return protocol.StageReply{RawVectors: vectors.Load(), RawBytes: bytes.Load()}, err
}

// graphDone reports how one new node's graph staging ended.
type graphDone struct {
	node *ManagedNode
	err  error
}

// promoteEntries makes the n new nodes entries as their graphs load, starting from t, the
// epoch the data flip installed. At every tick of entry_flip_interval, one flip adds all the
// nodes whose graph loaded since the previous flip; once every pull has ended, the last ones
// are added at once. So a node does not wait for the slowest pull, and one whose pull failed
// stays a data node and aggregator without holding up the others. It returns the last table
// it installed, the flips, and the nodes whose graph did not load.
func (c *Controller) promoteEntries(ctx context.Context, t *epoch.Table, done <-chan graphDone, n int,
	removed []*ManagedNode, triggered time.Time) (*epoch.Table, []protocol.EntryFlip, map[placement.NodeID]error, error) {

	var (
		flips  []protocol.EntryFlip
		failed = map[placement.NodeID]error{}
		ready  []*ManagedNode
	)
	promote := func() error {
		if len(ready) == 0 {
			return nil
		}
		// A node re-added after a stop-and-copy scale-in must not start paused.
		if err := forEach(ready, func(n *ManagedNode) error {
			return n.call(ctx, c.callTimeout(), protocol.SetAdmissionM, protocol.SetAdmissionReq{Paused: false}, nil)
		}); err != nil {
			return err
		}
		next := t.Clone()
		next.Epoch = t.Epoch + 1
		next.Entries = append(next.Entries, ids(ready)...)
		recips, err := c.managed(sortedKeys(next.Nodes))
		if err != nil {
			return err
		}
		if err := c.flip(ctx, t.Epoch, next, append(recips, removed...)); err != nil {
			return err
		}
		s := time.Since(triggered).Seconds()
		flips = append(flips, protocol.EntryFlip{Epoch: next.Epoch, Nodes: ids(ready), Seconds: s})
		c.event("reconfig.entry_flip_s", s)
		log.Printf("controller: epoch %d: nodes %v are entries (%.2f s after the trigger)", next.Epoch, ids(ready), s)
		t, ready = next, nil
		return nil
	}
	tick := time.NewTicker(c.cfg.EntryFlipInterval.Duration)
	defer tick.Stop()
	for pending := n; pending > 0; {
		select {
		case d := <-done:
			pending--
			if d.err != nil {
				failed[d.node.ID] = d.err
				log.Printf("controller: node %d stays a data node and aggregator: its graph did not load: %v", d.node.ID, d.err)
			} else {
				ready = append(ready, d.node)
			}
		case <-tick.C:
			if err := promote(); err != nil {
				return t, flips, failed, err
			}
		}
	}
	return t, flips, failed, promote()
}

// graphSources assigns every new entry the existing entry it copies the graph from:
// round-robin over the live entries of cur (by node ID), with a cursor that carries over to
// the next reconfiguration, so k new nodes spread over the entries instead of all pulling from
// one, and successive one-node scale-outs do not all land on the same entry. With no live
// entry the nodes copy from the build output (single machine). The second result lists the
// peer assignments, for the reply.
func (c *Controller) graphSources(cur *epoch.Table, adds []*ManagedNode) (map[placement.NodeID]protocol.Source, map[placement.NodeID]placement.NodeID) {
	entries := append([]placement.NodeID(nil), cur.Entries...)
	sort.Slice(entries, func(i, j int) bool { return entries[i] < entries[j] })
	var live []*ManagedNode
	for _, id := range entries {
		if n, ok := c.nodes.Get(id); ok && n.Alive() {
			live = append(live, n)
		}
	}
	c.mu.Lock()
	start := c.graphNext
	c.graphNext += len(adds)
	c.mu.Unlock()
	srcs := map[placement.NodeID]protocol.Source{}
	from := map[placement.NodeID]placement.NodeID{}
	liveIDs := make([]placement.NodeID, len(live))
	for i, n := range live {
		liveIDs[i] = n.ID
	}
	addIDs := make([]placement.NodeID, len(adds))
	for i, n := range adds {
		addIDs[i] = n.ID
	}
	for dst, i := range roundRobin(addIDs, len(liveIDs), start) {
		if i < 0 {
			srcs[dst] = protocol.Source{Kind: "local", Dir: c.cfg.IndexDir}
			continue
		}
		srcs[dst] = protocol.Source{Kind: "peer", Addr: live[i].Info.BulkAddr, File: "graph/graph.hnsw"}
		from[dst] = liveIDs[i]
	}
	return srcs, from
}

// roundRobin maps each of dsts, in order, to one of n sources: source (start+k) mod n for the
// k-th destination; -1 when there is no source.
func roundRobin(dsts []placement.NodeID, n, start int) map[placement.NodeID]int {
	out := make(map[placement.NodeID]int, len(dsts))
	for k, d := range dsts {
		if n == 0 {
			out[d] = -1
			continue
		}
		out[d] = (start + k) % n
	}
	return out
}

func without(s []placement.NodeID, id placement.NodeID) []placement.NodeID {
	out := s[:0:0]
	for _, x := range s {
		if x != id {
			out = append(out, x)
		}
	}
	return out
}

func sortedKeys[V any](m map[placement.NodeID]V) []placement.NodeID {
	out := make([]placement.NodeID, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

func union(a, b []placement.NodeID) []placement.NodeID {
	seen := map[placement.NodeID]bool{}
	var out []placement.NodeID
	for _, s := range [][]placement.NodeID{a, b} {
		for _, x := range s {
			if !seen[x] {
				seen[x] = true
				out = append(out, x)
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}
