package controller

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log"
	"sort"
	"time"

	"rtier/internal/design"
	"rtier/internal/placement"
	"rtier/internal/protocol"
)

// Update batches (U14). The update controller (internal/update) decides a batch and has the
// owners stage it; the controller coordinates, as for a reconfiguration, because it holds the
// epoch store and serializes everything that changes the epoch:
//
//  1. BeginUpdate takes the reconfiguration slot under a lease and returns the current epoch.
//     The update controller routes the inserts with its own copy of the navigation graph, by
//     that epoch's placement, sends their PQ
//     codes and raw vectors to their owners, and has every owner stage the next version of the
//     lists the batch changes next to the current one (STAGE_VERSIONS). Queries go on reading
//     the current versions.
//  2. CommitUpdate checks that every owner holds the versions the batch names, then flips: CAS,
//     install everywhere -- only the partitions' versions change, so all entries switch at
//     once -- grace (queries of the old epoch read the old versions until they end), and reclaim:
//     every node frees the deleted vectors' PQ codes (former owners' cached copies included) and
//     the owners drop the versions the batch replaced. A commit that fails before the flip drops
//     the staged versions itself, since the lease ends with the commit.
//  3. AbortUpdate gives the lease up without a flip. Dropping what was staged is the update
//     controller's job: it knows what it staged.
//
// A major epoch (splits and merges of posting lists) is a batch too. Besides the lists' next
// versions, it creates lists and patches the navigation graph: the update controller stages
// the patch next to the graph on every entry, CommitUpdate checks it there, the flip names the
// new graph version and lists, and after the grace period every entry merges the patch into
// its graph in place, before the lease ends -- so a later reconfiguration or major epoch starts
// from the merged graph.
//
// A lease that sees neither a commit nor an abort for update_lease expires, so that an update
// controller that died does not block reconfigurations for good; a later commit is refused.
// Batches run only with allow_updates.

// updateLease is the lease of the batch in progress.
type updateLease struct {
	token string
	epoch uint64
	timer *time.Timer
}

// BeginUpdate starts an update batch: it takes the reconfiguration slot under a lease and
// returns the epoch the batch is for.
func (c *Controller) BeginUpdate(ctx context.Context, req protocol.BeginUpdateReq) (*protocol.UpdateLease, error) {
	if !c.cfg.AllowUpdates {
		return nil, fmt.Errorf("controller: update batches are off (allow_updates)")
	}
	if c.Ready.IsBlocked() {
		return nil, fmt.Errorf("controller: initial deployment has not finished (%s)", c.LastError())
	}
	if !c.reconfigRunning.CompareAndSwap(false, true) {
		return nil, fmt.Errorf("controller: a reconfiguration or another update batch is in progress, retry later")
	}
	cur, err := c.store.Current()
	if err != nil {
		c.reconfigRunning.Store(false)
		return nil, err
	}
	// PQ codes still protected for a staging that ended without releasing them -- a rollback that
	// failed, or an update controller that died between PQ_PUT and STAGE_VERSIONS -- would hold
	// slots the batch's room check has to count as taken, for good. While the slot is held no
	// staging runs, so every protected code is such a leftover: the agents stop what may still run
	// of one and release them (the rollback of a staging).
	if err := c.releaseStaged(ctx, cur.Placement.Nodes()); err != nil {
		c.reconfigRunning.Store(false)
		return nil, fmt.Errorf("controller: releasing PQ codes left staged: %w", err)
	}
	d := c.cfg.UpdateLease.Duration
	if req.LeaseSeconds > 0 {
		d = time.Duration(req.LeaseSeconds * float64(time.Second))
	}
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		c.reconfigRunning.Store(false)
		return nil, err
	}
	l := &updateLease{token: hex.EncodeToString(b[:]), epoch: cur.Epoch}
	c.mu.Lock()
	c.lease = l
	l.timer = time.AfterFunc(d, func() { c.expireLease(l) })
	c.mu.Unlock()
	log.Printf("controller: update batch begins at epoch %d (lease %v)", cur.Epoch, d)
	return &protocol.UpdateLease{Token: l.token, Table: cur, Nodes: c.nodes.Summaries(cur), Expires: time.Now().Add(d)}, nil
}

// releaseStaged has the agents of nodes stop any staging still running and release the PQ
// codes left protected for one.
func (c *Controller) releaseStaged(ctx context.Context, ids []placement.NodeID) error {
	nodes, err := c.managed(ids)
	if err != nil {
		return err
	}
	return forEach(nodes, func(n *ManagedNode) error {
		return n.call(ctx, c.stageTimeout(), protocol.EvictM, protocol.EvictReq{ReleaseStagedPQ: true}, nil)
	})
}

// expireLease ends lease l if it still holds the slot.
func (c *Controller) expireLease(l *updateLease) {
	c.mu.Lock()
	if c.lease != l {
		c.mu.Unlock()
		return
	}
	c.lease = nil
	c.mu.Unlock()
	log.Printf("controller: the update lease of epoch %d expired without a commit or an abort", l.epoch)
	c.reconfigRunning.Store(false)
}

// takeLease ends the lease with this token and reports whether it still held the slot, which
// then stays taken for the caller to release.
func (c *Controller) takeLease(token string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	l := c.lease
	if l == nil || l.token != token || !l.timer.Stop() {
		return false // no such lease, or it is expiring: expireLease releases the slot
	}
	c.lease = nil
	return true
}

func errNoLease(token string) error {
	return fmt.Errorf("controller: no update lease %q: it expired, or its batch was committed or aborted", token)
}

// AbortUpdate ends a batch without a flip.
func (c *Controller) AbortUpdate(ctx context.Context, req protocol.AbortUpdateReq) error {
	if !c.takeLease(req.Token) {
		return errNoLease(req.Token)
	}
	c.reconfigRunning.Store(false)
	log.Printf("controller: update batch aborted")
	return nil
}

// CommitUpdate flips to the versions a batch staged and reclaims the ones it replaced. Like a
// reconfiguration it runs to completion even if the caller disconnects.
func (c *Controller) CommitUpdate(ctx context.Context, req protocol.CommitUpdateReq) (*protocol.UpdateReply, error) {
	ctx = context.WithoutCancel(ctx)
	if !c.takeLease(req.Token) {
		return nil, errNoLease(req.Token)
	}
	defer c.reconfigRunning.Store(false)
	reply, err := c.commitUpdate(ctx, req)
	c.setErr(err)
	return reply, err
}

func (c *Controller) commitUpdate(ctx context.Context, req protocol.CommitUpdateReq) (_ *protocol.UpdateReply, err error) {
	cur, err := c.store.Current()
	if err != nil {
		return nil, err
	}
	reply := &protocol.UpdateReply{FromEpoch: cur.Epoch, ToEpoch: cur.Epoch, Phases: map[string]float64{}}
	if req.FromEpoch != cur.Epoch {
		return reply, fmt.Errorf("controller: the batch was staged for epoch %d, the current epoch is %d", req.FromEpoch, cur.Epoch)
	}
	mark := time.Now()
	phase := func(name string) {
		s := time.Since(mark).Seconds()
		reply.Phases[name] = s
		c.event("update."+name+"_s", s)
		mark = time.Now()
	}
	np := cur.Placement.NumPartitions()
	next := cur.Clone()
	next.Epoch = cur.Epoch + 1
	if len(next.Versions) == 0 {
		next.Versions = make([]uint32, np)
	}
	// A major epoch: the graph's next version, and the lists it creates (never an existing one).
	if req.Graph != 0 && req.Graph != cur.Graph+1 {
		return reply, fmt.Errorf("controller: the batch patches the graph to version %d, the graph is at version %d", req.Graph, cur.Graph)
	}
	if req.Graph != 0 {
		next.Graph = req.Graph
	}
	for l, p := range req.Lists {
		if p < 0 || p >= np {
			return reply, fmt.Errorf("controller: the batch creates list %d in partition %d of %d", l, p, np)
		}
		if _, ok := cur.Lists[l]; ok {
			return reply, fmt.Errorf("controller: the batch creates list %d, which exists", l)
		}
		if next.Lists == nil {
			next.Lists = map[uint32]int{}
		}
		next.Lists[l] = p
	}
	reply.Graph = next.Graph
	// By owner: the versions to check before the flip, and the ones to drop after the grace
	// period. Versions only grow, so a version number never names two different list sets.
	staged := map[placement.NodeID][]protocol.PartitionVersion{}
	replaced := map[placement.NodeID][]protocol.PartitionVersion{}
	for _, pv := range req.Versions {
		p := pv.Partition
		if p < 0 || p >= np {
			return reply, fmt.Errorf("controller: the batch names partition %d of %d", p, np)
		}
		if next.Versions[p] != cur.Version(p) {
			return reply, fmt.Errorf("controller: the batch names partition %d twice", p)
		}
		if base := cur.Version(p); pv.Version <= base {
			return reply, fmt.Errorf("controller: partition %d is at version %d, the batch's version %d is not newer", p, base, pv.Version)
		}
		o := cur.Owner(p)
		staged[o] = append(staged[o], pv)
		replaced[o] = append(replaced[o], protocol.PartitionVersion{Partition: p, Version: cur.Version(p)})
		next.Versions[p] = pv.Version
		reply.Changed = append(reply.Changed, p)
	}
	sort.Ints(reply.Changed)
	if len(reply.Changed) == 0 && req.Graph == 0 {
		return reply, nil // the batch changed no list: nothing to flip
	}
	entries, err := c.managed(cur.Entries)
	if err != nil {
		return reply, err
	}
	if req.Graph == 0 {
		entries = nil
	}
	// Before the flip, a failure drops the staged versions and patches: no epoch names them, and
	// the batch's lease has ended, so the update controller may no longer touch the nodes.
	flipped := false
	defer func() {
		if !flipped && err != nil {
			c.dropVersions(staged)
			c.graphPatch(entries, protocol.GraphPatchReq{Version: req.Graph, Drop: true})
		}
	}()
	c.event("update.start", float64(cur.Epoch))
	log.Printf("controller: update batch: epoch %d, %d partitions at new versions", cur.Epoch, len(reply.Changed))

	// 1. CHECK: every owner holds the versions it staged -- the update controller says so, and a
	// flip without them would route queries to lists that are not there.
	owners, err := c.managed(sortedKeys(staged))
	if err != nil {
		return reply, err
	}
	if err := forEach(owners, func(n *ManagedNode) error {
		var st protocol.NodeStatus
		if err := n.call(ctx, c.callTimeout(), protocol.NodeStatusM, struct{}{}, &st); err != nil {
			return err
		}
		have := map[protocol.PartitionVersion]bool{}
		for _, pv := range st.Versions {
			have[pv] = true
		}
		for _, pv := range staged[n.ID] {
			if !have[pv] {
				return fmt.Errorf("node %d does not hold partition %d at version %d", n.ID, pv.Partition, pv.Version)
			}
		}
		return nil
	}); err != nil {
		return reply, fmt.Errorf("controller: update batch not staged: %w", err)
	}
	if err := forEach(entries, func(n *ManagedNode) error {
		var st protocol.NodeStatus
		if err := n.call(ctx, c.callTimeout(), protocol.NodeStatusM, struct{}{}, &st); err != nil {
			return err
		}
		if st.Graph != cur.Graph || st.GraphPatch != req.Graph {
			return fmt.Errorf("node %d has its graph at version %d and a patch to %d staged, the batch patches %d to %d",
				n.ID, st.Graph, st.GraphPatch, cur.Graph, req.Graph)
		}
		return nil
	}); err != nil {
		return reply, fmt.Errorf("controller: update batch not staged: %w", err)
	}
	phase("check")

	// 2. FLIP. From the compare-and-swap on, the new versions are authoritative; a failure keeps
	// the replaced ones (U13).
	recipients, err := c.managed(sortedKeys(cur.Nodes))
	if err != nil {
		return reply, err
	}
	if err := c.store.CompareAndSwap(cur.Epoch, next); err != nil {
		return reply, err
	}
	flipped = true
	afterFlip := func(err error) error {
		return fmt.Errorf("controller: update batch stopped after flipping to epoch %d; the replaced versions are "+
			"kept (%w): %v", next.Epoch, design.Undecided("U13", "failure handling during reconfiguration"), err)
	}
	if err := c.install(ctx, next, recipients); err != nil {
		return reply, afterFlip(err)
	}
	reply.ToEpoch = next.Epoch
	phase("flip")

	// 3. GRACE: no query of the old epoch -- the only ones that read the replaced versions -- is
	// still running.
	if err := forEach(recipients, func(n *ManagedNode) error {
		return n.call(ctx, c.stageTimeout(), protocol.WaitDrainedM, protocol.WaitDrainedReq{Epoch: cur.Epoch}, nil)
	}); err != nil {
		return reply, afterFlip(err)
	}
	phase("drain")

	// 4. RECLAIM: every node frees the PQ codes of the vectors the batch deleted, and the owners
	// drop the replaced versions -- in that order on each node, so that a deleted vector's code
	// goes with the last version that names it and never becomes a cached one. A deleted vector
	// is no cache: its ID is never named again. Former owners hold cached copies of deleted
	// vectors too; for them the step only frees memory (no list can name those copies), so a
	// node that does not answer is logged, not an error.
	if err := forEach(owners, func(n *ManagedNode) error {
		return n.call(ctx, c.stageTimeout(), protocol.DropVersionsM,
			protocol.DropVersionsReq{Versions: replaced[n.ID], Deleted: req.Deleted}, nil)
	}); err != nil {
		return reply, afterFlip(err)
	}
	if len(req.Deleted) > 0 {
		var others []*ManagedNode
		for _, n := range c.nodes.All() {
			if _, owner := staged[n.ID]; !owner && n.Alive() {
				others = append(others, n)
			}
		}
		if err := forEach(others, func(n *ManagedNode) error {
			return n.call(ctx, c.stageTimeout(), protocol.DropVersionsM, protocol.DropVersionsReq{Deleted: req.Deleted}, nil)
		}); err != nil {
			log.Printf("controller: freeing the PQ codes of deleted vectors on former owners: %v", err)
		}
	}
	// No query searches the graph without the patch any more: every entry merges it in place.
	if err := forEach(entries, func(n *ManagedNode) error {
		return n.call(ctx, c.stageTimeout(), protocol.GraphPatchM, protocol.GraphPatchReq{Version: req.Graph}, nil)
	}); err != nil {
		return reply, afterFlip(err)
	}
	phase("reclaim")
	log.Printf("controller: update batch done: epoch %d -> %d, partitions %v at new versions, phases %v",
		reply.FromEpoch, reply.ToEpoch, reply.Changed, reply.Phases)
	return reply, nil
}

// graphPatch sends req to nodes (best effort: a patch left staged only takes memory, and the
// next major epoch's patch replaces it).
func (c *Controller) graphPatch(nodes []*ManagedNode, req protocol.GraphPatchReq) {
	if err := forEach(nodes, func(n *ManagedNode) error {
		return n.call(context.Background(), c.stageTimeout(), protocol.GraphPatchM, req, nil)
	}); err != nil {
		log.Printf("controller: graph patch %d on the entries: %v", req.Version, err)
	}
}

// dropVersions has the owners drop versions no epoch names (best effort: what stays only takes
// memory, and a later batch that stages the same version replaces it).
func (c *Controller) dropVersions(byOwner map[placement.NodeID][]protocol.PartitionVersion) {
	owners, err := c.managed(sortedKeys(byOwner))
	if err == nil {
		err = forEach(owners, func(n *ManagedNode) error {
			return n.call(context.Background(), c.stageTimeout(), protocol.DropVersionsM, protocol.DropVersionsReq{Versions: byOwner[n.ID]}, nil)
		})
	}
	if err != nil {
		log.Printf("controller: dropping the versions of a batch that did not flip: %v", err)
	}
}
