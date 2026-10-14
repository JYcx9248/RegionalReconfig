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
//     The update controller routes the batch by that epoch's placement, sends the inserts' PQ
//     codes and raw vectors to their owners, and has every owner stage the next version of the
//     lists the batch changes next to the current one (STAGE_VERSIONS). Queries go on reading
//     the current versions.
//  2. CommitUpdate checks that every owner holds the versions the batch names, then flips: CAS,
//     install everywhere -- only the partitions' versions change, so all entries switch at
//     once -- grace (queries of the old epoch read the old versions until they end), and reclaim:
//     the owners drop the versions the batch replaced. A commit that fails before the flip drops
//     the staged versions itself, since the lease ends with the commit.
//  3. AbortUpdate gives the lease up without a flip. Dropping what was staged is the update
//     controller's job: it knows what it staged.
//
// A lease that sees neither a commit nor an abort for update_lease expires, so that an update
// controller that died does not block reconfigurations for good; a later commit is refused.

// updateLease is the lease of the batch in progress.
type updateLease struct {
	token string
	epoch uint64
	timer *time.Timer
}

// BeginUpdate starts an update batch: it takes the reconfiguration slot under a lease and
// returns the epoch the batch is for.
func (c *Controller) BeginUpdate(ctx context.Context, req protocol.BeginUpdateReq) (*protocol.UpdateLease, error) {
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
	if len(reply.Changed) == 0 {
		return reply, nil // the batch changed no list: nothing to flip
	}
	// Before the flip, a failure drops the staged versions: no epoch names them, and the batch's
	// lease has ended, so the update controller may no longer touch the nodes.
	flipped := false
	defer func() {
		if !flipped && err != nil {
			c.dropVersions(staged)
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

	// 4. RECLAIM: the owners drop the replaced versions. The PQ codes of the vectors the batch
	// deleted become cached with them; the update controller has every node free those.
	if err := forEach(owners, func(n *ManagedNode) error {
		return n.call(ctx, c.callTimeout(), protocol.DropVersionsM, protocol.DropVersionsReq{Versions: replaced[n.ID]}, nil)
	}); err != nil {
		return reply, afterFlip(err)
	}
	phase("reclaim")
	log.Printf("controller: update batch done: epoch %d -> %d, partitions %v at new versions, phases %v",
		reply.FromEpoch, reply.ToEpoch, reply.Changed, reply.Phases)
	return reply, nil
}

// dropVersions has the owners drop versions no epoch names (best effort: what stays only takes
// memory, and a later batch that stages the same version replaces it).
func (c *Controller) dropVersions(byOwner map[placement.NodeID][]protocol.PartitionVersion) {
	owners, err := c.managed(sortedKeys(byOwner))
	if err == nil {
		err = forEach(owners, func(n *ManagedNode) error {
			return n.call(context.Background(), c.callTimeout(), protocol.DropVersionsM, protocol.DropVersionsReq{Versions: byOwner[n.ID]}, nil)
		})
	}
	if err != nil {
		log.Printf("controller: dropping the versions of a batch that did not flip: %v", err)
	}
}
