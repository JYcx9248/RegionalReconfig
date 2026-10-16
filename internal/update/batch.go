package update

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sort"
	"sync"
	"time"

	"rtier/internal/ctrl"
	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
	"rtier/internal/placement"
	"rtier/internal/protocol"
)

// Controller is the controller's side of an update batch: *controller.Controller in-process,
// Remote over its API port.
type Controller interface {
	BeginUpdate(ctx context.Context, req protocol.BeginUpdateReq) (*protocol.UpdateLease, error)
	CommitUpdate(ctx context.Context, req protocol.CommitUpdateReq) (*protocol.UpdateReply, error)
	AbortUpdate(ctx context.Context, req protocol.AbortUpdateReq) error
}

// Remote is Controller over the controller's API port.
type Remote struct{ Conn *ctrl.Conn }

func (r Remote) BeginUpdate(ctx context.Context, req protocol.BeginUpdateReq) (*protocol.UpdateLease, error) {
	var out protocol.UpdateLease
	if err := r.Conn.Call(ctx, protocol.BeginUpdateA, req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (r Remote) CommitUpdate(ctx context.Context, req protocol.CommitUpdateReq) (*protocol.UpdateReply, error) {
	var out protocol.UpdateReply
	if err := r.Conn.Call(ctx, protocol.CommitUpdateA, req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (r Remote) AbortUpdate(ctx context.Context, req protocol.AbortUpdateReq) error {
	return r.Conn.Call(ctx, protocol.AbortUpdateA, req, nil)
}

// Updater runs update batches: the update controller's side (docs/drafts/
// dataset-updates-discussion.md, section 3). Prepare decides a batch under the controller's
// lease; Execute has the owners carry it out and the controller flip to it.
type Updater struct {
	Controller Controller
	Manifest   *partitioning.Manifest
	Alloc      *Allocator
	// Router routes the inserts (ASSIGN): the update controller's own rtier_node, holding the
	// index's navigation graph and codebook and nothing else (rtier-updater -router or -index).
	// Never a serving node: routing must not take the navigation CPU that queries need (section
	// 5 of the draft), and in a major epoch the update controller's graph is the one entries get.
	Router *nodeclient.Client

	mu      sync.Mutex
	clients map[string]*nodeclient.Client
}

// Batch is a decided batch. The decisions are data -- each insert's ID, location, lists and PQ
// code -- so that whoever applies the batch applies the same one: the owners, and the oracle of
// the end-to-end tests.
type Batch struct {
	Lease    *protocol.UpdateLease
	VecBytes int
	Vecs     []byte   // the inserted vectors, in the order of Inserts
	Inserts  []Insert // in the order of Vecs
	Deleted  []uint32 // ascending, without duplicates
	// Owners: the inserts each data node of the lease's epoch receives (ByOwner).
	Owners map[placement.NodeID][]int
	// Staged: by owner, the next version of each partition the batch changed (Execute).
	Staged map[placement.NodeID][]protocol.PartitionVersion
}

// Versions is every version the batch staged, by partition.
func (b *Batch) Versions() []protocol.PartitionVersion {
	var out []protocol.PartitionVersion
	for _, pvs := range b.Staged {
		out = append(out, pvs...)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Partition < out[j].Partition })
	return out
}

// Postings is every posting the batch adds: list, vector ID and location, insert by insert.
func (b *Batch) Postings() (lists, ids, locs []uint32) {
	for _, in := range b.Inserts {
		for _, l := range in.Lists {
			lists, ids, locs = append(lists, l), append(ids, in.ID), append(locs, in.Loc)
		}
	}
	return lists, ids, locs
}

// Per-call bounds: well under the data nodes' frames (256 MB) and update buffer (256 MB by
// default, rtier_node --update-buffer-mb).
const (
	pqPutBatch    = 1 << 16 // PQ codes per PQ_PUT
	rawAddBytes   = 1 << 20 // raw-vector bytes per RAW_ADD
	stagePostings = 1 << 22 // postings per STAGE_VERSIONS
	forgetBatch   = 1 << 20 // IDs per FORGET
)

// Prepare begins a batch: it takes the controller's lease, gives the inserts in vecs (vecBytes
// each, in the index's element type) IDs and locations, routes them by the build's rule, and
// checks that the owners have room for them. If it fails, it gives the lease up.
func (u *Updater) Prepare(ctx context.Context, vecs []byte, vecBytes int, deleted []uint32) (b *Batch, err error) {
	// A delete names a vector by the ID it was given. One that was never handed out cannot be
	// anywhere -- and one of this batch's inserts cannot be deleted by the same batch.
	next, _ := u.Alloc.Next()
	del := append([]uint32(nil), deleted...)
	sort.Slice(del, func(i, j int) bool { return del[i] < del[j] })
	del = dedup(del)
	if len(del) > 0 && uint64(del[len(del)-1]) >= next {
		return nil, fmt.Errorf("update: deleting vector %d, which was never inserted (the next ID is %d)", del[len(del)-1], next)
	}
	if len(vecs) > 0 && u.Router == nil {
		return nil, errors.New("update: no router: the update controller routes inserts with its own navigation graph")
	}
	lease, err := u.Controller.BeginUpdate(ctx, protocol.BeginUpdateReq{})
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, u.abort(ctx, lease))
		}
	}()
	t := lease.Table
	var ins []Insert
	if len(vecs) > 0 {
		if ins, err = Route(ctx, u.Router, u.Alloc, vecs, vecBytes); err != nil {
			return nil, err
		}
	}
	b = &Batch{Lease: lease, VecBytes: vecBytes, Vecs: vecs, Inserts: ins, Deleted: del,
		Owners: ByOwner(ins, u.Manifest, t), Staged: map[placement.NodeID][]protocol.PartitionVersion{}}
	if err := u.checkRoom(ctx, b); err != nil {
		return nil, err
	}
	return b, nil
}

// checkRoom: every data node has the IDs and locations the batch uses -- a partition can move
// to any of them -- and every owner the PQ codes of its inserts, on top of the codes it holds
// that it cannot evict: live ones, and staged ones, of which there are none once BeginUpdate has
// released the leftovers. That is the batch's peak: the codes of deleted vectors are freed only
// after the grace period. Over budget the batch fails here, before anything is sent: splitting
// the partitions that grew too large, or scaling out first (agreed item 10), is not
// implemented, and neither is growing the spaces (item 9).
func (u *Updater) checkRoom(ctx context.Context, b *Batch) error {
	var idEnd, locEnd uint64
	for _, in := range b.Inserts {
		idEnd, locEnd = max(idEnd, uint64(in.ID)+1), max(locEnd, uint64(in.Loc)+1)
	}
	t := b.Lease.Table
	nodes := t.Placement.Nodes()
	return parallel(ctx, len(nodes), func(ctx context.Context, i int) error {
		id := nodes[i]
		in, err := u.client(t.Nodes[id].NodeAddr).Info(ctx)
		if err != nil {
			return fmt.Errorf("update: node %d: %w", id, err)
		}
		if idEnd > in.IDCapacity || locEnd > in.Raw.Locations {
			return fmt.Errorf("update: node %d has room for %d vector IDs and %d locations, the batch needs %d and %d "+
				"(rtier_node --max-vectors, --max-pages)", id, in.IDCapacity, in.Raw.Locations, idEnd, locEnd)
		}
		held, n := in.PQ.Live+in.PQ.Staged, uint64(len(b.Owners[id]))
		if held+n > in.PQ.Capacity {
			return fmt.Errorf("update: node %d holds %d PQ codes it cannot evict and has room for %d, the batch brings %d "+
				"(U14: splitting partitions or scaling out before a batch is not implemented)", id, held, in.PQ.Capacity, n)
		}
		return nil
	})
}

// Execute has the owners stage b and the controller flip to it; the controller's reclaim frees
// the deleted vectors' PQ codes on every node. If the staging fails, Execute drops what it
// staged and gives the lease up. If the commit fails, the controller has dropped the staged
// versions when it failed before the flip, and stopped where it was after it (U13).
func (u *Updater) Execute(ctx context.Context, b *Batch) (*protocol.UpdateReply, error) {
	if err := u.stage(ctx, b); err != nil {
		u.undo(ctx, b)
		return nil, errors.Join(err, u.abort(ctx, b.Lease))
	}
	reply, err := u.Controller.CommitUpdate(ctx, protocol.CommitUpdateReq{Token: b.Lease.Token,
		FromEpoch: b.Lease.Table.Epoch, Versions: b.Versions(), Deleted: b.Deleted})
	if err != nil {
		// Whether the flip happened may be unknown here (a reply lost with the connection), so the
		// inserts are not declared dead: only their unreferenced codes go -- those of versions the
		// controller dropped -- which is safe either way.
		u.forget(ctx, b.ownerAddrs(), b.insertIDs(), false)
		return reply, err
	}
	return reply, nil
}

// stage sends each owner the PQ codes and raw vectors of its inserts, then has every data node
// stage the next version of its partitions that the batch changes. Nothing of it is visible
// before the flip: queries reach vectors only through lists, and they read the versions their
// epoch names.
func (u *Updater) stage(ctx context.Context, b *Batch) error {
	t := b.Lease.Table
	owners := make([]placement.NodeID, 0, len(b.Owners))
	for o := range b.Owners {
		owners = append(owners, o)
	}
	// 1. Codes and vectors, straight to the owners' data nodes. The raw vectors wait in memory
	// and are written to the SSD at the lowest priority, after the queries' reads and the
	// fetched vectors (U14): the flip does not wait for the disk.
	if err := parallel(ctx, len(owners), func(ctx context.Context, i int) error {
		o := owners[i]
		nc := u.client(t.Nodes[o].NodeAddr)
		is := b.Owners[o]
		m := len(b.Inserts[is[0]].Code)
		for lo := 0; lo < len(is); lo += pqPutBatch {
			hi := min(lo+pqPutBatch, len(is))
			ids, codes := make([]uint32, 0, hi-lo), make([]byte, 0, (hi-lo)*m)
			for _, k := range is[lo:hi] {
				ids, codes = append(ids, b.Inserts[k].ID), append(codes, b.Inserts[k].Code...)
			}
			if _, _, err := nc.PQPut(ctx, ids, m, codes); err != nil {
				return fmt.Errorf("update: PQ codes to node %d: %w", o, err)
			}
		}
		per := max(1, rawAddBytes/b.VecBytes)
		for lo := 0; lo < len(is); lo += per {
			hi := min(lo+per, len(is))
			locs, vecs := make([]uint32, 0, hi-lo), make([]byte, 0, (hi-lo)*b.VecBytes)
			for _, k := range is[lo:hi] {
				locs, vecs = append(locs, b.Inserts[k].Loc), append(vecs, b.Vecs[k*b.VecBytes:(k+1)*b.VecBytes]...)
			}
			// The node holds the call while its update buffer is full: give it time to write.
			actx, cancel := context.WithTimeout(ctx, 10*time.Minute)
			_, _, err := nc.RawAdd(actx, locs, b.VecBytes, vecs)
			cancel()
			if err != nil {
				return fmt.Errorf("update: raw vectors to node %d: %w", o, err)
			}
		}
		return nil
	}); err != nil {
		return err
	}

	// 2. The next versions. Postings go to the owner of their list's partition; the deleted IDs
	// go to every data node, which finds its copies itself (agreed item 4). Each call is all or
	// nothing for the partitions it names, and partitions the batch leaves alone get no version.
	type posting struct{ list, id, loc uint32 }
	byPart := map[int][]posting{}
	for _, in := range b.Inserts {
		for _, l := range in.Lists {
			p := u.Manifest.PartitionOf(l)
			byPart[p] = append(byPart[p], posting{l, in.ID, in.Loc})
		}
	}
	var mu sync.Mutex
	nodes := t.Placement.Nodes()
	return parallel(ctx, len(nodes), func(ctx context.Context, i int) error {
		o := nodes[i]
		nc := u.client(t.Nodes[o].NodeAddr)
		var parts []int
		for _, p := range t.Placement.PartitionsOf(o) {
			if len(b.Deleted) > 0 || len(byPart[p]) > 0 {
				parts = append(parts, p)
			}
		}
		for lo := 0; lo < len(parts); {
			var vs []nodeclient.VersionStage
			var lists, ids, locs []uint32
			hi := lo
			for ; hi < len(parts) && (hi == lo || len(lists)+len(byPart[parts[hi]]) <= stagePostings); hi++ {
				p := parts[hi]
				v := t.Version(p)
				vs = append(vs, nodeclient.VersionStage{Partition: p, Base: v, Next: v + 1})
				for _, x := range byPart[p] {
					lists, ids, locs = append(lists, x.list), append(ids, x.id), append(locs, x.loc)
				}
			}
			res, err := nc.StageVersions(ctx, vs, lists, ids, locs, b.Deleted)
			if err != nil {
				return fmt.Errorf("update: staging partitions %v on node %d: %w", parts[lo:hi], o, err)
			}
			mu.Lock()
			for k, r := range res {
				if r.Changed {
					b.Staged[o] = append(b.Staged[o], protocol.PartitionVersion{Partition: r.Partition, Version: vs[k].Next})
				}
			}
			mu.Unlock()
			lo = hi
		}
		return nil
	})
}

// undo drops what stage put on the nodes, while the batch still holds its lease (no other
// staging runs, and the batch will not flip): the staged versions, then the inserts' PQ codes --
// those still protected for the batch, and the rest as deleted, since their IDs are never named
// again. The raw vectors stay where they were sent: no list names them, and their locations are
// never handed out again.
func (u *Updater) undo(ctx context.Context, b *Batch) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), time.Minute)
	defer cancel()
	t := b.Lease.Table
	for o, pvs := range b.Staged {
		for _, pv := range pvs {
			if _, err := u.client(t.Nodes[o].NodeAddr).DropVersion(ctx, pv.Partition, pv.Version); err != nil {
				log.Printf("update: undo: dropping partition %d at version %d on node %d: %v", pv.Partition, pv.Version, o, err)
			}
		}
	}
	for o := range b.Owners {
		if _, err := u.client(t.Nodes[o].NodeAddr).PQRelease(ctx); err != nil {
			log.Printf("update: undo: releasing PQ codes on node %d: %v", o, err)
		}
	}
	u.forget(ctx, b.ownerAddrs(), b.insertIDs(), true)
}

// forget has the data nodes at addrs free the PQ codes of ids, inserts no list will name
// (nodeclient.ForgetVectors; deleted: also those a version still names, with that version). It
// only frees memory, so failures are logged.
func (u *Updater) forget(ctx context.Context, addrs []string, ids []uint32, deleted bool) {
	if len(ids) == 0 {
		return
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), time.Minute)
	defer cancel()
	parallel(ctx, len(addrs), func(ctx context.Context, i int) error {
		for lo := 0; lo < len(ids); lo += forgetBatch {
			if _, err := u.client(addrs[i]).ForgetVectors(ctx, ids[lo:min(lo+forgetBatch, len(ids))], deleted); err != nil {
				log.Printf("update: freeing the PQ codes of %d vectors on %s: %v", len(ids), addrs[i], err)
				return nil
			}
		}
		return nil
	})
}

func (u *Updater) abort(ctx context.Context, lease *protocol.UpdateLease) error {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), time.Minute)
	defer cancel()
	return u.Controller.AbortUpdate(ctx, protocol.AbortUpdateReq{Token: lease.Token})
}

func (u *Updater) client(addr string) *nodeclient.Client {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.clients == nil {
		u.clients = map[string]*nodeclient.Client{}
	}
	c, ok := u.clients[addr]
	if !ok {
		c = nodeclient.New(addr, 2)
		u.clients[addr] = c
	}
	return c
}

// Close closes the data-node connections.
func (u *Updater) Close() {
	u.mu.Lock()
	defer u.mu.Unlock()
	for _, c := range u.clients {
		c.Close()
	}
	u.clients = nil
}

func (b *Batch) ownerAddrs() []string {
	var out []string
	for o := range b.Owners {
		out = append(out, b.Lease.Table.Nodes[o].NodeAddr)
	}
	return out
}

func (b *Batch) insertIDs() []uint32 {
	out := make([]uint32, len(b.Inserts))
	for i, in := range b.Inserts {
		out[i] = in.ID
	}
	return out
}

func dedup(s []uint32) []uint32 {
	out := s[:0]
	for i, x := range s {
		if i == 0 || x != s[i-1] {
			out = append(out, x)
		}
	}
	return out
}

// parallel runs fn(ctx, 0..n-1) concurrently and returns the first error; the context passed
// to fn is cancelled as soon as one call fails.
func parallel(ctx context.Context, n int, fn func(ctx context.Context, i int) error) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	errs := make([]error, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if errs[i] = fn(ctx, i); errs[i] != nil {
				cancel()
			}
		}(i)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil && !errors.Is(err, context.Canceled) {
			return err
		}
	}
	return errors.Join(errs...)
}
