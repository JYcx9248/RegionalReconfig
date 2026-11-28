// Package update is the update controller (U14): it decides a batch of dataset updates -- the
// IDs and raw-vector locations of inserted vectors, the posting lists each insert joins and its
// PQ code, the owners that receive it -- and has the data nodes execute it: the owners stage the
// next version of the lists the batch changes, and the controller flips to them (batch.go). The
// design is in docs/drafts/dataset-updates-discussion.md.
package update

import (
	"context"
	"fmt"
	"runtime"
	"sort"
	"sync/atomic"

	"rtier/internal/epoch"
	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
	"rtier/internal/placement"
)

// Allocator hands out vector IDs and raw-vector locations, and only ever appends: an ID or a
// location is never given out twice, not even after a delete, because former owners keep
// cached PQ codes and raw vectors by ID and location, and a reused one could alias such a copy
// on a node that has not processed the delete yet. The spaces are the data nodes'
// (rtier_node --max-vectors, --max-pages). Growing them at an epoch boundary is not
// implemented: Take fails once they run out.
type Allocator struct {
	nextID, idEnd   uint64
	nextLoc, locEnd uint64
	nextList        uint64 // the first list ID major epochs have not handed out (0: none yet)
	listGen         uint32 // the index generation nextList counts in (U14 full rebuild)
}

// NewAllocator starts at the first free ID and location (the index's vector count and
// location count, before any insert) and stops at the ends of the nodes' spaces.
func NewAllocator(firstID, idEnd, firstLoc, locEnd uint64) (*Allocator, error) {
	if firstID > idEnd || firstLoc > locEnd || idEnd > 1<<32-1 || locEnd > 1<<32-1 {
		return nil, fmt.Errorf("update: bad spaces: IDs [%d, %d), locations [%d, %d)", firstID, idEnd, firstLoc, locEnd)
	}
	return &Allocator{nextID: firstID, idEnd: idEnd, nextLoc: firstLoc, locEnd: locEnd}, nil
}

// Take reserves the next n IDs and the next n locations.
func (a *Allocator) Take(n int) (ids, locs []uint32, err error) {
	if a.nextID+uint64(n) > a.idEnd {
		return nil, nil, fmt.Errorf("update: %d inserts do not fit the %d vector IDs left (U14: growing the "+
			"space at an epoch boundary is not implemented)", n, a.idEnd-a.nextID)
	}
	if a.nextLoc+uint64(n) > a.locEnd {
		return nil, nil, fmt.Errorf("update: %d inserts do not fit the %d raw-vector locations left (U14: "+
			"growing the space at an epoch boundary is not implemented)", n, a.locEnd-a.nextLoc)
	}
	ids, locs = make([]uint32, n), make([]uint32, n)
	for i := range ids {
		ids[i], locs[i] = uint32(a.nextID+uint64(i)), uint32(a.nextLoc+uint64(i))
	}
	a.nextID += uint64(n)
	a.nextLoc += uint64(n)
	return ids, locs, nil
}

// Next is the first ID and the first location not handed out yet.
func (a *Allocator) Next() (id, loc uint64) { return a.nextID, a.nextLoc }

// TakeLists reserves the next n list IDs for the lists a major epoch creates, from floor at the
// least (above every list that exists). Like vector IDs, list IDs are never handed out twice.
func (a *Allocator) TakeLists(n int, floor uint32) []uint32 {
	if a.nextList < uint64(floor) {
		a.nextList = uint64(floor)
	}
	out := make([]uint32, n)
	for i := range out {
		out[i] = uint32(a.nextList + uint64(i))
	}
	a.nextList += uint64(n)
	return out
}

// NextList is the first list ID not handed out yet (0 if none was) in index generation gen;
// SetNextList restores it. List IDs count within a generation: a full rebuild (U14) brings
// lists of its own, and the count starts over above them.
func (a *Allocator) NextList() (next uint64, gen uint32) { return a.nextList, a.listGen }
func (a *Allocator) SetNextList(next uint64, gen uint32) { a.nextList, a.listGen = next, gen }

// generation makes the allocator count list IDs in index generation gen.
func (a *Allocator) generation(gen uint32) {
	if a.listGen != gen {
		a.nextList, a.listGen = 0, gen
	}
}

// Insert is one routed insert: its ID and location, the posting lists it joins and its PQ code.
type Insert struct {
	ID, Loc uint32
	Lists   []uint32
	Code    []byte
}

// assignBatch bounds one ASSIGN call (its frame).
const assignBatch = 4096

// RouteStreams is how many ASSIGN calls Route keeps in flight: half of this machine's CPUs. The
// router runs here, next to the update controller (StartRouter, or the experiment runner's own),
// and the router runs each call on a connection thread of its own, as it does NAVIGATE. The
// other half stays with what shares the machine -- in an experiment, the cluster's controller and
// the load generator. One call at a time, a batch took hours: ASSIGN is the build's search on the
// graph for every vector, about 5 ms of a core each on Wikipedia (1024 dimensions, 1.6M heads),
// so the 775K inserts of a 5% batch need over an hour of one core.
func RouteStreams() int { return max(1, runtime.NumCPU()/2) }

// Route gives each of the vectors in vecs (vecBytes each, in the index's element type) an ID
// and a location from alloc, and routes it with the router node's ASSIGN: the lists the index's
// build would have given it, by the build's own rule on the navigation graph, so that inserted
// vectors are spread over lists the way built ones are, and its PQ code. The calls run side by
// side (RouteStreams), each over its own vectors, so the result is the same as one at a time.
func Route(ctx context.Context, router *nodeclient.Client, alloc *Allocator, vecs []byte, vecBytes int) ([]Insert, error) {
	if vecBytes <= 0 || len(vecs)%vecBytes != 0 {
		return nil, fmt.Errorf("update: %d bytes are not vectors of %d bytes", len(vecs), vecBytes)
	}
	n := len(vecs) / vecBytes
	ids, locs, err := alloc.Take(n)
	if err != nil {
		return nil, err
	}
	calls := (n + assignBatch - 1) / assignBatch
	got := make([][]nodeclient.Assignment, calls)
	var next atomic.Int64
	err = parallel(ctx, min(calls, RouteStreams()), func(ctx context.Context, _ int) error {
		for c := int(next.Add(1) - 1); c < calls; c = int(next.Add(1) - 1) {
			i, j := c*assignBatch, min(n, (c+1)*assignBatch)
			as, err := router.Assign(ctx, vecs[i*vecBytes:j*vecBytes], j-i)
			if err != nil {
				return err
			}
			got[c] = as
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	out := make([]Insert, 0, n)
	for c, as := range got {
		for k, a := range as {
			i := c*assignBatch + k
			if len(a.Lists) == 0 {
				return nil, fmt.Errorf("update: the router gave vector %d no list", i)
			}
			out = append(out, Insert{ID: ids[i], Loc: locs[i], Lists: a.Lists, Code: a.Code})
		}
	}
	return out, nil
}

// ByOwner returns the inserts each node of table t receives (indexes into ins, ascending): an
// insert goes to every owner of one of its lists' partitions. Each such owner takes the vector
// once, however many of its lists name it, since PQ codes and raw vectors are per node (U1).
func ByOwner(ins []Insert, m *partitioning.Manifest, t *epoch.Table) map[placement.NodeID][]int {
	out := map[placement.NodeID][]int{}
	for i, in := range ins {
		var owners []placement.NodeID
		for _, l := range in.Lists {
			o := t.Owner(t.PartitionOf(l, m))
			if !containsNode(owners, o) {
				owners = append(owners, o)
			}
		}
		for _, o := range owners {
			out[o] = append(out[o], i)
		}
	}
	for _, is := range out {
		sort.Ints(is)
	}
	return out
}

func containsNode(s []placement.NodeID, id placement.NodeID) bool {
	for _, x := range s {
		if x == id {
			return true
		}
	}
	return false
}
