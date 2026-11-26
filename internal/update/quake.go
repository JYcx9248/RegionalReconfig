package update

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"

	"rtier/internal/epoch"
	"rtier/internal/nodeclient"
	"rtier/internal/placement"
	"rtier/internal/protocol"
)

// PolicyQuake is Quake's maintenance (OSDI '25; github.com/marius-team/quake, main:
// MaintenancePolicy::perform_maintenance, MaintenanceCostEstimator, PartitionManager's
// split_partitions, delete_partitions and refine_partitions, kmeans_refine_partitions) as a
// major epoch (U14), on Quake's index: fusion_build --clustering kmeans, whose heads are the
// lists' k-means means and which puts every vector in one list, its nearest head's. One
// maintenance round is one major epoch, which PrepareQuake decides under the lease, in the order
// Quake's code takes:
//
//  1. The window: the last Window client queries the entries kept (QuerySample), each navigated
//     on the router's graph as it ran (its nprobe and ef): the lists it scanned. A list's hit
//     rate is the share of the window's queries that scanned it, and the scan fraction the mean
//     share of all postings a query scanned. With fewer queries than the window there is no
//     maintenance, as Quake waits for its window to fill.
//  2. The decisions, for every live list on its length and hit rate, by Quake's cost model on
//     the latency L(n) of scanning a list of n (top-10), which Quake profiles on random data and
//     interpolates (ListScanLatencyEstimator; here the router's CPU filter is profiled, the scan
//     our lists get): a list is deleted when the delete delta is below -DeleteThresholdNS -- with
//     DeleteRejection and more than MinSize postings, the delta on where its members would go,
//     their nearest heads but its own -- or else split when it holds more than MinSize postings
//     and the split delta is below -SplitThresholdNS. With MaxSize, by length alone: split above
//     it, delete below MinSize (Quake's size-bound mode, its LIRE baseline).
//  3. Deletes: every member of a deleted list joins the list of its nearest live head, the
//     deleted lists' left out -- a list that is about to split too.
//  4. Splits: Quake's split, a k-means of 2 on a split list's members (with those that joined
//     it), each member to the nearer mean, the two means the children's heads.
//  5. Refinement: the RefineRadius live heads nearest each child's (the child among them), and
//     every member of their lists to the nearest of their heads, for max(1, RefineIterations)
//     passes, every pass but the first on the means of the pass before; with two passes or more
//     the heads move to the means the last pass used (in the graph too, in place).
//  6. Every list left empty is removed.
//
// The router's graph is edited after the splits (the children in; the deleted lists' and the
// split lists' heads marked deleted) and that edit amended after the refinement (heads moved,
// empty lists' heads marked deleted): one patch, as every major epoch has.
//
// Quake's own differences, kept out: it decides on raw vectors, we on PQ codes, the router's (no
// data node computes); its window records the lists each query scanned as it ran, at their
// sizes then, ours reads them at the round from the queries the entries kept; its centroids
// drift with every insert and delete (update_centroid) and a list that lost more than
// partition_reduction_threshold of its vectors to deletes is deleted (delete_factor), ours stay
// where a split or a refinement put them, with no count of deletes per list across batches; its
// parent level is a flat index (or more levels, with their own maintenance), ours the graph,
// changed by the patches; it adapts nprobe (APS), our queries' nprobe is fixed; the scan
// fraction's total is the vector count when its policy started, ours the postings at the round;
// a mean that takes no member in a refinement pass is Quake's 0/0, here that list takes no
// member in the later passes and is removed as empty; it runs when its workload calls it, we
// once after an update batch.
const PolicyQuake = "quake"

// QuakeParams are Quake's maintenance parameters (MaintenancePolicyParams), with its defaults
// (DefaultQuakeParams; common.h).
type QuakeParams struct {
	Window            int     // window_size: the queries whose hits decide
	RefineRadius      int     // refinement_radius: heads nearest each child refined (0: none)
	RefineIterations  int     // refinement_iterations
	MinSize           int     // min_partition_size
	MaxSize           int     // max_partition_size; 0 (Quake's -1): the cost model decides
	Alpha             float64 // alpha: a split list's halves cost 2 * alpha * L(n/2) each query
	DeleteRejection   bool    // enable_delete_rejection
	SplitIterations   int     // split_knn_iterations: the 2-means' iterations
	DeleteThresholdNS float64 // delete_threshold_ns
	SplitThresholdNS  float64 // split_threshold_ns
}

// DefaultQuakeParams is Quake's defaults.
func DefaultQuakeParams() QuakeParams {
	return QuakeParams{Window: 1000, RefineRadius: 25, RefineIterations: 3, MinSize: 32, Alpha: 0.9,
		DeleteRejection: true, SplitIterations: 5, DeleteThresholdNS: 100, SplitThresholdNS: 100}
}

// QuakeStats is what one maintenance round found and did.
type QuakeStats struct {
	Window       int     `json:"window"`        // queries in the window
	Lists        int     `json:"lists"`         // live lists at the round
	Postings     int     `json:"postings"`      // their postings
	ScanFraction float64 `json:"scan_fraction"` // the window's mean share of the postings scanned
	Deletes      int     `json:"deletes"`
	Rejected     int     `json:"rejected"` // deletes the rejection turned down
	Splits       int     `json:"splits"`
	Refined      int     `json:"refined"`           // lists the refinement reassigned
	Reassigned   int     `json:"reassigned"`        // vectors it put in another list
	Moved        int     `json:"moved_heads"`       // heads it moved
	Emptied      int     `json:"removed_empty"`     // lists removed empty
	Skipped      string  `json:"skipped,omitempty"` // why no round ran
}

// Quake's scan-latency grid (common.h: DEFAULT_LATENCY_ESTIMATOR_RANGE_N, _K, _NTRIALS) and the
// cost estimator's k (MaintenancePolicy's MaintenanceCostEstimator(d, alpha, 10)).
var (
	quakeGridN = []int{1, 2, 4, 16, 64, 256, 1024, 4096, 16384, 65536}
	quakeGridK = []int{1, 4, 16, 64, 256}
)

const (
	quakeTrials = 5
	quakeK      = 10
	// Faiss's default seed (ClusteringParameters), which every one of Quake's splits runs with.
	quakeSplitSeed = 1234
)

// scanLatency is Quake's ListScanLatencyEstimator: mean scan times by list size and top-k on a
// grid, interpolated bilinearly inside it and extrapolated linearly from its last two rows or
// columns outside it (estimate_scan_latency).
type scanLatency struct {
	ns, ks []int
	grid   [][]float64 // [n][k], ns
}

func linearExtrapolate(f1, f2, frac float64) float64 { return f2 + (f2-f1)*frac }

// at is the grid's interpolation index and fraction for v (estimate_scan_latency's i_lower,
// i_upper, t), and whether v is within the grid.
func gridAt(vals []int, v int) (lo, hi int, t float64, within bool) {
	last := len(vals) - 1
	if v <= vals[last] {
		hi = sort.Search(len(vals), func(i int) bool { return vals[i] > v }) // std::upper_bound
		if hi == len(vals) {
			return last - 1, last, 1, true
		}
		lo = hi - 1
		return lo, hi, float64(v-vals[lo]) / float64(vals[hi]-vals[lo]), true
	}
	lo, hi = last-1, last
	return lo, hi, float64(v-vals[hi]) / float64(vals[hi]-vals[lo]), false
}

func (e *scanLatency) estimate(n, k int) float64 {
	if n == 0 || k == 0 {
		return 0
	}
	if n < e.ns[0] || k < e.ks[0] {
		return 0 // Quake throws; n, k >= 1 here
	}
	il, iu, t, nIn := gridAt(e.ns, n)
	jl, ju, u, kIn := gridAt(e.ks, k)
	f := e.grid
	switch {
	case nIn && kIn:
		return (1-t)*(1-u)*f[il][jl] + t*(1-u)*f[iu][jl] + (1-t)*u*f[il][ju] + t*u*f[iu][ju]
	case !nIn && kIn:
		lo, hi := linearExtrapolate(f[il][jl], f[iu][jl], t), linearExtrapolate(f[il][ju], f[iu][ju], t)
		return (1-u)*lo + u*hi
	case nIn && !kIn:
		lo, hi := linearExtrapolate(f[il][jl], f[il][ju], u), linearExtrapolate(f[iu][jl], f[iu][ju], u)
		return (1-t)*lo + t*hi
	default:
		lo, hi := linearExtrapolate(f[il][jl], f[iu][jl], t), linearExtrapolate(f[il][ju], f[iu][ju], t)
		return linearExtrapolate(lo, hi, u)
	}
}

// quakeCost is Quake's MaintenanceCostEstimator: the change in a query's expected scan time,
// in ns, if a list split or were deleted.
type quakeCost struct {
	lat   *scanLatency
	alpha float64
	k     int
}

func (c *quakeCost) l(n int) float64 { return c.lat.estimate(n, c.k) }

// splitDelta is compute_split_delta: one more list to search, and a list of size scanned at its
// hit rate replaced by two halves, alpha times as likely each.
func (c *quakeCost) splitDelta(size int, hit float64, total int) float64 {
	overhead := c.l(total+1) - c.l(total)
	return overhead + c.l(size/2)*hit*(2*c.alpha) - c.l(size)*hit
}

// deleteDelta is compute_delete_delta: one list fewer, its postings and hits spread over the
// others (avgHit: Quake passes the window's scan fraction there).
func (c *quakeCost) deleteDelta(size int, hit float64, total int, avgHit float64, avgSize int) float64 {
	if total <= 1 {
		return 0
	}
	overhead := c.l(total-1) - c.l(total)
	costOld := float64(total-1)*avgHit*c.l(avgSize) + hit*c.l(size)
	mergedSize := float64(avgSize) + float64(size)/float64(total-1)
	mergedHit := avgHit + hit/float64(total-1)
	var costNew float64
	if size < total {
		costNew = float64(size)*mergedHit*c.l(avgSize+1) + float64(total-size-1)*mergedHit*c.l(avgSize)
	} else {
		costNew = float64(total-1) * mergedHit * c.l(int(math.Ceil(mergedSize)))
	}
	return overhead + costNew - costOld
}

// deleteDeltaReassign is compute_delete_delta_w_reassign, as Quake's code has it: the list's
// own scan cost is added, and every list its members would go to grows by the whole list.
func (c *quakeCost) deleteDeltaReassign(size int, hit float64, total int, sizes []int, hits []float64) float64 {
	if total <= 1 {
		return 0
	}
	delta := c.l(total-1) - c.l(total) + hit*c.l(size)
	for i := range sizes {
		delta += (hits[i]+hit)*c.l(sizes[i]+size) - hits[i]*c.l(sizes[i])
	}
	return delta
}

// quakeDecisions is what a round's cost model chose.
type quakeDecisions struct {
	deletes, splits []uint32
	rejected        int
}

// decide is perform_maintenance's decisions, for every live list (lengths) on its hits in a
// window of `window` queries; members reads a list's members, for the delete rejection.
func (p QuakeParams) decide(c *quakeCost, lengths map[uint32]int, hits map[uint32]int, window int, scanFraction float64,
	rejectTargets func(l uint32) ([]uint32, error)) (quakeDecisions, error) {
	var d quakeDecisions
	lists := make([]uint32, 0, len(lengths))
	total := 0
	for l, n := range lengths {
		lists = append(lists, l)
		total += n
	}
	sort.Slice(lists, func(i, j int) bool { return lists[i] < lists[j] })
	if p.MaxSize > 0 {
		for _, l := range lists {
			switch n := lengths[l]; {
			case n > p.MaxSize:
				d.splits = append(d.splits, l)
			case n < p.MinSize:
				d.deletes = append(d.deletes, l)
			}
		}
		return d, nil
	}
	T := len(lists)
	if T == 0 {
		return d, nil
	}
	avgSize := total / T
	rate := func(l uint32) float64 { return float64(hits[l]) / float64(window) }
	for _, l := range lists {
		size, hit := lengths[l], rate(l)
		if c.deleteDelta(size, hit, T, scanFraction, avgSize) < -p.DeleteThresholdNS {
			if !p.DeleteRejection || size <= p.MinSize {
				d.deletes = append(d.deletes, l)
				continue
			}
			targets, err := rejectTargets(l)
			if err != nil {
				return d, err
			}
			sizes := make([]int, len(targets))
			rates := make([]float64, len(targets))
			for i, tl := range targets {
				sizes[i], rates[i] = lengths[tl], rate(tl)
			}
			if c.deleteDeltaReassign(size, hit, T, sizes, rates) < -p.DeleteThresholdNS {
				d.deletes = append(d.deletes, l)
			} else {
				d.rejected++
			}
			continue // not considered for a split (Quake's else branch)
		}
		if size > p.MinSize && c.splitDelta(size, hit, T) < -p.SplitThresholdNS {
			d.splits = append(d.splits, l)
		}
	}
	return d, nil
}

// qvec is a vector a round moves or reads: where its raw vector is, its PQ code, and the list it
// is in at the epoch's version.
type qvec struct {
	loc    uint32
	holder uint16
	code   []byte
	base   uint32
}

// PrepareQuake begins a major epoch that is one round of Quake's maintenance (PolicyQuake): it
// takes the controller's lease, brings the router's graph to the epoch's version, decides the
// round, and edits the router's graph, keeping the patch. A round with nothing to do gives the
// lease up and returns no epoch, and stats saying why. If it fails, it gives the lease up and
// brings the router back.
func (u *Updater) PrepareQuake(ctx context.Context) (mj *Major, st *QuakeStats, err error) {
	p := u.Quake
	if p.Window <= 0 || p.RefineRadius < 0 || p.RefineIterations < 0 || p.MinSize < 0 || p.MaxSize < 0 ||
		p.Alpha <= 0 || p.SplitIterations < 0 {
		return nil, nil, fmt.Errorf("update: Quake's parameters %+v", p)
	}
	if u.Router == nil {
		return nil, nil, errors.New("update: no router: Quake's maintenance edits the update controller's own navigation graph")
	}
	st = &QuakeStats{}
	lease, err := u.Controller.BeginUpdate(ctx, protocol.BeginUpdateReq{})
	if err != nil {
		return nil, nil, err
	}
	edited := false
	defer func() {
		if err != nil || mj == nil {
			if edited {
				err = errors.Join(err, u.syncRouter(context.WithoutCancel(ctx), lease.Table))
			}
			err = errors.Join(err, u.abort(ctx, lease))
		}
	}()
	t := lease.Table
	if err := u.follow(t); err != nil {
		return nil, nil, err
	}
	if err := u.syncRouter(ctx, t); err != nil {
		return nil, nil, err
	}
	in, err := u.Router.Info(ctx)
	if err != nil {
		return nil, nil, fmt.Errorf("update: router: %w", err)
	}

	// Every live list's length, from its owner at the epoch's version.
	lengths, err := u.listLengths(ctx, t)
	if err != nil {
		return nil, nil, err
	}
	st.Lists = len(lengths)
	for _, n := range lengths {
		st.Postings += n
	}

	// 1. The window, and each list's hits in it.
	hits := map[uint32]int{}
	if p.MaxSize == 0 {
		qs, err := u.Controller.QuerySamples(ctx, protocol.QuerySamplesReq{Max: p.Window})
		if err != nil {
			return nil, nil, fmt.Errorf("update: the entries' query samples: %w", err)
		}
		st.Window = len(qs.Queries)
		if st.Window < p.Window {
			st.Skipped = fmt.Sprintf("the window is not full: %d of %d queries", st.Window, p.Window)
			return nil, st, nil
		}
		scanned := make([][]uint32, len(qs.Queries))
		if err := parallelN(ctx, len(qs.Queries), 16, func(ctx context.Context, i int) error {
			q := qs.Queries[i]
			ls, err := u.Router.Navigate(ctx, t.Epoch, nodeclient.NewestGraph, q.Vec, q.NProbe, q.EF)
			scanned[i] = ls
			return err
		}); err != nil {
			return nil, nil, fmt.Errorf("update: navigating the window's queries: %w", err)
		}
		var frac float64
		for _, ls := range scanned {
			seen := map[uint32]bool{}
			n := 0
			for _, l := range ls {
				if !seen[l] {
					seen[l] = true
					hits[l]++
					n += lengths[l]
				}
			}
			if st.Postings > 0 {
				frac += float64(n) / float64(st.Postings)
			}
		}
		st.ScanFraction = frac / float64(len(scanned))
	}

	// 2. The decisions, on the cost model.
	if u.quakeLat == nil && p.MaxSize == 0 {
		ns, ks := make([]uint32, len(quakeGridN)), make([]uint32, len(quakeGridK))
		for i, v := range quakeGridN {
			ns[i] = uint32(v)
		}
		for i, v := range quakeGridK {
			ks[i] = uint32(v)
		}
		prof, err := u.Router.ScanProfile(ctx, ns, ks, quakeTrials)
		if err != nil {
			return nil, nil, fmt.Errorf("update: the router's scan profile: %w", err)
		}
		lat := &scanLatency{ns: quakeGridN, ks: quakeGridK, grid: make([][]float64, len(ns))}
		for i := range ns {
			lat.grid[i] = make([]float64, len(ks))
			for j := range ks {
				lat.grid[i][j] = float64(prof[i*len(ks)+j])
			}
		}
		u.quakeLat = lat
	}
	cost := &quakeCost{lat: u.quakeLat, alpha: p.Alpha, k: quakeK}
	members := map[uint32]*listMembers{}
	var mu sync.Mutex
	read := func(ctx context.Context, ls []uint32) error {
		var need []uint32
		mu.Lock()
		for _, l := range ls {
			if _, ok := members[l]; !ok {
				need = append(need, l)
			}
		}
		mu.Unlock()
		got, m, err := u.readLists(ctx, t, need)
		if err != nil {
			return err
		}
		mu.Lock()
		defer mu.Unlock()
		for l, lm := range got {
			members[l] = lm
		}
		if m > 0 {
			mj.M = m
		}
		return nil
	}
	mj = &Major{Lease: lease, Graph: t.Graph + 1, Lists: map[uint32]int{}, Parts: map[int]*PartEdits{},
		Codes: map[placement.NodeID]*CodeSet{}, Staged: map[placement.NodeID][]protocol.PartitionVersion{}, Quake: st}
	reject := func(l uint32) ([]uint32, error) {
		if err := read(ctx, []uint32{l}); err != nil {
			return nil, err
		}
		lm := members[l]
		if len(lm.IDs) == 0 {
			return nil, nil
		}
		near, err := u.nearestLive(ctx, lm.codes, mj.M, 2, nil)
		if err != nil {
			return nil, fmt.Errorf("update: where list %d's members would go: %w", l, err)
		}
		seen := map[uint32]bool{}
		var out []uint32
		for _, ns := range near {
			for _, tl := range ns {
				if tl != l && !seen[tl] {
					seen[tl] = true
					out = append(out, tl)
				}
			}
		}
		sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
		return out, nil
	}
	dec, err := p.decide(cost, lengths, hits, max(st.Window, 1), st.ScanFraction, reject)
	if err != nil {
		return nil, nil, err
	}
	st.Deletes, st.Splits, st.Rejected = len(dec.deletes), len(dec.splits), dec.rejected
	var empty []uint32 // live lists with no posting: Quake's last step removes them
	for l, n := range lengths {
		if n == 0 {
			empty = append(empty, l)
		}
	}
	isDel := map[uint32]bool{}
	for _, l := range dec.deletes {
		isDel[l] = true
	}
	for i := 0; i < len(dec.splits); i++ { // Quake's k-means of 2 needs two members
		if l := dec.splits[i]; lengths[l] < 2 {
			dec.splits = append(dec.splits[:i], dec.splits[i+1:]...)
			i--
		}
	}
	st.Splits = len(dec.splits)
	if len(dec.deletes)+len(dec.splits)+len(empty) == 0 {
		st.Skipped = "nothing to split, delete or remove"
		mj = nil
		return nil, st, nil
	}
	if err := read(ctx, append(append([]uint32(nil), dec.deletes...), dec.splits...)); err != nil {
		return nil, nil, err
	}

	// Each vector the round reads, once: Quake's index has one list per vector.
	vecs := map[uint32]*qvec{}
	note := func(l uint32) error {
		lm := members[l]
		for k, id := range lm.IDs {
			if v, ok := vecs[id]; ok && v.base != l {
				return fmt.Errorf("update: vector %d is in lists %d and %d: Quake's maintenance needs an index "+
					"with one list per vector (fusion_build --clustering kmeans)", id, v.base, l)
			}
			vecs[id] = &qvec{loc: lm.Locs[k], holder: lm.Holders[k], code: lm.code(k, mj.M), base: l}
		}
		return nil
	}
	for _, l := range append(append([]uint32(nil), dec.deletes...), dec.splits...) {
		if err := note(l); err != nil {
			return nil, nil, err
		}
	}

	// 3. Deletes: each member to the list of its nearest live head but the deleted ones'.
	final := map[uint32][]uint32{} // the lists the round changes -> their members after it
	joined := map[uint32][]uint32{}
	for _, l := range dec.deletes {
		lm := members[l]
		mj.Merges = append(mj.Merges, Merge{List: l, Partition: lm.p, Postings: len(lm.IDs), Into: noList})
		if len(lm.IDs) == 0 {
			continue
		}
		near, err := u.nearestLive(ctx, lm.codes, mj.M, 1, dec.deletes)
		if err != nil {
			return nil, nil, fmt.Errorf("update: deleting list %d: %w", l, err)
		}
		for k, id := range lm.IDs {
			if len(near[k]) == 0 {
				return nil, nil, errors.New("update: Quake's deletes leave no live list")
			}
			joined[near[k][0]] = append(joined[near[k][0]], id)
		}
	}
	for _, l := range dec.deletes {
		final[l] = nil
	}

	// 4. Splits: a k-means of 2 on the members, those that joined included.
	var children []uint32
	if len(dec.splits) > 0 {
		children = u.Alloc.TakeLists(2*len(dec.splits), listFloor(t, u.Manifest))
	}
	vb := in.Dim * dtypeSize(in.DType)
	var labels, dead []uint32
	var heads []byte
	for i, l := range dec.splits {
		lm := members[l]
		ids := append(append([]uint32(nil), lm.IDs...), joined[l]...)
		codes := make([]byte, 0, len(ids)*mj.M)
		for _, id := range ids {
			codes = append(codes, vecs[id].code...)
		}
		side, cent, err := u.Router.SplitMeans(ctx, codes, mj.M, p.SplitIterations, quakeSplitSeed)
		if err != nil {
			return nil, nil, fmt.Errorf("update: splitting list %d: %w", l, err)
		}
		if len(cent) != 2*vb {
			return nil, nil, fmt.Errorf("update: splitting list %d: %d bytes of means, want 2 of %d", l, len(cent), vb)
		}
		sp := Split{List: l, Partition: lm.p, Children: [2]uint32{children[2*i], children[2*i+1]},
			Heads: [2]uint32{noHead, noHead}}
		for c := 0; c < 2; c++ {
			mj.Lists[sp.Children[c]] = lm.p
			final[sp.Children[c]] = nil
		}
		for k, id := range ids {
			final[sp.Children[side[k]]] = append(final[sp.Children[side[k]]], id)
		}
		labels, heads = append(labels, sp.Children[:]...), append(heads, cent...)
		mj.Splits = append(mj.Splits, sp)
		final[l] = nil
		delete(joined, l)
	}
	dead = append(append(dead, dec.deletes...), dec.splits...)
	if len(labels)+len(dead) > 0 {
		if mj.Patch, err = u.Router.EditGraph(ctx, mj.Graph, labels, heads, dead, nil, nil); err != nil {
			return nil, nil, fmt.Errorf("update: editing the graph: %w", err)
		}
		edited = true
	}

	// 5. Refinement: the lists of the heads nearest each child's, their members to the nearest of
	// those heads, pass after pass.
	var moved []uint32
	var movedVecs []byte
	if len(children) > 0 && p.RefineRadius > 0 {
		near, err := u.Router.NearestLive(ctx, nil, 0, children, p.RefineRadius, nil)
		if err != nil {
			return nil, nil, fmt.Errorf("update: the lists near the children: %w", err)
		}
		inR := map[uint32]bool{}
		var R []uint32
		for _, ns := range near {
			for _, l := range ns {
				if !inR[l] {
					inR[l] = true
					R = append(R, l)
				}
			}
		}
		sort.Slice(R, func(i, j int) bool { return R[i] < R[j] })
		var old []uint32
		for _, l := range R {
			if _, created := mj.Lists[l]; !created {
				old = append(old, l)
			}
		}
		if err := read(ctx, old); err != nil {
			return nil, nil, err
		}
		for _, l := range old {
			if err := note(l); err != nil {
				return nil, nil, err
			}
		}
		var pool []uint32 // every member of R's lists, once, and its list before the refinement
		before := map[uint32]uint32{}
		for _, l := range R {
			var ids []uint32
			if _, created := mj.Lists[l]; created {
				ids = final[l]
			} else {
				ids = append(append(ids, members[l].IDs...), joined[l]...)
				delete(joined, l)
			}
			for _, id := range ids {
				before[id] = l
			}
			pool = append(pool, ids...)
			final[l] = nil
		}
		st.Refined = len(R)
		assign, cent, err := u.refine(ctx, pool, vecs, R, max(p.RefineIterations, 1), in.Dim, mj.M)
		if err != nil {
			return nil, nil, err
		}
		for k, id := range pool {
			l := R[assign[k]]
			final[l] = append(final[l], id)
			if before[id] != l {
				st.Reassigned++
			}
		}
		if p.RefineIterations >= 2 {
			for j, l := range R {
				if len(final[l]) > 0 {
					moved = append(moved, l)
					movedVecs = append(movedVecs, elementBytes(in.DType, cent[j*in.Dim:(j+1)*in.Dim])...)
				}
			}
			st.Moved = len(moved)
		}
	}

	// Receivers of deleted lists' members that no refinement read: they only gain.
	for l, ids := range joined {
		for _, id := range ids {
			v := vecs[id]
			tp := mj.partitionOf(t, u.Manifest, l)
			mj.part(tp).add(l, id, v.loc, v.holder)
			if to := t.Owner(tp); to != t.Owner(mj.partitionOf(t, u.Manifest, v.base)) {
				mj.sendCode(to, id, v.code)
			}
		}
	}

	// 6. Lists left empty go: the ones the round emptied, and those that were.
	var emptied []uint32
	for l, ids := range final {
		if len(ids) == 0 && !isDel[l] && !containsList(dec.splits, l) {
			emptied = append(emptied, l)
		}
	}
	for _, l := range empty {
		if _, changed := final[l]; !changed && !isDel[l] && len(joined[l]) == 0 {
			emptied = append(emptied, l)
		}
	}
	sort.Slice(emptied, func(i, j int) bool { return emptied[i] < emptied[j] })
	var deadAfter []uint32
	for _, l := range emptied {
		if _, created := mj.Lists[l]; created {
			delete(mj.Lists, l) // never was: no epoch names it
			deadAfter = append(deadAfter, l)
			continue
		}
		pe := mj.part(t.PartitionOf(l, u.Manifest))
		pe.Removed = append(pe.Removed, l)
		had := 0
		if lm, ok := members[l]; ok {
			had = len(lm.IDs) // all reassigned by the refinement
		}
		mj.Merges = append(mj.Merges, Merge{List: l, Partition: t.PartitionOf(l, u.Manifest), Postings: had, Into: noList})
		deadAfter = append(deadAfter, l)
	}
	st.Emptied = len(emptied)
	if len(moved)+len(deadAfter) > 0 {
		// The edit amended: one patch from the epoch's graph.
		if mj.Patch, err = u.Router.EditGraph(ctx, mj.Graph, nil, nil, deadAfter, moved, movedVecs); err != nil {
			return nil, nil, fmt.Errorf("update: editing the graph after the refinement: %w", err)
		}
		edited = true
	}

	// The edits: the deleted and split lists go, the children come, and every other list the
	// round changed gains and loses the difference.
	for _, l := range dec.deletes {
		pe := mj.part(members[l].p)
		pe.Removed = append(pe.Removed, l)
	}
	for _, sp := range mj.Splits {
		pe := mj.part(sp.Partition)
		pe.Removed = append(pe.Removed, sp.List)
		for _, ch := range sp.Children {
			if _, created := mj.Lists[ch]; created {
				pe.Created = append(pe.Created, ch)
			}
		}
	}
	for i := range mj.Splits {
		for c := 0; c < 2; c++ {
			mj.Splits[i].Sizes[c] = len(final[mj.Splits[i].Children[c]])
		}
	}
	changed := make([]uint32, 0, len(final))
	for l := range final {
		changed = append(changed, l)
	}
	sort.Slice(changed, func(i, j int) bool { return changed[i] < changed[j] })
	for _, l := range changed {
		if isDel[l] || containsList(dec.splits, l) || containsList(emptied, l) {
			continue
		}
		tp := mj.partitionOf(t, u.Manifest, l)
		pe := mj.part(tp)
		had := map[uint32]bool{}
		if _, created := mj.Lists[l]; !created {
			for _, id := range members[l].IDs {
				had[id] = true
			}
		}
		now := map[uint32]bool{}
		for _, id := range final[l] {
			now[id] = true
			if had[id] {
				continue
			}
			v := vecs[id]
			pe.add(l, id, v.loc, v.holder)
			if to := t.Owner(tp); to != t.Owner(mj.partitionOf(t, u.Manifest, v.base)) {
				mj.sendCode(to, id, v.code)
			}
		}
		if _, created := mj.Lists[l]; !created {
			for _, id := range members[l].IDs {
				if !now[id] {
					pe.drop(l, id)
				}
			}
		}
	}
	return mj, st, nil
}

// noHead is a split child's head vector ID under Quake: a mean, no data vector.
const noHead = math.MaxUint32

func containsList(ls []uint32, l uint32) bool {
	for _, x := range ls {
		if x == l {
			return true
		}
	}
	return false
}

// refine is kmeans_refine_partitions over the members pool (their codes in vecs) and the lists
// R: passes passes, the first to R's heads, every later one to the means of the pass before (a
// list that took no member takes none after). It returns each member's index in R and the
// centroid each list's last pass used (dim floats each; the head's for the first pass).
func (u *Updater) refine(ctx context.Context, pool []uint32, vecs map[uint32]*qvec, R []uint32, passes, dim, m int) ([]int, []float32, error) {
	codes := make([]byte, 0, len(pool)*m)
	for _, id := range pool {
		codes = append(codes, vecs[id].code...)
	}
	assign := make([]int, len(pool))
	out := make([]float32, len(R)*dim)
	if len(pool) == 0 {
		return assign, out, nil
	}
	var cent []float32 // the last pass's centroids, for the active lists
	active := make([]int, len(R))
	for j := range active {
		active[j] = j
	}
	var sums []float64
	var counts []int
	for pass := 0; pass < passes && len(active) > 0; pass++ {
		if pass > 0 {
			// The means of the pass before; a list that took no member drops out.
			var next []int
			var nc []float32
			for a, j := range active {
				if counts[a] == 0 {
					continue
				}
				next = append(next, j)
				for d := 0; d < dim; d++ {
					nc = append(nc, float32(sums[a*dim+d]/float64(counts[a])))
				}
			}
			active, cent = next, nc
			if len(active) == 0 {
				break
			}
		}
		sums, counts = make([]float64, len(active)*dim), make([]int, len(active))
		var heads []uint32
		if pass == 0 {
			heads = R
		}
		for i := 0; i < len(pool); i += codesBatch {
			j := min(len(pool), i+codesBatch)
			near, cnt, sm, err := u.Router.RefineAssign(ctx, codes[i*m:j*m], m, heads, cent, len(active), dim)
			if err != nil {
				return nil, nil, fmt.Errorf("update: refining: %w", err)
			}
			for k, a := range near {
				assign[i+k] = active[a]
			}
			for a := range cnt {
				counts[a] += int(cnt[a])
			}
			for x, v := range sm {
				sums[x] += float64(v)
			}
		}
	}
	// The centroid of every list of R its last pass used, by R's index; the heads' are not needed
	// (with one pass nothing moves).
	if cent != nil {
		for a, j := range active {
			copy(out[j*dim:(j+1)*dim], cent[a*dim:(a+1)*dim])
		}
	}
	return assign, out, nil
}

// nearestLive is NEAREST_LIVE over codes (m bytes each), in frames of codesBatch codes.
func (u *Updater) nearestLive(ctx context.Context, codes []byte, m, k int, excluded []uint32) ([][]uint32, error) {
	var out [][]uint32
	for i := 0; i < len(codes); i += codesBatch * m {
		j := min(len(codes), i+codesBatch*m)
		ls, err := u.Router.NearestLive(ctx, codes[i:j], m, nil, k, excluded)
		if err != nil {
			return nil, err
		}
		out = append(out, ls...)
	}
	return out, nil
}

// listLengths is every list's length at t's versions, from the owners.
func (u *Updater) listLengths(ctx context.Context, t *epoch.Table) (map[uint32]int, error) {
	byOwner := map[placement.NodeID][]int{}
	for p := 0; p < u.Manifest.NumPartitions; p++ {
		o := t.Owner(p)
		byOwner[o] = append(byOwner[o], p)
	}
	owners := make([]placement.NodeID, 0, len(byOwner))
	for o := range byOwner {
		owners = append(owners, o)
	}
	got := make([][]nodeclient.ListLength, len(owners))
	if err := parallel(ctx, len(owners), func(ctx context.Context, i int) error {
		ps := byOwner[owners[i]]
		vs := make([]uint32, len(ps))
		for k, p := range ps {
			vs[k] = t.Version(p)
		}
		ls, err := u.client(t.Nodes[owners[i]].NodeAddr).ListLengths(ctx, ps, vs)
		if err != nil {
			return fmt.Errorf("update: the list lengths of partitions %v on node %d: %w", ps, owners[i], err)
		}
		got[i] = ls
		return nil
	}); err != nil {
		return nil, err
	}
	out := map[uint32]int{}
	for _, ls := range got {
		for _, l := range ls {
			out[l.List] = int(l.Postings)
		}
	}
	return out, nil
}

// readLists reads lists from their owners at t's versions, the owners in parallel; m is the PQ
// code size (0 if every list is empty).
func (u *Updater) readLists(ctx context.Context, t *epoch.Table, lists []uint32) (map[uint32]*listMembers, int, error) {
	byOwner := map[placement.NodeID][]uint32{}
	for _, l := range lists {
		o := t.Owner(t.PartitionOf(l, u.Manifest))
		byOwner[o] = append(byOwner[o], l)
	}
	owners := make([]placement.NodeID, 0, len(byOwner))
	for o := range byOwner {
		owners = append(owners, o)
	}
	out := map[uint32]*listMembers{}
	var mu sync.Mutex
	m := 0
	err := parallel(ctx, len(owners), func(ctx context.Context, i int) error {
		for _, l := range byOwner[owners[i]] {
			lm, cm, err := u.readList(ctx, t, l)
			if err != nil {
				return err
			}
			mu.Lock()
			out[l] = &lm
			if cm > 0 {
				if m > 0 && cm != m {
					mu.Unlock()
					return fmt.Errorf("update: list %d has %d-byte PQ codes, others %d", l, cm, m)
				}
				m = cm
			}
			mu.Unlock()
		}
		return nil
	})
	return out, m, err
}

// parallelN runs fn for 0..n-1 on at most w goroutines.
func parallelN(ctx context.Context, n, w int, fn func(ctx context.Context, i int) error) error {
	next := make(chan int)
	go func() {
		defer close(next)
		for i := 0; i < n; i++ {
			select {
			case next <- i:
			case <-ctx.Done():
				return
			}
		}
	}()
	return parallel(ctx, min(w, n), func(ctx context.Context, _ int) error {
		for i := range next {
			if err := fn(ctx, i); err != nil {
				return err
			}
		}
		return nil
	})
}

// dtypeSize is the bytes of one element of the index's type (nodeclient.Info.DType).
func dtypeSize(dt string) int {
	switch dt {
	case "uint8", "int8":
		return 1
	default:
		return 4
	}
}

// elementBytes is v in the index's element type: a mean as the build writes a head
// (fusion_build's ElementOf: rounded to even and clamped for an integer type).
func elementBytes(dt string, v []float32) []byte {
	out := make([]byte, 0, len(v)*dtypeSize(dt))
	for _, x := range v {
		switch dt {
		case "uint8":
			out = append(out, uint8(math.Min(255, math.Max(0, math.RoundToEven(float64(x))))))
		case "int8":
			out = append(out, uint8(int8(math.Min(127, math.Max(-128, math.RoundToEven(float64(x)))))))
		default:
			out = binary.LittleEndian.AppendUint32(out, math.Float32bits(x))
		}
	}
	return out
}
