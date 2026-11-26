// rtier-updater: the update controller (U14). It applies one batch of dataset updates to a
// running cluster -- inserts from a vector file, deletes from a list of vector IDs: it takes the
// controller's lease, routes the inserts by the build's rule on its own copy of the navigation
// graph, sends their PQ codes and raw vectors to their owners, has the owners stage the next
// version of the lists the batch changes, and asks the controller to flip and reclaim.
//
//	rtier-updater -api ADDR -partitions DIR [-state FILE] [-inserts V.fbin] [-deletes IDS]
//	              [-router ADDR [-graph PATH] | -index DIR [-node-bin PATH]] [-patches DIR]
//	              [-split L,L,... -merge L,L,... | -split-len N -merge-len N [-major-max K]]
//	              [-policy threshold|spfresh [-reassign-range R] | -policy quake [-quake-* ...]]
//	              [-record FILE]
//
// Inserts are routed by the update controller's own rtier_node, which holds the index's
// navigation graph and codebook and nothing else -- never by a serving node, whose navigation
// CPU the queries need. -router names one that keeps running between batches (started as
// rtier_node --index DIR --partitions DIR --graph index --pq-capacity 1); -index starts one for
// this batch, which loads the graph first (tens of seconds for a large one).
//
// Inserted vectors get the IDs and raw-vector locations that follow the ones handed out before,
// in file order (the reply names them). FILE keeps where the next batch starts; without it, the
// first batch starts at the index's vector and location counts. IDs and locations are never
// handed out twice, not even by a batch that fails, so FILE is written once they are taken.
// -deletes is a text file of vector IDs (one per line, # comments) or a .bin file in the engine's
// u32 array format. The vector file must be in the index's element type. The controller must run
// with AllowUpdates.
//
// Major epochs split and merge posting lists, and patch the navigation graph to match: -split and
// -merge name lists for one, which runs after the batch, if any; with -split-len and -merge-len,
// one runs after the batch when the owners report a list longer or shorter than that (at most
// -major-max lists). The router's graph is the authoritative one: the major epoch edits it, and
// DIR (-patches, by default next to FILE) keeps the patch of every committed graph version, with
// which a router started later is brought to the cluster's version. -graph names the build's
// graph as a -router reads it, to reload it when an edit did not commit (-index: its graph.hnsw).
// -policy spfresh adds SPFresh's reassign to the major epoch (internal/update/spfresh.go): the
// members of each split's children and of the -reassign-range lists nearest its old head are
// checked, a vector once (one in a child only there), and a vector that no longer belongs to the
// list it was queued in is inserted again by the build's rule; a merged list's postings go into
// every list the rule gives them.
//
// -policy quake runs a round of Quake's maintenance after the batch, or alone without one
// (internal/update/quake.go), on Quake's index (fusion_build --clustering kmeans): it reads the
// entries' last -quake-window client queries, decides on its cost model which lists to delete and
// which to split, deletes, splits and refines them, and removes the lists left empty, as one
// major epoch. The -quake-* flags are Quake's maintenance parameters, with its defaults; with
// -quake-max-size the lists' lengths decide instead (Quake's size-bound mode).
//
// -record FILE writes, as JSON, what the batch and the major epoch did: their replies (epochs,
// partitions, and the phases from the staging to the reclaim), the inserted IDs and locations,
// and the time to decide and to apply each -- also when one failed, with the error. An
// experiment's runner keeps it (scripts/run_local.py, updates.json).
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"rtier/internal/ctrl"
	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
	"rtier/internal/protocol"
	"rtier/internal/update"
	"rtier/internal/vecio"
)

// state is where the next batch's IDs and locations start.
type state struct {
	NextID   uint64 `json:"next_id"`
	NextLoc  uint64 `json:"next_loc"`
	NextList uint64 `json:"next_list,omitempty"` // the first list ID no major epoch took
	ListGen  uint32 `json:"list_gen,omitempty"`  // in this index generation (U14 full rebuild)
}

type options struct {
	api, parts, statePath, inserts, deletes string
	router, index, nodeBin, graph, patches  string
	split, merge, record, policy            string
	splitLen, mergeLen, majorMax, reassign  int
	quake                                   update.QuakeParams
	timeout                                 time.Duration
}

// record is what -record writes.
type record struct {
	Batch         *protocol.UpdateReply `json:"batch,omitempty"`
	Inserted      int                   `json:"inserted"`
	IDs           []uint64              `json:"ids,omitempty"`       // [first, end) of the inserted vectors
	Locations     []uint64              `json:"locations,omitempty"` // [first, end) of their raw vectors
	Deleted       int                   `json:"deleted"`
	Postings      int                   `json:"postings_added"`
	DecideSeconds float64               `json:"decide_seconds,omitempty"` // the lease, the routing
	ApplySeconds  float64               `json:"apply_seconds,omitempty"`  // the staging to the reclaim
	Major         *protocol.UpdateReply `json:"major,omitempty"`
	Splits        []uint32              `json:"splits,omitempty"`
	Merges        []uint32              `json:"merges,omitempty"`
	MajorMoved    int                   `json:"major_moved_postings,omitempty"` // to another owner
	Reassign      *update.ReassignStats `json:"reassign,omitempty"`             // -policy spfresh
	Quake         *update.QuakeStats    `json:"quake,omitempty"`                // -policy quake
	PatchBytes    int                   `json:"major_patch_bytes,omitempty"`
	MajorDecide   float64               `json:"major_decide_seconds,omitempty"`
	MajorApply    float64               `json:"major_apply_seconds,omitempty"`
	// When the batch and the major epoch flipped, in seconds since this process started -- when
	// the runner triggered it (updates.json's trigger_seconds): the timelines draw them there.
	FlipSeconds      float64 `json:"flip_seconds,omitempty"`
	MajorFlipSeconds float64 `json:"major_flip_seconds,omitempty"`
	Error            string  `json:"error,omitempty"`
}

// started is when the process started, which the record's flip times count from.
var started = time.Now()

func main() {
	var o options
	flag.StringVar(&o.api, "api", "127.0.0.1:7101", "controller API address")
	flag.StringVar(&o.parts, "partitions", "", "partition directory (its manifest)")
	flag.StringVar(&o.statePath, "state", "update-state.json", "where the next batch's IDs and locations start")
	flag.StringVar(&o.inserts, "inserts", "", "vectors to insert (.u8bin / .i8bin / .fbin)")
	flag.StringVar(&o.deletes, "deletes", "", "vector IDs to delete (text, or .bin u32 array)")
	flag.StringVar(&o.router, "router", "", "the update controller's running router (an rtier_node with the graph)")
	flag.StringVar(&o.index, "index", "", "index directory: start a router on its graph for this batch")
	flag.StringVar(&o.nodeBin, "node-bin", "", "rtier_node to start with -index (default: engine/build next to bin/)")
	flag.StringVar(&o.graph, "graph", "", "the build's graph as the -router reads it, reloaded when an edit did not commit")
	flag.StringVar(&o.patches, "patches", "", "where the patches of committed graph versions are kept (default: next to -state)")
	flag.StringVar(&o.split, "split", "", "lists to split in a major epoch (comma-separated)")
	flag.StringVar(&o.merge, "merge", "", "lists to merge in a major epoch (comma-separated)")
	flag.IntVar(&o.splitLen, "split-len", 0, "after the batch, split the lists longer than this (0: never)")
	flag.IntVar(&o.mergeLen, "merge-len", 0, "after the batch, merge the lists shorter than this (0: never)")
	flag.IntVar(&o.majorMax, "major-max", 0, "at most this many lists per major epoch (0: no limit)")
	flag.StringVar(&o.policy, "policy", update.PolicyThreshold,
		"a major epoch's policy: threshold, spfresh (its reassign), or quake (a round of its maintenance)")
	flag.IntVar(&o.reassign, "reassign-range", 64, "-policy spfresh: lists nearest a split list's old head whose members are checked")
	o.quake = update.DefaultQuakeParams()
	flag.IntVar(&o.quake.Window, "quake-window", o.quake.Window, "-policy quake: the client queries whose hits decide (window_size)")
	flag.IntVar(&o.quake.RefineRadius, "quake-radius", o.quake.RefineRadius, "-policy quake: lists refined around each split's children (refinement_radius)")
	flag.IntVar(&o.quake.RefineIterations, "quake-iterations", o.quake.RefineIterations, "-policy quake: refinement passes (refinement_iterations)")
	flag.IntVar(&o.quake.MinSize, "quake-min-size", o.quake.MinSize, "-policy quake: min_partition_size")
	flag.IntVar(&o.quake.MaxSize, "quake-max-size", 0, "-policy quake: split lists longer than this and delete those shorter than -quake-min-size, by length alone (max_partition_size; 0: the cost model decides)")
	flag.Float64Var(&o.quake.Alpha, "quake-alpha", o.quake.Alpha, "-policy quake: alpha")
	flag.BoolVar(&o.quake.DeleteRejection, "quake-delete-rejection", o.quake.DeleteRejection, "-policy quake: enable_delete_rejection")
	flag.IntVar(&o.quake.SplitIterations, "quake-split-iterations", o.quake.SplitIterations, "-policy quake: the 2-means' iterations (split_knn_iterations)")
	flag.Float64Var(&o.quake.DeleteThresholdNS, "quake-delete-threshold-ns", o.quake.DeleteThresholdNS, "-policy quake: delete_threshold_ns")
	flag.Float64Var(&o.quake.SplitThresholdNS, "quake-split-threshold-ns", o.quake.SplitThresholdNS, "-policy quake: split_threshold_ns")
	flag.StringVar(&o.record, "record", "", "write what the batch and the major epoch did here, as JSON")
	flag.DurationVar(&o.timeout, "timeout", 30*time.Minute, "")
	flag.Parse()
	quake := o.policy == update.PolicyQuake
	if o.parts == "" || (o.inserts == "" && o.deletes == "" && o.split == "" && o.merge == "" && !quake) {
		flag.Usage()
		os.Exit(2)
	}
	if (o.policy != update.PolicyThreshold && o.policy != update.PolicySPFresh && !quake) || o.reassign < 0 {
		log.Fatalf("-policy %q -reassign-range %d: want threshold, spfresh or quake, and a range of 0 or more", o.policy, o.reassign)
	}
	if quake && (o.split != "" || o.merge != "" || o.splitLen > 0 || o.mergeLen > 0) {
		log.Fatal("-policy quake decides its own lists: no -split, -merge, -split-len or -merge-len")
	}
	var rec record
	err := run(o, &rec)
	if o.record != "" {
		if err != nil {
			rec.Error = err.Error()
		}
		if werr := writeRecord(o.record, &rec); werr != nil {
			err = errors.Join(err, werr)
		}
	}
	if err != nil {
		log.Fatal(err)
	}
}

// writeRecord writes rec to path whole or not at all.
func writeRecord(path string, rec *record) error {
	b, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(b, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func run(o options, rec *record) error {
	ctx, cancel := context.WithTimeout(context.Background(), o.timeout)
	defer cancel()
	m, err := partitioning.LoadManifest(o.parts)
	if err != nil {
		return err
	}
	var vecs []byte
	vecBytes, n := 0, 0
	var v *vecio.Vectors
	if o.inserts != "" {
		if o.router == "" && o.index == "" {
			return errors.New("inserts need a router: -router ADDR, or -index DIR to start one")
		}
		if v, err = vecio.ReadBin(o.inserts, 0); err != nil {
			return err
		}
		vecs, vecBytes, n = v.Data, v.Dim*v.DType.Size(), v.N
	}
	var del []uint32
	if o.deletes != "" {
		if del, err = readIDs(o.deletes); err != nil {
			return err
		}
	}
	conn, err := ctrl.Dial(ctx, o.api)
	if err != nil {
		return fmt.Errorf("controller API %s: %w", o.api, err)
	}
	defer conn.Close()
	conn.Start()

	// The spaces: the smallest any data node has, since partitions move.
	var st protocol.ClusterStatus
	if err := conn.Call(ctx, protocol.StatusA, nil, &st); err != nil {
		return err
	}
	if st.Table == nil {
		return errors.New("the cluster has no epoch yet")
	}
	var info *nodeclient.Info
	idEnd, locEnd := uint64(1<<32-1), uint64(1<<32-1)
	for _, id := range st.Table.Placement.Nodes() {
		nc := nodeclient.New(st.Table.Nodes[id].NodeAddr, 1)
		in, err := nc.Info(ctx)
		nc.Close()
		if err != nil {
			return fmt.Errorf("node %d: %w", id, err)
		}
		info, idEnd, locEnd = in, min(idEnd, in.IDCapacity), min(locEnd, in.Raw.Locations)
	}
	if v != nil && (string(v.DType) != info.DType || v.Dim != info.Dim) {
		return fmt.Errorf("%s holds %d-d %s vectors, the index %d-d %s", o.inserts, v.Dim, v.DType, info.Dim, info.DType)
	}
	s := state{NextID: info.NumVectors, NextLoc: m.Locations()}
	if b, err := os.ReadFile(o.statePath); err == nil {
		if err := json.Unmarshal(b, &s); err != nil {
			return fmt.Errorf("%s: %w", o.statePath, err)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	alloc, err := update.NewAllocator(s.NextID, idEnd, s.NextLoc, locEnd)
	if err != nil {
		return err
	}
	alloc.SetNextList(s.NextList, s.ListGen)
	splits, err := parseLists(o.split)
	if err != nil {
		return err
	}
	merges, err := parseLists(o.merge)
	if err != nil {
		return err
	}
	quake := o.policy == update.PolicyQuake
	major := len(splits)+len(merges) > 0 || o.splitLen > 0 || o.mergeLen > 0 || quake

	u := &update.Updater{Controller: update.Remote{Conn: conn}, Manifest: m, Alloc: alloc,
		GraphPath: o.graph, PatchDir: o.patches, Policy: o.policy, ReassignRange: o.reassign, Quake: o.quake}
	if u.PatchDir == "" {
		u.PatchDir = filepath.Join(filepath.Dir(o.statePath), "graph-patches")
	}
	if u.GraphPath == "" && o.index != "" {
		u.GraphPath = filepath.Join(o.index, "graph.hnsw")
	}
	defer u.Close()
	if n > 0 || major {
		if o.router == "" && o.index == "" {
			return errors.New("inserts and major epochs need a router: -router ADDR, or -index DIR to start one")
		}
		addr := o.router
		if addr == "" {
			bin := o.nodeBin
			if bin == "" {
				exe, err := os.Executable()
				if err != nil {
					return err
				}
				bin = filepath.Join(filepath.Dir(exe), "..", "engine", "build", "rtier_node")
			}
			var stop func() error
			if addr, stop, err = update.StartRouter(bin, o.index, o.parts, os.Stderr); err != nil {
				return err
			}
			defer stop()
			log.Printf("router: %s on %s at %s", bin, o.index, addr)
		}
		// A connection for each ASSIGN that routing keeps in flight.
		u.Router = nodeclient.New(addr, max(2, update.RouteStreams()))
		defer u.Router.Close()
	}
	if n > 0 || len(del) > 0 {
		start := time.Now()
		b, err := u.Prepare(ctx, vecs, vecBytes, del)
		// Whatever happened: what was taken is never handed out again.
		if serr := saveState(o.statePath, alloc); serr != nil {
			return errors.Join(err, serr)
		}
		if err != nil {
			return err
		}
		decided := time.Since(start)
		lists, _, _ := b.Postings()
		rec.Inserted, rec.Deleted, rec.Postings = n, len(b.Deleted), len(lists)
		rec.DecideSeconds = decided.Seconds()
		if n > 0 {
			rec.IDs = []uint64{uint64(b.Inserts[0].ID), uint64(b.Inserts[n-1].ID) + 1}
			rec.Locations = []uint64{uint64(b.Inserts[0].Loc), uint64(b.Inserts[n-1].Loc) + 1}
		}
		executing := time.Since(started).Seconds()
		reply, err := u.Execute(ctx, b)
		rec.Batch, rec.ApplySeconds = reply, time.Since(start).Seconds()-decided.Seconds()
		if reply != nil && reply.FlipSeconds > 0 {
			rec.FlipSeconds = executing + reply.FlipSeconds
		}
		if reply != nil {
			out, _ := json.MarshalIndent(reply, "", "  ")
			fmt.Println(string(out))
		}
		if err != nil {
			return err
		}
		if n > 0 {
			fmt.Printf("inserted %d vectors: IDs [%d, %d), locations [%d, %d)\n", n, b.Inserts[0].ID,
				uint64(b.Inserts[n-1].ID)+1, b.Inserts[0].Loc, uint64(b.Inserts[n-1].Loc)+1)
		}
		fmt.Printf("%d postings added, %d IDs deleted; epoch %d -> %d, %d partitions at new versions; "+
			"decided in %.2f s, applied in %.2f s\n", len(lists), len(b.Deleted), reply.FromEpoch, reply.ToEpoch,
			len(reply.Changed), decided.Seconds(), time.Since(start).Seconds()-decided.Seconds())
		if len(splits)+len(merges) == 0 && (o.splitLen > 0 || o.mergeLen > 0) {
			splits, merges = update.MajorDue(b.Lengths, uint32(max(o.splitLen, 0)), uint32(max(o.mergeLen, 0)), o.majorMax)
		}
	}
	if quake {
		return runQuake(ctx, u, o.statePath, alloc, rec)
	}
	if len(splits)+len(merges) == 0 {
		return nil
	}
	return runMajor(ctx, u, o.statePath, alloc, splits, merges, rec)
}

// runQuake runs one round of Quake's maintenance as a major epoch, if it finds anything to do.
func runQuake(ctx context.Context, u *update.Updater, statePath string, alloc *update.Allocator, rec *record) error {
	start := time.Now()
	mj, st, err := u.PrepareQuake(ctx)
	if serr := saveState(statePath, alloc); serr != nil {
		return errors.Join(err, serr)
	}
	rec.Quake = st
	if err != nil {
		return err
	}
	if mj == nil {
		fmt.Printf("Quake's maintenance: no round (%s)\n", st.Skipped)
		return nil
	}
	for _, sp := range mj.Splits {
		rec.Splits = append(rec.Splits, sp.List)
	}
	for _, mg := range mj.Merges {
		rec.Merges = append(rec.Merges, mg.List)
	}
	return execMajor(ctx, u, mj, start, rec)
}

// runMajor runs one major epoch that splits and merges those lists.
func runMajor(ctx context.Context, u *update.Updater, statePath string, alloc *update.Allocator, splits, merges []uint32,
	rec *record) error {
	start := time.Now()
	rec.Splits, rec.Merges = splits, merges
	mj, err := u.PrepareMajor(ctx, splits, merges)
	if serr := saveState(statePath, alloc); serr != nil {
		return errors.Join(err, serr)
	}
	if err != nil {
		return err
	}
	return execMajor(ctx, u, mj, start, rec)
}

// execMajor applies a major epoch decided since start.
func execMajor(ctx context.Context, u *update.Updater, mj *update.Major, start time.Time, rec *record) error {
	decided := time.Since(start)
	rec.MajorDecide, rec.PatchBytes, rec.Reassign = decided.Seconds(), len(mj.Patch), mj.Reassign
	for _, cs := range mj.Codes {
		rec.MajorMoved += len(cs.IDs)
	}
	executing := time.Since(started).Seconds()
	reply, err := u.ExecuteMajor(ctx, mj)
	rec.Major, rec.MajorApply = reply, time.Since(start).Seconds()-decided.Seconds()
	if reply != nil && reply.FlipSeconds > 0 {
		rec.MajorFlipSeconds = executing + reply.FlipSeconds
	}
	if reply != nil {
		out, _ := json.MarshalIndent(reply, "", "  ")
		fmt.Println(string(out))
	}
	if err != nil {
		return err
	}
	for _, sp := range mj.Splits {
		if mj.Quake != nil { // the heads are means
			fmt.Printf("list %d (partition %d) split into %d and %d: %d and %d postings\n",
				sp.List, sp.Partition, sp.Children[0], sp.Children[1], sp.Sizes[0], sp.Sizes[1])
			continue
		}
		fmt.Printf("list %d (partition %d) split into %d and %d: %d and %d postings, heads vectors %d and %d\n",
			sp.List, sp.Partition, sp.Children[0], sp.Children[1], sp.Sizes[0], sp.Sizes[1], sp.Heads[0], sp.Heads[1])
	}
	if q := mj.Quake; q != nil {
		fmt.Printf("Quake's maintenance: %d lists, %d queries, scan fraction %.4f; %d deleted (%d rejected), %d split, "+
			"%d refined (%d vectors reassigned, %d heads moved), %d removed empty\n", q.Lists, q.Window, q.ScanFraction,
			q.Deletes, q.Rejected, q.Splits, q.Refined, q.Reassigned, q.Moved, q.Emptied)
	}
	for _, mg := range mj.Merges {
		fmt.Printf("list %d (partition %d) merged: %d postings to their nearest heads' lists\n", mg.List, mg.Partition, mg.Postings)
	}
	moved := 0
	for _, cs := range mj.Codes {
		moved += len(cs.IDs)
	}
	fmt.Printf("major epoch: epoch %d -> %d, graph version %d (%d-byte patch), %d postings moved to another owner; "+
		"decided in %.2f s, applied in %.2f s\n", reply.FromEpoch, reply.ToEpoch, reply.Graph, len(mj.Patch), moved,
		decided.Seconds(), time.Since(start).Seconds()-decided.Seconds())
	return nil
}

// parseLists reads comma-separated list IDs.
func parseLists(s string) ([]uint32, error) {
	var out []uint32
	for _, f := range strings.Split(s, ",") {
		if f = strings.TrimSpace(f); f == "" {
			continue
		}
		v, err := strconv.ParseUint(f, 10, 32)
		if err != nil {
			return nil, fmt.Errorf("list %q: %v", f, err)
		}
		out = append(out, uint32(v))
	}
	return out, nil
}

func saveState(path string, a *update.Allocator) error {
	id, loc := a.Next()
	next, gen := a.NextList()
	b, _ := json.Marshal(state{NextID: id, NextLoc: loc, NextList: next, ListGen: gen})
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(b, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// readIDs reads vector IDs: a .bin file in the engine's u32 array format, or text.
func readIDs(path string) ([]uint32, error) {
	if strings.HasSuffix(path, ".bin") {
		return partitioning.ReadU32Array(path)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []uint32
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		v, err := strconv.ParseUint(line, 10, 32)
		if err != nil {
			return nil, fmt.Errorf("%s: %v", path, err)
		}
		out = append(out, uint32(v))
	}
	return out, sc.Err()
}
