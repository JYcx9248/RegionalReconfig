// rtier-updater: the update controller (U14). It applies one batch of dataset updates to a
// running cluster -- inserts from a vector file, deletes from a list of vector IDs: it takes the
// controller's lease, routes the inserts by the build's rule on its own copy of the navigation
// graph, sends their PQ codes and raw vectors to their owners, has the owners stage the next
// version of the lists the batch changes, and asks the controller to flip and reclaim.
//
//	rtier-updater -api ADDR -partitions DIR [-state FILE] [-inserts V.fbin] [-deletes IDS]
//	              [-router ADDR [-graph PATH] | -index DIR [-node-bin PATH]] [-patches DIR]
//	              [-split L,L,... -merge L,L,... | -split-len N -merge-len N [-major-max K]]
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
	split, merge                            string
	splitLen, mergeLen, majorMax            int
	timeout                                 time.Duration
}

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
	flag.DurationVar(&o.timeout, "timeout", 30*time.Minute, "")
	flag.Parse()
	if o.parts == "" || (o.inserts == "" && o.deletes == "" && o.split == "" && o.merge == "") {
		flag.Usage()
		os.Exit(2)
	}
	if err := run(o); err != nil {
		log.Fatal(err)
	}
}

func run(o options) error {
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
	major := len(splits)+len(merges) > 0 || o.splitLen > 0 || o.mergeLen > 0

	u := &update.Updater{Controller: update.Remote{Conn: conn}, Manifest: m, Alloc: alloc,
		GraphPath: o.graph, PatchDir: o.patches}
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
		u.Router = nodeclient.New(addr, 2)
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
		reply, err := u.Execute(ctx, b)
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
		lists, _, _ := b.Postings()
		fmt.Printf("%d postings added, %d IDs deleted; epoch %d -> %d, %d partitions at new versions; "+
			"decided in %.2f s, applied in %.2f s\n", len(lists), len(b.Deleted), reply.FromEpoch, reply.ToEpoch,
			len(reply.Changed), decided.Seconds(), time.Since(start).Seconds()-decided.Seconds())
		if len(splits)+len(merges) == 0 && (o.splitLen > 0 || o.mergeLen > 0) {
			splits, merges = update.MajorDue(b.Lengths, uint32(max(o.splitLen, 0)), uint32(max(o.mergeLen, 0)), o.majorMax)
		}
	}
	if len(splits)+len(merges) == 0 {
		return nil
	}
	return runMajor(ctx, u, o.statePath, alloc, splits, merges)
}

// runMajor runs one major epoch that splits and merges those lists.
func runMajor(ctx context.Context, u *update.Updater, statePath string, alloc *update.Allocator, splits, merges []uint32) error {
	start := time.Now()
	mj, err := u.PrepareMajor(ctx, splits, merges)
	if serr := saveState(statePath, alloc); serr != nil {
		return errors.Join(err, serr)
	}
	if err != nil {
		return err
	}
	decided := time.Since(start)
	reply, err := u.ExecuteMajor(ctx, mj)
	if reply != nil {
		out, _ := json.MarshalIndent(reply, "", "  ")
		fmt.Println(string(out))
	}
	if err != nil {
		return err
	}
	for _, sp := range mj.Splits {
		fmt.Printf("list %d (partition %d) split into %d and %d: %d and %d postings, heads vectors %d and %d\n",
			sp.List, sp.Partition, sp.Children[0], sp.Children[1], sp.Sizes[0], sp.Sizes[1], sp.Heads[0], sp.Heads[1])
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
