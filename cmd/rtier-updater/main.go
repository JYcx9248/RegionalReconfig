// rtier-updater: the update controller (U14). It applies one batch of dataset updates to a
// running cluster -- inserts from a vector file, deletes from a list of vector IDs: it takes the
// controller's lease, routes the inserts by the build's rule on an entry's navigation graph,
// sends their PQ codes and raw vectors to their owners, has the owners stage the next version of
// the lists the batch changes, and asks the controller to flip.
//
//	rtier-updater -api ADDR -partitions DIR [-state FILE] [-inserts V.fbin] [-deletes IDS] [-router ADDR]
//
// Inserted vectors get the IDs and raw-vector locations that follow the ones handed out before,
// in file order (the reply names them). FILE keeps where the next batch starts; without it, the
// first batch starts at the index's vector and location counts. IDs and locations are never
// handed out twice, not even by a batch that fails, so FILE is written once they are taken.
// -deletes is a text file of vector IDs (one per line, # comments) or a .bin file in the engine's
// u32 array format. The vector file must be in the index's element type.
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
	NextID  uint64 `json:"next_id"`
	NextLoc uint64 `json:"next_loc"`
}

func main() {
	api := flag.String("api", "127.0.0.1:7101", "controller API address")
	parts := flag.String("partitions", "", "partition directory (its manifest)")
	statePath := flag.String("state", "update-state.json", "where the next batch's IDs and locations start")
	inserts := flag.String("inserts", "", "vectors to insert (.u8bin / .i8bin / .fbin)")
	deletes := flag.String("deletes", "", "vector IDs to delete (text, or .bin u32 array)")
	routerAddr := flag.String("router", "", "data node that routes the inserts (default: the first entry's)")
	timeout := flag.Duration("timeout", 30*time.Minute, "")
	flag.Parse()
	if *parts == "" || (*inserts == "" && *deletes == "") {
		flag.Usage()
		os.Exit(2)
	}
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	m, err := partitioning.LoadManifest(*parts)
	if err != nil {
		log.Fatal(err)
	}
	conn, err := ctrl.Dial(ctx, *api)
	if err != nil {
		log.Fatalf("controller API %s: %v", *api, err)
	}
	defer conn.Close()
	conn.Start()

	// The spaces: the smallest any data node has, since partitions move.
	var st protocol.ClusterStatus
	if err := conn.Call(ctx, protocol.StatusA, nil, &st); err != nil {
		log.Fatal(err)
	}
	if st.Table == nil {
		log.Fatal("the cluster has no epoch yet")
	}
	var info *nodeclient.Info
	idEnd, locEnd := uint64(1<<32-1), uint64(1<<32-1)
	for _, id := range st.Table.Placement.Nodes() {
		nc := nodeclient.New(st.Table.Nodes[id].NodeAddr, 1)
		in, err := nc.Info(ctx)
		nc.Close()
		if err != nil {
			log.Fatalf("node %d: %v", id, err)
		}
		info, idEnd, locEnd = in, min(idEnd, in.IDCapacity), min(locEnd, in.Raw.Locations)
	}
	s := state{NextID: info.NumVectors, NextLoc: m.Locations()}
	if b, err := os.ReadFile(*statePath); err == nil {
		if err := json.Unmarshal(b, &s); err != nil {
			log.Fatalf("%s: %v", *statePath, err)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		log.Fatal(err)
	}
	alloc, err := update.NewAllocator(s.NextID, idEnd, s.NextLoc, locEnd)
	if err != nil {
		log.Fatal(err)
	}

	var vecs []byte
	vecBytes, n := 0, 0
	if *inserts != "" {
		v, err := vecio.ReadBin(*inserts, 0)
		if err != nil {
			log.Fatal(err)
		}
		if string(v.DType) != info.DType || v.Dim != info.Dim {
			log.Fatalf("%s holds %d-d %s vectors, the index %d-d %s", *inserts, v.Dim, v.DType, info.Dim, info.DType)
		}
		vecs, vecBytes, n = v.Data, v.Dim*v.DType.Size(), v.N
	}
	var del []uint32
	if *deletes != "" {
		if del, err = readIDs(*deletes); err != nil {
			log.Fatal(err)
		}
	}

	u := &update.Updater{Controller: update.Remote{Conn: conn}, Manifest: m, Alloc: alloc}
	defer u.Close()
	if *routerAddr != "" {
		u.Router = nodeclient.New(*routerAddr, 2)
		defer u.Router.Close()
	}
	start := time.Now()
	b, err := u.Prepare(ctx, vecs, vecBytes, del)
	saveState(*statePath, alloc) // whatever happened: what was taken is never handed out again
	if err != nil {
		log.Fatal(err)
	}
	decided := time.Since(start)
	reply, err := u.Execute(ctx, b)
	if reply != nil {
		out, _ := json.MarshalIndent(reply, "", "  ")
		fmt.Println(string(out))
	}
	if err != nil {
		log.Fatal(err)
	}
	if n > 0 {
		fmt.Printf("inserted %d vectors: IDs [%d, %d), locations [%d, %d)\n", n, b.Inserts[0].ID,
			uint64(b.Inserts[n-1].ID)+1, b.Inserts[0].Loc, uint64(b.Inserts[n-1].Loc)+1)
	}
	lists, _, _ := b.Postings()
	fmt.Printf("%d postings added, %d IDs deleted; epoch %d -> %d, %d partitions at new versions; "+
		"decided in %.2f s, applied in %.2f s\n", len(lists), len(b.Deleted), reply.FromEpoch, reply.ToEpoch,
		len(reply.Changed), decided.Seconds(), time.Since(start).Seconds()-decided.Seconds())
}

func saveState(path string, a *update.Allocator) {
	id, loc := a.Next()
	b, _ := json.Marshal(state{NextID: id, NextLoc: loc})
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(b, '\n'), 0o644); err != nil {
		log.Fatal(err)
	}
	if err := os.Rename(tmp, path); err != nil {
		log.Fatal(err)
	}
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
