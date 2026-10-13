package e2e

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"testing"
	"time"

	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
	"rtier/internal/update"
	"rtier/internal/vecio"
)

// TestInsertRouting checks the parts of an insert (U14) that exist so far, over the wire. A
// node started with room for inserts routes vectors by the build's rule: the index's own
// vectors get back the lists and PQ codes the build gave them. It takes PQ codes for new IDs.
// It serves inserted vectors at new locations from memory at once, and writes them to its SSD
// only while update writes are not paused.
func TestInsertRouting(t *testing.T) {
	f := getFixture(t)
	m, err := partitioning.LoadManifest(f.parts)
	if err != nil {
		t.Fatal(err)
	}
	const extraIDs, extraPages = 200, 8
	addr, stop, err := startNode(f, "--load", "all", "--graph", "index",
		"--max-vectors", strconv.Itoa(numVectors+extraIDs), "--max-pages", strconv.FormatUint(m.NumPages+extraPages, 10))
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	ctx := context.Background()
	nc := nodeclient.New(addr, 1)
	defer nc.Close()
	in, err := nc.Info(ctx)
	if err != nil {
		t.Fatal(err)
	}
	locEnd := (m.NumPages + extraPages) * m.VectorsPerPage
	if in.IDCapacity != numVectors+extraIDs || in.Raw.Locations != locEnd {
		t.Fatalf("spaces: %d IDs, %d locations; want %d, %d", in.IDCapacity, in.Raw.Locations, numVectors+extraIDs, locEnd)
	}

	// Route the index's first vectors as if they were inserted.
	const k = 200
	base, err := vecio.ReadBin(filepath.Join(f.dir, "base.fbin"), k)
	if err != nil {
		t.Fatal(err)
	}
	vb := len(base.Row(0))
	vecs := make([]byte, 0, k*vb)
	for i := 0; i < k; i++ {
		vecs = append(vecs, base.Row(i)...)
	}
	alloc, err := update.NewAllocator(numVectors, in.IDCapacity, m.Locations(), locEnd)
	if err != nil {
		t.Fatal(err)
	}
	ins, err := update.Route(ctx, nc, alloc, vecs, vb)
	if err != nil {
		t.Fatal(err)
	}
	built := builtLists(t, filepath.Join(f.index, "postings.bin"), k)
	codes := builtCodes(t, filepath.Join(f.index, "pq_codes.bin"), k)
	same := 0
	for i, x := range ins {
		got := append([]uint32(nil), x.Lists...)
		sort.Slice(got, func(a, b int) bool { return got[a] < got[b] })
		if reflect.DeepEqual(got, built[i]) && bytes.Equal(x.Code, codes[i]) {
			same++
		}
		if x.ID != uint32(numVectors+i) || x.Loc != uint32(m.Locations())+uint32(i) {
			t.Fatalf("insert %d got ID %d, location %d", i, x.ID, x.Loc)
		}
	}
	if same != k {
		t.Fatalf("%d of %d vectors routed to the build's lists with the build's PQ code", same, k)
	}

	// PQ codes for the new IDs, then the vectors at the new locations.
	ids := make([]uint32, k)
	locs := make([]uint32, k)
	var all []byte
	for i, x := range ins {
		ids[i], locs[i] = x.ID, x.Loc
		all = append(all, x.Code...)
	}
	if n, _, err := nc.PQPut(ctx, ids, len(ins[0].Code), all); err != nil || n != k {
		t.Fatalf("PQ_PUT of the new IDs: %d installed, %v", n, err)
	}
	if err := nc.PauseUpdateWrites(ctx, true); err != nil {
		t.Fatal(err)
	}
	if n, _, err := nc.RawAdd(ctx, locs, vb, vecs); err != nil || n != k {
		t.Fatalf("RAW_ADD: %d installed, %v", n, err)
	}
	read := func() []byte {
		_, got, err := nc.RawGet(ctx, locs, make([]uint32, k))
		if err != nil {
			t.Fatal(err)
		}
		return got
	}
	if !bytes.Equal(read(), vecs) {
		t.Fatal("inserted vectors read back differently from memory")
	}
	time.Sleep(50 * time.Millisecond)
	if in, err = nc.Info(ctx); err != nil || in.Raw.Added != 0 {
		t.Fatalf("written while paused: %d (%v)", in.Raw.Added, err)
	}
	if err := nc.PauseUpdateWrites(ctx, false); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(10 * time.Second)
	for in.Raw.Added < k && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
		if in, err = nc.Info(ctx); err != nil {
			t.Fatal(err)
		}
	}
	if in.Raw.Added != k {
		t.Fatalf("after resuming: %d of %d inserted vectors on the SSD", in.Raw.Added, k)
	}
	if !bytes.Equal(read(), vecs) {
		t.Fatal("inserted vectors read back differently from the SSD")
	}
}

// builtLists returns, for the first k vectors, the lists the build put them in (ascending), from
// the index's postings.bin: u64 lists, u64 ids, u64 offsets[lists + 1], u32 ids[ids].
func builtLists(t *testing.T, path string, k int) [][]uint32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	nl := int(binary.LittleEndian.Uint64(b))
	off := func(c int) int { return int(binary.LittleEndian.Uint64(b[16+8*c:])) }
	ids := b[16+8*(nl+1):]
	out := make([][]uint32, k)
	for c := 0; c < nl; c++ {
		for j := off(c); j < off(c+1); j++ {
			if id := binary.LittleEndian.Uint32(ids[4*j:]); int(id) < k {
				out[id] = append(out[id], uint32(c))
			}
		}
	}
	return out
}

// builtCodes returns the first k PQ codes of the index's pq_codes.bin (u64 n, u32 m, u32 pad,
// codes).
func builtCodes(t *testing.T, path string, k int) [][]byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	m := int(binary.LittleEndian.Uint32(b[8:]))
	out := make([][]byte, k)
	for i := range out {
		out[i] = b[16+i*m : 16+(i+1)*m]
	}
	return out
}
