package e2e

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"

	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
	"rtier/internal/update"
	"rtier/internal/vecio"
)

// TestRNGInsertRouting: on an index whose replicas the build pruned by SPANN's RNG rule
// (fusion_build --rng, the index SPFresh runs on), a node routes inserts by that rule (U14): the
// index's own vectors get back the lists and PQ codes the build gave them, fewer than the
// replica cap. ASSIGN_CODES over every vector's code, its frames side by side, gives what one
// frame at a time does.
func TestRNGInsertRouting(t *testing.T) {
	f := getRNGFixture(t)
	meta, err := os.ReadFile(filepath.Join(f.index, "meta.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(meta), "\nrng=1\n") {
		t.Fatalf("the fixture's index was not built with RNG pruning:\n%s", meta)
	}
	m, err := partitioning.LoadManifest(f.parts)
	if err != nil {
		t.Fatal(err)
	}
	const k = 300
	addr, stop, err := startNode(f, "--graph", "index", "--max-vectors", strconv.Itoa(numVectors+k),
		"--max-pages", strconv.FormatUint(m.NumPages+uint64(k)/m.VectorsPerPage+2, 10))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := stop(); err != nil {
			t.Error(err)
		}
	}()
	ctx := context.Background()
	nc := nodeclient.New(addr, 1)
	defer nc.Close()
	in, err := nc.Info(ctx)
	if err != nil {
		t.Fatal(err)
	}
	base, err := vecio.ReadBin(filepath.Join(f.dir, "base.fbin"), k)
	if err != nil {
		t.Fatal(err)
	}
	vb := len(base.Row(0))
	vecs := make([]byte, 0, k*vb)
	for i := 0; i < k; i++ {
		vecs = append(vecs, base.Row(i)...)
	}
	alloc, err := update.NewAllocator(numVectors, in.IDCapacity, m.Locations(), in.Raw.Locations)
	if err != nil {
		t.Fatal(err)
	}
	ins, err := update.Route(ctx, nc, alloc, vecs, vb)
	if err != nil {
		t.Fatal(err)
	}
	built := builtLists(t, filepath.Join(f.index, "postings.bin"), k)
	codes := builtCodes(t, filepath.Join(f.index, "pq_codes.bin"), k)
	same, total := 0, 0
	for i, x := range ins {
		got := append([]uint32(nil), x.Lists...)
		sort.Slice(got, func(a, b int) bool { return got[a] < got[b] })
		if reflect.DeepEqual(got, built[i]) && bytes.Equal(x.Code, codes[i]) {
			same++
		}
		total += len(got)
	}
	t.Logf("%d of %d vectors routed to the build's lists, %.2f lists each", same, k, float64(total)/k)
	if same != k {
		t.Fatalf("%d of %d vectors routed to the build's lists with the build's PQ code", same, k)
	}
	if total >= 8*k/2 {
		t.Fatalf("%d lists for %d vectors: the RNG rule pruned nothing", total, k)
	}

	// ASSIGN_CODES (SPFresh's reassign) side by side: every vector's PQ code, in several frames at
	// once, gives what one frame at a time gives, in order.
	all := builtCodes(t, filepath.Join(f.index, "pq_codes.bin"), numVectors)
	var every []byte
	for _, c := range all {
		every = append(every, c...)
	}
	cm := len(all[0])
	wide := nodeclient.New(addr, 4)
	defer wide.Close()
	got, err := update.AssignCodes(ctx, wide, every, cm)
	if err != nil {
		t.Fatal(err)
	}
	var want [][]uint32
	for i := 0; i < len(every); i += 4096 * cm {
		sets, err := nc.AssignCodes(ctx, every[i:min(len(every), i+4096*cm)], cm)
		if err != nil {
			t.Fatal(err)
		}
		want = append(want, sets...)
	}
	if len(got) != numVectors || !reflect.DeepEqual(got, want) {
		t.Fatalf("ASSIGN_CODES side by side: %d sets, not those of one frame at a time", len(got))
	}
}
