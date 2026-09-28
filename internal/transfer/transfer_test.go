package transfer

import (
	"bytes"
	"context"
	"errors"
	"math/rand"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"rtier/internal/design"
)

func TestTokenBucketPaces(t *testing.T) {
	b := NewTokenBucket(1<<20, 64<<10) // 1 MiB/s, 64 KiB burst
	start := time.Now()
	ctx := context.Background()
	for i := 0; i < 5; i++ {
		if err := b.Wait(ctx, 64<<10); err != nil {
			t.Fatal(err)
		}
	}
	// 5 x 64 KiB with a full 64 KiB bucket at start: ~4 x 62.5 ms.
	if el := time.Since(start); el < 200*time.Millisecond || el > 2*time.Second {
		t.Fatalf("elapsed %v", el)
	}
	unlimited := NewTokenBucket(0, 0)
	if err := unlimited.Wait(ctx, 1<<30); err != nil {
		t.Fatal(err)
	}
	cctx, cancel := context.WithCancel(ctx)
	cancel()
	slow := NewTokenBucket(1, 1)
	if err := slow.Wait(cctx, 10); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v", err)
	}
}

func TestPullFiles(t *testing.T) {
	src, dst := t.TempDir(), t.TempDir()
	rng := rand.New(rand.NewSource(1))
	want := map[string][]byte{}
	for _, name := range []string{"part-00001.seg", "sub/graph.hnsw", "empty.seg"} {
		n := 300000
		if name == "empty.seg" {
			n = 0
		}
		data := make([]byte, n)
		rng.Read(data)
		want[name] = data
		os.MkdirAll(filepath.Dir(filepath.Join(src, name)), 0o755)
		if err := os.WriteFile(filepath.Join(src, name), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	defer ln.Close()
	srv := &Server{Root: src, Limiter: NewTokenBucket(0, 0), ChunkBytes: 64 << 10}
	go srv.Serve(ln)

	files := []string{"part-00001.seg", "sub/graph.hnsw", "empty.seg"}
	st, err := Pull(context.Background(), ln.Addr().String(), files, dst, 0, Data)
	if err != nil {
		t.Fatal(err)
	}
	if st.Files != 3 || st.Bytes != 600000 || srv.BytesSent.Load() != 600000 {
		t.Fatalf("stats %+v sent %d", st, srv.BytesSent.Load())
	}
	for name, data := range want {
		got, err := os.ReadFile(filepath.Join(dst, name))
		if err != nil || !bytes.Equal(got, data) {
			t.Fatalf("%s differs (%v)", name, err)
		}
	}
	if _, err := Pull(context.Background(), ln.Addr().String(), []string{"../etc/passwd"}, dst, 0, Data); err == nil {
		t.Fatal("path traversal accepted")
	}
	if _, err := Pull(context.Background(), ln.Addr().String(), []string{"missing.seg"}, dst, 0, Background); err == nil {
		t.Fatal("missing file accepted")
	}
	if _, err := os.Stat(filepath.Join(dst, "missing.seg.part")); !os.IsNotExist(err) {
		t.Fatal("partial file left behind")
	}

	cp := t.TempDir()
	if st, err := CopyLocal(src, files, cp); err != nil || st.Bytes != 600000 {
		t.Fatalf("CopyLocal: %+v %v", st, err)
	}
}

func TestAdapters(t *testing.T) {
	a, _ := NewAdapter("fixed", 100)
	if r, err := a.Rate(Observation{}); err != nil || r != 100 {
		t.Fatal("fixed")
	}
	a, _ = NewAdapter("adaptive", 0)
	if _, err := a.Rate(Observation{}); !errors.Is(err, design.ErrUndecided) {
		t.Fatal("adaptive should be a placeholder")
	}
}

func TestPullPQ(t *testing.T) {
	const m = 6
	code := func(id uint32) []byte { // deterministic fake code
		return []byte{byte(id), byte(id >> 8), byte(id >> 16), byte(id >> 24), byte(id * 7), byte(id * 13)}
	}
	var served atomic.Int64
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	defer ln.Close()
	srv := &Server{ChunkBytes: 100, Limiter: NewTokenBucket(0, 0)} // tiny chunks: many frames
	srv.PQ = func(_ context.Context, ids []uint32) (int, []byte, error) {
		served.Add(int64(len(ids)))
		var out []byte
		for _, id := range ids {
			if id == 999999 {
				return 0, nil, errors.New("no PQ code for vector 999999")
			}
			out = append(out, code(id)...)
		}
		return m, out, nil
	}
	go srv.Serve(ln)

	ids := make([]uint32, 50000)
	for i := range ids {
		ids[i] = uint32(i*3 + 1)
	}
	got := map[uint32][]byte{}
	batches := 0
	st, err := PullPQ(context.Background(), ln.Addr().String(), ids, 7000, func(b []uint32, bm int, codes []byte) error {
		if bm != m || len(codes) != m*len(b) {
			t.Fatalf("batch of %d IDs: m=%d, %d bytes", len(b), bm, len(codes))
		}
		for i, id := range b {
			got[id] = codes[i*m : (i+1)*m]
		}
		batches++
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if st.Codes != int64(len(ids)) || st.Bytes != int64(m*len(ids)) || batches != 8 || served.Load() != int64(len(ids)) {
		t.Fatalf("stats %+v, %d batches, %d served", st, batches, served.Load())
	}
	for _, id := range ids {
		if string(got[id]) != string(code(id)) {
			t.Fatalf("vector %d: got %v want %v", id, got[id], code(id))
		}
	}
	if srv.BytesSent.Load() != int64(m*len(ids)) {
		t.Fatalf("server counted %d bytes", srv.BytesSent.Load())
	}

	// Nothing to fetch: no connection needed.
	if st, err := PullPQ(context.Background(), "127.0.0.1:1", nil, 0, nil); err != nil || st.Codes != 0 {
		t.Fatalf("empty pull: %+v %v", st, err)
	}
	// The source's error reaches the client; the sink's error stops the pull.
	if _, err := PullPQ(context.Background(), ln.Addr().String(), []uint32{5, 999999}, 0,
		func([]uint32, int, []byte) error { return nil }); err == nil || !strings.Contains(err.Error(), "999999") {
		t.Fatalf("source error: %v", err)
	}
	sinkErr := errors.New("capacity exhausted")
	if _, err := PullPQ(context.Background(), ln.Addr().String(), ids[:10], 0,
		func([]uint32, int, []byte) error { return sinkErr }); !errors.Is(err, sinkErr) {
		t.Fatalf("sink error: %v", err)
	}
	// A server without a PQ source refuses.
	ln2, _ := net.Listen("tcp", "127.0.0.1:0")
	defer ln2.Close()
	go (&Server{}).Serve(ln2)
	if _, err := PullPQ(context.Background(), ln2.Addr().String(), ids[:3], 0,
		func([]uint32, int, []byte) error { return nil }); err == nil || !strings.Contains(err.Error(), "does not serve PQ") {
		t.Fatalf("no source: %v", err)
	}
}

// Raw vectors come through the same record protocol at the class the client asks for; a
// server without a raw-vector source refuses.
func TestPullRaw(t *testing.T) {
	const vb = 40
	vec := func(loc uint32) []byte {
		v := make([]byte, vb)
		for i := range v {
			v[i] = byte(loc*5 + uint32(i))
		}
		return v
	}
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	defer ln.Close()
	srv := &Server{ChunkBytes: 1000, Limiter: NewTokenBucket(0, 0)}
	srv.PQ = func(context.Context, []uint32) (int, []byte, error) { return 0, nil, errors.New("not PQ") }
	srv.Raw = func(_ context.Context, locs []uint32) (int, []byte, error) {
		var out []byte
		for _, l := range locs {
			out = append(out, vec(l)...)
		}
		return vb, out, nil
	}
	go srv.Serve(ln)
	locs := make([]uint32, 10000)
	for i := range locs {
		locs[i] = uint32(i * 2)
	}
	n := 0
	st, err := PullRaw(context.Background(), ln.Addr().String(), locs, 3000, Background, nil, func(b []uint32, size int, vecs []byte) error {
		for i, l := range b {
			if size != vb || string(vecs[i*vb:(i+1)*vb]) != string(vec(l)) {
				t.Fatalf("location %d: wrong vector", l)
			}
		}
		n += len(b)
		return nil
	})
	if err != nil || n != len(locs) || st.Vectors != int64(len(locs)) || st.Bytes != int64(vb*len(locs)) || st.Codes != 0 {
		t.Fatalf("pull: %+v, %d vectors, %v", st, n, err)
	}
	// Before each batch the destination says what it still lacks: here, every other location
	// arrived meanwhile.
	n = 0
	st, err = PullRaw(context.Background(), ln.Addr().String(), locs, 3000, Data, func(b []uint32) ([]uint32, error) {
		var keep []uint32
		for i, l := range b {
			if i%2 == 0 {
				keep = append(keep, l)
			}
		}
		return keep, nil
	}, func(b []uint32, size int, vecs []byte) error {
		n += len(b)
		return nil
	})
	if err != nil || n != len(locs)/2 || st.Vectors != int64(len(locs)/2) {
		t.Fatalf("filtered pull: %+v, %d vectors, %v", st, n, err)
	}
	ln2, _ := net.Listen("tcp", "127.0.0.1:0")
	defer ln2.Close()
	go (&Server{}).Serve(ln2)
	if _, err := PullRaw(context.Background(), ln2.Addr().String(), locs[:3], 0, Data, nil,
		func([]uint32, int, []byte) error { return nil }); err == nil || !strings.Contains(err.Error(), "does not serve raw vectors") {
		t.Fatalf("no source: %v", err)
	}
}

// PQ pulls are paced by the sender's token bucket, like file pulls.
func TestPullPQRateLimited(t *testing.T) {
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	defer ln.Close()
	srv := &Server{ChunkBytes: 4 << 10, Limiter: NewTokenBucket(200_000, 10_000)}
	srv.PQ = func(_ context.Context, ids []uint32) (int, []byte, error) {
		return 6, make([]byte, 6*len(ids)), nil
	}
	go srv.Serve(ln)
	ids := make([]uint32, 20000) // 120 KB of codes: about 0.55 s at 200 KB/s after the burst
	for i := range ids {
		ids[i] = uint32(i)
	}
	start := time.Now()
	st, err := PullPQ(context.Background(), ln.Addr().String(), ids, 0, func([]uint32, int, []byte) error { return nil })
	if err != nil || st.Bytes != 120000 {
		t.Fatalf("pull: %+v %v", st, err)
	}
	if el := time.Since(start); el < 400*time.Millisecond {
		t.Fatalf("120 KB at 200 KB/s took %v: not paced", el)
	}
}

// A data transfer and a background transfer start together on one bucket. The data transfer
// must not be slowed down (the background one may take at most the bucket it finds full at the
// start), and the background one gets the rest only once the data transfer is over.
func TestBackgroundYieldsToData(t *testing.T) {
	ctx := context.Background()
	const chunk = 64 << 10
	b := NewTokenBucket(1<<20, chunk) // 1 MiB/s, 64 KiB burst: one chunk per 62.5 ms
	endData := b.Begin(Data)
	var dataDone, bgDone time.Time
	var wg sync.WaitGroup
	start := time.Now()
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < 8; i++ {
			if err := b.WaitClass(ctx, chunk, Data); err != nil {
				t.Error(err)
			}
		}
		dataDone = time.Now()
		endData()
	}()
	go func() {
		defer wg.Done()
		defer b.Begin(Background)()
		for i := 0; i < 3; i++ {
			if err := b.WaitClass(ctx, chunk, Background); err != nil {
				t.Error(err)
			}
		}
		bgDone = time.Now()
	}()
	wg.Wait()
	// 8 chunks at 62.5 ms: ~440 ms from a full bucket, ~500 ms if the background transfer
	// took the initial one. Sharing the rate would take ~625 ms or more.
	if d := dataDone.Sub(start); d > 580*time.Millisecond {
		t.Errorf("data transfer took %v: the background transfer slowed it down", d)
	}
	if !bgDone.After(dataDone) {
		t.Errorf("background transfer finished at %v, before the data transfer (%v)",
			bgDone.Sub(start), dataDone.Sub(start))
	}
}

// With a data transfer in progress but not sending (held back elsewhere), a background
// transfer still uses the idle rate: it is not blocked outright.
func TestBackgroundUsesIdleBandwidth(t *testing.T) {
	ctx := context.Background()
	const chunk = 64 << 10
	b := NewTokenBucket(1<<20, chunk)
	defer b.Begin(Data)()
	start := time.Now()
	for i := 0; i < 4; i++ {
		if err := b.WaitClass(ctx, chunk, Background); err != nil {
			t.Fatal(err)
		}
	}
	if d := time.Since(start); d < 150*time.Millisecond || d > time.Second {
		t.Fatalf("4 background chunks next to an idle data transfer took %v, want ~190 ms", d)
	}
	if _, err := ParseClass("urgent"); err == nil {
		t.Fatal("unknown priority accepted")
	}
	if c, err := ParseClass("background"); err != nil || c != Background || c.String() != "background" {
		t.Fatalf("ParseClass(background) = %v, %v", c, err)
	}
}

// A background pull goes through the same server path and arrives intact.
func TestPullBackground(t *testing.T) {
	src, dst := t.TempDir(), t.TempDir()
	data := bytes.Repeat([]byte("graph"), 60000)
	os.MkdirAll(filepath.Join(src, "graph"), 0o755)
	if err := os.WriteFile(filepath.Join(src, "graph", "graph.hnsw"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	defer ln.Close()
	srv := &Server{Root: src, Limiter: NewTokenBucket(8<<20, 0), ChunkBytes: 64 << 10}
	go srv.Serve(ln)
	st, err := Pull(context.Background(), ln.Addr().String(), []string{"graph/graph.hnsw"}, dst, 0, Background)
	if err != nil || st.Bytes != int64(len(data)) {
		t.Fatalf("background pull: %+v, %v", st, err)
	}
	got, err := os.ReadFile(filepath.Join(dst, "graph", "graph.hnsw"))
	if err != nil || !bytes.Equal(got, data) {
		t.Fatalf("graph differs (%v)", err)
	}
}
