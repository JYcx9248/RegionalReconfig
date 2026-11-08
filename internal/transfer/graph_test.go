package transfer

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
	"sync/atomic"
	"testing"
)

// serveGraph serves src's graph on a local port.
func serveGraph(t *testing.T, srv *Server) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	go srv.Serve(ln)
	return ln.Addr().String()
}

// A graph pull hands every chunk, in order, to put. A background pull waits for the sender's
// Yield before each chunk and for the receiver's gate before each read, a data pull for
// neither; errors at the source or in put end it, and so does a graph that changes under it.
func TestPullGraph(t *testing.T) {
	data := bytes.Repeat([]byte("navgraph"), 40000) // 320 KB
	var reads atomic.Int64
	src := func(ctx context.Context, version uint32, off int64, n int) ([]byte, int64, uint64, error) {
		reads.Add(1)
		if version != 3 {
			return nil, 0, 0, fmt.Errorf("graph version %d is not here", version)
		}
		return data[off:min(off+int64(n), int64(len(data)))], int64(len(data)), 7, nil
	}
	var yields atomic.Int64
	addr := serveGraph(t, &Server{Graph: src, Limiter: NewTokenBucket(0, 0), ChunkBytes: 64 << 10,
		Yield: func() error { yields.Add(1); return nil }})
	ctx := context.Background()
	var got []byte
	put := func(ctx context.Context, off int64, b []byte) error {
		if off != int64(len(got)) {
			return fmt.Errorf("chunk at %d after %d bytes", off, len(got))
		}
		got = append(got, b...)
		return nil
	}

	var gates atomic.Int64
	st, err := PullGraph(ctx, addr, 3, 0, Background, func(context.Context) error { gates.Add(1); return nil }, put)
	if err != nil || st.Bytes != int64(len(data)) || !bytes.Equal(got, data) {
		t.Fatalf("background pull: %+v, %v (%d of %d bytes)", st, err, len(got), len(data))
	}
	if y, g := yields.Load(), gates.Load(); y < 5 || g < 5 {
		t.Fatalf("background pull of 5 chunks: %d yields, %d gates", y, g)
	}

	yields.Store(0)
	got = nil
	before := reads.Load()
	if _, err := PullGraph(ctx, addr, 3, 100<<10, Data, nil, put); err != nil || !bytes.Equal(got, data) {
		t.Fatalf("data pull: %v", err)
	}
	if y, n := yields.Load(), reads.Load()-before; y != 0 || n != 4 { // 320 KB in 100 KiB chunks
		t.Fatalf("data pull: %d yields, %d reads for 4 chunks", y, n)
	}

	got = nil
	if _, err := PullGraph(ctx, addr, 4, 0, Data, nil, put); err == nil || !strings.Contains(err.Error(), "not here") {
		t.Fatalf("pull of a version the source does not have: %v", err)
	}
	full := func(context.Context, int64, []byte) error { return errors.New("no room") }
	if _, err := PullGraph(ctx, addr, 3, 0, Data, nil, full); err == nil || !strings.Contains(err.Error(), "no room") {
		t.Fatalf("pull whose put fails: %v", err)
	}

	// The source's graph changes at its third chunk: the pull fails instead of mixing two graphs.
	var calls atomic.Int64
	changing := func(ctx context.Context, version uint32, off int64, n int) ([]byte, int64, uint64, error) {
		return data[off:min(off+int64(n), int64(len(data)))], int64(len(data)), uint64(calls.Add(1) / 3), nil
	}
	got = nil
	if _, err := PullGraph(ctx, serveGraph(t, &Server{Graph: changing, ChunkBytes: 64 << 10}), 3, 0, Data, nil, put); err == nil ||
		!strings.Contains(err.Error(), "changed") {
		t.Fatalf("pull across a graph change: %v", err)
	}
	if _, err := PullGraph(ctx, serveGraph(t, &Server{}), 3, 0, Data, nil, put); err == nil ||
		!strings.Contains(err.Error(), "does not serve") {
		t.Fatalf("pull from a node that does not serve its graph: %v", err)
	}
}

// A partition's pull hands every chunk, in order, to put, at the data class (no yield); a
// partition or version the source does not hold ends it, and so does a segment loaded again
// under it.
func TestPullSegment(t *testing.T) {
	data := bytes.Repeat([]byte("postings"), 30000) // 240 KB
	var calls atomic.Int64
	src := func(ctx context.Context, p int, v uint32, off int64, n int) ([]byte, int64, uint64, error) {
		if p != 5 || v != 2 {
			return nil, 0, 0, fmt.Errorf("partition %d is not resident at version %d", p, v)
		}
		stamp := uint64(1)
		if calls.Add(1) > 10 { // loaded again after the first pull's chunks
			stamp = 2
		}
		return data[off:min(off+int64(n), int64(len(data)))], int64(len(data)), stamp, nil
	}
	var yields atomic.Int64
	addr := serveGraph(t, &Server{Segment: src, Limiter: NewTokenBucket(0, 0), ChunkBytes: 64 << 10,
		Yield: func() error { yields.Add(1); return nil }})
	ctx := context.Background()
	var got []byte
	put := func(ctx context.Context, off int64, b []byte) error {
		if off != int64(len(got)) {
			return fmt.Errorf("chunk at %d after %d bytes", off, len(got))
		}
		got = append(got, b...)
		return nil
	}
	st, err := PullSegment(ctx, addr, 5, 2, 0, Data, put)
	if err != nil || st.Bytes != int64(len(data)) || !bytes.Equal(got, data) || yields.Load() != 0 || calls.Load() != 4 {
		t.Fatalf("pull: %+v, %v (%d of %d bytes, %d yields, %d reads for 4 chunks)", st, err, len(got), len(data),
			yields.Load(), calls.Load())
	}
	got = nil
	if _, err := PullSegment(ctx, addr, 5, 3, 0, Data, put); err == nil || !strings.Contains(err.Error(), "not resident") {
		t.Fatalf("pull of a version the source does not hold: %v", err)
	}
	got = nil
	calls.Store(8)
	if _, err := PullSegment(ctx, addr, 5, 2, 0, Data, put); err == nil || !strings.Contains(err.Error(), "partition 5 changed") {
		t.Fatalf("pull across a load: %v", err)
	}
	if _, err := PullSegment(ctx, serveGraph(t, &Server{}), 5, 2, 0, Data, put); err == nil ||
		!strings.Contains(err.Error(), "does not serve its partitions") {
		t.Fatalf("pull from a node that does not serve its partitions: %v", err)
	}
}
