package nodeclient

import (
	"context"
	"errors"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"rtier/internal/frame"
)

// fakeNode answers PING with OK and everything else with BAD_REQUEST (enough for a client to
// return), holding each RERANK until release is closed, as a node holds one that waits for its
// holders' raw vectors. It counts the connections and the RERANKs in flight.
type fakeNode struct {
	ln      net.Listener
	release chan struct{}
	conns   atomic.Int64
	reranks atomic.Int64 // in flight
	peak    atomic.Int64 // the most RERANKs in flight at once
}

func newFakeNode(t *testing.T) *fakeNode {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	n := &fakeNode{ln: ln, release: make(chan struct{})}
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			n.conns.Add(1)
			go n.serve(c)
		}
	}()
	t.Cleanup(func() { ln.Close() })
	return n
}

func (n *fakeNode) serve(c net.Conn) {
	defer c.Close()
	for {
		var f frame.Frame
		if err := frame.Read(c, &f); err != nil {
			return
		}
		if f.Type == opRerank {
			v := n.reranks.Add(1)
			for p := n.peak.Load(); v > p && !n.peak.CompareAndSwap(p, v); p = n.peak.Load() {
			}
			<-n.release
			n.reranks.Add(-1)
		}
		resp := frame.Frame{Type: f.Type, Flags: frame.FlagResponse, ReqID: f.ReqID, Epoch: f.Epoch}
		if f.Type != opPing {
			resp.Status, resp.Body = StatusBadRequest, []byte("fake node")
		}
		if err := frame.Write(c, &resp); err != nil {
			return
		}
	}
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	for deadline := time.Now().Add(2 * time.Second); !cond(); time.Sleep(time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
	}
}

// A RERANK waiting at the node holds a RERANK slot, not one a FILTER needs: with one slot of
// each kind, a FILTER is answered meanwhile, and a second RERANK waits for the first.
func TestRerankHasItsOwnSlots(t *testing.T) {
	n := newFakeNode(t)
	c := New(n.ln.Addr().String(), 1)
	ctx := context.Background()
	q := []byte{1, 2, 3, 4}
	var answered *StatusError
	done := make(chan error, 2)
	for i := 0; i < 2; i++ {
		go func() {
			_, _, err := c.Rerank(ctx, 1, q, []uint32{7}, []uint32{0}, nil, 10)
			done <- err
		}()
	}
	waitFor(t, "a RERANK at the node", func() bool { return n.reranks.Load() == 1 })
	fctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	if _, _, err := c.Filter(fctx, 1, q, []uint32{0}, nil, 10); !errors.As(err, &answered) {
		t.Fatalf("FILTER next to a waiting RERANK: %v", err)
	}
	time.Sleep(20 * time.Millisecond)
	if p := n.peak.Load(); p != 1 {
		t.Fatalf("%d RERANKs in flight on one slot", p)
	}
	close(n.release)
	for i := 0; i < 2; i++ {
		if err := <-done; !errors.As(err, &answered) {
			t.Fatalf("RERANK: %v", err)
		}
	}
	if o := c.Open(); o != 2 {
		t.Fatalf("%d connections open, want 2 (one per kind in flight at once)", o)
	}
}

// Preconnect opens a connection for every slot of both kinds, and requests of either kind then
// open none.
func TestPreconnectFillsBothKinds(t *testing.T) {
	n := newFakeNode(t)
	close(n.release)
	c := New(n.ln.Addr().String(), 3)
	ctx := context.Background()
	if err := c.Preconnect(ctx); err != nil {
		t.Fatal(err)
	}
	if c.Open() != 6 || n.conns.Load() != 6 {
		t.Fatalf("%d open, %d accepted: want 6", c.Open(), n.conns.Load())
	}
	q := []byte{1, 2, 3, 4}
	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			c.Rerank(ctx, 1, q, []uint32{7}, []uint32{0}, nil, 10)
		}()
		go func() {
			defer wg.Done()
			c.Filter(ctx, 1, q, []uint32{0}, nil, 10)
		}()
	}
	wg.Wait()
	if err := c.Preconnect(ctx); err != nil || n.conns.Load() != 6 || c.Open() != 6 {
		t.Fatalf("after the requests and a second Preconnect: %v, %d accepted, %d open",
			err, n.conns.Load(), c.Open())
	}
	n.ln.Close()
	if err := New(n.ln.Addr().String(), 1).Preconnect(ctx); err == nil {
		t.Fatal("Preconnect to a node that is gone succeeded")
	}
}

// streamNode answers PQ_MISSING with three frames of one ID each (group 1), RAW_HELD_BY with two
// frames whose last counts the repointed postings, and RAW_MISSING with a frame and then an error.
func streamNode(t *testing.T) (addr string, conns *atomic.Int64) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	conns = new(atomic.Int64)
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			conns.Add(1)
			go func() {
				defer c.Close()
				for {
					var f frame.Frame
					if frame.Read(c, &f) != nil {
						return
					}
					send := func(flags uint8, status uint16, body []byte) {
						frame.Write(c, &frame.Frame{Type: f.Type, Flags: frame.FlagResponse | flags, Status: status, ReqID: f.ReqID, Body: body})
					}
					switch f.Type {
					case opPQMissing:
						for i := 0; i < 3; i++ {
							var w frame.Writer
							w.U32(1) // one run
							w.U32(1) // group 1
							w.U32(1)
							w.U32(uint32(10 + i))
							send(map[bool]uint8{true: 0, false: frame.FlagMore}[i == 2], StatusOK, w.B)
						}
					case opRawHeldBy:
						var w frame.Writer
						w.U64(0)
						w.U32(2)
						w.U32s([]uint32{5, 6})
						w.U32s([]uint32{50, 60})
						send(frame.FlagMore, StatusOK, w.B)
						w = frame.Writer{}
						w.U64(7)
						w.U32(0)
						send(0, StatusOK, w.B)
					case opRawMissing:
						var w frame.Writer
						w.U32(1)
						w.U16(3)
						w.U16(0)
						w.U32(1)
						w.U32s([]uint32{9})
						w.U32s([]uint32{90})
						send(frame.FlagMore, StatusOK, w.B)
						send(0, StatusRawAbsent, []byte("no holder recorded"))
					}
				}
			}()
		}
	}()
	return ln.Addr().String(), conns
}

// A streamed reply reaches fn frame by frame; an error from fn closes the connection (frames may
// still be on their way), and an error frame ends the stream with the connection kept.
func TestStreamedListings(t *testing.T) {
	addr, conns := streamNode(t)
	c := New(addr, 1)
	ctx := context.Background()
	var got []uint32
	if err := c.PQMissing(ctx, [][]string{{"a.seg"}, {"b.seg"}}, func(g int, ids []uint32) error {
		if g != 1 {
			t.Errorf("group %d", g)
		}
		got = append(got, ids...)
		return nil
	}); err != nil || len(got) != 3 || got[0] != 10 || got[2] != 12 {
		t.Fatalf("PQ_MISSING: %v, %v", got, err)
	}
	var held []uint32
	repointed, err := c.RawHeldBy(ctx, 4, func(m RawRefs) error {
		if m.Holder != 4 || len(m.Lists) != len(m.Locs) {
			t.Errorf("refs %+v", m)
		}
		held = append(held, m.Locs...)
		return nil
	})
	if err != nil || repointed != 7 || len(held) != 2 {
		t.Fatalf("RAW_HELD_BY: %d repointed, %v, %v", repointed, held, err)
	}
	var raw int
	var absent *StatusError
	err = c.RawMissing(ctx, []int{0}, func(groups []RawRefs) error {
		raw += len(groups)
		return nil
	})
	if !errors.As(err, &absent) || absent.Status != StatusRawAbsent || raw != 1 || c.Open() != 1 {
		t.Fatalf("RAW_MISSING ending in an error: %v, %d groups, %d open", err, raw, c.Open())
	}
	stopped := errors.New("enough")
	if err := c.PQMissing(ctx, [][]string{{"a.seg"}, {"b.seg"}}, func(int, []uint32) error { return stopped }); !errors.Is(err, stopped) {
		t.Fatalf("PQ_MISSING stopped by fn: %v", err)
	}
	if o := c.Open(); o != 0 || conns.Load() != 1 {
		t.Fatalf("%d open, %d accepted: the stopped stream's connection must go", o, conns.Load())
	}
	if err := c.PQMissing(ctx, [][]string{{"a.seg"}, {"b.seg"}}, func(int, []uint32) error { return nil }); err != nil || conns.Load() != 2 {
		t.Fatalf("after a stopped stream: %v, %d accepted", err, conns.Load())
	}
}
