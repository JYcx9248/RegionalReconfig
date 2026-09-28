package ctrl

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"sync"
	"testing"
	"time"
)

type echoReq struct{ X int }
type echoResp struct{ Y int }

func pair(t *testing.T, setupServer func(*Conn)) (*Conn, func()) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	go Serve(ln, setupServer)
	c, err := Dial(context.Background(), ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	return c, func() { c.Close(); ln.Close() }
}

func TestCallNotifyAndErrors(t *testing.T) {
	var mu sync.Mutex
	var notified []int
	gotNote := make(chan struct{}, 1)
	c, stop := pair(t, func(s *Conn) {
		s.Handle("Echo", func(ctx context.Context, b json.RawMessage) (any, error) {
			r, err := Decode[echoReq](b)
			return echoResp{Y: r.X * 2}, err
		})
		s.Handle("Fail", func(ctx context.Context, b json.RawMessage) (any, error) {
			return nil, errors.New("boom")
		})
		s.Handle("Panic", func(ctx context.Context, b json.RawMessage) (any, error) {
			panic("bad")
		})
		s.Handle("Slow", func(ctx context.Context, b json.RawMessage) (any, error) {
			time.Sleep(200 * time.Millisecond)
			return echoResp{Y: 1}, nil
		})
		s.OnNotify("Note", func(b json.RawMessage) {
			r, _ := Decode[echoReq](b)
			mu.Lock()
			notified = append(notified, r.X)
			mu.Unlock()
			gotNote <- struct{}{}
		})
	})
	defer stop()
	c.Start()
	ctx := context.Background()

	var out echoResp
	if err := c.Call(ctx, "Echo", echoReq{X: 21}, &out); err != nil || out.Y != 42 {
		t.Fatalf("Echo: %v %+v", err, out)
	}
	var re *RemoteError
	if err := c.Call(ctx, "Fail", nil, nil); !errors.As(err, &re) || re.Msg != "boom" {
		t.Fatalf("Fail: %v", err)
	}
	if err := c.Call(ctx, "Panic", nil, nil); !errors.As(err, &re) {
		t.Fatalf("Panic: %v", err)
	}
	if err := c.Call(ctx, "Nope", nil, nil); !errors.As(err, &re) {
		t.Fatalf("unknown method: %v", err)
	}

	// Concurrent calls, one slow: responses are matched by request ID.
	var wg sync.WaitGroup
	start := time.Now()
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			var o echoResp
			m := "Echo"
			if i == 0 {
				m = "Slow"
			}
			if err := c.Call(ctx, m, echoReq{X: i}, &o); err != nil {
				t.Errorf("%s: %v", m, err)
			} else if m == "Echo" && o.Y != 2*i {
				t.Errorf("mismatched response %d for %d", o.Y, i)
			}
		}(i)
	}
	wg.Wait()
	if time.Since(start) > 2*time.Second {
		t.Fatal("calls were serialized behind the slow handler")
	}

	if err := c.Notify("Note", echoReq{X: 5}); err != nil {
		t.Fatal(err)
	}
	select {
	case <-gotNote:
	case <-time.After(2 * time.Second):
		t.Fatal("notification not delivered")
	}

	tctx, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
	defer cancel()
	if err := c.Call(tctx, "Slow", nil, nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("timeout: %v", err)
	}
}

func TestPendingCallsFailOnClose(t *testing.T) {
	c, stop := pair(t, func(s *Conn) {
		s.Handle("Hang", func(ctx context.Context, b json.RawMessage) (any, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		})
	})
	defer stop()
	c.Start()
	errc := make(chan error, 1)
	go func() { errc <- c.Call(context.Background(), "Hang", nil, nil) }()
	time.Sleep(50 * time.Millisecond)
	c.Close()
	select {
	case err := <-errc:
		if !errors.Is(err, ErrClosed) {
			t.Fatalf("got %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("pending call not released")
	}
	<-c.Done()
}
