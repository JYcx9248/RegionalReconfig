package agent

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

// The graph waits while data moves -- a staging here, or a data-class transfer out -- and goes on
// once it has stopped; it counts the time it waited. Raw vectors do not hold it back: the yield
// has no input for them (the queries' fetches go on as long as the load does).
func TestGraphYield(t *testing.T) {
	var data atomic.Bool
	var waitedMs atomic.Int64
	const poll = 10 * time.Millisecond
	y := &graphYield{dataMoving: data.Load, poll: poll,
		waited: func(d time.Duration) { waitedMs.Add(d.Milliseconds()) }}
	ctx := context.Background()

	for range 100 { // nothing moves: no wait, none counted
		if err := y.Wait(ctx); err != nil {
			t.Fatal(err)
		}
	}
	if n := waitedMs.Load(); n != 0 {
		t.Errorf("%d ms counted as waited while nothing moved", n)
	}

	data.Store(true)
	done := make(chan error, 1)
	go func() { done <- y.Wait(ctx) }()
	select {
	case err := <-done:
		t.Fatalf("returned (%v) while data moved", err)
	case <-time.After(8 * poll):
	}
	data.Store(false)
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("still waiting after the data stopped")
	}
	if n := waitedMs.Load(); n < 8*poll.Milliseconds() {
		t.Errorf("a wait of over %v counted as %d ms", 8*poll, n)
	}

	data.Store(true)
	cctx, cancel := context.WithTimeout(ctx, 3*poll)
	defer cancel()
	if err := y.Wait(cctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("a wait past its deadline: %v", err)
	}
}
