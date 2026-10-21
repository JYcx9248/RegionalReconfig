package agent

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

// The graph waits while data moves and while the raw-vector counter changes from one poll to
// the next, and goes on once both have stopped.
func TestGraphYield(t *testing.T) {
	var data atomic.Bool
	var raw atomic.Uint64
	var polls, waitedMs atomic.Int64
	const poll = 10 * time.Millisecond
	y := &graphYield{dataMoving: data.Load, poll: poll, rawBytes: func(context.Context) (uint64, error) {
		polls.Add(1)
		return raw.Load(), nil
	}, waited: func(d time.Duration) { waitedMs.Add(d.Milliseconds()) }}
	ctx := context.Background()
	waiting := func(what string) chan error {
		done := make(chan error, 1)
		go func() { done <- y.Wait(ctx) }()
		select {
		case err := <-done:
			t.Fatalf("returned (%v) while %s", err, what)
		case <-time.After(8 * poll):
		}
		return done
	}
	resumes := func(done chan error, what string) {
		select {
		case err := <-done:
			if err != nil {
				t.Fatal(err)
			}
		case <-time.After(time.Second):
			t.Fatalf("still waiting after %s stopped", what)
		}
	}

	if err := y.Wait(ctx); err != nil { // nothing moves
		t.Fatal(err)
	}
	before := polls.Load()
	for i := 0; i < 100; i++ { // the chunks within one interval ask the node once at most
		if err := y.Wait(ctx); err != nil {
			t.Fatal(err)
		}
	}
	if n := polls.Load() - before; n > 1 {
		t.Errorf("%d polls for 100 waits within an interval", n)
	}
	if n := waitedMs.Load(); n != 0 {
		t.Errorf("%d ms counted as waited while nothing moved", n)
	}

	data.Store(true)
	done := waiting("data moved")
	data.Store(false)
	resumes(done, "the data")
	if n := waitedMs.Load(); n < 8*poll.Milliseconds() {
		t.Errorf("a wait of over %v counted as %d ms", 8*poll, n)
	}

	stop := make(chan struct{})
	go func() {
		for {
			select {
			case <-stop:
				return
			case <-time.After(time.Millisecond):
				raw.Add(4096)
			}
		}
	}()
	time.Sleep(2 * poll) // past the interval the last wait was answered in
	done = waiting("raw vectors moved")
	close(stop)
	resumes(done, "the raw vectors")

	data.Store(true)
	cctx, cancel := context.WithTimeout(ctx, 3*poll)
	defer cancel()
	if err := y.Wait(cctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("a wait past its deadline: %v", err)
	}
	data.Store(false)

	failing := &graphYield{dataMoving: data.Load, poll: poll, rawBytes: func(context.Context) (uint64, error) {
		return 0, errors.New("node down")
	}}
	if err := failing.Wait(ctx); err == nil {
		t.Fatal("a node that cannot be asked let the graph through")
	}
}
