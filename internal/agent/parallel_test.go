package agent

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

// parallelLimit runs every call, never more than limit at once; after a failure, or once its
// context ends, it starts no more, and it reports the failure, or the context's end if calls
// were left unstarted.
func TestParallelLimit(t *testing.T) {
	var running, peak, done atomic.Int64
	work := func(ctx context.Context, i int) error {
		if r := running.Add(1); r > peak.Load() {
			peak.Store(r)
		}
		time.Sleep(5 * time.Millisecond)
		running.Add(-1)
		done.Add(1)
		return nil
	}
	if err := parallelLimit(context.Background(), 20, 3, work); err != nil || done.Load() != 20 || peak.Load() > 3 {
		t.Fatalf("20 calls, 3 at once: %v, %d done, %d at once", err, done.Load(), peak.Load())
	}

	failed := errors.New("no room")
	done.Store(0)
	err := parallelLimit(context.Background(), 20, 2, func(ctx context.Context, i int) error {
		if i == 1 {
			return failed
		}
		return work(ctx, i)
	})
	if !errors.Is(err, failed) || done.Load() >= 19 {
		t.Fatalf("a failing call: %v, %d others done", err, done.Load())
	}

	ctx, cancel := context.WithCancel(context.Background())
	done.Store(0)
	err = parallelLimit(ctx, 20, 1, func(ctx context.Context, i int) error {
		if i == 2 {
			cancel()
		}
		return work(ctx, i)
	})
	if !errors.Is(err, context.Canceled) || done.Load() >= 20 {
		t.Fatalf("a context that ends: %v, %d done", err, done.Load())
	}
	if err := parallelLimit(context.Background(), 0, 4, work); err != nil {
		t.Fatalf("no calls: %v", err)
	}
}
