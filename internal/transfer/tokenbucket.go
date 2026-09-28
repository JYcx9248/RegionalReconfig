// Package transfer moves immutable index files (partition segments, the navigation graph)
// between nodes on a connection of their own, separate from the query path, and paces the
// sender with a token bucket so background copies do not starve foreground queries.
//
// Chunked streaming follows Koala's StateChunkSender (worker/stateCommUtil/stateCommRpcUtil.go:
// fixed-size chunks, then an end-of-stream marker) over the rtier framing instead of gRPC.
package transfer

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// Class is a transfer's priority on the sender's token bucket.
//
// A joining node becomes useful in two steps: once it holds its partitions' posting lists and
// PQ codes it can own them and answer FILTER and RERANK, which is what takes load off the old
// owners; only with the navigation graph can it also take client queries, and navigation is
// the smallest share of a query's work, which any existing entry can do meanwhile. So what the
// first step needs goes first: the graph never slows a data transfer down, and gets whatever
// bandwidth the data transfers leave unused. Raw vectors are not needed for the first step
// either -- RERANK fetches the ones it lacks from the old owner on demand (U9) -- so the stream
// that brings the rest after the flip is background too by default ("raw_priority").
type Class int

const (
	// Data is what a node needs before it can own partitions: segment files and PQ codes.
	Data Class = iota
	// Background is the navigation graph, needed only for the entry role, and the raw vectors
	// streamed in after the flip.
	Background
)

// ParseClass maps "" and "data" to Data and "background" to Background.
func ParseClass(s string) (Class, error) {
	switch s {
	case "", "data":
		return Data, nil
	case "background":
		return Background, nil
	}
	return Data, fmt.Errorf("transfer: unknown priority %q (data | background)", s)
}

func (c Class) String() string {
	if c == Background {
		return "background"
	}
	return "data"
}

// TokenBucket paces bytes. Rate 0 means unlimited.
type TokenBucket struct {
	mu      sync.Mutex
	rate    float64 // bytes per second
	burst   float64
	tokens  float64
	last    time.Time
	active  [2]int // transfers in progress per class (Begin)
	waiting [2]int // callers blocked in WaitClass per class
}

// NewTokenBucket creates a bucket that starts full. burst <= 0 defaults to 1/10 s of rate
// (at least 64 KiB).
func NewTokenBucket(rate, burst float64) *TokenBucket {
	if burst <= 0 {
		burst = math.Max(rate/10, 64<<10)
	}
	return &TokenBucket{rate: rate, burst: burst, tokens: burst, last: time.Now()}
}

// SetRate changes the rate (the hook for adaptive control, open question U8).
func (b *TokenBucket) SetRate(rate float64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.refillLocked(time.Now())
	b.rate = rate
}

// Rate returns the current rate in bytes per second.
func (b *TokenBucket) Rate() float64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.rate
}

// Begin marks a transfer of class c as in progress until the returned function is called.
// Background yields to Data: it takes nothing while a data sender is waiting for tokens, and
// while a data transfer is in progress it only takes a full bucket -- tokens the data
// transfers left unused, because they are between requests or held back by a slow receiver.
// A background transfer therefore never delays a data transfer, and still uses a link the data
// transfers leave idle.
func (b *TokenBucket) Begin(c Class) func() {
	b.mu.Lock()
	b.active[c]++
	b.mu.Unlock()
	var once sync.Once
	return func() {
		once.Do(func() {
			b.mu.Lock()
			b.active[c]--
			b.mu.Unlock()
		})
	}
}

func (b *TokenBucket) refillLocked(now time.Time) {
	if b.rate > 0 {
		b.tokens = math.Min(b.burst, b.tokens+now.Sub(b.last).Seconds()*b.rate)
	}
	b.last = now
}

// Wait blocks until n bytes may be sent as Data. Requests larger than the burst are taken in
// pieces.
func (b *TokenBucket) Wait(ctx context.Context, n int) error { return b.WaitClass(ctx, n, Data) }

// WaitClass blocks until n bytes of class c may be sent (see Begin for how classes share).
func (b *TokenBucket) WaitClass(ctx context.Context, n int, c Class) error {
	remaining := float64(n)
	blocked := false // counted in b.waiting[c]
	defer func() {
		if blocked {
			b.mu.Lock()
			b.waiting[c]--
			b.mu.Unlock()
		}
	}()
	for remaining > 0 {
		b.mu.Lock()
		if b.rate <= 0 {
			b.mu.Unlock()
			return nil
		}
		b.refillLocked(time.Now())
		take := math.Min(remaining, b.burst)
		need, yield := take, false
		if c == Background {
			if b.waiting[Data] > 0 {
				yield = true // a data sender is waiting for these tokens
			} else if b.active[Data] > 0 {
				need = b.burst // only what the data transfers left unused
			}
		}
		var wait time.Duration
		switch {
		case yield:
			wait = time.Duration(take / b.rate * float64(time.Second)) // look again after a chunk's worth
		case b.tokens >= need:
			b.tokens -= take
			remaining -= take
			if blocked {
				b.waiting[c]--
				blocked = false
			}
		default:
			wait = time.Duration((need - b.tokens) / b.rate * float64(time.Second))
			if !blocked {
				b.waiting[c]++
				blocked = true
			}
		}
		if wait > 0 && wait < time.Millisecond {
			wait = time.Millisecond
		}
		b.mu.Unlock()
		if wait > 0 {
			t := time.NewTimer(wait)
			select {
			case <-t.C:
			case <-ctx.Done():
				t.Stop()
				return ctx.Err()
			}
		}
	}
	return nil
}
