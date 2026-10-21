package agent

import (
	"context"
	"fmt"
	"sync"
	"time"

	"rtier/internal/transfer"
)

// graphPoll is how often a graph transfer looks at its data node's raw-vector traffic.
const graphPoll = 100 * time.Millisecond

// graphYield holds a navigation-graph transfer back while its node moves data (the decided part
// of U16, 2026-10-21). A new node needs posting lists, PQ codes and raw vectors to serve its
// partitions, and the graph only for the entry role, whose navigation -- the first step of a
// query -- any existing entry does meanwhile. So the graph goes last, at both ends of its
// transfer: it moves no byte while the agent moves segments or PQ codes, in (a staging) or out
// (writing the moved lists, the bucket's data class), or while the data node moves raw vectors,
// in or out, on demand or streamed, which no bucket paces (RAW_TRAFFIC). Unlike the bucket's
// background class it takes none of the bandwidth data leaves unused: it waits for the data to
// stop, paced or not. In the worst case -- queries fetching without pause -- a new node becomes
// an entry only once its fetching subsides, within the step timeout; the data flip never waits
// for it.
type graphYield struct {
	dataMoving func() bool                               // segments or PQ codes, in or out
	rawBytes   func(ctx context.Context) (uint64, error) // the data node's, sent plus received
	poll       time.Duration
	waited     func(time.Duration) // if not nil: how long a Wait held the graph back

	mu     sync.Mutex
	polled time.Time
	last   uint64
	moved  bool // between the last two polls
}

// Wait returns once neither data nor raw vectors are moving here.
func (y *graphYield) Wait(ctx context.Context) error {
	var start time.Time
	defer func() {
		if !start.IsZero() && y.waited != nil {
			y.waited(time.Since(start))
		}
	}()
	for {
		moved, err := y.rawMoved(ctx)
		if err != nil {
			return err
		}
		if !moved && !y.dataMoving() {
			return nil
		}
		if start.IsZero() {
			start = time.Now()
		}
		t := time.NewTimer(y.poll)
		select {
		case <-t.C:
		case <-ctx.Done():
			t.Stop()
			return ctx.Err()
		}
	}
}

// rawMoved reports whether the data node's raw-vector bytes changed between the last two polls.
// It asks the node at most once per interval, so the chunks of a transfer in between ask nobody.
func (y *graphYield) rawMoved(ctx context.Context) (bool, error) {
	y.mu.Lock()
	defer y.mu.Unlock()
	if !y.polled.IsZero() && time.Since(y.polled) < y.poll {
		return y.moved, nil
	}
	v, err := y.rawBytes(ctx)
	if err != nil {
		return false, fmt.Errorf("the data node's raw-vector traffic: %w", err)
	}
	y.moved = v != y.last
	y.last, y.polled = v, time.Now()
	return y.moved, nil
}

// dataMoving: a staging in flight (segments and PQ codes coming in), the moved lists being
// written as segments, or a data-class transfer going out.
func (a *Agent) dataMoving() bool {
	a.mu.Lock()
	staging := len(a.stagings) > 0
	a.mu.Unlock()
	return staging || a.saving.Load() > 0 || a.limiter.Active(transfer.Data) > 0
}
