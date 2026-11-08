package agent

import (
	"context"
	"time"

	"rtier/internal/transfer"
)

// graphPoll is how often a graph transfer held back by a staging looks again.
const graphPoll = 100 * time.Millisecond

// graphYield holds a navigation-graph transfer back while its agent stages data (the decided part
// of U16). A new node needs posting lists and PQ codes to serve its partitions, and the graph
// only for the entry role, whose navigation -- the first step of a query -- any existing entry
// does meanwhile. So the graph goes after the staging, at both ends of its transfer: it moves no
// byte while the agent moves posting lists or PQ codes, in (a staging) or out (the bucket's data
// class), whether the bucket paces them or not.
//
// It does not wait for raw vectors (decided 2026-11-08; from 2026-10-21 it waited for those too).
// After a lazy flip the queries fetch them for as long as the load lasts: on Amazon 300M, 5-35 MB/s
// for the whole cluster, a few percent of one link, never still for 100 ms. A graph that waited
// for them came only after the load, 282-297 s after the trigger, and held the rescale open, and
// the scale-in behind it. Since 2026-11-04 the graph moves from memory to memory, and raw vectors
// are bounded by the disks and the read budgets, not by the network: the graph shares only the
// network with them, and a little CPU. It takes what the token bucket gives the background class
// -- all of it, unless TransferRateBytesPerSec paces the agents -- with no limit of its own.
type graphYield struct {
	dataMoving func() bool // posting lists or PQ codes, in or out
	poll       time.Duration
	waited     func(time.Duration) // if not nil: how long a Wait held the graph back
}

// Wait returns once no data is moving here.
func (y *graphYield) Wait(ctx context.Context) error {
	var start time.Time
	defer func() {
		if !start.IsZero() && y.waited != nil {
			y.waited(time.Since(start))
		}
	}()
	for y.dataMoving() {
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
	return nil
}

// dataMoving: a staging in flight (posting lists and PQ codes coming in), or a data-class
// transfer going out.
func (a *Agent) dataMoving() bool {
	a.mu.Lock()
	staging := len(a.stagings) > 0
	a.mu.Unlock()
	return staging || a.limiter.Active(transfer.Data) > 0
}
