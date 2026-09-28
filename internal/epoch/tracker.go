package epoch

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Tracker is a node's view of the epoch, with the in-flight bookkeeping for the grace period.
//
// Every query pins the table it routes with (Enter, at its entry) and releases it when done;
// the whole query, including an aggregation forwarded to another node, runs under that one
// epoch. After the controller installs epoch e+1 everywhere, WaitDrained(e) returns once no
// query routed with an epoch <= e is still running on this node; only then may old copies be
// reclaimed. This
// replaces Koala's in-band InflightBarrier (api/collector/keybyCollector.go,
// internal/supplier/supplier.go handleInflightBarrier), which needs FIFO data channels.
type Tracker struct {
	mu       sync.Mutex
	cur      *Table
	inflight map[uint64]int
	changed  chan struct{} // closed when an old epoch drains or a new one is installed
}

func NewTracker() *Tracker {
	return &Tracker{inflight: map[uint64]int{}, changed: make(chan struct{})}
}

// Install makes t the current table. Epochs only move forward; re-installing the current
// epoch is a no-op (so the controller may retry).
func (tr *Tracker) Install(t *Table) error {
	tr.mu.Lock()
	defer tr.mu.Unlock()
	if tr.cur != nil {
		if t.Epoch == tr.cur.Epoch {
			return nil
		}
		if t.Epoch < tr.cur.Epoch {
			return fmt.Errorf("epoch: refusing to install epoch %d over %d", t.Epoch, tr.cur.Epoch)
		}
	}
	tr.cur = t
	tr.notifyLocked()
	return nil
}

// Current returns the installed table (nil before the first install).
func (tr *Tracker) Current() *Table {
	tr.mu.Lock()
	defer tr.mu.Unlock()
	return tr.cur
}

// Enter pins the current table for one query. The returned function must be called exactly
// once when the query is done. Returns (nil, nil) before the first install.
func (tr *Tracker) Enter() (*Table, func()) {
	tr.mu.Lock()
	t := tr.cur
	if t == nil {
		tr.mu.Unlock()
		return nil, nil
	}
	e := t.Epoch
	tr.inflight[e]++
	tr.mu.Unlock()
	var once sync.Once
	return t, func() { once.Do(func() { tr.exit(e) }) }
}

// Pin counts one query that runs on this node under epoch e without routing with this node's
// own table: an aggregation forwarded by an entry, which pinned e with Enter and keeps it
// pinned until the answer is back. The entry's pin is the one the grace period relies on (the
// entry is a node of epoch e, so the controller waits for it); this one makes the forwarded
// work visible here as well, to InFlight, WaitDrained and WaitIdle. e may be older than the
// installed epoch (a node that is leaving) or newer, or there may be none yet (a node that
// joins, right after the flip).
func (tr *Tracker) Pin(e uint64) func() {
	tr.mu.Lock()
	tr.inflight[e]++
	tr.mu.Unlock()
	var once sync.Once
	return func() { once.Do(func() { tr.exit(e) }) }
}

func (tr *Tracker) exit(e uint64) {
	tr.mu.Lock()
	defer tr.mu.Unlock()
	tr.inflight[e]--
	if tr.inflight[e] <= 0 {
		delete(tr.inflight, e)
		if tr.cur != nil && e < tr.cur.Epoch {
			tr.notifyLocked() // an old epoch drained: wake WaitDrained
		}
	}
}

// InFlight returns the number of running queries pinned to epoch e.
func (tr *Tracker) InFlight(e uint64) int {
	tr.mu.Lock()
	defer tr.mu.Unlock()
	return tr.inflight[e]
}

// WaitDrained blocks until an epoch newer than e is installed and no query pinned to an
// epoch <= e is running.
func (tr *Tracker) WaitDrained(ctx context.Context, e uint64) error {
	for {
		tr.mu.Lock()
		done := tr.cur != nil && tr.cur.Epoch > e
		for pinned := range tr.inflight {
			if pinned <= e {
				done = false
			}
		}
		ch := tr.changed
		tr.mu.Unlock()
		if done {
			return nil
		}
		select {
		case <-ch:
		case <-ctx.Done():
			return fmt.Errorf("epoch: waiting for epoch %d to drain: %w", e, ctx.Err())
		}
	}
}

// WaitIdle blocks until no query is pinned to any epoch (used by the stop-and-copy baseline
// after admission is paused).
func (tr *Tracker) WaitIdle(ctx context.Context) error {
	for {
		tr.mu.Lock()
		idle := len(tr.inflight) == 0
		ch := tr.changed
		tr.mu.Unlock()
		if idle {
			return nil
		}
		select {
		case <-ch:
		case <-time.After(5 * time.Millisecond): // current-epoch exits do not notify
		case <-ctx.Done():
			return fmt.Errorf("epoch: waiting for in-flight queries: %w", ctx.Err())
		}
	}
}

func (tr *Tracker) notifyLocked() {
	close(tr.changed)
	tr.changed = make(chan struct{})
}
