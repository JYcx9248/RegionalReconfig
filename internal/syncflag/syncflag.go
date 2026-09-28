// Package syncflag is a one-shot flag that goroutines can wait on.
//
// Adapted from Koala (internal/syncflag/syncflag.go, Apache-2.0): Signal and Reset now also
// manage a channel so that WaitContext can give up when a context ends.
package syncflag

import (
	"context"
	"sync"
	"sync/atomic"
)

// SyncFlag is used for synchronization and signaling.
type SyncFlag struct {
	// 0 = not signaled (blocked), 1 = signaled (unblocked)
	flag int32
	// Condition variable for waiting and signaling
	cond *sync.Cond
	ch   chan struct{} // closed on Signal; recreated on Reset
	mu   sync.Mutex
}

func NewSyncFlag() *SyncFlag {
	return &SyncFlag{cond: sync.NewCond(&sync.Mutex{}), ch: make(chan struct{})}
}

// Wait blocks until it is signaled.
func (sf *SyncFlag) Wait() {
	sf.cond.L.Lock()
	defer sf.cond.L.Unlock()
	for atomic.LoadInt32(&sf.flag) == 0 {
		sf.cond.Wait()
	}
}

// WaitContext blocks until it is signaled or ctx ends.
func (sf *SyncFlag) WaitContext(ctx context.Context) error {
	sf.mu.Lock()
	ch := sf.ch
	sf.mu.Unlock()
	if !sf.IsBlocked() {
		return nil
	}
	select {
	case <-ch:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// IsBlocked is a quick check whether the flag is still unsignaled.
func (sf *SyncFlag) IsBlocked() bool { return atomic.LoadInt32(&sf.flag) == 0 }

// Signal sets the flag and wakes up all waiting goroutines.
func (sf *SyncFlag) Signal() {
	sf.mu.Lock()
	if atomic.LoadInt32(&sf.flag) == 0 {
		atomic.StoreInt32(&sf.flag, 1)
		close(sf.ch)
	}
	sf.mu.Unlock()
	sf.cond.L.Lock()
	sf.cond.Broadcast()
	sf.cond.L.Unlock()
}

// Reset sets the flag back to the unsignaled state.
func (sf *SyncFlag) Reset() {
	sf.mu.Lock()
	if atomic.LoadInt32(&sf.flag) == 1 {
		atomic.StoreInt32(&sf.flag, 0)
		sf.ch = make(chan struct{})
	}
	sf.mu.Unlock()
}
