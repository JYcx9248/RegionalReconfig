package epoch

import (
	"context"
	"errors"
	"testing"
	"time"

	"rtier/internal/design"
	"rtier/internal/placement"
)

func table(e uint64) *Table {
	return &Table{
		Epoch:     e,
		Placement: placement.Table{Owners: []placement.NodeID{1, 2}},
		Nodes: map[placement.NodeID]NodeInfo{
			1: {ID: 1, NodeAddr: "a"}, 2: {ID: 2, NodeAddr: "b"},
		},
		Entries: []placement.NodeID{1},
	}
}

func TestMemStore(t *testing.T) {
	s := NewMemStore()
	if cur, _ := s.Current(); cur != nil {
		t.Fatal("empty store has a table")
	}
	if err := s.CompareAndSwap(0, table(1)); err != nil {
		t.Fatal(err)
	}
	if err := s.CompareAndSwap(0, table(1)); err == nil {
		t.Fatal("stale compare-and-swap succeeded")
	}
	if err := s.CompareAndSwap(1, table(3)); err == nil {
		t.Fatal("epoch skip accepted")
	}
	bad := table(2)
	bad.Entries = []placement.NodeID{9}
	if err := s.CompareAndSwap(1, bad); err == nil {
		t.Fatal("table with unknown node accepted")
	}
	if err := s.CompareAndSwap(1, table(2)); err != nil {
		t.Fatal(err)
	}
	if cur, _ := s.Current(); cur.Epoch != 2 || len(s.History()) != 2 {
		t.Fatal("history")
	}
	if _, err := (EtcdStore{}).Current(); !errors.Is(err, design.ErrUndecided) {
		t.Fatal("etcd store should be a placeholder")
	}
}

func TestTrackerGracePeriod(t *testing.T) {
	tr := NewTracker()
	if tb, _ := tr.Enter(); tb != nil {
		t.Fatal("Enter before install")
	}
	_ = tr.Install(table(1))
	t1, exit1 := tr.Enter()
	if t1.Epoch != 1 || tr.InFlight(1) != 1 {
		t.Fatal("pin")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if err := tr.WaitDrained(ctx, 1); err == nil {
		t.Fatal("drained before the next epoch was installed")
	}

	_ = tr.Install(table(2))
	if err := tr.Install(table(1)); err == nil {
		t.Fatal("epoch went backwards")
	}
	t2, exit2 := tr.Enter()
	if t2.Epoch != 2 {
		t.Fatal("new queries must use the new epoch")
	}
	done := make(chan error, 1)
	go func() { done <- tr.WaitDrained(context.Background(), 1) }()
	select {
	case <-done:
		t.Fatal("drained while an epoch-1 query is running")
	case <-time.After(50 * time.Millisecond):
	}
	exit1()
	exit1() // idempotent
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("WaitDrained did not return")
	}
	exit2()
	if tr.InFlight(1) != 0 || tr.InFlight(2) != 0 {
		t.Fatal("counts")
	}
}

// TestTrackerPin: a forwarded aggregation counts under the entry's epoch, whatever this node
// has installed -- nothing yet, or a newer epoch -- and holds WaitDrained and WaitIdle open.
func TestTrackerPin(t *testing.T) {
	tr := NewTracker()
	release := tr.Pin(1) // a node that joins: epoch 1 is not installed here yet
	if tr.InFlight(1) != 1 {
		t.Fatal("pin before the first install")
	}
	release()
	release() // idempotent
	if tr.InFlight(1) != 0 {
		t.Fatal("release")
	}

	_ = tr.Install(table(1))
	_ = tr.Install(table(2))
	release = tr.Pin(1) // a node that is leaving: it already installed epoch 2
	short := func() context.Context {
		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		t.Cleanup(cancel)
		return ctx
	}
	if err := tr.WaitDrained(short(), 1); err == nil {
		t.Fatal("epoch 1 drained while a forwarded epoch-1 aggregation is running")
	}
	if err := tr.WaitIdle(short()); err == nil {
		t.Fatal("idle while a forwarded aggregation is running")
	}
	done := make(chan error, 1)
	go func() { done <- tr.WaitDrained(context.Background(), 1) }()
	release()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("WaitDrained did not return after the release")
	}
	cur, exit := tr.Enter()
	exit()
	if cur.Epoch != 2 {
		t.Fatal("Pin must not change the table new queries route with")
	}
}
