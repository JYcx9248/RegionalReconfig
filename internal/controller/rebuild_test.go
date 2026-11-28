package controller

import (
	"errors"
	"runtime"
	"sync/atomic"
	"testing"
)

// TestEachPartition: placeGeneration's partitions, many more than its workers, each run once,
// and a failure is returned; after one, a worker starts no other partition (every call failing,
// at most one call per worker).
func TestEachPartition(t *testing.T) {
	const np = 1000
	calls := make([]atomic.Int32, np)
	if err := eachPartition(np, func(p int) error {
		calls[p].Add(1)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	for p := range calls {
		if n := calls[p].Load(); n != 1 {
			t.Fatalf("partition %d ran %d times", p, n)
		}
	}
	if err := eachPartition(0, func(int) error { return errors.New("no partition") }); err != nil {
		t.Fatalf("no partitions: %v", err)
	}
	bad := errors.New("unreadable segment")
	var ran atomic.Int32
	err := eachPartition(np, func(int) error {
		ran.Add(1)
		return bad
	})
	if workers := max(1, runtime.NumCPU()/2); !errors.Is(err, bad) || int(ran.Load()) > workers {
		t.Fatalf("every partition failing: %d calls on %d workers, error %v", ran.Load(), workers, err)
	}
}
