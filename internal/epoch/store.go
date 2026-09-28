package epoch

import (
	"fmt"
	"sync"

	"rtier/internal/design"
)

// Store holds the authoritative current epoch. The flip from epoch e to e+1 is a single
// compare-and-swap: either every later reader sees e+1 or the flip did not happen.
type Store interface {
	Current() (*Table, error)
	// CompareAndSwap installs next if the current epoch is prev (0 = no epoch yet).
	// next.Epoch must be prev+1.
	CompareAndSwap(prev uint64, next *Table) error
}

// NewStore returns the store named kind: "memory" or "etcd".
func NewStore(kind string) (Store, error) {
	switch kind {
	case "", "memory":
		return NewMemStore(), nil
	case "etcd":
		return EtcdStore{}, nil
	}
	return nil, fmt.Errorf("epoch: unknown store %q", kind)
}

// MemStore is a controller-local store: enough for the single-machine prototype, where the
// controller is the only writer. It keeps every installed table for inspection.
type MemStore struct {
	mu      sync.Mutex
	history []*Table
}

func NewMemStore() *MemStore { return &MemStore{} }

func (s *MemStore) Current() (*Table, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.history) == 0 {
		return nil, nil
	}
	return s.history[len(s.history)-1], nil
}

func (s *MemStore) CompareAndSwap(prev uint64, next *Table) error {
	if next.Epoch != prev+1 {
		return fmt.Errorf("epoch: next epoch %d is not %d+1", next.Epoch, prev)
	}
	if err := next.Validate(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var cur uint64
	if n := len(s.history); n > 0 {
		cur = s.history[n-1].Epoch
	}
	if cur != prev {
		return fmt.Errorf("epoch: compare-and-swap lost: current epoch is %d, expected %d", cur, prev)
	}
	s.history = append(s.history, next)
	return nil
}

// History returns every installed table, oldest first.
func (s *MemStore) History() []*Table {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]*Table(nil), s.history...)
}

// EtcdStore is the placeholder for an etcd-backed store (a transaction comparing the epoch
// key's revision). TODO(design) U10.
type EtcdStore struct{}

func (EtcdStore) Current() (*Table, error) {
	return nil, design.Undecided("U10", "etcd epoch store")
}

func (EtcdStore) CompareAndSwap(uint64, *Table) error {
	return design.Undecided("U10", "etcd epoch store")
}
