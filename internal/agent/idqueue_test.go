package agent

import (
	"slices"
	"sync"
	"testing"
	"time"
)

// An idQueue hands the batches out in order, never makes push wait, holds a worker until there
// is a batch or the queue is closed, and lets the workers take what is left after close.
func TestIDQueue(t *testing.T) {
	q := newIDQueue()
	for i := range 1000 { // far more than any worker takes meanwhile: push never waits
		q.push([]uint32{uint32(i)})
	}
	for i := range 3 {
		b, ok := q.pop()
		if !ok || len(b) != 1 || b[0] != uint32(i) {
			t.Fatalf("pop %d: %v %v", i, b, ok)
		}
	}

	got := make(chan uint32, 2000)
	var wg sync.WaitGroup
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				b, ok := q.pop()
				if !ok {
					return
				}
				got <- b[0]
			}
		}()
	}
	time.Sleep(10 * time.Millisecond) // the workers empty the queue, then wait
	for i := 1000; i < 1500; i++ {
		q.push([]uint32{uint32(i)})
	}
	q.close()
	wg.Wait()
	close(got)
	var all []uint32
	for v := range got {
		all = append(all, v)
	}
	slices.Sort(all)
	if len(all) != 1497 || all[0] != 3 || all[len(all)-1] != 1499 {
		t.Fatalf("the workers took %d batches, %v...%v", len(all), all[:1], all[len(all)-1:])
	}
	if _, ok := q.pop(); ok {
		t.Fatal("pop after close and empty: a batch")
	}
}
