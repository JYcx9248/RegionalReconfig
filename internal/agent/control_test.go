package agent

import (
	"reflect"
	"testing"
)

func TestRawStripes(t *testing.T) {
	for _, c := range []struct {
		n, batch, streams int
		want              [][2]int
	}{
		{0, 64, 4, nil},
		{10, 64, 4, [][2]int{{0, 10}}},                                       // one partial batch: one stream
		{128, 64, 8, [][2]int{{0, 64}, {64, 128}}},                           // fewer batches than streams
		{256, 64, 1, [][2]int{{0, 256}}},                                     // one stream: the old behavior
		{640, 64, 4, [][2]int{{0, 192}, {192, 384}, {384, 512}, {512, 640}}}, // 10 batches: 3, 3, 2, 2
		{650, 64, 4, [][2]int{{0, 192}, {192, 384}, {384, 576}, {576, 650}}}, // 11, the last one partial
	} {
		if got := rawStripes(c.n, c.batch, c.streams); !reflect.DeepEqual(got, c.want) {
			t.Errorf("rawStripes(%d, %d, %d) = %v, want %v", c.n, c.batch, c.streams, got, c.want)
		}
	}
	// Every split covers [0, n) in order with at most streams non-empty ranges of whole batches
	// (but the last), and their sizes differ by at most one batch.
	for _, batch := range []int{1, 7, 64, 256} {
		for _, streams := range []int{1, 2, 3, 4, 8} {
			for n := 0; n <= 3000; n += 37 {
				got := rawStripes(n, batch, streams)
				if len(got) > streams {
					t.Fatalf("rawStripes(%d, %d, %d): %d ranges", n, batch, streams, len(got))
				}
				lo, minB, maxB := 0, n+1, 0
				for i, r := range got {
					if r[0] != lo || r[1] <= r[0] || r[1] > n {
						t.Fatalf("rawStripes(%d, %d, %d) = %v: range %d", n, batch, streams, got, i)
					}
					if i < len(got)-1 && (r[1]-r[0])%batch != 0 {
						t.Fatalf("rawStripes(%d, %d, %d) = %v: range %d is not whole batches", n, batch, streams, got, i)
					}
					nb := (r[1] - r[0] + batch - 1) / batch
					minB, maxB = min(minB, nb), max(maxB, nb)
					lo = r[1]
				}
				if lo != n || (len(got) > 0 && maxB-minB > 1) {
					t.Fatalf("rawStripes(%d, %d, %d) = %v", n, batch, streams, got)
				}
			}
		}
	}
}
