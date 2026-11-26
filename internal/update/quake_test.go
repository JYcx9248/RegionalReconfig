package update

import (
	"math"
	"reflect"
	"testing"
)

// A latency grid bilinear in n and k is what Quake's estimator interpolates and extrapolates
// exactly, inside the grid and outside it.
func TestScanLatencyEstimate(t *testing.T) {
	f := func(n, k int) float64 { return 100 + 2*float64(n) + 3*float64(k) + 0.01*float64(n)*float64(k) }
	e := &scanLatency{ns: quakeGridN, ks: quakeGridK}
	for _, n := range quakeGridN {
		var row []float64
		for _, k := range quakeGridK {
			row = append(row, f(n, k))
		}
		e.grid = append(e.grid, row)
	}
	for _, c := range [][2]int{{1, 1}, {3, 10}, {1000, 10}, {65536, 256}, {65536, 10}, {100000, 10}, {7, 300},
		{200000, 1000}, {16384, 64}} {
		if got, want := e.estimate(c[0], c[1]), f(c[0], c[1]); math.Abs(got-want) > 1e-6*want {
			t.Errorf("L(%d, %d) = %v, want %v", c[0], c[1], got, want)
		}
	}
	if e.estimate(0, 10) != 0 || e.estimate(10, 0) != 0 {
		t.Error("an empty scan costs nothing")
	}
}

// linear is L(n) = 10 + n ns at every k.
func linear() *quakeCost {
	e := &scanLatency{ns: quakeGridN, ks: quakeGridK}
	for _, n := range quakeGridN {
		row := make([]float64, len(quakeGridK))
		for j := range row {
			row[j] = 10 + float64(n)
		}
		e.grid = append(e.grid, row)
	}
	return &quakeCost{lat: e, alpha: 0.9, k: quakeK}
}

// Quake's cost formulas (maintenance_cost_estimator.cpp), worked by hand on L(n) = 10 + n.
func TestQuakeCostDeltas(t *testing.T) {
	c := linear()
	near := func(name string, got, want float64) {
		t.Helper()
		if math.Abs(got-want) > 1e-9*math.Max(1, math.Abs(want)) {
			t.Errorf("%s = %v, want %v", name, got, want)
		}
	}
	// One list more: L(101) - L(100) = 1; two halves at 2 * 0.9 * 0.5 * L(500), against 0.5 * L(1000).
	near("split", c.splitDelta(1000, 0.5, 100), 1+510*0.5*1.8-1010*0.5)
	// One list fewer: L(99) - L(100) = -1. Old: 99 lists at the scan fraction and L(1000), and the
	// list's own; new (50 < 100 lists): 50 lists one posting longer, 49 as they were, each at the
	// merged hit rate.
	merged := 0.02 + 0.01/99
	near("delete", c.deleteDelta(50, 0.01, 100, 0.02, 1000), -1+merged*(50*1011+49*1010)-(99*0.02*1010+0.01*60))
	// A list as long as there are lists: every other list grows by its share, rounded up.
	merged = 0.02 + 0.01/99
	near("delete long", c.deleteDelta(500, 0.01, 100, 0.02, 1000),
		-1+99*merged*(10+math.Ceil(1000+500.0/99))-(99*0.02*1010+0.01*510))
	near("delete of the last list", c.deleteDelta(50, 0.01, 1, 0.02, 1000), 0)
	// With the rejection, as Quake's code has it: the list's own cost added, each receiver grown by
	// the whole list at the sum of both hit rates.
	near("delete with rejection", c.deleteDeltaReassign(50, 0.01, 100, []int{1000, 2000}, []float64{0.1, 0.2}),
		-1+0.01*60+(0.11*1060-0.1*1010)+(0.21*2060-0.2*2010))
}

// perform_maintenance's decisions: a long, hot list splits; a short, cold one is deleted, unless
// the rejection finds its members' lists would cost more; by length alone with a maximum.
func TestQuakeDecide(t *testing.T) {
	lengths := map[uint32]int{1: 1000, 2: 100, 3: 100, 4: 2}
	hits := map[uint32]int{1: 80, 2: 10, 3: 10}
	p := DefaultQuakeParams()
	p.DeleteThresholdNS, p.SplitThresholdNS, p.DeleteRejection = 0.1, 1, false
	none := func(uint32) ([]uint32, error) { t.Fatal("no rejection asked"); return nil, nil }
	d, err := p.decide(linear(), lengths, hits, 100, 0.3, none)
	if err != nil || !reflect.DeepEqual(d.splits, []uint32{1}) || !reflect.DeepEqual(d.deletes, []uint32{4}) {
		t.Fatalf("decisions %+v (%v): want list 1 split and list 4 deleted", d, err)
	}
	// The rejection, for a list longer than the minimum: where its members would go decides.
	p.DeleteRejection, p.MinSize = true, 1
	for _, c := range []struct {
		targets  []uint32
		deleted  bool
		rejected int
	}{{[]uint32{2}, true, 0}, {[]uint32{1, 2, 3}, false, 1}} {
		asked := 0
		d, err := p.decide(linear(), lengths, hits, 100, 0.3, func(l uint32) ([]uint32, error) {
			asked++
			if l != 4 {
				t.Errorf("rejection asked for list %d", l)
			}
			return c.targets, nil
		})
		if err != nil || asked != 1 || (len(d.deletes) == 1) != c.deleted || d.rejected != c.rejected ||
			!reflect.DeepEqual(d.splits, []uint32{1}) {
			t.Errorf("targets %v: decisions %+v (%v), asked %d", c.targets, d, err, asked)
		}
	}
	// By length alone.
	p = DefaultQuakeParams()
	p.MaxSize = 500
	d, err = p.decide(nil, lengths, nil, 1, 0, none)
	if err != nil || !reflect.DeepEqual(d.splits, []uint32{1}) || !reflect.DeepEqual(d.deletes, []uint32{4}) {
		t.Fatalf("size-bound decisions %+v (%v)", d, err)
	}
}

func TestElementBytes(t *testing.T) {
	if got := elementBytes("uint8", []float32{-3, 0.5, 1.5, 2.5, 254.6, 300}); !reflect.DeepEqual(got, []byte{0, 0, 2, 2, 255, 255}) {
		t.Errorf("uint8: %v", got)
	}
	if got := elementBytes("int8", []float32{-200, -1.5, 0.4, 127.6}); !reflect.DeepEqual(got, []byte{0x80, 0xFE, 0, 0x7F}) {
		t.Errorf("int8: %v", got)
	}
	if got := elementBytes("float", []float32{1.5}); !reflect.DeepEqual(got, []byte{0, 0, 0xC0, 0x3F}) {
		t.Errorf("float: %v", got)
	}
}
