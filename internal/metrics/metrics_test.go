package metrics

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestHistogramQuantiles(t *testing.T) {
	h := NewHistogram()
	for i := 1; i <= 1000; i++ {
		h.Record(time.Duration(i) * time.Microsecond)
	}
	s := h.Snapshot(true)
	if s.Count != 1000 || s.MaxUs != 1000 {
		t.Fatalf("summary %+v", s)
	}
	near := func(got, want float64) bool { return got >= want && got <= want*1.06 }
	if !near(s.P50Us, 500) || !near(s.P99Us, 990) || !near(s.P999Us, 999) {
		t.Fatalf("quantiles %+v", s)
	}
	if s2 := h.Snapshot(false); s2.Count != 0 {
		t.Fatal("reset")
	}
	h.Record(0)
	h.Record(10 * time.Minute) // beyond the last bucket: clamped
	if s := h.Snapshot(false); s.Count != 2 || s.P999Us != s.MaxUs {
		t.Fatalf("edges %+v", s)
	}
}

func TestRegistryAndSink(t *testing.T) {
	r := NewRegistry()
	r.Hist("entry.latency").Record(time.Millisecond)
	r.Counter("bulk.bytes").Add(4096)
	r.Gauge("partitions", func() float64 { return 3 })
	samples := r.Snapshot()
	got := map[string]float64{}
	for _, s := range samples {
		got[s.Type] = s.Value
	}
	if got["entry.latency.count"] != 1 || got["bulk.bytes"] != 4096 || got["partitions"] != 3 {
		t.Fatalf("samples %v", got)
	}
	if _, ok := got["entry.latency.p99_us"]; !ok {
		t.Fatal("missing p99")
	}

	path := filepath.Join(t.TempDir(), "m.jsonl")
	sink, err := OpenSink(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := sink.Write(Report{Node: "n1", Time: time.Now(), Samples: samples}); err != nil {
		t.Fatal(err)
	}
	sink.Event("controller", "reconfig.flip", 2)
	sink.Close()
	f, _ := os.Open(path)
	defer f.Close()
	n := 0
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var row map[string]any
		if err := json.Unmarshal(sc.Bytes(), &row); err != nil {
			t.Fatal(err)
		}
		n++
	}
	if n != len(samples)+1 {
		t.Fatalf("%d rows for %d samples", n, len(samples))
	}
}
