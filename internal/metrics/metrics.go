// Package metrics collects per-node metrics and ships them to the controller.
//
// It follows Koala's pipeline (metric/metric.go reports every MetricsInterval to the
// coordinator, coordinator/metricCollectorService.go stores rows of (operator_id, timestamp,
// metric_type, metric_value)) with two changes: latencies are histograms, so tails
// (p99, p99.9) are reported instead of Koala's per-interval averages, and the sink writes
// JSON lines (scripts/metrics_to_sqlite.py loads them into Koala's SQLite table layout).
package metrics

import (
	"encoding/json"
	"math"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// Histogram of durations with log-spaced buckets (5% wide) from 1 µs to ~100 s.
type Histogram struct {
	mu     sync.Mutex
	counts []uint64
	n      uint64
	sumUs  float64
	maxUs  float64
}

const (
	histRatio   = 1.05
	histBuckets = 380 // 1.05^380 µs ≈ 1.1e8 µs
)

var logRatio = math.Log(histRatio)

func NewHistogram() *Histogram { return &Histogram{counts: make([]uint64, histBuckets)} }

// Record adds one observation.
func (h *Histogram) Record(d time.Duration) {
	us := float64(d) / float64(time.Microsecond)
	i := 0
	if us > 1 {
		i = min(int(math.Log(us)/logRatio)+1, histBuckets-1)
	}
	h.mu.Lock()
	h.counts[i]++
	h.n++
	h.sumUs += us
	h.maxUs = max(h.maxUs, us)
	h.mu.Unlock()
}

// Summary of a histogram over one interval.
type Summary struct {
	Count                        uint64
	MeanUs, P50Us, P99Us, P999Us float64
	MaxUs                        float64
}

// Snapshot summarizes the histogram and, if reset, clears it.
func (h *Histogram) Snapshot(reset bool) Summary {
	h.mu.Lock()
	defer h.mu.Unlock()
	s := Summary{Count: h.n, MaxUs: h.maxUs}
	if h.n > 0 {
		s.MeanUs = h.sumUs / float64(h.n)
		s.P50Us = h.quantileLocked(0.50)
		s.P99Us = h.quantileLocked(0.99)
		s.P999Us = h.quantileLocked(0.999)
	}
	if reset {
		for i := range h.counts {
			h.counts[i] = 0
		}
		h.n, h.sumUs, h.maxUs = 0, 0, 0
	}
	return s
}

// quantileLocked returns the upper bound of the bucket holding quantile q.
func (h *Histogram) quantileLocked(q float64) float64 {
	rank := uint64(math.Ceil(q * float64(h.n)))
	var acc uint64
	for i, c := range h.counts {
		acc += c
		if acc >= rank {
			if i == histBuckets-1 { // overflow bucket
				return h.maxUs
			}
			return math.Min(math.Pow(histRatio, float64(i)), h.maxUs)
		}
	}
	return h.maxUs
}

// Sample is one metric value.
type Sample struct {
	Type  string  `json:"type"`
	Value float64 `json:"value"`
}

// Report is what a node sends every interval.
type Report struct {
	Node    string    `json:"node"`
	Time    time.Time `json:"time"`
	Samples []Sample  `json:"samples"`
}

// Registry holds a node's metrics.
type Registry struct {
	mu       sync.Mutex
	hists    map[string]*Histogram
	counters map[string]*atomic.Int64
	last     map[string]int64
	gauges   map[string]func() float64
	lastSnap time.Time
}

func NewRegistry() *Registry {
	return &Registry{
		hists:    map[string]*Histogram{},
		counters: map[string]*atomic.Int64{},
		last:     map[string]int64{},
		gauges:   map[string]func() float64{},
		lastSnap: time.Now(),
	}
}

// Hist returns (creating) the histogram name.
func (r *Registry) Hist(name string) *Histogram {
	r.mu.Lock()
	defer r.mu.Unlock()
	h, ok := r.hists[name]
	if !ok {
		h = NewHistogram()
		r.hists[name] = h
	}
	return h
}

// Counter returns (creating) the counter name.
func (r *Registry) Counter(name string) *atomic.Int64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	c, ok := r.counters[name]
	if !ok {
		c = &atomic.Int64{}
		r.counters[name] = c
	}
	return c
}

// Gauge registers a function read at every snapshot.
func (r *Registry) Gauge(name string, f func() float64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.gauges[name] = f
}

// Snapshot returns the interval's samples: for each histogram its count, rate, mean, p50,
// p99, p99.9 and max; for each counter its total and per-second rate; every gauge. It resets
// the histograms.
func (r *Registry) Snapshot() []Sample {
	r.mu.Lock()
	defer r.mu.Unlock()
	now := time.Now()
	secs := math.Max(now.Sub(r.lastSnap).Seconds(), 1e-9)
	r.lastSnap = now
	var out []Sample
	for name, h := range r.hists {
		s := h.Snapshot(true)
		out = append(out,
			Sample{name + ".count", float64(s.Count)},
			Sample{name + ".per_sec", float64(s.Count) / secs})
		if s.Count > 0 {
			out = append(out,
				Sample{name + ".mean_us", s.MeanUs}, Sample{name + ".p50_us", s.P50Us},
				Sample{name + ".p99_us", s.P99Us}, Sample{name + ".p999_us", s.P999Us},
				Sample{name + ".max_us", s.MaxUs})
		}
	}
	for name, c := range r.counters {
		v := c.Load()
		out = append(out, Sample{name, float64(v)}, Sample{name + ".per_sec", float64(v-r.last[name]) / secs})
		r.last[name] = v
	}
	for name, f := range r.gauges {
		out = append(out, Sample{name, f()})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Type < out[j].Type })
	return out
}

// Sink appends reports to a JSON-lines file, one row per sample:
// {"ts": ..., "node": ..., "type": ..., "value": ...} (Koala's metrics table as JSON).
type Sink struct {
	mu  sync.Mutex
	f   *os.File
	enc *json.Encoder
}

// OpenSink opens (appends to) path.
func OpenSink(path string) (*Sink, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, err
	}
	return &Sink{f: f, enc: json.NewEncoder(f)}, nil
}

type row struct {
	TS    string  `json:"ts"`
	Node  string  `json:"node"`
	Type  string  `json:"type"`
	Value float64 `json:"value"`
}

// Write stores one report.
func (s *Sink) Write(r Report) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	ts := r.Time.UTC().Format(time.RFC3339Nano)
	for _, x := range r.Samples {
		if math.IsNaN(x.Value) || math.IsInf(x.Value, 0) {
			continue
		}
		if err := s.enc.Encode(row{ts, r.Node, x.Type, x.Value}); err != nil {
			return err
		}
	}
	return nil
}

// Event records a one-off event (e.g. "reconfig.flip") as a sample with value 1.
func (s *Sink) Event(node, typ string, value float64) error {
	return s.Write(Report{Node: node, Time: time.Now(), Samples: []Sample{{typ, value}}})
}

// Close flushes and closes the file.
func (s *Sink) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.f.Close()
}
