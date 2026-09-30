package e2e

import (
	"encoding/csv"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"testing"
	"time"

	"rtier/internal/protocol"
)

// rescaleSpan is when one Rescale call ran, and the raw vectors that queries fetched on demand
// until rescale() returned (the reply does not count them).
type rescaleSpan struct {
	dataNodes        int
	start, end       time.Time
	reply            *protocol.RescaleReply
	fetched, fetches uint64 // vectors, and the round trips that fetched them
	fetchedBytes     uint64
}

// writeTimeline writes one run's latency timeline to dir, in the formats rtier-loadgen and
// run_local.py write (latency.csv, reconfigurations.json), so scripts/plot_run.py and
// scripts/compare_runs.py plot it like an experiment. TestReconfigUnderLoad calls it per
// protocol when RTIER_E2E_TIMELINE names a directory.
//
// The load is closed loop: a query's time is when its first attempt was sent (the next one
// waits for its answer), and its latency includes the retries of "unavailable" answers.
func writeTimeline(t *testing.T, dir string, st loadStats, spans []rescaleSpan) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	recs := append([]queryRecord(nil), st.records...)
	sort.Slice(recs, func(i, j int) bool { return recs[i].sent < recs[j].sent })
	f, err := os.Create(filepath.Join(dir, "latency.csv"))
	if err != nil {
		t.Fatal(err)
	}
	w := csv.NewWriter(f)
	w.Write([]string{"sched_ms", "latency_us", "status", "epoch", "recall", "attempts"})
	for _, r := range recs {
		status := "ok"
		if !r.ok {
			status = "error"
		}
		// recall stays empty: answers are checked against the single-node answer, not exact kNN.
		w.Write([]string{strconv.FormatFloat(float64(r.sent.Microseconds())/1000, 'f', 3, 64),
			strconv.FormatInt(r.latency.Microseconds(), 10), status,
			strconv.FormatUint(r.epoch, 10), "", strconv.Itoa(r.attempts)})
	}
	w.Flush()
	if err := w.Error(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	type entry struct {
		DataNodes  int                    `json:"data_nodes"`
		Trigger    float64                `json:"trigger_seconds"`
		Elapsed    float64                `json:"elapsed_seconds"`
		RawFetched uint64                 `json:"raw_fetched"`
		RawFetches uint64                 `json:"raw_fetches"`
		FetchedB   uint64                 `json:"raw_fetched_bytes"`
		Reply      *protocol.RescaleReply `json:"reply"`
	}
	var out []entry
	for _, s := range spans {
		out = append(out, entry{s.dataNodes, s.start.Sub(st.start).Seconds(), s.end.Sub(s.start).Seconds(),
			s.fetched, s.fetches, s.fetchedBytes, s.reply})
	}
	b, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "reconfigurations.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("latency timeline: %d queries, %d rescales -> %s", len(recs), len(spans), dir)
}
