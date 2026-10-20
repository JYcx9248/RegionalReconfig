package e2e

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"rtier/internal/config"
	"rtier/internal/metrics"
	"rtier/internal/query"
)

// TestNodeDiagnosticsForwarded: a data node's stats log and its agent's process and host samples
// reach the controller, which stamps every record with its own clock on arrival (as Koala's
// metric collector does), so that a run spanning machines has one time axis.
func TestNodeDiagnosticsForwarded(t *testing.T) {
	f := getFixture(t)
	stats := t.TempDir()
	cl := newCluster(t, f, "lazy", 1, func(c *config.Controller) { c.StatsDir = stats })
	nodeDir := t.TempDir()
	cl.agentCfg = func(name string, c *config.Agent) {
		c.NodeStatsFile = filepath.Join(nodeDir, "stats.jsonl")
		c.SampleInterval = config.Duration{Duration: 100 * time.Millisecond}
	}
	cl.addNode("n1", query.TwoPhase{}, "--stats-file", filepath.Join(nodeDir, "stats.jsonl"), "--stats-ms", "100")
	if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("initial deployment: %v (%s)", err, cl.ctl.LastError())
	}
	read := func(name string) []map[string]any {
		fh, err := os.Open(filepath.Join(stats, name))
		if err != nil {
			return nil
		}
		defer fh.Close()
		var out []map[string]any
		sc := bufio.NewScanner(fh)
		sc.Buffer(make([]byte, 1<<20), 1<<24)
		for sc.Scan() {
			var r map[string]any
			if json.Unmarshal(sc.Bytes(), &r) == nil {
				out = append(out, r)
			}
		}
		return out
	}
	deadline := time.Now().Add(10 * time.Second)
	var st, cg []map[string]any
	for time.Now().Before(deadline) {
		st, cg = read("stats-n1.jsonl"), read("cgroups.jsonl")
		if len(st) >= 3 && len(cg) >= 6 {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if len(st) < 3 || len(cg) < 6 {
		t.Fatalf("forwarded %d stats lines and %d samples", len(st), len(cg))
	}
	now := metrics.Monotonic()
	for _, r := range st {
		tt, ok1 := r["t"].(float64)
		_, ok2 := r["node_t"].(float64)
		if !ok1 || !ok2 || tt > now || tt < now-60 {
			t.Fatalf("stats line %v: want the controller's t (now %.3f) and the node's node_t", r, now)
		}
	}
	seen := map[string]bool{}
	for _, r := range cg {
		seen[r["node"].(string)] = true
	}
	for _, want := range []string{"n1", "n1-agent", "n1-host"} {
		if !seen[want] {
			t.Errorf("no %s sample in %v", want, seen)
		}
	}
}
