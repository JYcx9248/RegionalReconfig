package e2e

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"rtier/internal/nodeclient"
	"rtier/internal/query"
)

// TestBootstrapFromSnapshot: an experiment's warmup snapshot (scripts/run_local.py, "Type":
// "warmup"). The initial nodes bootstrap from the index, each saves the presence bits of its raw
// vectors (RAW_SAVE) and its PQ codes (PQ_SAVE), and the cluster stops. A new cluster's initial
// nodes start on copies of those raw files (--raw-reuse), on the saved codes where they are, as
// the runners leave them (--pq-file), and on the index without its page file and PQ codes, so
// that a single vector or code read from the index would fail the deployment: they deploy the
// same placement, copy nothing, and answer exactly; a scale-out then runs as it does from
// bootstrapped nodes.
func TestBootstrapFromSnapshot(t *testing.T) {
	f := getFixture(t)
	dir := t.TempDir()
	raw := func(run string, i int) string { return filepath.Join(dir, run, fmt.Sprintf("n%d", i), "raw.pages") }
	pq := filepath.Join(dir, "warmup", "n%d", "pq.codes")
	const initial = 2
	warm := newCluster(t, f, "lazy", initial)
	for i := 1; i <= initial; i++ {
		if err := os.MkdirAll(filepath.Dir(raw("warmup", i)), 0o755); err != nil {
			t.Fatal(err)
		}
		warm.addNode(fmt.Sprintf("n%d", i), query.TwoPhase{}, "--raw-file", raw("warmup", i))
	}
	if err := warm.wait(warm.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("initial deployment: %v (%s)", err, warm.ctl.LastError())
	}
	want := warm.ctl.Status().Table.Placement
	saved, savedPQ := make([]uint64, initial), make([]uint64, initial)
	for i, a := range warm.agents {
		c := nodeclient.New(a.Info().NodeAddr, 1)
		n, err := c.RawSave(warm.ctx)
		codes, perr := c.PQSave(warm.ctx, fmt.Sprintf(pq, i+1))
		info, ierr := c.InfoSettled(warm.ctx)
		c.Close()
		if err != nil || perr != nil || ierr != nil {
			t.Fatalf("n%d: RAW_SAVE %v, PQ_SAVE %v, INFO %v", i+1, err, perr, ierr)
		}
		if n == 0 || n != info.Raw.Present || n != info.Raw.FromIndex {
			t.Fatalf("n%d saved %d raw vectors, holds %+v: want every vector it copied", i+1, n, info.Raw)
		}
		if codes == 0 || codes != info.PQ.Resident || codes != info.PQ.FromIndex {
			t.Fatalf("n%d saved %d PQ codes, holds %+v: want every code it copied", i+1, codes, info.PQ)
		}
		saved[i], savedPQ[i] = n, codes
	}
	warm.close()
	warm.stops = nil // stopped: its cleanup has nothing left to stop

	// The snapshot, copied as the runner restores one.
	for i := 1; i <= initial; i++ {
		if err := os.MkdirAll(filepath.Dir(raw("run", i)), 0o755); err != nil {
			t.Fatal(err)
		}
		for _, suffix := range []string{"", ".present"} {
			b, err := os.ReadFile(raw("warmup", i) + suffix)
			if err == nil {
				err = os.WriteFile(raw("run", i)+suffix, b, 0o644)
			}
			if err != nil {
				t.Fatal(err)
			}
		}
	}
	cl := newCluster(t, f, "lazy", initial)
	cl.initialIndex = f.indexNoBootstrap
	for i := 1; i <= initial; i++ {
		cl.addNode(fmt.Sprintf("n%d", i), query.TwoPhase{}, "--raw-file", raw("run", i), "--raw-reuse",
			"--pq-file", fmt.Sprintf(pq, i))
	}
	cl.addNode("n3", query.TwoPhase{}) // a spare: starts empty, joins at the scale-out
	if err := cl.wait(cl.ctl.Ready, 60*time.Second); err != nil {
		t.Fatalf("deployment on the snapshot: %v (%s)", err, cl.ctl.LastError())
	}
	if got := cl.ctl.Status().Table.Placement; !reflect.DeepEqual(got.Owners, want.Owners) {
		t.Fatalf("placement %v, the snapshot's %v", got.Owners, want.Owners)
	}
	for i, a := range cl.agents[:initial] {
		c := nodeclient.New(a.Info().NodeAddr, 1)
		info, err := c.InfoSettled(cl.ctx)
		c.Close()
		if err != nil {
			t.Fatal(err)
		}
		if r := info.Raw; r.Restored != saved[i] || r.Present != r.Restored || r.FromIndex != 0 {
			t.Errorf("n%d's raw vectors %+v: want the %d saved, none copied from the index", i+1, r, saved[i])
		}
		if c := info.PQ; c.Restored != savedPQ[i] || c.Resident != c.Restored || c.FromIndex != 0 {
			t.Errorf("n%d's PQ codes %+v: want the %d saved, none read from the index", i+1, c, savedPQ[i])
		}
	}
	checkResidency(t, cl)
	exactPass(t, cl, nil)
	rescale(t, cl, 3)
	checkResidency(t, cl)
	exactPass(t, cl, nil)
}
