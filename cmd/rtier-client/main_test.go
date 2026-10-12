package main

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"rtier/internal/protocol"
)

// TestPlanFileRoundTrip: a saved plan reads back as the same plan, one line per node in
// numeric order, and a misspelt key is an error rather than an empty field.
func TestPlanFileRoundTrip(t *testing.T) {
	p := &protocol.Plan{FromEpoch: 7, Owners: map[string][]int{"10": {4, 5}, "2": {0, 1, 2}, "3": {3, 6, 7}}}
	b := formatPlan(p)
	if lines := strings.Split(strings.TrimSpace(string(b)), "\n"); len(lines) != 8 ||
		!strings.Contains(lines[3], `"2": [0, 1, 2]`) || !strings.Contains(lines[5], `"10": [4, 5]`) {
		t.Fatalf("layout:\n%s", b)
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "plan.json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	back, err := readPlan(path)
	if err != nil || !reflect.DeepEqual(back, p) {
		t.Fatalf("read back %+v (%v), want %+v", back, err, p)
	}
	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte(`{"owner": {"1": [0]}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := readPlan(bad); err == nil {
		t.Fatal(`"owner" instead of "owners" accepted`)
	}
}
