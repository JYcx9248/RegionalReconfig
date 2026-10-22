package config

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// One config.yaml configures the controller and the agents: each takes its own keys and keeps its
// defaults for the keys the file leaves out; a key shared by both (the controller's address, the
// stage timeout) sets both; a key neither knows is refused.
func TestLoad(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	write := func(s string) {
		if err := os.WriteFile(path, []byte(s), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(`ControllerAddr: 10.0.0.1:7100
Protocol: copy-then-flip
StageTimeout: 2h
InitialNodes: 2
NewNodeRoles: [aggregator, data]
NodeConns: 4
RawStreams: 3
MetricsInterval: 500ms
SampleInterval: 1000000000
`)
	c := DefaultController()
	if err := Load(path, &c); err != nil {
		t.Fatal(err)
	}
	if c.Listen != "10.0.0.1:7100" || c.Protocol != "copy-then-flip" || c.StageTimeout.Duration != 2*time.Hour ||
		c.InitialNodes != 2 || !reflect.DeepEqual(c.NewNodeRoles, []string{"aggregator", "data"}) {
		t.Errorf("controller %+v", c)
	}
	if d := DefaultController(); c.APIListen != d.APIListen || c.Placement != d.Placement || c.CallTimeout != d.CallTimeout {
		t.Errorf("the keys the file leaves out lost their defaults: %+v", c)
	}
	a := DefaultAgent()
	a.Name = "n1" // a command-line value: the file has no key for it
	if err := Load(path, &a); err != nil {
		t.Fatal(err)
	}
	if a.Controller != "10.0.0.1:7100" || a.StepTimeout.Duration != 2*time.Hour || a.NodeConns != 4 ||
		a.RawStreams != 3 || a.MetricsInterval.Duration != 500*time.Millisecond ||
		a.SampleInterval.Duration != time.Second || a.Name != "n1" || a.Selector != DefaultAgent().Selector {
		t.Errorf("agent %+v", a)
	}

	write("Protocol: lazy\nProtocl: lazy\n")
	if err := Load(path, &c); err == nil || !strings.Contains(err.Error(), "Protocl") {
		t.Errorf("a misspelled key: %v", err)
	}
	write("Name: n1\n") // an agent's own, not the file's
	if err := Load(path, &a); err == nil {
		t.Error("a command-line field set from the file")
	}
	write("StageTimeout: soon\n")
	if err := Load(path, &c); err == nil {
		t.Error("a duration that does not parse")
	}
}

// Defaults has every key once, and loads back to the defaults of both processes.
func TestDefaults(t *testing.T) {
	b, err := Defaults()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	c, a := DefaultController(), DefaultAgent()
	if err := Load(path, &c); err != nil {
		t.Fatal(err)
	}
	if err := Load(path, &a); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(c, DefaultController()) || !reflect.DeepEqual(a, DefaultAgent()) {
		t.Errorf("defaults do not round-trip:\n%s", b)
	}
	lines := 0
	for _, l := range strings.Split(string(b), "\n") {
		if l != "" && !strings.HasPrefix(l, " ") {
			lines++
		}
	}
	if lines != len(Keys()) {
		t.Errorf("%d top-level keys written, %d known", lines, len(Keys()))
	}
}
