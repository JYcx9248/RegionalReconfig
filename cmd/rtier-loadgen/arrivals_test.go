package main

import (
	"errors"
	"testing"
	"time"

	"rtier/internal/design"
)

func TestParseSteps(t *testing.T) {
	got, err := parseSteps("30:400, 1m:800", 100)
	if err != nil {
		t.Fatal(err)
	}
	want := []step{{0, 100}, {30 * time.Second, 400}, {time.Minute, 800}}
	if len(got) != len(want) {
		t.Fatalf("got %v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("step %d: got %v want %v", i, got[i], want[i])
		}
	}
	if s, err := parseSteps("", 100); err != nil || len(s) != 1 {
		t.Fatalf("no steps: %v %v", s, err)
	}
	for _, bad := range []string{"30", "30:-1", "60:100,30:200", "0:100", "x:100"} {
		if _, err := parseSteps(bad, 100); err == nil {
			t.Errorf("parseSteps(%q) accepted", bad)
		}
	}
}

// The rate in force is the one of the step covering the scheduled time, and a pause holds the
// load until the next step.
func TestPoissonSteps(t *testing.T) {
	steps, err := parseSteps("1:0,2:2000", 500)
	if err != nil {
		t.Fatal(err)
	}
	arr, err := newArrivals("poisson", steps, 8, 1)
	if err != nil {
		t.Fatal(err)
	}
	var at time.Duration
	counts := [3]int{} // arrivals in [0,1), [1,2), [2,3) seconds
	for i := 0; i < 4000; i++ {
		gap, qi := arr.Next()
		at += gap
		if at >= 3*time.Second {
			break
		}
		if qi < 0 || qi >= 8 {
			t.Fatalf("query index %d", qi)
		}
		counts[at/time.Second]++
	}
	if counts[0] < 400 || counts[0] > 600 {
		t.Errorf("first second: %d arrivals, want about 500", counts[0])
	}
	if counts[1] != 0 {
		t.Errorf("paused second: %d arrivals, want none", counts[1])
	}
	if counts[2] < 1800 || counts[2] > 2200 {
		t.Errorf("third second: %d arrivals, want about 2000", counts[2])
	}
}

func TestSemdnStillUndecided(t *testing.T) {
	steps, _ := parseSteps("", 100)
	if _, err := newArrivals("semdn", steps, 4, 1); !errors.Is(err, design.ErrUndecided) {
		t.Fatalf("semdn: %v", err)
	}
}
