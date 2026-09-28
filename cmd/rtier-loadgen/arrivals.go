package main

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"time"

	"rtier/internal/design"
)

// Arrivals yields inter-arrival gaps of an open-loop workload and which query to send.
type Arrivals interface {
	Next() (gap time.Duration, queryIndex int)
}

// step: from at (since the start of the run) the arrival rate is rate queries per second.
type step struct {
	at   time.Duration
	rate float64
}

// parseSteps turns "30:400,90:800" into the schedule after the initial rate: at 30 s the rate
// becomes 400 q/s, at 90 s 800 q/s. An offset without a unit is seconds ("90s", "2m" also
// work). Offsets must increase; rate 0 pauses the load. This is how a run raises the load past
// what the current nodes can serve -- the reason to scale out.
func parseSteps(s string, rate float64) ([]step, error) {
	steps := []step{{at: 0, rate: rate}}
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		at, r, ok := strings.Cut(part, ":")
		if !ok {
			return nil, fmt.Errorf("rate step %q: want <offset>:<rate>", part)
		}
		d, err := parseOffset(strings.TrimSpace(at))
		if err != nil {
			return nil, fmt.Errorf("rate step %q: %v", part, err)
		}
		v, err := strconv.ParseFloat(strings.TrimSpace(r), 64)
		if err != nil || v < 0 || math.IsInf(v, 0) {
			return nil, fmt.Errorf("rate step %q: rate must be >= 0", part)
		}
		if d <= steps[len(steps)-1].at {
			return nil, fmt.Errorf("rate step %q: offsets must increase and be > 0", part)
		}
		steps = append(steps, step{at: d, rate: v})
	}
	return steps, nil
}

func parseOffset(s string) (time.Duration, error) {
	if v, err := strconv.ParseFloat(s, 64); err == nil {
		return time.Duration(v * float64(time.Second)), nil
	}
	return time.ParseDuration(s)
}

// poisson: exponential gaps at the rate in force at the scheduled time; queries in round-robin
// order. Arrivals are memoryless, so a draw that would cross a step boundary is dropped and
// redrawn at the new rate.
type poisson struct {
	rng   *rand.Rand
	steps []step
	cur   int           // the step in force
	t     time.Duration // scheduled time of the last arrival
	nq    int
	i     int
}

// never ends the run: the main loop stops at the first gap past its duration.
const never = time.Duration(math.MaxInt64 / 4)

func (p *poisson) Next() (time.Duration, int) {
	start := p.t
	for {
		rate := p.steps[p.cur].rate
		next := p.cur + 1
		if rate <= 0 { // paused
			if next >= len(p.steps) {
				return never, 0
			}
			p.t, p.cur = p.steps[next].at, next
			continue
		}
		gap := time.Duration(p.rng.ExpFloat64() / rate * float64(time.Second))
		if next < len(p.steps) && p.t+gap >= p.steps[next].at {
			p.t, p.cur = p.steps[next].at, next
			continue
		}
		p.t += gap
		q := p.i % p.nq
		p.i++
		return p.t - start, q
	}
}

// newArrivals returns the arrival process named kind. steps is the rate schedule after the
// initial rate (parseSteps), empty for a constant rate.
//
// TODO(design) U11: the workload the proposal targets is SemDN-style bursty, task-coherent
// query streams from agents (bursts of related queries that hit a small set of lists). Its
// generator (or a trace to replay) is not designed yet; "semdn" is the placeholder.
func newArrivals(kind string, steps []step, nq int, seed int64) (Arrivals, error) {
	switch kind {
	case "", "poisson":
		if len(steps) == 0 || steps[0].rate <= 0 || nq <= 0 {
			return nil, fmt.Errorf("poisson arrivals need rate > 0 and queries")
		}
		return &poisson{rng: rand.New(rand.NewSource(seed)), steps: steps, nq: nq}, nil
	case "semdn":
		return nil, design.Undecided("U11", "SemDN-style bursty, task-coherent query streams")
	}
	return nil, fmt.Errorf("unknown arrival process %q", kind)
}
