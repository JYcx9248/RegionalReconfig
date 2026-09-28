package transfer

import (
	"fmt"
	"time"

	"rtier/internal/design"
)

// Observation is what a rate controller would look at.
type Observation struct {
	ForegroundP99 time.Duration // recent query tail latency on this node
	SLO           time.Duration
	BytesLeft     int64
}

// RateAdapter chooses the background transfer rate.
type RateAdapter interface {
	Name() string
	Rate(obs Observation) (float64, error)
}

// FixedRate always returns the same rate (bytes per second; 0 = unlimited).
type FixedRate float64

func (FixedRate) Name() string                        { return "fixed" }
func (r FixedRate) Rate(Observation) (float64, error) { return float64(r), nil }

// AdaptiveRate is the placeholder for adapting the rate to foreground latency.
// TODO(design) U8.
type AdaptiveRate struct{}

func (AdaptiveRate) Name() string { return "adaptive" }
func (AdaptiveRate) Rate(Observation) (float64, error) {
	return 0, design.Undecided("U8", "adaptive transfer rate")
}

// NewAdapter returns the adapter named kind ("fixed" or "adaptive").
func NewAdapter(kind string, fixed float64) (RateAdapter, error) {
	switch kind {
	case "", "fixed":
		return FixedRate(fixed), nil
	case "adaptive":
		return AdaptiveRate{}, nil
	}
	return nil, fmt.Errorf("transfer: unknown rate adapter %q", kind)
}
