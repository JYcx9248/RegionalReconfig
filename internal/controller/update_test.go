package controller

import (
	"context"
	"strings"
	"testing"

	"rtier/internal/config"
	"rtier/internal/protocol"
)

// TestUpdatesOffByDefault: without allow_updates the controller refuses an update batch's
// lease, so no batch can change the data of an experiment on a static dataset.
func TestUpdatesOffByDefault(t *testing.T) {
	c := &Controller{cfg: config.DefaultController()}
	if _, err := c.BeginUpdate(context.Background(), protocol.BeginUpdateReq{}); err == nil ||
		!strings.Contains(err.Error(), "allow_updates") {
		t.Fatalf("BeginUpdate with the default config: %v, want a refusal naming allow_updates", err)
	}
	if c.reconfigRunning.Load() {
		t.Fatal("the refused batch took the reconfiguration slot")
	}
}
