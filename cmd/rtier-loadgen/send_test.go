package main

import (
	"bufio"
	"context"
	"errors"
	"net"
	"sync/atomic"
	"testing"
	"time"

	"rtier/internal/frame"
	"rtier/internal/query"
)

// fakeEntry answers "unavailable" to its first `refuse` queries (a paused entry) and an
// empty result to the rest.
func fakeEntry(t *testing.T, refuse int64) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	var n atomic.Int64
	go func() {
		for {
			nc, err := ln.Accept()
			if err != nil {
				return
			}
			go func() {
				defer nc.Close()
				br := bufio.NewReader(nc)
				for {
					var f frame.Frame
					if err := frame.Read(br, &f); err != nil {
						return
					}
					resp := frame.Frame{Type: f.Type, Flags: frame.FlagResponse, ReqID: f.ReqID, Epoch: 1,
						Body: query.EncodeResult(nil)}
					if n.Add(1) <= refuse {
						resp.Status, resp.Body = query.StatusUnavailable, []byte("admission paused")
					}
					if err := frame.Write(nc, &resp); err != nil {
						return
					}
				}
			}()
		}
	}()
	return ln.Addr().String()
}

func sendTo(t *testing.T, addr string, until time.Time) (int, error, time.Duration) {
	t.Helper()
	ents := &entries{byAddr: map[string]*query.Client{}, conns: 2}
	ents.set([]string{addr})
	start := time.Now()
	_, attempts, err := send(context.Background(), ents,
		&query.Request{Vec: make([]byte, 8), Params: query.Params{K: 1, NProbe: 1, N: 1}}, until)
	return attempts, err, time.Since(start)
}

func TestSendGivesUpAfterThreeAttempts(t *testing.T) {
	attempts, err, _ := sendTo(t, fakeEntry(t, 5), time.Time{})
	if !errors.Is(err, query.ErrUnavailable) || attempts != 3 {
		t.Fatalf("got %d attempts, err %v; want 3 attempts, unavailable", attempts, err)
	}
}

func TestSendWaitsForAnswerUntilDeadline(t *testing.T) {
	// Refused 6 times, then answered: a client that waits gets the answer at attempt 7.
	attempts, err, _ := sendTo(t, fakeEntry(t, 6), time.Now().Add(5*time.Second))
	if err != nil || attempts != 7 {
		t.Fatalf("got %d attempts, err %v; want the answer at attempt 7", attempts, err)
	}
	// Never answered: it keeps trying until the deadline, then fails.
	attempts, err, took := sendTo(t, fakeEntry(t, 1<<30), time.Now().Add(200*time.Millisecond))
	if !errors.Is(err, query.ErrUnavailable) || attempts <= 3 || took < 200*time.Millisecond {
		t.Fatalf("got %d attempts in %v, err %v; want more than 3 attempts over at least 200ms, unavailable",
			attempts, took, err)
	}
}
