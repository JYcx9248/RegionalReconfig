package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"rtier/internal/ctrl"
	"rtier/internal/epoch"
	"rtier/internal/frame"
	"rtier/internal/placement"
	"rtier/internal/protocol"
	"rtier/internal/query"
)

// fakeEntry answers "unavailable" to its first `refuse` queries (a paused entry) and an
// empty result to the rest, under epoch 1.
func fakeEntry(t *testing.T, refuse int64) string {
	return fakeEntryAt(t, refuse, 1)
}

// fakeEntryAt is fakeEntry with its replies under epoch ep.
func fakeEntryAt(t *testing.T, refuse int64, ep uint64) string {
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
					resp := frame.Frame{Type: f.Type, Flags: frame.FlagResponse, ReqID: f.ReqID, Epoch: ep,
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
	ents.set([]string{addr}, 0)
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

// fakeController answers Status with an epoch table of epoch ep whose entries are at addrs,
// and counts the calls.
func fakeController(t *testing.T, ep uint64, addrs ...string) (string, *atomic.Int64) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ln.Close() })
	tbl := &epoch.Table{Epoch: ep, Nodes: map[placement.NodeID]epoch.NodeInfo{}}
	for i, a := range addrs {
		id := placement.NodeID(i + 1)
		tbl.Nodes[id] = epoch.NodeInfo{ID: id, QueryAddr: a}
		tbl.Entries = append(tbl.Entries, id)
	}
	var reads atomic.Int64
	go ctrl.Serve(ln, func(c *ctrl.Conn) {
		c.Handle(protocol.StatusA, func(context.Context, json.RawMessage) (any, error) {
			reads.Add(1)
			return protocol.ClusterStatus{Table: tbl}, nil
		})
	})
	return ln.Addr().String(), &reads
}

// staleList is a client whose list, of epoch 1, has only an entry that left the entry set in
// epoch 2 (it refuses every query under epoch 2); the controller has epoch 2's entry.
func staleList(t *testing.T) (*entries, *atomic.Int64) {
	t.Helper()
	api, reads := fakeController(t, 2, fakeEntryAt(t, 0, 2))
	ents := &entries{byAddr: map[string]*query.Client{}, conns: 64, api: api}
	ents.set([]string{fakeEntryAt(t, 1<<30, 2)}, 1)
	return ents, reads
}

var testReq = &query.Request{Vec: make([]byte, 8), Params: query.Params{K: 1, NProbe: 1, N: 1}}

func TestSendCatchesUpWithAStaleList(t *testing.T) {
	ents, reads := staleList(t)
	_, attempts, err := send(context.Background(), ents, testReq, time.Time{})
	if err != nil || attempts != 2 {
		t.Fatalf("got %d attempts, err %v; want the answer at attempt 2, from epoch 2's entry", attempts, err)
	}
	if reads.Load() != 1 || ents.stale.Load() != 1 || ents.listEpoch() != 2 {
		t.Fatalf("%d controller reads, %d stale, list of epoch %d; want 1, 1, 2",
			reads.Load(), ents.stale.Load(), ents.listEpoch())
	}
}

func TestConcurrentSendsShareOneRead(t *testing.T) {
	// Every query of a burst meets the stale list; one read of the controller serves them all.
	ents, reads := staleList(t)
	var wg sync.WaitGroup
	var failed atomic.Int64
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, _, err := send(context.Background(), ents, testReq, time.Time{}); err != nil {
				failed.Add(1)
			}
		}()
	}
	wg.Wait()
	if failed.Load() != 0 || reads.Load() != 1 {
		t.Fatalf("%d queries failed, %d controller reads; want 0 and 1", failed.Load(), reads.Load())
	}
}

func TestSendDoesNotReadUnderTheListsEpoch(t *testing.T) {
	// A paused entry refuses under the list's own epoch: a fresh list would be the same.
	api, reads := fakeController(t, 1)
	ents := &entries{byAddr: map[string]*query.Client{}, conns: 2, api: api}
	ents.set([]string{fakeEntryAt(t, 1<<30, 1)}, 1)
	_, attempts, err := send(context.Background(), ents, testReq, time.Time{})
	if !errors.Is(err, query.ErrUnavailable) || attempts != 3 || reads.Load() != 0 {
		t.Fatalf("got %d attempts, err %v, %d controller reads; want 3, unavailable, 0", attempts, err, reads.Load())
	}
}

func TestSetKeepsTheLaterList(t *testing.T) {
	// A read that ended after a later one does not bring an older list back.
	ents := &entries{byAddr: map[string]*query.Client{}, conns: 1}
	ents.set([]string{"127.0.0.1:1"}, 5)
	ents.set([]string{"127.0.0.1:2"}, 4)
	if ents.listEpoch() != 5 || len(ents.clients) != 1 || ents.byAddr["127.0.0.1:1"] != ents.clients[0] {
		t.Fatalf("list of epoch %d; want epoch 5's", ents.listEpoch())
	}
}
