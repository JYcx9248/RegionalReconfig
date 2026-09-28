package agent

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"net"
	"time"

	"rtier/internal/epoch"
	"rtier/internal/frame"
	"rtier/internal/query"
)

// serveQueries serves the query port: OpQuery from clients (this node as entry) and
// OpAggregate from entries (this node as aggregator). One request at a time per connection.
func (a *Agent) serveQueries(ctx context.Context, ln net.Listener) {
	for {
		c, err := ln.Accept()
		if err != nil {
			return
		}
		go func() {
			defer c.Close()
			br := bufio.NewReader(c)
			bw := bufio.NewWriter(c)
			for {
				var f frame.Frame
				if err := frame.Read(br, &f); err != nil {
					return
				}
				resp := a.handleQueryFrame(ctx, &f)
				if frame.Write(bw, &resp) != nil || bw.Flush() != nil {
					return
				}
			}
		}()
	}
}

func (a *Agent) handleQueryFrame(ctx context.Context, f *frame.Frame) frame.Frame {
	resp := frame.Frame{Type: f.Type, Flags: frame.FlagResponse, ReqID: f.ReqID}
	var (
		res query.Result
		err error
	)
	badRequest := func(err error) frame.Frame {
		resp.Status = query.StatusBadRequest
		resp.Body = []byte(err.Error())
		return resp
	}
	switch f.Type {
	case query.OpQuery:
		req, derr := query.DecodeRequest(f.Body)
		if derr != nil {
			return badRequest(derr)
		}
		res, err = a.entry(ctx, req)
	case query.OpAggregate:
		req, groups, derr := query.DecodeAggregate(f.Body)
		if derr == nil && f.Epoch == 0 {
			derr = errors.New("forwarded query without the epoch its entry pinned")
		}
		if derr != nil {
			return badRequest(derr)
		}
		res, err = a.forwarded(ctx, f.Epoch, req, groups)
	default:
		return badRequest(fmt.Errorf("unknown operation 0x%02x", f.Type))
	}
	resp.Epoch = res.Epoch
	if err != nil {
		resp.Status = query.StatusOf(err)
		resp.Body = []byte(err.Error())
		return resp
	}
	resp.Body = query.EncodeResult(res.Candidates)
	return resp
}

// entry handles a client query: navigate on the local graph, work out which nodes own the
// probed lists, then aggregate here or on the node the selector picks among them (U6).
//
// The epoch pinned here is the query's only one: routing, the choice of aggregator and the
// aggregation all use it, and the pin lasts until the answer is back, a forwarded aggregation
// included. It is taken before the admission and role checks. With that order, once the
// controller has paused admission, WaitIdle sees every query that got past the check (the
// stop-and-copy baseline really stops), and a node that has stopped being an entry cannot
// start a query under an epoch nobody waits for.
func (a *Agent) entry(ctx context.Context, req *query.Request) (query.Result, error) {
	t, exit := a.tracker.Enter()
	if t == nil {
		return query.Result{}, query.ErrNoEpoch
	}
	defer exit()
	if a.paused.Load() {
		return query.Result{Epoch: t.Epoch}, fmt.Errorf("%w: admission paused", query.ErrUnavailable)
	}
	if !t.IsEntry(a.ID()) {
		return query.Result{Epoch: t.Epoch}, fmt.Errorf("%w: node %d is not an entry in epoch %d",
			query.ErrUnavailable, a.ID(), t.Epoch)
	}
	start := time.Now()
	res, err := a.navigateAndAggregate(ctx, t, req)
	a.observe("entry", start, err)
	return res, err
}

func (a *Agent) navigateAndAggregate(ctx context.Context, t *epoch.Table, req *query.Request) (query.Result, error) {
	nav := time.Now()
	lists, err := a.local.Navigate(ctx, t.Epoch, req.Vec, req.Params.NProbe, req.Params.EF)
	if err != nil {
		return query.Result{}, err
	}
	a.reg.Hist("entry.navigate").Record(time.Since(nav))
	req.Lists = lists
	// The nodes the query has to reach, under this entry's epoch t: the aggregator is chosen
	// among them, and runs the query with these groups under t -- here, or on the node it is
	// forwarded to -- while this entry keeps t pinned.
	groups := query.GroupByOwner(lists, a.manifest, t)
	target, err := a.selector.Pick(t, a.ID(), req, groups)
	if err != nil {
		return query.Result{}, err
	}
	if !t.IsAggregator(target) {
		return query.Result{Epoch: t.Epoch}, fmt.Errorf("%w: node %d is not an aggregator in epoch %d",
			query.ErrUnavailable, target, t.Epoch)
	}
	if target == a.ID() {
		return a.aggregate(ctx, t.Epoch, req, groups)
	}
	n, ok := t.Node(target)
	if !ok {
		return query.Result{}, fmt.Errorf("selector picked unknown node %d", target)
	}
	a.reg.Counter("entry.forwarded").Add(1)
	return a.aggClient(n.QueryAddr).Aggregate(ctx, t.Epoch, req, groups)
}

// forwarded runs a query that another entry routed under epoch e and handed to this node. It
// runs under e whatever this node has installed: a node that joins may not have installed e
// yet (the entry installed it first), a node that is leaving may be past it (the entry
// admitted the query just before the flip). The entry keeps e pinned until the answer is
// back; Pin only makes the work visible to this node's own bookkeeping.
func (a *Agent) forwarded(ctx context.Context, e uint64, req *query.Request, groups []query.Group) (query.Result, error) {
	release := a.tracker.Pin(e)
	defer release()
	return a.aggregate(ctx, e, req, groups)
}

// aggregate runs the scatter/gather of a query routed under epoch e.
func (a *Agent) aggregate(ctx context.Context, e uint64, req *query.Request, groups []query.Group) (query.Result, error) {
	start := time.Now()
	res, err := a.agg.Run(ctx, e, req, groups)
	a.observe("aggregate", start, err)
	return res, err
}

func (a *Agent) observe(what string, start time.Time, err error) {
	if err == nil {
		a.reg.Hist(what + ".latency").Record(time.Since(start))
		return
	}
	var se *query.StatusError
	switch {
	case errors.As(err, &se) && se.Status == query.StatusUnavailable, errors.Is(err, query.ErrUnavailable):
		a.reg.Counter(what + ".unavailable").Add(1)
	case query.StatusOf(err) == query.StatusUndecided:
		a.reg.Counter(what + ".undecided").Add(1)
	default:
		a.reg.Counter(what + ".errors").Add(1)
	}
}
