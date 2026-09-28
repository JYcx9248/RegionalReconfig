package query

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"math"
	"net"
	"sync/atomic"
	"time"

	"rtier/internal/design"
	"rtier/internal/frame"
	"rtier/internal/nodeclient"
	"rtier/internal/placement"
)

// Query-port operations (served by rtier-agent).
const (
	OpQuery uint8 = 0x40 // client -> entry: params + vector
	// OpAggregate: entry -> aggregator: params + the navigated lists grouped by owner + vector;
	// the frame's epoch field is the epoch the entry pinned and routed with.
	OpAggregate uint8 = 0x41
)

// Query-port status codes.
const (
	StatusOK          uint16 = 0
	StatusBadRequest  uint16 = 1
	StatusInternal    uint16 = 4
	StatusUndecided   uint16 = 5 // a TODO(design) placeholder was reached
	StatusUnavailable uint16 = 6 // not an entry/aggregator in the current epoch, or admission paused
)

// Sanity bounds on query parameters (the data nodes enforce their own, smaller limits).
const (
	MaxK      = 1 << 14
	MaxN      = 1 << 20
	MaxNProbe = 1 << 16
)

// ErrUnavailable: the node does not serve this request right now; try another entry.
var ErrUnavailable = errors.New("query: node unavailable")

// StatusOf maps an error to a query-port status code.
func StatusOf(err error) uint16 {
	switch {
	case err == nil:
		return StatusOK
	case errors.Is(err, design.ErrUndecided):
		return StatusUndecided
	case errors.Is(err, ErrUnavailable), errors.Is(err, ErrNoEpoch):
		return StatusUnavailable
	}
	return StatusInternal
}

func putParams(w *frame.Writer, p Params) {
	w.U32(uint32(p.K))
	w.U32(uint32(p.NProbe))
	w.U32(uint32(p.N))
	w.U32(uint32(p.EF))
}

func getParams(r *frame.Reader) Params {
	return Params{K: int(r.U32()), NProbe: int(r.U32()), N: int(r.U32()), EF: int(r.U32())}
}

func checkRequest(req *Request) error {
	pr := req.Params
	if pr.K <= 0 || pr.NProbe <= 0 || pr.N <= 0 || pr.EF < 0 || len(req.Vec) == 0 ||
		pr.K > MaxK || pr.N > MaxN || pr.NProbe > MaxNProbe {
		return fmt.Errorf("query: bad parameters %+v", req.Params)
	}
	return nil
}

// EncodeRequest builds the body of OpQuery: params + vector.
func EncodeRequest(req *Request) []byte {
	var w frame.Writer
	putParams(&w, req.Params)
	w.Bytes(req.Vec)
	return w.B
}

// DecodeRequest parses an OpQuery body. The vector aliases body.
func DecodeRequest(body []byte) (*Request, error) {
	r := frame.NewReader(body)
	req := &Request{Params: getParams(r)}
	req.Vec = r.Rest()
	if r.Err != nil {
		return nil, r.Err
	}
	if err := checkRequest(req); err != nil {
		return nil, err
	}
	return req, nil
}

// EncodeAggregate builds the body of OpAggregate: params, the lists grouped by owner (u32
// groups, then per group u32 node, u32 n, u32 lists[n]), vector.
func EncodeAggregate(req *Request, groups []Group) []byte {
	var w frame.Writer
	putParams(&w, req.Params)
	w.U32(uint32(len(groups)))
	for _, g := range groups {
		w.U32(uint32(g.Node))
		w.U32(uint32(len(g.Lists)))
		w.U32s(g.Lists)
	}
	w.Bytes(req.Vec)
	return w.B
}

// DecodeAggregate parses an OpAggregate body. The vector aliases body; req.Lists stays nil
// (the lists travel in the groups).
func DecodeAggregate(body []byte) (*Request, []Group, error) {
	r := frame.NewReader(body)
	req := &Request{Params: getParams(r)}
	ng := int(r.U32())
	if ng <= 0 || ng > MaxNProbe {
		return nil, nil, fmt.Errorf("query: forwarded query with %d owner groups", ng)
	}
	groups := make([]Group, 0, ng)
	total := 0
	for i := 0; i < ng && r.Err == nil; i++ {
		node, n := r.U32(), int(r.U32())
		if node > math.MaxUint16 || n <= 0 || total+n > MaxNProbe {
			return nil, nil, fmt.Errorf("query: forwarded query with a bad group (node %d, %d lists)", node, n)
		}
		total += n
		groups = append(groups, Group{Node: placement.NodeID(node), Lists: r.U32s(n)})
	}
	req.Vec = r.Rest()
	if r.Err != nil {
		return nil, nil, r.Err
	}
	if err := checkRequest(req); err != nil {
		return nil, nil, err
	}
	return req, groups, nil
}

// EncodeResult builds a response body: u32 n, u32 ids[n], f32 dists[n].
func EncodeResult(cands []nodeclient.Candidate) []byte {
	var w frame.Writer
	w.U32(uint32(len(cands)))
	for _, c := range cands {
		w.U32(c.ID)
	}
	for _, c := range cands {
		w.F32(c.Dist)
	}
	return w.B
}

// DecodeResult parses a response body.
func DecodeResult(body []byte) ([]nodeclient.Candidate, error) {
	r := frame.NewReader(body)
	n := int(r.U32())
	ids, ds := r.U32s(n), r.F32s(n)
	if r.Err != nil {
		return nil, r.Err
	}
	out := make([]nodeclient.Candidate, n)
	for i := range out {
		out[i] = nodeclient.Candidate{ID: ids[i], Dist: ds[i]}
	}
	return out, nil
}

// StatusError is a non-OK reply on the query port.
type StatusError struct {
	Addr   string
	Status uint16
	Msg    string
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("query %s: status %d: %s", e.Addr, e.Status, e.Msg)
}

// Unwrap lets errors.Is match ErrUnavailable / design.ErrUndecided across the wire.
func (e *StatusError) Unwrap() error {
	switch e.Status {
	case StatusUnavailable:
		return ErrUnavailable
	case StatusUndecided:
		return design.ErrUndecided
	}
	return nil
}

type qconn struct {
	nc net.Conn
	br *bufio.Reader
}

// Client sends queries to one agent's query port (a pool of connections, one request at a
// time on each).
type Client struct {
	addr    string
	sem     chan struct{}
	idle    chan *qconn
	nextID  atomic.Uint64
	Timeout time.Duration
}

// NewClient creates a client with up to maxConns requests in flight.
func NewClient(addr string, maxConns int) *Client {
	maxConns = max(maxConns, 1)
	return &Client{addr: addr, sem: make(chan struct{}, maxConns), idle: make(chan *qconn, maxConns),
		Timeout: 30 * time.Second}
}

// Addr of the agent.
func (c *Client) Addr() string { return c.addr }

// Close drops idle connections.
func (c *Client) Close() {
	for {
		select {
		case q := <-c.idle:
			q.nc.Close()
		default:
			return
		}
	}
}

// Query sends a client query to an entry node.
func (c *Client) Query(ctx context.Context, req *Request) (Result, error) {
	return c.do(ctx, OpQuery, 0, EncodeRequest(req))
}

// Aggregate hands a query to the aggregator its entry picked: the entry navigated it and
// grouped its lists by owner under epoch e, which it keeps pinned until Aggregate returns.
func (c *Client) Aggregate(ctx context.Context, e uint64, req *Request, groups []Group) (Result, error) {
	return c.do(ctx, OpAggregate, e, EncodeAggregate(req, groups))
}

func (c *Client) do(ctx context.Context, op uint8, epoch uint64, body []byte) (Result, error) {
	select {
	case c.sem <- struct{}{}:
	case <-ctx.Done():
		return Result{}, ctx.Err()
	}
	defer func() { <-c.sem }()
	var q *qconn
	select {
	case q = <-c.idle:
	default:
		var d net.Dialer
		nc, err := d.DialContext(ctx, "tcp", c.addr)
		if err != nil {
			return Result{}, err
		}
		q = &qconn{nc: nc, br: bufio.NewReader(nc)}
	}
	deadline, ok := ctx.Deadline()
	if !ok {
		deadline = time.Now().Add(c.Timeout)
	}
	q.nc.SetDeadline(deadline)
	stop := context.AfterFunc(ctx, func() { q.nc.SetDeadline(time.Now()) })
	id := c.nextID.Add(1)
	err := frame.Write(q.nc, &frame.Frame{Type: op, ReqID: id, Epoch: epoch, Body: body})
	var resp frame.Frame
	if err == nil {
		err = frame.Read(q.br, &resp)
	}
	stop()
	if err == nil && resp.ReqID != id {
		err = fmt.Errorf("query: mismatched response from %s", c.addr)
	}
	if err != nil {
		q.nc.Close()
		if ctx.Err() != nil {
			return Result{}, ctx.Err()
		}
		return Result{}, err
	}
	select {
	case c.idle <- q:
	default:
		q.nc.Close()
	}
	if resp.Status != StatusOK {
		return Result{Epoch: resp.Epoch}, &StatusError{Addr: c.addr, Status: resp.Status, Msg: string(resp.Body)}
	}
	cands, err := DecodeResult(resp.Body)
	return Result{Epoch: resp.Epoch, Candidates: cands}, err
}
