// Package ctrl is the control-plane RPC between the controller, the node agents and the
// client/harness.
//
// It keeps Koala's control flow (one long-lived connection per worker, the coordinator sends
// a command and waits for the worker's answer; coordinator/managedWorker.go and
// worker/controlPlane.go) but replaces string-matched ACKs with request IDs and typed JSON
// bodies, so several requests can be outstanding at once and errors travel back to the
// caller instead of ending the process (Koala calls log.Fatalf on every failure).
//
// Transport: rtier frames (internal/frame). Frame type 1 = request, 2 = response,
// 3 = notification (no response); the body is a JSON envelope {"m": method, "err": ...,
// "body": ...}. JSON keeps the protocol readable from Python (scripts/rtier_ctl.py).
package ctrl

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"rtier/internal/frame"
)

const (
	typeRequest  uint8 = 1
	typeResponse uint8 = 2
	typeNotify   uint8 = 3
)

type envelope struct {
	Method string          `json:"m"`
	Err    string          `json:"err,omitempty"`
	Body   json.RawMessage `json:"body,omitempty"`
}

// Handler serves one request method. The returned value is JSON-encoded as the response body.
type Handler func(ctx context.Context, body json.RawMessage) (any, error)

// RemoteError is an error returned by the peer's handler.
type RemoteError struct {
	Method string
	Msg    string
}

func (e *RemoteError) Error() string { return fmt.Sprintf("%s: %s", e.Method, e.Msg) }

// ErrClosed is returned for calls on a closed connection.
var ErrClosed = errors.New("ctrl: connection closed")

type result struct {
	env envelope
	err error
}

// Conn is one bidirectional control connection. Either side may call the other.
type Conn struct {
	nc       net.Conn
	wmu      sync.Mutex
	bw       *bufio.Writer
	nextID   atomic.Uint64
	mu       sync.Mutex
	pending  map[uint64]chan result
	handlers map[string]Handler
	notifies map[string]func(json.RawMessage)
	ctx      context.Context
	cancel   context.CancelFunc
	done     chan struct{}
	closeErr error
	started  atomic.Bool
}

// NewConn wraps nc. Register handlers, then call Start.
func NewConn(nc net.Conn) *Conn {
	ctx, cancel := context.WithCancel(context.Background())
	return &Conn{
		nc:       nc,
		bw:       bufio.NewWriter(nc),
		pending:  make(map[uint64]chan result),
		handlers: make(map[string]Handler),
		notifies: make(map[string]func(json.RawMessage)),
		ctx:      ctx,
		cancel:   cancel,
		done:     make(chan struct{}),
	}
}

// Dial connects to a ctrl endpoint (the connection is not started).
func Dial(ctx context.Context, addr string) (*Conn, error) {
	var d net.Dialer
	nc, err := d.DialContext(ctx, "tcp", addr)
	if err != nil {
		return nil, err
	}
	return NewConn(nc), nil
}

// Handle registers a request handler. Must be called before Start.
func (c *Conn) Handle(method string, h Handler) { c.handlers[method] = h }

// OnNotify registers a notification handler. Must be called before Start.
func (c *Conn) OnNotify(method string, h func(json.RawMessage)) { c.notifies[method] = h }

// Start runs the read loop in the background.
func (c *Conn) Start() {
	if c.started.Swap(true) {
		return
	}
	go c.readLoop()
}

// Done is closed when the connection ends.
func (c *Conn) Done() <-chan struct{} { return c.done }

// Err is the reason the connection ended (valid after Done).
func (c *Conn) Err() error { return c.closeErr }

// RemoteAddr of the peer.
func (c *Conn) RemoteAddr() net.Addr { return c.nc.RemoteAddr() }

// Close ends the connection; pending calls fail with ErrClosed.
func (c *Conn) Close() error {
	err := c.nc.Close()
	if !c.started.Load() {
		c.shutdown(ErrClosed)
	}
	return err
}

// Call sends a request and waits for its response. resp may be nil.
func (c *Conn) Call(ctx context.Context, method string, req, resp any) error {
	body, err := json.Marshal(req)
	if err != nil {
		return err
	}
	id := c.nextID.Add(1)
	ch := make(chan result, 1)
	c.mu.Lock()
	if c.pending == nil {
		c.mu.Unlock()
		return ErrClosed
	}
	c.pending[id] = ch
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		if c.pending != nil {
			delete(c.pending, id)
		}
		c.mu.Unlock()
	}()
	if err := c.send(typeRequest, id, envelope{Method: method, Body: body}); err != nil {
		return err
	}
	select {
	case r := <-ch:
		if r.err != nil {
			return r.err
		}
		if r.env.Err != "" {
			return &RemoteError{Method: method, Msg: r.env.Err}
		}
		if resp != nil && len(r.env.Body) > 0 {
			return json.Unmarshal(r.env.Body, resp)
		}
		return nil
	case <-ctx.Done():
		return fmt.Errorf("%s: %w", method, ctx.Err())
	}
}

// Notify sends a one-way message.
func (c *Conn) Notify(method string, body any) error {
	b, err := json.Marshal(body)
	if err != nil {
		return err
	}
	return c.send(typeNotify, 0, envelope{Method: method, Body: b})
}

func (c *Conn) send(typ uint8, id uint64, env envelope) error {
	b, err := json.Marshal(env)
	if err != nil {
		return err
	}
	c.wmu.Lock()
	defer c.wmu.Unlock()
	select {
	case <-c.done:
		return ErrClosed
	default:
	}
	_ = c.nc.SetWriteDeadline(time.Now().Add(30 * time.Second))
	if err := frame.Write(c.bw, &frame.Frame{Type: typ, ReqID: id, Body: b}); err != nil {
		return err
	}
	return c.bw.Flush()
}

func (c *Conn) readLoop() {
	br := bufio.NewReader(c.nc)
	var err error
	for {
		var f frame.Frame
		if err = frame.Read(br, &f); err != nil {
			break
		}
		var env envelope
		if err = json.Unmarshal(f.Body, &env); err != nil {
			break
		}
		switch f.Type {
		case typeResponse:
			c.mu.Lock()
			ch := c.pending[f.ReqID]
			c.mu.Unlock()
			if ch != nil {
				ch <- result{env: env}
			}
		case typeRequest:
			// One goroutine per request: a long-running handler (e.g. WaitDrained) must not
			// block the connection. Callers that need ordering wait for the response first,
			// as Koala's coordinator does.
			go c.serve(f.ReqID, env)
		case typeNotify:
			if h := c.notifies[env.Method]; h != nil {
				h(env.Body)
			}
		default:
			err = fmt.Errorf("ctrl: unknown frame type %d", f.Type)
		}
		if err != nil {
			break
		}
	}
	c.shutdown(err)
}

func (c *Conn) serve(id uint64, env envelope) {
	resp := envelope{Method: env.Method}
	h := c.handlers[env.Method]
	if h == nil {
		resp.Err = "unknown method " + env.Method
	} else {
		out, err := safeCall(h, c.ctx, env.Body)
		if err != nil {
			resp.Err = err.Error()
		} else if out != nil {
			if b, merr := json.Marshal(out); merr != nil {
				resp.Err = merr.Error()
			} else {
				resp.Body = b
			}
		}
	}
	if err := c.send(typeResponse, id, resp); err != nil && !errors.Is(err, ErrClosed) {
		log.Printf("ctrl: reply to %s failed: %v", env.Method, err)
	}
}

// safeCall turns a handler panic into an error so one bad request cannot kill the process.
func safeCall(h Handler, ctx context.Context, body json.RawMessage) (out any, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("handler panic: %v", r)
		}
	}()
	return h(ctx, body)
}

func (c *Conn) shutdown(err error) {
	c.mu.Lock()
	pending := c.pending
	c.pending = nil
	c.mu.Unlock()
	if pending == nil {
		return
	}
	if err == nil {
		err = ErrClosed
	}
	c.closeErr = err
	c.cancel()
	close(c.done)
	_ = c.nc.Close()
	for _, ch := range pending {
		ch <- result{err: fmt.Errorf("%w (%v)", ErrClosed, err)}
	}
}

// Serve accepts connections on ln; setup registers handlers on each new Conn, which is then
// started. Serve returns when ln is closed.
func Serve(ln net.Listener, setup func(*Conn)) error {
	for {
		nc, err := ln.Accept()
		if err != nil {
			if errors.Is(err, net.ErrClosed) {
				return nil
			}
			return err
		}
		c := NewConn(nc)
		setup(c)
		c.Start()
	}
}

// Decode unmarshals a request body into T.
func Decode[T any](body json.RawMessage) (T, error) {
	var v T
	if len(body) == 0 {
		return v, nil
	}
	err := json.Unmarshal(body, &v)
	return v, err
}
