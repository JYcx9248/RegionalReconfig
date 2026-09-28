// Package nodeclient talks to the C++ data-node service (engine/tools/rtier_node.cpp) over the
// rtier wire protocol (engine/node/wire.h). One request per connection at a time; a Client
// keeps a small pool of connections for concurrency.
package nodeclient

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"sync/atomic"
	"time"

	"rtier/internal/frame"
)

// Operation codes (engine/node/wire.h).
const (
	opPing           uint8 = 0x01
	opInfo           uint8 = 0x02
	opLoadGraph      uint8 = 0x10
	opLoadPartition  uint8 = 0x11
	opEvictPartition uint8 = 0x12
	opPQMissing      uint8 = 0x13
	opPQGet          uint8 = 0x14
	opPQPut          uint8 = 0x15
	opPQRelease      uint8 = 0x16
	opRawMissing     uint8 = 0x17
	opRawGet         uint8 = 0x18
	opRawPut         uint8 = 0x19
	opRawCheck       uint8 = 0x1A
	opNavigate       uint8 = 0x20
	opFilter         uint8 = 0x21
	opRerank         uint8 = 0x22
	opSearchLocal    uint8 = 0x23
)

// Status codes (engine/node/wire.h).
const (
	StatusOK          uint16 = 0
	StatusBadRequest  uint16 = 1
	StatusNotResident uint16 = 2
	StatusNoGraph     uint16 = 3
	StatusInternal    uint16 = 4
	StatusUndecided   uint16 = 5
	StatusPQAbsent    uint16 = 7 // a PQ code the operation needs is not on the node
	StatusPQFull      uint16 = 8 // the node's PQ budget is exhausted
	StatusRawAbsent   uint16 = 9 // a raw vector the operation needs is not on the node and was not fetched
)

// StatusError is a non-OK response from a data node.
type StatusError struct {
	Addr   string
	Op     uint8
	Status uint16
	Msg    string
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("node %s op 0x%02x: status %d: %s", e.Addr, e.Op, e.Status, e.Msg)
}

// IsNotResident reports whether err says a partition is not resident on the node.
func IsNotResident(err error) bool {
	var se *StatusError
	return errors.As(err, &se) && se.Status == StatusNotResident
}

// IsPQAbsent reports whether err says the node lacks a PQ code the operation needs.
func IsPQAbsent(err error) bool {
	var se *StatusError
	return errors.As(err, &se) && se.Status == StatusPQAbsent
}

// IsRawAbsent reports whether err says the node lacks a raw vector and could not fetch it.
func IsRawAbsent(err error) bool {
	var se *StatusError
	return errors.As(err, &se) && se.Status == StatusRawAbsent
}

// IsNoGraph reports whether err says the node has no navigation graph.
func IsNoGraph(err error) bool {
	var se *StatusError
	return errors.As(err, &se) && se.Status == StatusNoGraph
}

// Candidate is a vector ID with a distance. Lists of candidates are ordered by (Dist, ID).
type Candidate struct {
	ID   uint32  `json:"id"`
	Dist float32 `json:"dist"`
}

// Less is the total order used everywhere in rtier: distance, then ID.
func Less(a, b Candidate) bool { return a.Dist < b.Dist || (a.Dist == b.Dist && a.ID < b.ID) }

// OpStats are the node's per-operation counters.
type OpStats struct {
	Count  uint64 `json:"count"`
	Errors uint64 `json:"errors"`
	BusyUs uint64 `json:"busy_us"`
}

// PQStats describe the node's PQ code store (engine/include/fusion/pq_store.h): one code per
// vector. Live codes are named by resident partitions, staged ones are held for a staging in
// progress, cached ones stayed after their partitions moved away (dropped only to make room).
// Received, FromIndex, Skipped, Evicted, Freed and Served are cumulative.
type PQStats struct {
	M         int    `json:"m"`
	Capacity  uint64 `json:"capacity"`
	Resident  uint64 `json:"resident"` // live + staged + cached
	Live      uint64 `json:"live"`
	Staged    uint64 `json:"staged"`
	Cached    uint64 `json:"cached"`
	Received  uint64 `json:"received"`
	FromIndex uint64 `json:"from_index"`
	Skipped   uint64 `json:"skipped"`
	Evicted   uint64 `json:"evicted"`
	Freed     uint64 `json:"freed"`
	Served    uint64 `json:"served"`
}

// RawStats describe the node's raw vectors (engine/include/fusion/raw_store.h): a sparse copy
// of the index's page file. Pending ones are named by loaded partitions and not here yet
// (after a migration, until the stream or an on-demand fetch brings them). FromIndex, Streamed,
// Fetched (on demand), Fetches (round trips), Skipped and Served are cumulative.
type RawStats struct {
	VecBytes  int    `json:"vec_bytes"`
	Locations uint64 `json:"locations"`
	Present   uint64 `json:"present"`
	Pending   uint64 `json:"pending"`
	FromIndex uint64 `json:"from_index"`
	Streamed  uint64 `json:"streamed"`
	Fetched   uint64 `json:"fetched"`
	Fetches   uint64 `json:"fetches"`
	Skipped   uint64 `json:"skipped"`
	Served    uint64 `json:"served"`
}

// Info is the node's INFO reply.
type Info struct {
	DType         string             `json:"dtype"`
	Dim           int                `json:"dim"`
	NumVectors    uint64             `json:"num_vectors"`
	NumLists      int                `json:"num_lists"`
	NumPartitions int                `json:"num_partitions"`
	MaxNProbe     int                `json:"max_nprobe"`
	MaxRerank     int                `json:"max_rerank"`
	Workers       int                `json:"workers"`
	Filter        string             `json:"filter"`
	Payload       string             `json:"payload"`
	GraphLoaded   bool               `json:"graph_loaded"`
	MaxEpochSeen  uint64             `json:"max_epoch_seen"`
	Resident      []int              `json:"resident"`
	PQ            PQStats            `json:"pq"`
	Raw           RawStats           `json:"raw"`
	Ops           map[string]OpStats `json:"ops"`
}

// FilterStats accompany a FILTER reply.
type FilterStats struct {
	Gathered uint32 // vector IDs gathered (replicas included)
	Unique   uint32 // distinct IDs scored with PQ
}

type conn struct {
	nc net.Conn
	br *bufio.Reader
}

// Client is a connection pool to one data node.
type Client struct {
	addr    string
	sem     chan struct{}
	idle    chan *conn
	nextID  atomic.Uint64
	Timeout time.Duration // per request when ctx has no deadline (default 30 s)
}

// New creates a client for addr with up to maxConns concurrent requests. Connections are
// opened lazily.
func New(addr string, maxConns int) *Client {
	maxConns = max(maxConns, 1)
	return &Client{
		addr:    addr,
		sem:     make(chan struct{}, maxConns),
		idle:    make(chan *conn, maxConns),
		Timeout: 30 * time.Second,
	}
}

// Addr is the node's address.
func (c *Client) Addr() string { return c.addr }

// Close closes idle connections.
func (c *Client) Close() {
	for {
		select {
		case cn := <-c.idle:
			cn.nc.Close()
		default:
			return
		}
	}
}

func (c *Client) get(ctx context.Context) (*conn, error) {
	select {
	case c.sem <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	select {
	case cn := <-c.idle:
		return cn, nil
	default:
	}
	var d net.Dialer
	nc, err := d.DialContext(ctx, "tcp", c.addr)
	if err != nil {
		<-c.sem
		return nil, err
	}
	if tc, ok := nc.(*net.TCPConn); ok {
		tc.SetNoDelay(true)
	}
	return &conn{nc: nc, br: bufio.NewReaderSize(nc, 64<<10)}, nil
}

func (c *Client) put(cn *conn, healthy bool) {
	if healthy {
		select {
		case c.idle <- cn:
		default:
			cn.nc.Close()
		}
	} else {
		cn.nc.Close()
	}
	<-c.sem
}

// do sends one request and returns the response body.
func (c *Client) do(ctx context.Context, op uint8, epoch uint64, body []byte) ([]byte, error) {
	cn, err := c.get(ctx)
	if err != nil {
		return nil, err
	}
	deadline, ok := ctx.Deadline()
	if !ok {
		deadline = time.Now().Add(c.Timeout)
	}
	cn.nc.SetDeadline(deadline)
	stop := context.AfterFunc(ctx, func() { cn.nc.SetDeadline(time.Now()) })
	id := c.nextID.Add(1)
	err = frame.Write(cn.nc, &frame.Frame{Type: op, ReqID: id, Epoch: epoch, Body: body})
	var resp frame.Frame
	if err == nil {
		err = frame.Read(cn.br, &resp)
	}
	stop()
	if err == nil && (resp.ReqID != id || resp.Type != op || !resp.IsResponse()) {
		err = fmt.Errorf("nodeclient: mismatched response from %s", c.addr)
	}
	if err != nil {
		c.put(cn, false)
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, fmt.Errorf("nodeclient %s: %w", c.addr, err)
	}
	c.put(cn, true)
	if resp.Status != StatusOK {
		return nil, &StatusError{Addr: c.addr, Op: op, Status: resp.Status, Msg: string(resp.Body)}
	}
	return resp.Body, nil
}

// Ping checks that the node answers.
func (c *Client) Ping(ctx context.Context) error {
	_, err := c.do(ctx, opPing, 0, nil)
	return err
}

// Info returns the node's description and counters.
func (c *Client) Info(ctx context.Context) (*Info, error) {
	b, err := c.do(ctx, opInfo, 0, nil)
	if err != nil {
		return nil, err
	}
	var in Info
	if err := json.Unmarshal(b, &in); err != nil {
		return nil, fmt.Errorf("nodeclient: bad INFO from %s: %w", c.addr, err)
	}
	return &in, nil
}

// LoadGraph makes the node load its navigation graph from path (a path on the node's host).
func (c *Client) LoadGraph(ctx context.Context, path string) error {
	_, err := c.do(ctx, opLoadGraph, 0, []byte(path))
	return err
}

// LoadPartition makes the node load partition p's segment from path. With fromIndex, the PQ
// codes and raw vectors the node lacks are copied from the index (bootstrap). Otherwise the PQ
// codes must have been installed with PQPut (or the load fails with StatusPQAbsent), and the
// raw vectors the node lacks are fetched from rawPeer, a data node's address, when a query
// needs them -- until RawPut brings them (StatusRawAbsent if some are missing and rawPeer is
// empty).
func (c *Client) LoadPartition(ctx context.Context, p int, path string, fromIndex bool, rawPeer string) error {
	var w frame.Writer
	w.U32(uint32(p))
	if fromIndex {
		w.U8(1)
	} else {
		w.U8(0)
	}
	w.U32(uint32(len(rawPeer)))
	w.Str(rawPeer)
	w.Bytes([]byte(path))
	_, err := c.do(ctx, opLoadPartition, 0, w.B)
	return err
}

// PQMissing returns, for each group of segment files (paths on the node's host), the vector
// IDs they name whose PQ code the node lacks and that no earlier group lists: every missing
// code appears exactly once, under the first group (source) that needs it. The codes the node
// already has are protected until the partitions load (or PQRelease), so making room for the
// missing ones never drops them.
func (c *Client) PQMissing(ctx context.Context, groups [][]string) ([][]uint32, error) {
	return c.missing(ctx, opPQMissing, groups)
}

// RawMissing returns, for each group of segment files, the raw-vector locations they name whose
// vector the node lacks and that no earlier group lists.
func (c *Client) RawMissing(ctx context.Context, groups [][]string) ([][]uint32, error) {
	return c.missing(ctx, opRawMissing, groups)
}

func (c *Client) missing(ctx context.Context, op uint8, groups [][]string) ([][]uint32, error) {
	var w frame.Writer
	w.U32(uint32(len(groups)))
	for _, g := range groups {
		w.U32(uint32(len(g)))
		for _, p := range g {
			w.U32(uint32(len(p)))
			w.Str(p)
		}
	}
	b, err := c.do(ctx, op, 0, w.B)
	if err != nil {
		return nil, err
	}
	r := frame.NewReader(b)
	n := int(r.U32())
	if r.Err != nil || n != len(groups) {
		return nil, fmt.Errorf("nodeclient: bad reply to op 0x%02x from %s", op, c.addr)
	}
	out := make([][]uint32, n)
	for i := range out {
		out[i] = r.U32s(int(r.U32()))
	}
	if r.Err == nil && r.Remaining() != 0 {
		return nil, fmt.Errorf("nodeclient: trailing bytes in the reply to op 0x%02x from %s", op, c.addr)
	}
	return out, r.Err
}

// RawGet returns the raw vectors at locs (vecBytes each, in order). The node fetches the ones
// it is itself still waiting for from its own source first.
func (c *Client) RawGet(ctx context.Context, locs []uint32) (int, []byte, error) {
	var w frame.Writer
	w.U32(uint32(len(locs)))
	w.U32s(locs)
	b, err := c.do(ctx, opRawGet, 0, w.B)
	if err != nil {
		return 0, nil, err
	}
	r := frame.NewReader(b)
	vb := int(r.U32())
	vecs := r.Rest()
	if r.Err != nil || len(vecs) != vb*len(locs) {
		return 0, nil, fmt.Errorf("nodeclient: bad RAW_GET reply from %s", c.addr)
	}
	return vb, vecs, nil
}

// RawAbsent returns which of locs the node does not hold (sorted, without duplicates).
func (c *Client) RawAbsent(ctx context.Context, locs []uint32) ([]uint32, error) {
	var w frame.Writer
	w.U32(uint32(len(locs)))
	w.U32s(locs)
	b, err := c.do(ctx, opRawCheck, 0, w.B)
	if err != nil {
		return nil, err
	}
	r := frame.NewReader(b)
	out := r.U32s(int(r.U32()))
	if r.Err == nil && r.Remaining() != 0 {
		return nil, fmt.Errorf("nodeclient: trailing bytes in RAW_CHECK reply from %s", c.addr)
	}
	return out, r.Err
}

// RawPut installs raw vectors (vecBytes each, in the order of locs); vectors the node already
// holds are skipped.
func (c *Client) RawPut(ctx context.Context, locs []uint32, vecBytes int, vecs []byte) (installed, skipped int, err error) {
	if len(vecs) != vecBytes*len(locs) {
		return 0, 0, fmt.Errorf("nodeclient: %d bytes of vectors for %d locations of %d bytes", len(vecs), len(locs), vecBytes)
	}
	var w frame.Writer
	w.U32(uint32(len(locs)))
	w.U32(uint32(vecBytes))
	w.U32s(locs)
	w.Bytes(vecs)
	b, err := c.do(ctx, opRawPut, 0, w.B)
	if err != nil {
		return 0, 0, err
	}
	r := frame.NewReader(b)
	installed, skipped = int(r.U32()), int(r.U32())
	return installed, skipped, r.Err
}

// PQGet returns the PQ codes of ids (m bytes each, in order) and m.
func (c *Client) PQGet(ctx context.Context, ids []uint32) (int, []byte, error) {
	var w frame.Writer
	w.U32(uint32(len(ids)))
	w.U32s(ids)
	b, err := c.do(ctx, opPQGet, 0, w.B)
	if err != nil {
		return 0, nil, err
	}
	r := frame.NewReader(b)
	m := int(r.U32())
	codes := r.Rest()
	if r.Err != nil || len(codes) != m*len(ids) {
		return 0, nil, fmt.Errorf("nodeclient: bad PQ_GET reply from %s", c.addr)
	}
	return m, codes, nil
}

// PQPut installs PQ codes (m bytes each, in the order of ids); codes the node already holds
// are skipped.
func (c *Client) PQPut(ctx context.Context, ids []uint32, m int, codes []byte) (installed, skipped int, err error) {
	if len(codes) != m*len(ids) {
		return 0, 0, fmt.Errorf("nodeclient: %d bytes of codes for %d IDs of %d bytes", len(codes), len(ids), m)
	}
	var w frame.Writer
	w.U32(uint32(len(ids)))
	w.U32(uint32(m))
	w.U32s(ids)
	w.Bytes(codes)
	b, err := c.do(ctx, opPQPut, 0, w.B)
	if err != nil {
		return 0, 0, err
	}
	r := frame.NewReader(b)
	installed, skipped = int(r.U32()), int(r.U32())
	return installed, skipped, r.Err
}

// PQRelease ends a staging that did not load: the node frees the codes installed for it and
// returns the codes it protected to its cache.
func (c *Client) PQRelease(ctx context.Context) (uint64, error) {
	b, err := c.do(ctx, opPQRelease, 0, nil)
	if err != nil {
		return 0, err
	}
	r := frame.NewReader(b)
	n := r.U64()
	return n, r.Err
}

// EvictPartition drops partition p; reports whether it was resident.
func (c *Client) EvictPartition(ctx context.Context, p int) (bool, error) {
	var w frame.Writer
	w.U32(uint32(p))
	b, err := c.do(ctx, opEvictPartition, 0, w.B)
	if err != nil {
		return false, err
	}
	return len(b) == 1 && b[0] == 1, nil
}

// Navigate returns the query's nprobe closest lists (closest first).
func (c *Client) Navigate(ctx context.Context, epoch uint64, q []byte, nprobe, ef int) ([]uint32, error) {
	var w frame.Writer
	w.U32(uint32(nprobe))
	w.U32(uint32(ef))
	w.Bytes(q)
	b, err := c.do(ctx, opNavigate, epoch, w.B)
	if err != nil {
		return nil, err
	}
	r := frame.NewReader(b)
	lists := r.U32s(int(r.U32()))
	return lists, r.Err
}

// Filter returns the PQ top-n over the given lists (all resident on this node), ordered by
// (PQ distance, ID).
func (c *Client) Filter(ctx context.Context, epoch uint64, q []byte, lists []uint32, topn int) ([]Candidate, FilterStats, error) {
	var w frame.Writer
	w.U32(uint32(topn))
	w.U32(uint32(len(lists)))
	w.U32s(lists)
	w.Bytes(q)
	b, err := c.do(ctx, opFilter, epoch, w.B)
	if err != nil {
		return nil, FilterStats{}, err
	}
	r := frame.NewReader(b)
	cands := readCandidates(r)
	st := FilterStats{Gathered: r.U32(), Unique: r.U32()}
	return cands, st, r.Err
}

// Rerank computes exact distances for every given ID and returns the k best, ordered by
// (distance, ID), and the pages it read. The node fetches the raw vectors it lacks first (the
// reply's last field counts them; see Info's Raw.Fetched).
func (c *Client) Rerank(ctx context.Context, epoch uint64, q []byte, ids []uint32, k int) ([]Candidate, uint32, error) {
	var w frame.Writer
	w.U32(uint32(k))
	w.U32(uint32(len(ids)))
	w.U32s(ids)
	w.Bytes(q)
	b, err := c.do(ctx, opRerank, epoch, w.B)
	if err != nil {
		return nil, 0, err
	}
	r := frame.NewReader(b)
	cands := readCandidates(r)
	pages := r.U32()
	return cands, pages, r.Err
}

// SearchParams for SearchLocal.
type SearchParams struct {
	K, NProbe, Rerank, EF int
	Heuristic             bool
}

// SearchLocal runs the whole FusionANNS pipeline on this node (every probed list must be
// resident): the single-node baseline.
func (c *Client) SearchLocal(ctx context.Context, epoch uint64, q []byte, p SearchParams) ([]Candidate, error) {
	var w frame.Writer
	w.U32(uint32(p.K))
	w.U32(uint32(p.NProbe))
	w.U32(uint32(p.Rerank))
	w.U32(uint32(p.EF))
	if p.Heuristic {
		w.U8(1)
	} else {
		w.U8(0)
	}
	w.Bytes([]byte{0, 0, 0})
	w.Bytes(q)
	b, err := c.do(ctx, opSearchLocal, epoch, w.B)
	if err != nil {
		return nil, err
	}
	r := frame.NewReader(b)
	cands := readCandidates(r)
	return cands, r.Err
}

func readCandidates(r *frame.Reader) []Candidate {
	n := int(r.U32())
	ids := r.U32s(n)
	ds := r.F32s(n)
	if r.Err != nil {
		return nil
	}
	out := make([]Candidate, n)
	for i := range out {
		out[i] = Candidate{ID: ids[i], Dist: ds[i]}
	}
	return out
}
