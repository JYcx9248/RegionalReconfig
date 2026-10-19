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
	"math"
	"net"
	"sort"
	"sync/atomic"
	"time"

	"rtier/internal/frame"
)

// Operation codes (engine/node/wire.h).
const (
	opPing           uint8 = 0x01
	opInfo           uint8 = 0x02
	infoSettle       uint8 = 1 // INFO flags
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
	opRawPull        uint8 = 0x1B
	opRawAdd         uint8 = 0x1C
	opPauseUpdates   uint8 = 0x1D
	opNavigate       uint8 = 0x20
	opFilter         uint8 = 0x21
	opRerank         uint8 = 0x22
	opSearchLocal    uint8 = 0x23
	opAssign         uint8 = 0x24
	opSplitList      uint8 = 0x25
	opNearestHeads   uint8 = 0x26
	opEditGraph      uint8 = 0x27
	opStageVersions  uint8 = 0x30
	opDropVersion    uint8 = 0x31
	opForget         uint8 = 0x32
	opSavePartition  uint8 = 0x33
	opSetNodes       uint8 = 0x34
	opRawHeldBy      uint8 = 0x35
	opListGet        uint8 = 0x36
	opPatchStage     uint8 = 0x37
	opPatchLoad      uint8 = 0x38
	opPatchMerge     uint8 = 0x39
	opPatchDrop      uint8 = 0x3A
	opPatchSave      uint8 = 0x3B
)

// NewestGraph asks NAVIGATE for the newest graph version the node has (a staged patch's, else
// its graph's).
const NewestGraph uint32 = 0xFFFFFFFF

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
	Deleted   uint64 `json:"deleted"` // deleted vectors' codes a resident version still names
}

// RawStats describe the node's raw vectors (engine/include/fusion/raw_store.h): a sparse copy
// of the index's page file. Pending ones are named by loaded partitions and not here yet
// (after a migration, until the stream or an on-demand fetch brings them). FromIndex, Streamed,
// Fetched (on demand), Fetches (round trips), Skipped and Served are cumulative.
type RawStats struct {
	VecBytes  int    `json:"vec_bytes"`
	Locations uint64 `json:"locations"`
	Present   uint64 `json:"present"`
	Cached    uint64 `json:"cached"` // fetched, in memory until the node's writer installs them
	Pending   uint64 `json:"pending"`
	FromIndex uint64 `json:"from_index"`
	Streamed  uint64 `json:"streamed"`
	Fetched   uint64 `json:"fetched"` // installed after an on-demand fetch (Cached not yet)
	Added     uint64 `json:"added"`   // inserted vectors installed on the SSD (RawAdd, U14)
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
	IDCapacity    uint64             `json:"id_capacity"` // vector IDs, room for inserts included (U14)
	Graph         GraphInfo          `json:"graph"`
	Resident      []int              `json:"resident"`
	Versions      [][2]uint32        `json:"versions"` // every resident (partition, version)
	PQ            PQStats            `json:"pq"`
	Raw           RawStats           `json:"raw"`
	Ops           map[string]OpStats `json:"ops"`
}

// GraphInfo is the node's navigation graph (U14 major epochs patch it).
type GraphInfo struct {
	Version  uint32 `json:"version"`  // the version of its bytes
	Patch    uint32 `json:"patch"`    // the staged patch's version, 0 = none
	Count    uint64 `json:"count"`    // heads
	Capacity uint64 `json:"capacity"` // room for heads
	Deleted  uint64 `json:"deleted"`  // heads marked deleted (routing points, never a result)
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

// Info returns the node's description and counters, without waiting for anything.
func (c *Client) Info(ctx context.Context) (*Info, error) { return c.info(ctx, nil) }

// InfoSettled is Info taken once the raw vectors the node fetched before the call are on its
// SSD: with no query fetching, Raw.Cached is 0 and every fetched vector counts as present.
func (c *Client) InfoSettled(ctx context.Context) (*Info, error) {
	return c.info(ctx, []byte{infoSettle})
}

// FetchedOnDemand: raw vectors the node has fetched on demand, installed or still cached.
func (r RawStats) FetchedOnDemand() uint64 { return r.Fetched + r.Cached }

func (c *Client) info(ctx context.Context, body []byte) (*Info, error) {
	b, err := c.do(ctx, opInfo, 0, body)
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

// LoadPartition makes the node load version v of partition p from the segment at path,
// replacing that version if it is resident (the partition's other versions stay). With fromIndex, the PQ
// codes and raw vectors the node lacks are copied from the index (bootstrap). Otherwise the PQ
// codes must have been installed with PQPut (or the load fails with StatusPQAbsent), and a raw
// vector the node lacks is fetched, when a query needs it, from the holder its posting records
// (a segment saved at a hand-off records one for every posting; StatusRawAbsent if one is
// missing with no holder recorded).
func (c *Client) LoadPartition(ctx context.Context, p int, v uint32, path string, fromIndex bool) error {
	var w frame.Writer
	w.U32(uint32(p))
	w.U32(v)
	if fromIndex {
		w.U8(1)
	} else {
		w.U8(0)
	}
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

// RawRefs are raw-vector locations held by one node (its cluster ID) and, for each, a posting
// list on this node that names it (parallel): what RawPull needs to bring them.
type RawRefs struct {
	Holder      uint16
	Locs, Lists []uint32
}

// RawMissing returns the raw vectors that the newest resident versions of partitions name and
// the node lacks, grouped by the node holding them, each once with a list that names it.
func (c *Client) RawMissing(ctx context.Context, partitions []int) ([]RawRefs, error) {
	var w frame.Writer
	w.U32(uint32(len(partitions)))
	for _, p := range partitions {
		w.U32(uint32(p))
	}
	b, err := c.do(ctx, opRawMissing, 0, w.B)
	if err != nil {
		return nil, err
	}
	r := frame.NewReader(b)
	out := make([]RawRefs, r.U32())
	for i := range out {
		out[i].Holder = r.U16()
		r.U16()
		n := int(r.U32())
		out[i].Locs, out[i].Lists = r.U32s(n), r.U32s(n)
	}
	if r.Err == nil && r.Remaining() != 0 {
		return nil, fmt.Errorf("nodeclient: trailing bytes in the RAW_MISSING reply from %s", c.addr)
	}
	return out, r.Err
}

// RawHeldBy is the drain of node `holder` before it leaves: the node makes every posting that
// names it, and whose vector it has now, name itself (repointed), and returns the vectors still
// to pull from it (RawPull), each with a list that names it.
func (c *Client) RawHeldBy(ctx context.Context, holder uint16) (repointed uint64, missing RawRefs, err error) {
	var w frame.Writer
	w.U16(holder)
	w.U16(0)
	b, err := c.do(ctx, opRawHeldBy, 0, w.B)
	if err != nil {
		return 0, RawRefs{}, err
	}
	r := frame.NewReader(b)
	repointed = r.U64()
	n := int(r.U32())
	missing = RawRefs{Holder: holder, Locs: r.U32s(n), Lists: r.U32s(n)}
	if r.Err == nil && r.Remaining() != 0 {
		return 0, RawRefs{}, fmt.Errorf("nodeclient: trailing bytes in the RAW_HELD_BY reply from %s", c.addr)
	}
	return repointed, missing, r.Err
}

// SetNodes tells the node its cluster ID (0: unchanged) and where other data nodes are: the
// holders its postings record, which it fetches raw vectors from.
func (c *Client) SetNodes(ctx context.Context, self uint16, peers map[uint16]string) error {
	ids := make([]int, 0, len(peers))
	for id := range peers {
		ids = append(ids, int(id))
	}
	sort.Ints(ids)
	var w frame.Writer
	w.U16(self)
	w.U16(0)
	w.U32(uint32(len(ids)))
	for _, id := range ids {
		w.U16(uint16(id))
		w.U16(0)
		w.U32(uint32(len(peers[uint16(id)])))
		w.Str(peers[uint16(id)])
	}
	_, err := c.do(ctx, opSetNodes, 0, w.B)
	return err
}

func (c *Client) missing(ctx context.Context, op uint8, groups [][]string) ([][]uint32, error) {
	r, err := c.missingReply(ctx, op, groups)
	if err != nil {
		return nil, err
	}
	out := make([][]uint32, len(groups))
	for i := range out {
		out[i] = r.U32s(int(r.U32()))
	}
	if r.Err == nil && r.Remaining() != 0 {
		return nil, fmt.Errorf("nodeclient: trailing bytes in the reply to op 0x%02x from %s", op, c.addr)
	}
	return out, r.Err
}

// missingReply sends a *_MISSING request and returns the reply positioned after its group count.
func (c *Client) missingReply(ctx context.Context, op uint8, groups [][]string) (*frame.Reader, error) {
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
	if n := int(r.U32()); r.Err != nil || n != len(groups) {
		return nil, fmt.Errorf("nodeclient: bad reply to op 0x%02x from %s", op, c.addr)
	}
	return r, nil
}

// RawGet returns the raw vectors at locs (vecBytes each, in order) from the node's own store
// (StatusRawAbsent for one it does not hold: a peer asks the holder, there is no chain).
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

// RawAdd installs inserted vectors (U14) at their locations, vecBytes each in the order of
// locs. The node serves them from memory at once and writes them to its SSD in the background,
// after the vectors it fetched for queries and not while update writes are paused; the call
// blocks while the node's buffer of unwritten inserted vectors is full. Vectors the node already
// holds are skipped.
func (c *Client) RawAdd(ctx context.Context, locs []uint32, vecBytes int, vecs []byte) (installed, skipped int, err error) {
	if len(vecs) != vecBytes*len(locs) {
		return 0, 0, fmt.Errorf("nodeclient: %d bytes of vectors for %d locations of %d bytes", len(vecs), len(locs), vecBytes)
	}
	var w frame.Writer
	w.U32(uint32(len(locs)))
	w.U32(uint32(vecBytes))
	w.U32s(locs)
	w.Bytes(vecs)
	b, err := c.do(ctx, opRawAdd, 0, w.B)
	if err != nil {
		return 0, 0, err
	}
	r := frame.NewReader(b)
	installed, skipped = int(r.U32()), int(r.U32())
	return installed, skipped, r.Err
}

// PauseUpdateWrites holds back (paused) or resumes writing inserted vectors to the node's SSD,
// e.g. while the node stages a reconfiguration, whose transfers go first.
func (c *Client) PauseUpdateWrites(ctx context.Context, paused bool) error {
	b := []byte{0}
	if paused {
		b[0] = 1
	}
	_, err := c.do(ctx, opPauseUpdates, 0, b)
	return err
}

// Assignment is the routing of one vector (U14): the posting lists the index's build would
// give it, by the build's rule, and its PQ code.
type Assignment struct {
	Lists []uint32
	Code  []byte
}

// Assign routes vectors (the index's element type, vecs holds n of them) on the node's
// navigation graph by the build's rule: an inserted vector goes where the build would have
// put it. The node needs its graph (StatusNoGraph otherwise).
func (c *Client) Assign(ctx context.Context, vecs []byte, n int) ([]Assignment, error) {
	if n <= 0 || len(vecs)%n != 0 {
		return nil, fmt.Errorf("nodeclient: ASSIGN of %d bytes as %d vectors", len(vecs), n)
	}
	var w frame.Writer
	w.U32(uint32(n))
	w.Bytes(vecs)
	b, err := c.do(ctx, opAssign, 0, w.B)
	if err != nil {
		return nil, err
	}
	r := frame.NewReader(b)
	rep, m := int(r.U32()), int(r.U32())
	lists := r.U32s(n * rep)
	codes := r.Rest()
	if r.Err != nil {
		return nil, r.Err
	}
	if len(codes) != n*m {
		return nil, fmt.Errorf("nodeclient: ASSIGN reply with %d code bytes for %d vectors of %d", len(codes), n, m)
	}
	out := make([]Assignment, n)
	for i := range out {
		for _, l := range lists[i*rep : (i+1)*rep] {
			if l != noList {
				out[i].Lists = append(out[i].Lists, l)
			}
		}
		out[i].Code = append([]byte(nil), codes[i*m:(i+1)*m]...)
	}
	return out, nil
}

// noList pads an ASSIGN reply's lists (fusion::kInvalidId).
const noList = math.MaxUint32

// VersionStage asks for the next version of a partition, built from its base version (U14).
type VersionStage struct {
	Partition int    `json:"partition"`
	Base      uint32 `json:"base"`
	Next      uint32 `json:"next"`
}

// StageResult says what StageVersions did to one partition.
type StageResult struct {
	Partition int
	Changed   bool // false: the batch does not touch it, no version was installed
	Added     uint32
	Removed   uint32
	// The lists of the next version whose length changed, and the ones it creates: what decides
	// the next major epoch (U14).
	Lengths []ListLength
}

// ListLength is a posting list's number of postings.
type ListLength struct {
	List     uint32
	Postings uint32
}

// Edits is what StageVersions changes besides the base lists.
type Edits struct {
	// Postings to add: Lists[i] gets (IDs[i], Locs[i]). Holders[i] is the node holding the
	// vector if it is not on this node (a posting that moves between owners); 0, or a nil
	// Holders, means it must be here.
	Lists, IDs, Locs []uint32
	Holders          []uint16
	Deleted          []uint32      // vector IDs deleted from every list
	Created          []CreatedList // lists the next version creates
	Removed          []uint32      // lists the next version no longer has
}

// CreatedList is a list a major epoch creates, in its partition.
type CreatedList struct {
	List      uint32
	Partition int
}

// ListPostings is a posting list as its owner holds it: every posting's holder by node ID.
type ListPostings struct {
	IDs, Locs []uint32
	Holders   []uint16
}

// StageVersions builds, for each of parts, the next version of its posting lists from the base
// version: without the removed lists and with the created ones, plus the added postings naming
// its lists, minus the deleted IDs. Partitions the batch does not change get no new version.
// All or nothing: StatusPQAbsent or StatusRawAbsent if an added vector's code is not on the node
// yet (PQPut) or its raw vector is neither there (RawAdd) nor held by the node its posting names,
// StatusNotResident if a base version is not there.
func (c *Client) StageVersions(ctx context.Context, parts []VersionStage, e Edits) ([]StageResult, error) {
	n := len(e.Lists)
	if len(e.IDs) != n || len(e.Locs) != n || (e.Holders != nil && len(e.Holders) != n) {
		return nil, fmt.Errorf("nodeclient: %d lists, %d IDs, %d locations, %d holders", n, len(e.IDs), len(e.Locs), len(e.Holders))
	}
	var w frame.Writer
	w.U32(uint32(len(parts)))
	for _, p := range parts {
		w.U32(uint32(p.Partition))
		w.U32(p.Base)
		w.U32(p.Next)
	}
	w.U32(uint32(n))
	w.U32s(e.Lists)
	w.U32s(e.IDs)
	w.U32s(e.Locs)
	for i := 0; i < n; i++ {
		var h uint16
		if e.Holders != nil {
			h = e.Holders[i]
		}
		w.U16(h)
	}
	w.U32(uint32(len(e.Deleted)))
	w.U32s(e.Deleted)
	w.U32(uint32(len(e.Created)))
	for _, cl := range e.Created {
		w.U32(cl.List)
		w.U32(uint32(cl.Partition))
	}
	w.U32(uint32(len(e.Removed)))
	w.U32s(e.Removed)
	b, err := c.do(ctx, opStageVersions, 0, w.B)
	if err != nil {
		return nil, err
	}
	r := frame.NewReader(b)
	out := make([]StageResult, r.U32())
	for i := range out {
		out[i].Partition = int(r.U32())
		flags := r.U32() // u8 changed, u8[3] pad
		out[i].Changed = flags&0xFF != 0
		out[i].Added, out[i].Removed = r.U32(), r.U32()
		nl := int(r.U32())
		for j := 0; j < nl && r.Err == nil; j++ {
			out[i].Lengths = append(out[i].Lengths, ListLength{List: r.U32(), Postings: r.U32()})
		}
	}
	return out, r.Err
}

// ListGet returns a posting list of version v of its partition (StatusNotResident if the node
// does not hold it), every posting's holder by node ID.
func (c *Client) ListGet(ctx context.Context, list, v uint32) (ListPostings, error) {
	var w frame.Writer
	w.U32(list)
	w.U32(v)
	b, err := c.do(ctx, opListGet, 0, w.B)
	if err != nil {
		return ListPostings{}, err
	}
	r := frame.NewReader(b)
	n := int(r.U32())
	lp := ListPostings{IDs: r.U32s(n), Locs: r.U32s(n), Holders: make([]uint16, n)}
	for i := range lp.Holders {
		lp.Holders[i] = r.U16()
	}
	return lp, r.Err
}

// PatchStage stages a graph patch next to the node's graph (U14 major epochs): NAVIGATE of the
// patch's version searches through it, of the graph's own version without it.
func (c *Client) PatchStage(ctx context.Context, patch []byte) error {
	_, err := c.do(ctx, opPatchStage, 0, patch)
	return err
}

// PatchLoad stages the graph patch in a file on the node's machine.
func (c *Client) PatchLoad(ctx context.Context, path string) error {
	_, err := c.do(ctx, opPatchLoad, 0, []byte(path))
	return err
}

// PatchMerge merges the staged patch of version v into the graph in place, after the grace
// period of the epochs that searched the graph without it (a no-op if the graph is at v).
func (c *Client) PatchMerge(ctx context.Context, v uint32) error {
	var w frame.Writer
	w.U32(v)
	_, err := c.do(ctx, opPatchMerge, 0, w.B)
	return err
}

// PatchDrop unstages the patch of version v; it reports whether it was staged.
func (c *Client) PatchDrop(ctx context.Context, v uint32) (bool, error) {
	var w frame.Writer
	w.U32(v)
	b, err := c.do(ctx, opPatchDrop, 0, w.B)
	if err != nil {
		return false, err
	}
	return len(b) == 1 && b[0] == 1, nil
}

// PatchSave writes the staged patch of version v to path on the node's machine.
func (c *Client) PatchSave(ctx context.Context, v uint32, path string) error {
	var w frame.Writer
	w.U32(v)
	w.Str(path)
	_, err := c.do(ctx, opPatchSave, 0, w.B)
	return err
}

// SplitList asks the router to split a list by the PQ codes of its members (m bytes each): side
// 0 or 1 per member, balanced, and the member closest to each side's mean (the new heads).
func (c *Client) SplitList(ctx context.Context, codes []byte, m int) ([]uint8, [2]uint32, error) {
	var heads [2]uint32
	if m <= 0 || len(codes)%m != 0 {
		return nil, heads, fmt.Errorf("nodeclient: %d bytes of %d-byte codes", len(codes), m)
	}
	n := len(codes) / m
	var w frame.Writer
	w.U32(uint32(n))
	w.U32(uint32(m))
	w.Bytes(codes)
	b, err := c.do(ctx, opSplitList, 0, w.B)
	if err != nil {
		return nil, heads, err
	}
	r := frame.NewReader(b)
	heads[0], heads[1] = r.U32(), r.U32()
	side := append([]uint8(nil), r.Bytes(n)...)
	return side, heads, r.Err
}

// NearestHeads asks the router for the nearest live head of each decoded PQ code.
func (c *Client) NearestHeads(ctx context.Context, codes []byte, m, ef int) ([]uint32, error) {
	if m <= 0 || len(codes)%m != 0 {
		return nil, fmt.Errorf("nodeclient: %d bytes of %d-byte codes", len(codes), m)
	}
	n := len(codes) / m
	var w frame.Writer
	w.U32(uint32(n))
	w.U32(uint32(m))
	w.U32(uint32(ef))
	w.Bytes(codes)
	b, err := c.do(ctx, opNearestHeads, 0, w.B)
	if err != nil {
		return nil, err
	}
	r := frame.NewReader(b)
	lists := r.U32s(n)
	return lists, r.Err
}

// EditGraph has the router insert the heads labels (vecs: one vector each, in the index's
// element type) and mark the heads deleted deleted, taking its graph to version v; it returns
// the patch from v-1 to v.
func (c *Client) EditGraph(ctx context.Context, v uint32, labels []uint32, vecs []byte, deleted []uint32) ([]byte, error) {
	var w frame.Writer
	w.U32(v)
	w.U32(uint32(len(labels)))
	w.U32s(labels)
	w.Bytes(vecs)
	w.U32(uint32(len(deleted)))
	w.U32s(deleted)
	b, err := c.do(ctx, opEditGraph, 0, w.B)
	if err != nil {
		return nil, err
	}
	return append([]byte(nil), b...), nil
}

// DropVersion evicts version v of partition p (after the grace period of the epochs that saw
// it); it reports whether that version was resident.
func (c *Client) DropVersion(ctx context.Context, p int, v uint32) (bool, error) {
	var w frame.Writer
	w.U32(uint32(p))
	w.U32(v)
	b, err := c.do(ctx, opDropVersion, 0, w.B)
	if err != nil {
		return false, err
	}
	return len(b) == 1 && b[0] == 1, nil
}

// ForgetVectors frees the PQ codes of ids that no resident version names, and returns how
// many it freed. With deleted the vectors are deleted (U14): the codes a version still names
// are marked and freed with their last reference instead of being cached. Without it the
// call only frees codes known to be unused, and is safe whatever their versions' fate.
func (c *Client) ForgetVectors(ctx context.Context, ids []uint32, deleted bool) (uint64, error) {
	var w frame.Writer
	var flags uint32
	if deleted {
		flags = 1
	}
	w.U32(flags)
	w.U32(uint32(len(ids)))
	w.U32s(ids)
	b, err := c.do(ctx, opForget, 0, w.B)
	if err != nil {
		return 0, err
	}
	r := frame.NewReader(b)
	n := r.U64()
	return n, r.Err
}

// SavePartition writes version v of partition p as a segment file at path (on the node's
// host), so that a node the partition moves to can load that version.
func (c *Client) SavePartition(ctx context.Context, p int, v uint32, path string) error {
	var w frame.Writer
	w.U32(uint32(p))
	w.U32(v)
	w.Bytes([]byte(path))
	_, err := c.do(ctx, opSavePartition, 0, w.B)
	return err
}

// RawPull has the node fetch the vectors at locs that it lacks from the sources of their lists
// (lists[i] names locs[i]) -- data node to data node, as RERANK fetches -- and install them as
// RawPut does; vectors it already holds are skipped.
func (c *Client) RawPull(ctx context.Context, locs, lists []uint32) (installed, skipped int, err error) {
	if len(lists) != len(locs) {
		return 0, 0, fmt.Errorf("nodeclient: RAW_PULL of %d locations with %d lists", len(locs), len(lists))
	}
	var w frame.Writer
	w.U32(uint32(len(locs)))
	w.U32s(locs)
	w.U32s(lists)
	b, err := c.do(ctx, opRawPull, 0, w.B)
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

// Navigate returns the query's nprobe closest lists (closest first) on graph version graph (its
// epoch's; NewestGraph: the newest the node has).
func (c *Client) Navigate(ctx context.Context, epoch uint64, graph uint32, q []byte, nprobe, ef int) ([]uint32, error) {
	var w frame.Writer
	w.U32(uint32(nprobe))
	w.U32(uint32(ef))
	w.U32(graph)
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
// (PQ distance, ID). versions[i] is the version of lists[i]'s partition in the query's epoch
// (nil: all 0); StatusNotResident if the node does not hold that version.
func (c *Client) Filter(ctx context.Context, epoch uint64, q []byte, lists, versions []uint32, topn int) ([]Candidate, FilterStats, error) {
	vs, err := versionsFor(lists, versions)
	if err != nil {
		return nil, FilterStats{}, err
	}
	var w frame.Writer
	w.U32(uint32(topn))
	w.U32(uint32(len(lists)))
	w.U32s(lists)
	w.U32s(vs)
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
// (distance, ID), and the pages it read. lists are the query's lists on this node (the ones its
// FILTER got), at their versions (as in Filter): the node fetches a raw vector it lacks first,
// from the source of a list among them that names it (the reply's last field counts them; see
// Info's Raw.Fetched).
func (c *Client) Rerank(ctx context.Context, epoch uint64, q []byte, ids, lists, versions []uint32, k int) ([]Candidate, uint32, error) {
	vs, err := versionsFor(lists, versions)
	if err != nil {
		return nil, 0, err
	}
	var w frame.Writer
	w.U32(uint32(k))
	w.U32(uint32(len(ids)))
	w.U32s(ids)
	w.U32(uint32(len(lists)))
	w.U32s(lists)
	w.U32s(vs)
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

// versionsFor returns the partition version of each of lists: versions itself, or all 0 when
// it is nil (a caller without an epoch table, on a dataset without updates).
func versionsFor(lists, versions []uint32) ([]uint32, error) {
	if versions == nil {
		return make([]uint32, len(lists)), nil
	}
	if len(versions) != len(lists) {
		return nil, fmt.Errorf("nodeclient: %d versions for %d lists", len(versions), len(lists))
	}
	return versions, nil
}
