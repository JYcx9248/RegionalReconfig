package transfer

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"hash/crc32"
	"net"
	"time"

	"rtier/internal/frame"
)

// Record pulls: PQ codes and raw vectors, the node-level state of rtier's data nodes (the
// decided part of design question U1). Both are fixed-size records named by a u32 key -- a
// vector ID for a PQ code, a canonical location for a raw vector -- and a destination asks each
// source for exactly the ones it lacks: its data node computes them, deduplicated against what
// it holds and across sources (nodeclient.PQMissing, nodeclient.RawMissing). The source reads
// them from its own data node and streams them back through the same token bucket as files.
//
//	client -> server  PULL_PQ   u32 n, u32 ids[n]                (always the data class)
//	client -> server  PULL_RAW  u8 class, u32 n, u32 locs[n]
//	server -> client  *_DATA    u32 size, u32 first, records of keys[first..] (whole records)
//	server -> client  *_DONE    u32 size, u32 n, u32 crc32 of all records in order
//	server -> client  ERROR     message
//
// One connection carries any number of requests (the client sends one per batch of keys).
const (
	typePullPQ  uint8 = 0x54
	typePQData  uint8 = 0x55
	typePQDone  uint8 = 0x56
	typePullRaw uint8 = 0x57
	typeRawData uint8 = 0x58
	typeRawDone uint8 = 0x59
)

// DefaultPQBatch is the number of IDs per PULL_PQ request.
const DefaultPQBatch = 1 << 16

// DefaultRawBatch is the number of locations per PULL_RAW request (vectors are up to a page, so
// a batch is at most 32 MB).
const DefaultRawBatch = 1 << 13

// Keys the server asks its data node for at a time.
const (
	nodePQBatch  = 1 << 14
	nodeRawBatch = 1 << 12
)

// RecordSource reads records for the server: size bytes per key, in the order of keys. The
// agent wires it to its data node (PQ_GET for PQ codes, RAW_GET for raw vectors).
type RecordSource func(ctx context.Context, keys []uint32) (size int, data []byte, err error)

// PQSource reads PQ codes (m bytes per vector ID).
type PQSource = RecordSource

// RawSource reads raw vectors (vec_bytes per location).
type RawSource = RecordSource

type recordKind struct {
	what             string // for messages
	pull, data, done uint8
	nodeBatch        int
	defaultPullBatch int
}

var (
	pqRecords  = recordKind{"PQ codes", typePullPQ, typePQData, typePQDone, nodePQBatch, DefaultPQBatch}
	rawRecords = recordKind{"raw vectors", typePullRaw, typeRawData, typeRawDone, nodeRawBatch, DefaultRawBatch}
)

// sendPQ answers one PULL_PQ request.
func (s *Server) sendPQ(bw *bufio.Writer, body []byte) error {
	if s.PQ == nil {
		return errors.New("transfer: this node does not serve PQ codes")
	}
	r := frame.NewReader(body)
	ids := r.U32s(int(r.U32()))
	if r.Err != nil || r.Remaining() != 0 {
		return errors.New("transfer: bad PULL_PQ request")
	}
	return s.sendRecords(bw, pqRecords, s.PQ, Data, ids)
}

// sendRaw answers one PULL_RAW request.
func (s *Server) sendRaw(bw *bufio.Writer, body []byte) error {
	if s.Raw == nil {
		return errors.New("transfer: this node does not serve raw vectors")
	}
	r := frame.NewReader(body)
	c := Class(r.U8())
	locs := r.U32s(int(r.U32()))
	if r.Err != nil || r.Remaining() != 0 || (c != Data && c != Background) {
		return errors.New("transfer: bad PULL_RAW request")
	}
	return s.sendRecords(bw, rawRecords, s.Raw, c, locs)
}

func (s *Server) sendRecords(bw *bufio.Writer, k recordKind, src RecordSource, class Class, keys []uint32) error {
	if s.Limiter != nil {
		defer s.Limiter.Begin(class)()
	}
	chunk := s.ChunkBytes
	if chunk <= 0 {
		chunk = 1 << 20
	}
	chunk = min(chunk, frame.MaxFrameBytes/2)
	ctx := context.Background()
	crc := crc32.NewIEEE()
	size := 0
	for first := 0; first < len(keys); first += k.nodeBatch {
		batch := keys[first:min(first+k.nodeBatch, len(keys))]
		bs, data, err := src(ctx, batch)
		if err != nil {
			return err
		}
		if bs <= 0 || len(data) != bs*len(batch) || (size != 0 && bs != size) {
			return fmt.Errorf("transfer: %s source returned %d bytes of %d-byte records for %d keys", k.what, len(data), bs, len(batch))
		}
		size = bs
		per := max(chunk/size, 1) // whole records per frame
		for off := 0; off < len(batch); off += per {
			part := data[off*size : min(off+per, len(batch))*size]
			if s.Limiter != nil {
				if err := s.Limiter.WaitClass(ctx, len(part), class); err != nil {
					return err
				}
			}
			crc.Write(part)
			var w frame.Writer
			w.U32(uint32(size))
			w.U32(uint32(first + off))
			w.Bytes(part)
			if err := frame.Write(bw, &frame.Frame{Type: k.data, Flags: frame.FlagResponse, Body: w.B}); err != nil {
				return err
			}
			s.BytesSent.Add(int64(len(part)))
		}
	}
	var w frame.Writer
	w.U32(uint32(size))
	w.U32(uint32(len(keys)))
	w.U32(crc.Sum32())
	if err := frame.Write(bw, &frame.Frame{Type: k.done, Flags: frame.FlagResponse, Body: w.B}); err != nil {
		return err
	}
	return bw.Flush()
}

// PullPQ fetches the PQ codes of ids from the server at addr, batch IDs per request (0 =
// DefaultPQBatch), and hands every verified batch to sink (the agent: its data node's PQ_PUT)
// before requesting the next one. Stats.Codes counts the codes.
func PullPQ(ctx context.Context, addr string, ids []uint32, batch int,
	sink func(ids []uint32, m int, codes []byte) error) (Stats, error) {
	st, err := pullRecords(ctx, addr, pqRecords, Data, ids, batch, nil, sink)
	st.Codes, st.Vectors = st.Vectors, 0
	return st, err
}

// PullRaw fetches the raw vectors at locs from the server at addr at priority class on the
// server's bucket, batch locations per request (0 = DefaultRawBatch), and hands every verified
// batch to sink (the agent: its data node's RAW_PUT). Before each request, still (if not nil)
// keeps only the locations the destination still lacks: queries fetch vectors on demand while
// the stream runs, and those need not come twice. Stats.Vectors counts the vectors received.
func PullRaw(ctx context.Context, addr string, locs []uint32, batch int, class Class,
	still func(locs []uint32) ([]uint32, error),
	sink func(locs []uint32, vecBytes int, vecs []byte) error) (Stats, error) {
	return pullRecords(ctx, addr, rawRecords, class, locs, batch, still, sink)
}

func pullRecords(ctx context.Context, addr string, k recordKind, class Class, keys []uint32, batch int,
	still func(keys []uint32) ([]uint32, error), sink func(keys []uint32, size int, data []byte) error) (Stats, error) {
	start := time.Now()
	if len(keys) == 0 {
		return Stats{}, nil
	}
	if batch <= 0 {
		batch = k.defaultPullBatch
	}
	var d net.Dialer
	c, err := d.DialContext(ctx, "tcp", addr)
	if err != nil {
		return Stats{}, err
	}
	defer c.Close()
	stop := context.AfterFunc(ctx, func() { c.SetDeadline(time.Now()) })
	defer stop()
	br := bufio.NewReaderSize(c, 256<<10)
	var st Stats
	for first := 0; first < len(keys); first += batch {
		part := keys[first:min(first+batch, len(keys))]
		if still != nil {
			if part, err = still(part); err != nil {
				return st, err
			}
			if len(part) == 0 {
				continue
			}
		}
		var w frame.Writer
		if k.pull == typePullRaw {
			w.U8(uint8(class))
		}
		w.U32(uint32(len(part)))
		w.U32s(part)
		err := frame.Write(c, &frame.Frame{Type: k.pull, Body: w.B})
		var size int
		var data []byte
		if err == nil {
			size, data, err = readRecords(br, addr, k, len(part))
		}
		if err != nil {
			if ctx.Err() != nil {
				return st, ctx.Err()
			}
			return st, err
		}
		if err := sink(part, size, data); err != nil {
			return st, err
		}
		st.Vectors += int64(len(part))
		st.Bytes += int64(len(data))
	}
	st.Elapsed = time.Since(start)
	return st, nil
}

// readRecords reads the frames answering one pull of n keys.
func readRecords(br *bufio.Reader, addr string, k recordKind, n int) (int, []byte, error) {
	var data []byte
	var crc uint32
	size := 0
	for {
		var f frame.Frame
		if err := frame.Read(br, &f); err != nil {
			return 0, nil, fmt.Errorf("transfer: %s pull from %s: %w", k.what, addr, err)
		}
		switch f.Type {
		case k.data:
			r := frame.NewReader(f.Body)
			fs, first := int(r.U32()), int(r.U32())
			part := r.Rest()
			if r.Err != nil || fs <= 0 || (size != 0 && fs != size) || len(part)%fs != 0 ||
				first*fs != len(data) || len(data)+len(part) > n*fs {
				return 0, nil, fmt.Errorf("transfer: bad %s chunk from %s", k.what, addr)
			}
			size = fs
			data = append(data, part...)
			crc = crc32.Update(crc, crc32.IEEETable, part)
		case k.done:
			r := frame.NewReader(f.Body)
			ds, dn, dcrc := int(r.U32()), int(r.U32()), r.U32()
			if r.Err != nil || dn != n || (n > 0 && (ds != size || len(data) != n*size)) || dcrc != crc {
				return 0, nil, fmt.Errorf("transfer: %s from %s arrived corrupted (%d bytes for %d keys, crc %08x/%08x)",
					k.what, addr, len(data), n, crc, dcrc)
			}
			return size, data, nil
		case typeError:
			return 0, nil, fmt.Errorf("transfer: %s: %s", addr, string(f.Body))
		default:
			return 0, nil, fmt.Errorf("transfer: unexpected frame type %d", f.Type)
		}
	}
}
