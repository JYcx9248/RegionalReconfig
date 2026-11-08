package transfer

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"net"
	"time"

	"rtier/internal/frame"
)

// Pulls from memory (U16): the navigation graph from one entry's memory into another's, and a
// partition's posting lists from its owner's memory into its next owner's. Queries use both only
// in memory, so neither goes through a disk: the source's data node answers for any range of the
// image -- hnswlib's index format of its loaded graph (GRAPH_GET), the segment format of a
// resident partition (SEGMENT_GET, every posting with its current holder) -- and the
// destination's builds the graph or the segment from the bytes in order as they arrive
// (GRAPH_PUT then GRAPH_INSTALL, SEGMENT_PUT then SEGMENT_INSTALL once the PQ codes are in), so
// neither end holds a second copy either. The agents relay the bytes through the source's token
// bucket, chunk by chunk as for a file: the lists at the data class, the graph at the background
// class by default, with the graph's yield to data at both ends.
//
//	client -> server  PULL_GRAPH    u8 class, u32 version, u32 chunk
//	client -> server  PULL_SEGMENT  u8 class, u32 partition, u32 version, u32 chunk
//	server -> client  CHUNK         u32 0, u64 offset, data           (a file's chunk)
//	server -> client  DONE          JSON pullDone, for the one stream (a file pull's)
//	server -> client  ERROR         message
//
// Every chunk must come from one image: the source checks its data node's stamp -- of the graph,
// which changes with each graph it loads and each change in place; of the segment, which differs
// for each one loaded -- and a transfer that sees it change fails instead of mixing two images'
// bytes. A segment's holders may change between chunks (a fetch makes a posting name the node
// that fetched): each is a holder still, since no node drops a vector it has held.
const (
	typePullGraph   uint8 = 0x5A
	typePullSegment uint8 = 0x5B
)

// GraphSource reads bytes [off, off+n) of the graph at version from the server's data node
// (GRAPH_GET), with the format's size and the node's graph stamp.
type GraphSource func(ctx context.Context, version uint32, off int64, n int) (data []byte, size int64, stamp uint64, err error)

// SegmentSource reads bytes [off, off+n) of the image of version v of partition p from the
// server's data node (SEGMENT_GET), with the image's size and the segment's stamp.
type SegmentSource func(ctx context.Context, p int, v uint32, off int64, n int) (data []byte, size int64, stamp uint64, err error)

// imageRead reads bytes [off, off+n) of one image, with its size and stamp.
type imageRead func(ctx context.Context, off int64, n int) (data []byte, size int64, stamp uint64, err error)

// sendGraph answers one PULL_GRAPH request.
func (s *Server) sendGraph(bw *bufio.Writer, body []byte) error {
	if s.Graph == nil {
		return errors.New("transfer: this node does not serve its graph")
	}
	r := frame.NewReader(body)
	class, version, chunk := Class(r.U8()), r.U32(), int(r.U32())
	if r.Err != nil || r.Remaining() != 0 || (class != Data && class != Background) {
		return errors.New("transfer: bad PULL_GRAPH request")
	}
	return s.sendImage(bw, class, chunk, "the graph", func(ctx context.Context, off int64, n int) ([]byte, int64, uint64, error) {
		return s.Graph(ctx, version, off, n)
	})
}

// sendSegment answers one PULL_SEGMENT request.
func (s *Server) sendSegment(bw *bufio.Writer, body []byte) error {
	if s.Segment == nil {
		return errors.New("transfer: this node does not serve its partitions")
	}
	r := frame.NewReader(body)
	class, p, version, chunk := Class(r.U8()), int(r.U32()), r.U32(), int(r.U32())
	if r.Err != nil || r.Remaining() != 0 || (class != Data && class != Background) {
		return errors.New("transfer: bad PULL_SEGMENT request")
	}
	return s.sendImage(bw, class, chunk, fmt.Sprintf("partition %d", p), func(ctx context.Context, off int64, n int) ([]byte, int64, uint64, error) {
		return s.Segment(ctx, p, version, off, n)
	})
}

// sendImage sends the image read reads, in chunks of chunk bytes (0: the server's), at class on
// the bucket, then its size and CRC-32.
func (s *Server) sendImage(bw *bufio.Writer, class Class, chunk int, what string, read imageRead) error {
	if s.Limiter != nil {
		defer s.Limiter.Begin(class)()
	}
	if chunk <= 0 {
		chunk = s.ChunkBytes
	}
	if chunk <= 0 {
		chunk = 1 << 20
	}
	chunk = min(chunk, frame.MaxFrameBytes/2)
	ctx := context.Background()
	crc := crc32.NewIEEE()
	var off, size int64
	var stamp uint64
	for first := true; first || off < size; first = false {
		if class == Background && s.Yield != nil {
			if err := s.Yield(); err != nil {
				return err
			}
		}
		data, sz, st, err := read(ctx, off, chunk)
		if err != nil {
			return err
		}
		if first {
			size, stamp = sz, st
		} else if sz != size || st != stamp {
			return fmt.Errorf("transfer: %s changed during its transfer, at byte %d of %d", what, off, size)
		}
		if len(data) == 0 && off < size {
			return fmt.Errorf("transfer: no bytes of %s at %d of %d", what, off, size)
		}
		if s.Limiter != nil {
			if err := s.Limiter.WaitClass(ctx, len(data), class); err != nil {
				return err
			}
		}
		crc.Write(data)
		var w frame.Writer
		w.U32(0)
		w.U64(uint64(off))
		w.Bytes(data)
		if err := frame.Write(bw, &frame.Frame{Type: typeChunk, Flags: frame.FlagResponse, Body: w.B}); err != nil {
			return err
		}
		off += int64(len(data))
		s.BytesSent.Add(int64(len(data)))
	}
	b, _ := json.Marshal(pullDone{Sizes: []int64{off}, CRCs: []uint32{crc.Sum32()}})
	if err := frame.Write(bw, &frame.Frame{Type: typeDone, Flags: frame.FlagResponse, Body: b}); err != nil {
		return err
	}
	return bw.Flush()
}

// PullGraph fetches the graph at version from the server at addr, at priority class on the
// sender's bucket, and hands each chunk, in order, to put (the destination's data node,
// GRAPH_PUT). gate (if not nil) is called before each chunk is read, as in PullGated. It returns
// once every byte has gone to put and arrived intact -- the sender's size and CRC-32 -- for the
// caller to install the graph; a pull that fails leaves the destination with part of a graph,
// which the caller drops.
func PullGraph(ctx context.Context, addr string, version uint32, chunk int, class Class,
	gate func(context.Context) error, put func(ctx context.Context, off int64, data []byte) error) (Stats, error) {
	var w frame.Writer
	w.U8(uint8(class))
	w.U32(version)
	w.U32(uint32(max(chunk, 0)))
	return pullImage(ctx, addr, typePullGraph, w.B, "the graph", gate, put)
}

// PullSegment fetches version v of partition p's lists from the server at addr, its owner, at
// priority class on the sender's bucket, and hands each chunk, in order, to put (the
// destination's data node, SEGMENT_PUT). It returns once every byte has gone to put and arrived
// intact, as PullGraph; a pull that fails leaves the destination with part of the segment,
// which the caller drops.
func PullSegment(ctx context.Context, addr string, p int, v uint32, chunk int, class Class,
	put func(ctx context.Context, off int64, data []byte) error) (Stats, error) {
	var w frame.Writer
	w.U8(uint8(class))
	w.U32(uint32(p))
	w.U32(v)
	w.U32(uint32(max(chunk, 0)))
	return pullImage(ctx, addr, typePullSegment, w.B, fmt.Sprintf("partition %d", p), nil, put)
}

// pullImage sends one pull request from memory (req of type kind) and hands the image's chunks
// to put, checking them against the sender's size and CRC-32.
func pullImage(ctx context.Context, addr string, kind uint8, req []byte, what string,
	gate func(context.Context) error, put func(ctx context.Context, off int64, data []byte) error) (Stats, error) {
	start := time.Now()
	var d net.Dialer
	c, err := d.DialContext(ctx, "tcp", addr)
	if err != nil {
		return Stats{}, err
	}
	defer c.Close()
	stop := context.AfterFunc(ctx, func() { c.SetDeadline(time.Now()) })
	defer stop()
	if err := frame.Write(c, &frame.Frame{Type: kind, Body: req}); err != nil {
		return Stats{}, err
	}
	br := bufio.NewReaderSize(c, 256<<10)
	var size int64
	var crc uint32
	for {
		if gate != nil {
			if err := gate(ctx); err != nil {
				return Stats{}, err
			}
		}
		var f frame.Frame
		if err := frame.Read(br, &f); err != nil {
			if ctx.Err() != nil {
				return Stats{}, ctx.Err()
			}
			return Stats{}, fmt.Errorf("transfer: pull of %s from %s: %w", what, addr, err)
		}
		switch f.Type {
		case typeChunk:
			r := frame.NewReader(f.Body)
			idx, off := r.U32(), int64(r.U64())
			data := r.Rest()
			if r.Err != nil || idx != 0 || off != size {
				return Stats{}, fmt.Errorf("transfer: bad chunk of %s from %s", what, addr)
			}
			if err := put(ctx, off, data); err != nil {
				return Stats{}, err
			}
			crc = crc32.Update(crc, crc32.IEEETable, data)
			size += int64(len(data))
		case typeError:
			return Stats{}, fmt.Errorf("transfer: %s: %s", addr, string(f.Body))
		case typeDone:
			var done pullDone
			if err := json.Unmarshal(f.Body, &done); err != nil || len(done.Sizes) != 1 || len(done.CRCs) != 1 {
				return Stats{}, fmt.Errorf("transfer: bad completion from %s", addr)
			}
			if done.Sizes[0] != size || done.CRCs[0] != crc {
				return Stats{}, fmt.Errorf("transfer: %s arrived corrupted (%d/%d bytes, crc %08x/%08x)",
					what, size, done.Sizes[0], crc, done.CRCs[0])
			}
			return Stats{Bytes: size, Elapsed: time.Since(start)}, nil
		default:
			return Stats{}, fmt.Errorf("transfer: unexpected frame type %d", f.Type)
		}
	}
}
