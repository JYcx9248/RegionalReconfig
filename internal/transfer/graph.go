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

// Graph pulls: the navigation graph from one entry's memory into another's (U16). Queries search
// the graph only in memory, so it never goes through a disk: the source's data node answers for
// any range of its loaded graph in hnswlib's index format (GRAPH_GET), and the destination's
// builds its graph from the bytes in order as they arrive (GRAPH_PUT, then GRAPH_INSTALL), so
// neither end holds a second copy either. The agents relay the bytes, through the source's token
// bucket and with the graph's yield to data at both ends, chunk by chunk as for a file.
//
//	client -> server  PULL_GRAPH  u8 class, u32 version, u32 chunk
//	server -> client  CHUNK       u32 0, u64 offset, data           (a file's chunk)
//	server -> client  DONE        JSON pullDone, for the one stream (a file pull's)
//	server -> client  ERROR       message
//
// Every chunk must come from one graph: the source checks its data node's graph stamp, which
// changes with each graph it loads and each change in place, and a transfer that sees it change
// fails instead of mixing two graphs' bytes.
const typePullGraph uint8 = 0x5A

// GraphSource reads bytes [off, off+n) of the graph at version from the server's data node
// (GRAPH_GET), with the format's size and the node's graph stamp.
type GraphSource func(ctx context.Context, version uint32, off int64, n int) (data []byte, size int64, stamp uint64, err error)

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
		data, sz, st, err := s.Graph(ctx, version, off, chunk)
		if err != nil {
			return err
		}
		if first {
			size, stamp = sz, st
		} else if sz != size || st != stamp {
			return fmt.Errorf("transfer: the graph changed during its transfer, at byte %d of %d", off, size)
		}
		if len(data) == 0 && off < size {
			return fmt.Errorf("transfer: no graph bytes at %d of %d", off, size)
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
	start := time.Now()
	var d net.Dialer
	c, err := d.DialContext(ctx, "tcp", addr)
	if err != nil {
		return Stats{}, err
	}
	defer c.Close()
	stop := context.AfterFunc(ctx, func() { c.SetDeadline(time.Now()) })
	defer stop()

	var w frame.Writer
	w.U8(uint8(class))
	w.U32(version)
	w.U32(uint32(max(chunk, 0)))
	if err := frame.Write(c, &frame.Frame{Type: typePullGraph, Body: w.B}); err != nil {
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
			return Stats{}, fmt.Errorf("transfer: graph pull from %s: %w", addr, err)
		}
		switch f.Type {
		case typeChunk:
			r := frame.NewReader(f.Body)
			idx, off := r.U32(), int64(r.U64())
			data := r.Rest()
			if r.Err != nil || idx != 0 || off != size {
				return Stats{}, fmt.Errorf("transfer: bad graph chunk from %s", addr)
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
				return Stats{}, fmt.Errorf("transfer: the graph arrived corrupted (%d/%d bytes, crc %08x/%08x)",
					size, done.Sizes[0], crc, done.CRCs[0])
			}
			return Stats{Bytes: size, Elapsed: time.Since(start)}, nil
		default:
			return Stats{}, fmt.Errorf("transfer: unexpected frame type %d", f.Type)
		}
	}
}
