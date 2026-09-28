package transfer

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"log"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"

	"rtier/internal/frame"
)

// Frame types of the pull protocol.
const (
	typePull  uint8 = 0x50 // client -> server: JSON pullRequest
	typeChunk uint8 = 0x51 // server -> client: u32 file index, u64 offset, data
	typeDone  uint8 = 0x52 // server -> client: JSON pullDone
	typeError uint8 = 0x53 // server -> client: message
)

type pullRequest struct {
	Files    []string `json:"files"`
	Chunk    int      `json:"chunk"`
	Priority string   `json:"priority,omitempty"` // Class; "" = data
}

type pullDone struct {
	Sizes []int64  `json:"sizes"`
	CRCs  []uint32 `json:"crcs"`
}

// Server serves files under Root, and PQ codes and raw vectors read through PQ and Raw
// (records.go), to pulling peers. All connections share one token bucket: it caps this node's
// outgoing background bandwidth, and within it data transfers (segments, PQ codes) go before
// background ones (the graph, and by default the raw vectors streamed after the flip).
type Server struct {
	Root       string
	PQ         PQSource     // nil = PULL_PQ is refused
	Raw        RawSource    // nil = PULL_RAW is refused
	Limiter    *TokenBucket // nil = unlimited
	ChunkBytes int          // default chunk size (Koala: StateMigrationChunkSize, 1 MiB)
	BytesSent  atomic.Int64
}

// Serve accepts pull connections until ln is closed.
func (s *Server) Serve(ln net.Listener) error {
	for {
		c, err := ln.Accept()
		if err != nil {
			if errors.Is(err, net.ErrClosed) {
				return nil
			}
			return err
		}
		go s.serveConn(c)
	}
}

func (s *Server) serveConn(c net.Conn) {
	defer c.Close()
	br := bufio.NewReader(c)
	bw := bufio.NewWriterSize(c, 256<<10)
	for {
		var f frame.Frame
		if err := frame.Read(br, &f); err != nil {
			return
		}
		var err error
		switch f.Type {
		case typePull:
			var req pullRequest
			if err = json.Unmarshal(f.Body, &req); err == nil {
				err = s.sendFiles(bw, &req)
			}
		case typePullPQ:
			err = s.sendPQ(bw, f.Body)
		case typePullRaw:
			err = s.sendRaw(bw, f.Body)
		default:
			return
		}
		if err != nil {
			log.Printf("transfer: pull from %s failed: %v", c.RemoteAddr(), err)
			var w frame.Writer
			w.Str(err.Error())
			_ = frame.Write(bw, &frame.Frame{Type: typeError, Flags: frame.FlagResponse, Body: w.B})
			_ = bw.Flush()
			return
		}
	}
}

func (s *Server) sendFiles(bw *bufio.Writer, req *pullRequest) error {
	class, err := ParseClass(req.Priority)
	if err != nil {
		return err
	}
	if s.Limiter != nil {
		defer s.Limiter.Begin(class)()
	}
	chunk := req.Chunk
	if chunk <= 0 {
		chunk = s.ChunkBytes
	}
	if chunk <= 0 {
		chunk = 1 << 20
	}
	chunk = min(chunk, frame.MaxFrameBytes/2)
	done := pullDone{}
	buf := make([]byte, chunk)
	ctx := context.Background()
	for i, name := range req.Files {
		path, err := s.resolve(name)
		if err != nil {
			return err
		}
		fh, err := os.Open(path)
		if err != nil {
			return err
		}
		crc := crc32.NewIEEE()
		var off int64
		for {
			n, rerr := io.ReadFull(fh, buf)
			if n > 0 {
				if s.Limiter != nil {
					if err := s.Limiter.WaitClass(ctx, n, class); err != nil {
						fh.Close()
						return err
					}
				}
				crc.Write(buf[:n])
				var w frame.Writer
				w.U32(uint32(i))
				w.U64(uint64(off))
				w.Bytes(buf[:n])
				if err := frame.Write(bw, &frame.Frame{Type: typeChunk, Flags: frame.FlagResponse, Body: w.B}); err != nil {
					fh.Close()
					return err
				}
				off += int64(n)
				s.BytesSent.Add(int64(n))
			}
			if rerr == io.EOF || rerr == io.ErrUnexpectedEOF {
				break
			}
			if rerr != nil {
				fh.Close()
				return rerr
			}
		}
		fh.Close()
		done.Sizes = append(done.Sizes, off)
		done.CRCs = append(done.CRCs, crc.Sum32())
	}
	b, _ := json.Marshal(done)
	if err := frame.Write(bw, &frame.Frame{Type: typeDone, Flags: frame.FlagResponse, Body: b}); err != nil {
		return err
	}
	return bw.Flush()
}

// resolve maps a requested name to a path under Root, rejecting anything that escapes it.
func (s *Server) resolve(name string) (string, error) {
	clean := filepath.Clean(name)
	if filepath.IsAbs(clean) || clean == "." || strings.HasPrefix(clean, "..") {
		return "", fmt.Errorf("transfer: refusing to serve %q", name)
	}
	return filepath.Join(s.Root, clean), nil
}

// Stats describes one pull.
type Stats struct {
	Files   int
	Codes   int64 // PQ codes (PullPQ)
	Vectors int64 // raw vectors (PullRaw)
	Bytes   int64
	Elapsed time.Duration
}

// Pull fetches files from the server at addr into destDir (same relative names), at priority
// class on the sender's bucket. Each file is written to "<name>.part", checked against the
// sender's size and CRC-32, then renamed.
func Pull(ctx context.Context, addr string, files []string, destDir string, chunk int, class Class) (Stats, error) {
	start := time.Now()
	var d net.Dialer
	c, err := d.DialContext(ctx, "tcp", addr)
	if err != nil {
		return Stats{}, err
	}
	defer c.Close()
	stop := context.AfterFunc(ctx, func() { c.SetDeadline(time.Now()) })
	defer stop()

	req := pullRequest{Files: files, Chunk: chunk}
	if class != Data {
		req.Priority = class.String()
	}
	body, _ := json.Marshal(req)
	if err := frame.Write(c, &frame.Frame{Type: typePull, Body: body}); err != nil {
		return Stats{}, err
	}
	outs := make([]*os.File, len(files))
	crcs := make([]uint32, len(files))
	sizes := make([]int64, len(files))
	cleanup := func() {
		for i, f := range outs {
			if f != nil {
				f.Close()
				os.Remove(tmpName(destDir, files[i]))
			}
		}
	}
	for i, name := range files {
		p := tmpName(destDir, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			cleanup()
			return Stats{}, err
		}
		if outs[i], err = os.Create(p); err != nil {
			cleanup()
			return Stats{}, err
		}
	}
	br := bufio.NewReaderSize(c, 256<<10)
	for {
		var f frame.Frame
		if err := frame.Read(br, &f); err != nil {
			cleanup()
			if ctx.Err() != nil {
				return Stats{}, ctx.Err()
			}
			return Stats{}, fmt.Errorf("transfer: pull from %s: %w", addr, err)
		}
		switch f.Type {
		case typeChunk:
			r := frame.NewReader(f.Body)
			idx, off := int(r.U32()), int64(r.U64())
			data := r.Rest()
			if r.Err != nil || idx >= len(files) || off != sizes[idx] {
				cleanup()
				return Stats{}, fmt.Errorf("transfer: bad chunk from %s", addr)
			}
			if _, err := outs[idx].Write(data); err != nil {
				cleanup()
				return Stats{}, err
			}
			crcs[idx] = crc32.Update(crcs[idx], crc32.IEEETable, data)
			sizes[idx] += int64(len(data))
		case typeError:
			cleanup()
			return Stats{}, fmt.Errorf("transfer: %s: %s", addr, string(f.Body))
		case typeDone:
			var done pullDone
			if err := json.Unmarshal(f.Body, &done); err != nil || len(done.Sizes) != len(files) {
				cleanup()
				return Stats{}, fmt.Errorf("transfer: bad completion from %s", addr)
			}
			st := Stats{Files: len(files)}
			for i := range files {
				if done.Sizes[i] != sizes[i] || done.CRCs[i] != crcs[i] {
					cleanup()
					return Stats{}, fmt.Errorf("transfer: %s arrived corrupted (%d/%d bytes, crc %08x/%08x)",
						files[i], sizes[i], done.Sizes[i], crcs[i], done.CRCs[i])
				}
				st.Bytes += sizes[i]
			}
			for i, name := range files {
				if err := outs[i].Close(); err != nil {
					cleanup()
					return Stats{}, err
				}
				outs[i] = nil
				if err := os.Rename(tmpName(destDir, name), filepath.Join(destDir, name)); err != nil {
					return Stats{}, err
				}
			}
			st.Elapsed = time.Since(start)
			return st, nil
		default:
			cleanup()
			return Stats{}, fmt.Errorf("transfer: unexpected frame type %d", f.Type)
		}
	}
}

func tmpName(dir, name string) string { return filepath.Join(dir, name) + ".part" }

// CopyLocal copies files from srcDir into destDir (hard link when possible). Used to seed the
// initial deployment from the shared build output on a single machine.
func CopyLocal(srcDir string, files []string, destDir string) (Stats, error) {
	start := time.Now()
	st := Stats{Files: len(files)}
	for _, name := range files {
		src, dst := filepath.Join(srcDir, name), filepath.Join(destDir, name)
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return st, err
		}
		fi, err := os.Stat(src)
		if err != nil {
			return st, err
		}
		st.Bytes += fi.Size()
		os.Remove(dst)
		if os.Link(src, dst) == nil {
			continue
		}
		in, err := os.Open(src)
		if err != nil {
			return st, err
		}
		out, err := os.Create(dst + ".part")
		if err == nil {
			_, err = io.Copy(out, in)
			if cerr := out.Close(); err == nil {
				err = cerr
			}
		}
		in.Close()
		if err == nil {
			err = os.Rename(dst+".part", dst)
		}
		if err != nil {
			return st, err
		}
	}
	st.Elapsed = time.Since(start)
	return st, nil
}
