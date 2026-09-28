// Package frame implements the rtier wire framing, shared by every rtier connection: the Go
// control plane, the query path and bulk transfer, and the C++ data-node service
// (engine/node/wire.h describes the same layout).
//
// The framing is Koala's state-comm TCP format (MAGIC_START | length | type | body |
// MAGIC_END; worker/stateCommUtil/stateCommTcpUtil.go), extended with a request ID, an epoch
// and a status field, and switched to little endian so vectors and ID arrays need no
// conversion:
//
//	off  size  field
//	0    4     magic_start   0x31465452 ("RTF1")
//	4    4     length        bytes after this field: 20 + len(body) + 4
//	8    1     type
//	9    1     flags         bit 0: response
//	10   2     status        0 = OK
//	12   8     req_id
//	20   8     epoch
//	28   ...   body
//	end  4     magic_end     0x444E4546 ("FEND")
package frame

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

const (
	MagicStart    uint32 = 0x31465452
	MagicEnd      uint32 = 0x444E4546
	HeaderBytes          = 28
	MaxFrameBytes        = 256 << 20
	FlagResponse  uint8  = 1
)

// Frame is one message.
type Frame struct {
	Type   uint8
	Flags  uint8
	Status uint16
	ReqID  uint64
	Epoch  uint64
	Body   []byte
}

// IsResponse reports whether the response flag is set.
func (f *Frame) IsResponse() bool { return f.Flags&FlagResponse != 0 }

var le = binary.LittleEndian

// Append encodes f onto dst.
func Append(dst []byte, f *Frame) ([]byte, error) {
	n := 20 + len(f.Body) + 4
	if n > MaxFrameBytes {
		return dst, fmt.Errorf("frame: body of %d bytes exceeds the frame limit", len(f.Body))
	}
	var h [HeaderBytes]byte
	le.PutUint32(h[0:], MagicStart)
	le.PutUint32(h[4:], uint32(n))
	h[8] = f.Type
	h[9] = f.Flags
	le.PutUint16(h[10:], f.Status)
	le.PutUint64(h[12:], f.ReqID)
	le.PutUint64(h[20:], f.Epoch)
	dst = append(dst, h[:]...)
	dst = append(dst, f.Body...)
	return le.AppendUint32(dst, MagicEnd), nil
}

// Write encodes f and writes it with a single Write call.
func Write(w io.Writer, f *Frame) error {
	buf, err := Append(make([]byte, 0, HeaderBytes+len(f.Body)+4), f)
	if err != nil {
		return err
	}
	_, err = w.Write(buf)
	return err
}

// ErrMalformed is returned for frames that violate the format.
var ErrMalformed = errors.New("frame: malformed frame")

// Read reads one frame into f. It returns io.EOF only on a clean end of stream before a frame
// starts; a stream that ends inside a frame gives io.ErrUnexpectedEOF.
func Read(r io.Reader, f *Frame) error {
	var h [8]byte
	if _, err := io.ReadFull(r, h[:]); err != nil {
		return err
	}
	if le.Uint32(h[0:]) != MagicStart {
		return fmt.Errorf("%w: bad magic", ErrMalformed)
	}
	n := le.Uint32(h[4:])
	if n < 24 || n > MaxFrameBytes {
		return fmt.Errorf("%w: bad length %d", ErrMalformed, n)
	}
	rest := make([]byte, n)
	if _, err := io.ReadFull(r, rest); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return err
	}
	if le.Uint32(rest[n-4:]) != MagicEnd {
		return fmt.Errorf("%w: bad trailer", ErrMalformed)
	}
	f.Type = rest[0]
	f.Flags = rest[1]
	f.Status = le.Uint16(rest[2:])
	f.ReqID = le.Uint64(rest[4:])
	f.Epoch = le.Uint64(rest[12:])
	f.Body = rest[20 : n-4]
	return nil
}

// Writer builds little-endian bodies.
type Writer struct{ B []byte }

func (w *Writer) U8(v uint8)     { w.B = append(w.B, v) }
func (w *Writer) U16(v uint16)   { w.B = le.AppendUint16(w.B, v) }
func (w *Writer) U32(v uint32)   { w.B = le.AppendUint32(w.B, v) }
func (w *Writer) U64(v uint64)   { w.B = le.AppendUint64(w.B, v) }
func (w *Writer) F32(v float32)  { w.B = le.AppendUint32(w.B, math.Float32bits(v)) }
func (w *Writer) Bytes(b []byte) { w.B = append(w.B, b...) }
func (w *Writer) Str(s string)   { w.B = append(w.B, s...) }
func (w *Writer) U32s(v []uint32) {
	for _, x := range v {
		w.B = le.AppendUint32(w.B, x)
	}
}
func (w *Writer) F32s(v []float32) {
	for _, x := range v {
		w.B = le.AppendUint32(w.B, math.Float32bits(x))
	}
}

// Reader parses little-endian bodies. The first short read sets Err and makes every later
// call return zero values, so callers check Err once at the end.
type Reader struct {
	B   []byte
	off int
	Err error
}

func NewReader(b []byte) *Reader { return &Reader{B: b} }

func (r *Reader) take(n int) []byte {
	if r.Err != nil {
		return nil
	}
	if n < 0 || r.off+n > len(r.B) {
		r.Err = fmt.Errorf("%w: body too short", ErrMalformed)
		return nil
	}
	b := r.B[r.off : r.off+n]
	r.off += n
	return b
}

func (r *Reader) U8() uint8 {
	if b := r.take(1); b != nil {
		return b[0]
	}
	return 0
}
func (r *Reader) U16() uint16 {
	if b := r.take(2); b != nil {
		return le.Uint16(b)
	}
	return 0
}
func (r *Reader) U32() uint32 {
	if b := r.take(4); b != nil {
		return le.Uint32(b)
	}
	return 0
}
func (r *Reader) U64() uint64 {
	if b := r.take(8); b != nil {
		return le.Uint64(b)
	}
	return 0
}
func (r *Reader) F32() float32 { return math.Float32frombits(r.U32()) }

// Bytes returns the next n bytes (aliasing the body).
func (r *Reader) Bytes(n int) []byte { return r.take(n) }

func (r *Reader) U32s(n int) []uint32 {
	b := r.take(4 * n)
	if b == nil {
		return nil
	}
	v := make([]uint32, n)
	for i := range v {
		v[i] = le.Uint32(b[4*i:])
	}
	return v
}

func (r *Reader) F32s(n int) []float32 {
	b := r.take(4 * n)
	if b == nil {
		return nil
	}
	v := make([]float32, n)
	for i := range v {
		v[i] = math.Float32frombits(le.Uint32(b[4*i:]))
	}
	return v
}

// Rest returns the unread remainder.
func (r *Reader) Rest() []byte {
	if r.Err != nil {
		return nil
	}
	b := r.B[r.off:]
	r.off = len(r.B)
	return b
}

// Remaining is the number of unread bytes.
func (r *Reader) Remaining() int { return len(r.B) - r.off }
