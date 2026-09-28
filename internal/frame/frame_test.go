package frame

import (
	"bytes"
	"errors"
	"io"
	"testing"
)

func TestRoundTrip(t *testing.T) {
	var w Writer
	w.U32(7)
	w.F32(1.5)
	w.U32s([]uint32{1, 2, 3})
	w.Str("path")
	in := &Frame{Type: 0x21, Flags: FlagResponse, Status: 2, ReqID: 99, Epoch: 5, Body: w.B}
	var buf bytes.Buffer
	if err := Write(&buf, in); err != nil {
		t.Fatal(err)
	}
	if buf.Len() != HeaderBytes+len(w.B)+4 {
		t.Fatalf("encoded %d bytes", buf.Len())
	}
	var out Frame
	if err := Read(&buf, &out); err != nil {
		t.Fatal(err)
	}
	if out.Type != in.Type || out.Flags != in.Flags || out.Status != in.Status ||
		out.ReqID != in.ReqID || out.Epoch != in.Epoch || !out.IsResponse() {
		t.Fatalf("header mismatch: %+v", out)
	}
	r := NewReader(out.Body)
	if r.U32() != 7 || r.F32() != 1.5 {
		t.Fatal("scalar mismatch")
	}
	if v := r.U32s(3); len(v) != 3 || v[2] != 3 {
		t.Fatalf("array mismatch: %v", v)
	}
	if string(r.Rest()) != "path" || r.Err != nil {
		t.Fatal("rest mismatch")
	}
	if err := Read(&buf, &out); err != io.EOF {
		t.Fatalf("want io.EOF at end of stream, got %v", err)
	}
}

func TestMalformed(t *testing.T) {
	var buf bytes.Buffer
	_ = Write(&buf, &Frame{Type: 1, Body: []byte{1, 2, 3}})
	raw := buf.Bytes()

	bad := append([]byte(nil), raw...)
	bad[0] ^= 0xFF
	var f Frame
	if err := Read(bytes.NewReader(bad), &f); !errors.Is(err, ErrMalformed) {
		t.Fatalf("bad magic: got %v", err)
	}
	bad = append([]byte(nil), raw...)
	bad[len(bad)-1] ^= 0xFF
	if err := Read(bytes.NewReader(bad), &f); !errors.Is(err, ErrMalformed) {
		t.Fatalf("bad trailer: got %v", err)
	}
	if err := Read(bytes.NewReader(raw[:len(raw)-2]), &f); err != io.ErrUnexpectedEOF {
		t.Fatalf("truncated: got %v", err)
	}
	r := NewReader([]byte{1, 2})
	_ = r.U32()
	if r.Err == nil || r.U8() != 0 {
		t.Fatal("short read not reported")
	}
}
