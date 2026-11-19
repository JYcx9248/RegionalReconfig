package main

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// The answers file as scripts/score_answers.py reads it: the header, then fixed-size records in
// send order, each with its answer's length, the answer padded to k.
func TestWriteAnswers(t *testing.T) {
	path := filepath.Join(t.TempDir(), "answers.bin")
	ans := []answer{
		{query: 30, epoch: 4, sched: 2500 * time.Millisecond, ids: []uint32{7, 9}, dists: []float32{0.5, 1.25}},
		{query: 0, epoch: 2, sched: 1500 * time.Microsecond, ids: []uint32{3, 1, 4}, dists: []float32{0, 1, 2}},
	}
	if err := writeAnswers(path, 3, 10, ans); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	const k, rec = 3, 24 + 8*3
	if len(b) != 24+2*rec || string(b[:8]) != answersMagic {
		t.Fatalf("%d bytes, magic %q", len(b), b[:8])
	}
	le := binary.LittleEndian
	if le.Uint32(b[8:]) != k || le.Uint32(b[12:]) != 10 || le.Uint64(b[16:]) != 2 {
		t.Fatalf("header: k %d, stride %d, count %d", le.Uint32(b[8:]), le.Uint32(b[12:]), le.Uint64(b[16:]))
	}
	type got struct {
		query, n uint32
		epoch    uint64
		schedUS  uint64
		ids      [k]uint32
		dists    [k]float32
	}
	parse := func(r []byte) (g got) {
		g.query, g.n, g.epoch, g.schedUS = le.Uint32(r), le.Uint32(r[4:]), le.Uint64(r[8:]), le.Uint64(r[16:])
		for i := 0; i < k; i++ {
			g.ids[i] = le.Uint32(r[24+4*i:])
			g.dists[i] = math.Float32frombits(le.Uint32(r[24+4*k+4*i:]))
		}
		return g
	}
	// In send order: query 0 (sent at 1.5 ms) first.
	first, second := parse(b[24:24+rec]), parse(b[24+rec:])
	if first.query != 0 || first.n != 3 || first.epoch != 2 || first.schedUS != 1500 ||
		first.ids != [k]uint32{3, 1, 4} || first.dists != [k]float32{0, 1, 2} {
		t.Errorf("first record %+v", first)
	}
	if second.query != 30 || second.n != 2 || second.epoch != 4 || second.schedUS != 2500000 ||
		second.ids != [k]uint32{7, 9, math.MaxUint32} || second.dists[0] != 0.5 || second.dists[1] != 1.25 ||
		!math.IsNaN(float64(second.dists[2])) {
		t.Errorf("second record %+v", second)
	}
}
