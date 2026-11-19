package main

import (
	"bufio"
	"encoding/binary"
	"math"
	"os"
	"sort"
	"time"
)

// Answers of the scored queries (-answers): what scripts/score_answers.py scores against the
// ground truth of the data version each query read. Update batches (U14) change the data, so
// latency.csv's recall, against the one ground truth of -gt, holds only until the first batch
// flips; a query's epoch tells which version it read. The answers are kept in memory, as
// latency.csv's records are, and written with them at the end: the run writes nothing while the
// nodes' disks are being measured.
//
// The file, little endian: the magic "RTANSWR1", u32 k, u32 stride (a query's ground-truth row is
// query / stride), u64 count, then count records of u32 query, u32 n (the answer's length, at
// most k), u64 epoch, u64 the scheduled send time in microseconds since the start, k u32 IDs and
// k f32 distances, of which the first n are the answer's (the rest 0xFFFFFFFF and NaN).
const answersMagic = "RTANSWR1"

type answer struct {
	query int
	epoch uint64
	sched time.Duration
	ids   []uint32
	dists []float32
}

func writeAnswers(path string, k, stride int, ans []answer) error {
	sort.Slice(ans, func(i, j int) bool { return ans[i].sched < ans[j].sched })
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriterSize(f, 1<<20)
	hdr := make([]byte, 24)
	copy(hdr, answersMagic)
	binary.LittleEndian.PutUint32(hdr[8:], uint32(k))
	binary.LittleEndian.PutUint32(hdr[12:], uint32(stride))
	binary.LittleEndian.PutUint64(hdr[16:], uint64(len(ans)))
	w.Write(hdr)
	rec := make([]byte, 24+8*k)
	nan := math.Float32bits(float32(math.NaN()))
	for _, a := range ans {
		n := min(len(a.ids), k)
		binary.LittleEndian.PutUint32(rec[0:], uint32(a.query))
		binary.LittleEndian.PutUint32(rec[4:], uint32(n))
		binary.LittleEndian.PutUint64(rec[8:], a.epoch)
		binary.LittleEndian.PutUint64(rec[16:], uint64(a.sched.Microseconds()))
		for i := 0; i < k; i++ {
			id, d := uint32(math.MaxUint32), nan
			if i < n {
				id, d = a.ids[i], math.Float32bits(a.dists[i])
			}
			binary.LittleEndian.PutUint32(rec[24+4*i:], id)
			binary.LittleEndian.PutUint32(rec[24+4*k+4*i:], d)
		}
		if _, err := w.Write(rec); err != nil {
			f.Close()
			return err
		}
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
