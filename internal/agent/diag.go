package agent

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"rtier/internal/ctrl"
	"rtier/internal/metrics"
	"rtier/internal/protocol"
)

// A node's diagnostics go to the controller, as Koala's workers send their metrics to the
// coordinator's collector: the controller stamps them on arrival, so a run has one clock
// whatever machines it spans, and nothing has to be collected from the nodes afterwards.
//
//   - forwardStats: the data node's stats log (rtier_node --stats-file), line by line, into the
//     controller's stats-<node>.jsonl;
//   - sampleLoop: the data node's and the agent's processes and their machine, into the
//     controller's cgroups.jsonl, in the format of scripts/run_local.py's sampler (which samples
//     the nodes itself when they all run on its machine).

// forwardStats tails NodeStatsFile (-node-stats) and forwards every complete line to the controller.
func (a *Agent) forwardStats(ctx context.Context) {
	path := a.cfg.NodeStatsFile
	if path == "" {
		return
	}
	var off int64
	var partial []byte
	send := func() {
		f, err := os.Open(path)
		if err != nil {
			return // not written yet
		}
		defer f.Close()
		if _, err := f.Seek(off, io.SeekStart); err != nil {
			return
		}
		b, err := io.ReadAll(f)
		if err != nil || len(b) == 0 {
			return
		}
		off += int64(len(b))
		b = append(partial, b...)
		cut := bytes.LastIndexByte(b, '\n')
		if cut < 0 {
			partial = b
			return
		}
		partial = append([]byte(nil), b[cut+1:]...)
		var lines []json.RawMessage
		for _, l := range bytes.Split(b[:cut], []byte{'\n'}) {
			if l = bytes.TrimSpace(l); len(l) > 0 && json.Valid(l) {
				lines = append(lines, json.RawMessage(append([]byte(nil), l...)))
			}
		}
		if len(lines) == 0 {
			return
		}
		err = a.conn.Notify(protocol.NodeStatsN, protocol.NodeLines{Node: a.cfg.Name, Lines: lines})
		if err != nil && !errors.Is(err, ctrl.ErrClosed) {
			log.Printf("agent %s: stats: %v", a.cfg.Name, err)
		}
	}
	tk := time.NewTicker(250 * time.Millisecond)
	defer tk.Stop()
	for {
		select {
		case <-ctx.Done():
			send() // what the node wrote last
			return
		case <-tk.C:
			send()
		}
	}
}

// sampleLoop samples every SampleInterval and forwards the records.
func (a *Agent) sampleLoop(ctx context.Context) {
	iv := a.cfg.SampleInterval.Duration
	if iv <= 0 {
		return
	}
	var pid int
	for pid == 0 {
		ictx, cancel := context.WithTimeout(ctx, 5*time.Second)
		if in, err := a.localBulk.Info(ictx); err == nil {
			pid = in.PID
		}
		cancel()
		if pid == 0 {
			select {
			case <-ctx.Done():
				return
			case <-time.After(time.Second):
			}
		}
	}
	s := &procSampler{disk: dataDisk(a.cfg.WorkDir), cpus: allowedCPUs(pid), gpu: visibleGPU(pid)}
	tk := time.NewTicker(iv)
	defer tk.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tk.C:
		}
		var lines []json.RawMessage
		for _, r := range []map[string]any{
			s.process(a.cfg.Name, pid), s.process(a.cfg.Name+"-agent", os.Getpid()), s.host(a.cfg.Name),
		} {
			if r == nil {
				continue
			}
			if b, err := json.Marshal(r); err == nil {
				lines = append(lines, b)
			}
		}
		err := a.conn.Notify(protocol.NodeSamplesN, protocol.NodeLines{Node: a.cfg.Name, Lines: lines})
		if err != nil && !errors.Is(err, ctrl.ErrClosed) {
			log.Printf("agent %s: samples: %v", a.cfg.Name, err)
		}
	}
}

// procSampler reads what run_local.py's CgroupSampler reads, for one node's machine.
type procSampler struct {
	disk string // MAJ:MIN of the disk of the agent's work directory
	cpus []int  // the data node's CPUs
	gpu  string // CUDA_VISIBLE_DEVICES of the data node, if any
}

// clkTck: USER_HZ, the unit of /proc/<pid>/stat's CPU times (100 on every Linux build we use).
const clkTck = 100

// cgroupOf is the cgroup of a process that runs in a scope of its own (systemd-run's run-*.scope),
// "" otherwise: a process in its login session's scope shares its counters with the session.
func cgroupOf(pid int) string {
	b, err := os.ReadFile(fmt.Sprintf("/proc/%d/cgroup", pid))
	if err != nil {
		return ""
	}
	for _, l := range strings.Split(string(b), "\n") {
		if p, ok := strings.CutPrefix(l, "0::"); ok && strings.HasPrefix(filepath.Base(p), "run-") &&
			strings.HasSuffix(p, ".scope") {
			return "/sys/fs/cgroup" + p
		}
	}
	return ""
}

func keyed(path string) map[string]int64 {
	out := map[string]int64{}
	b, err := os.ReadFile(path)
	if err != nil {
		return out
	}
	for _, l := range strings.Split(string(b), "\n") {
		f := strings.Fields(l)
		if len(f) == 2 {
			if v, err := strconv.ParseInt(f[1], 10, 64); err == nil {
				out[f[0]] = v
			}
		}
	}
	return out
}

func procIO(pid int) map[string]int64 {
	io := map[string]int64{}
	b, _ := os.ReadFile(fmt.Sprintf("/proc/%d/io", pid))
	for _, l := range strings.Split(string(b), "\n") {
		if k, v, ok := strings.Cut(l, ":"); ok {
			if n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64); err == nil {
				io[k] = n
			}
		}
	}
	return map[string]int64{"rbytes": io["read_bytes"], "wbytes": io["write_bytes"]}
}

func (s *procSampler) process(name string, pid int) map[string]any {
	r := map[string]any{"t": metrics.Monotonic(), "node": name}
	cg := cgroupOf(pid)
	if cg == "" {
		b, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
		if err != nil {
			return nil
		}
		f := strings.Fields(string(b[bytes.LastIndexByte(b, ')')+1:]))
		if len(f) < 13 {
			return nil
		}
		ut, _ := strconv.ParseInt(f[11], 10, 64)
		st, _ := strconv.ParseInt(f[12], 10, 64)
		r["cpu_usec"] = (ut + st) * 1_000_000 / clkTck
		var rss int64
		if b, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid)); err == nil {
			for _, l := range strings.Split(string(b), "\n") {
				if v, ok := strings.CutPrefix(l, "VmRSS:"); ok {
					kb, _ := strconv.ParseInt(strings.Fields(v)[0], 10, 64)
					rss = kb * 1024
				}
			}
		}
		r["mem"] = map[string]int64{"rss": rss}
		r["io"], r["io_from"] = procIO(pid), "proc"
		return r
	}
	r["cpu_usec"] = keyed(cg + "/cpu.stat")["usage_usec"]
	io := map[string]int64{"rbytes": 0, "wbytes": 0, "rios": 0, "wios": 0}
	if b, err := os.ReadFile(cg + "/io.stat"); err == nil {
		for _, l := range strings.Split(string(b), "\n") {
			f := strings.Fields(l)
			if len(f) > 0 && f[0] == s.disk {
				for _, kv := range f[1:] {
					if k, v, ok := strings.Cut(kv, "="); ok {
						if _, want := io[k]; want {
							io[k], _ = strconv.ParseInt(v, 10, 64)
						}
					}
				}
			}
		}
		r["io"], r["io_from"] = io, "cgroup"
	} else {
		r["io"], r["io_from"] = procIO(pid), "proc"
	}
	mem := keyed(cg + "/memory.stat")
	r["mem"] = map[string]int64{"anon": mem["anon"], "file": mem["file"], "file_dirty": mem["file_dirty"],
		"file_writeback": mem["file_writeback"]}
	psi := map[string]int64{}
	for _, res := range []string{"cpu", "io", "memory"} {
		b, err := os.ReadFile(cg + "/" + res + ".pressure")
		if err != nil {
			continue
		}
		for _, l := range strings.Split(string(b), "\n") {
			f := strings.Fields(l)
			if len(f) == 0 {
				continue
			}
			for _, kv := range f[1:] {
				if v, ok := strings.CutPrefix(kv, "total="); ok {
					psi[res+"_"+f[0]+"_us"], _ = strconv.ParseInt(v, 10, 64)
				}
			}
		}
	}
	r["psi"] = psi
	return r
}

// host is the machine's record: its data disk's counters (everyone's I/O), how busy the data
// node's CPUs were (its processes and anyone else's), the load average and the node's GPU.
func (s *procSampler) host(name string) map[string]any {
	r := map[string]any{"t": metrics.Monotonic(), "node": name + "-host"}
	if b, err := os.ReadFile("/proc/loadavg"); err == nil {
		if f := strings.Fields(string(b)); len(f) > 0 {
			r["load1"], _ = strconv.ParseFloat(f[0], 64)
		}
	}
	if b, err := os.ReadFile("/proc/diskstats"); err == nil {
		for _, l := range strings.Split(string(b), "\n") {
			f := strings.Fields(l)
			if len(f) > 12 && f[0]+":"+f[1] == s.disk {
				n := func(i int) int64 { v, _ := strconv.ParseInt(f[i], 10, 64); return v }
				r["disk"] = map[string]int64{"rios": n(3), "rbytes": n(5) * 512, "wios": n(7), "wbytes": n(9) * 512,
					"busy_ms": n(12)}
			}
		}
	}
	if b, err := os.ReadFile("/proc/stat"); err == nil {
		want := map[int]bool{}
		for _, c := range s.cpus {
			want[c] = true
		}
		var busy int64
		for _, l := range strings.Split(string(b), "\n") {
			f := strings.Fields(l)
			if len(f) < 9 || !strings.HasPrefix(f[0], "cpu") || f[0] == "cpu" {
				continue
			}
			c, err := strconv.Atoi(f[0][3:])
			if err != nil || (len(want) > 0 && !want[c]) {
				continue
			}
			// user nice system idle iowait irq softirq steal: everything but idle and iowait
			for _, i := range []int{1, 2, 3, 6, 7, 8} {
				v, _ := strconv.ParseInt(f[i], 10, 64)
				busy += v
			}
		}
		r["cpus_busy_usec"] = map[string]int64{name: busy * 1_000_000 / clkTck}
	}
	if s.gpu != "" {
		if g := gpuSample(s.gpu); g != nil {
			r["gpus"] = map[string]any{s.gpu: g}
		}
	}
	return r
}

// dataDisk is MAJ:MIN of the disk holding path, as io.stat and diskstats name it (a partition's
// I/O is accounted to its disk).
func dataDisk(path string) string {
	var st syscall.Stat_t
	if syscall.Stat(path, &st) != nil {
		return ""
	}
	mm := fmt.Sprintf("%d:%d", (st.Dev>>8)&0xfff|(st.Dev>>32)&^0xfff, st.Dev&0xff|(st.Dev>>12)&^0xff)
	if _, err := os.Stat("/sys/dev/block/" + mm + "/partition"); err == nil {
		if b, err := os.ReadFile("/sys/dev/block/" + mm + "/../dev"); err == nil {
			mm = strings.TrimSpace(string(b))
		}
	}
	return mm
}

// allowedCPUs: the CPUs a process may run on (Cpus_allowed_list), nil if it is not known.
func allowedCPUs(pid int) []int {
	b, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid))
	if err != nil {
		return nil
	}
	var out []int
	sc := bufio.NewScanner(bytes.NewReader(b))
	for sc.Scan() {
		v, ok := strings.CutPrefix(sc.Text(), "Cpus_allowed_list:")
		if !ok {
			continue
		}
		for _, part := range strings.Split(strings.TrimSpace(v), ",") {
			lo, hi, found := strings.Cut(part, "-")
			a, err1 := strconv.Atoi(lo)
			b := a
			var err2 error
			if found {
				b, err2 = strconv.Atoi(hi)
			}
			if err1 != nil || err2 != nil {
				continue
			}
			for c := a; c <= b; c++ {
				out = append(out, c)
			}
		}
	}
	return out
}

// visibleGPU is the CUDA_VISIBLE_DEVICES a process was started with.
func visibleGPU(pid int) string {
	b, err := os.ReadFile(fmt.Sprintf("/proc/%d/environ", pid))
	if err != nil {
		return ""
	}
	for _, kv := range bytes.Split(b, []byte{0}) {
		if v, ok := bytes.CutPrefix(kv, []byte("CUDA_VISIBLE_DEVICES=")); ok {
			return string(v)
		}
	}
	return ""
}

// gpuSample is the utilization and memory in use of the GPU g names (an nvidia-smi index or
// UUID), nil without nvidia-smi.
func gpuSample(g string) map[string]int64 {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil
	}
	for _, l := range strings.Split(string(out), "\n") {
		f := strings.Split(l, ",")
		if len(f) != 4 {
			continue
		}
		for i := range f {
			f[i] = strings.TrimSpace(f[i])
		}
		if f[0] == g || strings.EqualFold(f[1], g) {
			u, err1 := strconv.ParseInt(f[2], 10, 64)
			m, err2 := strconv.ParseInt(f[3], 10, 64)
			if err1 == nil && err2 == nil {
				return map[string]int64{"util": u, "mem_mib": m}
			}
		}
	}
	return nil
}
