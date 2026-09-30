// End-to-end tests: real C++ data-node processes (rtier_node), the controller and the agents
// in-process, queries running while the cluster scales out and in.
//
// They need the engine binaries: `make test` builds them and sets RTIER_ENGINE_BIN (the
// directory holding fusion_build, rtier_segment and rtier_node). Without it they are skipped.
//
// Queries run through query.TwoPhase, the implemented half of U5: the tests check that its
// answer equals the single-node oracle exactly, in every epoch and during every
// reconfiguration, whatever the placement.
package e2e

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"rtier/internal/agent"
	"rtier/internal/config"
	"rtier/internal/controller"
	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
	"rtier/internal/placement"
	"rtier/internal/query"
	"rtier/internal/syncflag"
	"rtier/internal/vecio"
)

const (
	numVectors    = 8000
	dim           = 16
	numQueries    = 64
	numPartitions = 8
	maxRerank     = 4096
	testNProbe    = 16
	testK         = 10
)

type fixture struct {
	bin, dir, index, parts string
	// indexNoPages is the index without its page file (vectors_*.bin). Nodes that join after
	// the initial deployment run on it: they can only get raw vectors from their peers, which
	// is what shows that nothing but the bootstrap reads the index's pages (no global replica).
	indexNoPages string
	queries      *vecio.Vectors
	oracle       [][]nodeclient.Candidate // single-node exhaustive answers
	partIDs      [][]uint32               // per partition: vector IDs its lists name
}

// idsOf is the set of vector IDs named by the lists of the given partitions: the PQ codes a
// node holding exactly those partitions keeps (one per vector).
func (f *fixture) idsOf(parts []int) map[uint32]struct{} {
	s := map[uint32]struct{}{}
	for _, p := range parts {
		for _, id := range f.partIDs[p] {
			s[id] = struct{}{}
		}
	}
	return s
}

// partitionIDs reads the index's posting lists (postings.bin: u64 lists, u64 ids,
// u64 offsets[lists+1], u32 ids[]) and groups their vector IDs by partition.
func partitionIDs(postings string, assign []uint32, parts int) ([][]uint32, error) {
	b, err := os.ReadFile(postings)
	if err != nil {
		return nil, err
	}
	le := binary.LittleEndian
	if len(b) < 16 {
		return nil, fmt.Errorf("%s: truncated", postings)
	}
	nl, ni := int(le.Uint64(b)), int(le.Uint64(b[8:]))
	if nl != len(assign) || len(b) != 16+8*(nl+1)+4*ni {
		return nil, fmt.Errorf("%s: %d lists, %d IDs, %d bytes (assignment has %d lists)", postings, nl, ni, len(b), len(assign))
	}
	off := func(c int) int { return int(le.Uint64(b[16+8*c:])) }
	ids := b[16+8*(nl+1):]
	sets := make([]map[uint32]struct{}, parts)
	for i := range sets {
		sets[i] = map[uint32]struct{}{}
	}
	for c := 0; c < nl; c++ {
		for k := off(c); k < off(c+1); k++ {
			sets[assign[c]][le.Uint32(ids[4*k:])] = struct{}{}
		}
	}
	out := make([][]uint32, parts)
	for p, set := range sets {
		for id := range set {
			out[p] = append(out[p], id)
		}
		sort.Slice(out[p], func(i, j int) bool { return out[p][i] < out[p][j] })
	}
	return out, nil
}

var (
	fixOnce sync.Once
	fix     *fixture
	fixErr  error
)

func getFixture(t *testing.T) *fixture {
	t.Helper()
	bin := os.Getenv("RTIER_ENGINE_BIN")
	if bin == "" {
		t.Skip("RTIER_ENGINE_BIN not set (run `make test`)")
	}
	fixOnce.Do(func() { fix, fixErr = buildFixture(bin) })
	if fixErr != nil {
		t.Fatal(fixErr)
	}
	return fix
}

func mixture(rng *rand.Rand, centers [][]float32, n int) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		c := centers[rng.Intn(len(centers))]
		v := make([]float32, dim)
		for j := range v {
			v[j] = c[j] + float32(rng.NormFloat64())
		}
		out[i] = v
	}
	return out
}

func buildFixture(bin string) (*fixture, error) {
	dir, err := os.MkdirTemp("", "rtier-e2e-")
	if err != nil {
		return nil, err
	}
	f := &fixture{bin: bin, dir: dir, index: filepath.Join(dir, "index"), parts: filepath.Join(dir, "parts")}
	rng := rand.New(rand.NewSource(7))
	centers := make([][]float32, 40)
	for i := range centers {
		centers[i] = make([]float32, dim)
		for j := range centers[i] {
			centers[i][j] = float32(rng.NormFloat64() * 4)
		}
	}
	base := vecio.FromFloat32(mixture(rng, centers, numVectors))
	f.queries = vecio.FromFloat32(mixture(rng, centers, numQueries))
	if err := vecio.WriteBin(filepath.Join(dir, "base.fbin"), base); err != nil {
		return nil, err
	}
	if out, err := exec.Command(filepath.Join(bin, "fusion_build"), "--base", filepath.Join(dir, "base.fbin"),
		"--out", f.index, "--pq-m", "8", "--pq-train", "4000", "--threads", "2").CombinedOutput(); err != nil {
		return nil, fmt.Errorf("fusion_build: %v\n%s", err, out)
	}
	// Test-only list -> partition assignment (contiguous ranges); the real one is U2.
	heads, err := partitioning.ReadU32Array(filepath.Join(f.index, "heads.bin")) // one head per list
	if err != nil {
		return nil, err
	}
	numLists := len(heads)
	assign := make([]uint32, numLists)
	for c := range assign {
		assign[c] = uint32(c * numPartitions / numLists)
	}
	if err := partitioning.WriteU32Array(filepath.Join(dir, "assign.bin"), assign); err != nil {
		return nil, err
	}
	if out, err := exec.Command(filepath.Join(bin, "rtier_segment"), "--index", f.index,
		"--assign", filepath.Join(dir, "assign.bin"), "--out", f.parts).CombinedOutput(); err != nil {
		return nil, fmt.Errorf("rtier_segment: %v\n%s", err, out)
	}
	if f.partIDs, err = partitionIDs(filepath.Join(f.index, "postings.bin"), assign, numPartitions); err != nil {
		return nil, err
	}
	f.indexNoPages = filepath.Join(dir, "index-nopages")
	if err := os.MkdirAll(f.indexNoPages, 0o755); err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(f.index)
	if err != nil {
		return nil, err
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), "vectors_") {
			continue
		}
		if err := os.Symlink(filepath.Join(f.index, e.Name()), filepath.Join(f.indexNoPages, e.Name())); err != nil {
			return nil, err
		}
	}

	// Oracle: one node holding everything, exhaustive single-node search.
	addr, stop, err := startNode(f, "--load", "all", "--graph", "index")
	if err != nil {
		return nil, err
	}
	nc := nodeclient.New(addr, 1)
	for i := 0; i < numQueries; i++ {
		res, err := nc.SearchLocal(context.Background(), 0, f.queries.Row(i),
			nodeclient.SearchParams{K: testK, NProbe: testNProbe, Rerank: maxRerank})
		if err != nil {
			stop()
			return nil, err
		}
		f.oracle = append(f.oracle, res)
	}
	nc.Close()
	return f, stop()
}

var nodeSeq atomic.Int64

// startNode launches rtier_node on the fixture's index and waits for its READY line. Its stderr
// goes to a log file in the fixture directory; stop() interrupts the node, waits for it, and
// reports sanitizer findings (when the engine is built with -fsanitize) or a failed exit.
func startNode(f *fixture, extra ...string) (string, func() error, error) {
	return startNodeOn(f, f.index, extra...)
}

// startNodeOn is startNode on another index directory (f.indexNoPages).
func startNodeOn(f *fixture, index string, extra ...string) (string, func() error, error) {
	args := append([]string{"--index", index, "--partitions", f.parts, "--listen", "127.0.0.1:0",
		"--workers", "2", "--max-nprobe", "64", "--max-rerank", fmt.Sprint(maxRerank), "--no-direct"}, extra...)
	cmd := exec.Command(filepath.Join(f.bin, "rtier_node"), args...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", nil, err
	}
	logPath := filepath.Join(f.dir, fmt.Sprintf("node-%d.log", nodeSeq.Add(1)))
	logf, err := os.Create(logPath)
	if err != nil {
		return "", nil, err
	}
	cmd.Stderr = logf
	if err := cmd.Start(); err != nil {
		logf.Close()
		return "", nil, err
	}
	stop := func() error {
		cmd.Process.Signal(os.Interrupt)
		done := make(chan error, 1)
		go func() { done <- cmd.Wait() }()
		var werr error
		select {
		case werr = <-done:
		case <-time.After(10 * time.Second):
			cmd.Process.Kill()
			werr = fmt.Errorf("rtier_node did not stop within 10s")
		}
		logf.Close()
		out, _ := os.ReadFile(logPath)
		if strings.Contains(string(out), "Sanitizer") || strings.Contains(string(out), "runtime error:") {
			return fmt.Errorf("sanitizer report in %s", logPath)
		}
		if werr != nil {
			return fmt.Errorf("rtier_node exit: %v (log %s)", werr, logPath)
		}
		return nil
	}
	line, err := bufio.NewReader(stdout).ReadString('\n')
	if err != nil || !strings.HasPrefix(line, "READY tcp=") {
		stop()
		return "", nil, fmt.Errorf("rtier_node did not start: %q %v (log %s)", line, err, logPath)
	}
	return "127.0.0.1:" + strings.TrimSpace(strings.TrimPrefix(line, "READY tcp=")), stop, nil
}

// heldTracker records which partitions each node has owned so far: with a static dataset a
// node keeps the PQ codes of partitions that moved away, so they are what it holds.
type heldTracker map[placement.NodeID]map[int]bool

func (h heldTracker) observe(t placement.Table) {
	for p, n := range t.Owners {
		if h[n] == nil {
			h[n] = map[int]bool{}
		}
		h[n][p] = true
	}
}

func (h heldTracker) partitions(n placement.NodeID) []int {
	var ps []int
	for p := range h[n] {
		ps = append(ps, p)
	}
	sort.Ints(ps)
	return ps
}

type cluster struct {
	t        *testing.T
	f        *fixture
	initial  int    // nodes of the initial deployment: the first ones added
	protocol string // the controller's reconfiguration protocol
	ctx      context.Context
	cancel   context.CancelFunc
	ctl      *controller.Controller
	agents   []*agent.Agent
	stops    []func() error
	wg       sync.WaitGroup
	held     heldTracker   // partitions each node has owned (see observe)
	rescales []rescaleSpan // every Rescale call of rescale(), in order (timeline_test.go)
}

// observe records the current placement in cl.held.
func (cl *cluster) observe() {
	if t := cl.ctl.Status().Table; t != nil {
		cl.held.observe(t.Placement)
	}
}

func newCluster(t *testing.T, f *fixture, protocol string, initial int, opts ...func(*config.Controller)) *cluster {
	ctx, cancel := context.WithCancel(context.Background())
	cl := &cluster{t: t, f: f, initial: initial, protocol: protocol, ctx: ctx, cancel: cancel, held: heldTracker{}}
	t.Cleanup(cl.close)
	cfg := config.DefaultController()
	cfg.Listen, cfg.APIListen = "127.0.0.1:0", "127.0.0.1:0"
	cfg.IndexDir, cfg.PartitionsDir = f.index, f.parts
	cfg.InitialNodes = initial
	cfg.Protocol = protocol
	cfg.MetricsPath = filepath.Join(t.TempDir(), "metrics.jsonl")
	cfg.CallTimeout = config.Duration{Duration: 20 * time.Second}
	cfg.EntryFlipInterval = config.Duration{Duration: 50 * time.Millisecond}
	for _, o := range opts {
		o(&cfg)
	}
	ctl, err := controller.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	cl.ctl = ctl
	cl.wg.Add(1)
	go func() { defer cl.wg.Done(); ctl.Run(ctx) }()
	ctl.Listening.Wait()
	return cl
}

// addNode starts an rtier_node and its agent; waits until the agent registered.
func (cl *cluster) addNode(name string, strategy query.Strategy, nodeArgs ...string) *agent.Agent {
	return cl.addNodeWith(name, agent.Options{Strategy: strategy}, nodeArgs...)
}

// addNodeWith is addNode with the agent's test options. The nodes of the initial deployment
// bootstrap from the index; the others join later and run without the index's page file.
func (cl *cluster) addNodeWith(name string, opts agent.Options, nodeArgs ...string) *agent.Agent {
	index := cl.f.index
	if len(cl.agents) >= cl.initial {
		index = cl.f.indexNoPages
	}
	// RTIER_E2E_BACKEND=gpu runs the cluster's data nodes on the GPU filter backend; the
	// single-node oracle stays on the CPU, so every answer also checks GPU against CPU.
	if b := os.Getenv("RTIER_E2E_BACKEND"); b != "" {
		nodeArgs = append([]string{"--backend", b}, nodeArgs...)
	}
	addr, stop, err := startNodeOn(cl.f, index, nodeArgs...)
	if err != nil {
		cl.t.Fatal(err)
	}
	cl.stops = append(cl.stops, stop)
	ctlAddr, _ := cl.ctl.Addrs()
	cfg := config.DefaultAgent()
	cfg.Name, cfg.Controller, cfg.NodeAddr = name, ctlAddr, addr
	cfg.WorkDir = filepath.Join(cl.t.TempDir(), name)
	cfg.PartitionsDir = cl.f.parts
	cfg.TransferRate = 64 << 20 // paced background copies, 64 MB/s
	cfg.ChunkBytes = 64 << 10
	cfg.MetricsInterval = config.Duration{Duration: 200 * time.Millisecond}
	a, err := agent.New(cfg, opts)
	if err != nil {
		cl.t.Fatal(err)
	}
	cl.wg.Add(1)
	go func() {
		defer cl.wg.Done()
		if err := a.Run(cl.ctx); err != nil && cl.ctx.Err() == nil {
			cl.t.Errorf("agent %s: %v", name, err)
		}
	}()
	if err := cl.wait(a.Registered, 30*time.Second); err != nil {
		cl.t.Fatalf("agent %s did not register: %v", name, err)
	}
	cl.agents = append(cl.agents, a)
	return a
}

// wait blocks on a sync flag for at most d.
func (cl *cluster) wait(flag *syncflag.SyncFlag, d time.Duration) error {
	ctx, cancel := context.WithTimeout(cl.ctx, d)
	defer cancel()
	return flag.WaitContext(ctx)
}

func (cl *cluster) close() {
	cl.cancel()
	cl.wg.Wait()
	for _, s := range cl.stops {
		if err := s(); err != nil {
			cl.t.Error(err)
		}
	}
}

func (cl *cluster) entryClients() []*query.Client {
	st := cl.ctl.Status()
	var out []*query.Client
	for _, id := range st.Table.Entries {
		out = append(out, query.NewClient(st.Table.Nodes[id].QueryAddr, 4))
	}
	return out
}
