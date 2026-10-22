// rtier-loadgen: open-loop query load against the entry nodes.
//
// Queries are sent at their scheduled arrival times whether or not earlier ones have
// returned (open loop), and latency is measured from the scheduled time, so client-side
// queueing during a reconfiguration shows up in the tail instead of hiding it. Entries are
// read from the controller's epoch table every 500 ms, so new entries are used after a
// scale-out, and at once when an entry refuses a query under a newer epoch than the list's:
// that entry has left the entry set, and the list is stale.
//
// With -users the load is closed loop instead: a number of concurrent users that follows a
// schedule (-users-steps), each sending its next query when its last one returned; a query's
// time is when it was sent, and the offered load adapts to how fast the cluster answers.
//
// An "unavailable" answer (an entry that left, or paused admission under stop-and-copy) is
// retried three times by default, and the query fails after that; a retry after an entry that
// left goes to the entries of its epoch. With -retry-for the client
// waits for its answer instead: it keeps retrying until the answer comes or the time is up,
// so a pause shows up as latency of the answered queries. Averages over answered queries
// only compare protocols when no query fails, which is what -retry-for is for.
//
//	rtier-loadgen -api 127.0.0.1:7101 -queries q.u8bin -gt gt.ibin -rate 200 -duration 60s -out lat.csv
package main

import (
	"context"
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"rtier/internal/ctrl"
	"rtier/internal/metrics"
	"rtier/internal/protocol"
	"rtier/internal/query"
	"rtier/internal/vecio"
)

type record struct {
	sched    time.Duration // scheduled send time since start
	latency  time.Duration // completion - scheduled; for an abandoned query, its wait until the stop
	status   string
	epoch    uint64
	recall   float64
	hasGT    bool // recall is measured: the query has a ground-truth row
	attempts int  // queries sent, retries included
	users    int  // concurrent users when it was sent (closed loop; 0 in open loop)
}

// entries is the client's list of entry nodes, read from the controller's epoch table.
type entries struct {
	mu      sync.Mutex
	clients []*query.Client
	byAddr  map[string]*query.Client
	epoch   uint64 // the epoch the list was read from (0 with -entries)
	next    atomic.Uint64
	conns   int

	// Discovery, from the controller's API ("" with -entries). One read of it runs at a time,
	// and rounds counts them, so that the queries that find the list stale together share one.
	api    string
	read   sync.Mutex
	rounds atomic.Uint64
	stale  atomic.Int64 // reads a refusal under a newer epoch asked for
}

// set makes addrs, the entries of an epoch, the list, unless the list is from a later epoch
// already (a read that started before another one and ended after it).
func (e *entries) set(addrs []string, epoch uint64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if epoch < e.epoch {
		return
	}
	var cs []*query.Client
	for _, a := range addrs {
		c, ok := e.byAddr[a]
		if !ok {
			c = query.NewClient(a, e.conns)
			e.byAddr[a] = c
		}
		cs = append(cs, c)
	}
	e.clients, e.epoch = cs, epoch
}

func (e *entries) listEpoch() uint64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.epoch
}

// refresh reads the list from the controller: every 500 ms, whether or not it changed.
func (e *entries) refresh(ctx context.Context) error {
	e.read.Lock()
	defer e.read.Unlock()
	e.rounds.Add(1)
	return e.load(ctx)
}

// catchUp brings the list up to epoch, the epoch an entry refused a query under: the list is
// older, so that entry has left the entry set, and the entries to try are that epoch's. The
// queries that find the list stale together wait for one read, the first one's: none reads
// again once a read ran after it looked. It reports whether the list is now from that epoch or
// a later one.
func (e *entries) catchUp(ctx context.Context, epoch uint64) bool {
	if e.api == "" {
		return false
	}
	seen := e.rounds.Load()
	e.read.Lock()
	if e.rounds.Load() == seen && e.listEpoch() < epoch {
		e.rounds.Add(1)
		e.stale.Add(1)
		if err := e.load(ctx); err != nil {
			if ctx.Err() == nil {
				log.Printf("entry discovery: %v", err)
			}
		} else {
			log.Printf("entries: a query refused under epoch %d, the list read again (epoch %d)", epoch, e.listEpoch())
		}
	}
	e.read.Unlock()
	return e.listEpoch() >= epoch
}

func (e *entries) pick() *query.Client {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(e.clients) == 0 {
		return nil
	}
	return e.clients[e.next.Add(1)%uint64(len(e.clients))]
}

func main() {
	api := flag.String("api", "127.0.0.1:7101", "controller API (entry discovery)")
	fixed := flag.String("entries", "", "comma-separated entry query addresses (disables discovery)")
	qpath := flag.String("queries", "", "query vectors (.u8bin/.i8bin/.fbin)")
	gtpath := flag.String("gt", "", "ground truth (.ibin), optional")
	gtStride := flag.Int("gt-stride", 1, "the ground truth's row r is query r*stride: a query file larger than "+
		"a run, with ground truth for some of it all through the run")
	rate := flag.Float64("rate", 100, "queries per second (the rate at the start of the run)")
	steps := flag.String("rate-steps", "", "rate schedule after -rate, e.g. \"30:400,90:800\": "+
		"at 30 s the rate becomes 400 q/s, at 90 s 800 q/s (0 pauses the load)")
	duration := flag.Duration("duration", 30*time.Second, "run time")
	arrivals := flag.String("arrivals", "poisson", "poisson | semdn (U11, placeholder)")
	k := flag.Int("k", 10, "results")
	nprobe := flag.Int("nprobe", 64, "posting lists probed")
	n := flag.Int("n", 200, "candidates re-ranked (fixed n)")
	ef := flag.Int("ef", 0, "graph search width (0 = 2*nprobe)")
	conns := flag.Int("conns", 64, "max in-flight queries per entry")
	out := flag.String("out", "", "per-query CSV (sched_ms, latency_us, status, epoch, recall, attempts, users)")
	seed := flag.Int64("seed", 1, "arrival seed")
	retryFor := flag.Duration("retry-for", 0, "keep retrying an unavailable answer until this long "+
		"after the scheduled time (0: three attempts); the wait counts in the latency")
	users := flag.Int("users", 0, "closed loop with this many concurrent users at the start, each sending "+
		"its next query when the last one returned (0: open loop at -rate)")
	userSteps := flag.String("users-steps", "", "user schedule after -users, e.g. \"30:16,60:4\": "+
		"16 concurrent users from 30 s, 4 from 60 s")
	flag.Parse()
	if *gtStride < 1 {
		log.Fatalf("-gt-stride is %d: at least 1", *gtStride)
	}

	qs, err := vecio.ReadBin(*qpath, 0)
	if err != nil {
		log.Fatal(err)
	}
	var gt *vecio.GroundTruth
	if *gtpath != "" {
		if gt, err = vecio.ReadGroundTruth(*gtpath); err != nil {
			log.Fatal(err)
		}
	}
	rateSched, err := parseSteps(*steps, *rate)
	if err != nil {
		log.Fatal(err)
	}
	var userSched []step
	if *users > 0 {
		if userSched, err = parseSteps(*userSteps, float64(*users)); err != nil {
			log.Fatal(err)
		}
	}
	arr, err := newArrivals(*arrivals, rateSched, qs.N, *seed)
	if err != nil {
		log.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ents := &entries{byAddr: map[string]*query.Client{}, conns: *conns}
	if *fixed != "" {
		ents.set(strings.Split(*fixed, ","), 0)
	} else {
		ents.api = *api
		if err := ents.refresh(ctx); err != nil {
			log.Fatal(err)
		}
		go func() {
			for ctx.Err() == nil {
				time.Sleep(500 * time.Millisecond)
				if err := ents.refresh(ctx); err != nil && ctx.Err() == nil {
					log.Printf("entry discovery: %v", err)
				}
			}
		}()
	}

	params := query.Params{K: *k, NProbe: *nprobe, N: *n, EF: *ef}
	var (
		mu   sync.Mutex
		recs []record
		wg   sync.WaitGroup
	)
	lat := metrics.NewHistogram()
	var sent, ok, failed atomic.Int64
	start := time.Now()
	// The runner times its reconfigurations from this line: the file reads before it take a while
	// with a large query file, and its own clock would run ahead of latency.csv's by that much.
	fmt.Printf("START %.6f\n", metrics.Monotonic())
	// The queries sent and not answered yet: a stop (the runner gives up on a backlog that
	// outlasts its drain time) records them as abandoned, with how long they waited, rather than
	// losing every record of the run -- the slowest protocol's.
	pending := map[uint64]record{}
	var nextSeq uint64
	var finish sync.Once
	stopped := false
	done := func() {
		finish.Do(func() {
			mu.Lock()
			all := append([]record(nil), recs...)
			if stopped {
				now := time.Since(start)
				for _, r := range pending {
					r.latency = now - r.sched
					all = append(all, r)
				}
				log.Printf("stopped: %d queries unanswered, recorded as abandoned", len(pending))
			}
			mu.Unlock()
			summarize(all, gt != nil, ents.stale.Load())
			if *out != "" {
				if err := writeCSV(*out, all); err != nil {
					log.Fatal(err)
				}
			}
		})
	}
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigs
		mu.Lock()
		stopped = true
		mu.Unlock()
		done()
		os.Exit(0)
	}()
	go report(ctx, start, &sent, &ok, &failed, lat)
	// one sends query qi, scheduled (open loop) or sent (closed loop) at sched, and records it.
	one := func(sched time.Duration, qi, active int) {
		rec := record{sched: sched, status: "ok", users: active}
		mu.Lock()
		seq := nextSeq
		nextSeq++
		pending[seq] = record{sched: sched, status: "abandoned", users: active}
		mu.Unlock()
		var until time.Time
		if *retryFor > 0 {
			until = start.Add(sched + *retryFor)
		}
		res, attempts, err := send(ctx, ents, &query.Request{Vec: qs.Row(qi), Params: params}, until)
		rec.latency = time.Since(start.Add(sched))
		rec.attempts = attempts
		rec.epoch = res.Epoch
		if err != nil {
			rec.status = statusName(err)
			failed.Add(1)
		} else {
			ok.Add(1)
			lat.Record(rec.latency)
			if row := qi / *gtStride; gt != nil && qi%*gtStride == 0 && row < gt.NQ {
				ids := make([]uint32, len(res.Candidates))
				ds := make([]float32, len(res.Candidates))
				for j, c := range res.Candidates {
					ids[j], ds[j] = c.ID, c.Dist
				}
				rec.recall = gt.Recall(row, ids, ds, *k)
				rec.hasGT = true
			}
		}
		mu.Lock()
		delete(pending, seq)
		recs = append(recs, rec)
		mu.Unlock()
	}
	if userSched != nil {
		// Closed loop: user u sends its next query when its last one returned, while the
		// schedule has more than u users.
		for u := 0; u < maxUsers(userSched); u++ {
			wg.Add(1)
			go func(u int) {
				defer wg.Done()
				rng := rand.New(rand.NewSource(*seed + int64(u)))
				for {
					now := time.Since(start)
					if now >= *duration {
						return
					}
					active := int(valueAt(userSched, now))
					if u >= active {
						time.Sleep(5 * time.Millisecond)
						continue
					}
					sent.Add(1)
					one(now, rng.Intn(qs.N), active)
				}
			}(u)
		}
	} else {
		var sched time.Duration
		for {
			gap, qi := arr.Next()
			sched += gap
			if sched >= *duration {
				break
			}
			if d := time.Until(start.Add(sched)); d > 0 {
				time.Sleep(d)
			}
			sent.Add(1)
			wg.Add(1)
			go func(sched time.Duration, qi int) {
				defer wg.Done()
				one(sched, qi, 0)
			}(sched, qi)
		}
	}
	wg.Wait()
	cancel()
	done()
}

// maxBackoff caps the pause between retries of a waiting client. It bounds how late a
// client notices that admission resumed, and keeps the retry traffic of a long pause to about
// 20 per waiting query per second.
const maxBackoff = 50 * time.Millisecond

// send tries up to three entries: an entry that just left (scale-in) or is paused
// (stop-and-copy) answers "unavailable". With until set, it keeps retrying until then.
// It returns the number of attempts.
func send(ctx context.Context, ents *entries, req *query.Request, until time.Time) (query.Result, int, error) {
	var res query.Result
	var err error
	attempt := 0
	for attempt < 3 || time.Now().Before(until) {
		c := ents.pick()
		if c == nil {
			return res, attempt, errors.New("no entry nodes")
		}
		res, err = c.Query(ctx, req)
		attempt++
		if err == nil || !errors.Is(err, query.ErrUnavailable) {
			return res, attempt, err
		}
		// Refused under a newer epoch than the list's: the list is stale, and the next attempt
		// goes to that epoch's entries at once. Refused under the list's epoch or an earlier one
		// -- admission paused, or an entry that has not installed the list's epoch yet -- a
		// fresh list would be the same: wait, then retry.
		if res.Epoch > ents.listEpoch() && ents.catchUp(ctx, res.Epoch) {
			continue
		}
		time.Sleep(min(time.Duration(attempt)*5*time.Millisecond, maxBackoff))
	}
	return res, attempt, err
}

func statusName(err error) string {
	var se *query.StatusError
	if errors.As(err, &se) {
		switch se.Status {
		case query.StatusUnavailable:
			return "unavailable"
		case query.StatusUndecided:
			return "undecided"
		}
		return "error"
	}
	return "transport"
}

// load reads the entries of the controller's current epoch.
func (e *entries) load(ctx context.Context) error {
	c, err := ctrl.Dial(ctx, e.api)
	if err != nil {
		return err
	}
	defer c.Close()
	c.Start()
	cctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	var st protocol.ClusterStatus
	if err := c.Call(cctx, protocol.StatusA, nil, &st); err != nil {
		return err
	}
	if st.Table == nil {
		return errors.New("controller has no epoch yet")
	}
	var addrs []string
	for _, id := range st.Table.Entries {
		if n, ok := st.Table.Nodes[id]; ok {
			addrs = append(addrs, n.QueryAddr)
		}
	}
	e.set(addrs, st.Table.Epoch)
	return nil
}

func report(ctx context.Context, start time.Time, sent, ok, failed *atomic.Int64, lat *metrics.Histogram) {
	tk := time.NewTicker(time.Second)
	defer tk.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case now := <-tk.C:
			s := lat.Snapshot(true)
			log.Printf("t=%5.1fs sent=%d ok=%d failed=%d  last 1s: p50=%.0fus p99=%.0fus",
				now.Sub(start).Seconds(), sent.Load(), ok.Load(), failed.Load(), s.P50Us, s.P99Us)
		}
	}
}

func summarize(recs []record, withRecall bool, staleReads int64) {
	var lats []float64
	var recall, sum float64
	retried, measured := 0, 0
	counts := map[string]int{}
	for _, r := range recs {
		counts[r.status]++
		if r.attempts > 1 {
			retried++
		}
		if r.status == "ok" {
			lats = append(lats, float64(r.latency.Microseconds()))
			sum += lats[len(lats)-1]
			if r.hasGT {
				recall += r.recall
				measured++
			}
		}
	}
	sort.Float64s(lats)
	q := func(p float64) float64 {
		if len(lats) == 0 {
			return 0
		}
		return lats[min(len(lats)-1, int(p*float64(len(lats))))]
	}
	mean := 0.0
	if len(lats) > 0 {
		mean = sum / float64(len(lats))
	}
	fmt.Printf("queries %d: %v, %d retried, the entry list read %d times after a refusal under a newer epoch\n"+
		"latency us: mean %.0f  p50 %.0f  p99 %.0f  p99.9 %.0f  max %.0f\n",
		len(recs), counts, retried, staleReads, mean, q(0.5), q(0.99), q(0.999), q(1))
	if withRecall && measured > 0 {
		fmt.Printf("recall %.4f (%d queries with ground truth)\n", recall/float64(measured), measured)
	}
}

func writeCSV(path string, recs []record) error {
	sort.Slice(recs, func(i, j int) bool { return recs[i].sched < recs[j].sched })
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := csv.NewWriter(f)
	w.Write([]string{"sched_ms", "latency_us", "status", "epoch", "recall", "attempts", "users"})
	for _, r := range recs {
		recall := "" // no ground truth for it: not a recall of 0
		if r.hasGT {
			recall = strconv.FormatFloat(r.recall, 'f', 4, 64)
		}
		w.Write([]string{
			strconv.FormatFloat(float64(r.sched.Microseconds())/1000, 'f', 3, 64),
			strconv.FormatInt(r.latency.Microseconds(), 10), r.status,
			strconv.FormatUint(r.epoch, 10), recall,
			strconv.Itoa(r.attempts), strconv.Itoa(r.users),
		})
	}
	w.Flush()
	if err := w.Error(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
