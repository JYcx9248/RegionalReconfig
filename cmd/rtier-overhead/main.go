// rtier-overhead: how much work the two-phase query strategy adds when a query's posting lists
// are spread over N owners instead of one.
//
// It emulates N owners on ONE data node that holds every partition. The lists a query
// navigates to are grouped by the owner an even placement of N nodes gives their partitions,
// and each group gets its own FILTER and RERANK call -- the calls N separate nodes would
// receive, minus the network. Per N it reports the fan-out, the candidates scored with PQ (a
// vector that sits in lists of two owners is scored by both), the pages read to re-rank (a page
// holding candidates of two owners is read by both), the server-side busy time of each
// operation (the node's INFO counters), the per-call overhead outside it (RPC), the
// aggregator's merge time, and how unevenly the work lands on the owners. Every answer is
// checked against the N = 1 answer.
//
//	rtier-overhead -node 127.0.0.1:7000 -partitions DIR -queries Q.u8bin [-nq 500] \
//	    [-nodes 1,2,4,8,16] [-json out.json]
//
// The node must have the graph and every partition loaded (rtier_node --graph index --load all).
// What this measures is work, not latency: calls run one at a time. The data node's backend
// (CPU or GPU) and the transport decide the per-call constants; the counts (fan-out, scored
// candidates, pages) do not depend on them.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
	"rtier/internal/placement"
	"rtier/internal/query"
	"rtier/internal/vecio"
)

// Owner is one emulated node's share of the work, summed over all queries.
type Owner struct {
	FilterCalls int     `json:"filter_calls"`
	Scored      int     `json:"scored"`
	RerankCalls int     `json:"rerank_calls"`
	Pages       int     `json:"pages"`
	WallUs      float64 `json:"wall_us"`
}

// Result is one cluster size; per-query averages unless noted.
type Result struct {
	N          int     `json:"n"`
	FanF       float64 `json:"fan_filter"` // owners called with FILTER
	FanR       float64 `json:"fan_rerank"` // owners called with RERANK (those holding part of the top-n)
	Gathered   float64 `json:"gathered"`   // posting entries gathered, summed over owners
	Scored     float64 `json:"scored"`     // distinct candidates scored per owner, summed
	Pages      float64 `json:"pages"`      // pages read to re-rank, summed over owners
	MergeIn    float64 `json:"merge_in"`   // candidates entering the top-n merge
	FilterUs   float64 `json:"filter_busy_us"`
	RerankUs   float64 `json:"rerank_busy_us"`
	FilterWall float64 `json:"filter_wall_us"`
	RerankWall float64 `json:"rerank_wall_us"`
	MergeUs    float64 `json:"merge_us"`
	Mismatches int     `json:"mismatches"`
	Owners     []Owner `json:"owners"`
}

// Report is the whole run.
type Report struct {
	Queries     int      `json:"queries"`
	NProbe      int      `json:"nprobe"`
	TopN        int      `json:"n"`
	K           int      `json:"k"`
	Partitions  int      `json:"partitions"`
	NavBusyUs   float64  `json:"navigate_busy_us"`
	NavWallUs   float64  `json:"navigate_wall_us"`
	MonoFixedUs float64  `json:"search_local_fixed_us"`
	MonoHeurUs  float64  `json:"search_local_heuristic_us"`
	MonoSame    int      `json:"search_local_fixed_same"`
	HeurOverlap float64  `json:"search_local_heuristic_topk_overlap"`
	Results     []Result `json:"results"`
}

func main() {
	addr := flag.String("node", "", "data node address")
	pdir := flag.String("partitions", "", "partition directory (manifest)")
	qpath := flag.String("queries", "", "query vectors")
	nq := flag.Int("nq", 500, "queries")
	nodesFlag := flag.String("nodes", "1,2,4,8,16", "cluster sizes to emulate")
	nprobe := flag.Int("nprobe", 64, "lists probed")
	n := flag.Int("n", 200, "candidates re-ranked (global top-n)")
	k := flag.Int("k", 10, "results")
	jsonOut := flag.String("json", "", "write the full report here")
	split := flag.String("split", "lowest", "who re-ranks a candidate several owners reported: lowest (as in internal/query) | hash")
	cmp := flag.Bool("compare", false, "compare where re-ranking happens (two-phase, local, local-quota) instead")
	ownersOnly := flag.Bool("owners", false, "only report how many owners each query's lists fall on")
	gtPath := flag.String("gt", "", "ground truth (.ibin) for recall in -compare")
	var cost Costs
	flag.Float64Var(&cost.RPC, "cost-rpc", 66.5, "model: us per RPC")
	flag.Float64Var(&cost.FilterCall, "cost-filter-call", 22.0, "model: us per FILTER call")
	flag.Float64Var(&cost.Scored, "cost-scored", 0.093, "model: us per candidate scored with PQ")
	flag.Float64Var(&cost.RerankCall, "cost-rerank-call", 70.6, "model: us per RERANK call")
	flag.Float64Var(&cost.Page, "cost-page", 6.2, "model: us per page read to re-rank")
	flag.Float64Var(&cost.Merge, "cost-merge", 29.7, "model: us per merge")
	flag.Float64Var(&cost.Merged, "cost-merged", 0.177, "model: us per candidate entering a merge")
	navFixed := flag.Float64("cost-navigate", 0, "model: us per navigation (0 = as measured in this run)")
	flag.Parse()
	if *addr == "" || *pdir == "" || *qpath == "" {
		flag.Usage()
		os.Exit(2)
	}
	ctx := context.Background()
	m, err := partitioning.LoadManifest(*pdir)
	check(err)
	qs, err := vecio.ReadBin(*qpath, *nq)
	check(err)
	c := nodeclient.New(*addr, 2)
	info, err := c.Info(ctx)
	check(err)
	if !info.GraphLoaded || len(info.Resident) != m.NumPartitions {
		log.Fatalf("node must hold the graph and all %d partitions (graph %v, %d resident)",
			m.NumPartitions, info.GraphLoaded, len(info.Resident))
	}
	rep := Report{Queries: qs.N, NProbe: *nprobe, TopN: *n, K: *k, Partitions: m.NumPartitions}

	// Navigation is the same for every N: once per query, at the entry.
	before := busy(ctx, c)
	lists := make([][]uint32, qs.N)
	var navWall time.Duration
	for q := 0; q < qs.N; q++ {
		t := time.Now()
		lists[q], err = c.Navigate(ctx, 0, qs.Row(q), *nprobe, 0)
		navWall += time.Since(t)
		check(err)
	}
	rep.NavBusyUs = float64(busy(ctx, c)["navigate"]-before["navigate"]) / float64(qs.N)
	rep.NavWallUs = us(navWall) / float64(qs.N)

	var sizes []int
	for _, s := range strings.Split(*nodesFlag, ",") {
		v, err := strconv.Atoi(strings.TrimSpace(s))
		check(err)
		sizes = append(sizes, v)
	}
	if *ownersOnly {
		owners(m, lists, sizes)
		return
	}
	if *cmp {
		var gt *vecio.GroundTruth
		if *gtPath != "" {
			gt, err = vecio.ReadGroundTruth(*gtPath)
			check(err)
		}
		nav := rep.NavWallUs
		if *navFixed > 0 {
			nav = *navFixed
		}
		compare(ctx, c, m, qs, lists, gt, sizes, *n, *k, *split, nav, cost)
		return
	}

	// Monolithic single-node baselines: the whole pipeline in one call, fixed n and heuristic.
	monoAns := map[bool][][]nodeclient.Candidate{}
	for _, h := range []bool{false, true} {
		before := busy(ctx, c)
		for q := 0; q < qs.N; q++ {
			a, err := c.SearchLocal(ctx, 0, qs.Row(q), nodeclient.SearchParams{K: *k, NProbe: *nprobe, Rerank: *n, Heuristic: h})
			check(err)
			monoAns[h] = append(monoAns[h], a)
		}
		v := float64(busy(ctx, c)["search_local"]-before["search_local"]) / float64(qs.N)
		if h {
			rep.MonoHeurUs = v
		} else {
			rep.MonoFixedUs = v
		}
	}

	var ref [][]nodeclient.Candidate
	for _, size := range sizes {
		nodes := make([]placement.NodeID, size)
		for i := range nodes {
			nodes[i] = placement.NodeID(i + 1)
		}
		tbl, err := placement.EvenPolicy{}.Initial(m.NumPartitions, nodes)
		check(err)
		r := Result{N: size, Owners: make([]Owner, size)}
		before := busy(ctx, c)
		answers := make([][]nodeclient.Candidate, qs.N)
		for q := 0; q < qs.N; q++ {
			vec := qs.Row(q)
			owners, groups := group(lists[q], m, tbl)
			filtered := make([][]nodeclient.Candidate, len(groups))
			for i, g := range groups {
				t := time.Now()
				cands, st, err := c.Filter(ctx, 0, vec, g, *n)
				check(err)
				d := us(time.Since(t))
				o := &r.Owners[owners[i]-1]
				o.FilterCalls++
				o.Scored += int(st.Unique)
				o.WallUs += d
				r.FilterWall += d
				r.Gathered += float64(st.Gathered)
				r.Scored += float64(st.Unique)
				r.MergeIn += float64(len(cands))
				filtered[i] = cands
			}
			t := time.Now()
			merged := query.MergeTopK(filtered, *n)
			byOwner := splitByOrigin(filtered, merged, *split)
			r.MergeUs += us(time.Since(t))
			reranked := make([][]nodeclient.Candidate, len(groups))
			for i := range groups {
				if len(byOwner[i]) == 0 {
					continue
				}
				t := time.Now()
				cands, pages, err := c.Rerank(ctx, 0, vec, byOwner[i], *k)
				check(err)
				d := us(time.Since(t))
				o := &r.Owners[owners[i]-1]
				o.RerankCalls++
				o.Pages += int(pages)
				o.WallUs += d
				r.RerankWall += d
				r.Pages += float64(pages)
				r.FanR++
				reranked[i] = cands
			}
			t = time.Now()
			answers[q] = query.MergeTopK(reranked, *k)
			r.MergeUs += us(time.Since(t))
			r.FanF += float64(len(groups))
			if ref != nil && !equal(answers[q], ref[q]) {
				r.Mismatches++
			}
		}
		if ref == nil {
			ref = answers
		}
		after := busy(ctx, c)
		r.FilterUs = float64(after["filter"]-before["filter"]) / float64(qs.N)
		r.RerankUs = float64(after["rerank"]-before["rerank"]) / float64(qs.N)
		for _, p := range []*float64{&r.FanF, &r.FanR, &r.Gathered, &r.Scored, &r.Pages,
			&r.FilterWall, &r.RerankWall, &r.MergeUs, &r.MergeIn} {
			*p /= float64(qs.N)
		}
		rep.Results = append(rep.Results, r)
	}
	for q := range ref {
		if equal(monoAns[false][q], ref[q]) {
			rep.MonoSame++
		}
		rep.HeurOverlap += shared(monoAns[true][q], ref[q], *k) / float64(len(ref))
	}

	fmt.Printf("%d queries, nprobe %d, n %d, k %d, %d partitions; navigate %.0f us busy, %.0f us with RPC (every N)\n",
		qs.N, *nprobe, *n, *k, m.NumPartitions, rep.NavBusyUs, rep.NavWallUs)
	fmt.Printf("single node, one SEARCH_LOCAL call: fixed n %.0f us (same answer as N=1 on %d/%d), "+
		"heuristic re-rank %.0f us (top-%d overlap with fixed n %.1f%%)\n\n",
		rep.MonoFixedUs, rep.MonoSame, len(ref), rep.MonoHeurUs, *k, 100*rep.HeurOverlap)
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(w, "N\towners F/R\tscored\tx1\tpages\tx1\tfilter us\trerank us\tRPC us\tmerge us\twork x1\tS(N)\thot owner\tmismatch\t")
	base := rep.Results[0]
	w1 := work(rep, base)
	for _, r := range rep.Results {
		wn := work(rep, r)
		rpc := (r.FilterWall + r.RerankWall) - (r.FilterUs + r.RerankUs)
		fmt.Fprintf(w, "%d\t%.2f/%.2f\t%.0f\t%.2f\t%.1f\t%.2f\t%.0f\t%.0f\t%.0f\t%.0f\t%.2f\t%.2f\t%.2f\t%d\t\n",
			r.N, r.FanF, r.FanR, r.Scored, r.Scored/base.Scored, r.Pages, r.Pages/base.Pages,
			r.FilterUs, r.RerankUs, rpc, r.MergeUs, wn/w1, float64(r.N)*w1/wn, hot(r), r.Mismatches)
	}
	w.Flush()
	fmt.Println("\nscored = distinct candidates scored with PQ, summed over owners; pages = pages read to re-rank;")
	fmt.Println("x1 = relative to N=1; work = navigate + filter + rerank + RPC + merge per query;")
	fmt.Println("S(N) = N * work(1)/work(N), perfectly balanced; hot owner = max/mean of per-owner work units.")
	if *jsonOut != "" {
		b, err := json.MarshalIndent(rep, "", " ")
		check(err)
		check(os.WriteFile(*jsonOut, b, 0o644))
	}
}

// work is the per-query CPU work of the whole cluster: navigation, filter and re-rank calls
// (server busy time plus the RPC overhead around them), and the aggregator's merges.
func work(rep Report, r Result) float64 {
	return rep.NavWallUs + r.FilterWall + r.RerankWall + r.MergeUs
}

// hot is max/mean over owners of a unit-free load: scored candidates, pages and calls, each
// normalized by its cluster-wide mean, weighted equally. Deterministic (no timing noise); a
// rough indicator of how uneven an even partition count is, not a capacity figure.
func hot(r Result) float64 {
	if len(r.Owners) < 2 {
		return 1
	}
	var ms, mp, mc float64
	for _, o := range r.Owners {
		ms += float64(o.Scored)
		mp += float64(o.Pages)
		mc += float64(o.FilterCalls + o.RerankCalls)
	}
	nn := float64(len(r.Owners))
	ms, mp, mc = ms/nn, mp/nn, mc/nn
	peak := 0.0
	for _, o := range r.Owners {
		l := (float64(o.Scored)/ms + float64(o.Pages)/mp + float64(o.FilterCalls+o.RerankCalls)/mc) / 3
		peak = max(peak, l)
	}
	return peak
}

// group splits lists by owner in tbl, ascending node IDs (query.GroupByOwner without an epoch).
func group(lists []uint32, m *partitioning.Manifest, tbl placement.Table) ([]placement.NodeID, [][]uint32) {
	idx := map[placement.NodeID]int{}
	var owners []placement.NodeID
	var groups [][]uint32
	for _, l := range lists {
		o := tbl.Owners[m.PartitionOf(l)]
		i, ok := idx[o]
		if !ok {
			i = len(owners)
			idx[o] = i
			owners = append(owners, o)
			groups = append(groups, nil)
		}
		groups[i] = append(groups[i], l)
	}
	order := make([]int, len(owners))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return owners[order[a]] < owners[order[b]] })
	so := make([]placement.NodeID, len(order))
	sg := make([][]uint32, len(order))
	for i, j := range order {
		so[i], sg[i] = owners[j], groups[j]
	}
	return so, sg
}

// splitByOrigin assigns each merged candidate to one of the groups that reported it. "lowest"
// mirrors internal/query (the lowest node ID wins); "hash" picks among the reporters by a hash
// of the vector ID, which spreads shared candidates evenly and still does not depend on timing.
func splitByOrigin(parts [][]nodeclient.Candidate, merged []nodeclient.Candidate, mode string) [][]uint32 {
	reporters := make(map[uint32][]int, len(merged))
	for i, p := range parts {
		for _, c := range p {
			reporters[c.ID] = append(reporters[c.ID], i)
		}
	}
	out := make([][]uint32, len(parts))
	for _, c := range merged {
		rs := reporters[c.ID]
		i := rs[0]
		if mode == "hash" && len(rs) > 1 {
			h := c.ID * 2654435761 // multiplicative hash
			i = rs[int(h>>16)%len(rs)]
		}
		out[i] = append(out[i], c.ID)
	}
	return out
}

func busy(ctx context.Context, c *nodeclient.Client) map[string]uint64 {
	in, err := c.Info(ctx)
	check(err)
	out := map[string]uint64{}
	for op, st := range in.Ops {
		out[op] = st.BusyUs
	}
	return out
}

func equal(a, b []nodeclient.Candidate) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// shared is the fraction of b's IDs (at most k) that a also returns.
func shared(a, b []nodeclient.Candidate, k int) float64 {
	in := map[uint32]bool{}
	for _, c := range a {
		in[c.ID] = true
	}
	hit, tot := 0, min(k, len(b))
	for _, c := range b[:tot] {
		if in[c.ID] {
			hit++
		}
	}
	if tot == 0 {
		return 1
	}
	return float64(hit) / float64(tot)
}

func us(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e3 }

func check(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

// owners reports, per cluster size, how the probed lists of a query spread over owners: the
// share of queries whose lists sit on one node, the mean number of owners, and the chance that
// the entry (uniform over the nodes) is itself one of them.
func owners(m *partitioning.Manifest, lists [][]uint32, sizes []int) {
	fmt.Printf("%d queries; owners per query under an even placement of %d partitions\n\n", len(lists), m.NumPartitions)
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(w, "N\tmean owners\tone owner\t<= 2 owners\tentry is an owner\t")
	for _, size := range sizes {
		nodes := make([]placement.NodeID, size)
		for i := range nodes {
			nodes[i] = placement.NodeID(i + 1)
		}
		tbl, err := placement.EvenPolicy{}.Initial(m.NumPartitions, nodes)
		check(err)
		var sum, one, two, entry float64
		for _, l := range lists {
			o, _ := group(l, m, tbl)
			f := float64(len(o))
			sum += f
			if len(o) == 1 {
				one++
			}
			if len(o) <= 2 {
				two++
			}
			entry += f / float64(size)
		}
		q := float64(len(lists))
		fmt.Fprintf(w, "%d\t%.2f\t%.1f%%\t%.1f%%\t%.1f%%\t\n", size, sum/q, 100*one/q, 100*two/q, 100*entry/q)
	}
	w.Flush()
}
