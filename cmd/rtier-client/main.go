// rtier-client: talk to the controller API, an entry node, or a data node.
//
//	rtier-client [-api ADDR] status
//	rtier-client [-api ADDR] wait-ready [-timeout 60s]
//	rtier-client [-api ADDR] rescale [-dry-run] [-save-plan OUT] N           # N data nodes, placed by the policy
//	rtier-client [-api ADDR] rescale [-dry-run] [-save-plan OUT] -plan FILE  # the placement FILE names
//	rtier-client [-api ADDR] design             # open design questions (placeholders)
//
// A plan file is a protocol.Plan: {"from_epoch": E, "owners": {"NODE": [partitions], ...}},
// NODE a node ID or name; "from_epoch" is optional. -dry-run changes nothing and prints the
// plan the request would carry out; -save-plan writes it to OUT, one line per node, to edit
// and pass back with -plan (the controller refuses it once the epoch has moved on, unless
// "from_epoch" is taken out).
//
//	rtier-client query -entry ADDR -queries Q.u8bin [-gt GT.ibin] [-k 10 -nprobe 64 -n 200]
//	rtier-client search-local -node ADDR -queries Q.u8bin [...]   # single-node baseline
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
	"time"

	"rtier/internal/ctrl"
	"rtier/internal/design"
	"rtier/internal/nodeclient"
	"rtier/internal/protocol"
	"rtier/internal/query"
	"rtier/internal/vecio"
)

func main() {
	api := flag.String("api", "127.0.0.1:7101", "controller API address")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: rtier-client [-api ADDR] status | wait-ready | rescale [-dry-run] [-save-plan OUT] N|-plan FILE | design | query ... | search-local ...")
	}
	flag.Parse()
	if flag.NArg() < 1 {
		flag.Usage()
		os.Exit(2)
	}
	ctx := context.Background()
	args := flag.Args()[1:]
	switch cmd := flag.Arg(0); cmd {
	case "status":
		var st protocol.ClusterStatus
		call(ctx, *api, protocol.StatusA, nil, &st, time.Minute)
		printJSON(st)
	case "wait-ready":
		fs := flag.NewFlagSet(cmd, flag.ExitOnError)
		timeout := fs.Duration("timeout", 5*time.Minute, "")
		fs.Parse(args)
		var st protocol.ClusterStatus
		call(ctx, *api, protocol.WaitReadyA, protocol.WaitReadyReq{TimeoutSeconds: timeout.Seconds()}, &st, *timeout+10*time.Second)
		fmt.Printf("ready: epoch %d, %d nodes\n", st.Table.Epoch, len(st.Nodes))
	case "rescale":
		fs := flag.NewFlagSet(cmd, flag.ExitOnError)
		planPath := fs.String("plan", "", "reconfigure to the placement this plan file names")
		dryRun := fs.Bool("dry-run", false, "work out the plan and what would move, change nothing")
		savePlan := fs.String("save-plan", "", "write the reply's plan to this file")
		fs.Parse(args)
		req := protocol.RescaleReq{DryRun: *dryRun}
		switch {
		case *planPath != "" && fs.NArg() == 0:
			p, err := readPlan(*planPath)
			if err != nil {
				log.Fatal(err)
			}
			req.Plan = p
		case *planPath == "" && fs.NArg() == 1:
			n, err := strconv.Atoi(fs.Arg(0))
			if err != nil {
				log.Fatal(err)
			}
			req.DataNodes = n
		default:
			log.Fatal("usage: rescale [-dry-run] [-save-plan OUT] N | rescale [-dry-run] [-save-plan OUT] -plan FILE")
		}
		var rep protocol.RescaleReply
		call(ctx, *api, protocol.RescaleA, req, &rep, 24*time.Hour)
		if *savePlan != "" && rep.Plan != nil {
			if err := os.WriteFile(*savePlan, formatPlan(rep.Plan), 0o644); err != nil {
				log.Fatal(err)
			}
		}
		printJSON(rep)
	case "design":
		for _, id := range design.IDs() {
			q := design.Open[id]
			fmt.Printf("%s  %s\n     options: %s\n     where:   %s\n", q.ID, q.Title, q.Options, q.Where)
		}
	case "query", "search-local":
		runQueries(ctx, cmd, args)
	default:
		flag.Usage()
		os.Exit(2)
	}
}

func call(ctx context.Context, addr, method string, req, resp any, timeout time.Duration) {
	c, err := ctrl.Dial(ctx, addr)
	if err != nil {
		log.Fatalf("controller API %s: %v", addr, err)
	}
	defer c.Close()
	c.Start()
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	if err := c.Call(ctx, method, req, resp); err != nil {
		log.Fatal(err)
	}
}

func printJSON(v any) {
	b, _ := json.MarshalIndent(v, "", "  ")
	fmt.Println(string(b))
}

// readPlan reads a plan file strictly: a misspelt key would otherwise leave a field empty.
func readPlan(path string) (*protocol.Plan, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	var p protocol.Plan
	if err := dec.Decode(&p); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &p, nil
}

// formatPlan lays a plan out for editing: one line per node, nodes in numeric order
// (MarshalIndent would put every partition on a line of its own and sort "10" before "2").
func formatPlan(p *protocol.Plan) []byte {
	keys := make([]string, 0, len(p.Owners))
	for k := range p.Owners {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		a, aerr := strconv.Atoi(keys[i])
		b, berr := strconv.Atoi(keys[j])
		if aerr == nil && berr == nil {
			return a < b
		}
		return keys[i] < keys[j]
	})
	var sb strings.Builder
	fmt.Fprintf(&sb, "{\n  \"from_epoch\": %d,\n  \"owners\": {\n", p.FromEpoch)
	for i, k := range keys {
		key, _ := json.Marshal(k)
		ps, _ := json.Marshal(p.Owners[k])
		sep := ","
		if i == len(keys)-1 {
			sep = ""
		}
		fmt.Fprintf(&sb, "    %s: %s%s\n", key, strings.ReplaceAll(string(ps), ",", ", "), sep)
	}
	sb.WriteString("  }\n}\n")
	return []byte(sb.String())
}

func runQueries(ctx context.Context, cmd string, args []string) {
	fs := flag.NewFlagSet(cmd, flag.ExitOnError)
	entry := fs.String("entry", "", "entry node query address (query)")
	node := fs.String("node", "", "data node address (search-local)")
	qpath := fs.String("queries", "", "query vectors (.u8bin/.i8bin/.fbin)")
	gtpath := fs.String("gt", "", "ground truth (.ibin), optional")
	maxQ := fs.Int("nq", 0, "use only the first nq queries")
	k := fs.Int("k", 10, "results")
	nprobe := fs.Int("nprobe", 64, "posting lists probed")
	n := fs.Int("n", 200, "candidates re-ranked (fixed n)")
	ef := fs.Int("ef", 0, "graph search width (0 = 2*nprobe)")
	heuristic := fs.Bool("heuristic", false, "search-local: heuristic early stop instead of fixed n")
	fs.Parse(args)
	qs, err := vecio.ReadBin(*qpath, *maxQ)
	if err != nil {
		log.Fatal(err)
	}
	var gt *vecio.GroundTruth
	if *gtpath != "" {
		if gt, err = vecio.ReadGroundTruth(*gtpath); err != nil {
			log.Fatal(err)
		}
	}
	var qc *query.Client
	var nc *nodeclient.Client
	if cmd == "query" {
		qc = query.NewClient(*entry, 1)
	} else {
		nc = nodeclient.New(*node, 1)
	}
	var recall float64
	start := time.Now()
	for i := 0; i < qs.N; i++ {
		var cands []nodeclient.Candidate
		if qc != nil {
			res, err := qc.Query(ctx, &query.Request{Vec: qs.Row(i), Params: query.Params{K: *k, NProbe: *nprobe, N: *n, EF: *ef}})
			if err != nil {
				log.Fatalf("query %d: %v", i, err)
			}
			cands = res.Candidates
		} else {
			cands, err = nc.SearchLocal(ctx, 0, qs.Row(i), nodeclient.SearchParams{K: *k, NProbe: *nprobe, Rerank: *n, EF: *ef, Heuristic: *heuristic})
			if err != nil {
				log.Fatalf("query %d: %v", i, err)
			}
		}
		if gt != nil && i < gt.NQ {
			ids := make([]uint32, len(cands))
			ds := make([]float32, len(cands))
			for j, c := range cands {
				ids[j], ds[j] = c.ID, c.Dist
			}
			recall += gt.Recall(i, ids, ds, *k)
		}
	}
	el := time.Since(start)
	fmt.Printf("%d queries in %.2fs (%.1f us/query sequential)", qs.N, el.Seconds(), float64(el.Microseconds())/float64(qs.N))
	if gt != nil {
		fmt.Printf(", recall@%d %.4f", *k, recall/float64(min(qs.N, gt.NQ)))
	}
	fmt.Println()
}
