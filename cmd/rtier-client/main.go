// rtier-client: talk to the controller API, an entry node, or a data node.
//
//	rtier-client [-api ADDR] status
//	rtier-client [-api ADDR] wait-ready [-timeout 60s]
//	rtier-client [-api ADDR] rescale N          # N data nodes (scale out or in)
//	rtier-client [-api ADDR] design             # open design questions (placeholders)
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
	"strconv"
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
		fmt.Fprintln(os.Stderr, "usage: rtier-client [-api ADDR] status | wait-ready | rescale N | design | query ... | search-local ...")
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
		if len(args) != 1 {
			log.Fatal("usage: rescale N")
		}
		n, err := strconv.Atoi(args[0])
		if err != nil {
			log.Fatal(err)
		}
		var rep protocol.RescaleReply
		call(ctx, *api, protocol.RescaleA, protocol.RescaleReq{DataNodes: n}, &rep, 24*time.Hour)
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
