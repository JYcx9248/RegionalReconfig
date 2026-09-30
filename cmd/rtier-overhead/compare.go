package main

import (
	"context"
	"fmt"
	"math"
	"os"
	"text/tabwriter"

	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
	"rtier/internal/placement"
	"rtier/internal/query"
	"rtier/internal/vecio"
)

// Costs are per-unit costs of the model used to turn counts into work and latency (us). The
// defaults are what a fit of the fan-out runs gave on the machine this tool was written on
// (2-vCPU VM, CPU PQ filter, virtio disk, loopback RPC); pass your own for other hardware.
type Costs struct {
	RPC, FilterCall, Scored, RerankCall, Page, Merge, Merged float64
}

// Where the re-ranking happens, and on which candidates.
//
//	two-phase     owners return their PQ top-n; the aggregator merges the global top-n and has
//	              each candidate re-ranked by an owner that reported it (two round trips)
//	two-phase+1   the same, except that a query whose lists all sit on one owner is filtered and
//	              re-ranked there in a single call (same answer, one round trip)
//	local         every owner re-ranks its own PQ top-n and returns its exact top-k (one round
//	              trip; a superset of the global top-n is re-ranked)
//	local-quota   every owner re-ranks the first n x (its share of the scored candidates) of its
//	              PQ top-n (one round trip; about n re-ranked in all, not the global top-n)
var strategies = []string{"two-phase", "two-phase+1", "local", "local-quota"}

type stratResult struct {
	reranked, pages, calls float64 // per query; calls = RPCs to owners
	work, latency          float64 // modeled, us per query
	recall                 float64
	differs                int // answers that differ from the single-node answer
}

func compare(ctx context.Context, c *nodeclient.Client, m *partitioning.Manifest, qs *vecio.Vectors,
	lists [][]uint32, gt *vecio.GroundTruth, sizes []int, n, k int, split string, navUs float64, cost Costs) {

	ref := make([][]nodeclient.Candidate, qs.N) // single-node answers (two-phase at N=1)
	type row struct {
		n   int
		fan float64
		res map[string]*stratResult
	}
	var rows []row
	for _, size := range sizes {
		nodes := make([]placement.NodeID, size)
		for i := range nodes {
			nodes[i] = placement.NodeID(i + 1)
		}
		tbl, err := placement.EvenPolicy{}.Initial(m.NumPartitions, nodes)
		check(err)
		r := row{n: size, res: map[string]*stratResult{}}
		for _, s := range strategies {
			r.res[s] = &stratResult{}
		}
		for q := 0; q < qs.N; q++ {
			vec := qs.Row(q)
			_, groups := group(lists[q], m, tbl)
			r.fan += float64(len(groups))
			filtered := make([][]nodeclient.Candidate, len(groups))
			scored := make([]float64, len(groups))
			var total float64
			for i, g := range groups {
				cands, st, err := c.Filter(ctx, 0, vec, g, n)
				check(err)
				filtered[i], scored[i] = cands, float64(st.Unique)
				total += scored[i]
			}
			inputs := map[string][][]uint32{}
			tp := splitByOrigin(filtered, query.MergeTopK(filtered, n), split)
			inputs["two-phase"], inputs["two-phase+1"] = tp, tp
			local := make([][]uint32, len(groups))
			quota := make([][]uint32, len(groups))
			for i, f := range filtered {
				ids := make([]uint32, len(f))
				for j, cd := range f {
					ids[j] = cd.ID
				}
				local[i] = ids
				share := int(math.Ceil(float64(n) * scored[i] / total))
				quota[i] = ids[:min(share, len(ids))]
			}
			inputs["local"], inputs["local-quota"] = local, quota
			var returned float64
			for _, f := range filtered {
				returned += float64(len(f))
			}
			for _, s := range strategies {
				res := r.res[s]
				var reranked [][]nodeclient.Candidate
				ownerFilter := make([]float64, len(groups))
				ownerRerank := make([]float64, len(groups))
				fused := s == "local" || s == "local-quota" || (s == "two-phase+1" && len(groups) == 1)
				if fused {
					res.calls += float64(len(groups))
				} else {
					res.calls += float64(len(groups)) // FILTER; RERANK calls are added below
				}
				for i, ids := range inputs[s] {
					ownerFilter[i] = cost.FilterCall + cost.Scored*scored[i]
					if len(ids) == 0 {
						continue
					}
					cands, pages, err := c.Rerank(ctx, 0, vec, ids, groups[i], k)
					check(err)
					reranked = append(reranked, cands)
					ownerRerank[i] = cost.RerankCall + cost.Page*float64(pages)
					res.reranked += float64(len(ids))
					res.pages += float64(pages)
					if !fused {
						res.calls++
					}
				}
				ans := query.MergeTopK(reranked, k)
				mergeK := cost.Merge + cost.Merged*float64(len(reranked)*k)
				var work, lat float64
				if fused { // FILTER and RERANK in one call per owner
					for i := range groups {
						t := cost.RPC + ownerFilter[i] + ownerRerank[i]
						work += t
						lat = math.Max(lat, t)
					}
					work += mergeK
					lat += mergeK
				} else {
					mergeN := cost.Merge + cost.Merged*returned
					var lf, lr float64
					for i := range groups {
						work += cost.RPC + ownerFilter[i]
						lf = math.Max(lf, cost.RPC+ownerFilter[i])
						if ownerRerank[i] > 0 {
							work += cost.RPC + ownerRerank[i]
							lr = math.Max(lr, cost.RPC+ownerRerank[i])
						}
					}
					work += mergeN + mergeK
					lat = lf + mergeN + lr + mergeK
				}
				res.work += work + navUs
				res.latency += lat + navUs
				if ref[q] == nil && s == "two-phase" {
					ref[q] = ans
				}
				if !equal(ans, ref[q]) {
					res.differs++
				}
				if gt != nil && q < gt.NQ {
					ids := make([]uint32, len(ans))
					ds := make([]float32, len(ans))
					for j, cd := range ans {
						ids[j], ds[j] = cd.ID, cd.Dist
					}
					res.recall += gt.Recall(q, ids, ds, k)
				}
			}
		}
		nq := float64(qs.N)
		r.fan /= nq
		for _, res := range r.res {
			res.reranked /= nq
			res.pages /= nq
			res.calls /= nq
			res.work /= nq
			res.latency /= nq
			res.recall /= nq
		}
		rows = append(rows, r)
	}

	w1 := rows[0].res["two-phase"].work
	fmt.Printf("re-rank placement, %d queries, n %d, k %d; work and latency modeled from the counts "+
		"(rpc %.0f, filter %.0f+%.3f/scored, rerank %.0f+%.1f/page, merge %.0f+%.3f/candidate us)\n\n",
		qs.N, n, k, cost.RPC, cost.FilterCall, cost.Scored, cost.RerankCall, cost.Page, cost.Merge, cost.Merged)
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(w, "N\towners\tstrategy\tre-ranked\tpages\tcalls\twork us\tS(N)\tlatency us\trecall@k\t≠ single node\t")
	for _, r := range rows {
		for _, s := range strategies {
			res := r.res[s]
			rec := "-"
			if gt != nil {
				rec = fmt.Sprintf("%.4f", res.recall)
			}
			fmt.Fprintf(w, "%d\t%.2f\t%s\t%.0f\t%.1f\t%.2f\t%.0f\t%.2f\t%.0f\t%s\t%.1f%%\t\n",
				r.n, r.fan, s, res.reranked, res.pages, res.calls, res.work, float64(r.n)*w1/res.work,
				res.latency, rec, 100*float64(res.differs)/float64(qs.N))
		}
	}
	w.Flush()
}
