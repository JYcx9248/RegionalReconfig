#!/usr/bin/env python3
"""Each partition's load in a run (branch partition-load), and what a placement plan would do with it.

The data nodes count, per partition, what FILTER and RERANK did for its lists (fusion::PartitionLoad:
the lists probed and their postings, the candidates re-ranked and the pages read for them) and log
the counters in their stats lines (stats-<node>.jsonl). This adds them up over a window of a run,
prints each partition's share and its owner, and checks the pages against what each node read
(reads: charged - peer). With --plan it adds up each node of the plan, flags a node that would both
give and take partitions, and, with --partitions, counts the PQ codes each node would hold.

    python3 scripts/partition_load.py RUN [--from S] [--to S] [--json OUT]
        [--plan PLAN.json] [--partitions DIR] [--client bin/rtier-client]

RUN is a results directory. The window is the whole run, or --from/--to seconds after the load's
start (clock.json; a calibration has none). Shares are of the pages read, the load on the SSDs,
which bounds our nodes. A plan's numbers add up its nodes' partitions as measured under the
placement of the window, so they are an estimate: a candidate that several nodes report is
re-ranked at the lowest node ID (internal/query, splitByOrigin), and moving a partition moves those
reads between partitions. A run with the plan measures the result.

Plan keys are node names (n1, n2, ...) or IDs, which the runners give in the same order (node i is
n<i>). --partitions DIR is the run's partition directory, for bin/rtier-client pq-share: the codes
each node of the plan would hold, and, for a node that gives and takes, both sets at once until the
reclaim; the run's pq-budget.json, if any, has the budget they must fit.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

KEYS = ("lists", "postings", "reranked", "pages")


def lines(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def node_key(k: str) -> str:
    return f"n{k}" if str(k).isdigit() else str(k)


def window(recs, t0, t1):
    """The first and last records in [t0, t1] (either bound None: open)."""
    inside = [r for r in recs if (t0 is None or r["t"] >= t0) and (t1 is None or r["t"] <= t1)]
    if len(inside) < 2:
        return None
    return inside[0], inside[-1]


def delta(a, b):
    pa, pb = a["partitions"], b["partitions"]
    d = {k: [y - x for x, y in zip(pa[k], pb[k])] for k in KEYS}
    d["rerank_pages"] = pb["rerank_pages"] - pa["rerank_pages"]
    d["unattributed"] = pb["unattributed"] - pa["unattributed"]
    ra, rb = a.get("reads", {}), b.get("reads", {})
    d["own_reads"] = (rb.get("charged", 0) - rb.get("peer", 0)) - (ra.get("charged", 0) - ra.get("peer", 0))
    return d


def pct(x, tot):
    return 100.0 * x / tot if tot else 0.0


def pq_share(client, parts, owners):
    """bin/rtier-client pq-share on a plan of these owners: the distinct vectors per node."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"owners": owners}, f)
    try:
        r = subprocess.run([client, "pq-share", "-partitions", parts, "-plan", f.name],
                           capture_output=True, text=True)
    finally:
        os.unlink(f.name)
    if r.returncode != 0:
        sys.exit(f"pq-share: {r.stderr.strip()}")
    return {node_key(k): v for k, v in json.loads(r.stdout)["vectors"].items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="a results directory")
    ap.add_argument("--from", dest="t_from", type=float, help="window start, seconds after the load's start")
    ap.add_argument("--to", dest="t_to", type=float, help="window end, seconds after the load's start")
    ap.add_argument("--plan", help="a placement plan ({\"owners\": {NODE: [partitions]}})")
    ap.add_argument("--partitions", help="the run's partition directory, for the PQ codes of the plan's nodes")
    ap.add_argument("--client", default="bin/rtier-client", help="rtier-client, for pq-share")
    ap.add_argument("--json", help="write each partition's load here")
    args = ap.parse_args()

    stats = {os.path.basename(f)[len("stats-"):-len(".jsonl")]: lines(f)
             for f in glob.glob(os.path.join(args.run, "stats-*.jsonl"))}
    stats = {n: [r for r in v if "partitions" in r] for n, v in stats.items()}
    stats = {n: v for n, v in stats.items() if v}
    if not stats:
        sys.exit(f"{args.run}: no stats-<node>.jsonl with per-partition counters (a run of branch partition-load)")
    nodes = sorted(stats, key=lambda n: (len(n), n))
    t0 = t1 = None
    if args.t_from is not None or args.t_to is not None:
        try:
            zero = json.load(open(os.path.join(args.run, "clock.json")))["load_start"]
        except OSError:
            sys.exit(f"{args.run}: --from/--to need clock.json (the load's start)")
        t0 = None if args.t_from is None else zero + args.t_from
        t1 = None if args.t_to is None else zero + args.t_to

    per_node = {}
    for n in nodes:
        w = window(stats[n], t0, t1)
        if w:
            per_node[n] = delta(*w)
    if not per_node:
        sys.exit("no node has two stats records in the window")
    nparts = max(len(d["pages"]) for d in per_node.values())

    # Each partition: totals over the nodes, and its owner in the window (the node that probed it most).
    parts = []
    for p in range(nparts):
        row = {k: sum(d[k][p] for d in per_node.values() if p < len(d[k])) for k in KEYS}
        counted = {n: d["lists"][p] for n, d in per_node.items() if p < len(d["lists"]) and d["lists"][p]}
        row["owner"] = max(counted, key=counted.get) if counted else None
        row["counted_on"] = sorted(counted, key=lambda n: (len(n), n))
        parts.append(row)
    tot = {k: sum(r[k] for r in parts) for k in KEYS}

    print(f"{args.run}: {len(per_node)} nodes, {nparts} partitions, window "
          f"{'whole run' if t0 is None and t1 is None else f'{args.t_from}-{args.t_to} s after the load start'}")
    print(f"{'part':>4} {'owner':>6} {'pages %':>8} {'reranked %':>10} {'postings %':>10} {'lists %':>8}")
    for p, r in enumerate(parts):
        moved = "" if len(r["counted_on"]) <= 1 else f"  (counted on {', '.join(r['counted_on'])})"
        print(f"{p:>4} {r['owner'] or '-':>6} {pct(r['pages'], tot['pages']):>8.2f} "
              f"{pct(r['reranked'], tot['reranked']):>10.2f} {pct(r['postings'], tot['postings']):>10.2f} "
              f"{pct(r['lists'], tot['lists']):>8.2f}{moved}")
    hot = sorted(range(nparts), key=lambda p: -parts[p]["pages"])
    print("hottest partitions by pages: " + ", ".join(f"{p} {pct(parts[p]['pages'], tot['pages']):.1f}%" for p in hot[:8]))

    print(f"\n{'node':>6} {'pages %':>8} {'own reads %':>11} {'pages':>12} {'rerank pages':>12} {'own reads':>12} "
          f"{'unattributed':>12}")
    own_tot = sum(d["own_reads"] for d in per_node.values())
    bad = []
    for n in nodes:
        d = per_node.get(n)
        if not d:
            continue
        pages = sum(d["pages"])
        print(f"{n:>6} {pct(pages, tot['pages']):>8.2f} {pct(d['own_reads'], own_tot):>11.2f} {pages:>12} "
              f"{d['rerank_pages']:>12} {d['own_reads']:>12} {d['unattributed']:>12}")
        if pages != d["rerank_pages"] or d["unattributed"]:
            bad.append(n)
    busiest = max(sum(d["pages"]) for d in per_node.values())
    owners_now = {}
    for p, r in enumerate(parts):
        if r["owner"]:
            owners_now.setdefault(r["owner"], []).append(p)
    nn = len(owners_now)
    print(f"busiest node: {pct(busiest, tot['pages']):.1f}% of the pages, {busiest * nn / tot['pages']:.2f}x an even "
          f"share of {nn} nodes" if tot["pages"] else "no page read in the window")
    if bad:
        print(f"WARNING: nodes {', '.join(bad)} counted pages that differ from what re-ranking read, or candidates "
              "no list named")
    print("(own reads: every device read but those served to peers -- re-ranking's, and the fetch writer's "
          "read-backs; at a fixed placement with every vector local they are re-ranking's pages)")

    if args.plan:
        plan = json.load(open(args.plan))
        owners = {node_key(k): sorted(v) for k, v in plan.get("owners", {}).items()}
        named = sorted(p for v in owners.values() for p in v)
        if named != list(range(nparts)):
            sys.exit(f"{args.plan}: the plan must give each of the {nparts} partitions to one node")
        print(f"\nplan {args.plan}: {len(owners)} nodes")
        print(f"{'node':>6} {'pages % now':>11} {'pages % plan':>12} {'gives':>24} {'takes':>24}")
        two_way = []
        for n in sorted(set(owners) | set(owners_now), key=lambda n: (len(n), n)):
            now, then = set(owners_now.get(n, [])), set(owners.get(n, []))
            gives, takes = sorted(now - then), sorted(then - now)
            share_now = pct(sum(parts[p]["pages"] for p in now), tot["pages"])
            share_plan = pct(sum(parts[p]["pages"] for p in then), tot["pages"])
            print(f"{n:>6} {share_now:>11.2f} {share_plan:>12.2f} {str(gives) if gives else '-':>24} "
                  f"{str(takes) if takes else '-':>24}")
            if gives and takes:
                two_way.append(n)
        peak = max(sum(parts[p]["pages"] for p in v) for v in owners.values())
        print(f"busiest node of the plan: {pct(peak, tot['pages']):.1f}% of the pages, "
              f"{peak * len(owners) / tot['pages']:.2f}x an even share of {len(owners)} nodes (an estimate)")
        if two_way:
            print(f"nodes {', '.join(two_way)} both give and take: until the reclaim each holds the PQ codes of "
                  "both sets")
        if args.partitions:
            final = pq_share(args.client, args.partitions, owners)
            during = pq_share(args.client, args.partitions,
                              {n: sorted(set(owners.get(n, [])) | set(owners_now.get(n, []))) for n in owners})
            budget = None
            try:
                budget = json.load(open(os.path.join(args.run, "pq-budget.json"))).get("budget")
            except OSError:
                pass
            print(f"\n{'node':>6} {'PQ codes':>12} {'while staging':>14}" + (f"   budget {budget}" if budget else ""))
            for n in sorted(owners, key=lambda n: (len(n), n)):
                over = " OVER" if budget and max(final.get(n, 0), during.get(n, 0)) > budget else ""
                print(f"{n:>6} {final.get(n, 0):>12} {during.get(n, 0):>14}{over}")

    if args.json:
        out = {"run": os.path.abspath(args.run), "from": args.t_from, "to": args.t_to,
               "nodes": {n: {"pages": sum(d["pages"]), "rerank_pages": d["rerank_pages"], "own_reads": d["own_reads"],
                             "unattributed": d["unattributed"]} for n, d in per_node.items()},
               "partitions": [{"partition": p, **{k: r[k] for k in KEYS}, "owner": r["owner"]}
                              for p, r in enumerate(parts)]}
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
