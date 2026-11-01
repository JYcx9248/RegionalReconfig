#!/usr/bin/env python3
"""Where a run's reads went, per data node and window: what each node's read budget charged
(stats-<node>.jsonl, "reads"), how long its readers waited for the budget, and what the whole
data disk served (cgroups.jsonl "host" records: everyone's reads, other users' included). On a
cluster (run_cluster.py) each node has a disk of its own: its host's ("<node>-host" records).

It tells the per-node budget from the shared disk as the limit of a run: at a node's budget, its
reads stay flat at the budget and its wait climbs; on a saturated disk, no node reaches its
budget and the disk's total stops growing. Ends with each node's share of the reads (the busiest
one sets the cluster's capacity) and, with GPUs, the GPUs' peak utilization and memory.

    python3 scripts/node_reads.py results/wiki-gpu/calibrate-4 [--window 20] [--last 320]

Times are seconds from the load's start (clock.json) when the run has one, else from the end
(calibration runs). Needs runs with --read-iops for the per-node columns; the disk columns need
"NodeResources" on one machine, the agents' samples on a cluster (cgroups.jsonl).
"""

import argparse
import glob
import json
import os


def lines(path):
    try:
        return [json.loads(l) for l in open(path) if l.strip()]
    except OSError:
        return []


def per_s(xs, a, b, f):
    xs = [x for x in xs if a <= x["t"] < b]
    return (f(xs[-1]) - f(xs[0])) / (xs[-1]["t"] - xs[0]["t"]) if len(xs) > 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="a results directory")
    ap.add_argument("--window", type=float, default=20, help="seconds per row (default 20)")
    ap.add_argument("--last", type=float, default=0, help="only the last this many seconds")
    args = ap.parse_args()

    st = {os.path.basename(f)[len("stats-"):-len(".jsonl")]: lines(f)
          for f in sorted(glob.glob(os.path.join(args.run, "stats-*.jsonl")))}
    st = {n: [x for x in v if "reads" in x] for n, v in st.items()}
    st = {n: v for n, v in st.items() if v}
    if not st:
        raise SystemExit(f"no stats-<node>.jsonl with read-budget counters in {args.run}")
    names = sorted(st, key=lambda n: (len(n), n))
    cg = lines(os.path.join(args.run, "cgroups.jsonl"))
    # One machine: "host" records (run_local.py), of the one disk every node reads. A cluster:
    # each node's agent sends "<node>-host" records, of its own host's disk and CPUs.
    host = [h for h in cg if h.get("node") == "host"]
    hosts = {} if host else {n: [h for h in cg if h.get("node") == n + "-host"] for n in names}
    hosts = {n: v for n, v in hosts.items() if v}
    disk = [h for h in host if "disk" in h]
    disks = {n: [h for h in v if "disk" in h] for n, v in hosts.items()}
    start = min(v[0]["t"] for v in st.values())
    end = min(v[-1]["t"] for v in st.values())
    try:
        zero, ref = json.load(open(os.path.join(args.run, "clock.json")))["load_start"], "load start"
    except (OSError, ValueError, KeyError):
        zero, ref = end, "end"
    lo = max(start, end - args.last) if args.last else start

    print(f"{args.run}: per {args.window:g} s, t = seconds from the {ref}; reads/s charged by each node's "
          f"budget; wait = seconds its readers spent waiting for the budget per second (summed over "
          f"readers, so it can exceed 1); others = disk reads not ours (a few hundred either way is "
          f"sampling noise: the node and host samples are not taken at the same instants)")
    # How busy each node's CPUs were (its processes and anyone else's on them), as a share of
    # them: near 100% on the busiest node means the CPUs, not the reads, limit the run. One
    # machine's experiment gives their number (CPUsPerNode), a cluster's host records do.
    if hosts:
        ncpu = {n: v[-1].get("cpus", 0) for n, v in hosts.items()}
        cpu = {n: [h for h in hosts[n] if h.get("cpus_busy_usec")] for n in names if ncpu.get(n)}
    else:
        try:
            k = json.load(open(os.path.join(args.run, "experiment.json")))["NodeResources"]["CPUsPerNode"]
        except (OSError, ValueError, KeyError):
            k = 0
        ncpu = {n: k for n in names}
        cpu = {n: [h for h in host if h.get("cpus_busy_usec")] for n in names} if k else {}
    cpu = cpu if len(cpu) == len(names) else {}
    disk_cols = (" ".join(f"{n + ' disk/s':>11s}" for n in names) + " " +
                 " ".join(f"{n + ' others':>9s}" for n in names)) if hosts else f"{'disk reads/s':>13s} {'others':>8s}"
    print(f"{'t':>7s} " + " ".join(f"{n + ' reads/s':>11s}" for n in names) + " " +
          " ".join(f"{n + ' wait':>8s}" for n in names) + " " + disk_cols +
          ("".join(f" {n + ' cpu':>7s}" for n in names) if cpu else ""))
    a = lo
    while a < end:
        b = a + args.window
        r = {n: per_s(st[n], a, b, lambda x: x["reads"]["charged"]) for n in names}
        w = {n: per_s(st[n], a, b, lambda x: x["reads"]["wait_us"]) / 1e6 for n in names}
        if hosts:
            d = {n: per_s(disks.get(n, []), a, b, lambda x: x["disk"]["rios"]) for n in names}
            disk_vals = (" ".join(f"{d[n]:11.0f}" for n in names) + " " +
                         " ".join(f"{d[n] - r[n]:9.0f}" for n in names))
        else:
            d = per_s(disk, a, b, lambda x: x["disk"]["rios"])
            disk_vals = f"{d:13.0f} {d - sum(r.values()):8.0f}"
        c = {n: per_s(cpu[n], a, b, lambda x, n=n: x["cpus_busy_usec"].get(n, 0)) / 1e6 / ncpu[n] for n in names} if cpu else {}
        print(f"{a - zero:7.0f} " + " ".join(f"{r[n]:11.0f}" for n in names) + " " +
              " ".join(f"{w[n]:8.2f}" for n in names) + " " + disk_vals +
              ("".join(f" {c[n]:7.0%}" for n in names) if c else ""))
        a = b

    total = {n: st[n][-1]["reads"]["charged"] - st[n][0]["reads"]["charged"] for n in names}
    all_reads = sum(total.values())
    budget = st[names[0]][-1]["reads"]["budget"]
    print(f"\nreads per node over the run (budget {budget}/s each): " +
          ", ".join(f"{n} {total[n]:,} ({total[n] / all_reads:.0%})" for n in names if all_reads))
    gpus = {}
    # A cluster's hosts number their GPUs alike: each is named with its node there.
    for prefix, recs in ([(n + " ", hosts[n]) for n in hosts] if hosts else [("", host)]):
        for h in recs:
            for g, v in h.get("gpus", {}).items():
                m = gpus.setdefault(f"{prefix}GPU {g}", [0, 0])
                m[0], m[1] = max(m[0], v["util"]), max(m[1], v["mem_mib"])
    if gpus:
        print("GPUs, peak over the run: " + ", ".join(f"{g}: {u}% util, {m} MiB"
                                                      for g, (u, m) in sorted(gpus.items())))


if __name__ == "__main__":
    main()
