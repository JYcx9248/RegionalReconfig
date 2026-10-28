#!/usr/bin/env python3
"""Turn one experiment's results into a timeline: what the load saw while the cluster changed.

    python3 scripts/plot_run.py results/run1              # -> timeline.svg, timeline.csv, summary
    python3 scripts/plot_run.py results/run1 --window 0.2
    python3 scripts/plot_run.py results/run1 --latency tail   # p50 and p99 instead of the mean

Reads what run_local.py (or run_cluster.py) leaves in the results directory:

  latency.csv            one row per query: sched_ms, latency_us, status, epoch, recall
                         (empty when the query has no ground truth), attempts (older runs have
                         no attempts column)
  reconfigurations.json  one entry per reconfiguration (a rescale, a plan or a full rebuild):
                         when it was triggered, how long it took, what it moved (older runs hold
                         the bare replies)
  cgroups.jsonl          where there is one (NodeResources, or a cluster's agents): each node's
                         samples, with clock.json, the load's start on their clock

and writes, over the same time axis:

  timeline.svg   two panels -- queries per second (offered and answered) and latency (the mean
                 of the answered queries, or p50 and p99) -- with the epochs marked, so a flip
                 is visible as a line and a reconfiguration as a band. The dashed "offered"
                 line hides under "answered" exactly while the cluster keeps up: where it
                 appears, the load is past what the nodes serve. With cgroups.jsonl a third:
                 each data node's memory -- its anonymous memory (what it holds: PQ codes,
                 posting lists, graph, caches; not the page cache), or its RSS where it has no
                 cgroup of its own -- one line a node, up to eight (past that, the most and the
                 least of them); what a full rebuild's placement and generations cost in memory
  timeline.csv   the same windows as numbers, for your own plotting
  memory.csv     the memory samples the third panel draws

No dependencies: the SVG is written here, in light and dark.

A query counts as offered in the window of its scheduled time, as answered (or failed) in the
window of its answer, and its latency in the window of its scheduled time -- what a query sent
then waited, as compare_runs.py counts them. By the scheduled time alone, every query of a
backlog would count as answered in the window it was sent, and the answered line would follow
the offered one through the dip it is meant to show. A flip is marked where the first query of
its epoch was answered: no query of an epoch is answered before its flip.

What to read off it: time-to-online (trigger to the epoch line), the latency cost of the
transition (how far the latency rises above its level before the trigger, and for how long),
and whether throughput actually follows the load after the new node joins.

The mean covers answered queries only. A query that failed (e.g. refused while a
stop-and-copy paused admission, or abandoned when the runner stopped waiting for the backlog)
is missing from it and from "answered", and counts as failed in timeline.csv; run the load generator with -retry-for ("RetryForSeconds" in the experiment)
so such a query waits for its answer and its wait counts in the mean.
"""

import argparse
import csv
import json
import math
import os
import sys

# The categorical slots of the reference palette in their order, light and dark (validated in
# that order on the adjacent pairs, as lines are: slots 1 and 2 for the load's panels, one a data
# node for the memory panel).
SERIES = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"), ("#eda100", "#c98500"),
          ("#e87ba4", "#d55181"), ("#008300", "#008300"), ("#6250d6", "#9085e9"), ("#e34948", "#e66767")]


def read_latency(path):
    """Rows of latency.csv as dicts, sorted by scheduled time."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "t": float(r["sched_ms"]) / 1000.0,
                "us": float(r["latency_us"]),
                "status": r["status"],
                "epoch": int(r["epoch"]),
                "recall": float(r["recall"]) if r.get("recall") else None,
                "attempts": int(r.get("attempts") or 1),
                "users": int(r.get("users") or 0),
            })
    rows.sort(key=lambda r: r["t"])
    return rows


def pctl(sorted_values, q):
    if not sorted_values:
        return None
    i = min(len(sorted_values) - 1, int(q * len(sorted_values)))
    return sorted_values[i]


def answered_at(r):
    """When the query was answered (or failed, or was abandoned), in seconds into the run."""
    return r["t"] + r["us"] / 1e6


def windows(rows, width):
    """Per window of `width` seconds, up to the last query's scheduled time: the offered rate
    (by scheduled time), the answered and failed rates (by the time of the answer; answers after
    the last window are left out, not piled into it), the mean and percentiles of the latency of
    the answered queries sent in the window."""
    if not rows:
        return []
    out = []
    end = rows[-1]["t"]
    n = max(1, int(math.ceil((end + 1e-9) / width)))
    sent = [[] for _ in range(n)]
    answered, failed = [0] * n, [0] * n
    for r in rows:
        sent[min(n - 1, int(r["t"] / width))].append(r)
        k = int(answered_at(r) / width)
        if k < n:
            if r["status"] == "ok":
                answered[k] += 1
            else:
                failed[k] += 1
    for i, b in enumerate(sent):
        lat = sorted(r["us"] for r in b if r["status"] == "ok")
        out.append({
            "t": i * width,
            "offered": len(b) / width,
            "answered": answered[i] / width,
            "failed": failed[i] / width,
            "mean_us": sum(lat) / len(lat) if lat else None,
            "p50_us": pctl(lat, 0.5),
            "p99_us": pctl(lat, 0.99),
        })
    return out


def epoch_marks(rows):
    """(time, epoch) of the first query answered in each epoch after the first: the flips."""
    first = {}
    for r in rows:
        e = r["epoch"]
        if e and (e not in first or answered_at(r) < first[e]):
            first[e] = answered_at(r)
    return sorted((t, e) for e, t in first.items() if e != min(first))


def summarize(rows):
    lat = sorted(r["us"] for r in rows if r["status"] == "ok")
    status, epochs = {}, {}
    recalls = [r["recall"] for r in rows if r["status"] == "ok" and r["recall"] is not None]
    for r in rows:
        status[r["status"]] = status.get(r["status"], 0) + 1
        epochs[r["epoch"]] = epochs.get(r["epoch"], 0) + 1
    span = (rows[-1]["t"] - rows[0]["t"]) if len(rows) > 1 else 0.0
    return {
        "queries": len(rows),
        "status": status,
        "epochs": epochs,
        "seconds": round(span, 3),
        "answered_per_s": round(len(lat) / span, 1) if span else 0.0,
        "retried": sum(1 for r in rows if r["attempts"] > 1),
        "mean_us": round(sum(lat) / len(lat), 1) if lat else None,
        "p50_us": pctl(lat, 0.5), "p99_us": pctl(lat, 0.99),
        "p999_us": pctl(lat, 0.999), "max_us": pctl(lat, 1.0),
        "recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "recall_queries": len(recalls),
    }


def node_memory(results):
    """Each data node's memory over the run, from cgroups.jsonl: {name: [(t, bytes, kind), ...]},
    t on latency.csv's axis (clock.json's load_start). kind: "anon" from its cgroup, or "rss"
    from /proc for a node without one. The agents and the host have samples of their own."""
    try:
        zero = json.load(open(os.path.join(results, "clock.json")))["load_start"]
        f = open(os.path.join(results, "cgroups.jsonl"))
    except (OSError, ValueError, KeyError):
        return {}
    out = {}
    with f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue  # a run cut short leaves its last line half written
            name, mem = r.get("node", ""), r.get("mem") or {}
            if not (name[:1] == "n" and name[1:].isdigit()) or "t" not in r:
                continue
            kind = "anon" if "anon" in mem else "rss"
            if kind in mem:
                out.setdefault(name, []).append((r["t"] - zero, mem[kind], kind))
    return dict(sorted(out.items(), key=lambda kv: int(kv[0][1:])))


def memory_series(mem, span):
    """The memory panel within [0, span]: (lines, title, unit), one line a node, n<i> in slot i,
    or with a node past the palette's eight slots the most and the least of them, second by
    second -- a ninth hue would be one nobody can tell apart. In GiB, or MiB below one."""
    inside = {n: [(t, b) for t, b, _ in pts if 0 <= t <= span] for n, pts in mem.items()}
    inside = {n: pts for n, pts in inside.items() if pts}
    if not inside:
        return None
    unit, scale = ("GiB", 2.0 ** 30) if max(b for pts in inside.values() for _, b in pts) >= 2 ** 30 \
        else ("MiB", 2.0 ** 20)
    inside = {n: [(t, b / scale) for t, b in pts] for n, pts in inside.items()}
    if all(int(n[1:]) <= len(SERIES) for n in inside):
        return [(n, pts, "s%d" % int(n[1:])) for n, pts in inside.items()], "memory per data node", unit
    per_s = {}
    for pts in inside.values():
        for t, v in pts:
            per_s.setdefault(int(t), []).append(v)
    secs = sorted(per_s)
    return ([("most", [(t + 0.5, max(per_s[t])) for t in secs], "s1"),
             ("least", [(t + 0.5, min(per_s[t])) for t in secs], "s2 dashed")],
            "memory of the %d data nodes: the most and the least" % len(inside), unit)


def reconfigurations(path):
    """Rescales as (trigger_s, elapsed_s, label). Older runs have no timing: then ()."""
    try:
        with open(path) as f:
            entries = json.load(f)
    except (OSError, ValueError):
        return []
    out = []
    for e in entries:
        if not isinstance(e, dict) or "trigger_seconds" not in e:
            continue
        reply = e.get("reply", {})
        label = "rescale to %s" % e.get("data_nodes", "?")
        if e.get("plan"):  # a placement of its own, possibly at the same size
            label += " (plan)"
        if e.get("rebuild"):  # a full rebuild (U14): the next index generation
            label = "full rebuild"
        out.append((float(e["trigger_seconds"]), float(e.get("elapsed_seconds", 0)), label, reply))
    return out


# ---------------------------------------------------------------- SVG

W, PANEL_H, GAP = 900, 200, 52
LEFT, RIGHT, TOP, BOTTOM = 64, 118, 52, 56


def nice_ticks(hi, count=5):
    """Ticks from 0 to >= hi on 1/2/5 steps."""
    if hi <= 0:
        return [0, 1]
    raw = hi / count
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            stepped = m * mag
            break
    ticks, v = [], 0.0
    while v < hi + stepped * 0.5:
        ticks.append(v)
        v += stepped
    if ticks[-1] < hi:  # the top tick is the panel's top: nothing may lie above it
        ticks.append(v)
    return ticks


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def panel(svg, y0, title, series, span, marks, bands, unit, at_ends=False):
    """One panel: series = [(name, [(t, v), ...], css class), ...], values in `unit`. One series
    gets no legend: the title names it. at_ends: each name stands beside its line's last value,
    moved apart where they would overlap, instead of one under the other at the top."""
    values = [v for _, pts, _ in series for _, v in pts if v is not None]
    hi = max(values) if values else 1.0
    ticks = nice_ticks(hi)
    top, bot = ticks[-1] or 1.0, 0.0
    x = lambda t: LEFT + (W - LEFT - RIGHT) * (t / span if span else 0)
    y = lambda v: y0 + PANEL_H - PANEL_H * ((v - bot) / (top - bot))

    svg.append('<text class="title" x="%d" y="%d">%s</text>' % (LEFT, y0 - 12, esc(title)))
    named = series if len(series) > 1 else []  # identity never colour alone
    rows = [y0 + 14 + i * 16 for i in range(len(named))]
    if at_ends:
        ends = [next((y(v) + 4 for _, v in reversed(pts) if v is not None), y0 + PANEL_H) for _, pts, _ in named]
        order = sorted(range(len(named)), key=lambda i: ends[i])
        prev = y0 + 10 - 14
        for i in order:  # top to bottom, 14 px apart at least, then back up from the bottom edge
            rows[i] = prev = max(ends[i], prev + 14)
        nxt = y0 + PANEL_H + 4 + 14
        for i in reversed(order):
            rows[i] = nxt = min(rows[i], nxt - 14)
    for (name, _, cls), ly in zip(named, rows):
        lx = W - RIGHT + 8
        svg.append('<line class="%s line" x1="%d" y1="%d" x2="%d" y2="%d"/>'
                   % (cls, lx, ly - 4, lx + 16, ly - 4))
        svg.append('<text class="legend" x="%d" y="%d">%s</text>' % (lx + 22, ly, esc(name)))

    for v in ticks:  # recessive grid, labels in ink
        svg.append('<line class="grid" x1="%d" y1="%.1f" x2="%d" y2="%.1f"/>'
                   % (LEFT, y(v), W - RIGHT, y(v)))
        svg.append('<text class="tick" x="%d" y="%.1f" text-anchor="end">%s</text>'
                   % (LEFT - 8, y(v) + 4, esc(fmt(v))))
    svg.append('<text class="tick" x="6" y="%.1f">%s</text>' % (y0 + 12, esc(unit)))

    for t0, dur, _ in bands:
        svg.append('<rect class="band" x="%.1f" y="%d" width="%.1f" height="%d"/>'
                   % (x(t0), y0, max(1.5, x(t0 + dur) - x(t0)), PANEL_H))
    for t, _ in marks:
        svg.append('<line class="event" x1="%.1f" y1="%d" x2="%.1f" y2="%d"/>'
                   % (x(t), y0, x(t), y0 + PANEL_H))

    for name, pts, cls in series:
        run = []
        for t, v in pts:
            if v is None:
                if len(run) > 1:
                    svg.append('<polyline class="%s line" points="%s"/>' % (cls, " ".join(run)))
                run = []
                continue
            run.append("%.1f,%.1f" % (x(t), y(v)))
        if len(run) > 1:
            svg.append('<polyline class="%s line" points="%s"/>' % (cls, " ".join(run)))


def fmt(v):
    if v == 0:
        return "0"
    if v >= 10:
        return "%.0f" % v
    if v >= 1:
        return "%.1f" % v
    return "%.2f" % v


def svg_document(win, marks, bands, span, title, latency="mean", memory=None):
    top1 = TOP
    top2 = TOP + PANEL_H + GAP
    top3 = top2 + PANEL_H + GAP if memory else None
    last = top3 or top2
    h = last + PANEL_H + BOTTOM
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" '
           'font-family="system-ui, -apple-system, Segoe UI, sans-serif">' % (W, h, W, h)]
    svg.append("""<style>
  svg { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --grid:#e4e3de; %s }
  @media (prefers-color-scheme: dark) {
    svg { --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --grid:#33322f; %s }
  }
  .bg { fill: var(--surface); }
  .title { fill: var(--ink); font-size: 13px; font-weight: 600; }
  .legend, .tick { fill: var(--ink2); font-size: 11px; }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .event { stroke: var(--ink2); stroke-width: 1; stroke-dasharray: 3 3; opacity: .7; }
  .band { fill: var(--ink2); opacity: .10; }
  .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
%s
  .hit { fill: transparent; }
  .hit:hover { fill: var(--ink2); opacity: .08; }
  /* the first of two series is dashed: when they coincide exactly -- every query answered,
     which is the good case -- it must still be visible under the second. */
  .dashed { stroke-dasharray: 5 3; }
</style>""" % (" ".join("--s%d:%s;" % (i, light) for i, (light, _) in enumerate(SERIES, 1)),
               " ".join("--s%d:%s;" % (i, dark) for i, (_, dark) in enumerate(SERIES, 1)),
               "\n".join("  .s%d { stroke: var(--s%d); fill: none; }" % (i, i) for i in range(1, len(SERIES) + 1))))
    svg.append('<rect class="bg" x="0" y="0" width="%d" height="%d"/>' % (W, h))
    svg.append('<text class="title" x="%d" y="18">%s</text>' % (LEFT, esc(title)))

    ms = lambda v: v / 1000.0 if v else None
    thr = [("offered", [(w["t"], w["offered"]) for w in win], "s1 dashed"),
           ("answered", [(w["t"], w["answered"]) for w in win], "s2")]
    panel(svg, top1, "queries per second", thr, span, marks, bands, "q/s")
    if latency == "tail":
        lat = [("p50", [(w["t"], ms(w["p50_us"])) for w in win], "s1 dashed"),
               ("p99", [(w["t"], ms(w["p99_us"])) for w in win], "s2")]
        panel(svg, top2, "latency", lat, span, marks, bands, "ms")
    else:
        lat = [("mean", [(w["t"], ms(w["mean_us"])) for w in win], "s1")]
        panel(svg, top2, "mean latency of the answered queries", lat, span, marks, bands, "ms")

    at = lambda t: LEFT + (W - LEFT - RIGHT) * (t / span if span else 0)
    if memory:
        lines, mtitle, unit = memory
        panel(svg, top3, mtitle, lines, span, marks, bands, unit, at_ends=True)
        # Hover: every node's memory in each second, as the panel's native tooltip.
        per_s = {}
        for name, pts, _ in lines:
            for t, v in pts:
                per_s.setdefault(int(t), {})[name] = v
        for sec in sorted(per_s):
            if sec >= span:
                continue
            readout = "%d–%d s: " % (sec, sec + 1) + ", ".join(
                "%s %.2f %s" % (n, v, unit) for n, v in per_s[sec].items())
            svg.append('<rect class="hit" x="%.1f" y="%d" width="%.1f" height="%d"><title>%s</title></rect>'
                       % (at(sec), top3, max(1.0, at(min(sec + 1, span)) - at(sec)), PANEL_H, esc(readout)))
    for t in nice_ticks(span, 8):
        svg.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%s</text>'
                   % (at(t), last + PANEL_H + 18, esc(fmt(t))))
    svg.append('<text class="tick" x="%d" y="%d" text-anchor="middle">seconds into the run</text>'
               % ((LEFT + W - RIGHT) / 2, last + PANEL_H + 38))
    for t, label in marks:  # the flips, named once, above the top panel
        svg.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%s</text>'
                   % (at(t), top1 - 12, esc(label)))
    for t0, dur, label in bands:  # the reconfiguration window, named inside it
        svg.append('<text class="tick" x="%.1f" y="%d">%s</text>'
                   % (at(t0) + 6, top1 + PANEL_H - 8, esc(label)))
    svg.append("</svg>")
    return "\n".join(svg)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", help="results directory (run_local.py's output)")
    ap.add_argument("--window", type=float, default=0.5, help="window in seconds (default 0.5)")
    ap.add_argument("--out", default="timeline.svg", help="SVG name inside the results directory")
    ap.add_argument("--latency", choices=("mean", "tail"), default="mean",
                    help="latency panel: the mean (default) or p50 and p99")
    args = ap.parse_args()

    lat_path = os.path.join(args.results, "latency.csv")
    if not os.path.exists(lat_path):
        sys.exit("no latency.csv in %s (did the load generator run?)" % args.results)
    rows = read_latency(lat_path)
    if not rows:
        sys.exit("latency.csv is empty")
    win = windows(rows, args.window)
    marks = [(t, "epoch %d" % e) for t, e in epoch_marks(rows)]
    rcs = reconfigurations(os.path.join(args.results, "reconfigurations.json"))
    bands = [(t0, dur, label) for t0, dur, label, _ in rcs]
    span = rows[-1]["t"]
    marks = [(t, label) for t, label in marks if t <= span]  # a flip only the backlog saw, after the load

    mem = node_memory(args.results)
    svg = svg_document(win, marks, bands, span, os.path.basename(os.path.abspath(args.results)),
                       args.latency, memory_series(mem, span) if mem else None)
    out = os.path.join(args.results, args.out)
    with open(out, "w") as f:
        f.write(svg)
    with open(os.path.join(args.results, "timeline.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "offered_qps", "answered_qps", "failed_qps", "p50_us", "p99_us", "mean_us"])
        for x in win:
            w.writerow(["%.3f" % x["t"], "%.1f" % x["offered"], "%.1f" % x["answered"],
                        "%.1f" % x["failed"], x["p50_us"] or "", x["p99_us"] or "",
                        "%.1f" % x["mean_us"] if x["mean_us"] else ""])
    if mem:
        with open(os.path.join(args.results, "memory.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "node", "bytes", "kind"])
            for name, pts in mem.items():
                for t, b, kind in pts:
                    w.writerow(["%.3f" % t, name, b, kind])

    s = summarize(rows)
    print(json.dumps(s, indent=2, sort_keys=True))
    for t, e in epoch_marks(rows):
        print("flip: %s at %.3f s" % (e, t))
    for t0, dur, label, reply in rcs:
        if "generation" in reply:  # a full rebuild: what its placement found in place, and fetched
            print("%s: triggered at %.3f s, took %.3f s, %s of %s vectors local, %s segment bytes, warm-up %s "
                  "queries fetched %s vectors, phases %s"
                  % (label, t0, dur, reply.get("local"), reply.get("vectors"), reply.get("segment_bytes"),
                     reply.get("warmup_queries"), reply.get("warmup_fetched"), reply.get("phases")))
            continue
        print("%s: triggered at %.3f s, took %.3f s, %s bytes, %s PQ codes, phases %s"
              % (label, t0, dur, reply.get("bytes"), reply.get("pq_codes"), reply.get("phases")))
    # The transition's cost: the highest windowed mean and p99 while an epoch was changing.
    if bands and win:
        for key, name in (("mean_us", "mean"), ("p99_us", "p99")):
            during = [w[key] for w in win
                      for t0, dur, _ in bands if t0 <= w["t"] <= t0 + dur + 1 and w[key]]
            before = [w[key] for w in win if w[key] and w["t"] < bands[0][0]]
            if during and before:
                print("%s during the reconfigurations: up to %.0f us per %g s window (before: %.0f us)"
                      % (name, max(during), args.window, sum(before) / len(before) if key == "mean_us"
                         else pctl(sorted(before), 0.99)))
    print("wrote %s, timeline.csv%s" % (out, " and memory.csv" if mem else ""))


if __name__ == "__main__":
    main()
