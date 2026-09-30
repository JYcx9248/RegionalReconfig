#!/usr/bin/env python3
"""Turn one experiment's results into a timeline: what the load saw while the cluster changed.

    python3 scripts/plot_run.py results/run1              # -> timeline.svg, timeline.csv, summary
    python3 scripts/plot_run.py results/run1 --window 0.2
    python3 scripts/plot_run.py results/run1 --latency tail   # p50 and p99 instead of the mean

Reads what run_local.py leaves in the results directory:

  latency.csv            one row per query: sched_ms, latency_us, status, epoch, recall,
                         attempts (older runs have no attempts column)
  reconfigurations.json  one entry per rescale: when it was triggered, how long it took, what
                         it moved (written by run_local.py; older runs hold the bare replies)

and writes, over the same time axis:

  timeline.svg   two panels -- queries per second (offered and answered) and latency (the mean
                 of the answered queries, or p50 and p99) -- with the epochs marked, so a flip is visible as a line and a reconfiguration
                 as a band. The dashed "offered" line hides under "answered" exactly while the
                 cluster keeps up: where it appears, the load is past what the nodes serve
  timeline.csv   the same windows as numbers, for your own plotting

No dependencies: the SVG is written here, in light and dark.

What to read off it: time-to-online (trigger to the epoch line), the latency cost of the
transition (how far the latency rises above its level before the trigger, and for how long),
and whether throughput actually follows the load after the new node joins.

The mean covers answered queries only. A query that failed (e.g. refused while a
stop-and-copy paused admission) is missing from it and shows up as "failed" in the top
panel instead; run the load generator with -retry-for ("RetryForSeconds" in the experiment)
so such a query waits for its answer and its wait counts in the mean.
"""

import argparse
import csv
import json
import math
import os
import sys

# Categorical slots 1 and 2 of the reference palette, light and dark (validated as a pair).
SERIES = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926")]


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
                "recall": float(r.get("recall") or 0.0),
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


def windows(rows, width):
    """Per window of `width` seconds (by scheduled time): offered and answered rate, the mean
    and percentiles of the answered queries' latency."""
    if not rows:
        return []
    out = []
    end = rows[-1]["t"]
    n = max(1, int(math.ceil((end + 1e-9) / width)))
    buckets = [[] for _ in range(n)]
    for r in rows:
        buckets[min(n - 1, int(r["t"] / width))].append(r)
    for i, b in enumerate(buckets):
        lat = sorted(r["us"] for r in b if r["status"] == "ok")
        bad = sum(1 for r in b if r["status"] != "ok")
        out.append({
            "t": i * width,
            "offered": len(b) / width,
            "answered": len(lat) / width,
            "failed": bad / width,
            "mean_us": sum(lat) / len(lat) if lat else None,
            "p50_us": pctl(lat, 0.5),
            "p99_us": pctl(lat, 0.99),
        })
    return out


def epoch_marks(rows):
    """(time, epoch) of the first query answered in each epoch after the first: the flips."""
    marks, seen, first = [], set(), None
    for r in rows:
        e = r["epoch"]
        if e == 0 or e in seen:
            continue
        seen.add(e)
        if first is None:
            first = e
            continue
        marks.append((r["t"], e))
    return marks


def summarize(rows):
    lat = sorted(r["us"] for r in rows if r["status"] == "ok")
    status, epochs = {}, {}
    recalls = [r["recall"] for r in rows if r["status"] == "ok" and r["recall"] > 0]
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
    }


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
    return ticks


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def panel(svg, y0, title, series, span, marks, bands, unit):
    """One panel: series = [(name, [(t, v), ...], css class), ...] (at most two), values in
    `unit`. One series gets no legend: the title names it."""
    values = [v for _, pts, _ in series for _, v in pts if v is not None]
    hi = max(values) if values else 1.0
    ticks = nice_ticks(hi)
    top, bot = ticks[-1] or 1.0, 0.0
    x = lambda t: LEFT + (W - LEFT - RIGHT) * (t / span if span else 0)
    y = lambda v: y0 + PANEL_H - PANEL_H * ((v - bot) / (top - bot))

    svg.append('<text class="title" x="%d" y="%d">%s</text>' % (LEFT, y0 - 12, esc(title)))
    for i, (name, _, cls) in enumerate(series if len(series) > 1 else []):  # identity never colour alone
        lx = W - RIGHT + 8
        ly = y0 + 14 + i * 16
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


def svg_document(win, marks, bands, span, title, latency="mean"):
    top1 = TOP
    top2 = TOP + PANEL_H + GAP
    h = top2 + PANEL_H + BOTTOM
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" '
           'font-family="system-ui, -apple-system, Segoe UI, sans-serif">' % (W, h, W, h)]
    svg.append("""<style>
  svg { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --grid:#e4e3de; --s1:%s; --s2:%s; }
  @media (prefers-color-scheme: dark) {
    svg { --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --grid:#33322f; --s1:%s; --s2:%s; }
  }
  .bg { fill: var(--surface); }
  .title { fill: var(--ink); font-size: 13px; font-weight: 600; }
  .legend, .tick { fill: var(--ink2); font-size: 11px; }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .event { stroke: var(--ink2); stroke-width: 1; stroke-dasharray: 3 3; opacity: .7; }
  .band { fill: var(--ink2); opacity: .10; }
  .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .s1 { stroke: var(--s1); fill: none; }
  .s2 { stroke: var(--s2); fill: none; }
  /* the first of two series is dashed: when they coincide exactly -- every query answered,
     which is the good case -- it must still be visible under the second. */
  .dashed { stroke-dasharray: 5 3; }
</style>""" % (SERIES[0][0], SERIES[1][0], SERIES[0][1], SERIES[1][1]))
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
    for t in nice_ticks(span, 8):
        svg.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%s</text>'
                   % (at(t), top2 + PANEL_H + 18, esc(fmt(t))))
    svg.append('<text class="tick" x="%d" y="%d" text-anchor="middle">seconds into the run</text>'
               % ((LEFT + W - RIGHT) / 2, top2 + PANEL_H + 38))
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

    svg = svg_document(win, marks, bands, span, os.path.basename(os.path.abspath(args.results)),
                       args.latency)
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

    s = summarize(rows)
    print(json.dumps(s, indent=2, sort_keys=True))
    for t, e in epoch_marks(rows):
        print("flip: %s at %.3f s" % (e, t))
    for t0, dur, label, reply in rcs:
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
    print("wrote %s and timeline.csv" % out)


if __name__ == "__main__":
    main()
