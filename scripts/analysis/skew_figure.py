#!/usr/bin/env python3
"""The skew-mitigation figure for the paper, as Koala's (NSDI '27) Figure 11 with more panels in one
row: (a) the cluster's throughput, the offered rate and the queries answered per second; then each
data node's load, a line per node in one colour, a panel per --node-metric; then the backlog. The
load increase, the rebalance's trigger and its flip are marked through all of them. No dependencies:
plain SVG at the page's width (7 in), light only; rsvg-convert -f pdf makes the PDF.

    python3 scripts/analysis/skew_figure.py RUN [RUN ...] --out FIG [--node-metric reads]
        [--node-metric candidates] [--window 2] [--backlog-window 1]

A task of Koala's processes a record alone, so its per-task lines add up to the job's throughput. A
query of ours goes to several data nodes (a FILTER to each owner of its lists, then each candidate's
RERANK at one owner that reported it), so the cluster's throughput gets a panel of its own, and a
node's load has two measures, a panel each (--node-metric, repeated for both, the panels in its
order; reads alone without it):

- reads: the node's page reads per second (stats reads.charged): its queries' re-ranking, the reads
  it serves to peers' fetches and the pages its fetch writer reads back. What its disk, a c240g5
  node's bottleneck, pays: an old owner keeps reading for the partitions it gave away until their
  vectors have moved.
- candidates: the candidates the node re-ranked per second (fetch.rerank_disk + rerank_mem). Each
  candidate is re-ranked at one node, so the lines add up to about (a)'s answered rate times the
  candidates a query re-ranks (its "N"): the work the placement gives each node, which a plan moves
  at its flip.

The runs are repetitions of one protocol and one plan (run_one.sh's rep*/PROTOCOL): each line is the
mean over them. The offered rate is the runs' schedule ("Rate", "RateSteps"), the load increase its
first rise, the rebalance the plan's trigger, the flip its mean over the runs. The time axis runs from
the warm-up's end to where the load stopped. FIG.csv holds what is plotted. The caption, in LaTeX,
carries the title.
"""

import argparse
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from compare_runs import PALETTE, flow, load, node_flows, node_stats, offered_schedule, rate_at  # noqa: E402
from plot_run import esc, flip_time, load_end  # noqa: E402

FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
W = 504                        # pt: the page's width
Y0, Y1 = 22, 80                # the plots' top and bottom
H = Y1 + 34                    # below them: tick labels, "Time (s)", the subcaptions
LEFT, RIGHT, GAP = 36, 6, 34   # room for the first y label, after the last plot, between plots
LINE = PALETTE[0][0]           # lazy's colour in every figure of ours
OFFERED = {"stroke": "#333333", "width": 0.8, "dash": "2.5 1.5"}
MARK = "#b3b3b3"
DASH = {"load": "4 1.5 1 1.5", "trigger": "3 2", "flip": "0.8 1.6"}
# --node-metric: node_flows' key, the panel's y label and subcaption.
METRICS = {"reads": ("reads", "Page reads (/s)", "Page reads per node"),
           "candidates": ("cands", "Candidates (/s)", "Re-ranks per node")}


def columns(n):
    """Each of n plots' left and right, across the page."""
    w = (W - LEFT - RIGHT - GAP * (n - 1)) / n
    return [(LEFT + i * (w + GAP), LEFT + i * (w + GAP) + w) for i in range(n)]


def ticks(hi):
    """Round ticks from 0 to the lowest top past hi that 2 to 4 steps of 1, 2, 2.5 or 5 times a power
    of ten reach (the fewer steps of two equal tops)."""
    if hi <= 0:
        return [0.0, 1.0]
    best = None
    for mag in (10 ** k for k in range(-3, 10)):
        for m in (1, 2, 2.5, 5):
            step = m * mag
            n = int(math.ceil(hi / step - 1e-9))
            if 2 <= n <= 4 and (best is None or (n * step, n) < (best[0] * best[1], best[1])):
                best = (step, n)
    step, n = best if best else (hi, 1)
    return [step * i for i in range(n + 1)]


def label(v):
    return ("%.1f" % v).rstrip("0").rstrip(".") if v != int(v) else "%d" % v


def polyline(pts, style):
    dash = ' stroke-dasharray="%s"' % style["dash"] if style.get("dash") else ""
    return ('<polyline points="%s" fill="none" stroke="%s" stroke-width="%g"%s stroke-linejoin="round"/>'
            % (pts, style["stroke"], style["width"], dash))


def peak(series):
    return max((v for pts, _ in series for _, v in pts if v is not None), default=1.0)


def panel(svg, xs, t0, t1, series, marks, ylabel, scale, caption, legend=None, hi=None):
    """One plot: the marks [(kind, t, name)] under the series [(points [(t, v)], style)], then its
    frame, ticks, y label with matplotlib's offset ("1e3"), x label and subcaption, and a legend
    [(name, style)] in its lower right corner. A style: stroke, width, dash, and area (shaded down to
    0). The y axis reaches past hi, the series' peak unless given (plots that share a range)."""
    x0, x1 = xs
    ts = ticks((peak(series) if hi is None else hi) / scale * 1.04)
    top = ts[-1]
    x = lambda t: x0 + (x1 - x0) * (t - t0) / (t1 - t0)  # noqa: E731
    y = lambda v: Y1 - (Y1 - Y0) * (v / scale) / top  # noqa: E731
    for kind, t, _ in marks:
        if t0 <= t <= t1:
            svg.append('<line x1="%.2f" y1="%d" x2="%.2f" y2="%d" stroke="%s" stroke-width="1" '
                       'stroke-dasharray="%s"/>' % (x(t), Y0, x(t), Y1, MARK, DASH[kind]))
    for pts, style in series:
        pts = [(t, v) for t, v in pts if v is not None and t0 <= t <= t1]
        if not pts:
            continue
        line = " ".join("%.2f,%.2f" % (x(t), y(v)) for t, v in pts)
        if style.get("area"):
            svg.append('<polygon points="%.2f,%d %s %.2f,%d" fill="#e3e3e3" stroke="none"/>'
                       % (x(pts[0][0]), Y1, line, x(pts[-1][0]), Y1))
        svg.append(polyline(line, style))
    svg.append('<rect x="%.2f" y="%d" width="%.2f" height="%d" fill="none" stroke="#000" stroke-width="0.6"/>'
               % (x0, Y0, x1 - x0, Y1 - Y0))
    for v in ts:
        yv = Y1 - (Y1 - Y0) * v / top
        svg.append('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#000" stroke-width="0.6"/>'
                   % (x0, yv, x0 - 2.5, yv))
        svg.append('<text x="%.2f" y="%.2f" font-size="6.5" text-anchor="end">%s</text>'
                   % (x0 - 3.8, yv + 2.3, label(v)))
    step = 100 if t1 - t0 > 150 else 50
    for t in range(int(math.ceil(t0 / step)) * step, int(t1) + 1, step):
        svg.append('<line x1="%.2f" y1="%d" x2="%.2f" y2="%.1f" stroke="#000" stroke-width="0.6"/>'
                   % (x(t), Y1, x(t), Y1 + 2.5))
        svg.append('<text x="%.2f" y="%.1f" font-size="6.5" text-anchor="middle">%d</text>' % (x(t), Y1 + 9.5, t))
    if scale != 1:
        svg.append('<text x="%.2f" y="%.1f" font-size="5.5">1e%d</text>' % (x0, Y0 - 2.5, round(math.log10(scale))))
    # The y label clears the widest tick label (6.5 pt digits, about 3.6 pt each).
    width = max(len(label(v)) for v in ts)
    svg.append('<text transform="translate(%.2f,%.1f) rotate(-90)" font-size="7" text-anchor="middle">%s</text>'
               % (x0 - 9 - 3.6 * width, (Y0 + Y1) / 2, esc(ylabel)))
    svg.append('<text x="%.2f" y="%.1f" font-size="6.5" text-anchor="middle">Time (s)</text>'
               % ((x0 + x1) / 2, Y1 + 18))
    svg.append('<text x="%.2f" y="%.1f" font-size="7.5" text-anchor="middle">%s</text>'
               % ((x0 + x1) / 2, Y1 + 29.5, esc(caption)))
    for i, (name, style) in enumerate(reversed(legend or [])):
        ky = Y1 - 5 - 8 * i
        svg.append(polyline("%.2f,%.1f %.2f,%.1f" % (x1 - 47, ky - 2, x1 - 37, ky - 2), style))
        svg.append('<text x="%.2f" y="%.1f" font-size="6">%s</text>' % (x1 - 35, ky, esc(name)))


def shares(nf, key, width, t0, t1):
    """Each node's share of the sum of a node_flows key, over the windows inside [t0, t1]."""
    means = {}
    for n, ws in nf.items():
        vs = [w[key] for w in ws if w[key] is not None and w["t"] >= t0 and w["t"] + width <= t1]
        means[n] = sum(vs) / len(vs) if vs else 0.0
    total = sum(means.values()) or 1.0
    return {n: v / total for n, v in means.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="repetitions of one protocol and plan (results directories)")
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--node-metric", choices=sorted(METRICS), action="append",
                    help="a panel of each node's page reads or of the candidates it re-ranked, per second; "
                         "repeat for both, in the order wanted (default: reads)")
    ap.add_argument("--window", type=float, default=2.0, help="seconds per point of the rates (at least 1)")
    ap.add_argument("--backlog-window", type=float, default=1.0, help="seconds per point of the backlog")
    args = ap.parse_args()
    metrics = args.node_metric or ["reads"]
    if len(set(metrics)) < len(metrics):
        sys.exit("--node-metric: give each at most once")

    exp = json.load(open(os.path.join(args.runs[0], "experiment.json")))
    skip = float(exp.get("Load", {}).get("WarmupSeconds", 0))
    ends = [e for e in (load_end(d) for d in args.runs) if e]
    if not ends:
        sys.exit("no run says where its load stopped")
    end = min(ends)
    sched = offered_schedule(args.runs[0])
    if not sched:
        sys.exit("%s: not an open-loop run, so no offered rate" % args.runs[0])
    if any(offered_schedule(d) != sched for d in args.runs[1:]):
        sys.exit("the runs have different offered rates: give repetitions of one experiment")
    runs = load("runs", args.runs, skip)
    rcs = [r for _, r in runs]
    if not all(rcs):
        sys.exit("a run without a reconfiguration: nothing rebalanced")
    trigger = sum(r[0][0] for r in rcs) / len(rcs)
    flips = [flip_time(r[0][0], r[0][3]) for r in rcs]
    flip = sum(flips) / len(flips) if None not in flips else None
    rises = [t for (t, r), (_, r0) in zip(sched[1:], sched) if r > r0]

    w = max(args.window, 1.0)
    merged = sorted((r for rows, _ in runs for r in rows), key=lambda r: r["t"])
    fl = flow([r for r in merged if r["t"] < end], w, int(math.ceil(end / w)), len(runs))
    answered = [(x["t"] + w / 2, x["answered"]) for x in fl if x["t"] >= skip and x["t"] + w <= end]
    offered = [(skip, rate_at(sched, skip))]
    for t, r in sched:
        if skip < t < end:
            offered += [(t, offered[-1][1]), (t, r)]
    offered.append((end, offered[-1][1]))
    nf = node_flows([node_stats(d) for d in args.runs], w, int(math.ceil(end / w)))
    per_node = [(m, {n: [(x["t"] + w / 2, x[METRICS[m][0]]) for x in ws if x["t"] + w <= end]
                     for n, ws in nf.items()}) for m in metrics]
    bw = args.backlog_window
    bl = flow([r for r in merged if r["t"] < end], bw, int(math.ceil(end / bw)), len(runs))
    backlog = [(x["t"] + bw, x["backlog"]) for x in bl if x["t"] + bw <= end]

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%dpt" height="%dpt" '
           'font-family="%s" fill="#000">' % (W, H, W, H, FONT),
           '<rect x="0" y="0" width="%d" height="%d" fill="#fff"/>' % (W, H)]
    marks = [("load", t, "Load increase") for t in rises[:1]] + [("trigger", trigger, "Rebalance")]
    if flip is not None:
        marks.append(("flip", flip, "Flip"))
    cols = columns(2 + len(metrics))
    letters = "abcdefgh"
    blue = {"stroke": LINE, "width": 1}
    panel(svg, cols[0], skip, end, [(answered, blue), (offered, OFFERED)], marks, "Throughput (q/s)", 1,
          "(a) Throughput", legend=[("offered", OFFERED), ("answered", blue)])
    # The nodes' panels share a range, so that their lines' heights compare.
    thin = {"stroke": LINE, "width": 0.8}
    shared = max(peak([(pts, thin) for pts in nodes.values()]) for _, nodes in per_node)
    for i, ((m, nodes), xs) in enumerate(zip(per_node, cols[1:-1])):
        panel(svg, xs, skip, end, [(pts, thin) for pts in nodes.values()], marks, METRICS[m][1], 1000,
              "(%s) %s" % (letters[i + 1], METRICS[m][2]), hi=shared)
    panel(svg, cols[-1], skip, end, [(backlog, dict(blue, area=True))], marks, "Backlog (queries)", 1000,
          "(%s) Backlog" % letters[len(cols) - 1])
    # Legend across the top, as Koala's: a sample of each mark and its name.
    widths = [16 + 3 + 0.6 * 7 * len(name) for _, _, name in marks]
    lx = (W - sum(widths) - 10 * (len(widths) - 1)) / 2
    for (kind, _, name), wd in zip(marks, widths):
        svg.append('<line x1="%.1f" y1="8" x2="%.1f" y2="8" stroke="%s" stroke-width="1.2" stroke-dasharray="%s"/>'
                   % (lx, lx + 16, MARK, DASH[kind]))
        svg.append('<text x="%.1f" y="10.5" font-size="7">%s</text>' % (lx + 19, esc(name)))
        lx += wd + 10
    svg.append("</svg>")
    with open(args.out + ".svg", "w") as f:
        f.write("\n".join(svg))
    with open(args.out + ".csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["panel", "series", "t", "value"])
        for name, pts in [("offered", offered), ("answered", answered)]:
            out.writerows(("a", name, "%g" % t, "%.1f" % v) for t, v in pts)
        for i, (m, nodes) in enumerate(per_node):
            for n, pts in nodes.items():
                out.writerows((letters[i + 1], "%s %s" % (n, m), "%g" % t, "" if v is None else "%.0f" % v)
                              for t, v in pts)
        out.writerows((letters[len(cols) - 1], "backlog", "%g" % t, "%.1f" % v) for t, v in backlog)

    most = max(backlog, key=lambda p: p[1])
    print("%d runs, %d nodes; load increase at %s s, rebalance at %.1f s, flip at %s s; backlog peak %.0f at %.0f s"
          % (len(runs), len(nf), ", ".join("%g" % t for t in rises[:1]) or "-", trigger,
             "%.1f" % flip if flip is not None else "-", most[1], most[0]))
    for m in metrics:
        for name, (a, b) in [("the 30 s before the rebalance", (trigger - 30, trigger)),
                             ("the load's last minute", (end - 60, end))]:
            sh = shares(nf, METRICS[m][0], w, a, b)
            top = max(sh, key=sh.get)
            print("%s, %s: %s; busiest %s, %.2fx an even share"
                  % (m, name, " ".join("%s %.1f%%" % (n, 100 * v) for n, v in sh.items()), top, sh[top] * len(sh)))
    print("wrote %s.svg and %s.csv" % (args.out, args.out))


if __name__ == "__main__":
    main()
