#!/usr/bin/env python3
"""Figures for the Wikipedia 4 -> 8 runs on CloudLab c240g5 (a P100 and an S3500 per node,
2026-11-03/04). No dependencies: plain SVG in light and dark, as compare_runs.py writes them.

    python3 scripts/analysis/cloudlab_gpu_figures.py [--runs results/cloudlab-gpu] [--out results/cloudlab-gpu-figures]
        [--series 4to8] [--calibrations 4to8] [--before SERIES] [--fetch-bound-mb 2048]

The runs directory is what came back from the control node (the runbook's section 8); the
figures go to a directory of their own, so that copying the runs again leaves them alone. A
series is a directory of the runs with rep1/ and rep2/ (lazy, copy-then-flip) and 4to8to4/, as
run_one.sh wrote them; the calibrations (calibrate-4, calibrate-8) can come from another series,
as for a rerun after a change. --fetch-bound-mb is the fetch cache's bound the runs had (64 before
2026-11-04, 2048 since). Every number a figure's notes cite is computed from the runs. Writes,
into --out:

  4to8.svg, 4to8-rep1.svg, 4to8-rep2.svg, 4to8to4.svg
                    compare_runs.py: lazy against copy-then-flip (both repetitions pooled, then
                    each one), and 4 -> 8 -> 4 under lazy; a .csv and a .json beside each
  capacity.svg      the calibrations: answered queries and mean latency against the offered
                    rate, on 4 and on 8 nodes
  lazy-fetch.svg    lazy from the trigger until the new nodes are warm: answered queries, the
                    new nodes' fetch cache against its bound, the time per fetch, their disk writes
  node-reads.svg    the reads per second charged to each node's budget, lazy and copy-then-flip
  ctf-stream.svg    copy-then-flip's raw-vector stream from each initial node, under the load and
                    without it (calibrate-8 scales out before its load)

  lazy-before-after.svg
                    with --before: lazy in that series against lazy in this one (both repetitions
                    pooled), for a change that only lazy's path sees

and a PNG of each when rsvg-convert is installed. Runs that are missing are skipped.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from compare_runs import FIXED_SLOT, PALETTE, tick_label, ticks_to  # noqa: E402
from plot_run import esc, flip_time, nice_ticks, read_latency, windows  # noqa: E402

LAZY, CTF = FIXED_SLOT["lazy"], FIXED_SLOT["copy-then-flip"]
# Two highlighted series that are not protocols (cluster sizes; the two hot new nodes): palette
# slots 2 and 6, green and purple. They stay apart for every colour-vision type in both modes
# (OKLab dE x 100: 35.8 / 26.6 at worst in light, 24.6 / 9.4 in dark), where four nodes in
# four slots would not (green against pink: 1.6 in dark); the other nodes are context, in grey.
HOT_A, HOT_B = 2, 6
CTX = ("#bdbcb4", "#6e6d66")      # context lines: apart from both highlights (dE >= 18 / 15)
REF_GREY = ("#8d8c84", "#8d8c84")  # copy-then-flip without load, against its orange (17 / 10)
# reads/s against the 40K budget in 5K steps: one hue (OKLab 250 deg), light to dark, re-stepped
# for the dark surface; adjacent steps dE ~9, at worst 7.3 under colour-vision deficiency, so
# every cell also names its value on hover.
HEAT = (["#eff6fd", "#cbdaeb", "#a7bfd9", "#84a5c7", "#618ab5", "#3e70a3", "#165790", "#003d7d"],
        ["#1f2730", "#2f3e4d", "#40566c", "#526f8d", "#6489b0", "#77a4d3", "#8bc0f8", "#9fddff"])
SURFACE = ({"bg": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e1e0d9", "axis": "#c3c2b7"},
           {"bg": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "grid": "#2c2c2a", "axis": "#383835"})

W, LEFT, RIGHT = 900, 64, 150


# ---------------------------------------------------------------------------------------------
# data

def load_json(path):
    with open(path) as f:
        return json.load(f)


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def field(rec, path):
    for k in path.split("."):
        rec = rec[k]
    return rec


def load_start(run):
    p = os.path.join(run, "clock.json")
    return load_json(p)["load_start"] if os.path.exists(p) else None


def series(recs, path, t0):
    return [(r["t"] - t0, field(r, path)) for r in recs]


def value_at(pts, t):
    """A cumulative counter at t, linear between samples; None outside the samples."""
    if not pts or t < pts[0][0] or t > pts[-1][0]:
        return None
    lo, hi = 0, len(pts) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pts[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    (ta, va), (tb, vb) = pts[lo], pts[hi]
    return va if tb == ta else va + (vb - va) * (t - ta) / (tb - ta)


def deltas(pts, lo, hi, step):
    """Per bin of `step` seconds: (bin middle, the counter's increase in it), None outside the samples."""
    out, t = [], lo
    while t + step <= hi + 1e-9:
        a, b = value_at(pts, t), value_at(pts, t + step)
        out.append((t + step / 2, None if a is None or b is None else b - a))
        t += step
    return out


def rates(pts, lo, hi, step, scale=1.0):
    return [(t, None if d is None else d / step * scale) for t, d in deltas(pts, lo, hi, step)]


def host_records(run, node):
    return [r for r in jsonl(os.path.join(run, "cgroups.jsonl")) if r.get("node") == node + "-host"]


def events(run):
    """(trigger, flip, [entry flips]) of the run's first reconfiguration, in seconds of the load
    (plot_run.flip_time: the flip as README's "The flip" defines it)."""
    rc = load_json(os.path.join(run, "reconfigurations.json"))[0]
    reply, trig = rc["reply"], rc["trigger_seconds"]
    flip = flip_time(trig, reply)
    return trig, flip, [trig + e["seconds"] for e in reply.get("entry_flips", [])], rc


# ---------------------------------------------------------------------------------------------
# SVG

def style():
    def rules(m):
        s = SURFACE[m]
        r = [".bg { fill: %s; }" % s["bg"], ".dot { stroke: %s; }" % s["bg"],
             ".title { fill: %s; }" % s["ink"], ".sub, .legend, .tick, .note { fill: %s; }" % s["ink2"],
             ".grid { stroke: %s; }" % s["grid"], ".axis { stroke: %s; }" % s["axis"],
             ".ref, .ev, .cu, .hit line { stroke: %s; }" % s["ink2"],
             ".ctx { stroke: %s; } .fctx { fill: %s; }" % (CTX[m], CTX[m]),
             ".gry { stroke: %s; } .fgry { fill: %s; }" % (REF_GREY[m], REF_GREY[m])]
        r += [".c%d { stroke: %s; } .f%d { fill: %s; }" % (i, c[m], i, c[m]) for i, c in enumerate(PALETTE)]
        r += [".h%d { fill: %s; }" % (i, c) for i, c in enumerate(HEAT[m])]
        return "\n".join("  " + x for x in r)

    # Colours per class, not CSS variables, so that rsvg-convert renders as browsers do.
    return """<style>
  .title { font-size: 14px; font-weight: 600; }
  .ptitle { font-size: 12px; font-weight: 600; }
  .sub, .legend, .tick, .note { font-size: 11px; }
  .tick { font-variant-numeric: tabular-nums; }
  .grid, .axis { stroke-width: 1; }
  .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .thin { stroke-width: 1.5; }
  .ref { stroke-width: 1; stroke-dasharray: 4 3; fill: none; }
  .ev { stroke-width: 1; stroke-dasharray: 2 3; }
  .sched { stroke-width: 2.5; stroke-dasharray: 8 5; }
  .dot { stroke-width: 2; }
  .hit rect { fill: transparent; }
  .hit line { stroke-width: 1; opacity: 0; }
  .hit:hover line { opacity: .6; }
%s
  @media (prefers-color-scheme: dark) {
%s
  }
</style>""" % (rules(0), rules(1))


class Svg:
    def __init__(self):
        self.body = []

    def add(self, s):
        self.body.append(s)

    def text(self, x, y, s, cls="tick", anchor="start", weight=None):
        w = ' font-weight="%s"' % weight if weight else ""
        self.add('<text class="%s" x="%.1f" y="%.1f" text-anchor="%s"%s>%s</text>' % (cls, x, y, anchor, w, esc(s)))

    def legend(self, x, y, items):
        """items: (label, class of the line or swatch, kind: line / dash / box)."""
        for i, (label, cls, kind) in enumerate(items):
            yy = y + i * 18
            if kind == "box":
                self.add('<rect class="%s" x="%d" y="%d" width="14" height="10" rx="2"/>' % (cls, x, yy - 9))
            elif kind == "line":
                self.add('<line class="line %s" x1="%d" y1="%d" x2="%d" y2="%d"/>' % (cls, x, yy - 4, x + 16, yy - 4))
            self.text(x + 22, yy, label, "legend")

    def save(self, out, name, h, title, subtitle):
        path = os.path.join(out, name)
        head = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" '
                'font-family="system-ui, -apple-system, Segoe UI, sans-serif">' % (W, h, W, h))
        doc = [head, style(), '<rect class="bg" x="0" y="0" width="%d" height="%d"/>' % (W, h),
               '<text class="title" x="%d" y="22">%s</text>' % (LEFT, esc(title)),
               '<text class="sub" x="%d" y="40">%s</text>' % (LEFT, esc(subtitle))] + self.body + ["</svg>"]
        with open(path, "w") as f:
            f.write("\n".join(doc) + "\n")
        print("wrote", path)
        return path


class Panel:
    """A chart area: y linear from 0 (ymax) or log (lo, hi decades); x a range of seconds or rates."""

    def __init__(self, svg, x0, y0, pw, ph, xr, title, unit, ymax=1.0, log=None, yfmt=None, grid=True,
                 title_dy=-10):
        self.svg, self.x0, self.y0, self.pw, self.ph, self.xr = svg, x0, y0, pw, ph, xr
        if log:
            lo, hi = log
            span = math.log10(hi) - math.log10(lo)
            self.y = lambda v: y0 + ph - ph * (math.log10(min(max(v, lo), hi)) - math.log10(lo)) / span
            ticks = [lo * 10 ** i for i in range(int(round(span)) + 1)]
            label = yfmt or (lambda v: "%g" % v)
        else:
            ticks = ticks_to(ymax or 1.0)
            top = ticks[-1] or 1.0
            self.y = lambda v: y0 + ph - ph * min(max(v, 0.0), top) / top
            step = ticks[1] - ticks[0] if len(ticks) > 1 else 1
            label = yfmt or (lambda v, step=step: tick_label(v, step))
        svg.text(x0, y0 + title_dy, title, "title ptitle")
        for v in ticks if grid else []:
            svg.add('<line class="grid" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>' % (x0, self.y(v), x0 + pw, self.y(v)))
            svg.text(x0 - 8, self.y(v) + 4, label(v), anchor="end")
        svg.text(x0 + pw + 8, y0 + ph + 4, unit)
        svg.add('<line class="axis" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>' % (x0, y0 + ph, x0 + pw, y0 + ph))

    def x(self, t):
        return self.x0 + self.pw * (t - self.xr[0]) / (self.xr[1] - self.xr[0])

    def line(self, pts, cls):
        run = []
        for t, v in list(pts) + [(None, None)]:
            if t is not None and not self.xr[0] <= t <= self.xr[1]:
                continue
            if v is None:  # no sample: a gap, not a zero
                if len(run) > 1:
                    self.svg.add('<polyline class="line %s" points="%s"/>' % (cls, " ".join("%.1f,%.1f" % p for p in run)))
                run = []
                continue
            run.append((self.x(t), self.y(v)))

    def hline(self, v, label=None, cls="ref", anchor="end", dy=-4):
        y = self.y(v)
        self.svg.add('<line class="%s" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>' % (cls, self.x0, y, self.x0 + self.pw, y))
        if label:
            x = self.x0 + self.pw - 4 if anchor == "end" else self.x0 + 4
            self.svg.text(x, y + dy, label, "note", anchor)

    def vline(self, t, label=None, anchor="start", row=0, above=False):
        """A moment across the panel; its label inside the top, or above the panel where marks
        fill it (a heatmap)."""
        x = self.x(t)
        self.svg.add('<line class="ev" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>' % (x, self.y0, x, self.y0 + self.ph))
        if label:
            y = self.y0 - 3 if above else self.y0 + 11 + 13 * row
            self.svg.text(x + (3 if anchor == "start" else -3), y, label, "note", anchor)


def time_axis(svg, p, y, lo, hi, label="seconds since the load started", count=8):
    step_ticks = nice_ticks(hi - lo, count)
    step = step_ticks[1] - step_ticks[0]
    t = math.ceil(lo / step) * step
    while t <= hi + 1e-9:
        svg.text(p.x(t), y + 16, tick_label(t, step), anchor="middle")
        t += step
    svg.text(p.x0 + p.pw / 2, y + 32, label, anchor="middle")


def hover(svg, panels, lo, hi, step, describe):
    """A crosshair per bin over the stacked panels, one readout per bin (describe(t0) -> lines)."""
    top, bottom = panels[0].y0, panels[-1].y0 + panels[-1].ph
    p = panels[0]
    t = lo
    while t < hi - 1e-9:
        x0, x1 = p.x(t), p.x(min(t + step, hi))
        svg.add('<g class="hit"><title>%s</title><rect x="%.1f" y="%.1f" width="%.1f" height="%.1f"/>'
                '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/></g>'
                % (esc("\n".join(describe(t))), x0, top, max(x1 - x0, 0.5), bottom - top,
                   (x0 + x1) / 2, top, (x0 + x1) / 2, bottom))
        t += step


def fmt_k(v):
    if v is None:
        return "–"
    if abs(v) < 1000:
        return "%.0f" % v
    return "%dK" % (v // 1000) if v % 1000 == 0 else "%.1fK" % (v / 1000)


# ---------------------------------------------------------------------------------------------
# figures

def compare_specs(runset, before=None):
    """compare_runs.py figures: (name, title, run dirs under --runs, labels, slots)."""
    tag = "" if runset == "4to8" else " [%s]" % runset
    head = "Wikipedia 4 → 8 at 90 s, load 390 → 780 q/s at 60 s, CloudLab c240g5"
    specs = [
        ("4to8", head + " (2 runs per protocol)" + tag,
         ["%s/rep%d/%s" % (runset, r, p) for p in ("lazy", "copy-then-flip") for r in (1, 2)], None, None),
        ("4to8-rep1", head + " (repetition 1)" + tag,
         ["%s/rep1/lazy" % runset, "%s/rep1/copy-then-flip" % runset], None, None),
        ("4to8-rep2", head + " (repetition 2)" + tag,
         ["%s/rep2/lazy" % runset, "%s/rep2/copy-then-flip" % runset], None, None),
        ("4to8to4", "Wikipedia 4 → 8 at 90 s → 4 at 300 s under lazy; load 390 → 780 q/s at 60 s → 390 at 280 s" + tag,
         ["%s/4to8to4" % runset], "lazy", None),
    ]
    if before:
        # Both lines are lazy: lazy's own colour for this series, a validated neighbour of it (the
        # palette's third slot) for the one before -- not copy-then-flip's orange.
        specs.append(("lazy-before-after", head + ": lazy in %s and in %s (2 runs each)" % (before, runset),
                      ["%s/rep%d/lazy" % (s_, r) for s_ in (before, runset) for r in (1, 2)],
                      ",".join(["lazy (%s)" % before] * 2 + ["lazy (%s)" % runset] * 2),  # no comma: the separator
                      "%d,%d" % (HOT_A, LAZY)))
    return specs


def timelines(runs, out, specs):
    for name, title, dirs, labels, slots in specs:
        paths = [os.path.join(runs, d) for d in dirs]
        if not all(os.path.exists(os.path.join(p, "latency.csv")) for p in paths):
            print("skipped", name, "(runs missing)")
            continue
        cmd = [sys.executable, os.path.join(os.path.dirname(HERE), "compare_runs.py")] + paths + \
              ["--out", os.path.join(out, name), "--title", title]
        if labels:
            cmd += ["--labels", labels]
        if slots:
            cmd += ["--slots", slots]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        print("wrote", os.path.join(out, name + ".svg"))


def capacity(runs, out, runset, calibrations):
    cals = [("4 nodes", HOT_A, os.path.join(runs, calibrations, "calibrate-4")),
            ("8 nodes", HOT_B, os.path.join(runs, calibrations, "calibrate-8"))]
    cals = [(l, s, load_json(os.path.join(d, "calibration.json"))) for l, s, d in cals
            if os.path.exists(os.path.join(d, "calibration.json"))]
    if len(cals) < 2:
        print("skipped capacity.svg (calibrations missing)")
        return
    exp = os.path.join(runs, runset, "rep1", "lazy", "experiment.json")
    if not os.path.exists(exp):
        print("skipped capacity.svg (no %s)" % exp)
        return
    rate = load_json(exp)["Load"]
    lo_rate, hi_rate = float(rate["Rate"]), float(rate["RateSteps"].split(",")[0].split(":")[1])
    svg = Svg()
    pw = W - LEFT - RIGHT
    xr = (0, 1700)
    top_ans = max(s["answered_in_window_per_s"] for _, _, st in cals for s in st)
    p1 = Panel(svg, LEFT, 84, pw, 230, xr, "answered queries per second (in the 30 s window of each step)", "q/s",
               ymax=max(top_ans, 1000))
    p2 = Panel(svg, LEFT, 384, pw, 170, xr, "mean latency (log scale)", "ms", log=(10, 100000),
               yfmt=lambda v: "{:,.0f}".format(v))
    # answered = offered: below it, the cluster falls behind and the backlog grows
    p1.svg.add('<line class="ref" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>'
               % (p1.x(0), p1.y(0), p1.x(1050), p1.y(1050)))
    svg.text(p1.x(1050) - 4, p1.y(1050) - 6, "answered = offered", "note", "end")
    for p in (p1, p2):
        for r, lab in ((lo_rate, "runs: %.0f" % lo_rate), (hi_rate, "→ %.0f q/s" % hi_rate)):
            p.vline(r, lab if p is p1 else None)
    plateaus = []
    for label, slot, steps in cals:
        pts1 = [(s["offered_per_s"], s["answered_in_window_per_s"]) for s in steps]
        pts2 = [(s["offered_per_s"], s["mean_us"] / 1000.0) for s in steps]
        p1.line(pts1, "c%d" % slot)
        p2.line(pts2, "c%d" % slot)
        for s, (x, a), (_, m) in zip(steps, pts1, pts2):
            tip = "%s, offered %d q/s: answered %.1f q/s, mean %.1f ms, p99 %.0f ms, %d behind at the end" % (
                label, s["offered_per_s"], a, m, s["p99_us"] / 1000.0, s["behind_at_end"])
            for p, v in ((p1, a), (p2, m)):
                svg.add('<circle class="dot f%d" cx="%.1f" cy="%.1f" r="4"><title>%s</title></circle>'
                        % (slot, p.x(x), p.y(v), esc(tip)))
        plateau = sum(a for _, a in pts1[-3:]) / 3
        plateaus.append(plateau)
        x, a = pts1[-1]
        # beside the plateau, ending at its last step, on the side no other line takes: clear of
        # the legend in the right margin
        svg.text(p1.x(x), p1.y(a) + (20 if slot == HOT_A else -12), "%s: %.0f q/s" % (label, plateau), "note", "end")
    time_axis(svg, p2, p2.y0 + p2.ph, *xr, label="offered queries per second (Poisson arrivals)")
    svg.legend(W - RIGHT + 16, p1.y0 + 10, [(l, "c%d" % s, "line") for l, s, _ in cals])
    notes = ["The plateau is the mean of the last three steps. Every step is bound by the busiest node's",
             "budget of 40,000 reads/s (node_reads.py): its share of the reads is above 1/n, so 8 nodes carry",
             "%.2f times what 4 carry. The runs offer %.0f q/s, then %.0f: above 4 nodes' capacity, below 8 nodes'."
             % (plateaus[1] / plateaus[0], lo_rate, hi_rate)]
    for i, n in enumerate(notes):
        svg.text(LEFT, p2.y0 + p2.ph + 62 + 15 * i, n, "note")
    return svg.save(out, "capacity.svg", p2.y0 + p2.ph + 62 + 15 * len(notes) + 10,
                    "Capacity on CloudLab c240g5: 4 and 8 nodes, 40,000 reads/s per node",
                    "the calibrations: 30 s per offered rate, the same queries (NProbe 64, N 200, k 10); "
                    "8 nodes after copy-then-flip moved every raw vector")


def lazy_fetch(runs, out, runset, calibrations, bound_mb, lo=80.0, hi=200.0):
    rel = "%s/rep1/lazy" % runset
    run = os.path.join(runs, rel)
    if not os.path.exists(os.path.join(run, "stats-n5.jsonl")):
        print("skipped lazy-fetch.svg (run missing)")
        return
    t0 = load_start(run)
    trig, flip, entries, _ = events(run)
    cal4 = os.path.join(runs, calibrations, "calibrate-4", "calibration.json")
    c4 = sum(s["answered_in_window_per_s"] for s in load_json(cal4)[-3:]) / 3 if os.path.exists(cal4) else None
    exp = load_json(os.path.join(run, "experiment.json"))["Load"]
    offered = float(exp["RateSteps"].split(",")[0].split(":")[1])
    win = {round(w["t"]): w["answered"] for w in windows(read_latency(os.path.join(run, "latency.csv")), 1.0)}
    answered = [(t + 0.5, win.get(t)) for t in range(int(lo), int(hi))]
    new = ["n5", "n6", "n7", "n8"]
    cls = {"n5": "c%d" % HOT_A, "n8": "c%d" % HOT_B, "n6": "ctx thin", "n7": "ctx thin"}
    st = {n: jsonl(os.path.join(run, "stats-%s.jsonl" % n)) for n in new}
    queued = {n: [(r["t"] - t0, r["fetch"]["queued"]) for r in st[n]] for n in new}
    step = 2.0  # a fetch call can take a second: per 2 s, most bins have some
    per_call = {}
    for n in new:
        us, calls = series(st[n], "fetch.fetch_us", t0), series(st[n], "fetch.fetch_calls", t0)
        per_call[n] = [(t, (u / c / 1000.0) if u is not None and c else None)
                       for (t, u), (_, c) in zip(deltas(us, lo, hi, step), deltas(calls, lo, hi, step))]
    call_bin = {n: {int((t - lo) // step): v for t, v in per_call[n]} for n in new}
    writes = {n: rates([(r["t"] - t0, r["disk"]["wbytes"]) for r in host_records(run, n)], lo, hi, step, 1e-6)
              for n in new}
    # The cache's bound (engine/src/node_engine.cpp, kMaxQueuedBytes: 64 MB until 2026-11-04, 2 GB
    # since) in vectors of this index; drawn when it is within reach of what the cache held.
    reply = events(run)[3]["reply"]  # the bytes per raw vector, as the rescale counted its fetches
    vec_bytes = reply["raw_fetched_bytes"] // reply["raw_fetched"] if reply.get("raw_fetched") else 4096
    bound = bound_mb * (1 << 20) // vec_bytes
    peak = {n: max(v for _, v in queued[n]) for n in new}
    top = max(peak.values()) or 1
    show_bound = bound <= 2 * top
    last = {n: st[n][-1]["fetch"] for n in new}
    # the holders' budget waits in the 30 s after the flip: what a fetch from them waits behind
    src_wait = {}
    for n in ("n1", "n4"):
        pts = series(jsonl(os.path.join(run, "stats-%s.jsonl" % n)), "reads.wait_us", t0)
        a, b = value_at(pts, flip), value_at(pts, flip + 30)
        src_wait[n] = None if a is None or b is None else (b - a) / 30 / 1e6

    svg = Svg()
    pw = W - LEFT - RIGHT
    xr = (lo, hi)
    ys = [84, 254, 404, 554]
    p1 = Panel(svg, LEFT, ys[0], pw, 120, xr, "answered queries per second (1 s windows)", "q/s",
               ymax=max(v for _, v in answered if v is not None))
    p2 = Panel(svg, LEFT, ys[1], pw, 100, xr, "fetched raw vectors waiting for the SSD (the fetch cache)", "vectors",
               ymax=max(top, bound if show_bound else 0) * 1.05)
    p3 = Panel(svg, LEFT, ys[2], pw, 100, xr, "time per fetch call (log scale)", "ms", log=(1, 10000),
               yfmt=lambda v: "{:,.0f}".format(v))
    p4 = Panel(svg, LEFT, ys[3], pw, 100, xr, "disk writes on each new node's SSD", "MB/s",
               ymax=max(v for n in new for _, v in writes[n] if v is not None))
    for p in (p1, p2, p3, p4):
        p.vline(trig, "trigger" if p is p1 else None, "end")
        p.vline(flip, "flip" if p is p1 else None)
        for i, e in enumerate(sorted(entries)):
            p.vline(e, ("new entries" if i == 0 else None) if p is p1 else None)
    p1.hline(offered, "offered %.0f q/s" % offered, cls="cu sched", anchor="start", dy=-5)
    if c4:  # labelled at the right end, where the line runs above it
        p1.hline(c4, "4 nodes' capacity, %.0f q/s" % c4, dy=12)
    p1.line(answered, "c%d" % LAZY)
    size = "%d GB" % (bound_mb // 1024) if bound_mb >= 1024 else "%d MB" % bound_mb
    if show_bound:
        p2.hline(bound, "the bound, %s: past it the query that fetched writes its vectors itself" % size, dy=-5)
    for n in ("n6", "n7", "n5", "n8"):  # context first, highlights over it
        p2.line(queued[n], cls[n])
        p3.line(per_call[n], cls[n])
        p4.line(writes[n], cls[n])
    time_axis(svg, p4, p4.y0 + p4.ph, lo, hi)
    svg.legend(W - RIGHT + 16, p1.y0 + 10, [("lazy", "c%d" % LAZY, "line"), ("offered", "cu sched", "line")])
    svg.legend(W - RIGHT + 16, p2.y0 + 10, [("n5 (from n1)", cls["n5"], "line"), ("n8 (from n4)", cls["n8"], "line"),
                                            ("n6, n7", "ctx", "line")])

    def describe(t):
        k = int(t)
        lines = ["%.0f–%.0f s" % (t, t + 1), "answered: %s q/s" % fmt_k(win.get(k))]
        for n in new:
            q = value_at(queued[n], t + 0.5)
            pc = call_bin[n].get(int((t - lo) // step))
            lines.append("%s: %s in the cache, %s ms per fetch" % (n, fmt_k(q), "–" if pc is None else "%.0f" % pc))
        return lines

    hover(svg, [p1, p2, p3, p4], lo, hi, 1.0, describe)
    sub = ("4 → 8, %s: the new nodes own half of the partitions from the flip on and fetch each raw vector "
           "when a query first needs it" % rel.replace("/", ", "))
    gb = lambda v: v * vec_bytes / 1e9
    calls = lambda n: last[n]["fetch_us"] / max(last[n]["fetch_calls"], 1) / 1000.0
    waits = ", ".join("%s %.1f s per second" % (n, w) for n, w in src_wait.items() if w is not None)
    notes = ["n5 and n8 took the hot halves of n1's and n4's partitions. Fetch cache at its fullest: n5 %.2f GB, "
             "n8 %.2f GB," % (gb(peak["n5"]), gb(peak["n8"])),
             "against a bound of %s; vectors the fetching queries had to install themselves: n5 %d, n8 %d. Mean "
             "fetch call:" % (size, last["n5"]["sync_installed"], last["n8"]["sync_installed"]),
             "n5 %.0f ms, n8 %.0f ms. A fetch also waits for its holder's read budget: in the 30 s after the flip "
             "its readers" % (calls("n5"), calls("n8")),
             "waited %s (summed over the readers)." % waits]
    for i, n in enumerate(notes):
        svg.text(LEFT, p4.y0 + p4.ph + 54 + 15 * i, n, "note")
    return svg.save(out, "lazy-fetch.svg", p4.y0 + p4.ph + 54 + 15 * len(notes) + 10,
                    "Lazy right after the flip: the new nodes' fetch cache, their fetches and their disks", sub)


def node_reads(runs, out, runset, lo=30.0, hi=360.0, step=5.0):
    rels = ("%s/rep1/lazy" % runset, "%s/rep1/copy-then-flip" % runset)
    runs_ = [(r, os.path.join(runs, r)) for r in rels if os.path.exists(os.path.join(runs, r, "stats-n1.jsonl"))]
    if len(runs_) < len(rels):
        print("skipped node-reads.svg (runs missing)")
        return
    nodes = ["n%d" % i for i in range(1, 9)]
    svg = Svg()
    pw = W - LEFT - RIGHT
    row, gap = 16, 2
    y = 94
    budget = 40000
    p = None
    for rel, run in runs_:
        t0 = load_start(run)
        trig, flip, entries, rc = events(run)
        proto = rc["reply"]["protocol"]
        ph = len(nodes) * (row + gap)
        # the title a row up: the moments' labels take the row right above the cells
        p = Panel(svg, LEFT, y, pw, ph, (lo, hi), "%s (%s)" % (proto, rel.split("/")[1]), "", grid=False,
                  title_dy=-20)
        for i, n in enumerate(nodes):
            st = jsonl(os.path.join(run, "stats-%s.jsonl" % n))
            rs = rates(series(st, "reads.charged", t0), lo, hi, step)
            ry = y + i * (row + gap)
            svg.text(LEFT - 8, ry + row - 4, n, anchor="end")
            for t, v in rs:
                if v is None:
                    continue
                k = min(len(HEAT[0]) - 1, int(v / (budget / len(HEAT[0]))))
                svg.add('<rect class="h%d" x="%.1f" y="%.1f" width="%.1f" height="%d"><title>%s, %.0f–%.0f s: %s reads/s</title></rect>'
                        % (k, p.x(t - step / 2), ry, p.x(t + step / 2) - p.x(t - step / 2) - 0.6, row,
                           n, t - step / 2, t + step / 2, "{:,.0f}".format(v)))
        # the added nodes below the initial ones
        sep = y + 4 * (row + gap) - gap / 2
        svg.add('<line class="axis" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>' % (LEFT, sep, LEFT + pw, sep))
        p.vline(trig, "trigger", "end", above=True)
        if flip <= hi:
            p.vline(flip, "flip", above=True)
        else:
            svg.text(LEFT + pw - 4, y - 20, "flip at %.0f s, after the load" % flip, "note", "end")
        for i, e in enumerate(sorted(entries)):
            if e <= hi:
                p.vline(e, "new entries" if i == 0 else None, above=True)
        time_axis(svg, p, y + ph, lo, hi)
        y += ph + 80
    # the colour bar: 5K steps up to the budget
    bx, by = W - RIGHT + 16, 94
    svg.text(bx, by - 6, "reads/s", "legend")
    for k in range(len(HEAT[0])):
        yy = by + k * 14
        svg.add('<rect class="h%d" x="%d" y="%d" width="14" height="12" rx="2"/>' % (len(HEAT[0]) - 1 - k, bx, yy))
        lo_k = budget / len(HEAT[0]) * (len(HEAT[0]) - 1 - k)
        svg.text(bx + 22, yy + 10, ("≥ %s" % fmt_k(lo_k)) if k == 0 else "%s–%s" % (fmt_k(lo_k), fmt_k(lo_k + budget / len(HEAT[0]))), "legend")
    svg.text(bx, by + 14 * len(HEAT[0]) + 14, "budget 40K/s", "legend")
    notes = ["n1-n4 hold the data from the start; n5-n8 join at the flip. Under lazy, the new nodes take the reads as",
             "their fetches warm them up; a node at the end of the scale is at its budget and bounds the capacity.",
             "Under copy-then-flip, the new nodes serve nothing before the flip: the initial nodes carry the queries",
             "and the raw stream on one budget."]
    for i, n in enumerate(notes):
        svg.text(LEFT, y - 24 + 15 * i, n, "note")
    return svg.save(out, "node-reads.svg", y + 15 * len(notes) - 14,
                    "Reads per second charged to each node's budget, per 5 s",
                    "the same queries in both runs; the colour scale ends at the budget, 40,000 reads/s")


def calibration_trigger(run):
    """calibrate-8 scales out before its load: its trigger on the controller's clock is the
    ladder's START (calibration.log) less the rescale's elapsed time."""
    with open(os.path.join(run, "calibration.log")) as f:
        start = float(f.readline().split()[1])
    return start - load_json(os.path.join(run, "reconfigurations.json"))[0]["elapsed_seconds"]


def ctf_stream(runs, out, runset, calibrations, lo=0.0, hi=530.0):
    loaded = [os.path.join(runs, runset, r, "copy-then-flip") for r in ("rep1", "rep2")]
    loaded = [r for r in loaded if os.path.exists(os.path.join(r, "stats-n1.jsonl"))]
    calm = os.path.join(runs, calibrations, "calibrate-8")
    if not loaded or not os.path.exists(os.path.join(calm, "calibration.log")):
        print("skipped ctf-stream.svg (runs missing)")
        return
    sources = {"n1": "n5", "n2": "n6", "n3": "n7", "n4": "n8"}
    # each source's share of the query reads on 4 nodes, before the trigger
    share, total = {}, 0
    t0 = load_start(loaded[0])
    for n in sources:
        pts = series(jsonl(os.path.join(loaded[0], "stats-%s.jsonl" % n)), "reads.charged", t0)
        share[n] = value_at(pts, 90) - value_at(pts, 30)
        total += share[n]
    curves = {}
    for n in sources:
        cs = []
        for r in loaded:
            trig = load_start(r) + events(r)[0]
            cs.append(("load", [(rec["t"] - trig, rec["raw_bytes"]["sent"] / 1e9)
                                for rec in jsonl(os.path.join(r, "stats-%s.jsonl" % n))]))
        trig = calibration_trigger(calm)
        cs.append(("calm", [(rec["t"] - trig, rec["raw_bytes"]["sent"] / 1e9)
                            for rec in jsonl(os.path.join(calm, "stats-%s.jsonl" % n))]))
        curves[n] = cs
    load_end = 360 - events(loaded[0])[0]
    svg = Svg()
    cols, gx = 2, 40
    pw = (W - LEFT - RIGHT - gx) / cols
    ph, gy = 130, 74
    ymax = max(v for cs in curves.values() for _, pts in cs for _, v in pts)
    panels = []
    for i, n in enumerate(sources):
        cx, cy = LEFT + (i % cols) * (pw + gx), 84 + (i // cols) * (ph + gy)
        p = Panel(svg, cx, cy, pw, ph, (lo, hi), "%s → %s: %.0f%% of the queries' reads" % (n, sources[n], 100 * share[n] / total),
                  "GB" if i % cols == cols - 1 else "", ymax=ymax)
        p.vline(load_end, "load ends" if i == 0 else None)
        for kind, pts in curves[n]:
            p.line([(t, v) for t, v in pts if lo <= t <= hi], "c%d" % CTF if kind == "load" else "gry")
        time_axis(svg, p, cy + ph, lo, hi, label="seconds since the trigger" if i // cols == 1 else "", count=5)
        panels.append(p)
    svg.legend(W - RIGHT + 16, 94, [("under the load", "c%d" % CTF, "line"), ("(2 runs)", "", "none"),
                                    ("no load", "gry", "line"), ("(calibrate-8)", "", "none")])
    yb = 84 + 2 * (ph + gy) - 10
    took = sorted(events(r)[3]["elapsed_seconds"] for r in loaded)
    calm_took = load_json(os.path.join(calm, "reconfigurations.json"))[0]["elapsed_seconds"]
    notes = ["Copy-then-flip streams each moved partition's raw vectors from its initial owner (RAW_PULL), reading",
             "against the same 40K/s budget as the queries. Under the load, the busiest initial nodes spend theirs on",
             "queries and their streams barely move until the load ends and the backlog drains: the rescale takes",
             "%s s after the trigger, against %.0f s with no load. The node with the fewest queries finishes first."
             % ("-".join("%.0f" % t for t in (took[0], took[-1])) if len(took) > 1 else "%.0f" % took[0], calm_took)]
    for i, s in enumerate(notes):
        svg.text(LEFT, yb + 15 * i, s, "note")
    return svg.save(out, "ctf-stream.svg", yb + 15 * len(notes) + 6,
                    "Copy-then-flip's raw-vector stream from each initial node: 43.1 GB in all",
                    "the raw vectors each initial node has sent to its new node, cumulative")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default="results/cloudlab-gpu")
    ap.add_argument("--out", default="results/cloudlab-gpu-figures")
    ap.add_argument("--series", default="4to8", help="the runs to draw: a directory under --runs (default 4to8)")
    ap.add_argument("--calibrations", default="4to8",
                    help="the directory under --runs with calibrate-4 and calibrate-8 (default 4to8)")
    ap.add_argument("--before", help="another series, whose lazy runs lazy-before-after.svg draws against these")
    ap.add_argument("--fetch-bound-mb", type=int, default=2048,
                    help="the fetch cache's bound the runs had (default 2048; 64 before 2026-11-04, as in 4to8)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    specs = compare_specs(args.series, args.before)
    timelines(args.runs, args.out, specs)
    made = [capacity(args.runs, args.out, args.series, args.calibrations),
            lazy_fetch(args.runs, args.out, args.series, args.calibrations, args.fetch_bound_mb),
            node_reads(args.runs, args.out, args.series),
            ctf_stream(args.runs, args.out, args.series, args.calibrations)]
    made += [os.path.join(args.out, n + ".svg") for n, _, _, _, _ in specs]
    if shutil.which("rsvg-convert"):
        for path in made:
            if path and os.path.exists(path):
                subprocess.run(["rsvg-convert", "-w", "1800", path, "-o", path[:-4] + ".png"], check=True)


if __name__ == "__main__":
    main()
