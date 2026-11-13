#!/usr/bin/env python3
"""Figures for the 4 -> 8 runs on CloudLab c240g5 (a P100 and an S3500 per node): Wikipedia
(2026-11-03/05) and Amazon 300M (2026-11-07). No dependencies: plain SVG in light and dark, as
compare_runs.py writes them. The titles take the dataset, the reconfigurations and the load from
the runs' experiment.json.

    python3 scripts/analysis/cloudlab_gpu_figures.py [--runs results/cloudlab-gpu] [--out results/cloudlab-gpu-figures/SERIES]
        [--series 4to8] [--calibrations 4to8] [--before SERIES] [--fetch-bound-mb 2048] [--drained] [--lazy-series SERIES]
        [--window 5]

The runs directory is what came back from the control node (the runbook's section 8); the
figures go to a directory of their own, so that copying the runs again leaves them alone: by
default one per series, results/cloudlab-gpu-figures/SERIES, so that no series' figures are
taken for another's (the first Wikipedia runs', 4to8, before the fixes of 2026-11-04). A
series is a directory of the runs with rep1/ and rep2/ (lazy, copy-then-flip) and 4to8to4/, as
run_one.sh wrote them; a protocol may have run in one repetition only, not the same one for each
(amazon-4to8-v4: rep1/copy-then-flip and rep2/lazy), and the figures then take the repetitions
that ran. The calibrations (calibrate-4, calibrate-8) can come from another series, as for a
rerun after a change. --fetch-bound-mb is the fetch cache's bound the runs had (64 before
2026-11-04, 2048 since). Every number a figure's notes cite is computed from the runs. Writes,
into --out:

  4to8.svg, 4to8-rep1.svg, 4to8-rep2.svg, 4to8to4.svg
                    compare_runs.py: lazy against copy-then-flip (the repetitions pooled, then
                    each one that ran both), and 4 -> 8 -> 4 under lazy; a .csv and a .json beside
                    each. Every compare_runs.py timeline is drawn per --window seconds (WINDOW,
                    5 s): its own default, 0.5 s, draws minutes of load as noise
  4to8-stream-first.svg
                    with rep1/ and rep2/ copy-then-flip-stream-first: lazy, copy-then-flip, and
                    copy-then-flip whose stream the queries yield to (--peer-reads-first)
  capacity.svg      the calibrations: answered queries and mean latency against the offered
                    rate, on 4 and on 8 nodes
  lazy-fetch.svg    lazy from the trigger until the new nodes are warm: answered queries, the
                    new nodes' fetch cache against its bound, the time per fetch, their disk writes
  lazy-stall.svg    when lazy flips after the load's end (Amazon 300M): the new nodes' staging up
                    to the flip, and the answers that stop while the initial nodes reclaim
  node-reads.svg    the reads per second charged to each node's budget, lazy and copy-then-flip
                    (and copy-then-flip with the stream first, when its runs are there)
  ctf-stream.svg    copy-then-flip's raw-vector stream from each initial node, under the load (on
                    one budget; with the stream first, when its runs are there) and without it
                    (calibrate-8 scales out before its load)
  ctf-budget.svg    with the stream-first runs: each initial node's reads while copy-then-flip
                    copies, its stream's against its queries', on one budget and the stream first

  lazy-before-after.svg
                    with --before: lazy in that series against lazy in this one (the repetitions
                    pooled), for a change that only lazy's path sees
  4to8to4-before-after.svg
                    with --before, when both series' 4to8to4 runs have the same load and
                    reconfigurations: 4 -> 8 -> 4 in that series against this one
  lazy-staging.svg  with --before: lazy's rescale step by step in that series and in this one
                    (the first repetition of each that ran lazy): the answers, each new node's
                    lists, PQ codes and partition loads up to the flip and its graph up to its
                    entry, the initial nodes' saving of the moved partitions as files and their
                    reclaim
  lazy-entries.svg  with --before, when this series' new entries come online while lazy's backlog
                    drains: each group of entries' answers (the initial ones', the new ones'),
                    the backlog in both series, and the latency of the queries sent from the
                    first new entry until the backlog cleared, as a distribution
  lazy-keepup.svg   lazy through the whole load, every repetition: answered against offered, the
                    backlog, the latency against 8 warm nodes', and the raw vectors fetched against
                    what copy-then-flip copies (calibrate-8)
  keepup-all.svg    lazy and copy-then-flip's two baselines until every query is answered, as
                    compare_runs.py draws them (per --window, the latency on a log scale, no title:
                    a paper figure); with --lazy-series, lazy's runs from a series with a longer load
  NAME-drained.svg  with --drained: each compare_runs.py timeline again, past the load's end until
                    every query is answered (compare_runs.py --until answer), for a series whose
                    backlogs outlast the load

and a PNG of each when rsvg-convert is installed. Runs that are missing are skipped.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from compare_runs import FIXED_SLOT, PALETTE, tick_label, ticks_to  # noqa: E402
from plot_run import answered_at, esc, flip_time, load_end, nice_ticks, read_latency, windows  # noqa: E402

LAZY, CTF = FIXED_SLOT["lazy"], FIXED_SLOT["copy-then-flip"]
# Two highlighted series that are not protocols (cluster sizes; the two hot new nodes): palette
# slots 2 and 6, green and purple. They stay apart for every colour-vision type in both modes
# (OKLab dE x 100: 35.8 / 26.6 at worst in light, 24.6 / 9.4 in dark), where four nodes in
# four slots would not (green against pink: 1.6 in dark); the other nodes are context, in grey.
HOT_A, HOT_B = 2, 6
# Seconds per point of the compare_runs.py timelines and keepup-all.svg (--window). compare_runs.py's
# own default, 0.5 s, suits the end-to-end tests' seconds; minutes of load at hundreds of queries a
# second it draws as noise.
WINDOW = 5.0
CTX = ("#bdbcb4", "#6e6d66")      # context lines: apart from both highlights (dE >= 18 / 15)
# copy-then-flip whose stream goes first: the slot compare_runs.py gives its label in
# 4to8-stream-first.svg, the first one free after lazy's and copy-then-flip's.
SF = 2
# A neutral line beside copy-then-flip's orange and the stream-first green (the queries; no load):
# the surfaces' secondary ink, apart from both in lightness (OKLab dE x 100 at the worst colour-
# vision type: 18.3 and 23.5 in light, 20.6 and 15.7 in dark). A mid grey sits at the green's
# lightness: 3.7 under deuteranopia.
NEUTRAL = ("#52514e", "#c3c2b7")
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


def reps(runs, runset, protocol, need="latency.csv"):
    """The repetitions of a protocol that the series ran (rep1/, rep2/, those with need), in order."""
    rs = [os.path.join(runs, runset, r, protocol) for r in ("rep1", "rep2")]
    return [r for r in rs if os.path.exists(os.path.join(r, need))]


def lead_run(runs, runset):
    """The run whose experiment.json states the series' trigger and load (every protocol's file has
    the same): its first lazy run, else its first copy-then-flip run."""
    rs = reps(runs, runset, "lazy", "experiment.json") + reps(runs, runset, "copy-then-flip", "experiment.json")
    return rs[0] if rs else os.path.join(runs, runset, "rep1", "lazy")


def read_budget(run):
    """The data nodes' read budget a run had (NodeResources.ReadIOPSPerNode, reads/s), 0 for none:
    no experiment has set one since 2026-11-12, and the nodes then count their reads unlimited."""
    res = load_json(os.path.join(run, "experiment.json")).get("NodeResources") or {}
    return int(res.get("ReadIOPSPerNode") or 0)


def rep_name(run):
    """"repetition 2" for .../rep2/lazy."""
    r = os.path.basename(os.path.dirname(run))
    return "repetition " + r[3:] if r.startswith("rep") else r


def runs_count(n):
    return "%d run%s" % (n, "" if n == 1 else "s")


def same_schedule(a, b):
    """Whether two runs had the same load and reconfigurations (experiment.json), so that their
    timelines compare window by window."""
    try:
        ea, eb = (load_json(os.path.join(r, "experiment.json")) for r in (a, b))
    except OSError:
        return False
    return all(os.path.exists(os.path.join(r, "latency.csv")) for r in (a, b)) and \
        ea.get("Load") == eb.get("Load") and ea.get("Reconfigurations") == eb.get("Reconfigurations")


def metrics_series(run, types):
    """The agents' metrics (metrics.jsonl) of the given types on the load's clock: {(node, type):
    [(t, value)]}. The controller stamps each report as it arrives ("ts", its wall clock), and its
    own reconfig.start at the first trigger, whose time on the load's clock reconfigurations.json
    has: the offset between the two clocks."""
    import datetime

    def wall(s):
        day, _, frac = s.rstrip("Z").partition(".")
        return (datetime.datetime.fromisoformat(day).replace(tzinfo=datetime.timezone.utc).timestamp()
                + float("0." + (frac or "0")))
    recs = jsonl(os.path.join(run, "metrics.jsonl"))
    start = next(r for r in recs if r["type"] == "reconfig.start")
    base = wall(start["ts"]) - events(run)[0]
    out = {}
    for r in recs:
        if r["type"] in types:
            out.setdefault((r["node"].split(":")[0], r["type"]), []).append((wall(r["ts"]) - base, r["value"]))
    return out


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


def plan_moves(run):
    """Which initial node handed which new node partitions in the run's first reconfiguration:
    ({new node: [initial nodes]}, {initial node: [new nodes]}), by name. The initial placement is
    contiguous ranges of partitions (U3); the plan names the target's owners by node ID, and node
    i is n<i> (the runners start them in order)."""
    initial = int(load_json(os.path.join(run, "experiment.json")).get("InitialNodes", 1))
    owners = events(run)[3]["reply"]["plan"]["owners"]
    parts = sum(len(v) for v in owners.values())
    src_of, dst_of = {}, {}
    for node, ps in sorted(owners.items(), key=lambda kv: int(kv[0])):
        if int(node) <= initial:
            continue
        for q in ps:
            old, new = "n%d" % (q * initial // parts + 1), "n%s" % node
            src_of.setdefault(new, []) if old in src_of.get(new, []) else src_of.setdefault(new, []).append(old)
            dst_of.setdefault(old, []) if new in dst_of.get(old, []) else dst_of.setdefault(old, []).append(new)
    return src_of, dst_of


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
             ".neu { stroke: %s; } .fneu { fill: %s; }" % (NEUTRAL[m], NEUTRAL[m])]
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
  .dash { stroke-dasharray: 6 4; }
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

def dataset(c):
    """The dataset an experiment ran on, by its work directory (wiki-c1024, amazon-300m)."""
    w = os.path.basename(str(c.get("WorkDir", "")).rstrip("/"))
    if w.startswith("wiki"):
        return "Wikipedia"
    if w.startswith("amazon-"):
        return "Amazon " + w.split("-", 1)[1].upper()
    return w or "rtier"


def schedule(run, protocol=False):
    """A run's reconfigurations and load as a title states them, from its experiment.json:
    "Amazon 300M 4 → 8 at 90 s, load 350 → 700 q/s at 60 s, CloudLab c240g5"; with protocol,
    "... 4 → 8 at 90 s → 4 at 300 s under lazy; load 350 → 700 q/s at 60 s → 350 at 280 s, ..."."""
    try:
        c = load_json(os.path.join(run, "experiment.json"))
        load, recs = c["Load"], c.get("Reconfigurations") or []
        sizes = "%d" % c.get("InitialNodes", 1) + "".join(
            " → %d at %s s" % (r["DataNodes"], r["TriggerTimeSeconds"]) for r in recs if "DataNodes" in r)
        steps = [s.split(":") for s in str(load.get("RateSteps", "")).split(",") if s]
        rates = ("%s → %s q/s at %s s" % (load["Rate"], steps[0][1], steps[0][0]) +
                 "".join(" → %s at %s s" % (r, t) for t, r in steps[1:])) if steps else "%s q/s" % load["Rate"]
        under = " under %s" % c.get("Protocol", "lazy") if protocol else ""
        return "%s %s%s%s load %s, CloudLab c240g5" % (dataset(c), sizes, under, ";" if protocol else ",", rates)
    except (OSError, KeyError, IndexError, ValueError, TypeError):
        return "4 → 8, CloudLab c240g5"


def setting(runs, runset):
    """The 4 -> 8 runs' trigger and load as the titles state them, from the series' lead run."""
    return schedule(lead_run(runs, runset))


def compare_specs(runs, runset, before=None):
    """compare_runs.py figures: (name, title, run dirs under --runs, labels, slots[, more of its
    options]). The pooled figures take the repetitions that ran."""
    tag = "" if runset == "4to8" else " [%s]" % runset
    head = setting(runs, runset)
    rel = lambda rs: [os.path.relpath(r, runs) for r in rs]  # noqa: E731
    lazy, ctf = rel(reps(runs, runset, "lazy")), rel(reps(runs, runset, "copy-then-flip"))
    per = ("%s per protocol" % runs_count(len(lazy)) if len(lazy) == len(ctf)
           else "lazy %s, copy-then-flip %s" % (runs_count(len(lazy)), runs_count(len(ctf))))
    specs = [("4to8", head + " (%s)" % per + tag, lazy + ctf, None, None)] if lazy and ctf else []
    specs += [
        ("4to8-rep1", head + " (repetition 1)" + tag,
         ["%s/rep1/lazy" % runset, "%s/rep1/copy-then-flip" % runset], None, None),
        ("4to8-rep2", head + " (repetition 2)" + tag,
         ["%s/rep2/lazy" % runset, "%s/rep2/copy-then-flip" % runset], None, None),
        ("4to8to4", schedule(os.path.join(runs, runset, "4to8to4"), protocol=True) + tag,
         ["%s/4to8to4" % runset], "lazy", None),
        # copy-then-flip whose stream the queries yield to (--peer-reads-first), next to the two.
        ("4to8-stream-first", head + " (2 runs per line)" + tag,
         ["%s/rep%d/%s" % (runset, r, p) for p in ("lazy", "copy-then-flip", "copy-then-flip-stream-first")
          for r in (1, 2)],
         ",".join(["lazy"] * 2 + ["copy-then-flip"] * 2 + ["ctf stream-first"] * 2), None),
    ]
    if before:
        # Both lines are lazy: lazy's own colour for this series, a validated neighbour of it (the
        # palette's third slot) for the one before -- not copy-then-flip's orange. The labels stay
        # short enough for the legend (about 16 characters); the title names the two series.
        old = rel(reps(runs, before, "lazy"))
        if old and lazy:
            counts = ("%s each" % runs_count(len(old)) if len(old) == len(lazy)
                      else "%s and %s" % (runs_count(len(old)), runs_count(len(lazy))))
            specs.append(("lazy-before-after", head + ": lazy before (%s) and after (%s), %s" % (before, runset, counts),
                          old + lazy,
                          ",".join(["lazy (before)"] * len(old) + ["lazy (after)"] * len(lazy)),  # no comma in a label
                          "%d,%d" % (HOT_A, LAZY)))
        a, b = os.path.join(runs, before, "4to8to4"), os.path.join(runs, runset, "4to8to4")
        if same_schedule(a, b):  # a scale-in that the change moved, say
            # on a log scale: the scale-in's milliseconds beside the backlog's seconds
            specs.append(("4to8to4-before-after", schedule(b, protocol=True) + ": before (%s) and after (%s)"
                          % (before, runset), rel([a, b]), "lazy (before),lazy (after)", "%d,%d" % (HOT_A, LAZY),
                          ["--log-latency"]))
    return specs


def timelines(runs, out, specs, until="load", window=WINDOW):
    """compare_runs.py's timelines, per window seconds. until "answer": past the load's end until
    every query is answered, the backlogs drained, as NAME-drained.svg beside the ones cut at the
    load's end."""
    for name, title, dirs, labels, slots, *more in specs:
        paths = [os.path.join(runs, d) for d in dirs]
        if not all(os.path.exists(os.path.join(p, "latency.csv")) for p in paths):
            print("skipped", name, "(runs missing)")
            continue
        name += "-drained" if until == "answer" else ""
        cmd = [sys.executable, os.path.join(os.path.dirname(HERE), "compare_runs.py")] + paths + \
              ["--out", os.path.join(out, name), "--title", title, "--until", until, "--window", "%g" % window]
        if labels:
            cmd += ["--labels", labels]
        if slots:
            cmd += ["--slots", slots]
        subprocess.run(cmd + (more[0] if more else []), check=True, stdout=subprocess.DEVNULL)
        print("wrote", os.path.join(out, name + ".svg"))


def capacity(runs, out, runset, calibrations):
    cals = [("4 nodes", HOT_A, os.path.join(runs, calibrations, "calibrate-4")),
            ("8 nodes", HOT_B, os.path.join(runs, calibrations, "calibrate-8"))]
    cals = [(l, s, load_json(os.path.join(d, "calibration.json"))) for l, s, d in cals
            if os.path.exists(os.path.join(d, "calibration.json"))]
    if len(cals) < 2:
        print("skipped capacity.svg (calibrations missing)")
        return
    exp = os.path.join(lead_run(runs, runset), "experiment.json")
    if not os.path.exists(exp):
        print("skipped capacity.svg (no %s)" % exp)
        return
    rate = load_json(exp)["Load"]
    lo_rate, hi_rate = float(rate["Rate"]), float(rate["RateSteps"].split(",")[0].split(":")[1])
    svg = Svg()
    pw = W - LEFT - RIGHT
    top = max(s["offered_per_s"] for _, _, st in cals for s in st)
    xr = (0, 1700 if top <= 1700 else ticks_to(top)[-1])  # every step on the axis, its dot and label too
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
    shares = [read_shares(os.path.join(runs, calibrations, c)) for c in ("calibrate-4", "calibrate-8")]
    budget = read_budget(os.path.join(runs, calibrations, "calibrate-4"))
    text = "The plateau is the mean of the last three steps. "
    if all(shares) and budget:
        text += ("A query re-ranks %.0f candidates in %.0f page reads. The busiest node's budget of %s reads/s bounds "
                 "every step (node_reads.py): it takes %.0f%% of the reads on 4 nodes and %.0f%% on 8, %.2f and %.2f "
                 "times an even share, which predicts %.0f and %.0f q/s. "
                 % (shares[0]["cands"], shares[0]["pages"], "{:,}".format(budget), 100 * shares[0]["top"],
                    100 * shares[1]["top"], 4 * shares[0]["top"], 8 * shares[1]["top"], shares[0]["predicted"],
                    shares[1]["predicted"]))
    elif all(shares):  # no read limit: what bounds a step is the busiest node's device, or more
        text += ("A query re-ranks %.0f candidates in %.0f page reads. The busiest node takes %.0f%% of the reads on 4 "
                 "nodes and %.0f%% on 8, %.2f and %.2f times an even share (node_reads.py). "
                 % (shares[0]["cands"], shares[0]["pages"], 100 * shares[0]["top"], 100 * shares[1]["top"],
                    4 * shares[0]["top"], 8 * shares[1]["top"]))
    text += ("8 nodes carry %.2f times what 4 carry. The runs offer %.0f q/s, then %.0f: above 4 nodes' capacity, "
             "below 8 nodes'." % (plateaus[1] / plateaus[0], lo_rate, hi_rate))
    notes = textwrap.wrap(text, 100)
    for i, n in enumerate(notes):
        svg.text(LEFT, p2.y0 + p2.ph + 62 + 15 * i, n, "note")
    load = load_json(os.path.join(runs, calibrations, "calibrate-4", "experiment.json")).get("Load", {})
    return svg.save(out, "capacity.svg", p2.y0 + p2.ph + 62 + 15 * len(notes) + 10,
                    "Capacity on CloudLab c240g5: 4 and 8 nodes, %s"
                    % ("%s reads/s per node" % "{:,}".format(budget) if budget else "no read limit"),
                    "the calibrations: 30 s per offered rate, the same queries (NProbe %s, N %s, k %s); "
                    "8 nodes after copy-then-flip moved every raw vector"
                    % (load.get("NProbe", "?"), load.get("N", "?"), load.get("K", "?")))


def read_shares(run):
    """A calibration's page reads per node, from the first re-ranking on (before it, calibrate-8's
    old nodes read only for its copy, which the budget charges too): the candidates and page reads
    per query, the busiest node's share, and the capacity its budget predicts. None without the
    stats."""
    try:
        steps = load_json(os.path.join(run, "calibration.json"))
        queries = sum(st["queries"] for st in steps)
        st = {}
        for i in range(1, 65):
            path = os.path.join(run, "stats-n%d.jsonl" % i)
            if not os.path.exists(path):
                break
            st[i] = [(r["t"], r["reads"]["charged"], r["fetch"]["rerank_disk"], r["reads"]["budget"])
                     for r in jsonl(path) if "reads" in r]
        t0 = min(next((x[0] for x in v if x[2] > v[0][2]), float("inf")) for v in st.values())

        def before(v):
            return [x for x in v if x[0] < t0][-1] if v[0][0] < t0 else (t0, 0, 0, 0)
        pages = {n: v[-1][1] - before(v)[1] for n, v in st.items()}
        cands = sum(v[-1][2] - before(v)[2] for v in st.values())
        total, top = sum(pages.values()), max(pages.values())
        budget = st[1][-1][3]
        return {"cands": cands / queries, "pages": total / queries, "top": top / total, "budget": budget,
                "predicted": budget * queries / top}
    except (OSError, ValueError, KeyError, IndexError, StopIteration, ZeroDivisionError):
        return None


def lazy_fetch(runs, out, runset, calibrations, bound_mb, lo=80.0, hi=None):
    rs = reps(runs, runset, "lazy", "stats-n5.jsonl")
    if not rs:
        print("skipped lazy-fetch.svg (run missing)")
        return
    run = rs[0]
    rel = os.path.relpath(run, runs)
    t0 = load_start(run)
    trig, flip, entries, _ = events(run)
    new = ["n5", "n6", "n7", "n8"]
    st = {n: jsonl(os.path.join(run, "stats-%s.jsonl" % n)) for n in new}
    end = load_end(run) or 360.0
    rows = read_latency(os.path.join(run, "latency.csv"))
    if hi is None and flip >= end:
        # A flip after the load's end (the new nodes' lists and PQ codes took that long to stage):
        # until the last answer, the backlog's.
        hi = max(answered_at(r) for r in rows) + 5.0
    elif hi is None:
        # Until the new nodes' fetch caches have emptied, and 20 s more: about 200 s with the
        # Wikipedia queries. Without locality (U11's worst case) they do not empty during the load.
        busy = [r["t"] - t0 for n in new for r in st[n] if r["fetch"]["queued"] and r["t"] - t0 <= end]
        hi = min(end, max(200.0, (max(busy) if busy else flip) + 20.0))
    # The two new nodes that fetched the most, in node order, highlighted; the others context.
    fetched = {n: st[n][-1]["fetch"]["fetched"] for n in new}
    hot = sorted(sorted(new, key=lambda n: -fetched[n])[:2], key=lambda n: int(n[1:]))
    src_of, _ = plan_moves(run)
    named = lambda n: "%s (from %s)" % (n, ", ".join(src_of.get(n, [])) or "?")  # noqa: E731
    cal4 = os.path.join(runs, calibrations, "calibrate-4", "calibration.json")
    c4 = sum(s["answered_in_window_per_s"] for s in load_json(cal4)[-3:]) / 3 if os.path.exists(cal4) else None
    exp = load_json(os.path.join(run, "experiment.json"))["Load"]
    offered = float(exp["RateSteps"].split(",")[0].split(":")[1])
    win = {round(w["t"]): w["answered"] for w in windows(rows, 1.0)}
    answered = [(t + 0.5, win.get(t)) for t in range(int(lo), int(hi))]
    cls = {n: "ctx thin" for n in new}
    cls.update({hot[0]: "c%d" % HOT_A, hot[1]: "c%d" % HOT_B})
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
    for n in sorted({o for h in hot for o in src_of.get(h, [])}, key=lambda n: int(n[1:])):
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
    shown = sorted(e for e in entries if e <= hi)  # later ones: after the window
    for p in (p1, p2, p3, p4):
        p.vline(trig, "trigger" if p is p1 else None, "end")
        if end < hi:
            p.vline(end, "load ends" if p is p1 else None, "end")
        # entries right after the flip take the right of its line: the flip's label goes left
        p.vline(flip, "flip" if p is p1 else None, "end" if shown and p.x(shown[0]) - p.x(flip) < 60 else "start")
        for i, e in enumerate(shown):
            p.vline(e, ("new entries" if i == 0 else None) if p is p1 else None)
    if end < hi:  # offered until the load's end only
        p1.line([(lo, offered), (end, offered)], "cu sched")
        svg.text(p1.x(lo) + 6, p1.y(offered) - 5, "offered %.0f q/s" % offered, "note")
    else:
        p1.hline(offered, "offered %.0f q/s" % offered, cls="cu sched", anchor="start", dy=-5)
    if c4:  # labelled at the right end, where the line runs above it
        p1.hline(c4, "4 nodes' capacity, %.0f q/s" % c4, dy=12)
    p1.line(answered, "c%d" % LAZY)
    size = "%d GB" % (bound_mb // 1024) if bound_mb >= 1024 else "%d MB" % bound_mb
    if show_bound:
        p2.hline(bound, "the bound, %s: past it the query that fetched writes its vectors itself" % size, dy=-5)
    for n in [n for n in new if n not in hot] + hot:  # context first, highlights over it
        p2.line(queued[n], cls[n])
        p3.line(per_call[n], cls[n])
        p4.line(writes[n], cls[n])
    time_axis(svg, p4, p4.y0 + p4.ph, lo, hi)
    svg.legend(W - RIGHT + 16, p1.y0 + 10, [("lazy", "c%d" % LAZY, "line"), ("offered", "cu sched", "line")])
    svg.legend(W - RIGHT + 16, p2.y0 + 10, [(named(hot[0]), cls[hot[0]], "line"), (named(hot[1]), cls[hot[1]], "line"),
                                            (", ".join(n for n in new if n not in hot), "ctx", "line")])

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
    a, b = hot
    text = ("%s and %s fetched the most. Fetch cache at its fullest: %s %.2f GB, %s %.2f GB, against a bound of %s; "
            "vectors the fetching queries had to install themselves: %s %d, %s %d. Mean fetch call: %s %.0f ms, %s %.0f "
            "ms. A fetch also waits for its holder's read budget: in the 30 s after the flip its readers waited %s "
            "(summed over the readers)."
            % (named(a), named(b), a, gb(peak[a]), b, gb(peak[b]), size, a, last[a]["sync_installed"], b,
               last[b]["sync_installed"], a, calls(a), b, calls(b), waits))
    if flip >= end:
        text += (" The flip comes %.0f s after the trigger, after the load's end: lazy-stall.svg shows where that "
                 "time goes, and why the answers stop after the flip." % (flip - trig))
    notes = textwrap.wrap(text, 108)
    for i, n in enumerate(notes):
        svg.text(LEFT, p4.y0 + p4.ph + 54 + 15 * i, n, "note")
    return svg.save(out, "lazy-fetch.svg", p4.y0 + p4.ph + 54 + 15 * len(notes) + 10,
                    "Lazy right after the flip: the new nodes' fetch cache, their fetches and their disks", sub)


def lazy_stall(runs, out, runset, step=5.0):
    """Lazy when its flip comes after the load's end: where the time from the trigger to the flip
    goes (the new nodes' staging), and the answers that stop after the flip, while the initial
    nodes drop the partitions that moved (the reclaim). Drawn only for such a series."""
    rs = [os.path.join(runs, runset, "rep%d" % r, "lazy") for r in (1, 2)]
    rs = [r for r in rs if os.path.exists(os.path.join(r, "stats-n5.jsonl"))]
    if not rs:
        print("skipped lazy-stall.svg (runs missing)")
        return None
    run = rs[0]
    t0, end = load_start(run), load_end(run) or 360.0
    per = []
    for r in rs:
        trig, flip, _, rc = events(r)
        ph = rc["reply"]["phases"]
        rows = read_latency(os.path.join(r, "latency.csv"))
        per.append({"trig": trig, "flip": flip, "stage": ph.get("stage_data", 0.0), "reclaim": ph.get("reclaim", 0.0),
                    "reclaimed": flip + ph.get("drain", 0.0) + ph.get("reclaim", 0.0), "rows": rows,
                    "failed": sum(1 for x in rows if x["status"] != "ok"),
                    "last": max(answered_at(x) for x in rows)})
    if min(q["flip"] for q in per) < end:
        print("skipped lazy-stall.svg (lazy flips during the load)")
        return None
    first = per[0]
    lo, hi = first["trig"] - 20.0, max(q["last"] for q in per) + 5.0
    new, old = ["n5", "n6", "n7", "n8"], ["n1", "n2", "n3", "n4"]
    st = {n: jsonl(os.path.join(run, "stats-%s.jsonl" % n)) for n in new + old}

    def held(ps, t):  # a cumulative counter: 0 before its first sample, its last value after the last
        return 0 if not ps or t < ps[0][0] else ps[-1][1] if t > ps[-1][0] else value_at(ps, t)

    def summed(nodes, path):
        pts = [series(st[n], path, t0) for n in nodes]
        times = sorted({t for ps in pts for t, _ in ps if lo - step <= t <= hi + step})
        return [(t, sum(held(ps, t) for ps in pts)) for t in times]

    def share(nodes, path):  # percent of its final value
        pts = summed(nodes, path)
        top = max((v for _, v in pts), default=0) or 1
        return [(t, 100.0 * v / top) for t, v in pts]

    answered = [[(w["t"] + step / 2, w["answered"]) for w in windows(q["rows"], step) if lo <= w["t"] <= hi] for q in per]
    exp = load_json(os.path.join(run, "experiment.json"))["Load"]
    steps = [(float(a), float(b)) for a, b in (x.split(":") for x in str(exp.get("RateSteps", "")).split(",") if x)]

    def offered_at(t):
        r = float(exp["Rate"])
        for a, b in steps:
            r = b if t >= a else r
        return r if t < end else 0.0
    sched = []
    for t in sorted({lo, hi, end} | {a for a, _ in steps if lo < a < hi}):
        if sched:
            sched.append((t, sched[-1][1]))  # the step, drawn upright
        sched.append((t, offered_at(t)))
    pq, parts = share(new, "ops.pq_put.count"), share(new, "ops.load_partition.count")
    rr_old = rates(summed(old, "ops.rerank.count"), lo, hi, step)
    rr_new = rates(summed(new, "ops.rerank.count"), lo, hi, step)

    svg = Svg()
    pw = W - LEFT - RIGHT
    xr = (lo, hi)
    ys = [84, 264, 424]
    p1 = Panel(svg, LEFT, ys[0], pw, 130, xr, "answered queries per second (%g s windows), both runs" % step, "q/s",
               ymax=max(max(v for _, v in sched), max(v or 0 for a in answered for _, v in a)) * 1.05)
    p2 = Panel(svg, LEFT, ys[1], pw, 100, xr, "the new nodes' staging (repetition 1): what has arrived and loaded", "%",
               ymax=100.0)
    p3 = Panel(svg, LEFT, ys[2], pw, 100, xr, "RERANKs per second on the initial and on the new nodes (repetition 1)",
               "/s", ymax=max(v or 0 for _, v in rr_old + rr_new) * 1.05)
    for p in (p1, p2, p3):
        p.vline(first["trig"], "trigger" if p is p1 else None, "end")
        p.vline(end, "load ends" if p is p1 else None, "end")
        p.vline(first["flip"], "flip" if p is p1 else None)
        p.vline(first["reclaimed"], "reclaim done" if p is p1 else None, "end")
    p1.line(sched, "cu sched")
    for a in answered:
        p1.line(a, "c%d" % LAZY)
    p2.line(pq, "c%d" % HOT_A)
    p2.line(parts, "c%d" % HOT_B)
    p3.line(rr_old, "c%d" % HOT_A)
    p3.line(rr_new, "c%d" % HOT_B)
    time_axis(svg, p3, ys[2] + 100, lo, hi)
    svg.legend(W - RIGHT + 16, ys[0] + 10, [("lazy", "c%d" % LAZY, "line"), ("(%s)" % runs_count(len(per)), "", "none"),
                                            ("offered", "cu sched", "line")])
    svg.legend(W - RIGHT + 16, ys[1] + 10, [("PQ codes", "c%d" % HOT_A, "line"), ("installed", "", "none"),
                                            ("partitions", "c%d" % HOT_B, "line"), ("loaded", "", "none")])
    svg.legend(W - RIGHT + 16, ys[2] + 10, [("n1-n4", "c%d" % HOT_A, "line"), ("n5-n8", "c%d" % HOT_B, "line")])

    def mean_rate(q, a, b):
        return sum(1 for x in q["rows"] if x["status"] == "ok" and a <= answered_at(x) < b) / max(b - a, 1e-9)
    before = [mean_rate(q, q["flip"] - 40, q["flip"] - 5) for q in per]
    during = [mean_rate(q, q["flip"] + 5, q["reclaimed"] - 5) for q in per]
    loads = [n for n in new if st[n][-1]["ops"]["load_partition"]["count"]]
    each = (sum(st[n][-1]["ops"]["load_partition"]["busy_us"] for n in loads) /
            max(sum(st[n][-1]["ops"]["load_partition"]["count"] for n in loads), 1) / 1e6)
    rng = lambda key, f="%.0f": span_of([q[key] for q in per], f)  # noqa: E731
    text = ("Lazy flips %s s after the trigger, after the load's end: staging the new nodes' partitions takes %s s "
            "(stage_data). Their lists and PQ codes arrive first; then each new node loads its partitions one at a "
            "time, %.0f s each. After the flip the initial nodes drop the partitions that moved (the reclaim, %s s). "
            "Meanwhile the cluster answers %s q/s, against %s before the flip, and %s queries fail: the client gives "
            "up on a query after 30 s. Loading a partition and dropping it take and release one PQ reference per "
            "posting under the PQ store's mutex, a whole partition at a time, and every RERANK takes the same mutex "
            "to locate its candidates (PQStore::Ref, Unref, Locate): the reranks stop on every node while the "
            "reclaim runs."
            % (span_of([q["flip"] - q["trig"] for q in per]), rng("stage"), each, rng("reclaim"), span_of(during),
               span_of(before), "/".join("%d" % q["failed"] for q in per)))
    notes = textwrap.wrap(text, 108)
    yb = ys[2] + 100 + 62
    for i, line in enumerate(notes):
        svg.text(LEFT, yb + 15 * i, line, "note")
    return svg.save(out, "lazy-stall.svg", yb + 15 * len(notes) + 6,
                    "Lazy at 300M vectors: a flip after the load, then answers stop while the old owners reclaim",
                    "%s; from the trigger until every query is answered" % schedule(run))


def counter_rises(st, op, key, t0, lo, hi):
    """When one of a node's op counters rose within [lo, hi] (seconds of the load): [(the sample
    before, the sample it rose at)]. A call's count rises when it starts, its busy time when it ends."""
    pts = [(r["t"] - t0, r["ops"][op][key]) for r in st if op in r.get("ops", {})]
    return [(ta, tb) for (ta, va), (tb, vb) in zip(pts, pts[1:]) if vb > va and tb >= lo and ta <= hi]


def staging_bars(run):
    """A lazy rescale's steps from its nodes' counters (1 s samples), in seconds since the trigger:
    per new node its phases [(phase, start, end)], its partition loads (each one's end, the busy
    time per load) and its graph (from the bulk of its chunks to the install); the initial nodes' phases
    (the moved partitions saved as files, before 2026-11-07, and the reclaim); the flip, each new
    entry, the load's end."""
    t0 = load_start(run)
    trig, flip, _, rc = events(run)
    reply, ph = rc["reply"], rc["reply"]["phases"]
    lo, hi = trig - 1.0, trig + rc["elapsed_seconds"] + 5.0
    pre = ph.get("plan", 0.0) + ph.get("stage_aggregator", 0.0)
    saved = pre + ph.get("save_segments", 0.0)
    nodes = {}
    for n in ["n%d" % a for a in reply.get("added", [])]:
        st = jsonl(os.path.join(run, "stats-%s.jsonl" % n))

        def rose(op, key="count"):
            return [(a - trig, b - trig) for a, b in counter_rises(st, op, key, t0, lo, hi)]
        bars = []
        seg, pqm, pqp = rose("segment_put"), rose("pq_missing"), rose("pq_put")
        if seg:  # straight from the old owners' memory (since 2026-11-07)
            bars.append(("lists", seg[0][0], seg[-1][1]))
        elif pqm and ph.get("save_segments"):  # files the old owners saved first: their copy
            bars.append(("lists", saved, pqm[0][0]))
        if pqm and pqp:
            bars.append(("PQ codes", pqm[0][0], pqp[-1][1]))
        op = "segment_install" if rose("segment_install") else "load_partition"
        starts, ends = rose(op), rose(op, "busy_us")
        if starts and ends:
            bars.append(("loads", starts[0][0], ends[-1][1]))
        last = st[-1]["ops"].get(op, {})
        # how many loads started together: 1 when they went one at a time
        counts = [r["ops"][op]["count"] for r in st if op in r.get("ops", {}) and lo <= r["t"] - t0 <= hi]
        at_once = max((b - a for a, b in zip(counts, counts[1:])), default=0)
        gp, gi = rose("graph_put"), rose("graph_install")
        # The graph starts where the bulk of its chunks does: its first chunk or few can leave as
        # the rescale starts, before the yield sees the staging, and then wait with the rest (n5 and
        # n6 in amazon-4to8-v3 and v4, 1-5 chunks of about 3,100).
        puts = [(r["t"] - t0 - trig, r["ops"]["graph_put"]["count"]) for r in st
                if "graph_put" in r.get("ops", {}) and lo <= r["t"] - t0 <= hi]
        first_n = puts[0][1] if puts else 0
        total = puts[-1][1] - first_n if puts else 0
        bulk = next((ta for (ta, _), (_, vb) in zip(puts, puts[1:]) if total and vb - first_n >= 0.02 * total), None)
        # The graph ends at its install; the run's last sample can come before the last entry's
        # (the runner stops when the rescale returns), and its entry flip then marks the end.
        entry = next((e["seconds"] for e in reply.get("entry_flips", []) if int(n[1:]) in e["nodes"]), None)
        done = gi[0][1] if gi else entry
        nodes[n] = dict(bars=bars, ends=[b for _, b in ends], op=op, loads=last.get("count", 0), at_once=at_once,
                        per_load=last.get("busy_us", 0) / 1e6 / max(last.get("count", 0), 1),
                        graph=(bulk if bulk is not None else gp[0][0], done) if gp and done is not None else None)
    old = [("lists", pre, saved)] if ph.get("save_segments") else []
    reclaim = flip - trig + ph.get("drain", 0.0)
    old.append(("reclaim", reclaim, reclaim + ph.get("reclaim", 0.0)))
    entries = {"n%d" % n: e["seconds"] for e in reply.get("entry_flips", []) for n in e["nodes"]}
    return dict(nodes=nodes, old=old, flip=flip - trig, entries=entries, end=(load_end(run) or 360.0) - trig,
                elapsed=rc["elapsed_seconds"], trig=trig, phases=ph)


def lazy_staging(runs, out, runset, before, step=5.0):
    """With --before: where lazy's rescale spends its time, in that series and in this one (the
    first repetition of each that ran lazy), on one axis of seconds since the trigger. Per series,
    the answers, then per new node its posting lists, PQ codes and partition loads up to the flip,
    and its graph up to its entry; the initial nodes' saving of the moved partitions as files
    (before 2026-11-07) and their reclaim after the flip. For a change to the staging (2026-11-07:
    lists from memory, partitions loaded 8 at once with a PQ reference per partition and vector)
    or to when the graph moves (2026-11-08, U16: after the staging, not after the fetches)."""
    if not before:
        return None
    sets = [("before", before), ("after", runset)]
    paths = [(reps(runs, s, "lazy", "stats-n5.jsonl") or [None])[0] for _, s in sets]
    if None in paths:
        print("skipped lazy-staging.svg (runs missing)")
        return None
    data, lo_ = [], -10.0
    for (name, s), run in zip(sets, paths):
        b = staging_bars(run)
        rows = read_latency(os.path.join(run, "latency.csv"))
        trig, end = b["trig"], b["end"]
        exp = load_json(os.path.join(run, "experiment.json"))["Load"]
        rate, sched = float(exp["Rate"]), []
        steps = sorted((float(a), float(r)) for a, r in (x.split(":") for x in str(exp.get("RateSteps", "")).split(",") if x))
        for t, r in [(0.0, rate)] + steps:  # the offered rate, its steps drawn upright
            if sched:
                sched.append((t - trig, sched[-1][1]))
            sched.append((t - trig, r))
        sched += [(end, sched[-1][1]), (end, 0.0)]
        sched = [(lo_, next(r for t, r in reversed(sched) if t <= lo_))] + [q for q in sched if q[0] > lo_]
        b.update(name=name, series=s, run=run, sched=sched,
                 answered=[(w["t"] + step / 2 - trig, w["answered"]) for w in windows(rows, step)],
                 failed=sum(1 for x in rows if x["status"] != "ok"),
                 last=max(answered_at(x) for x in rows) - trig, sources=plan_moves(run)[0])
        data.append(b)
    lo = lo_
    hi = max(max(b["elapsed"], b["last"], b["old"][-1][2]) for b in data) + 10.0
    left = LEFT + 40  # room for the rows' names, "n5 ← n1, n2"
    pw = W - left - RIGHT
    tp_h, lane, gap, pitch = 70, 12, 3, 26  # the answers' panel; a row's lanes: data, then graph
    names = sorted(data[0]["nodes"], key=lambda n: int(n[1:]))
    block = 34 + tp_h + 26 + pitch * (len(names) + 1)
    # headroom over the answers for the events' labels
    ymax = max(max(v for _, v in b["answered"]) for b in data) * 1.25
    svg = Svg()
    cls = {"lists": "f0", "PQ codes": "f1", "loads": "f2", "reclaim": "fneu"}
    what = {"lists": "posting lists", "PQ codes": "PQ codes", "loads": "partition loads", "reclaim": "reclaim"}

    def bar(x0, x1, y, h, c, tip):
        # 2 px of surface between bars that touch: each gives up 1 px at both ends
        w = max(x1 - x0 - 2, 2.0)
        svg.add('<rect class="%s" x="%.1f" y="%.1f" width="%.1f" height="%d" rx="2"><title>%s</title></rect>'
                % (c, x0 + 1, y, w, h, esc(tip)))

    y = 84
    p = None
    for b in data:
        p = Panel(svg, left, y + 20, pw, tp_h, (lo, hi), "%s: %s%s, %s · answered queries per second, per %g s"
                  % (b["name"], b["series"], revision(b["run"], " (commit ", ")"), rep_name(b["run"]), step), "q/s",
                  ymax=ymax)
        p.line([(t, v) for t, v in b["sched"] if lo <= t <= hi], "ref")
        p.line([(t, v) for t, v in b["answered"] if lo <= t <= hi], "neu")
        gy = y + 20 + tp_h + 22  # the rows
        rows_h = pitch * (len(names) + 1)
        for t, label, anchor in ((0.0, "trigger", "start"), (b["flip"], "flip", "start"), (b["end"], "load ends", "end"),
                                 (b["elapsed"], "rescale ends", "start")):
            x = p.x(t)
            svg.add('<line class="ev" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>' % (x, y + 20, x, gy + rows_h - 6))
            svg.text(x + (3 if anchor == "start" else -3), y + 31, label, "note", anchor)
        for i, n in enumerate(names + ["old"]):
            ry = gy + i * pitch
            if n == "old":
                svg.text(left - 8, ry + 10, "n1–n4", anchor="end")
                for ph_, a, z in b["old"]:
                    tip = ("n1–n4: the moved partitions saved as files (save_segments), %.1f–%.1f s" % (a, z)
                           if ph_ == "lists" else "n1–n4: reclaim of the moved partitions, %.1f–%.1f s after the trigger" % (a, z))
                    bar(p.x(a), p.x(z), ry, lane, cls[ph_], tip)
                continue
            node = b["nodes"][n]
            svg.text(left - 8, ry + 10, "%s ← %s" % (n, ", ".join(b["sources"].get(n, []))), anchor="end")
            for ph_, a, z in node["bars"]:
                tip = "%s: %s, %.1f–%.1f s after the trigger (%.1f s)" % (n, what[ph_], a, z, z - a)
                if ph_ == "loads":
                    tip += "; %d loads (%s), %.1f s each" % (node["loads"], node["op"].upper(), node["per_load"])
                bar(p.x(a), p.x(z), ry, lane, cls[ph_], tip)
                if ph_ == "loads" and node["at_once"] <= 1:  # each load's end, as a cut in the bar
                    for e in node["ends"][:-1]:
                        svg.add('<line class="dot" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>' % (p.x(e), ry, p.x(e), ry + lane))
            if node["graph"]:
                a, z = node["graph"]
                bar(p.x(a), p.x(z), ry + lane + gap, 5, "fctx", "%s: the graph, moving from %.1f s, installed %.1f s "
                    "after the trigger" % (n, a, z))
            if n in b["entries"]:
                e = b["entries"][n]
                svg.add('<circle class="fneu" cx="%.1f" cy="%.1f" r="3.5"><title>%s</title></circle>'
                        % (p.x(e), ry + lane + gap + 2.5, esc("%s is an entry %.1f s after the trigger" % (n, e))))
        y += block
    base = y - block + 20 + tp_h + 22 + pitch * (len(names) + 1) - 6  # under the last rows
    svg.add('<line class="axis" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>' % (left, base, left + pw, base))
    time_axis(svg, p, base, lo, hi, label="seconds since the trigger")
    svg.legend(W - RIGHT + 16, 84 + 20 + tp_h + 32,
               [("posting lists", "f0", "box"), ("PQ codes", "f1", "box"), ("partition loads", "f2", "box"),
                ("graph", "fctx", "box"), ("entry online", "", "none"), ("reclaim", "fneu", "box"),
                ("answered", "neu", "line"), ("offered", "ref", "line")])
    ey = 84 + 20 + tp_h + 32 + 4 * 18  # the entry's dot beside its label
    svg.add('<circle class="fneu" cx="%d" cy="%d" r="3.5"/>' % (W - RIGHT + 16 + 7, ey - 4))

    def phase(b, name, f):  # over the new nodes: f of each one's span of a phase
        return [f(a, z) for nd in b["nodes"].values() for ph_, a, z in nd["bars"] if ph_ == name]
    old, new = data
    rng = lambda vals, f="%.0f": span_of(vals, f) if vals else "–"  # noqa: E731

    def describe(b, name):  # one series' steps, from its counters: the same words for any change
        lists_by = rng(phase(b, "lists", lambda a, z: z))
        save = b["phases"].get("save_segments", 0.0)
        lists = ("the initial nodes first save the moved partitions as files (%.0f s) and the new nodes copy them "
                 "by %s s" % (save, lists_by) if save else "the lists come from the initial nodes' memory by %s s"
                 % lists_by)
        together = max(nd["at_once"] for nd in b["nodes"].values())
        reclaim = b["phases"].get("reclaim", 0.0)
        return ("%s (%s): %s, the PQ codes in %s s (by %s s), and the partitions load %s, %s s of work each, by "
                "%s s; lazy flips %.0f s after the trigger. The reclaim takes %s s, and %d queries fail. "
                % (name, b["series"], lists, rng(phase(b, "PQ codes", lambda a, z: z - a)),
                   rng(phase(b, "PQ codes", lambda a, z: z)), "one at a time" if together <= 1 else "%d at once" % together,
                   rng([nd["per_load"] for nd in b["nodes"].values()], "%.1f"), rng(phase(b, "loads", lambda a, z: z)),
                   b["flip"], ("%.1f" if reclaim < 10 else "%.0f") % reclaim, b["failed"])
                + graph_text(b))

    def graph_text(b):  # when each graph moved, against its node's own staging; the entries
        g = [(nd["graph"], max(z for _, _, z in nd["bars"])) for nd in b["nodes"].values() if nd["graph"] and nd["bars"]]
        ent = list(b["entries"].values())
        if not g or not ent:
            return ""
        return ("Its graphs start %s s after their node's staging ends and move in %s s each; the new entries come "
                "online %s s after the trigger%s, and the rescale ends with the last. "
                % (rng([max(a - staged, 0.0) for (a, _), staged in g]), rng([z - a for (a, z), _ in g]), rng(ent),
                   ", after the load" if min(ent) > b["end"] else ""))
    text = describe(old, "Before") + describe(new, "After")
    notes = textwrap.wrap(text, 112)
    yb = base + 56
    for i, s in enumerate(notes):
        svg.text(LEFT, yb + 15 * i, s, "note")
    return svg.save(out, "lazy-staging.svg", yb + 15 * len(notes) + 6,
                    "Lazy's rescale step by step: %s against %s" % (before, runset),
                    schedule(paths[1]))


LOADGEN_CONNS = 64  # rtier-loadgen -conns, its default: the runners do not set it


def lazy_entries(runs, out, runset, before, step=5.0):
    """With --before, when this series' new entries come online while lazy's backlog drains (since
    2026-11-08 the graph no longer waits for the queries' fetches, U16): where the backlog waits.
    The load generator sends each query, when it arrives, to the next entry of its list (round-
    robin), at most LOADGEN_CONNS in flight at an entry, the others queued at that entry in arrival
    order. A new entry starts with an empty queue, and answers its share of the arrivals as they
    come, while the initial entries' queues hold the backlog. Per 5 s: each group of entries'
    answers (their agents' counts) and the backlog, in both series; then the latency of the queries
    sent from the first new entry until the backlogs cleared, as a distribution."""
    if not before:
        return None
    import bisect
    after, olds = reps(runs, runset, "lazy", "metrics.jsonl"), reps(runs, before, "lazy", "metrics.jsonl")
    if not after or not olds:
        print("skipped lazy-entries.svg (runs missing)")
        return None

    def facts(run):
        rows = read_latency(os.path.join(run, "latency.csv"))
        sent = sorted(r["t"] for r in rows)
        ans = sorted(answered_at(r) for r in rows if r["status"] == "ok")
        trig, flip, _, rc = events(run)
        end = load_end(run) or 360.0
        backlog = [(t, bisect.bisect_left(sent, t) - bisect.bisect_left(ans, t)) for t in (trig + i for i in range(int(end - trig) + 1))]
        peak = max(backlog, key=lambda p: p[1])
        cleared = next((t for t, b in backlog if t > peak[0] and b <= 100), None)
        entries = {"n%d" % n: trig + e["seconds"] for e in rc["reply"].get("entry_flips", []) for n in e["nodes"]}
        tail = sorted(r["us"] / 1e6 for r in rows if r["status"] == "ok" and r["t"] >= trig)
        return dict(run=run, rows=rows, sent=sent, ans=ans, trig=trig, flip=flip, end=end, rc=rc, backlog=backlog,
                    peak=peak, cleared=cleared, entries=entries,
                    miss=sum(1 for r in rows if r["t"] >= trig and (r["status"] != "ok" or r["us"] > 200e3))
                    / max(sum(1 for r in rows if r["t"] >= trig), 1),
                    mean=sum(tail) / len(tail) if tail else None)
    afters, olds = [facts(r) for r in after], [facts(r) for r in olds]
    a = afters[0]  # the experiment, its nodes and its schedule: the same in every run
    if any(f["cleared"] is None for f in afters + olds):
        print("skipped lazy-entries.svg (a backlog of %s or %s does not clear during the load)" % (runset, before))
        return None
    if any(not f["entries"] or min(f["entries"].values()) >= f["cleared"] for f in afters):
        print("skipped lazy-entries.svg (no new entry of %s comes online while its backlog drains)" % runset)
        return None
    exp = load_json(os.path.join(a["run"], "experiment.json"))
    initial = ["n%d" % i for i in range(1, int(exp.get("InitialNodes", 4)) + 1)]
    added = ["n%d" % n for n in a["rc"]["reply"].get("added", [])]
    load = exp["Load"]
    steps_ = sorted((float(t), float(r)) for t, r in (x.split(":") for x in str(load.get("RateSteps", "")).split(",") if x))
    offered_at = lambda t: next((r for s, r in reversed(steps_) if s <= t), float(load["Rate"]))  # noqa: E731
    first = min(t for f in afters for t in f["entries"].values())
    last = max(t for f in afters for t in f["entries"].values())
    w0, w1 = first, max(f["cleared"] for f in afters + olds)  # the queries the figure's last panel takes
    r0, r1 = last, min(f["cleared"] for f in afters)  # the entries' rates: every run of this series drains
    lo, hi = a["trig"] - 10.0, min(a["end"], w1 + 30.0)
    bins = [lo + step * i for i in range(int((hi - lo) / step))]

    def group(f, names_):  # q/s the entries of names_ answered, per bin: each one's mean rate in it, summed
        m = metrics_series(f["run"], ("entry.latency.per_sec",))
        out_ = []
        for t in bins:
            tot = 0.0
            for n in names_:
                v = [x for tt, x in m.get((n, "entry.latency.per_sec"), []) if t <= tt < t + step]
                tot += sum(v) / len(v) if v else 0.0
            out_.append((t + step / 2, tot))
        return out_
    a_init, a_new = [group(f, initial) for f in afters], [group(f, added) for f in afters]
    o_init = [group(o, initial) for o in olds]

    def window_lat(f):  # the latency (s) of the queries sent in [w0, w1), sorted
        return sorted(r["us"] / 1e6 for r in f["rows"] if r["status"] == "ok" and w0 <= r["t"] < w1)
    lat = {"after": [window_lat(f) for f in afters], "before": [window_lat(o) for o in olds]}
    within = lambda xs, s: bisect.bisect_right(xs, s) / len(xs) if xs else 0.0  # noqa: E731
    pct = lambda xs, q: xs[min(int(q * len(xs)), len(xs) - 1)] if xs else float("nan")  # noqa: E731

    def mean_in(pts, t0_, t1_):
        v = [x for t, x in pts if t0_ <= t - step / 2 and t + step / 2 <= t1_]
        return sum(v) / len(v) if v else float("nan")

    svg = Svg()
    pw = W - LEFT - RIGHT
    ymax = max(max(v for _, v in s) for s in a_init + a_new + o_init + [[(0, offered_at(lo)), (0, offered_at(hi))]])
    pa = Panel(svg, LEFT, 84, pw, 150, (lo, hi), "answered queries per second at each group of entries (their agents' "
               "counts), per %g s" % step, "q/s", ymax=ymax * 1.15)
    pb = Panel(svg, LEFT, 294, pw, 110, (lo, hi), "backlog: queries sent, not answered yet", "queries",
               ymax=max(p[1] for f in afters + olds for p in f["backlog"] if lo <= p[0] <= hi))
    sched = [(lo, offered_at(lo))]
    for s, r in steps_:
        if lo < s < hi:
            sched += [(s, sched[-1][1]), (s, r)]
    sched.append((hi, offered_at(hi)))
    pa.line(sched, "cu sched")
    pa.line([(t, v) for t, v in [(last, offered_at(last) / 2), (hi, offered_at(hi) / 2)]], "ref")
    svg.text(pa.x(hi) - 4, pa.y(offered_at(hi) / 2) + 14, "half the offered rate", "note", "end")
    for o in o_init:
        pa.line(o, "ctx")
    for ai, an in zip(a_init, a_new):
        pa.line(ai, "c%d" % LAZY)
        pa.line(an, "c%d dash" % LAZY)
    for o in olds:
        pb.line([p for p in o["backlog"] if lo <= p[0] <= hi], "ctx")
    for f in afters:
        pb.line([p for p in f["backlog"] if lo <= p[0] <= hi], "c%d" % LAZY)
    # The trigger's label right of its line and the flip's left of its own, or one label for both where
    # lazy flips seconds after the trigger (Wikipedia since 2026-11-08): two would overlap.
    flips = sorted(set(f["flip"] for f in afters))
    close = pa.x(flips[-1]) - pa.x(a["trig"]) < 80
    cleared = sorted(f["cleared"] for f in afters)
    for p in (pa, pb):
        p.vline(a["trig"], "trigger" if p is pa and not close else None)
        for i, t in enumerate(flips):  # right of the last run's flip, or left of the first one's
            mine = p is pa and i == (len(flips) - 1 if close else 0)
            p.vline(t, ("trigger, flip" if close else "flip") if mine else None, "start" if close else "end")
        for i, t in enumerate(sorted(t for f in afters for t in f["entries"].values())):
            p.vline(t, "new entries online (after)" if p is pa and i == 0 else None)
        # right of the last run's line where the figure leaves room for it before the legend, else
        # left of the first one's: no other run's line crosses the label
        right = pa.x(hi) - pa.x(cleared[-1]) > 110
        for i, t in enumerate(cleared):
            mine = p is pa and i == (len(cleared) - 1 if right else 0)
            p.vline(t, "backlog cleared" if mine else None, "start" if right else "end", row=1)
    time_axis(svg, pb, pb.y0 + pb.ph, lo, hi)
    # the panels' titles name the entries, the figure's title the two series
    svg.legend(W - RIGHT + 16, pa.y0 + 10, [("initial (after)", "c%d" % LAZY, "line"), ("new (after)", "c%d dash" % LAZY, "line"),
                                            ("initial (before)", "ctx", "line"), ("offered", "cu sched", "line")])
    svg.legend(W - RIGHT + 16, pb.y0 + 10, [("after", "c%d" % LAZY, "line"), ("before", "ctx", "line")])

    def describe(t):
        k = min(int(round((t - lo) / step)), len(bins) - 1)
        lines = ["%.0f–%.0f s" % (t, t + step)]
        for f, ai, an in zip(afters, a_init, a_new):
            lines.append("after (%s): initial entries %.0f q/s, new entries %.0f q/s; backlog %s"
                         % (rep_name(f["run"]), ai[k][1], an[k][1], fmt_k(value_at(f["backlog"], t + step))))
        for o, oi in zip(olds, o_init):
            lines.append("before (%s): initial entries %.0f q/s; backlog %s"
                         % (rep_name(o["run"]), oi[k][1], fmt_k(value_at(o["backlog"], t + step))))
        return lines
    hover(svg, [pa, pb], lo, bins[-1] + step, step, describe)

    # the distribution: share answered within a time, on a log axis from 10 ms to 100 s
    yc = pb.y0 + pb.ph + 92
    # ymax under 1: the ticks then end at 100% (ticks_to clears its argument by 5%)
    pc = Panel(svg, LEFT, yc, pw, 150, (-2.0, 2.0), "queries sent from %.0f s (the first new entry) to %.0f s (both "
               "backlogs cleared): share answered within a latency" % (w0, w1), "",
               ymax=0.95, yfmt=lambda v: "%.0f%%" % (100 * v))
    for d, label in zip(range(-2, 3), ("10 ms", "100 ms", "1 s", "10 s", "100 s")):
        svg.add('<line class="grid" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>' % (pc.x(d), yc, pc.x(d), yc + pc.ph))
        svg.text(pc.x(d), yc + pc.ph + 16, label, anchor="middle")
    svg.text(pc.x0 + pc.pw / 2, yc + pc.ph + 32, "latency, from the query's arrival (log scale)", anchor="middle")
    pc.vline(math.log10(0.2), "200 ms")
    grid = [-2.0 + 0.02 * i for i in range(201)]
    for xs in lat["before"]:
        pc.line([(g, within(xs, 10 ** g)) for g in grid], "ctx")
    for xs in lat["after"]:
        pc.line([(g, within(xs, 10 ** g)) for g in grid], "c%d" % LAZY)
    for i in range(40):  # a readout per tenth of a decade
        g0, g1 = -2.0 + 0.1 * i, -2.0 + 0.1 * (i + 1)
        tip = ["within %s" % (("%.0f ms" % (1000 * 10 ** g1)) if g1 < 0 else "%.1f s" % 10 ** g1)]
        tip += ["after (%s): %.0f%%" % (rep_name(f["run"]), 100 * within(xs, 10 ** g1)) for f, xs in zip(afters, lat["after"])]
        tip += ["before (%s): %.0f%%" % (rep_name(o["run"]), 100 * within(xs, 10 ** g1)) for o, xs in zip(olds, lat["before"])]
        svg.add('<g class="hit"><title>%s</title><rect x="%.1f" y="%.1f" width="%.1f" height="%.1f"/></g>'
                % (esc("\n".join(tip)), pc.x(g0), yc, pc.x(g1) - pc.x(g0), pc.ph))
    svg.legend(W - RIGHT + 16, yc + 10, [("after", "c%d" % LAZY, "line"), ("before", "ctx", "line")])

    rng = lambda vals, f="%.0f": span_of(vals, f)  # noqa: E731
    o_ent = [t - o["trig"] for o in olds for t in o["entries"].values()]
    before_txt = ("Before (%s), the new entries came online %s s after the trigger, %s: the backlog drained through the "
                  "%d initial entries' queues, in arrival order. "
                  % (before, rng(o_ent), "after the backlog had cleared (at %s s)" % rng([o["cleared"] for o in olds])
                     if min(o_ent) + olds[0]["trig"] >= max(o["cleared"] for o in olds) else "while it drained",
                     len(initial)))
    share = [within(xs, 0.2) for xs in lat["before"]]
    text = ("The load generator sends each query, as it arrives, to the next entry of its list (round-robin), at most %d in "
            "flight at an entry; the others wait in that entry's queue, in arrival order. " % LOADGEN_CONNS + before_txt +
            "After (%s), they come online %s s after the trigger, at %s s, while %sK queries wait. A new entry starts with an "
            "empty queue and answers its share of the arrivals as they come: the %d new entries %s q/s together, half "
            "the offered rate, while the initial entries answer their queues at %s q/s together, against %s before, so "
            "the queries in them wait longer. Of the queries sent from %.0f s until both backlogs cleared, %s%% are "
            "answered within 200 ms, against %s%% before; the slowest 1%% wait %s s and more, against %s s. The backlog "
            "clears at %s s, against %s s%s. From the trigger on, "
            "%s%% of the queries miss 200 ms (%s%% before), and their mean latency is %s s (%s s)."
            % (runset, rng([t - f["trig"] for f in afters for t in f["entries"].values()]), rng([first, last]),
               rng([value_at(f["backlog"], min(f["entries"].values())) / 1e3 for f in afters], "%.1f"), len(added),
               rng([mean_in(an, r0, r1) for an in a_new]), rng([mean_in(ai, r0, r1) for ai in a_init]),
               rng([mean_in(oi, r0, r1) for oi in o_init]),
               w0, rng([100 * within(xs, 0.2) for xs in lat["after"]]), rng([100 * s for s in share]),
               rng([pct(xs, 0.99) for xs in lat["after"]]), rng([pct(xs, 0.99) for xs in lat["before"]]),
               rng(cleared), rng([o["cleared"] for o in olds]),
               # the same drain, its queries in another order -- not where the runs clear far apart
               ": the cluster answers as many queries, in another order"
               if min(cleared) >= min(o["cleared"] for o in olds) - 10 and max(cleared) <= max(o["cleared"] for o in olds) + 10
               else "",
               rng([100 * f["miss"] for f in afters]), rng([100 * o["miss"] for o in olds]),
               rng([f["mean"] for f in afters], "%.1f"), rng([o["mean"] for o in olds], "%.1f")))
    notes = textwrap.wrap(text, 112)
    yb = yc + pc.ph + 62
    for i, s in enumerate(notes):
        svg.text(LEFT, yb + 15 * i, s, "note")
    return svg.save(out, "lazy-entries.svg", yb + 15 * len(notes) + 6,
                    "Lazy's backlog once the new entries serve: %s against %s" % (before, runset),
                    "%s; lazy, %s against %s" % (schedule(a["run"]),
                                                 ", ".join(os.path.basename(os.path.dirname(f["run"])) for f in afters),
                                                 ", ".join(os.path.basename(os.path.dirname(o["run"])) for o in olds)))


def node_reads(runs, out, runset, lo=30.0, hi=360.0, step=5.0):
    picked = [reps(runs, runset, p, "stats-n1.jsonl") for p in ("lazy", "copy-then-flip")]
    if not all(picked):
        print("skipped node-reads.svg (runs missing)")
        return
    sf = reps(runs, runset, "copy-then-flip-stream-first", "stats-n1.jsonl")
    runs_ = [(os.path.relpath(rs[0], runs), rs[0]) for rs in picked + ([sf] if sf else [])]
    names = {"copy-then-flip-stream-first": "copy-then-flip, stream first"}
    nodes = ["n%d" % i for i in range(1, 9)]
    svg = Svg()
    pw = W - LEFT - RIGHT
    row, gap = 16, 2
    y = 94
    budget = read_budget(runs_[0][1])
    scale = budget  # the colour scale's end: the budget, else the busiest node's rate in round steps
    if not scale:
        top = max((v for _, run in runs_ for n in nodes
                   for _, v in rates(series(jsonl(os.path.join(run, "stats-%s.jsonl" % n)), "reads.charged", load_start(run)),
                                     lo, hi, step) if v is not None), default=0)
        unit = 1000 * len(HEAT[0])
        scale = max(unit, math.ceil(top / unit) * unit)
    p = None
    for rel, run in runs_:
        t0 = load_start(run)
        trig, flip, entries, rc = events(run)
        proto = rc["reply"]["protocol"]
        ph = len(nodes) * (row + gap)
        # the title a row up: the moments' labels take the row right above the cells
        p = Panel(svg, LEFT, y, pw, ph, (lo, hi), "%s (%s)" % (names.get(rel.split("/")[2], proto), rel.split("/")[1]),
                  "", grid=False, title_dy=-20)
        for i, n in enumerate(nodes):
            st = jsonl(os.path.join(run, "stats-%s.jsonl" % n))
            rs = rates(series(st, "reads.charged", t0), lo, hi, step)
            ry = y + i * (row + gap)
            svg.text(LEFT - 8, ry + row - 4, n, anchor="end")
            for t, v in rs:
                if v is None:
                    continue
                k = min(len(HEAT[0]) - 1, int(v / (scale / len(HEAT[0]))))
                svg.add('<rect class="h%d" x="%.1f" y="%.1f" width="%.1f" height="%d"><title>%s, %.0f–%.0f s: %s reads/s</title></rect>'
                        % (k, p.x(t - step / 2), ry, p.x(t + step / 2) - p.x(t - step / 2) - 0.6, row,
                           n, t - step / 2, t + step / 2, "{:,.0f}".format(v)))
        # the added nodes below the initial ones
        sep = y + 4 * (row + gap) - gap / 2
        svg.add('<line class="axis" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>' % (LEFT, sep, LEFT + pw, sep))
        p.vline(trig, "trigger", "end", above=True)
        shown = sorted(e for e in entries if e <= hi)
        if flip <= hi:
            # An entry flip right after it (copy-then-flip's graph waits for the stream) takes the
            # right of the line: the flip's label goes left.
            near = bool(shown) and p.x(shown[0]) - p.x(flip) < 40
            p.vline(flip, "flip", "end" if near else "start", above=True)
        else:
            svg.text(LEFT + pw - 4, y - 20, "flip at %.0f s, after the load" % flip, "note", "end")
        for i, e in enumerate(shown):
            p.vline(e, "new entries" if i == 0 else None, above=True)
        time_axis(svg, p, y + ph, lo, hi)
        y += ph + 80
    # the colour bar: 8 steps up to the scale's end (the budget, 5K steps at 40K/s)
    bx, by = W - RIGHT + 16, 94
    svg.text(bx, by - 6, "reads/s", "legend")
    for k in range(len(HEAT[0])):
        yy = by + k * 14
        svg.add('<rect class="h%d" x="%d" y="%d" width="14" height="12" rx="2"/>' % (len(HEAT[0]) - 1 - k, bx, yy))
        lo_k = scale / len(HEAT[0]) * (len(HEAT[0]) - 1 - k)
        svg.text(bx + 22, yy + 10, ("≥ %s" % fmt_k(lo_k)) if k == 0 else "%s–%s" % (fmt_k(lo_k), fmt_k(lo_k + scale / len(HEAT[0]))), "legend")
    svg.text(bx, by + 14 * len(HEAT[0]) + 14, "budget %s/s" % fmt_k(budget) if budget else "no read limit", "legend")
    if all(events(run)[1] > hi for _, run in runs_):
        notes = textwrap.wrap("n1-n4 hold the data from the start; n5-n8 join at the flip, which comes after the "
                              "load's end under every protocol here: through the load the initial nodes carry every "
                              "query, and %s the capacity."
                              % ("those at their budget bound" if budget else "the busiest of them bounds"), 108)
    elif budget:
        notes = ["n1-n4 hold the data from the start; n5-n8 join at the flip. Under lazy, the new nodes take the reads as",
                 "their fetches warm them up; a node at the end of the scale is at its budget and bounds the capacity.",
                 "Under copy-then-flip, the new nodes serve nothing before the flip: the initial nodes carry the queries",
                 "and the raw stream on one budget."]
    else:
        notes = textwrap.wrap("n1-n4 hold the data from the start; n5-n8 join at the flip. Under lazy, the new nodes take "
                              "the reads as their fetches warm them up; the busiest node bounds the capacity. Under "
                              "copy-then-flip, the new nodes serve nothing before the flip: the initial nodes carry the "
                              "queries and the raw stream on one device.", 108)
    if len(runs_) > 2:
        notes += textwrap.wrap("With the stream first, the stream's reads go ahead of the queries' (ctf-budget.svg "
                               "splits them) until the flip at %.0f s; then the new nodes take their share."
                               % events(runs_[2][1])[1], 108)
    for i, n in enumerate(notes):
        svg.text(LEFT, y - 24 + 15 * i, n, "note")
    return svg.save(out, "node-reads.svg", y + 15 * len(notes) - 14,
                    "Reads per second charged to each node's budget, per 5 s" if budget
                    else "Device reads per second on each node, per 5 s",
                    "the same queries in every run; the colour scale ends at %s"
                    % ("the budget, %s reads/s" % "{:,}".format(budget) if budget
                       else "%s reads/s, round above the busiest node's (no read limit)" % "{:,}".format(scale)))


def calibration_trigger(run):
    """calibrate-8 scales out before its load: its trigger on the controller's clock is the
    ladder's START (calibration.log) less the rescale's elapsed time."""
    with open(os.path.join(run, "calibration.log")) as f:
        start = float(f.readline().split()[1])
    return start - load_json(os.path.join(run, "reconfigurations.json"))[0]["elapsed_seconds"]


def span_of(vals, f="%.0f"):
    """A range as the notes print it: one value when the runs agree, else lowest–highest."""
    a, b = f % min(vals), f % max(vals)
    return a if a == b else "%s–%s" % (a, b)


def stage_raw(run):
    return events(run)[3]["reply"]["phases"]["stage_raw"]


def revision(run, before="", after=""):
    """The commit a run ran (revision.txt's first line), short, between before and after; "" if
    unknown."""
    try:
        with open(os.path.join(run, "revision.txt")) as f:
            words = f.readline().split()
    except OSError:
        return ""
    return before + words[1][:7] + after if len(words) > 1 and words[0] == "commit" else ""


def copy_runs(runs, runset, protocol):
    """rep1/ and rep2/ of a protocol in the series, those with stats."""
    return reps(runs, runset, protocol, "stats-n1.jsonl")


def ctf_stream(runs, out, runset, calibrations, lo=0.0, hi=None):
    loaded, first = copy_runs(runs, runset, "copy-then-flip"), copy_runs(runs, runset, "copy-then-flip-stream-first")
    calm = os.path.join(runs, calibrations, "calibrate-8")
    if not loaded or not os.path.exists(os.path.join(calm, "calibration.log")):
        print("skipped ctf-stream.svg (runs missing)")
        return
    sources = {o: ", ".join(ns) for o, ns in sorted(plan_moves(loaded[0])[1].items(), key=lambda kv: int(kv[0][1:]))}
    # each source's share of the query reads on 4 nodes, before the trigger
    share, total = {}, 0
    t0 = load_start(loaded[0])
    for n in sources:
        pts = series(jsonl(os.path.join(loaded[0], "stats-%s.jsonl" % n)), "reads.charged", t0)
        share[n] = value_at(pts, 90) - value_at(pts, 30)
        total += share[n]

    def sent(run, n, trig):
        return [(rec["t"] - trig, rec["raw_bytes"]["sent"] / 1e9) for rec in jsonl(os.path.join(run, "stats-%s.jsonl" % n))]

    # No load last: dashed on top, it shows where the stream-first line runs along it.
    curves = {n: [("c%d" % CTF, sent(r, n, load_start(r) + events(r)[0])) for r in loaded] +
                 [("c%d" % SF, sent(r, n, load_start(r) + events(r)[0])) for r in first] +
                 [("neu dash", sent(calm, n, calibration_trigger(calm)))] for n in sources}
    end = load_end(loaded[0]) - events(loaded[0])[0]  # the load's end, in seconds since the trigger
    ours = [pts for n in sources for cls, pts in curves[n] if cls == "c%d" % CTF and pts]
    # The stream's first bytes under the load: it starts once the lists and PQ codes have moved.
    began = min((next((t for t, v in pts if v > 0), float("inf")) for pts in ours), default=float("inf"))
    if hi is None:
        # Until every stream has ended: 530 s for the Wikipedia runs, whose figure keeps that.
        done = [next(t for t, v in pts if v >= pts[-1][1]) for _, cs in curves.items() for _, pts in cs if pts]
        hi = max(530.0, max(done, default=0.0) * 1.03)
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
        p.vline(end, "load ends" if i == 0 else None)
        for cls, pts in curves[n]:
            p.line([(t, v) for t, v in pts if lo <= t <= hi], cls)
        time_axis(svg, p, cy + ph, lo, hi, label="seconds since the trigger" if i // cols == 1 else "", count=5)
        panels.append(p)
    items = [("one budget", "c%d" % CTF, "line"), ("(%s)" % runs_count(len(loaded)), "", "none")]
    if first:
        items += [("stream first", "c%d" % SF, "line"), ("(%s)" % runs_count(len(first)), "", "none")]
    svg.legend(W - RIGHT + 16, 94, items + [("no load", "neu dash", "line"), ("(calibrate-8)", "", "none")])
    yb = 84 + 2 * (ph + gy) - 10
    calm_raw = load_json(os.path.join(calm, "reconfigurations.json"))[0]["reply"]["phases"]["stage_raw"]
    if began >= end:
        text = ("Copy-then-flip streams each moved partition's raw vectors from its initial owner (RAW_PULL) once "
                "the partitions' lists and PQ codes have moved, %.0f s after the trigger: after the load's end. Its "
                "stream meets no queries but the backlog's, and takes %s s, against %.0f s with no load."
                % (began, span_of([stage_raw(r) for r in loaded]), calm_raw))
    else:
        text = ("Copy-then-flip streams each moved partition's raw vectors from its initial owner (RAW_PULL). On one "
                "budget of 40K reads/s, in arrival order with the queries' reads, the busiest initial nodes spend "
                "theirs on queries, and their streams barely move until the load ends and the backlog drains: the "
                "stream takes %s s, against %.0f s with no load." % (span_of([stage_raw(r) for r in loaded]), calm_raw))
    if first:
        took = [stage_raw(r) for r in first]
        like = ("about as long as with none" if max(abs(t / calm_raw - 1) for t in took) < 0.1
                else "against %.0f s with none" % calm_raw)
        text += (" With the stream first (rtier_node --peer-reads-first), it takes %s s under the load, %s."
                 % (span_of(took), like))
    text += " The node with the fewest queries finishes first." if began < end else ""
    # The calibrations can come from another series, staged by older code: say where its stream
    # starts, so that its line is read for its slope, not its place.
    calm_began = min((next((t for t, v in pts if v > 0), float("inf"))
                      for cs in curves.values() for cls, pts in cs if cls == "neu dash" and pts), default=float("inf"))
    if math.isfinite(calm_began) and math.isfinite(began) and abs(calm_began - began) > 30:
        text += (" The run with no load (%s/calibrate-8%s) began to stream %.0f s after its trigger, against %.0f s in "
                 "these runs%s; compare the slopes." % (calibrations, revision(calm, ", commit "), calm_began, began,
                                                        revision(loaded[0], " (commit ", ")")))
    notes = textwrap.wrap(text, 108)
    for i, s in enumerate(notes):
        svg.text(LEFT, yb + 15 * i, s, "note")
    return svg.save(out, "ctf-stream.svg", yb + 15 * len(notes) + 6,
                    "Copy-then-flip's raw-vector stream from each initial node: %.1f GB in all"
                    % (events(loaded[0])[3]["reply"]["raw_bytes"] / 1e9),
                    "the raw vectors each initial node has sent to its new node, cumulative")


def answered_between(run, a, b):
    """Queries answered per second between a and b seconds of the load."""
    return sum(1 for r in read_latency(os.path.join(run, "latency.csv"))
               if r["status"] == "ok" and a <= answered_at(r) < b) / (b - a)


def ctf_budget(runs, out, runset, step=5.0):
    """Each initial node's reads while copy-then-flip copies, its raw stream's against its queries':
    on one budget in arrival order, and with the stream first (the first repetition of each)."""
    cols = [("copy-then-flip", "one budget, in arrival order", CTF),
            ("copy-then-flip-stream-first", "the stream first (--peer-reads-first)", SF)]
    picked = [(copy_runs(runs, runset, p), head, slot) for p, head, slot in cols]
    if not all(rs for rs, _, _ in picked):
        print("skipped ctf-budget.svg (runs missing)")
        return
    sources = {o: ", ".join(ns) for o, ns in sorted(plan_moves(picked[0][0][0])[1].items(), key=lambda kv: int(kv[0][1:]))}
    budget = 40000
    svg = Svg()
    gx, ph, gy, top = 40, 92, 64, 124
    pw = (W - LEFT - RIGHT - gx) / 2
    facts = []
    for c, (rs, head, slot) in enumerate(picked):
        run = rs[0]
        t0 = load_start(run)
        trig, flip, _, rc = events(run)
        reply = rc["reply"]
        vb = reply["raw_bytes"] / reply["raw_vectors"]  # a streamed vector's bytes: one read each
        start = sum(reply["phases"].get(k, 0.0) for k in ("plan", "stage_aggregator", "save_segments", "pause", "stage_data"))
        lo, hi = 0.0, load_end(run) - trig  # from the trigger to the load's end
        cx = LEFT + c * (pw + gx)
        svg.text(cx, top - 46, head, "title ptitle")
        svg.text(cx, top - 30, "flip %.0f s after the trigger%s" % (flip - trig, ", after the load" if flip - trig > hi else ""),
                 "note")
        panels, rows, per = [], {}, {}
        for i, (src, dst) in enumerate(sources.items()):
            st = jsonl(os.path.join(run, "stats-%s.jsonl" % src))
            sent, disk = series(st, "raw_bytes.sent", t0 + trig), series(st, "fetch.rerank_disk", t0 + trig)
            stream, queries = rates(sent, lo, hi, step, 1.0 / vb), rates(disk, lo, hi, step)
            cy = top + i * (ph + gy)
            # the budget as the top tick (ticks_to clears its ymax by 5%)
            p = Panel(svg, cx, cy, pw, ph, (lo, hi), "%s → %s" % (src, dst), "reads/s" if c == 1 else "",
                      ymax=budget / 1.06, yfmt=fmt_k)
            p.hline(budget, "budget" if i == 0 else None)
            if flip - trig <= hi:
                p.vline(flip - trig, "flip" if i == 0 else None)
            p.line(queries, "neu")
            p.line(stream, "c%d" % slot)
            time_axis(svg, p, cy + ph, lo, hi, label="seconds since the trigger" if i == len(sources) - 1 else "", count=6)
            panels.append(p)
            rows[src] = (stream, queries)
            # the rates while this node streams, within the load
            done = next((t for t, v in sent if v >= sent[-1][1]), hi)
            a, b = start, min(done, hi)
            per[src] = ((value_at(sent, b) - value_at(sent, a)) / vb / (b - a), (value_at(disk, b) - value_at(disk, a)) / (b - a))

        def describe(t, rows=rows, head=head):
            k = int(round(t / step))
            lines = ["%s, %.0f–%.0f s after the trigger" % (head, t, t + step)]
            for src, (s_, q_) in rows.items():
                if k < len(s_):
                    lines.append("%s: stream %s/s, queries %s/s" % (src, fmt_k(s_[k][1]), fmt_k(q_[k][1])))
            return lines
        hover(svg, panels, lo, hi, step, describe)
        facts.append(dict(per=per, qps=answered_between(run, trig, min(flip, trig + hi)), flip=flip - trig))
    svg.legend(W - RIGHT + 16, top - 46, [("raw stream", "c%d" % CTF, "line"), ("(one budget)", "", "none"),
                                     ("raw stream", "c%d" % SF, "line"), ("(stream first)", "", "none"),
                                     ("queries", "neu", "line"), ("(re-ranking)", "", "none"),
                                     ("budget", "ref", "line")])
    one, sf = facts
    k = lambda v: fmt_k(round(v, -2))  # noqa: E731
    rng = lambda vs: "%s–%s" % (k(min(vs)), k(max(vs)))  # noqa: E731
    busiest = sorted(sources, key=lambda s: -one["per"][s][1])[:2]
    text = ("On one budget, in arrival order, the busiest initial nodes spend it on their queries: while they stream, "
            "%s re-rank %s vectors/s from disk and stream %s. With the stream first, n1-n4 stream %s vectors/s and "
            "their queries get %s reads/s. The cluster answers %.0f q/s during the copy instead of %.0f, and the flip "
            "comes %.0f s after the trigger instead of %.0f s."
            % (" and ".join(busiest), " and ".join(k(one["per"][s][1]) for s in busiest),
               " and ".join(k(one["per"][s][0]) for s in busiest), rng([sf["per"][s][0] for s in sources]),
               rng([sf["per"][s][1] for s in sources]), sf["qps"], one["qps"], sf["flip"], one["flip"]))
    notes = textwrap.wrap(text, 108)
    yb = top + 3 * (ph + gy) + ph + 62
    for i, s in enumerate(notes):
        svg.text(LEFT, yb + 15 * i, s, "note")
    return svg.save(out, "ctf-budget.svg", yb + 15 * len(notes) + 6,
                    "Copy-then-flip's copy against the queries at each initial node (repetition 1)",
                    "per 5 s: the vectors each initial node streams (RAW_GET) and those its queries re-rank from disk, "
                    "both on its 40K reads/s budget")


def lazy_keepup(runs, out, runset, calibrations, lo=30.0, step=5.0):
    """Lazy through the whole load, every repetition: whether it holds the offered rate while its
    new nodes still fetch, against what copy-then-flip copies (calibrate-8's scale-out) and what
    8 nodes with every raw vector local answer in (calibrate-8's ladder)."""
    rs = reps(runs, runset, "lazy", "stats-n5.jsonl")
    if rs and all(events(r)[1] >= (load_end(r) or 360.0) for r in rs):
        print("skipped lazy-keepup.svg (lazy flips after the load's end: lazy-stall.svg)")
        return None
    calm = os.path.join(runs, calibrations, "calibrate-8")
    if not rs or not os.path.exists(os.path.join(calm, "calibration.json")):
        print("skipped lazy-keepup.svg (runs missing)")
        return
    import bisect
    copy = load_json(os.path.join(calm, "reconfigurations.json"))[0]["reply"]["raw_bytes"]
    ladder = load_json(os.path.join(calm, "calibration.json"))
    c8 = sum(s["answered_in_window_per_s"] for s in ladder[-3:]) / 3
    cal4 = os.path.join(runs, calibrations, "calibrate-4", "calibration.json")
    c4 = sum(s["answered_in_window_per_s"] for s in load_json(cal4)[-3:]) / 3 if os.path.exists(cal4) else None
    exp = load_json(os.path.join(rs[0], "experiment.json"))["Load"]
    hi = load_end(rs[0])
    rate0, (s1, rate1) = float(exp["Rate"]), [float(x) for x in str(exp["RateSteps"]).split(",")[0].split(":")]
    # what 8 warm nodes answer in at the offered rate: the ladder's steps on either side of it
    below = max((s for s in ladder if s["offered_per_s"] <= rate1), key=lambda s: s["offered_per_s"], default=ladder[0])
    above = min((s for s in ladder if s["offered_per_s"] >= rate1), key=lambda s: s["offered_per_s"], default=ladder[-1])
    warm = (below["mean_us"] + above["mean_us"]) / 2 / 1000.0
    per = []  # per run: answered, backlog, mean latency, fetched GB per bin; events; facts
    for run in rs:
        t0 = load_start(run)
        trig, flip, entries, rc = events(run)
        rows = read_latency(os.path.join(run, "latency.csv"))
        sent = sorted(r["t"] for r in rows)
        ans = sorted(answered_at(r) for r in rows if r["status"] == "ok")
        bins = [lo + step * i for i in range(int((hi - lo) / step))]
        win = {round(w["t"]): w for w in windows(rows, step)}
        answered = [(t + step / 2, (bisect.bisect_left(ans, t + step) - bisect.bisect_left(ans, t)) / step) for t in bins]
        backlog = [(t + step, bisect.bisect_left(sent, t + step) - bisect.bisect_left(ans, t + step)) for t in bins]
        mean = [(t + step / 2, win[round(t)]["mean_us"] / 1000.0 if round(t) in win and win[round(t)]["mean_us"] else None)
                for t in bins]
        reply = rc["reply"]
        vb = reply["raw_fetched_bytes"] / reply["raw_fetched"] if reply.get("raw_fetched") else 4096
        def counted(pts, t):  # a cumulative counter at t; past its last sample, its last value
            return pts[-1][1] if t >= pts[-1][0] else (value_at(pts, t) or 0)
        cum = [series(jsonl(os.path.join(run, "stats-%s.jsonl" % n)), "fetch.fetched", t0) for n in ("n5", "n6", "n7", "n8")]
        fetched = [sum(counted(pts, t + step) for pts in cum) for t in bins]
        gb = [(t + step, f * vb / 1e9) for t, f in zip(bins, fetched)]
        peak = max(backlog, key=lambda p: p[1])
        cleared = next((t for t, b in backlog if t > peak[0] and b <= 100), None)
        steady = [r for r in rows if r["status"] == "ok" and cleared and cleared + 20 <= answered_at(r) < hi]
        lat = sorted(r["us"] / 1000.0 for r in steady)
        fetch_rate = lambda t: (value_at(gb, t + step) - value_at(gb, t)) / step * 1000 if value_at(gb, t) is not None and value_at(gb, t + step) is not None else None
        per.append(dict(run=run, answered=answered, backlog=backlog, mean=mean, gb=gb, trig=trig, flip=flip,
                        entries=entries, cleared=cleared,
                        steady_qps=len(steady) / (hi - cleared - 20) if cleared and hi - cleared > 20 else None,
                        steady_mean=sum(lat) / len(lat) if lat else None, steady_p99=lat[int(0.99 * len(lat))] if lat else None,
                        rate_from=fetch_rate(cleared + 20) if cleared else None, rate_to=fetch_rate(hi - step - step),
                        fetched_end=value_at(gb, hi) or gb[-1][1]))
    svg = Svg()
    pw = W - LEFT - RIGHT
    xr = (lo, hi)
    ys = [84, 300, 470, 670]
    # room above the 8 nodes' line for its label: a drain can answer close to it at the load's end
    p1 = Panel(svg, LEFT, ys[0], pw, 160, xr, "answered and offered queries per second (per %.0f s)" % step, "q/s",
               ymax=max(max(v for p in per for _, v in p["answered"]), 1.1 * c8))
    p2 = Panel(svg, LEFT, ys[1], pw, 110, xr, "backlog: queries sent, not answered yet", "queries",
               ymax=max(v for p in per for _, v in p["backlog"]))
    p3 = Panel(svg, LEFT, ys[2], pw, 140, xr, "mean latency of the answered queries (log scale)", "ms", log=(1, 100000),
               yfmt=lambda v: "{:,.0f}".format(v))
    p4 = Panel(svg, LEFT, ys[3], pw, 140, xr, "raw vectors the new nodes fetched, cumulative", "GB", ymax=copy / 1e9)
    sched = [(lo, rate0 if lo < s1 else rate1), (s1, rate0), (s1, rate1), (hi, rate1)] if lo < s1 else [(lo, rate1), (hi, rate1)]
    p1.line(sched, "cu sched")
    # above its line, at the right: the events' labels are at the top left, and a backlog that
    # drains until the load's end answers just below the line there
    p1.hline(c8, "8 nodes' capacity, %.0f q/s" % c8)
    if c4:
        p1.hline(c4, "4 nodes' capacity, %.0f q/s" % c4, dy=14)
    p3.hline(warm, "8 nodes with every raw vector local, at %.0f q/s: %.1f ms" % (rate1, warm), dy=14)
    p4.hline(copy / 1e9, "copy-then-flip streams %.1f GB before its flip" % (copy / 1e9), dy=14)
    first = per[0]
    for p in (p1, p2, p3, p4):
        p.vline(first["trig"], "trigger" if p is p1 else None, "end")
        p.vline(first["flip"], "flip" if p is p1 else None)
        # Labelled in the backlog's panel, whose top is clear once the backlog drains: right of the
        # last run's line, or left of the first one's where the last is near the right edge (the
        # legend's side), so that no other run's line crosses the label.
        cl = sorted(q["cleared"] for q in per if q["cleared"])
        right = bool(cl) and cl[-1] <= lo + 0.75 * (hi - lo)
        for i, t in enumerate(cl):
            mine = p is p2 and i == (len(cl) - 1 if right else 0)
            p.vline(t, "backlog cleared" if mine else None, "start" if right else "end")
    for q in per:
        p1.line(q["answered"], "c%d" % LAZY)
        p2.line(q["backlog"], "c%d" % LAZY)
        p3.line(q["mean"], "c%d" % LAZY)
        p4.line(q["gb"], "c%d" % LAZY)
    time_axis(svg, p4, ys[3] + 140, lo, hi)
    svg.legend(W - RIGHT + 16, ys[0] + 10, [("lazy", "c%d" % LAZY, "line"), ("(%s)" % runs_count(len(per)), "", "none"),
                                            ("offered", "cu sched", "line")])
    rng = lambda key, f="%.0f": span_of([q[key] for q in per if q[key] is not None], f)  # noqa: E731
    exp = load_json(os.path.join(rs[0], "experiment.json"))
    queries, name = exp.get("Queries", ""), dataset(exp)
    text = ("%s, %.0f q/s, then %.0f from %.0f s; 4 → 8 at %.0f s. "
            % ("Queries drawn from the base without repetition (U11's worst case)" if "worst" in os.path.basename(queries)
               else "The Wikipedia queries (the file is shorter than a run: they repeat)" if name == "Wikipedia"
               else "The %s queries" % name,
               rate0, rate1, s1, first["trig"]))
    done, left = [q for q in per if q["cleared"]], [q for q in per if not q["cleared"]]
    names = lambda qs: " and ".join(rep_name(q["run"]) for q in qs)  # noqa: E731
    if done and (left or any(q["steady_mean"] is None for q in done)):
        # too late for a steady state before the load's end, or not in every repetition
        text += "Lazy's backlog clears under the load at %s s%s" % (
            span_of([q["cleared"] for q in done]), " in %s" % names(done) if left else ".")
        if left:
            text += "; in %s, %s queries still wait when the load ends." % (
                names(left), span_of([q["backlog"][-1][1] for q in left]))
        text += " "
    elif done:
        still = (", while its new nodes still fetch %s MB/s, and %s MB/s at the load's end" % (rng("rate_from"), rng("rate_to"))
                 if max(q["rate_from"] or 0 for q in per) >= 1 else "; its new nodes have stopped fetching by then")
        # "as warm nodes do" only when it is so (within 15%); else the two numbers
        like = ("as 8 nodes with every raw vector local do" if max(q["steady_mean"] for q in per) <= 1.15 * warm
                else "against %.1f ms on 8 nodes with every raw vector local" % warm)
        text += ("Lazy's backlog clears under the load at %s s. From 20 s later to the load's end it answers %s q/s "
                 "against %.0f offered, at a mean latency of %s ms (p99 %s ms), %s%s. "
                 % (rng("cleared"), rng("steady_qps"), rate1, rng("steady_mean", "%.1f"), rng("steady_p99"), like, still))
    else:
        text += ("Lazy's backlog does not clear before the load ends: %s queries still wait then. "
                 % span_of([q["backlog"][-1][1] for q in per]))
    text += ("By the load's end its new nodes have fetched %s GB, %s%% of what copy-then-flip streams before it "
             "serves anything new." % (rng("fetched_end", "%.1f"), span_of([100 * q["fetched_end"] * 1e9 / copy for q in per])))
    online = [e - q["trig"] for q in per for e in q["entries"]]  # the first new entry to the last
    if online:
        text += " The new entries come online %s s after the trigger%s." % (
            span_of(online), " (U16's strict yield: data moves until then)" if min(online) > hi - first["trig"] else "")
    notes = textwrap.wrap(text, 108)
    yb = ys[3] + 140 + 62
    for i, s in enumerate(notes):
        svg.text(LEFT, yb + 15 * i, s, "note")
    return svg.save(out, "lazy-keepup.svg", yb + 15 * len(notes) + 6,
                    "Lazy through the whole load: the offered rate held while its new nodes still fetch",
                    "%s; the backlog, latency and fetched data of each"
                    % ("both repetitions" if len(per) == 2 else ", ".join(rep_name(q["run"]) for q in per)))


def keepup_all(runs, out, runset, lazy_set, window=WINDOW):
    """Lazy and copy-then-flip's two baselines until every query is answered, drawn by
    compare_runs.py as the series' other timelines are: offered and answered queries, the backlog
    and the mean latency; the reconfigurations; the data each transferred, per run. Lazy's runs
    come from lazy_set, a series whose load may go on longer: the figure then ends where the
    baselines' load stops, the last moment the three ran one load (past it a line's backlog
    balances against its own load). With one load it goes on until every query is answered. Per
    5 s windows,
    the latency on a log scale (lazy's return to a warm cluster's milliseconds, against minutes),
    and no title or subtitle: a paper figure, whose caption carries them."""
    dirs = (["%s/rep%d/lazy" % (lazy_set, r) for r in (1, 2)] +
            ["%s/rep%d/%s" % (runset, r, p) for p in ("copy-then-flip", "copy-then-flip-stream-first") for r in (1, 2)])
    paths = [os.path.join(runs, d) for d in dirs]
    if not all(os.path.exists(os.path.join(p, "latency.csv")) for p in paths):
        print("skipped keepup-all.svg (runs missing)")
        return None
    path = os.path.join(out, "keepup-all")
    ends = sorted({e for e in (load_end(p) for p in paths) if e})
    subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), "compare_runs.py")] + paths +
                   ["--labels", ",".join(["lazy"] * 2 + ["copy-then-flip"] * 2 + ["ctf stream-first"] * 2),
                    "--until", "answer" if len(ends) < 2 else "%g" % ends[0],
                    "--window", "%g" % window, "--log-latency", "--title", "", "--out", path],
                   check=True, stdout=subprocess.DEVNULL)
    print("wrote", path + ".svg")
    return path + ".svg"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default="results/cloudlab-gpu")
    ap.add_argument("--out", help="where the figures go (default: results/cloudlab-gpu-figures/SERIES)")
    ap.add_argument("--series", default="4to8", help="the runs to draw: a directory under --runs (default 4to8)")
    ap.add_argument("--calibrations", default="4to8",
                    help="the directory under --runs with calibrate-4 and calibrate-8 (default 4to8)")
    ap.add_argument("--before", help="another series, whose lazy runs lazy-before-after.svg draws against these")
    ap.add_argument("--fetch-bound-mb", type=int, default=2048,
                    help="the fetch cache's bound the runs had (default 2048; 64 before 2026-11-04, as in 4to8)")
    ap.add_argument("--lazy-series", help="keepup-all.svg: lazy's runs from this series (a longer load), "
                    "against --series' copy-then-flip runs (default: --series)")
    ap.add_argument("--drained", action="store_true",
                    help="also draw each timeline until every query is answered, the backlogs drained (NAME-drained.svg)")
    ap.add_argument("--window", type=float, default=WINDOW,
                    help="seconds per point of the compare_runs.py timelines and keepup-all.svg (default %g)" % WINDOW)
    args = ap.parse_args()
    args.out = args.out or os.path.join("results/cloudlab-gpu-figures", args.series)
    os.makedirs(args.out, exist_ok=True)
    specs = compare_specs(args.runs, args.series, args.before)
    timelines(args.runs, args.out, specs, window=args.window)
    if args.drained:
        timelines(args.runs, args.out, specs, until="answer", window=args.window)
    made = [capacity(args.runs, args.out, args.series, args.calibrations),
            lazy_fetch(args.runs, args.out, args.series, args.calibrations, args.fetch_bound_mb),
            lazy_stall(args.runs, args.out, args.series),
            lazy_staging(args.runs, args.out, args.series, args.before),
            lazy_entries(args.runs, args.out, args.series, args.before),
            node_reads(args.runs, args.out, args.series),
            ctf_stream(args.runs, args.out, args.series, args.calibrations),
            ctf_budget(args.runs, args.out, args.series),
            lazy_keepup(args.runs, args.out, args.series, args.calibrations),
            keepup_all(args.runs, args.out, args.series, args.lazy_series or args.series, args.window)]
    made += [os.path.join(args.out, spec[0] + suffix + ".svg") for spec in specs
             for suffix in ("", "-drained") if suffix == "" or args.drained]
    if shutil.which("rsvg-convert"):
        for path in made:
            if path and os.path.exists(path):
                subprocess.run(["rsvg-convert", "-w", "1800", path, "-o", path[:-4] + ".png"], check=True)


if __name__ == "__main__":
    main()
