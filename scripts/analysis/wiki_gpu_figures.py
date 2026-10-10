#!/usr/bin/env python3
"""Figures for the wiki-c1024 GPU runs on the lab server (no dependencies: plain SVG).

    python3 scripts/analysis/wiki_gpu_figures.py [--runs results/wiki-gpu] [--out results/wiki-gpu-figures]

The runs directory is what was downloaded from the server; the figures go to a directory of
their own, so that downloading the runs again does not delete them. Writes, into --out:

  compare_runs.py timelines, one per pair (lazy vs copy-then-flip, or a repeat on other GPUs)
  capacity.svg            answered vs offered for every calibration; peak throughput per budget
  protocols-vs-load.svg   lazy vs copy-then-flip at 40K reads/s, three loads
  protocols-vs-budget.svg lazy vs copy-then-flip under overload at 40K / 60K / 80K reads/s
  copy-rate.svg           copy-then-flip's raw-vector copy per source over time, 40K, three loads
  copy-rate-budgets.svg   the same under overload at 40K / 60K / 80K reads/s
  trigger-stall.svg       the dip right after the trigger: writeback on the shared disk

and a PNG of every SVG when rsvg-convert is installed. Runs that are missing are skipped.
"""

import argparse
import csv
import glob
import json
import math
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMPARE = os.path.join(os.path.dirname(HERE), "compare_runs.py")
RUNS, OUT = "results/wiki-gpu", "results/wiki-gpu-figures"
C2_40K = 330  # 2-node capacity at 40K reads/s per node (calibrate-2, 2026-10-07; not among the downloaded runs)
BATCH_MB = 64 * 4096 / 1e6  # every PULL_RAW batch of these runs is full: 64 vectors of 4 KB

BLUE, ORANGE, GREY, GREEN, PURPLE, RED = "#2a76d2", "#e8673a", "#777777", "#2e9d5b", "#8a56c2", "#c23b3b"
PALETTE = [BLUE, ORANGE, GREEN, PURPLE, RED, "#c79a1e", "#1aa3a3", "#555555", "#d45fa0"]

# compare_runs.py outputs: (name, title, [(label, run dir)])
PAIRS = [
    ("2to4-40k-early", "2 -> 4 at 40K reads/s, load 210 -> 265 q/s",
     [("lazy", "2to4-early/lazy"), ("copy-then-flip", "2to4-early/copy-then-flip")]),
    ("2to4-40k-mild", "2 -> 4 at 40K reads/s, load 210 -> 315 q/s",
     [("lazy", "2to4-mild/lazy"), ("copy-then-flip", "2to4-mild/copy-then-flip")]),
    ("2to4-40k-full", "2 -> 4 at 40K reads/s, load 210 -> 420 q/s",
     [("lazy", "2to4/lazy"), ("copy-then-flip", "2to4/copy-then-flip")]),
    ("2to4-60k-full", "2 -> 4 at 60K reads/s, load 325 -> 650 q/s",
     [("lazy", "2to4-60k/lazy"), ("copy-then-flip", "2to4-60k/copy-then-flip")]),
    ("2to4-80k-full", "2 -> 4 at 80K reads/s, load 400 -> 800 q/s",
     [("lazy", "2to4-80k/lazy"), ("copy-then-flip", "2to4-80k/copy-then-flip")]),
    ("2to4-100k-full", "2 -> 4 at 100K reads/s, load 550 -> 1100 q/s",
     [("lazy", "2to4-100k/lazy"), ("copy-then-flip", "2to4-100k/copy-then-flip")]),
    ("2to4-60k-ctf-new", "2 -> 4 at 60K reads/s, 325 -> 650 q/s: copy-then-flip with the old stream and the data node path (new)",
     [("lazy", "2to4-60k/lazy"), ("ctf old", "2to4-60k/copy-then-flip"),
      ("ctf new", "2to4-60k/copy-then-flip-optimize")]),
    ("2to4to2-40k-lazy", "2 -> 4 -> 2, lazy, 40K reads/s, 210 -> 420 -> 210 q/s", [("lazy", "2to4to2")]),
    ("gpu-check-ctf-early", "copy-then-flip 2 -> 4 at 40K (load 265): GPU models",
     [("4x PCIe 40GB (GPUs 3-6)", "2to4-early/copy-then-flip"), ("2x SXM4 80GB + 2x PCIe 40GB", "2to4/copy-then-flip-early")]),
    ("gpu-check-ctf-mild", "copy-then-flip 2 -> 4 at 40K (load 315): GPU models",
     [("4x PCIe 40GB (GPUs 3-6)", "2to4-mild/copy-then-flip"), ("2x SXM4 80GB + 2x PCIe 40GB", "2to4/copy-then-flip-mild")]),
    ("gpu-check-2to4to2", "2 -> 4 -> 2 lazy at 40K: GPU models",
     [("4x PCIe 40GB (GPUs 3-6)", "2to4to2"), ("2x SXM4 80GB + 2x PCIe 40GB", "2to4to2/lazy")]),
]


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def ticks(lo, hi, n=5):
    span = hi - lo
    if span <= 0:
        return [lo]
    p = 10 ** math.floor(math.log10(span / n))
    step = next(m * p for m in (1, 2, 2.5, 5, 10) if span / (m * p) <= n)
    t, out = math.ceil(lo / step) * step, []
    while t <= hi + 1e-9:
        out.append(round(t, 6))
        t += step
    return out


def fmt(v):
    if abs(v) >= 1000 and v == int(v):
        return f"{int(v):,}"
    return f"{v:g}"


class Panel:
    """One chart area inside an SVG: x/y ranges, axes, lines, bars, points, labels."""

    def __init__(self, svg, x, y, w, h, xr, yr, title="", xlabel="", ylabel="", xticks=None, yticks=None,
                 xfmt=fmt, yfmt=fmt):
        self.s, self.x, self.y, self.w, self.h, self.xr, self.yr = svg, x, y, w, h, xr, yr
        s = svg
        s.append(f'<text x="{x}" y="{y - 12}" class="t">{esc(title)}</text>')
        for v in (yticks or ticks(*yr)):
            py = self.py(v)
            s.append(f'<line x1="{x}" x2="{x + w}" y1="{py}" y2="{py}" class="g"/>')
            s.append(f'<text x="{x - 6}" y="{py + 4}" class="l" text-anchor="end">{esc(yfmt(v))}</text>')
        if xticks is not False:
            for v in (xticks or ticks(*xr)):
                px = self.px(v)
                s.append(f'<text x="{px}" y="{y + h + 16}" class="l" text-anchor="middle">{esc(xfmt(v))}</text>')
        s.append(f'<line x1="{x}" x2="{x + w}" y1="{y + h}" y2="{y + h}" class="a"/>')
        if xlabel:
            s.append(f'<text x="{x + w / 2}" y="{y + h + 34}" class="l" text-anchor="middle">{esc(xlabel)}</text>')
        if ylabel:
            s.append(f'<text x="{x - 46}" y="{y + h / 2}" class="l" text-anchor="middle" '
                     f'transform="rotate(-90 {x - 46} {y + h / 2})">{esc(ylabel)}</text>')

    def px(self, v):
        return self.x + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w

    def py(self, v):
        v = min(max(v, self.yr[0]), self.yr[1])
        return self.y + self.h - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h

    def line(self, pts, color, width=2, dash="", marker=False):
        if not pts:
            return
        d = " ".join(f"{self.px(a):.1f},{self.py(b):.1f}" for a, b in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.s.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{width}"{da}/>')
        if marker:
            for a, b in pts:
                self.s.append(f'<circle cx="{self.px(a):.1f}" cy="{self.py(b):.1f}" r="3.2" fill="{color}"/>')

    def bar(self, x0, x1, v, color, label=None):
        top, base = self.py(v), self.py(self.yr[0])
        self.s.append(f'<rect x="{self.px(x0):.1f}" y="{top:.1f}" width="{self.px(x1) - self.px(x0):.1f}" '
                      f'height="{max(base - top, 0.5):.1f}" fill="{color}"/>')
        if label is not None:
            self.s.append(f'<text x="{(self.px(x0) + self.px(x1)) / 2:.1f}" y="{top - 5:.1f}" class="v" '
                          f'text-anchor="middle">{esc(label)}</text>')

    def text(self, a, b, s, anchor="start", cls="l", color=None, dy=0):
        c = f' style="fill:{color}"' if color else ""  # a style attribute: the class's fill would win over fill=
        self.s.append(f'<text x="{self.px(a):.1f}" y="{self.py(b) + dy:.1f}" class="{cls}" '
                      f'text-anchor="{anchor}"{c}>{esc(s)}</text>')

    def vband(self, a, b, color, opacity=0.12):
        self.s.append(f'<rect x="{self.px(a):.1f}" y="{self.y}" width="{self.px(b) - self.px(a):.1f}" '
                      f'height="{self.h}" fill="{color}" opacity="{opacity}"/>')


def legend(svg, x, y, items):
    for i, (label, color, style) in enumerate(items):
        yy = y + i * 18
        if style == "hollow":
            svg.append(f'<rect x="{x}" y="{yy - 9}" width="14" height="10" fill="none" stroke="{color}" stroke-width="2"/>')
        elif style == "dash":
            svg.append(f'<line x1="{x}" x2="{x + 16}" y1="{yy - 4}" y2="{yy - 4}" stroke="{color}" stroke-width="2" stroke-dasharray="5 3"/>')
        elif style != "none":
            svg.append(f'<rect x="{x}" y="{yy - 9}" width="14" height="10" fill="{color}"/>')
        svg.append(f'<text x="{x + 22}" y="{yy}" class="l">{esc(label)}</text>')


def notes(svg, x, y, lines):
    for i, n in enumerate(lines):
        svg.append(f'<text x="{x}" y="{y + i * 16}" class="l">{esc(n)}</text>')


def write(name, w, h, body, title, subtitle=""):
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif">'
            '<style>.t{font-size:13px;font-weight:bold;fill:#222}.l{font-size:11px;fill:#444}'
            '.v{font-size:10px;fill:#333}.g{stroke:#e4e4e4}.a{stroke:#999}'
            '.h{font-size:17px;font-weight:bold;fill:#111}.s{font-size:12px;fill:#555}</style>'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>'
            f'<text x="24" y="30" class="h">{esc(title)}</text>'
            f'<text x="24" y="50" class="s">{esc(subtitle)}</text>')
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        f.write(head + "".join(body) + "</svg>\n")
    print("wrote", path)


def load(path):
    with open(path) as f:
        return json.load(f)


def run(rel):
    return os.path.join(RUNS, rel)


def have(*rels):
    return all(os.path.exists(os.path.join(run(r), "latency.csv")) for r in rels)


# ---------------------------------------------------------------------------------------------
# compare_runs.py timelines

def timelines():
    for name, title, pairs in PAIRS:
        if not have(*(d for _, d in pairs)):
            print("skipped", name, "(runs missing)")
            continue
        subprocess.run([sys.executable, COMPARE] + [run(d) for _, d in pairs] +
                       ["--labels", ",".join(l for l, _ in pairs), "--window", "1",
                        "--out", os.path.join(OUT, name), "--title", title], check=True, stdout=subprocess.DEVNULL)
        print("wrote", os.path.join(OUT, name + ".svg"))


def compared(name):
    """{label: compare_runs.py summary} of one timeline."""
    return {x["label"]: x for x in load(os.path.join(OUT, name + ".json"))}


# ---------------------------------------------------------------------------------------------
# Capacity: answered vs offered for every calibration, and the plateau per read budget

def calibrations():
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, "calibrate-*"))):
        if not os.path.exists(os.path.join(d, "calibration.json")):
            continue
        c = load(os.path.join(d, "experiment.json"))
        r = c.get("NodeResources", {})
        rows = load(os.path.join(d, "calibration.json"))
        out.append({
            "dir": os.path.basename(d), "nodes": c.get("InitialNodes", c.get("Nodes")),
            "budget": r.get("ReadIOPSPerNode", 0), "cpus": r.get("CPUsPerNode"),
            "rows": [(x["offered_per_s"], x["answered_in_window_per_s"], x["p99_us"] / 1000) for x in rows],
            "plateau": max(x["answered_in_window_per_s"] for x in rows),
        })
    return out


def calibrated(nodes, budget):
    """Peak throughput of the calibrations with that many nodes and that budget (mean of repeats)."""
    v = [c["plateau"] for c in calibrations() if c["nodes"] == nodes and c["budget"] == budget]
    if not v and nodes == 2 and budget == 40000:
        return C2_40K
    return sum(v) / len(v) if v else None


def c2_from_run(budget):
    """C2 read off the copy-then-flip run at that budget: while the load exceeds C2 and the copy
    has not flipped, the two old nodes answer at their capacity (within ~1% of the calibrations
    at 40K-80K: 328 / 490 / 653 against 330 / 496 / 660)."""
    for b, _, ctf, _ in BUDGETS:
        if b == budget and have(ctf):
            dur = load(os.path.join(run(ctf), "experiment.json"))["Load"]["DurationSeconds"]
            ans = per_second_answers(run(ctf))
            return sum(ans.get(s, 0) for s in range(150, dur - 10)) / (dur - 160)
    return None


def capacity(nodes, budget):
    """The calibrated capacity, else an estimate (estimated() says which): C2 from the
    copy-then-flip run, C4 from C2 and the 4-node / 2-node ratio measured at the other budgets."""
    v = calibrated(nodes, budget)
    if v or nodes not in (2, 4):
        return v
    c2 = calibrated(2, budget) or c2_from_run(budget)
    if nodes == 2 or not c2:
        return c2
    r = [calibrated(4, b) / calibrated(2, b) for b, *_ in BUDGETS if calibrated(4, b) and calibrated(2, b)]
    return c2 * sum(r) / len(r) if r else None


def estimated(nodes, budget):
    return calibrated(nodes, budget) is None


def cap_label(nodes, budget):
    v = capacity(nodes, budget)
    return "?" if v is None else (f"~{v:,.0f} (est.)" if estimated(nodes, budget) else f"{v:,.0f}")


def fig_capacity():
    cal = calibrations()
    body = []

    def label(c):
        if c["budget"] >= 1_000_000:
            return f'{c["nodes"]} nodes, no budget, {c["cpus"]} CPUs'
        return f'{c["nodes"]} nodes, {c["budget"] // 1000}K reads/s'

    keys = list(dict.fromkeys(label(x) for x in cal))
    p = Panel(body, 80, 90, 520, 330, (0, 3100), (0, 2800), "answered within the sending window vs offered",
              "offered load (q/s)", "answered (q/s)")
    p.line([(0, 0), (2800, 2800)], "#bbbbbb", 1, "4 3")
    seen = set()
    for c in cal:
        key = label(c)
        color = PALETTE[keys.index(key) % len(PALETTE)]
        p.line([(a, b) for a, b, _ in c["rows"]], color, 2, "5 3" if key in seen else "", marker=True)
        seen.add(key)
    legend(body, 620, 100, [(k, PALETTE[i % len(PALETTE)], "solid") for i, k in enumerate(keys)] +
           [("dashed: second run of the same setting", GREY, "dash")])

    budgets = sorted(b for b, *_ in BUDGETS)
    q = Panel(body, 700, 330, 330, 210, (30, budgets[-1] / 1000 + 10), (0, 1400), "peak throughput vs per-node read budget",
              "read budget per node (K reads/s)", "q/s", xticks=[b // 1000 for b in budgets])
    for nodes, color in ((2, ORANGE), (4, BLUE)):
        by = {}
        for c in cal:
            if c["nodes"] == nodes and c["budget"] < 1e6:
                by.setdefault(c["budget"] / 1000, []).append(c["plateau"])
        if nodes == 2:
            by.setdefault(40, []).append(C2_40K)
        pts = sorted((a, sum(v) / len(v)) for a, v in by.items())
        q.line(pts, color, 2, marker=True)
        for a, v in sorted(by.items()):
            s = f"{min(v):.0f}" if len(v) == 1 else (f"{min(v):.0f}" if round(min(v)) == round(max(v)) else
                                                    f"{min(v):.0f}-{max(v):.0f}") + f" ({len(v)} runs)"
            q.text(a, sum(v) / len(v), s, "middle", "v", dy=-8)
        # Budgets without a calibration: the estimate, hollow, joined to the measured line by a dashed one.
        for b in budgets:
            if b / 1000 in by or not capacity(nodes, b):
                continue
            v = capacity(nodes, b)
            q.line([pts[-1], (b / 1000, v)], color, 1.5, "4 3")
            q.s.append(f'<circle cx="{q.px(b / 1000):.1f}" cy="{q.py(v):.1f}" r="4" fill="#fff" stroke="{color}" stroke-width="2"/>')
            q.text(b / 1000, v, f"~{v:.0f} ({'from the run' if nodes == 2 else 'est.'})", "end", "v", dy=-8)
    q.text(31, 1350, "4 nodes", color=BLUE)
    q.text(31, 1260, "2 nodes (40K: earlier run)", color=ORANGE)
    q.text(31, 1170, "hollow: no calibration yet, estimated", cls="v")
    ratios = ", ".join(f"{b // 1000}K {calibrated(4, b) / calibrated(2, b):.2f}x" for b in budgets
                       if calibrated(4, b) and calibrated(2, b))
    body.append(f'<text x="700" y="600" class="l">4 nodes / 2 nodes: {esc(ratios)}</text>')
    body.append('<text x="700" y="616" class="l">(the busiest node serves 61% of the reads of 2 nodes, 38% of 4)</text>')
    est = [b for b in budgets if estimated(2, b) or estimated(4, b)]
    if est:
        body.append(f'<text x="700" y="632" class="l">{esc(", ".join(f"{b // 1000}K" for b in est))}: C2 from the copy-then-flip run '
                    f'(answers while the load exceeds C2),</text>')
        body.append('<text x="700" y="648" class="l">C4 = C2 x the 4-node / 2-node ratio measured at the other budgets</text>')
    write("capacity.svg", 1100, 665, body, "Capacity of the wiki-c1024 cluster (A100, 8 CPUs per node, NProbe 64, N 200)",
          "each point: 30 s at one offered rate; the plateau is where the busiest node's read budget (or its CPUs, or the shared disk) runs out")


# ---------------------------------------------------------------------------------------------
# lazy vs copy-then-flip: grouped bars

def protocol_bars(rows, name, title, subtitle, foot):
    """rows: [(x label, x sublabel, lazy summary, copy-then-flip summary, repeat summary or None)]."""
    body = []
    panels = [
        ("rescale time (s)", lambda x: x["rescales"][0]["elapsed_s"], "{:.0f}"),
        ("answered > 200 ms after the trigger (%)", lambda x: 100 * x["open_loop"]["slo_miss_fraction"], "{:.1f}%"),
        ("peak backlog (queries)", lambda x: x["open_loop"]["peak_backlog"], "{:,.0f}"),
    ]
    pw = max(280, 92 * len(rows))  # ~90 px per group keeps the group labels apart
    for k, (ptitle, get, f) in enumerate(panels):
        top = max(get(x) for r in rows for x in r[2:] if x)
        yr = (0, 100) if "%" in f else (0, next(t for t in ticks(0, top * 1.6) if t >= top * 1.08) if top > 0 else 1)
        p = Panel(body, 80 + k * (pw + 60), 100, pw, 300, (-0.5, len(rows) - 0.5), yr, ptitle, xticks=False)
        for i, (xl, xs, lz, cf, rep) in enumerate(rows):
            p.bar(i - 0.36, i - 0.02, get(lz), BLUE, f.format(get(lz)))
            p.bar(i + 0.02, i + 0.36, get(cf), ORANGE, f.format(get(cf)))
            if rep:
                v = get(rep)
                p.s.append(f'<line x1="{p.px(i + 0.02):.1f}" x2="{p.px(i + 0.36):.1f}" y1="{p.py(v):.1f}" '
                           f'y2="{p.py(v):.1f}" stroke="#222" stroke-width="2" stroke-dasharray="4 2"/>')
            p.text(i, yr[0], xl, "middle", "l", dy=16)
            p.text(i, yr[0], xs, "middle", "v", dy=29)
    items = [("lazy", BLUE, "solid"), ("copy-then-flip", ORANGE, "solid")]
    if any(r[4] for r in rows):
        items.append(("copy-then-flip, repeat on other GPUs (2x SXM4 80GB + 2x PCIe)", "#222", "dash"))
    legend(body, 80, 470, items)
    notes(body, 80, 545, foot)
    write(name, max(1100, 80 + 3 * (pw + 60)), 545 + 16 * len(foot) + 10, body, title, subtitle)


def fig_protocols_vs_load():
    rows = []
    for name, rate, check in (("2to4-40k-early", 265, "gpu-check-ctf-early"), ("2to4-40k-mild", 315, "gpu-check-ctf-mild"),
                              ("2to4-40k-full", 420, None)):
        if not os.path.exists(os.path.join(OUT, name + ".json")):
            return
        j = compared(name)
        rep = load(os.path.join(OUT, check + ".json"))[1] if check and os.path.exists(os.path.join(OUT, check + ".json")) else None
        rows.append((f"{rate} q/s", f"{rate / C2_40K:.0%} of C2", j["lazy"], j["copy-then-flip"], rep))
    protocol_bars(rows, "protocols-vs-load.svg", "lazy vs copy-then-flip, 2 -> 4 at 40K reads/s, three load levels",
                  "one run per protocol and load; the dashed line is a second copy-then-flip run on other GPUs: a measure of run-to-run spread",
                  ["copy-then-flip's rescale ended after the 360 s load in all three (352 s, 437 s, 624 s after the trigger at 90 s).",
                   "Load 210 q/s, stepped at 60 s to the rate shown; 2 -> 4 at 90 s; 40K reads/s, 8 CPUs, 24 GB, one A100 per node.",
                   f"C2 = 2-node capacity, about {C2_40K} q/s. The early and mild pairs ran on GPUs 3-6 (4x PCIe 40GB), the 420 q/s pair on the PCIe cards as numbered on 2026-10-07."])


BUDGETS = [(40000, "2to4-40k-full", "2to4/copy-then-flip", "2to4"), (60000, "2to4-60k-full", "2to4-60k/copy-then-flip", "2to4-60k"),
           (80000, "2to4-80k-full", "2to4-80k/copy-then-flip", "2to4-80k"),
           (100000, "2to4-100k-full", "2to4-100k/copy-then-flip", "2to4-100k")]


def budgets_present():
    """The BUDGETS whose pair of runs is there (and was compared)."""
    return [b for b in BUDGETS if os.path.exists(os.path.join(OUT, b[1] + ".json")) and have(b[2])]


def names(bs):
    return " / ".join(f"{b // 1000}K" for b, *_ in bs)


def step_rate(d):
    """(base rate, rate after the 60 s step) of a run's load."""
    L = load(os.path.join(run(d), "experiment.json"))["Load"]
    return float(L["Rate"]), float(str(L["RateSteps"]).split(",")[0].split(":")[1])


def fig_protocols_vs_budget():
    bs = budgets_present()
    if not bs:
        return
    rows, lz_rs, cf_rs = [], [], []
    for budget, name, ctf, _ in bs:
        j = compared(name)
        base, peak = step_rate(ctf)
        rows.append((f"{budget // 1000}K: {peak:.0f} q/s", f"{peak / capacity(2, budget):.0%} of C2",
                     j["lazy"], j["copy-then-flip"], None))
        lz_rs.append(j["lazy"]["rescales"][0]["elapsed_s"])
        cf_rs.append(j["copy-then-flip"]["rescales"][0]["elapsed_s"])
    caps = ", ".join(f"{b // 1000}K {cap_label(2, b)} / {cap_label(4, b)}" for b, *_ in bs)
    loads = ", ".join(f"{b // 1000}K {step_rate(c)[0]:.0f} -> {step_rate(c)[1]:.0f}" for b, _, c, _ in bs)
    protocol_bars(rows, "protocols-vs-budget.svg", f"lazy vs copy-then-flip under overload, 2 -> 4 at {names(bs)} reads/s per node",
                  "one run per protocol and budget; load stepped at 60 s above the 2-node capacity C2 (below the 4-node capacity C4); 2 -> 4 at 90 s",
                  [f"copy-then-flip's rescale took {min(cf_rs):.0f}-{max(cf_rs):.0f} s at every budget: while the load exceeds C2 the hot source's budget goes to",
                   f"the queries and its copy stalls; the rest is copied after the 360 s load has ended and its backlog is answered. lazy: {min(lz_rs):.0f}-{max(lz_rs):.0f} s.",
                   f"Capacities C2 / C4 (q/s): {caps}.",
                   f"Loads (q/s, step at 60 s): {loads}."])


# ---------------------------------------------------------------------------------------------
# copy-then-flip's raw-vector copy per source over time

def source_series(d, n):
    """(t since the load started, RAW_GETs served so far) of source n, one per stats sample."""
    z = load(os.path.join(d, "clock.json"))["load_start"]
    with open(os.path.join(d, f"stats-{n}.jsonl")) as f:
        L = [json.loads(l) for l in f if l.strip()]
    c0 = L[0]["ops"]["raw_get"]["count"]
    return [(x["t"] - z, x["ops"]["raw_get"]["count"] - c0) for x in L]


def mb_per_get(d):
    """MB of raw vectors per RAW_GET at the sources of copy-then-flip run d (whose sources serve
    no other RAW_GET): a 64-vector batch (BATCH_MB) with the old stream, ~1 MiB on the data node
    path."""
    gets = sum(source_series(d, n)[-1][1] for n in ("n1", "n2"))
    raw = load(os.path.join(d, "reconfigurations.json"))[0]["reply"]["raw_bytes"]
    return raw / 1e6 / gets if gets else BATCH_MB


def old_stream(d):
    return abs(mb_per_get(d) - BATCH_MB) < 0.01


def source_streams(d):
    """Per source of the raw copy: (name, MB/s while the load runs, seconds a RAW_GET takes, budget
    wait in reader-seconds per second, when its stream ended, charged reads/s, GB), hot source
    first. The rate is taken from 20 s after the stream started until it or the load ended."""
    z = load(os.path.join(d, "clock.json"))["load_start"]
    dur = load(os.path.join(d, "experiment.json"))["Load"]["DurationSeconds"]
    per = mb_per_get(d)
    out = []
    for n in ("n1", "n2"):
        with open(os.path.join(d, f"stats-{n}.jsonl")) as f:
            L = [json.loads(l) for l in f if l.strip()]
        g = [(x["t"] - z, x["ops"]["raw_get"]["count"], x["reads"]["charged"], x["reads"]["wait_us"],
              x["ops"]["raw_get"]["busy_us"]) for x in L]
        total = g[-1][1] - g[0][1]
        t0 = next(t for t, c, *_ in g if c > g[0][1])
        t1 = next(t for t, c, *_ in g if c - g[0][1] >= total)
        w = [x for x in g if t0 + 20 <= x[0] <= min(t1, dur) - 5]
        dt = w[-1][0] - w[0][0]
        gets = (w[-1][1] - w[0][1]) / dt
        busy = (w[-1][4] - w[0][4]) / 1e6 / max(w[-1][1] - w[0][1], 1)  # seconds one RAW_GET takes at the source
        out.append((n, gets * per, busy, (w[-1][3] - w[0][3]) / dt / 1e6, t1,
                    (w[-1][2] - w[0][2]) / dt, total * per / 1000))
    return sorted(out, key=lambda x: -x[5])


def copy_figure(cols, name, title, subtitle, foot):
    """cols: [(panel title, run dir)], one column each: MB/s per source and GB copied over time."""
    body, win = [], 10
    ends = max(s[4] for _, d in cols for s in source_streams(run(d)))
    xr = (80, max(730, 10 * math.ceil((ends + 15) / 10)))  # one time axis for every column
    series = {}  # (column, source) -> MB/s over win-second windows
    for k, (_, d) in enumerate(cols):
        path, per = run(d), mb_per_get(run(d))
        for n, *_ in source_streams(path):
            ser, t_end = source_series(path, n), next(s[4] for s in source_streams(path) if s[0] == n)
            series[k, n] = [((ser[i][0] + ser[i + win][0]) / 2, (ser[i + win][1] - ser[i][1]) * per / (ser[i + win][0] - ser[i][0]))
                            for i in range(0, len(ser) - win, win) if xr[0] <= ser[i][0] <= xr[1] and ser[i][0] <= t_end]
    top = max(v for pts in series.values() for _, v in pts)
    ymax = next(t for t in (100, 150, 200, 250, 300, 400, 500, 800, 1000) if t >= top * 1.05)  # the same for every column
    for k, (ptitle, d) in enumerate(cols):
        path = run(d)
        dur = load(os.path.join(path, "experiment.json"))["Load"]["DurationSeconds"]
        streams, per = source_streams(path), mb_per_get(path)
        x0 = 80 + k * 340
        xt = [90, dur, round(max(s[4] for s in streams))]
        pr = Panel(body, x0, 100, 290, 170, xr, (0, ymax), ptitle, xticks=xt,
                   ylabel="MB/s per source" if k == 0 else "")
        pc = Panel(body, x0, 320, 290, 170, xr, (0, 28), "copied so far (GB)", "seconds since the load started",
                   xticks=xt, ylabel="GB" if k == 0 else "")
        for p in (pr, pc):
            p.vband(dur, xr[1], GREY, 0.08)
        pr.text(dur + 6, ymax * 0.95, "load over", cls="v")
        if old_stream(path):
            pr.line([(xr[0], 80), (xr[1], 80)], GREY, 1.2, "5 3")
        for (n, mbps, sec, wait, t_end, _, gb), color, role in zip(streams, (RED, BLUE), ("hot", "cold")):
            ser = source_series(path, n)
            pr.line(series[k, n], color, 2)
            pc.line([(t, c * per / 1000) for t, c in ser if xr[0] <= t <= min(t_end + 5, xr[1])], color, 2)
            pc.text(t_end, gb, f"done {t_end:.0f} s", "end" if t_end > 600 else "start", "v", color, dy=-6)
            body.append(f'<text x="{x0}" y="{548 + (role == "cold") * 15}" class="v" style="fill:{color}">'
                        f'{role} {n}: {mbps:.0f} MB/s under load, RAW_GET {sec * 1000:.1f} ms each, wait {wait:.1f} s/s</text>')
    items = [("hot source n2 (~60% of the query reads), copies 26.6 GB to n4", RED, "solid"),
             ("cold source n1, copies 24.6 GB to n3", BLUE, "solid")]
    if any(old_stream(run(d)) for _, d in cols):
        items.append(("~80 MB/s: the old stream's own bound under load, one 64-vector (256 KB) request in flight per source", GREY, "dash"))
    legend(body, 80, 605, items)
    notes(body, 80, 675, foot)
    write(name, max(1100, 80 + 340 * len(cols) + 20), 675 + 16 * len(foot) + 12, body, title, subtitle)


def fig_copy_rate():
    cols = [(f"{r} q/s ({r / C2_40K:.0%} of C2)", d) for r, d in
            ((265, "2to4-early/copy-then-flip"), (315, "2to4-mild/copy-then-flip"), (420, "2to4/copy-then-flip"))]
    if not have(*(d for _, d in cols)):
        return
    copy_figure(cols, "copy-rate.svg", "copy-then-flip at 40K reads/s per node: the raw-vector copy of each source over time",
                "2 -> 4 at 90 s, load 210 q/s stepped at 60 s to the rate shown; rates over 10 s windows; MB/s = PULL_RAW batches x 64 vectors x 4 KB",
                ["265 q/s: both sources have read budget to spare; both streams run at the stream's own bound (an implementation limit, to be fixed).",
                 "315 q/s: the hot source is at its 40K reads/s; each batch waits for read tokens behind the queries: ~50 MB/s while the load runs.",
                 "420 q/s: the hot source is saturated by queries; its stream nearly stops (6 MB/s) until the load has ended and the 2 old nodes have",
                 "              answered their backlog (27,675 queries at 360 s, the last answered at 445 s); then it copies the rest at ~90 MB/s.",
                 "The cold source runs at the bound throughout. The rescale waits for the slower stream: 352 s, 437 s and 624 s after the trigger.",
                 "Without queries (after the load) a stream reaches ~85-95 MB/s."])


def fig_copy_rate_stream_fix():
    cols = [("old stream: 1 x 64 KB via the agents", "2to4-60k/copy-then-flip"),
            ("data node path: 4 x 1 MiB", "2to4-60k/copy-then-flip-optimize")]
    if not have(*(d for _, d in cols)):
        return
    (ho, co), (hn, cn) = (source_streams(run(d)) for _, d in cols)
    ro, rn = (load(os.path.join(run(d), "reconfigurations.json"))[0]["elapsed_seconds"] for _, d in cols)
    copy_figure(cols, "copy-rate-60k-stream-fix.svg",
                "copy-then-flip at 60K reads/s per node: the raw-vector copy before and after the stream fix",
                "2 -> 4 at 90 s, load 325 q/s stepped at 60 s to 650 (131% of C2); rates over 10 s windows",
                [f"Cold source: {co[1]:.0f} -> {cn[1]:.0f} MB/s ({cn[1] / co[1]:.1f}x), done at {co[4]:.0f} -> {cn[4]:.0f} s. "
                 f"Hot source while overloaded: {ho[1]:.0f} -> {hn[1]:.0f} MB/s ({hn[1] / ho[1]:.1f}x).",
                 "The hot source stays starved by the queries: each request now carries 4x the vectors and waits ~4x as long for read tokens, so four in flight copy ~2x faster,",
                 f"and copy-then-flip still cannot flip while the load lasts (rescale {ro:.0f} -> {rn:.0f} s, ended after the load).",
                 "Queries during the load: unchanged (same backlog and latency): the copy's extra reads barely reach the budget."])


def fig_copy_rate_budgets():
    bs = budgets_present()
    if not bs:
        return
    cols, hot, rs = [], [], []
    for budget, name, ctf, _ in bs:
        base, peak = step_rate(ctf)
        cols.append((f"{budget // 1000}K, {peak:.0f} q/s ({peak / capacity(2, budget):.0%} of C2)", ctf))
        hot.append(source_streams(run(ctf))[0])
        rs.append(compared(name)["copy-then-flip"]["rescales"][0]["elapsed_s"])
    copy_figure(cols, "copy-rate-budgets.svg",
                f"copy-then-flip under overload at {names(bs)} reads/s per node: the raw-vector copy of each source",
                "2 -> 4 at 90 s; load stepped at 60 s above the 2-node capacity C2; rates over 10 s windows; MB/s = PULL_RAW batches x 64 vectors x 4 KB",
                ["At every budget the hot source is saturated by queries while the load exceeds C2. A larger budget mostly serves more queries:",
                 f"the hot stream gets {' / '.join(f'{h[1]:.0f}' for h in hot)} MB/s at {names(bs)} "
                 f"(each batch waits {' / '.join(f'{h[2] * 1000:.0f}' for h in hot)} ms for read tokens).",
                 "Its stream resumes only once the load has ended and the old nodes have answered their backlog; the cold source runs at the bound.",
                 f"So under sustained overload copy-then-flip's rescale is set by when the overload ends, not by the budget ({min(rs):.0f}-{max(rs):.0f} s here)."])


# ---------------------------------------------------------------------------------------------
# The stall right after the trigger, common to both protocols: writeback on the shared disk

def per_second_answers(d):
    ans = {}
    with open(os.path.join(d, "latency.csv")) as f:
        for r in csv.DictReader(f):
            if r["status"] == "ok":
                t = int((float(r["sched_ms"]) + int(r["latency_us"]) / 1000) / 1000)
                ans[t] = ans.get(t, 0) + 1
    return ans


def host_series(d):
    z = load(os.path.join(d, "clock.json"))["load_start"]
    H = []
    with open(os.path.join(d, "cgroups.jsonl")) as f:
        for l in f:
            if l.strip():
                h = json.loads(l)
                if h.get("node") == "host" and "disk" in h:
                    H.append(h)
    return [((b["t"] + a["t"]) / 2 - z, (b["disk"]["wbytes"] - a["disk"]["wbytes"]) / (b["t"] - a["t"]) / 1e6,
             (b["disk"]["rios"] - a["disk"]["rios"]) / (b["t"] - a["t"]) / 1000,
             (b["disk"]["busy_ms"] - a["disk"]["busy_ms"]) / (b["t"] - a["t"]) / 10) for a, b in zip(H, H[1:])]


def fig_trigger_stall():
    if not have("2to4-mild/lazy", "2to4-mild/copy-then-flip"):
        return
    body = []
    for k, (proto, color) in enumerate((("lazy", BLUE), ("copy-then-flip", ORANGE))):
        d = run(os.path.join("2to4-mild", proto))
        stage = load(os.path.join(d, "reconfigurations.json"))[0]["reply"]["phases"]["stage_data"]
        hs = [x for x in host_series(d) if 80 <= x[0] <= 130]
        ans = per_second_answers(d)
        y0 = 90 + k * 300
        p1 = Panel(body, 90, y0, 900, 100, (80, 130), (0, 2200), f"{proto}: disk writes (MB/s) and disk busy (%)", xticks=False)
        p1.vband(90, 90 + stage, GREY)
        p1.line([(t, w) for t, w, _, _ in hs], RED, 2)
        p1.line([(t, b * 22) for t, _, _, b in hs], GREY, 1.5, "4 2")
        p1.text(90.5, 2050, "staging lists + PQ", cls="v")
        p1.text(129.5, 2050, "red: writes MB/s; grey dashed: busy % (x22)", "end", "v")
        p2 = Panel(body, 90, y0 + 140, 900, 100, (80, 130), (0, 450), f"{proto}: answered q/s (offered 315) and disk reads (K/s)",
                   "seconds since the load started" if k else "")
        p2.vband(90, 90 + stage, GREY)
        p2.line([(t + 0.5, ans.get(t, 0)) for t in range(80, 130)], color, 2)
        p2.line([(t, r * 5) for t, _, r, _ in hs], GREEN, 1.5, "4 2")
        p2.text(129.5, 420, "green dashed: disk reads K/s (x5)", "end", "v")
    notes(body, 90, 690, [
        "Both runs: 210 -> 315 q/s at 60 s (95% of the 2-node capacity), 2 -> 4 at 90 s, 40K reads/s per node, all four nodes on one shared data disk.",
        "Writeback of what the new nodes stage and load (graph 2 x 7 GB, PQ, lists) comes in ~2 GB/s bursts: the disk is 85-90% busy,",
        "and the old owners' query reads drop from ~65K to ~20K/s. On a real cluster the new nodes write to their own disks:",
        "this stall is a single-machine artifact, the same for both protocols."])
    write("trigger-stall.svg", 1080, 756, body, "The dip right after the trigger: writeback bursts on the shared disk",
          "mild load, 1 s windows; grey band: staging of posting lists and PQ codes")


def main():
    global RUNS, OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default=RUNS, help=f"the downloaded runs (default {RUNS})")
    ap.add_argument("--out", default=OUT, help=f"where the figures go (default {OUT})")
    args = ap.parse_args()
    RUNS, OUT = args.runs, args.out
    os.makedirs(OUT, exist_ok=True)
    timelines()
    fig_capacity()
    fig_protocols_vs_load()
    fig_protocols_vs_budget()
    fig_copy_rate()
    fig_copy_rate_budgets()
    fig_copy_rate_stream_fix()
    fig_trigger_stall()
    if shutil.which("rsvg-convert"):
        for svg in sorted(glob.glob(os.path.join(OUT, "*.svg"))):
            subprocess.run(["rsvg-convert", "-w", "1400", "-b", "white", svg, "-o", svg[:-4] + ".png"], check=True)
        print("wrote a PNG of every SVG")


if __name__ == "__main__":
    main()
