#!/usr/bin/env python3
"""Overlay the mean latency over time of several runs: one line per run (e.g. per protocol).

    python3 scripts/compare_runs.py results/e2e/lazy results/e2e/copy-then-flip \\
        results/e2e/stop-and-copy --window 0.02 --out results/e2e/latency-mean
    python3 scripts/compare_runs.py results/e2e/rep*/lazy results/e2e/rep*/copy-then-flip \\
        results/e2e/rep*/stop-and-copy --window 0.02      # repetitions pooled per protocol

Each directory holds a latency.csv and, optionally, a reconfigurations.json, as written by
run_local.py (one run per protocol, same config) or by TestReconfigUnderLoad with
RTIER_E2E_TIMELINE set. A run is labelled with its directory name unless --labels is given,
and runs that share a label are pooled into one line: their queries are merged, so a window's
mean covers every run's queries sent in it. Pool only runs with the same schedule (the
reconfigurations start within about a window of each other). Writes:

  OUT.svg   the mean latency of the answered queries per window, one line per label, and under
            it one strip per label marking the reconfigurations of each of its runs: runs need
            not trigger them at the same moment (the end-to-end test starts each one when the
            previous returned)
  OUT.csv   the same windows as numbers (the chart's table view)
  OUT.json  per label: the mean latency before the first reconfiguration, during each one (the
            queries each run sent while its own reconfiguration ran), after the last one and
            over the whole run; pooled over the runs, with the lowest and highest run

A query's latency counts in the window of the time it was sent: its scheduled time with the
open-loop load generator, its first attempt in the closed-loop test; answered queries per second
count in the window of their answer. With the open-loop load (no "users" in latency.csv) the
figure shows the offered rate instead of the users, and the backlog: queries sent and not
answered yet. It grows whenever the offered rate exceeds what the cluster answers, and after a
scale-out it drains only with the capacity left over -- what a closed loop hides, since its
users just wait. OUT.json then also has the peak backlog, when it drained, and the queries
answered later than --slo-ms (failed ones included) among those sent from the first
reconfiguration on. --skip drops the queries sent in the
first seconds of each run (warm-up: the end-to-end test's first queries meet a cold start that
has nothing to do with reconfiguration). Means cover answered queries only,
so they compare runs only when no query failed: a refused query that gave up is missing from
them (see "RetryForSeconds" in run_local.py; the end-to-end test's clients always wait).

No dependencies: the SVG is written here, in light and dark.
"""

import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_run import esc, fmt, nice_ticks, read_latency, reconfigurations, windows  # noqa: E402

# Categorical slots of the reference palette in their fixed order, (light, dark); validated
# as a set (adjacent pairs, both modes; the first three all-pairs too).
PALETTE = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"),
           ("#eda100", "#c98500"), ("#e87ba4", "#d55181"), ("#008300", "#008300"),
           ("#4a3aa7", "#9085e9"), ("#e34948", "#e66767")]
# A protocol keeps its colour in every figure, whichever runs are compared.
FIXED_SLOT = {"lazy": 0, "copy-then-flip": 1, "stop-and-copy": 2, "lazy-stream": 3}

W, LEFT, RIGHT, TOP, PLOT_H = 900, 64, 132, 60, 260
STRIP_ROW, STRIP_GAP = 20, 64  # one strip row per label, below the x axis


def slots(labels):
    """Palette slot per label: the fixed one for a known protocol, else the next unused one."""
    used = {FIXED_SLOT[l] for l in labels if l in FIXED_SLOT}
    free = [i for i in range(len(PALETTE)) if i not in used]
    out = []
    for l in labels:
        if l in FIXED_SLOT:
            out.append(FIXED_SLOT[l])
        elif free:
            out.append(free.pop(0))
        else:
            sys.exit("more than %d lines: compare fewer at a time" % len(PALETTE))
    return out


def ticks_to(hi):
    """nice_ticks, extended until the top tick clears hi by 5% (the peak stays inside)."""
    ticks = nice_ticks(hi)
    step = ticks[1] - ticks[0] if len(ticks) > 1 else 1.0
    while ticks[-1] < hi * 1.05:
        ticks.append(ticks[-1] + step)
    return ticks


def tick_label(v, step):
    """v with as many decimals as the tick step needs, the same on every tick of an axis."""
    return "%.*f" % (max(0, -int(math.floor(math.log10(step) + 1e-9))), v)


def offered_schedule(d):
    """The open-loop rate schedule of a run as [(from_s, rate)] ("Rate", "RateSteps" in its
    experiment.json), or None."""
    try:
        load = json.load(open(os.path.join(d, "experiment.json"))).get("Load", {})
    except (OSError, ValueError):
        return None
    if load.get("Users") or "Rate" not in load:
        return None
    steps = load.get("RateSteps") or []
    if isinstance(steps, str):
        steps = [x.split(":") for x in steps.split(",") if x]
    return [(0.0, float(load["Rate"]))] + sorted((float(t), float(r)) for t, r in steps)


def rate_at(sched, t):
    r = sched[0][1]
    for t0, v in sched:
        if t >= t0:
            r = v
    return r


def flow(rows, width, n, runs):
    """Per window, averaged over pooled runs: queries answered in it (by the time of the
    answer), and at its end the backlog -- sent but not answered yet (a query that failed
    leaves it when it fails)."""
    done, sent, left = [0] * n, [0] * n, [0] * n
    for r in rows:
        k = int((r["t"] + r["us"] / 1e6) / width)
        if k < n:
            left[k] += 1
            done[k] += r["status"] == "ok"
        sent[min(n - 1, int(r["t"] / width))] += 1
    out, backlog = [], 0
    for k in range(n):
        backlog += sent[k] - left[k]
        out.append({"t": k * width, "answered": done[k] / width / runs, "backlog": backlog / runs})
    return out


def openloop_summary(rows, fl, width, trigger, runs, slo_us):
    """Peak backlog after the first reconfiguration's trigger, when it drained (the first
    window after the peak back within the queries in flight: twice the backlog of the run's
    last fifth plus 10, at least 50 -- a later blip of a few dozen is not a backlog; a run whose
    last fifth still holds over 100 has not drained), and the queries sent from the trigger on
    that were answered later than slo_us or failed."""
    after = [w for w in fl if w["t"] + width > trigger]
    if not after:
        return {}
    tail = sorted(w["backlog"] for w in fl[-max(1, len(fl) // 5):])
    floor = tail[len(tail) // 2]
    limit = max(2 * floor + 10, 50)
    peak = max(after, key=lambda w: w["backlog"])
    drained = None
    if floor > 100:
        pass  # still draining at the end
    elif peak["backlog"] <= limit:
        drained = trigger  # never behind
    else:
        drained = next((w["t"] + width for w in after if w["t"] > peak["t"] and w["backlog"] <= limit), None)
    late = [r for r in rows if r["t"] >= trigger]
    miss = sum(1 for r in late if r["status"] != "ok" or r["us"] > slo_us)
    return {"peak_backlog": round(peak["backlog"]), "peak_backlog_at_s": round(peak["t"] + width, 3),
            "backlog_drained_at_s": round(drained, 3) if drained is not None else None,
            "slo_ms": slo_us / 1000.0, "slo_misses": round(miss / runs),
            "slo_miss_fraction": round(miss / len(late), 5) if late else None}


def mean_of(rows, t0, t1):
    """(mean latency in us, count) of the answered queries sent in [t0, t1)."""
    lat = [r["us"] for r in rows if r["status"] == "ok" and t0 <= r["t"] < t1]
    return (sum(lat) / len(lat) if lat else None), len(lat)


def pooled(parts):
    """Per-run (mean, count) -> the pooled mean, the count, and the lowest and highest run."""
    parts = [(m, n) for m, n in parts if m is not None]
    n = sum(k for _, k in parts)
    if not n:
        return {"mean_us": None, "queries": 0}
    out = {"mean_us": round(sum(m * k for m, k in parts) / n, 1), "queries": n}
    if len(parts) > 1:
        out["min_us"] = round(min(m for m, _ in parts), 1)
        out["max_us"] = round(max(m for m, _ in parts), 1)
    return out


def load(label, dirs, skip):
    """The runs of one label as [(rows sent from `skip` on, reconfigurations)]."""
    runs = []
    for d in dirs:
        path = os.path.join(d, "latency.csv")
        if not os.path.exists(path):
            sys.exit("no latency.csv in %s" % d)
        rows = [r for r in read_latency(path) if r["t"] >= skip]
        if not rows:
            sys.exit("%s has no query sent after %g s" % (path, skip))
        runs.append((rows, reconfigurations(os.path.join(d, "reconfigurations.json"))))
    if len({len(rcs) for _, rcs in runs}) > 1:
        sys.exit("%s: the runs have different numbers of reconfigurations; pool only runs of one schedule"
                 % label)
    return runs


def summarize(label, runs, win, width, skip):
    s = {
        "label": label,
        "runs": len(runs),
        "queries": sum(len(rows) for rows, _ in runs),
        "failed": sum(1 for rows, _ in runs for r in rows if r["status"] != "ok"),
        "retried": sum(1 for rows, _ in runs for r in rows if r["attempts"] > 1),
        "skipped_s": skip,
        "whole_run": pooled([mean_of(rows, skip, math.inf) for rows, _ in runs]),
        "before": pooled([mean_of(rows, skip, rcs[0][0] if rcs else math.inf) for rows, rcs in runs]),
        "rescales": [],
    }
    for j in range(len(runs[0][1])):
        spans = [rcs[j] for _, rcs in runs]
        e = pooled([mean_of(rows, rcs[j][0], rcs[j][0] + rcs[j][1]) for rows, rcs in runs])
        lo, hi = min(t0 for t0, _, _, _ in spans), max(t0 + d for t0, d, _, _ in spans)
        peak = [w["mean_us"] for w in win if w["mean_us"] and w["t"] + width > lo and w["t"] < hi]
        e.update(name=spans[0][2], trigger_s=round(sum(t0 for t0, _, _, _ in spans) / len(spans), 3),
                 elapsed_s=round(sum(d for _, d, _, _ in spans) / len(spans), 3),
                 peak_window_mean_us=round(max(peak), 1) if peak else None)
        s["rescales"].append(e)
    if runs[0][1]:
        s["after"] = pooled([mean_of(rows, rcs[-1][0] + rcs[-1][1], math.inf) for rows, rcs in runs])
    return s


# Transfer components, in the order the text lists them.
PARTS = [("lists", "posting lists"), ("pq", "PQ codes"), ("raw_streamed", "raw streamed"),
         ("raw_fetched", "raw fetched on demand"), ("graph", "graph")]


def transfers(d):
    """Bytes a run's reconfigurations moved, by component, summed over its rescales. On-demand
    fetches go on after a lazy rescale returns, so with the run's metrics.jsonl they are the
    nodes' raw.fetched counters over the whole run, not what the replies counted."""
    try:
        entries = json.load(open(os.path.join(d, "reconfigurations.json")))
    except (OSError, ValueError):
        return None
    out = {k: 0 for k, _ in PARTS}
    fetched, fetched_bytes = 0, 0
    for e in entries:
        r = e.get("reply") or {}
        out["pq"] += r.get("pq_bytes", 0)
        out["lists"] += r.get("bytes", 0) - r.get("pq_bytes", 0)
        out["raw_streamed"] += r.get("raw_bytes", 0)
        fetched += r.get("raw_fetched", e.get("raw_fetched", 0))
        fetched_bytes += r.get("raw_fetched_bytes", e.get("raw_fetched_bytes", 0))
        out["graph"] += r.get("graph_bytes", 0)
    out["raw_fetched"] = fetched_bytes
    metrics = os.path.join(d, "metrics.jsonl")
    vec_bytes = fetched_bytes / fetched if fetched else base_vec_bytes(d)
    if vec_bytes and os.path.exists(metrics):
        first, last = {}, {}
        with open(metrics) as f:
            for line in f:
                m = json.loads(line)
                if m.get("type") == "raw.fetched":
                    first.setdefault(m["node"], m["value"])
                    last[m["node"]] = m["value"]
        total = sum(last[n] - first[n] for n in last)
        out["raw_fetched"] = max(fetched_bytes, round(total * vec_bytes))
    return out


def base_vec_bytes(d):
    """Bytes per raw vector of the run's base file (experiment.json), or 0 if unknown."""
    try:
        base = json.load(open(os.path.join(d, "experiment.json")))["Base"]
        with open(base, "rb") as f:
            dim = int.from_bytes(f.read(8)[4:8], "little")
    except (OSError, ValueError, KeyError):
        return 0
    return dim * (4 if base.endswith((".fbin", ".fvecs")) else 1)


def mb(v):
    return "%.1f MB" % (v / 1e6) if v >= 1e5 else "%.0f KB" % (v / 1e3)


def svg_document(lines, t_min, span, width, title, subtitle, users, moved, log_latency=False,
                 flows=None, offered=None):
    """lines: [(label, slot, pooled windows, [reconfigurations of each run], runs)]; users:
    [(t, n)] per window or None; moved: {label: bytes by component, mean over its runs} or {};
    flows: {label: flow()}; offered: [(t, rate)] per window with the open-loop load, else None
    (then no backlog panel). One time axis (t_min to span) for every panel."""
    plot_w = W - LEFT - RIGHT
    x = lambda t: LEFT + plot_w * ((t - t_min) / (span - t_min) if span > t_min else 0)
    panels = []  # (title, unit, height, series [(slot or None, [(t, v)])], dashed legend)
    if users:
        panels.append(("concurrent users", "users", 70, [(None, users)]))
    if offered:
        panels.append(("offered load (open loop)", "q/s", 70, [(None, offered)]))
    panels.append(("answered queries per second", "q/s", 150,
                   [(slot, [(w["t"] + width / 2, w["answered"])
                            for w in flows[label] if w["t"] + width <= span])  # the last window is partial
                    for label, slot, _, _, _ in lines]))
    if offered:
        panels.append(("backlog: queries sent, not answered yet", "queries", 120,
                       [(slot, [(w["t"] + width, w["backlog"]) for w in flows[label] if w["t"] + width <= span])
                        for label, slot, _, _, _ in lines]))
    panels.append(("mean latency of the answered queries" + (" (log scale)" if log_latency else ""), "ms", 210,
                   [(slot, [(w["t"] + width / 2, w["mean_us"] / 1000.0 if w["mean_us"] else None) for w in win])
                    for _, slot, win, _, _ in lines]))
    tops, yy = [], TOP + 24
    for _, _, ph, _ in panels:
        tops.append(yy)
        yy += ph + 44
    axis_y = tops[-1] + panels[-1][2]
    strip_top = axis_y + STRIP_GAP
    bars_top = strip_top + STRIP_ROW * len(lines) + (40 if moved else 0)
    h = bars_top + (34 * len(lines) + 30 if moved else 20)

    def colours(mode, surface, ink, ink2, grid, axis):
        rules = [".bg { fill: %s; }" % surface, ".dot { stroke: %s; }" % surface, ".title { fill: %s; }" % ink,
                 ".sub, .legend, .tick { fill: %s; }" % ink2, ".grid { stroke: %s; }" % grid,
                 ".axis { stroke: %s; }" % axis, ".hit line { stroke: %s; }" % ink2, ".cu { stroke: %s; }" % ink2]
        rules += [".c%d { stroke: %s; } .f%d { fill: %s; }" % (i, c[mode], i, c[mode])
                  for i, c in enumerate(PALETTE)]
        return "\n".join("  " + r for r in rules)

    # Colours are written out per class (no CSS variables) so that converters such as
    # rsvg-convert render the figure as browsers do; dark mode overrides the same classes.
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" '
           'font-family="system-ui, -apple-system, Segoe UI, sans-serif">' % (W, h, W, h)]
    svg.append("""<style>
  .title { font-size: 14px; font-weight: 600; }
  .ptitle { font-size: 12px; font-weight: 600; }
  .sub, .legend, .tick { font-size: 11px; }
  .tick { font-variant-numeric: tabular-nums; }
  .grid, .axis { stroke-width: 1; }
  .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .span { stroke: none; }
  .dot { stroke-width: 2; }
  .hit rect { fill: transparent; }
  .hit line { stroke-width: 1; opacity: 0; }
  .hit:hover line { opacity: .6; }
%s
  .ptitle { fill: inherit; }
  @media (prefers-color-scheme: dark) {
%s
  }
</style>""" % (colours(0, "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"),
               colours(1, "#1a1a19", "#ffffff", "#c3c2b7", "#2c2c2a", "#383835")))
    svg.append('<rect class="bg" x="0" y="0" width="%d" height="%d"/>' % (W, h))
    svg.append('<text class="title" x="%d" y="22">%s</text>' % (LEFT, esc(title)))
    svg.append('<text class="sub" x="%d" y="40">%s</text>' % (LEFT, esc(subtitle)))

    for pi, ((ptitle, unit, ph, series), y0) in enumerate(zip(panels, tops)):
        values = [v for _, pts in series for _, v in pts if v is not None]
        if log_latency and pi == len(panels) - 1 and values:
            # Latency spans milliseconds to a stop-and-copy pause of seconds: decades.
            lo = 10 ** math.floor(math.log10(min(v for v in values if v > 0)))
            hi = 10 ** math.ceil(math.log10(max(values)))
            span_l = math.log10(hi) - math.log10(lo) or 1
            y = lambda v, y0=y0, ph=ph, lo=lo, s=span_l: y0 + ph - ph * (math.log10(max(v, lo)) - math.log10(lo)) / s
            ticks = [lo * 10 ** i for i in range(int(round(span_l)) + 1)]
            label = lambda v: fmt(v)
        else:
            ticks = ticks_to(max(values) if values else 1.0)
            top = ticks[-1] or 1.0
            y = lambda v, y0=y0, ph=ph, top=top: y0 + ph - ph * (v / top)
            step = ticks[1] - ticks[0] if len(ticks) > 1 else 1
            label = lambda v, step=step: tick_label(v, step)
            ticks = ticks[:: 2 if ph < 100 else 1]
        svg.append('<text class="title ptitle" x="%d" y="%d">%s</text>' % (LEFT, y0 - 10, esc(ptitle)))
        for v in ticks:
            svg.append('<line class="grid" x1="%d" y1="%.1f" x2="%d" y2="%.1f"/>' % (LEFT, y(v), W - RIGHT, y(v)))
            svg.append('<text class="tick" x="%d" y="%.1f" text-anchor="end">%s</text>'
                       % (LEFT - 8, y(v) + 4, esc(label(v))))
        svg.append('<text class="tick" x="%d" y="%d">%s</text>' % (W - RIGHT + 8, y0 + ph + 4, esc(unit)))
        svg.append('<line class="axis" x1="%d" y1="%d" x2="%d" y2="%d"/>' % (LEFT, y0 + ph, W - RIGHT, y0 + ph))
        for slot, pts in series:
            cls = "cu" if slot is None else "c%d" % slot
            if slot is None:  # a schedule: steps, not a trend
                d = []
                for i, (t, v) in enumerate(pts):
                    if v is None or t < t_min:
                        continue
                    d.append(("M" if not d else "H%.1f V" % x(t)) + ("%.1f,%.1f" % (x(t), y(v)) if not d else "%.1f" % y(v)))
                if d:
                    d.append("H%.1f" % x(span))
                    svg.append('<path class="line %s" d="%s"/>' % (cls, " ".join(d)))
                continue
            run = []
            for t, v in pts + [(None, None)]:
                if t is not None and t < t_min:
                    continue
                if v is None:  # no answered query in the window: a gap, not a zero
                    if len(run) > 1:
                        svg.append('<polyline class="line %s" points="%s"/>'
                                   % (cls, " ".join("%.1f,%.1f" % p for p in run)))
                    elif run:
                        svg.append('<circle class="dot f%d" cx="%.1f" cy="%.1f" r="4"/>' % ((slot,) + run[0]))
                    run = []
                    continue
                run.append((x(t), y(v)))

    xt = nice_ticks(span - t_min, 8)
    for t in xt:
        t += t_min
        if t <= span * 1.001:
            svg.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%s</text>'
                       % (x(t), axis_y + 18, esc(tick_label(t, xt[1] - xt[0]))))
    svg.append('<text class="tick" x="%d" y="%d" text-anchor="middle">seconds since the load started</text>'
               % (LEFT + plot_w / 2, axis_y + 36))

    ly0 = tops[1 if users or offered else 0]  # legend beside the first panel with one line per run
    for i, (label, slot, _, _, _) in enumerate(lines):
        lx, ly = W - RIGHT + 16, ly0 + 10 + i * 18
        svg.append('<line class="line c%d" x1="%d" y1="%d" x2="%d" y2="%d"/>' % (slot, lx, ly - 4, lx + 16, ly - 4))
        svg.append('<text class="legend" x="%d" y="%d">%s</text>' % (lx + 22, ly, esc(label)))

    # Reconfigurations: one strip per label, on the same time axis, every run's spans in it.
    svg.append('<text class="sub" x="%d" y="%d">reconfigurations (target data nodes)</text>'
               % (LEFT, strip_top - 10))
    for i, (label, slot, _, runs_rcs, _) in enumerate(lines):
        ry = strip_top + i * STRIP_ROW
        svg.append('<line class="grid" x1="%d" y1="%d" x2="%d" y2="%d"/>' % (LEFT, ry + 6, W - RIGHT, ry + 6))
        for j in range(len(runs_rcs[0]) if runs_rcs else 0):
            end = 0.0
            for rcs in runs_rcs:
                t0, dur, name, _ = rcs[j]
                x0, x1 = x(t0), max(x(t0 + dur), x(t0) + 3)
                end = max(end, x1)
                svg.append('<rect class="span f%d" x="%.1f" y="%d" width="%.1f" height="8" rx="2">'
                           '<title>%s: %s, %.2f s from %.2f s</title></rect>'
                           % (slot, x0, ry + 2, x1 - x0, esc(label), esc(name), dur, t0))
            svg.append('<text class="tick" x="%.1f" y="%d">%s</text>'
                       % (end + 3, ry + 10, esc(runs_rcs[0][j][2].replace("rescale to ", "→"))))
        lx = W - RIGHT + 16
        svg.append('<line class="line c%d" x1="%d" y1="%d" x2="%d" y2="%d"/>' % (slot, lx, ry + 6, lx + 16, ry + 6))
        svg.append('<text class="legend" x="%d" y="%d">%s</text>' % (lx + 22, ry + 10, esc(label)))

    # Data moved by each pipeline: one bar per label, the parts spelled out beside it.
    if moved:
        svg.append('<text class="sub" x="%d" y="%d">data transferred by the reconfigurations, per run</text>'
                   % (LEFT, bars_top - 10))
        biggest = max(sum(v.values()) for v in moved.values()) or 1
        bx, bw = LEFT + 96, 220
        for i, (label, slot, _, _, _) in enumerate(lines):
            if label not in moved:
                continue
            parts, total = moved[label], sum(moved[label].values())
            by = bars_top + i * 34
            svg.append('<text class="legend" x="%d" y="%d">%s</text>' % (LEFT, by + 10, esc(label)))
            svg.append('<rect class="span f%d" x="%d" y="%d" width="%.1f" height="12" rx="2"/>'
                       % (slot, bx, by, max(2.0, bw * total / biggest)))
            detail = " · ".join("%s %s" % (name, mb(parts[k])) for k, name in PARTS if parts[k])
            svg.append('<text class="tick" x="%d" y="%d"><tspan font-weight="600">%s</tspan></text>'
                       % (bx + bw + 10, by + 10, esc(mb(total))))
            svg.append('<text class="tick" x="%d" y="%d">%s</text>' % (bx, by + 26, esc(detail)))

    # Hover: a crosshair per window over every panel, one readout with every line.
    n = max(len(win) for _, _, win, _, _ in lines)
    for k in range(n):
        t = k * width
        if t + width <= t_min:
            continue
        text = ["%.2f–%.2f s" % (t, t + width)]
        if users and k < len(users) and users[k][1] is not None:
            text.append("users: %d" % users[k][1])
        if offered and k < len(offered):
            text.append("offered: %.0f q/s" % offered[k][1])
        for label, _, win, _, runs in lines:
            w = win[k] if k < len(win) else None
            f = flows[label][k] if k < len(flows[label]) else None
            text.append("%s: %s, %s%s" % (label, "%.0f q/s" % f["answered"] if f else "–",
                                          "%.2f ms" % (w["mean_us"] / 1000.0) if w and w["mean_us"] else "–",
                                          ", backlog %.0f" % f["backlog"] if f and offered else ""))
        x0, x1 = x(max(t, t_min)), x(min(t + width, span))
        svg.append('<g class="hit"><title>%s</title><rect x="%.1f" y="%d" width="%.1f" height="%d"/>'
                   '<line x1="%.1f" y1="%d" x2="%.1f" y2="%d"/></g>'
                   % (esc("\n".join(text)), x0, tops[0], max(x1 - x0, 1), axis_y - tops[0],
                      (x0 + x1) / 2, tops[0], (x0 + x1) / 2, axis_y))
    svg.append("</svg>")
    return "\n".join(svg)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="results directories (latency.csv, reconfigurations.json)")
    ap.add_argument("--labels", help="comma-separated names, one per directory (default: the directory "
                    "names); directories with the same name are pooled")
    ap.add_argument("--window", type=float, default=0.5, help="window in seconds (default 0.5)")
    ap.add_argument("--out", help="output path without extension (default: latency-mean next to the runs)")
    ap.add_argument("--title", default="Mean latency over time")
    ap.add_argument("--log-latency", action="store_true", help="latency panel on a log scale")
    ap.add_argument("--slo-ms", type=float, default=200.0,
                    help="open loop: count the queries answered later than this (default 200 ms)")
    ap.add_argument("--skip", type=float, default=None,
                    help="leave out the queries sent in the first SKIP seconds of each run (warm-up; "
                    "default: \"WarmupSeconds\" in the first run's experiment.json, else 0)")
    args = ap.parse_args()

    names = args.labels.split(",") if args.labels else [os.path.basename(os.path.abspath(d)) for d in args.runs]
    if len(names) != len(args.runs):
        sys.exit("--labels names %d runs, %d given" % (len(names), len(args.runs)))
    groups = {}
    for d, name in zip(args.runs, names):
        groups.setdefault(name, []).append(d)
    labels = list(groups)
    if args.skip is None:
        try:
            args.skip = float(json.load(open(os.path.join(args.runs[0], "experiment.json")))
                              .get("Load", {}).get("WarmupSeconds", 0))
        except (OSError, ValueError):
            args.skip = 0.0
    out = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.runs[0])))
                                   if len(groups[labels[0]]) > 1 else
                                   os.path.dirname(os.path.abspath(args.runs[0])), "latency-mean")

    lines, summary, span, users, moved, flows = [], [], 0.0, None, {}, {}
    sched = offered_schedule(args.runs[0])
    for label, slot in zip(labels, slots(labels)):
        runs = load(label, groups[label], args.skip)
        merged = sorted((r for rows, _ in runs for r in rows), key=lambda r: r["t"])
        win = windows(merged, args.window)
        flows[label] = flow(merged, args.window, len(win), len(runs))
        lines.append((label, slot, win, [rcs for _, rcs in runs], len(runs)))
        s = summarize(label, runs, win, args.window, args.skip)
        if sched and runs[0][1]:
            trigger = sum(rcs[0][0] for _, rcs in runs) / len(runs)
            s["open_loop"] = openloop_summary(merged, flows[label], args.window, trigger, len(runs),
                                              args.slo_ms * 1000)
        per_run = [t for t in (transfers(d) for d in groups[label]) if t]
        if per_run:
            moved[label] = {k: sum(t[k] for t in per_run) / len(per_run) for k, _ in PARTS}
            s["transferred_bytes"] = {k: round(v) for k, v in moved[label].items()}
            s["transferred_bytes"]["total"] = round(sum(moved[label].values()))
        summary.append(s)
        span = max(span, merged[-1]["t"])
        if users is None and any(r["users"] for r in merged):  # the schedule, from the first line
            users = []
            for w in range(len(win)):
                us = [r["users"] for r in merged if w * args.window <= r["t"] < (w + 1) * args.window]
                users.append((w * args.window, max(us) if us else None))

    offered = None
    if sched and not users:
        n = max(len(win) for _, _, win, _, _ in lines)
        offered = [(k * args.window, rate_at(sched, k * args.window)) for k in range(n)]
    counts = {len(groups[l]) for l in labels}
    subtitle = "per %s ms window: answers when they came, latency by send time" % fmt(args.window * 1000)
    if counts != {1}:
        subtitle += ", pooled over %s runs per line" % (counts.pop() if len(counts) == 1 else "the")
    if args.skip:
        subtitle += "; first %s ms left out (warm-up)" % fmt(args.skip * 1000)
    with open(out + ".svg", "w") as f:
        f.write(svg_document(lines, args.skip, span, args.window, args.title, subtitle, users, moved,
                             args.log_latency, flows, offered))
    with open(out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s"] + [c for l in labels for c in (l + " mean_us", l + " queries sent",
                                                          l + " answered_per_s", l + " backlog")])
        for k in range(max(len(win) for _, _, win, _, _ in lines)):
            if (k + 1) * args.window <= args.skip:
                continue
            row = ["%.3f" % (k * args.window)]
            for label, _, win, _, runs in lines:
                x = win[k] if k < len(win) else None
                f = flows[label][k] if k < len(flows[label]) else None
                row += ["%.1f" % x["mean_us"] if x and x["mean_us"] else "",
                        "%d" % round(x["offered"] * args.window / runs) if x else "",
                        "%.1f" % f["answered"] if f else "", "%.1f" % f["backlog"] if f else ""]
            w.writerow(row)
    with open(out + ".json", "w") as f:
        json.dump(summary, f, indent=2)

    def ms(p):
        if not p.get("mean_us"):
            return "–"
        rng = " [%.2f–%.2f]" % (p["min_us"] / 1000, p["max_us"] / 1000) if "min_us" in p else ""
        return "%.2f ms%s" % (p["mean_us"] / 1000, rng)
    for s in summary:
        print("%s (%d run%s): %d queries (%d failed, %d retried), mean %s; before the first rescale %s"
              % (s["label"], s["runs"], "" if s["runs"] == 1 else "s", s["queries"], s["failed"], s["retried"],
                 ms(s["whole_run"]), ms(s["before"])))
        for r in s["rescales"]:
            print("  %-14s at %6.3f s, %4.0f ms: mean %s over %d queries, peak window %s"
                  % (r["name"], r["trigger_s"], r["elapsed_s"] * 1000, ms(r), r["queries"],
                     ms({"mean_us": r["peak_window_mean_us"]})))
        if "after" in s:
            print("  after the last rescale: %s" % ms(s["after"]))
        o = s.get("open_loop")
        if o:
            print("  backlog: peak %d queries at %.1f s, drained %s; %d queries (%.2f%%) over %g ms from the trigger on"
                  % (o["peak_backlog"], o["peak_backlog_at_s"],
                     "at %.1f s" % o["backlog_drained_at_s"] if o["backlog_drained_at_s"] is not None else "never",
                     o["slo_misses"], 100 * (o["slo_miss_fraction"] or 0), o["slo_ms"]))
        if "transferred_bytes" in s:
            tb = s["transferred_bytes"]
            print("  transferred per run: %s (%s)" % (mb(tb["total"]), ", ".join(
                "%s %s" % (name, mb(tb[k])) for k, name in PARTS if tb[k])))
    print("wrote %s.svg, .csv, .json" % out)


if __name__ == "__main__":
    main()
