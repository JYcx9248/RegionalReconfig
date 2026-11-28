#!/usr/bin/env python3
"""The task-migration figure (E6), as Koala's (NSDI '27) Figure 12: a column per run -- one
FetchExtra setting or the drain each -- with (a) the queries answered per second and (b) the
raw-vector bytes the migration's new owner received per second (shaded): fetched on demand and,
with FetchExtra, in the background, or with a drain, pulled after the flip. A run migrates by a
Migrate entry (FetchExtra moves the rest) or by a placement plan that gives one data node's
partitions to one that joins: the old owner then leaves as in a scale-in, drained after the flip.
A dashed line marks the migration's trigger, a dotted one the moment no posting named the old
owner any more (FetchExtra done, or drained): it went offline, labeled by its name -- "source"
would read as the queries' source, which runs to the load's end. With FetchExtra 0 and no drain
there is none -- the old owner stays the holder of what no query fetched. No
dependencies: plain SVG at the page's width (7 in), light only; rsvg-convert -f pdf makes the PDF.

    python3 scripts/analysis/migration_figure.py RUN [RUN ...] --out FIG [--labels A,B,C] [--window 10]

Each RUN holds latency.csv, clock.json, experiment.json, reconfigurations.json (one task migration)
and the new owner's stats-<node>.jsonl, a record a second stamped on the controller's clock, as
clock.json's load_start is. Values are means over a trailing window (--window seconds; Koala's are
10). The time axis runs from the warm-up's end to where the load stopped. FIG.json holds, per run,
when the old owner went offline and the bytes the new owner received, in the background (all but
what its queries fetched on demand) and in all;
FIG.csv the series drawn. The caption, in LaTeX, carries the title.
"""

import argparse
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from compare_runs import PALETTE  # noqa: E402
from plot_run import answered_at, esc, read_latency  # noqa: E402

FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
W = 504                                       # pt: the page's width
ROWS = [(30, 88), (126, 184)]                 # each row's plot top and bottom
H = 212
PANELS = [(42, 182), (198, 338), (354, 494)]  # each column's plot left and right (three runs)
LINE = PALETTE[0][0]                          # lazy's colour in every figure of ours
MARK = "#9a9a9a"
DASH = {"trigger": "3 2", "offline": "0.8 1.6"}


def ticks3(hi):
    """About three round ticks from 0 that reach past hi."""
    if hi <= 0:
        return [0.0, 1.0]
    for mag in (10 ** k for k in range(-3, 10)):
        for m in (1, 2, 2.5, 5):
            step = m * mag
            if hi / step <= 3:
                return [step * i for i in range(int(math.ceil(hi / step)) + 1)]
    return [0.0, hi]


def label(v):
    return ("%.1f" % v).rstrip("0").rstrip(".") if v != int(v) else "%d" % v


def trailing_mean(values, window):
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def migration(d):
    """The run's task migration (reconfigurations.json): (entry, old owner, new owner, kind) -- a
    Migrate entry ("migrate"), or a plan that moves one data node's partitions to one that joins
    ("drain": the old owner leaves the placement and is drained after the flip)."""
    for e in json.load(open(os.path.join(d, "reconfigurations.json"))):
        if e.get("migrate"):
            return e, e["migrate"]["From"], e["migrate"]["To"], "migrate"
        reply = e.get("reply") or {}
        if e.get("plan") and len(reply.get("removed") or []) == 1 and len(reply.get("added") or []) == 1:
            # The runners start n<i> as the i-th node, and the IDs follow that order.
            return e, "n%d" % reply["removed"][0], "n%d" % reply["added"][0], "drain"
    sys.exit("%s: no task migration in reconfigurations.json" % d)


def answered(d, seconds):
    """Queries answered in each second of the run, by when the answer came."""
    per = [0] * seconds
    for r in read_latency(os.path.join(d, "latency.csv")):
        s = int(answered_at(r))
        if r["status"] == "ok" and 0 <= s < seconds:
            per[s] += 1
    return per


def received(d, node, t0, seconds, vb=None):
    """The raw-vector bytes node received in each second of the run, in all and as its queries'
    fetches on demand (the rest -- FetchExtra's requests, a drain's pulls -- is the background),
    from the counters of its stats log, each delta spread evenly over the time between its two
    records; and the bytes a vector takes: vb, else received over fetched, which a drain's pulls
    (received, not fetched) would inflate."""
    rows = []
    with open(os.path.join(d, "stats-%s.jsonl" % node)) as f:
        for line in f:
            r = json.loads(line)
            if "t" in r and "raw_bytes" in r:
                rows.append((r["t"] - t0, r["raw_bytes"]["received"], r["fetch"]["fetched"],
                             r["fetch"].get("extra", {}).get("fetched", 0)))
    rows.sort()
    if not vb:
        vecs = rows[-1][2] - rows[0][2] if rows else 0
        vb = (rows[-1][1] - rows[0][1]) / vecs if vecs else 0.0
    total, ondemand = [0.0] * seconds, [0.0] * seconds
    for (ta, ba, fa, ea), (tb, bb, fb, eb) in zip(rows, rows[1:]):
        if tb <= ta:
            continue
        for out, delta in ((total, bb - ba), (ondemand, ((fb - fa) - (eb - ea)) * vb)):
            lo, hi = max(ta, 0.0), min(tb, float(seconds))
            s = int(lo)
            while s < hi:
                out[s] += delta * (min(hi, s + 1) - max(lo, s)) / (tb - ta)
                s += 1
    return total, ondemand, vb


def panel(svg, xs, row, t0, t1, series, marks, top, ylabel=None, note=None):
    """One plot: the marks [(kind, t)] under the series [(values per second, area)], its frame, the
    ticks of the row's scale (top, the highest tick) and, in the first column, the y label."""
    x0, x1 = xs
    y0, y1 = row
    x = lambda t: x0 + (x1 - x0) * (t - t0) / (t1 - t0)  # noqa: E731
    y = lambda v: y1 - (y1 - y0) * min(v, top) / top  # noqa: E731
    for kind, t in marks:
        if t is not None and t0 <= t <= t1:
            svg.append('<line x1="%.2f" y1="%d" x2="%.2f" y2="%d" stroke="%s" stroke-width="1" '
                       'stroke-dasharray="%s"/>' % (x(t), y0, x(t), y1, MARK, DASH[kind]))
    for values, area in series:
        pts = [(s + 0.5, v) for s, v in enumerate(values) if t0 <= s + 0.5 <= t1]
        if not pts:
            continue
        line = " ".join("%.2f,%.2f" % (x(t), y(v)) for t, v in pts)
        if area:
            svg.append('<polygon points="%.2f,%d %s %.2f,%d" fill="#dde6f5" stroke="none"/>'
                       % (x(pts[0][0]), y1, line, x(pts[-1][0]), y1))
        svg.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="1" stroke-linejoin="round"/>'
                   % (line, LINE))
    svg.append('<rect x="%d" y="%d" width="%d" height="%d" fill="none" stroke="#000" stroke-width="0.6"/>'
               % (x0, y0, x1 - x0, y1 - y0))
    for v in ticks3(top)[:-1] + [top]:
        yv = y1 - (y1 - y0) * v / top
        svg.append('<line x1="%d" y1="%.2f" x2="%.1f" y2="%.2f" stroke="#000" stroke-width="0.6"/>'
                   % (x0, yv, x0 - 2.5, yv))
        if ylabel:
            svg.append('<text x="%.1f" y="%.2f" font-size="6.5" text-anchor="end">%s</text>'
                       % (x0 - 3.8, yv + 2.3, label(v)))
    step = 200 if t1 - t0 > 600 else 100
    for t in range(int(math.ceil(t0 / step)) * step, int(t1) + 1, step):
        svg.append('<line x1="%.2f" y1="%d" x2="%.2f" y2="%.1f" stroke="#000" stroke-width="0.6"/>'
                   % (x(t), y1, x(t), y1 + 2.5))
        svg.append('<text x="%.2f" y="%.1f" font-size="6.5" text-anchor="middle">%d</text>' % (x(t), y1 + 9.5, t))
    if ylabel:
        width = max(len(label(v)) for v in ticks3(top))
        svg.append('<text transform="translate(%.1f,%.1f) rotate(-90)" font-size="7" text-anchor="middle">%s</text>'
                   % (x0 - 9 - 3.6 * width, (y0 + y1) / 2, esc(ylabel)))
    if note:  # on a white ground, over the marks it may cross
        width = 3.1 * max(len(line) for line in note) + 4
        svg.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#fff" fill-opacity="0.9"/>'
                   % (x1 - 1 - width, y0 + 1.5, width, 7 * len(note) + 2))
    for i, line in enumerate(note or []):
        svg.append('<text x="%.1f" y="%.1f" font-size="6" text-anchor="end">%s</text>'
                   % (x1 - 3, y0 + 8 + 7 * i, esc(line)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="results directories, a column each (at most three)")
    ap.add_argument("--out", required=True, help="FIG: FIG.svg, FIG.json and FIG.csv are written")
    ap.add_argument("--labels", help="column titles, comma-separated (default: from each run's FetchExtra)")
    ap.add_argument("--window", type=int, default=10, help="trailing mean, in seconds (default 10, Koala's)")
    a = ap.parse_args()
    if len(a.runs) > len(PANELS):
        sys.exit("at most %d runs" % len(PANELS))
    labels = a.labels.split(",") if a.labels else []
    if labels and len(labels) != len(a.runs):
        sys.exit("--labels: one per run")

    cols, summary = [], []
    for i, d in enumerate(a.runs):
        exp = json.load(open(os.path.join(d, "experiment.json")))
        load = exp.get("Load", {})
        seconds = int(load.get("DurationSeconds", 60))
        t0 = json.load(open(os.path.join(d, "clock.json")))["load_start"]
        e, source, target, kind = migration(d)
        reply = e.get("reply") or {}
        per = answered(d, seconds)
        vb = reply["raw_fetched_bytes"] / reply["raw_fetched"] if reply.get("raw_fetched") else None
        total, ondemand, vb = received(d, target, t0, seconds, vb)
        background = max(0.0, sum(total) - sum(ondemand))
        trigger = float(e["trigger_seconds"])
        flip = trigger + float(reply.get("flip_seconds", 0))
        # Measured from the flip: FetchExtra done (migrate_seconds) or the drain's end.
        moved = reply.get("migrate_seconds") if kind == "migrate" else reply.get("drain_seconds")
        offline = flip + float(moved) if moved else None
        n = int(exp.get("FetchExtra", 0))
        name = labels[i] if labels else ("Fetch on demand + drain" if kind == "drain" else
                                         "Fetch on demand" if n == 0 else "FetchExtra %d" % n)
        cols.append({"name": name, "tput": trailing_mean(per, a.window),
                     "mbps": trailing_mean([b / 1e6 for b in total], a.window),
                     "trigger": trigger, "offline": offline, "seconds": seconds,
                     "warmup": float(load.get("WarmupSeconds", 0))})
        summary.append({"run": d, "label": name, "kind": kind, "fetch_extra": n, "source": source, "target": target,
                        "trigger_s": round(trigger, 1), "flip_s": round(flip, 1),
                        "offline_s": round(offline, 1) if offline is not None else None,
                        "migrate_seconds": reply.get("migrate_seconds"), "drain_seconds": reply.get("drain_seconds"),
                        "drain_vectors": reply.get("drain_vectors"), "holding": reply.get("holding"),
                        "received_gb": round(sum(total) / 1e9, 3), "background_gb": round(background / 1e9, 3),
                        "vec_bytes": round(vb, 1), "answered": sum(per), "duration_s": seconds})

    t0 = max(c["warmup"] for c in cols)
    t1 = min(c["seconds"] for c in cols)
    tops = [ticks3(max(max(c[k][int(t0):int(t1)] or [0]) for c in cols) * 1.04)[-1] for k in ("tput", "mbps")]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%dpt" height="%dpt" '
           'font-family="%s">' % (W, H, W, H, FONT), '<rect x="0" y="0" width="%d" height="%d" fill="#fff"/>' % (W, H)]
    for k, c in enumerate(cols):
        marks = [("trigger", c["trigger"]), ("offline", c["offline"])]
        s = summary[k]
        note = ["%.1f GB from %s" % (s["received_gb"], s["source"])]
        if s["background_gb"] >= 0.05:
            note.append("%.1f GB %s" % (s["background_gb"], "by the drain" if s["kind"] == "drain" else
                                        "in the background"))
        note.append("%s offline at %d s" % (s["source"], c["offline"]) if c["offline"] is not None
                    else "%s keeps the rest" % s["source"])
        panel(svg, PANELS[k], ROWS[0], t0, t1, [(c["tput"], False)], marks, tops[0],
              ylabel="Answered (q/s)" if k == 0 else None)
        panel(svg, PANELS[k], ROWS[1], t0, t1, [(c["mbps"], True)], marks, tops[1],
              ylabel="Received (MB/s)" if k == 0 else None, note=note)
        svg.append('<text x="%.1f" y="%d" font-size="7.5" text-anchor="middle">%s</text>'
                   % ((PANELS[k][0] + PANELS[k][1]) / 2, ROWS[0][0] - 5, esc(c["name"])))
    mid = (PANELS[0][0] + PANELS[len(cols) - 1][1]) / 2
    svg.append('<text x="%.1f" y="%d" font-size="7.5" text-anchor="middle">(a) Throughput over time (s)</text>'
               % (mid, ROWS[0][1] + 21))
    svg.append('<text x="%.1f" y="%d" font-size="7.5" text-anchor="middle">(b) Raw vectors the new owner '
               'received over time (s)</text>' % (mid, ROWS[1][1] + 21))
    lx = mid - 150
    olds = sorted({s["source"] for s in summary})
    for kind, name in (("trigger", "Task migration"),
                       ("offline", "%s offline" % olds[0] if len(olds) == 1 else "Old owner offline")):
        svg.append('<line x1="%.1f" y1="8" x2="%.1f" y2="8" stroke="%s" stroke-width="1" stroke-dasharray="%s"/>'
                   % (lx, lx + 14, MARK, DASH[kind]))
        svg.append('<text x="%.1f" y="10.3" font-size="6.5">%s</text>' % (lx + 17, name))
        lx += 92
    svg.append('<rect x="%.1f" y="4.5" width="14" height="7" fill="#dde6f5" stroke="%s" stroke-width="0.6"/>'
               % (lx, LINE))
    svg.append('<text x="%.1f" y="10.3" font-size="6.5">Raw-vector migration</text>' % (lx + 17))
    svg.append("</svg>")
    with open(a.out + ".svg", "w") as f:
        f.write("\n".join(svg) + "\n")
    json.dump(summary, open(a.out + ".json", "w"), indent=2)
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "second", "answered_per_s", "received_mb_per_s"])
        for c in cols:
            for s in range(int(t0), int(t1)):
                w.writerow([c["name"], s, round(c["tput"][s], 2), round(c["mbps"][s], 3)])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
