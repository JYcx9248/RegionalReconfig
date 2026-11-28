#!/usr/bin/env python3
"""The dataset-update figure for the paper (a draft): one column per experiment, each with the
cluster's throughput (the offered rate and the queries answered per second) and the recall of the
answers over time, against the data current at each query -- the system's answers, which apply the
batches, and those an index that never applied them would give; under them a strip of the updates'
phases. No dependencies: plain SVG at the page's width (7 in), light only, in the style of
skew_figure.py; rsvg-convert -f pdf makes the PDF.

    python3 scripts/analysis/update_figure.py --col "CAPTION" RUN [RUN ...] [--col ...] \
        --base-gt GT.ibin --versions DIR --deletes-dir DIR --out FIG [--window 5] [--recall-window 20]

A column's runs are repetitions of one experiment: each line is their mean, each event its mean time.
A query's data version is the number of batches flipped by the epoch it ran in (updates.json:
to_epoch). Its answer is scored against that version's ground truth, found in --versions by the
batches it holds and their first IDs (scripts/score_answers.py keeps them there; version 0 is
--base-gt). The never-updated index's answer to a query is the system's own before the first flip
(every query has one: the query file cycles in seconds, and a query's answer depends only on the
index), scored against the same version, a vector the batches deleted by then counting as a miss
(the deletes files, by name, in --deletes-dir): what a client of an index that ignores the updates
gets -- recall falling with each batch, and deleted vectors returned. The strip: each batch's routing on the control node
(updates.json: decide_seconds) and its staging up to its flip, the major epoch after it (its flip:
major_flip_seconds), and a full rebuild (reconfigurations.json) from its trigger to its end. Dotted
lines mark the batches' flips, where the data changes, dash-dotted ones a rebuild's flip to the next
generation. No backlog: under these loads (0.6 x C4) it stays two orders of magnitude below the
rescales' (user, 2026-11-26). FIG.csv holds what is plotted. The caption, in LaTeX, carries the title.
"""

import argparse
import csv
import json
import math
import os
import struct
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from compare_runs import PALETTE, flow, load, offered_schedule  # noqa: E402
from plot_run import esc, load_end  # noqa: E402
import score_answers as sa  # noqa: E402

FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
W = 504                                  # pt: the page's width
LEFT, RIGHT, GAP = 40, 12, 34             # room for the first y label, after the last column, between columns
TOP = 30                                 # above the panels: the legend
PANELS = [("tput", 40), ("recall", 40)]  # each panel's height, in pt
VGAP = 6                                 # between panels
STRIP_H, STRIP_GAP = 9, 5                # the phases' strip, right under the panels, above the time axis
LINE = PALETTE[0][0]                     # the system's colour in every figure of ours
OFFERED = {"stroke": "#333333", "width": 0.8, "dash": "2.5 1.5"}
STALE = {"stroke": "#8c8c8c", "width": 0.8, "dash": "2 1.2"}
MARK = "#b3b3b3"
DASH = {"batch": "0.8 1.6", "rebuild": "3 1.2 0.8 1.2"}
PHASE = {"routing": ("#d9d9d9", "Routing (control node)"), "staging": (PALETTE[1][0], "Staging"),
         "epoch": (PALETTE[2][0], "Major epoch"), "rebuild": (PALETTE[3][0], "Full rebuild")}


def label(v, digits=0):
    if digits:
        return "%.*f" % (digits, v)
    return ("%.1f" % v).rstrip("0").rstrip(".") if v != int(v) else "%d" % v


def wrap(text, max_chars):
    """text in as few lines as fit max_chars each, the lines as even as the words allow."""
    words = text.split()
    for n in range(1, len(words) + 1):
        width = max(max(len(w) for w in words), math.ceil(len(text) / n))
        while width <= max_chars:
            lines, cur = [], ""
            for w in words:
                if cur and len(cur) + 1 + len(w) > width:
                    lines.append(cur)
                    cur = w
                else:
                    cur = cur + " " + w if cur else w
            lines.append(cur)
            if len(lines) <= n:
                return lines
            width += 1
    return [text]


def time_step(seconds, points):
    """The time axis' step: the smallest round one that leaves room for its labels (24 pt apart)."""
    for step in (50, 100, 200, 250, 500, 1000, 2000):
        if step * points / seconds >= 24:
            return step
    return 5000


def ticks(hi):
    """Round ticks from 0 past hi: 2 to 4 steps of 1, 2, 2.5 or 5 times a power of ten."""
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


def polyline(pts, style):
    dash = ' stroke-dasharray="%s"' % style["dash"] if style.get("dash") else ""
    return ('<polyline points="%s" fill="none" stroke="%s" stroke-width="%g"%s stroke-linejoin="round"/>'
            % (pts, style["stroke"], style["width"], dash))


def mean_time(values):
    vs = [v for v in values if v is not None]
    return sum(vs) / len(vs) if vs else None


def version_files(versions_dir, base_gt):
    """The ground truth of each data version kept in versions_dir: [(batches as (first ID, count)), GT]."""
    out = []
    for name in sorted(os.listdir(versions_dir)):
        if not name.endswith(".json"):
            continue
        rec = json.load(open(os.path.join(versions_dir, name)))
        path = os.path.join(versions_dir, name[:-5] + ".ibin")
        if os.path.exists(path):
            out.append(([(b["first"], b["count"]) for b in rec["batches"]], path))
    return out


class Truth:
    """The ground truth of a run's data versions: version v = its first v batches."""

    def __init__(self, base_gt, versions_dir):
        self.base = sa.read_gt(base_gt)
        self.files = version_files(versions_dir, base_gt)
        self.loaded = {}

    def of(self, batches):
        if not batches:
            return self.base
        want = [(b["first"], b["count"]) for b in batches]
        for have, path in self.files:
            if have == want:
                if path not in self.loaded:
                    self.loaded[path] = sa.read_gt(path)
                return self.loaded[path]
        sys.exit("no ground truth in the versions for batches %s" % want)


def deleted_ids(path):
    """A deletes file's IDs (a u64 count, then u32 IDs)."""
    b = open(path, "rb").read()
    (n,) = struct.unpack_from("<Q", b, 0)
    return set(struct.unpack_from("<%dI" % n, b, 8))


def recall_without(gt, row, ids, dists, k, gone):
    """score_answers.recall, with every vector in gone (deleted) a miss: the tie rule would count a
    deleted vector as near as the k-th true neighbour as a hit."""
    _, K, gids, gd = gt
    k = min(k, K)
    truth = set(gids[row * K: row * K + k])
    kth = gd[row * K + k - 1] if gd else None
    return sum(1 for i in range(min(k, len(ids)))
               if ids[i] not in gone and (ids[i] in truth or (kth is not None and dists[i] <= kth))) / k


def column(caption, runs, truth, deletes_dir, window, recall_window):
    """What one column plots, its runs pooled."""
    exp = json.load(open(os.path.join(runs[0], "experiment.json")))
    skip = float(exp.get("Load", {}).get("WarmupSeconds", 0))
    end = min(e for e in (load_end(d) for d in runs) if e)
    sched = offered_schedule(runs[0])
    pooled = load("runs", runs, skip)
    merged = sorted((r for rows, _ in pooled for r in rows), key=lambda r: r["t"])
    fl = flow([r for r in merged if r["t"] < end], window, int(math.ceil(end / window)), len(runs))
    answered = [(x["t"] + window / 2, x["answered"]) for x in fl if x["t"] >= skip and x["t"] + window <= end]
    offered = [(skip, sched[0][1]), (end, sched[-1][1])]

    # Recall per window, by the time each query was sent, against the data version it read: the
    # system's answer, and the never-updated index's (the query's answer before the first flip).
    sums = defaultdict(lambda: [0, 0.0, 0.0])
    entries, rebuilds = [], []
    for d in runs:
        k, stride, answers = sa.read_answers(os.path.join(d, "answers.bin"))
        batches, _ = sa.events_of(d)
        gts = [truth.of(batches[:v]) for v in range(len(batches) + 1)]
        flips = [b["to_epoch"] for b in batches]
        gone = [set()]
        for b in batches:
            gone.append(gone[-1] | (deleted_ids(os.path.join(deletes_dir, os.path.basename(b["deletes"])))
                                    if b["deletes"] else set()))
        before = {}
        for q, epoch, us, ids, dists in answers:
            if epoch < flips[0] and q not in before:
                before[q] = (ids, dists)
        for q, epoch, us, ids, dists in answers:
            t = us / 1e6
            if t < skip or t >= end:
                continue
            if q not in before:
                sys.exit("%s: query %d has no answer before the first flip" % (d, q))
            v = sum(1 for e in flips if e <= epoch)
            row = q // stride
            acc = sums[int(t // recall_window)]
            acc[0] += 1
            acc[1] += sa.recall(gts[v], row, ids, dists, k)
            acc[2] += recall_without(gts[v], row, before[q][0], before[q][1], k, gone[v])
        upd = json.load(open(os.path.join(d, "updates.json")))
        entries.append([{"t0": e["trigger_seconds"],
                         "routed": e["trigger_seconds"] + e.get("decide_seconds", 0),
                         "flip": e["trigger_seconds"] + e["flip_seconds"] if e.get("flip_seconds") else None,
                         "epoch": e["trigger_seconds"] + e["major_flip_seconds"] if e.get("major_flip_seconds") else None,
                         "policy": (e.get("update") or {}).get("Policy", "threshold")}
                        for e in upd])
        path = os.path.join(d, "reconfigurations.json")
        rb = []
        for e in (json.load(open(path)) if os.path.exists(path) else []):
            rep = e.get("reply") or {}
            if e.get("rebuild") or "generation" in rep:
                ph = rep.get("phases") or {}
                t0, t1 = e["trigger_seconds"], e["trigger_seconds"] + e["elapsed_seconds"]
                rb.append({"t0": t0, "end": t1, "flip": t1 - ph.get("release", 0)})
        rebuilds.append(rb)
    recall = [((i + 0.5) * recall_window, s1 / n, s0 / n) for i, (n, s1, s0) in sorted(sums.items())
              if n and i * recall_window >= skip and (i + 1) * recall_window <= end]
    # Events, their times averaged over the runs (the same updates in each).
    ev = [{key: mean_time([run[i][key] for run in entries]) for key in ("t0", "routed", "flip", "epoch")}
          | {"policy": entries[0][i]["policy"]} for i in range(min(len(r) for r in entries))]
    rbs = [{key: mean_time([run[i][key] for run in rebuilds]) for key in ("t0", "end", "flip")}
           for i in range(min(len(r) for r in rebuilds))] if rebuilds else []
    return {"caption": caption, "skip": skip, "end": end, "answered": answered, "offered": offered,
            "recall": recall, "entries": ev, "rebuilds": rbs, "runs": len(runs)}


def draw(cols, out):
    n = len(cols)
    cw = (W - LEFT - RIGHT - GAP * (n - 1)) / n
    ys, y = {}, TOP
    for name, h in PANELS:
        ys[name] = (y, y + h)
        y += h + VGAP
    strip_top = ys[PANELS[-1][0]][1] + STRIP_GAP
    axis_bottom = strip_top + STRIP_H
    # Each subcaption in as many lines as the column's width needs (7 pt type, ~3.7 pt a character).
    captions = [wrap(c["caption"], int(cw / 3.7)) for c in cols]
    H = axis_bottom + 31 + 8.5 * (max(len(c) for c in captions) - 1)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%dpt" height="%dpt" '
           'font-family="%s" fill="#000">' % (W, H, W, H, FONT),
           '<rect x="0" y="0" width="%d" height="%d" fill="#fff"/>' % (W, H)]
    # Shared ranges, so that the columns compare.
    tput_top = ticks(max(v for c in cols for _, v in c["answered"] + c["offered"]) * 1.04)
    lo = min(min(r0, r1) for c in cols for _, r1, r0 in c["recall"])
    hi = max(max(r0, r1) for c in cols for _, r1, r0 in c["recall"])
    if hi - lo > 0.1:  # the never-updated index falls far: steps of 0.1, room under its lowest point
        rec_step, rec_digits = 0.1, 1
        rec_lo, rec_hi = math.floor(lo * 10 - 0.5) / 10, math.ceil(hi * 10) / 10
    else:  # steps of 0.02, one more under the lowest line for the legend
        rec_step, rec_digits = 0.02, 2
        rec_lo, rec_hi = math.floor(lo * 50 - 0.2) / 50 - 0.02, math.ceil(hi * 50 + 0.2) / 50
    rec_ticks = [rec_lo + rec_step * i for i in range(int(round((rec_hi - rec_lo) / rec_step)) + 1)]
    for ci, c in enumerate(cols):
        x0 = LEFT + ci * (cw + GAP)
        x1 = x0 + cw
        t0, t1 = c["skip"], c["end"]
        x = lambda t: x0 + (x1 - x0) * (t - t0) / (t1 - t0)  # noqa: E731
        marks = [("batch", e["flip"]) for e in c["entries"] if e["flip"]] + \
                [("rebuild", r["flip"]) for r in c["rebuilds"]]
        for name, _ in PANELS:
            py0, py1 = ys[name]
            if name == "tput":
                tks, scale, lo_v = tput_top, 1.0, 0.0
                series = [(c["answered"], {"stroke": LINE, "width": 0.9}), (c["offered"], OFFERED)]
                ylabel = "Queries/s"
            else:
                tks, scale, lo_v = rec_ticks, 1.0, rec_lo
                series = [([(t, r0) for t, _, r0 in c["recall"]], STALE),
                          ([(t, r1) for t, r1, _ in c["recall"]], {"stroke": LINE, "width": 0.9})]
                ylabel = "Recall@10"
            top = tks[-1]
            yv = lambda v: py1 - (py1 - py0) * (v - lo_v) / (top - lo_v)  # noqa: E731
            for kind, t in marks:
                if t0 <= t <= t1:
                    svg.append('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="%s" stroke-width="0.8" '
                               'stroke-dasharray="%s"/>' % (x(t), py0, x(t), py1, MARK, DASH[kind]))
            for pts, style in series:
                pts = [(t, min(max(v, lo_v), top)) for t, v in pts if t0 <= t <= t1]
                if not pts:
                    continue
                line = " ".join("%.2f,%.2f" % (x(t), yv(v)) for t, v in pts)
                if style.get("area"):
                    svg.append('<polygon points="%.2f,%.2f %s %.2f,%.2f" fill="#dde8f6" stroke="none"/>'
                               % (x(pts[0][0]), py1, line, x(pts[-1][0]), py1))
                svg.append(polyline(line, style))
            svg.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="none" stroke="#000" '
                       'stroke-width="0.6"/>' % (x0, py0, x1 - x0, py1 - py0))
            digits = rec_digits if name == "recall" else 0
            for v in tks:
                svg.append('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#000" stroke-width="0.6"/>'
                           % (x0, yv(v), x0 - 2.5, yv(v)))
                svg.append('<text x="%.2f" y="%.2f" font-size="6" text-anchor="end">%s</text>'
                           % (x0 - 3.8, yv(v) + 2.1, label(v, digits)))
            width = max(len(label(v, digits)) for v in tks)
            svg.append('<text transform="translate(%.2f,%.2f) rotate(-90)" font-size="6.5" text-anchor="middle">%s</text>'
                       % (x0 - 8.5 - 3.4 * width, (py0 + py1) / 2, esc(ylabel)))
            if name == "tput":
                legend = [("offered", OFFERED), ("answered", {"stroke": LINE, "width": 0.9})]
            elif name == "recall":
                legend = [("updated", {"stroke": LINE, "width": 0.9}), ("never updated", STALE)]
            else:
                legend = []
            lx, ky = x0 + 5, py1 - 3.5
            for lname, style in legend:
                svg.append(polyline("%.2f,%.2f %.2f,%.2f" % (lx, ky - 2, lx + 10, ky - 2), style))
                svg.append('<text x="%.2f" y="%.2f" font-size="5.5">%s</text>' % (lx + 12, ky, esc(lname)))
                lx += 12 + 3.1 * len(lname) + 8
        step = time_step(t1 - t0, x1 - x0)
        for t in range(int(math.ceil(t0 / step)) * step, int(t1) + 1, step):
            svg.append('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#000" stroke-width="0.6"/>'
                       % (x(t), axis_bottom, x(t), axis_bottom + 2.5))
            svg.append('<text x="%.2f" y="%.2f" font-size="6" text-anchor="middle">%d</text>'
                       % (x(t), axis_bottom + 8.5, t))
        svg.append('<text x="%.2f" y="%.2f" font-size="6" text-anchor="middle">Time (s)</text>'
                   % ((x0 + x1) / 2, axis_bottom + 15))
        for li, text in enumerate(captions[ci]):
            svg.append('<text x="%.2f" y="%.2f" font-size="7" text-anchor="middle">%s</text>'
                       % ((x0 + x1) / 2, axis_bottom + 26 + 8.5 * li, esc(text)))
        # The strip: one row of the updates' phases.
        sy = strip_top
        svg.append('<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="#e6e6e6" stroke-width="0.6"/>'
                   % (x0, sy + STRIP_H / 2, x1, sy + STRIP_H / 2))
        svg.append('<text x="%.2f" y="%.2f" font-size="6.5" text-anchor="end">Updates</text>'
                   % (x0 - 3.8, sy + STRIP_H / 2 + 2.2))

        def bar(a, b, colour):
            if a is None or b is None or b <= a:
                return
            w = max(x(b) - x(a) - 0.5, 0.8)
            svg.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" fill="%s"/>'
                       % (x(a) + 0.25, sy, w, STRIP_H, colour))
        for i, e in enumerate(c["entries"], 1):
            bar(e["t0"], e["routed"], PHASE["routing"][0])
            bar(e["routed"], e["flip"], PHASE["staging"][0])
            bar(e["flip"], e["epoch"], PHASE["epoch"][0])
            svg.append('<text x="%.2f" y="%.2f" font-size="5.5" text-anchor="middle">B%d</text>'
                       % ((x(e["t0"]) + x(e["routed"])) / 2, sy + STRIP_H / 2 + 2, i))
        for r in c["rebuilds"]:
            bar(r["t0"], r["end"], PHASE["rebuild"][0])
            svg.append('<text x="%.2f" y="%.2f" font-size="5.5">rebuild</text>'
                       % (x(r["end"]) + 2, sy + STRIP_H / 2 + 2))
    # The legend across the top: the marks and the strip's phases.
    items = [("line", "batch", "Batch flip")]
    if any(c["rebuilds"] for c in cols):
        items.append(("line", "rebuild", "Rebuild flip"))
    items += [("box", key, PHASE[key][1]) for key in ("routing", "staging", "epoch")]
    if any(c["rebuilds"] for c in cols):
        items.append(("box", "rebuild", PHASE["rebuild"][1]))
    widths = [14 + 3 + 3.55 * len(name) for _, _, name in items]
    lx = (W - sum(widths) - 7 * (len(widths) - 1)) / 2
    for (kind, key, name), wd in zip(items, widths):
        if kind == "line":
            svg.append('<line x1="%.1f" y1="9" x2="%.1f" y2="9" stroke="%s" stroke-width="1.1" stroke-dasharray="%s"/>'
                       % (lx, lx + 14, MARK, DASH[key]))
        else:
            svg.append('<rect x="%.1f" y="5.5" width="14" height="7" fill="%s"/>' % (lx, PHASE[key][0]))
        svg.append('<text x="%.1f" y="11.3" font-size="6.5">%s</text>' % (lx + 17, esc(name)))
        lx += wd + 7
    svg.append("</svg>")
    with open(out + ".svg", "w") as f:
        f.write("\n".join(svg))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--col", nargs="+", action="append", required=True, metavar=("CAPTION", "RUN"),
                    help="a column: its subcaption, then the runs (repetitions of one experiment)")
    ap.add_argument("--base-gt", required=True, help="the base data's ground truth (the experiments' GroundTruth)")
    ap.add_argument("--versions", required=True, help="the data versions' ground truth (score_answers.py's)")
    ap.add_argument("--deletes-dir", required=True, help="the batches' deletes files, by name (updates.json: Deletes)")
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--window", type=float, default=5.0, help="seconds per point of the throughput")
    ap.add_argument("--recall-window", type=float, default=20.0, help="seconds per point of the recall")
    args = ap.parse_args()
    truth = Truth(args.base_gt, args.versions)
    cols = []
    for spec in args.col:
        if len(spec) < 2:
            sys.exit("--col CAPTION RUN [RUN ...]")
        cols.append(column(spec[0], spec[1:], truth, args.deletes_dir, args.window, args.recall_window))
    draw(cols, args.out)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["column", "series", "t", "value"])
        for i, c in enumerate(cols):
            col = "abcdefgh"[i]
            w.writerows((col, "offered", "%g" % t, "%.1f" % v) for t, v in c["offered"])
            w.writerows((col, "answered", "%g" % t, "%.1f" % v) for t, v in c["answered"])
            w.writerows((col, "recall, updated", "%g" % t, "%.4f" % r1) for t, r1, _ in c["recall"])
            w.writerows((col, "recall, never updated", "%g" % t, "%.4f" % r0) for t, _, r0 in c["recall"])
    for i, c in enumerate(cols):
        r1 = [r for _, r, _ in c["recall"]]
        r0 = [r for _, _, r in c["recall"]]
        print("(%s) %s: %d runs; recall, updated %.4f-%.4f, never updated %.4f-%.4f"
              % ("abcdefgh"[i], c["caption"], c["runs"], min(r1), max(r1), min(r0), max(r0)))
    print("wrote %s.svg and %s.csv" % (args.out, args.out))


if __name__ == "__main__":
    main()
