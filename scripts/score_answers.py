#!/usr/bin/env python3
"""Recall by data version for runs with update batches (U14), and whether every query got one
answer wherever the index was the same.

usage: score_answers.py RUN_DIR [RUN_DIR ...] [--gt-dir DIR] [--threads N]
       score_answers.py --plan EXPERIMENT.json [--gt-dir DIR] [--threads N]

A run with batches keeps the answers of its scored queries -- the ones its GroundTruth covers,
every GroundTruthStride-th -- with the epoch each ran in (answers.bin, rtier-loadgen -answers).
A batch changes the data from its flip on (updates.json: its to_epoch), so each answer is scored
against the ground truth of the data version its epoch read: the base without the vectors the
batches flipped by then deleted, and with the ones they inserted, under the IDs the update
controller gave them (updates.json: ids). latency.csv's recall, against the base's ground truth,
holds only until the first batch flips.

The ground truth of each version is made once (fusion_gt --exclude --insert, the same rows and k
as the base's) and kept by content -- the base ground truth's digest, every batch file's digest
and its first ID -- next to the base's (<GroundTruth>.versions/, or --gt-dir): the runs of one
experiment, whatever their protocol, apply the same batches and share it. It takes as long as the
base's: a brute force over the base and the inserts.

Prints, and writes into each run directory (scores.json):
- the recall per data version and per epoch, against the version's ground truth, and against the
  base's (what latency.csv reported);
- the scored queries that got different answers on the same index -- the same data version, the
  same major epochs (they split and merge lists) and the same full rebuilds (a next generation of
  the same data) -- within a run and, given several runs of one experiment, across them: a
  query's answer depends only on the index its epoch names.

--plan makes the ground truth of every data version an experiment's batches will make, before
any run: each batch's inserts take the IDs after the index's vectors (meta.txt num_vectors) and
the earlier batches', as the update controller hands them out when every batch commits. A run
whose batches did so finds them made (the same content); one where a batch failed makes its own.
Its machine needs only the data, the batch files and fusion_gt -- not the cluster.

Relative paths in experiment.json are taken from the repository, as run_one.sh starts runs there.
Standard library only, as the runners.
"""
import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_local as rl  # noqa: E402

MAGIC = b"RTANSWR1"


def at_root(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(rl.ROOT, path)


def read_answers(path: str):
    """answers.bin (cmd/rtier-loadgen/answers.go): k, stride, and (query, epoch, sched_us, ids, dists)."""
    with open(path, "rb") as f:
        b = f.read()
    if b[:8] != MAGIC:
        sys.exit(f"{path}: not an answers file")
    k, stride, n = struct.unpack_from("<IIQ", b, 8)
    size = 24 + 8 * k
    if len(b) != 24 + n * size:
        sys.exit(f"{path}: {len(b)} bytes for {n} answers of k={k}")
    out = []
    for i in range(n):
        off = 24 + i * size
        q, m, epoch, sched = struct.unpack_from("<IIQQ", b, off)
        out.append((q, epoch, sched, struct.unpack_from(f"<{m}I", b, off + 24),
                    struct.unpack_from(f"<{m}f", b, off + 24 + 4 * k)))
    return k, stride, out


def read_gt(path: str):
    """A big-ann ground truth: (rows, k, ids, dists or None), as vecio.ReadGroundTruth."""
    with open(path, "rb") as f:
        b = f.read()
    nq, k = struct.unpack_from("<II", b)
    ids = struct.unpack_from(f"<{nq * k}I", b, 8)
    dists = struct.unpack_from(f"<{nq * k}f", b, 8 + 4 * nq * k) if len(b) >= 8 + 8 * nq * k else None
    return nq, k, ids, dists


def recall(gt, row: int, ids, dists, k: int) -> float:
    """|answer ∩ top-k| / k, a result tied with the k-th distance counting too: vecio's Recall,
    which latency.csv's recall is."""
    _, K, gids, gd = gt
    k = min(k, K)
    truth = set(gids[row * K: row * K + k])
    kth = gd[row * K + k - 1] if gd else None
    hit = sum(1 for i in range(min(k, len(ids))) if ids[i] in truth or (kth is not None and dists[i] <= kth))
    return hit / k


def events_of(run: str):
    """The run's committed batches, in order (to_epoch, inserts, first ID, count, deletes), and
    the to_epochs of its major epochs."""
    path = os.path.join(run, "updates.json")
    if not os.path.exists(path):
        return [], []
    batches, majors = [], []
    for e in json.load(open(path)):
        b, m, up = e.get("batch") or {}, e.get("major") or {}, e.get("update") or {}
        if b.get("to_epoch", 0) > b.get("from_epoch", 0):
            ids = e.get("ids") or [0, 0]
            inserts = at_root(up["Inserts"]) if e.get("inserted") and up.get("Inserts") else None
            deletes = at_root(up["Deletes"]) if up.get("Deletes") else None
            batches.append({"to_epoch": b["to_epoch"], "inserts": inserts, "first": ids[0], "count": ids[1] - ids[0],
                            "deletes": deletes, "digests": {inserts: up.get("InsertsDigest"),
                                                            deletes: up.get("DeletesDigest")}})
            if batches[-1]["inserts"] and batches[-1]["count"] != e["inserted"]:
                sys.exit(f"{path}: a batch inserted {e['inserted']} vectors, IDs {ids}")
        if m.get("to_epoch", 0) > m.get("from_epoch", 0):
            majors.append(m["to_epoch"])
    return batches, majors


def rebuilds_of(run: str) -> list:
    """The epochs from which each full rebuild's next generation answered (reconfigurations.json:
    the reply's to_epoch). The data stays, the index changes."""
    path = os.path.join(run, "reconfigurations.json")
    if not os.path.exists(path):
        return []
    out = []
    for e in json.load(open(path)):
        r = e.get("reply") or {}
        if (e.get("rebuild") or "generation" in r) and r.get("to_epoch"):
            out.append(r["to_epoch"])
    return out


class Versions:
    """The ground truth of each data version of one experiment, made once and kept by content."""

    def __init__(self, cfg: dict, gt_dir: str, threads: int):
        self.cfg = cfg
        self.base_path = at_root(cfg["GroundTruth"])
        self.base = read_gt(self.base_path)
        self.dir = gt_dir or self.base_path + ".versions"
        self.threads = threads
        self.digests, self.loaded = {}, {}
        self.base_digest = self.digest(self.base_path)

    def digest(self, path: str) -> str:
        if path not in self.digests:
            self.digests[path] = rl.file_digest(path)
        return self.digests[path]

    def key(self, batches: list) -> str:
        """The data version after batches: its content, not where its files are."""
        k = {"base_gt": self.base_digest, "batches": [
            [self.digest(b["inserts"]) if b["inserts"] else None, b["first"] if b["inserts"] else None,
             self.digest(b["deletes"]) if b["deletes"] else None] for b in batches]}
        return hashlib.sha256(json.dumps(k, sort_keys=True).encode()).hexdigest()[:16]

    def gt(self, batches: list):
        """The ground truth after batches (the base's for none), made if it is not there."""
        if not batches:
            return self.base
        key = self.key(batches)
        if key in self.loaded:
            return self.loaded[key]
        path = os.path.join(self.dir, key + ".ibin")
        if not os.path.exists(path):
            self.make(path, batches)
        g = read_gt(path)
        if g[0] != self.base[0] or g[1] != self.base[1]:
            sys.exit(f"{path}: {g[0]} rows of {g[1]}, the base's ground truth {self.base[0]} of {self.base[1]}")
        self.loaded[key] = g
        return g

    def make(self, path: str, batches: list) -> None:
        cfg = self.cfg
        rows, k = self.base[0], self.base[1]
        stride = int(cfg.get("GroundTruthStride", 1))
        ebin = os.path.join(rl.ROOT, cfg.get("EngineBin", "engine/build"))
        cmd = [os.path.join(ebin, "fusion_gt")]
        base = at_root(cfg["Base"]) if cfg.get("Base") else ""
        if base and os.path.isfile(base):
            cmd += ["--base", base] + (["--max-n", str(rl.build_max_n(cfg))] if rl.build_max_n(cfg) else [])
        else:  # the base file is gone: the index's page file holds every vector (fusion_gt --index)
            index = at_root(cfg.get("IndexDir") or os.path.join(cfg.get("WorkDir", "/tmp/rtier-run"), "index"))
            cmd += ["--index", index]
        # The base's rows: the first (rows - 1) * stride + 1 queries, every stride-th.
        cmd += ["--queries", at_root(cfg["Queries"]), "--stride", str(stride), "--k", str(k),
                "--nq", str((rows - 1) * stride + 1)]
        excl = [b["deletes"] for b in batches if b["deletes"]]
        ins = [f"{b['inserts']}@{b['first']}" for b in batches if b["inserts"]]
        if excl:
            cmd += ["--exclude", ",".join(excl)]
        if ins:
            cmd += ["--insert", ",".join(ins)]
        if self.threads:
            cmd += ["--threads", str(self.threads)]
        os.makedirs(self.dir, exist_ok=True)
        tmp = path + ".tmp"
        print(f"ground truth of data version {len(batches)}: " + " ".join(cmd[1:]), flush=True)
        subprocess.run(cmd + ["--out", tmp], check=True)
        os.replace(tmp, path)
        json.dump({"base_ground_truth": self.base_path, "batches": batches, "command": cmd},
                  open(path[:-len(".ibin")] + ".json", "w"), indent=2)


def score(run: str, versions: Versions, groups: dict) -> dict:
    """Scores one run; adds its answers to groups[(data version, majors, query)] -> {answer: [runs]}."""
    k, stride, answers = read_answers(os.path.join(run, "answers.bin"))
    want = int(versions.cfg.get("GroundTruthStride", 1))
    if stride != want:
        sys.exit(f"{run}: answers kept every {stride}-th query, the ground truth has every {want}-th")
    batches, majors = events_of(run)
    rebuilds = rebuilds_of(run)
    # The files must be the ones the run applied: the ground truth is made of them.
    for b in batches:
        for path, digest in b["digests"].items():
            if path and digest and versions.digest(path) != digest:
                sys.exit(f"{run}: {path} changed since the run applied it")
            if path and not digest:
                print(f"{run}: updates.json has no digest of {path} (a run before 2026-11-17): taken as it is")

    def data_version(epoch: int) -> int:
        return sum(1 for b in batches if b["to_epoch"] <= epoch)

    gt_of, key_of = {}, {}  # by data version: its ground truth, and its content key

    def version(v: int):
        if v not in gt_of:
            gt_of[v] = versions.gt(batches[:v])
            key_of[v] = versions.key(batches[:v]) if v else "base"
        return gt_of[v], key_of[v]

    by_version, by_epoch = defaultdict(lambda: [0, 0.0, 0.0]), defaultdict(lambda: [0, 0.0, 0.0])
    same = defaultdict(set)  # (data version, majors, query) -> the answers in this run
    for q, epoch, _, ids, dists in answers:
        v = data_version(epoch)
        row = q // stride
        if q % stride or row >= versions.base[0]:
            sys.exit(f"{run}: query {q} has no ground-truth row")
        gt, key = version(v)
        r = recall(gt, row, ids, dists, k)
        r0 = recall(versions.base, row, ids, dists, k)
        for acc in (by_version[v], by_epoch[epoch]):
            acc[0] += 1
            acc[1] += r
            acc[2] += r0
        index = (key, sum(1 for t in majors if t <= epoch), sum(1 for t in rebuilds if t <= epoch), q)
        same[index].add(ids)
        groups[index].setdefault(ids, set()).add(run)
    first_epoch = {0: 1, **{i + 1: b["to_epoch"] for i, b in enumerate(batches)}}
    out = {
        "k": k, "stride": stride, "answers": len(answers),
        "versions": [{"version": v, "from_epoch": first_epoch[v], "answers": n,
                      "recall": round(s / n, 4), "recall_vs_base": round(s0 / n, 4)}
                     for v, (n, s, s0) in sorted(by_version.items())],
        "epochs": [{"epoch": e, "version": data_version(e), "answers": n, "recall": round(s / n, 4),
                    "recall_vs_base": round(s0 / n, 4)} for e, (n, s, s0) in sorted(by_epoch.items())],
        "queries_on_one_index": len(same),
        "queries_with_two_answers": sum(1 for a in same.values() if len(a) > 1),
    }
    json.dump(out, open(os.path.join(run, "scores.json"), "w"), indent=2)
    return out


def plan(path: str, gt_dir: str, threads: int) -> None:
    """The ground truth of every data version the experiment's batches make, if each commits."""
    cfg = json.load(open(path))
    if not cfg.get("GroundTruth"):
        sys.exit(f"{path}: the experiment has no GroundTruth")
    index = at_root(cfg.get("IndexDir") or os.path.join(cfg.get("WorkDir", "/tmp/rtier-run"), "index"))
    meta = dict(line.rstrip("\n").split("=", 1) for line in open(os.path.join(index, "meta.txt")) if "=" in line)
    versions = Versions(cfg, gt_dir, threads)
    batches, next_id = [], int(meta["num_vectors"])
    for rc in cfg.get("Reconfigurations", []):
        up = rc.get("Update") or {}
        if not (up.get("Inserts") or up.get("Deletes")):
            continue  # a rescale, a rebuild, or a major epoch alone: the data stays
        b = {"to_epoch": None, "inserts": None, "first": 0, "count": 0, "deletes": None}
        if up.get("Inserts"):
            b["inserts"] = at_root(up["Inserts"])
            with open(b["inserts"], "rb") as f:
                b["count"] = struct.unpack("<I", f.read(4))[0]
            b["first"], next_id = next_id, next_id + b["count"]
        if up.get("Deletes"):
            b["deletes"] = at_root(up["Deletes"])
        batches.append(b)
        versions.gt(batches)
        print(f"data version {len(batches)}: {os.path.join(versions.dir, versions.key(batches) + '.ibin')} "
              f"(inserts from ID {b['first']}, {b['count']} of them)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", help="run directories (answers.bin, updates.json, experiment.json)")
    ap.add_argument("--plan", default="", help="an experiment (JSON): make its data versions' ground truth now")
    ap.add_argument("--gt-dir", default="", help="where the versions' ground truth is kept "
                                                 "(default: <GroundTruth>.versions/)")
    ap.add_argument("--threads", type=int, default=0, help="fusion_gt's threads (default: all)")
    a = ap.parse_args()
    if a.plan:
        if a.runs:
            sys.exit("--plan takes an experiment, not runs")
        return plan(a.plan, a.gt_dir, a.threads)
    if not a.runs:
        sys.exit("give run directories, or --plan EXPERIMENT.json")
    cfg = json.load(open(os.path.join(a.runs[0], "experiment.json")))
    if not cfg.get("GroundTruth"):
        sys.exit(f"{a.runs[0]}: the experiment has no GroundTruth")
    versions = Versions(cfg, a.gt_dir, a.threads)
    groups = defaultdict(dict)
    for run in a.runs:
        other = json.load(open(os.path.join(run, "experiment.json")))
        for key in ("Base", "Queries", "GroundTruth", "GroundTruthStride", "BuildArgs"):
            if other.get(key) != cfg.get(key):
                sys.exit(f"{run}: another experiment's data ({key}) than {a.runs[0]}'s")
        s = score(run, versions, groups)
        print(f"{run}: {s['answers']} answers (k={s['k']}, every {s['stride']}-th query)")
        for v in s["versions"]:
            print(f"  data version {v['version']} (epoch {v['from_epoch']} on): {v['answers']} answers, "
                  f"recall {v['recall']:.4f} (against the base's ground truth {v['recall_vs_base']:.4f})")
        print(f"  {s['queries_with_two_answers']} of {s['queries_on_one_index']} queries got two answers on one index")
    if len(a.runs) > 1:
        shared = [g for g in groups.values() if len(set().union(*g.values())) > 1]
        differ = sum(1 for g in shared if len(g) > 1)
        print(f"across the {len(a.runs)} runs: {differ} of {len(shared)} queries scored on one index in more than "
              f"one run got different answers")


if __name__ == "__main__":
    main()
