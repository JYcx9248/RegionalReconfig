#!/usr/bin/env python3
"""TEST ONLY: assign posting lists to partitions with GP, the graph partitioning of Gottesbueren,
Dhulipala, Jayaram and Lacki, "Unleashing Graph Partitioning for Large-Scale Nearest Neighbor
Search" (PVLDB 18, VLDB '25). A candidate for open design question U2, not a decision.

GP partitions a point set in two steps:
  1. An approximate 10-nearest-neighbor graph, by "recursive splitting with dense ball
     clusters": sample pivots, put every point in the cluster of its closest pivot, split
     clusters again or, once small enough, compare all pairs in them.
  2. KaMinPar partitions that graph into k blocks of balanced size (epsilon = 5%) that cut as
     few edges as possible (gp/gp_kaminpar.cpp).

Parameters as the paper gives them (Section 2.1, Section 4, Appendix F.1):
  - k = 10 neighbors, squared L2 distance;
  - three independent repetitions;
  - at the top level, 950 pivots and every point assigned to its 3 closest; on the levels
    below, 0.005 x the points left as pivots and every point assigned to its closest;
  - clusters of at most 2500 points are compared all-pairs; a point keeps its 10 closest
    candidates over every cluster it was in;
  - epsilon = 5%.

Where the paper says nothing, its code decides (github.com/larsgottesbueren/gp-ann,
src/knn_graph.h and src/partitioning.cpp):
  - at least 3 and at most 1500 pivots;
  - clusters of fewer than 50 points are merged into buckets of up to 2500;
  - past depth 14, or past depth 10 if a cluster kept more than 80% of its parent, the
    cluster is shuffled and cut into chunks;
  - the graph is symmetrized by adding the reverse edges, with no weights.

Departures from that code, toward the paper or fixing a bug:
  - its repetitions reuse one seed and so repeat one clustering three times; here each
    repetition has its own seed;
  - its recursion stops at 5000 points; here at the paper's 2500;
  - a point that sits twice in a merged bucket (two of its top-level clusters were small)
    becomes its own neighbor there; here it is compared once;
  - its random split of near-duplicate clusters fills every chunk from the first one; here
    each chunk is its own slice.

rtier partitions posting lists, not points, so GP's blocks have to be mapped onto lists:
  --points heads    GP partitions the list heads (centroids), as Pyramid partitions the
                    bottom layer of its meta-HNSW; a list goes where its head goes.
  --points vectors  GP partitions the dataset's vectors, the paper's own input; a list goes
                    to the block holding the plurality of its postings.

Two departures for rtier, both for --points heads and both off by default:
  --graph-source index   instead of building a kNN graph, use the navigation graph's level-0
                         edges between the heads (the index's graph.hnsw, hnswlib's format;
                         --index-degree K keeps each head's first K, the closest): the graph
                         the build made anyway, which as the partition graph's weights did as
                         well as a kNN graph (U3).
  --weights postings     each head weighs its list's postings, so the partitions balance the
                         postings rather than the number of lists.
KaMinPar numbers its blocks along its recursive bisection, so consecutive blocks tend to be
neighbors and an even placement of consecutive partitions (placement.EvenPolicy) keeps a node's
partitions together. On BIGANN-1M (64 partitions, nprobe 64), shuffling the numbers raised the
owners per query on 4 nodes from 1.93 to 3.45.

    python3 scripts/testing/make_gp_assignment.py INDEX_DIR NUM_PARTITIONS OUT.bin \\
        --points heads|vectors [--base BASE.u8bin] --kaminpar PATH/gp_kaminpar \\
        [--graph GRAPH.bin] [--threads N]

--graph keeps the kNN graph: an existing file is read instead of building it again, so one
graph serves several NUM_PARTITIONS.
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

DTYPES = {"uint8": np.uint8, "int8": np.int8, "float": np.float32}
EXT = {".u8bin": np.uint8, ".i8bin": np.int8, ".fbin": np.float32}

K = 10                  # neighbors per point (Section 4)
REPETITIONS = 3         # independent repetitions (F.1)
FANOUT = 3              # closest pivots per point on the top level; 1 below (F.1)
TOP_LEADERS = 950       # pivots on the top level (F.1)
FRAC_LEADERS = 0.005    # pivots on lower levels, as a fraction of the points left (F.1)
MAX_CLUSTER = 2500      # clusters up to this size are compared all-pairs (F.1)
EPSILON = 0.05          # shard size imbalance (Section 4)
SEED = 555              # the paper's code's seed
# Not in the paper; values of the paper's code (src/knn_graph.h).
MIN_LEADERS, MAX_LEADERS = 3, 1500
MIN_CLUSTER, MAX_MERGED = 50, 2500
MAX_DEPTH, CONCERNING_DEPTH, SHRINK = 14, 10, 0.8
CHUNK = 16384           # rows per distance block (memory only)


def log(msg):
    print(msg, flush=True)


def load_index(index_dir):
    """Heads (as float32), and the lists as CSR (offsets, vector IDs)."""
    meta = dict(l.strip().split("=", 1) for l in open(os.path.join(index_dir, "meta.txt")) if "=" in l)
    with open(os.path.join(index_dir, "centroids.bin"), "rb") as f:
        n, dim = struct.unpack("<II", f.read(8))
        heads = np.fromfile(f, dtype=DTYPES[meta["dtype"]], count=n * dim).reshape(n, dim)
    with open(os.path.join(index_dir, "postings.bin"), "rb") as f:
        num_lists, num_ids = struct.unpack("<QQ", f.read(16))
        offsets = np.fromfile(f, dtype=np.uint64, count=num_lists + 1)
        ids = np.fromfile(f, dtype=np.uint32, count=num_ids)
    if num_lists != n or len(ids) != num_ids:
        sys.exit("centroids.bin has %d heads, postings.bin %d lists" % (n, num_lists))
    return heads.astype(np.float32), offsets.astype(np.int64), ids


def load_vectors(path):
    dtype = EXT.get(os.path.splitext(path)[1])
    if dtype is None:
        sys.exit("%s: want .u8bin, .i8bin or .fbin" % path)
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        return np.fromfile(f, dtype=dtype, count=n * d).reshape(n, d).astype(np.float32)


class KNNGraphBuilder:
    """Approximate kNN graph by recursive splitting with dense ball clusters (Section 2.1)."""

    def __init__(self, x):
        self.x = x
        self.sq = (x * x).sum(1)
        self.best_i = np.full((len(x), K), -1, dtype=np.int64)
        self.best_d = np.full((len(x), K), np.inf, dtype=np.float32)
        self.buckets = 0

    def closest(self, ids, leaders, fanout):
        """The fanout closest leaders (indexes into leaders) of every point in ids."""
        lx, lsq = self.x[leaders], self.sq[leaders]
        out = np.empty((len(ids), fanout), dtype=np.int64)
        for s in range(0, len(ids), CHUNK):
            idx = ids[s:s + CHUNK]
            d = self.sq[idx][:, None] - 2.0 * (self.x[idx] @ lx.T) + lsq[None, :]
            if fanout == 1:
                out[s:s + len(idx), 0] = d.argmin(1)
            else:
                out[s:s + len(idx)] = np.argpartition(d, fanout - 1, axis=1)[:, :fanout]
        return out

    def sketch(self, ids, depth, fanout, seed):
        """Splits ids into buckets of at most MAX_CLUSTER points (RecursivelySketch)."""
        if len(ids) <= MAX_CLUSTER:
            return [ids]
        num = TOP_LEADERS if depth == 0 else int(len(ids) * FRAC_LEADERS)
        num = max(min(num, MAX_LEADERS), MIN_LEADERS)
        # A uniform sample without replacement, in the order of ids (as std::sample).
        pos = np.sort(np.random.default_rng(seed).choice(len(ids), size=num, replace=False))
        leaders = ids[pos]
        lead = self.closest(ids, leaders, fanout).ravel()
        pts = np.repeat(ids, fanout)
        order = np.argsort(lead, kind="stable")
        clusters = np.split(pts[order], np.cumsum(np.bincount(lead, minlength=num))[:-1])
        clusters.sort(key=len, reverse=True)
        buckets, merged = [], []
        while clusters and len(clusters[-1]) < MIN_CLUSTER:
            c = clusters.pop()
            if not merged or len(c) + sum(len(p) for p in merged[-1]) > MAX_MERGED:
                merged.append([])
            merged[-1].append(c)
        buckets.extend(np.concatenate(m) for m in merged if sum(len(p) for p in m))
        for c in clusters:
            if depth > MAX_DEPTH or (depth > CONCERNING_DEPTH and len(c) > SHRINK * len(ids)):
                # Duplicates and near-duplicates: split at random.
                c = np.random.default_rng(seed + depth + len(ids)).permutation(c)
                buckets.extend(c[i:i + MAX_CLUSTER] for i in range(0, len(c), MAX_CLUSTER))
            else:
                buckets.extend(self.sketch(c, depth + 1, 1, seed))
        return buckets

    def crunch(self, b):
        """All pairs in bucket b; every point keeps its K closest seen so far."""
        # A point sits in a merged bucket twice if two of its top-level clusters were small
        # and merged together; the paper's code would then make it its own neighbor.
        b = b[np.sort(np.unique(b, return_index=True)[1])]
        if len(b) < 2:
            return
        xb, sb = self.x[b], self.sq[b]
        d = sb[:, None] - 2.0 * (xb @ xb.T) + sb[None, :]
        np.fill_diagonal(d, np.inf)
        kk = min(K, len(b) - 1)
        nn = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        ci = np.concatenate([self.best_i[b], b[nn]], 1)
        cd = np.concatenate([self.best_d[b], np.take_along_axis(d, nn, 1)], 1)
        o = np.argsort(ci, axis=1, kind="stable")
        si, sd = np.take_along_axis(ci, o, 1), np.take_along_axis(cd, o, 1)
        dup = np.zeros(si.shape, dtype=bool)
        dup[:, 1:] = (si[:, 1:] == si[:, :-1]) & (si[:, 1:] >= 0)
        sd[dup] = np.inf
        keep = np.argpartition(sd, K - 1, axis=1)[:, :K]
        self.best_i[b] = np.take_along_axis(si, keep, 1)
        self.best_d[b] = np.take_along_axis(sd, keep, 1)
        self.buckets += 1

    def build(self):
        all_ids = np.arange(len(self.x), dtype=np.int64)
        for rep in range(REPETITIONS):
            t = time.monotonic()
            buckets = self.sketch(all_ids, 0, FANOUT, SEED + rep)
            for b in buckets:
                self.crunch(b)
            sizes = np.array([len(b) for b in buckets])
            log("  repetition %d: %d buckets (avg %.0f, max %d points), %.1f s" %
                (rep, len(buckets), sizes.mean(), sizes.max(), time.monotonic() - t))
        # Each point's neighbors, closest first (as the paper's code lists them).
        order = np.argsort(self.best_d, axis=1, kind="stable")
        nb = np.take_along_axis(self.best_i, order, 1)
        valid = nb >= 0
        xadj = np.concatenate([[0], np.cumsum(valid.sum(1))]).astype(np.uint64)
        return xadj, nb[valid].astype(np.uint32)


def read_hnsw_level0(index_dir, degree):
    """The navigation graph's level-0 edges by label (list ID), as CSR: each head's links in
    the order hnswlib keeps them (closest first), at most degree of them (0: all). The file is
    hnswlib's saveIndex output (internal/navgraph reads it the same way): a 96-byte header, then
    one record per element -- the link count (low 16 bits of a u32), maxM0 link slots of
    internal IDs, the vector, the u64 label."""
    path = os.path.join(index_dir, "graph.hnsw")
    with open(path, "rb") as f:
        hdr = f.read(96)
        off0, _, count, size, label_off, _ = struct.unpack("<6Q", hdr[:48])
        maxm0 = struct.unpack("<Q", hdr[64:72])[0]
        if off0 != 0 or maxm0 == 0 or label_off + 8 > size:
            sys.exit("%s: not an hnswlib index" % path)
        rec = np.fromfile(f, dtype=np.uint8, count=count * size).reshape(count, size)
    n_links = rec[:, 0:4].copy().view("<u4")[:, 0] & 0xFFFF
    links = rec[:, 4:4 + 4 * maxm0].copy().view("<u4")
    labels = rec[:, label_off:label_off + 8].copy().view("<u8")[:, 0].astype(np.int64)
    if degree > 0:
        n_links = np.minimum(n_links, degree)
    keep = np.arange(maxm0)[None, :] < n_links[:, None]
    src = np.repeat(labels, keep.sum(1))
    dst = labels[links[keep]]
    order = np.argsort(src, kind="stable")  # by label, each head's links in their order
    src, dst = src[order], dst[order]
    xadj = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=count))]).astype(np.uint64)
    return xadj, dst.astype(np.uint32)


def write_graph(path, xadj, adj):
    with open(path, "wb") as f:
        f.write(struct.pack("<QQ", len(xadj) - 1, len(adj)))
        f.write(xadj.astype("<u8").tobytes())
        f.write(adj.astype("<u4").tobytes())


def read_graph(path):
    """A graph written by write_graph, without self-loops (older files may have a few)."""
    with open(path, "rb") as f:
        n, m = struct.unpack("<QQ", f.read(16))
        xadj = np.fromfile(f, dtype="<u8", count=n + 1).astype(np.int64)
        adj = np.fromfile(f, dtype="<u4", count=m)
    src = np.repeat(np.arange(n), np.diff(xadj))
    keep = src != adj
    xadj = np.concatenate([[0], np.cumsum(np.bincount(src[keep], minlength=n))]).astype(np.uint64)
    return xadj, adj[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("index_dir")
    ap.add_argument("partitions", type=int)
    ap.add_argument("out")
    ap.add_argument("--points", choices=["heads", "vectors"], required=True)
    ap.add_argument("--base", help="the dataset's vectors (--points vectors)")
    ap.add_argument("--kaminpar", required=True, help="gp_kaminpar (scripts/testing/gp)")
    ap.add_argument("--graph", help="kNN graph file: read if it exists, else written")
    ap.add_argument("--graph-source", choices=["knn", "index"], default="knn",
                    help="index: the navigation graph's level-0 edges instead of a kNN graph (--points heads)")
    ap.add_argument("--index-degree", type=int, default=0, help="with --graph-source index: links kept per head (0: all)")
    ap.add_argument("--weights", choices=["unit", "postings"], default="unit",
                    help="postings: a head weighs its list's postings (--points heads)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    if a.partitions < 2:
        sys.exit("NUM_PARTITIONS must be at least 2")
    if a.points == "vectors" and not a.base:
        sys.exit("--points vectors needs --base")
    if a.points == "vectors" and (a.graph_source != "knn" or a.weights != "unit"):
        sys.exit("--graph-source index and --weights postings are for --points heads")

    heads, offsets, ids = load_index(a.index_dir)
    sizes = np.diff(offsets)
    if a.graph_source == "index":
        t = time.monotonic()
        xadj, adj = read_hnsw_level0(a.index_dir, a.index_degree)
        log("navigation graph level 0: %d heads, %d edges (%s per head), %.1f s" %
            (len(xadj) - 1, len(adj), a.index_degree or "all", time.monotonic() - t))
    elif a.graph and os.path.exists(a.graph):
        xadj, adj = read_graph(a.graph)
        log("kNN graph from %s: %d points, %d edges" % (a.graph, len(xadj) - 1, len(adj)))
    else:
        x = heads if a.points == "heads" else load_vectors(a.base)
        log("kNN graph (k = %d) over %d %s, %d dims" % (K, len(x), a.points, x.shape[1]))
        t = time.monotonic()
        xadj, adj = KNNGraphBuilder(x).build()
        log("kNN graph: %d edges in %.1f s" % (len(adj), time.monotonic() - t))
        if a.graph:
            write_graph(a.graph, xadj, adj)
    n = len(xadj) - 1
    want = len(heads) if a.points == "heads" else int(ids.max()) + 1
    if n != want:
        sys.exit("the graph has %d points, the %s have %d" % (n, a.points, want))

    with tempfile.TemporaryDirectory() as tmp:
        gpath, ppath = os.path.join(tmp, "graph.bin"), os.path.join(tmp, "part.bin")
        write_graph(gpath, xadj, adj)
        cmd = [a.kaminpar, gpath, str(a.partitions), str(EPSILON), str(a.threads), ppath]
        if a.weights == "postings":
            wpath = os.path.join(tmp, "weights.bin")
            with open(wpath, "wb") as f:
                f.write(struct.pack("<Q", len(sizes)))
                f.write(sizes.astype("<u4").tobytes())
            cmd.append(wpath)
        t = time.monotonic()
        subprocess.run(cmd, check=True)
        log("KaMinPar: %.1f s" % (time.monotonic() - t))
        block = np.fromfile(ppath, dtype="<u4", count=n)
    src = np.repeat(np.arange(n), np.diff(xadj).astype(np.int64))
    log("GP cuts %.1f%% of the %d edges" % (100.0 * (block[src] != block[adj]).mean(), len(adj)))

    if a.points == "heads":
        out = block.astype(np.uint32)
    else:
        owner = np.repeat(np.arange(len(sizes)), sizes)
        counts = np.bincount(owner * a.partitions + block[ids], minlength=len(sizes) * a.partitions)
        counts = counts.reshape(len(sizes), a.partitions)
        out = counts.argmax(1).astype(np.uint32)  # the plurality; ties to the lowest block
        log("%.1f%% of the postings sit in the partition their vector's block maps to" %
            (100.0 * counts.max(1).sum() / max(1, len(ids))))
    post = np.bincount(out, weights=sizes, minlength=a.partitions)
    lists = np.bincount(out, minlength=a.partitions)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<Q", len(out)))
        f.write(out.astype("<u4").tobytes())
    log("%d lists -> %d partitions (TEST-ONLY GP assignment, --points %s, --graph-source %s, --weights %s): "
        "lists per partition %d..%d (max %.3f x average), postings per partition %d..%d (max %.3f x average) -> %s" %
        (len(out), a.partitions, a.points, a.graph_source, a.weights, lists.min(), lists.max(),
         lists.max() / lists.mean(), post.min(), post.max(), post.max() / post.mean(), a.out))


if __name__ == "__main__":
    main()
