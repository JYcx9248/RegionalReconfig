#!/usr/bin/env python3
"""TEST ONLY: assign posting lists to partitions by where their centroids are.

How lists should be grouped into partitions is open design question U2; this is a placeholder
for experiments, like make_range_assignment.py, not a decision. Lists are split in two by a
2-means over their centroids, each half again, and so on (recursive balanced bisection,
balanced by postings, so the segments come out the same size). Partitions are numbered in
the order of the tree's leaves, so any run of consecutive partitions covers a region of the
space: the even placement gives each node such a run, and a scale-out takes the tail of each
run, a sub-region too. A query's nearby lists then fall into few partitions and few nodes.

    python3 scripts/testing/make_spatial_assignment.py INDEX_DIR NUM_PARTITIONS OUT.bin
"""

import os
import struct
import sys

import numpy as np

DTYPES = {"uint8": np.uint8, "int8": np.int8, "float": np.float32}


def load_index(index_dir):
    meta = dict(l.strip().split("=", 1) for l in open(os.path.join(index_dir, "meta.txt")) if "=" in l)
    with open(os.path.join(index_dir, "centroids.bin"), "rb") as f:
        n, dim = struct.unpack("<II", f.read(8))
        cent = np.fromfile(f, dtype=DTYPES[meta["dtype"]], count=n * dim).reshape(n, dim)
    with open(os.path.join(index_dir, "postings.bin"), "rb") as f:
        num_lists, _ = struct.unpack("<QQ", f.read(16))
        offsets = np.fromfile(f, dtype=np.uint64, count=num_lists + 1)
    if num_lists != n:
        sys.exit("centroids.bin has %d heads, postings.bin %d lists" % (n, num_lists))
    return cent.astype(np.float32), np.diff(offsets).astype(np.float64)


def two_means(x, rng, iters=12):
    """Two centers of x (rows), by Lloyd's iterations from two distant points."""
    a = x[rng.integers(len(x))]
    b = x[np.argmax(((x - a) ** 2).sum(1))]
    c = np.stack([x[np.argmax(((x - b) ** 2).sum(1))], b])
    for _ in range(iters):
        d = (x * x).sum(1)[:, None] - 2 * x @ c.T + (c * c).sum(1)[None, :]
        lab = d.argmin(1)
        for j in (0, 1):
            if (lab == j).any():
                c[j] = x[lab == j].mean(0)
    return c


def bisect(x, w, idx, parts, first, out, rng):
    """Assigns lists idx (rows of x, weights w) to partitions first .. first + parts - 1."""
    if parts == 1:
        out[idx] = first
        return
    left = parts // 2
    xs = x[idx]
    c = two_means(xs, rng)
    # Order by how much closer a list is to the first center than to the second, and cut at
    # the weight the left half should get: balanced, and each side still a region.
    score = ((xs - c[0]) ** 2).sum(1) - ((xs - c[1]) ** 2).sum(1)
    order = np.argsort(score, kind="stable")
    cum = np.cumsum(w[idx][order])
    cut = int(np.searchsorted(cum, cum[-1] * left / parts))
    cut = min(max(cut, left), len(idx) - (parts - left))  # every partition gets at least one list
    bisect(x, w, idx[order[:cut]], left, first, out, rng)
    bisect(x, w, idx[order[cut:]], parts - left, first + left, out, rng)


def main(index_dir, parts, out_path):
    x, w = load_index(index_dir)
    out = np.zeros(len(x), dtype=np.uint32)
    bisect(x, w, np.arange(len(x)), parts, 0, out, np.random.default_rng(1))
    sizes = np.bincount(out, weights=w, minlength=parts)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(x)))
        f.write(out.astype("<u4").tobytes())
    print("%d lists -> %d partitions (TEST-ONLY spatial assignment), postings per partition "
          "min %d max %d -> %s" % (len(x), parts, sizes.min(), sizes.max(), out_path))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: make_spatial_assignment.py INDEX_DIR NUM_PARTITIONS OUT.bin")
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
