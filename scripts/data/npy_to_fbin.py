#!/usr/bin/env python3
"""Convert embedding shards (.npy, one 2-D array per file, e.g. float16) into one .fbin -- u32
count, u32 dim, then float32 rows -- the base-vector format fusion_build, fusion_gt and the
load generator read.

Shards are taken in natural file-name order (shard_2 before shard_10), so row i of the output
is row i of their concatenation; the shard list and the row count are printed, check them
against the dataset's. Rows are renormalized to unit length (unless --keep-norms): embeddings
stored as unit vectors in float16 come back a few 1e-4 off, and with unit vectors the engine's
L2 ranks neighbours exactly as inner product does. --sample N keeps N rows drawn uniformly
without replacement, in their original order, and --ids writes their original row numbers
(u64 count, then u64 ids), so that a subset can be traced back to its chunks.

    python3 scripts/data/npy_to_fbin.py SHARD_DIR OUT.fbin [--sample N] [--seed S] [--ids OUT.ids]
"""

import argparse
import os
import re
import struct
import sys

import numpy as np

CHUNK = 1 << 18  # rows converted at a time


def natural_key(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def shards(d: str) -> list:
    out = []
    for name in sorted((n for n in os.listdir(d) if n.endswith(".npy")), key=natural_key):
        a = np.load(os.path.join(d, name), mmap_mode="r")
        if a.ndim != 2:
            print(f"skipping {name}: shape {a.shape}", file=sys.stderr)
            continue
        out.append((name, a))
    if not out:
        sys.exit(f"no 2-D .npy shards in {d}")
    dims = {a.shape[1] for _, a in out}
    if len(dims) != 1:
        sys.exit(f"shards disagree on the dimension: {sorted(dims)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("shard_dir")
    ap.add_argument("out")
    ap.add_argument("--sample", type=int, default=0, help="keep this many rows, drawn uniformly")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ids", help="with --sample: write the kept rows' original numbers here")
    ap.add_argument("--keep-norms", action="store_true", help="do not renormalize the rows")
    args = ap.parse_args()

    sh = shards(args.shard_dir)
    total, dim = sum(a.shape[0] for _, a in sh), sh[0][1].shape[1]
    print(f"{len(sh)} shards ({sh[0][0]} ... {sh[-1][0]}), {total:,} rows x {dim}, {sh[0][1].dtype}")
    if args.sample:
        if args.sample > total:
            sys.exit(f"--sample {args.sample} is more than the {total} rows")
        keep = np.sort(np.random.default_rng(args.seed).choice(total, args.sample, replace=False))
    else:
        keep = None
    n_out = args.sample or total

    norms = [np.inf, 0.0]
    written, base = 0, 0
    with open(args.out, "wb") as f:
        f.write(struct.pack("<II", n_out, dim))
        for name, a in sh:
            rows = a.shape[0]
            if keep is not None:
                lo, hi = np.searchsorted(keep, [base, base + rows])
                sel = keep[lo:hi] - base
            for s in range(0, rows if keep is None else len(sel), CHUNK):
                x = np.asarray(a[s:s + CHUNK] if keep is None else a[sel[s:s + CHUNK]], dtype=np.float32)
                nrm = np.linalg.norm(x, axis=1)
                if len(nrm):
                    norms = [min(norms[0], float(nrm.min())), max(norms[1], float(nrm.max()))]
                if not args.keep_norms:
                    if (nrm == 0).any():
                        sys.exit(f"{name}: a zero vector cannot be normalized (use --keep-norms)")
                    x /= nrm[:, None]
                x.tofile(f)
                written += len(x)
            base += rows
            print(f"  {name}: {written:,} of {n_out:,} rows written", flush=True)
    if written != n_out:
        sys.exit(f"wrote {written} rows, expected {n_out}")
    print(f"{args.out}: {n_out:,} x {dim} float32 ({os.path.getsize(args.out) / 1e9:.1f} GB); "
          f"input norms {norms[0]:.5f} .. {norms[1]:.5f}" +
          ("" if args.keep_norms else ", renormalized"))
    if args.ids and keep is not None:
        with open(args.ids, "wb") as f:
            f.write(struct.pack("<Q", len(keep)))
            keep.astype(np.uint64).tofile(f)


if __name__ == "__main__":
    main()
