#!/usr/bin/env python3
"""A uniform random subset of a base (.i8bin, .u8bin or .fbin: u32 count, u32 dim, then the rows):
exactly N rows, drawn with a fixed seed and kept in their order, for a dataset cut to what the
GPUs hold (the Amazon corpus to the P100's 12 GB).

    python3 scripts/data/subset_base.py $V/corpus.i8bin $V/corpus-300m.i8bin --n 300000000

Rows are drawn one by one, not a chunk of the embedding at a time: the base follows the corpus's
order, one category's file after another, so whole chunks would drop runs of one category.
OUT.rows.npy lists the source row of each kept row (uint32, ascending), and OUT.json how the
vectors were made: the source's own record (its scale too, for --scale-from) and the subset's.
The rows file, not the seed, is what makes the subset again, since numpy's generators may
change. OUT appears only once it is complete.
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

ITEM = {".i8bin": 1, ".u8bin": 1, ".fbin": 4}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base", help="the source base")
    ap.add_argument("out", help="the subset, in the same format")
    ap.add_argument("--n", type=int, required=True, help="how many rows to keep")
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--chunk", type=int, default=1 << 20, help="rows copied at a time")
    args = ap.parse_args()

    ext = os.path.splitext(args.base)[1]
    if ext not in ITEM or os.path.splitext(args.out)[1] != ext:
        sys.exit(f"base and out must both be one of {', '.join(ITEM)}")
    with open(args.base, "rb") as f:
        count, dim = struct.unpack("<II", f.read(8))
    row = dim * ITEM[ext]
    if os.path.getsize(args.base) != 8 + count * row:
        sys.exit(f"{args.base}: {os.path.getsize(args.base)} bytes, not 8 + {count} x {row}")
    if not 0 < args.n <= count:
        sys.exit(f"--n must be in 1..{count}")
    record = {}
    if os.path.exists(args.base + ".json"):
        with open(args.base + ".json") as f:
            record = json.load(f)
        # embed_texts.py's output holds rows of zeros until every chunk is in.
        if record.get("complete") is False:
            sys.exit(f"{args.base}.json: not complete")

    log(f"{args.base}: {count} rows of {dim}; drawing {args.n} (seed {args.seed})")
    rows = np.random.default_rng(args.seed).permutation(count)[: args.n]
    rows.sort()
    rows = rows.astype(np.uint32)
    src = np.memmap(args.base, dtype=np.uint8, mode="r", offset=8, shape=(count, row))

    tmp = args.out + ".partial"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<II", args.n, dim))
        step = max(1, args.n // 20)
        for i in range(0, args.n, args.chunk):
            f.write(src[rows[i : i + args.chunk]].tobytes())
            if (i + args.chunk) // step > i // step:
                log(f"{min(i + args.chunk, args.n)} of {args.n} rows")
    np.save(args.out + ".rows.npy", rows)
    record["subset"] = {
        "of": os.path.abspath(args.base), "of_count": count, "n": args.n, "seed": args.seed,
        "rows": os.path.basename(args.out) + ".rows.npy", "numpy": np.__version__,
        "made": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.out + ".json", "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, args.out)
    log(f"{args.out}: {args.n} rows, {os.path.getsize(args.out)} bytes")


if __name__ == "__main__":
    main()
