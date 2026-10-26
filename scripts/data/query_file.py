#!/usr/bin/env python3
"""A query file (.i8bin, like the base) for the load generator, fusion_gt and fusion_search: the
vectors of a stretch of AmazonQAC's search stream, in time order, a term searched twice coming
twice -- the log's own repetition and skew.

    python3 scripts/data/query_file.py VEC/queries-stream.npy VEC/queries.i8bin OUT.i8bin
    python3 scripts/data/query_file.py ... OUT.i8bin --from 86400 --seconds 3600 --first 500000

STREAM and TERMS are amazon_texts.py's stream and embed_texts.py's vectors of the same terms.
--from and --seconds pick the searches by time (seconds since 2023-09-01 00:00:00), --first the
first N of those; nothing picks the whole stream. OUT.i8bin.json records the stretch.
"""

import argparse
import json
import os
import struct
import sys

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stream", help="queries-stream.npy (amazon_texts.py queries)")
    ap.add_argument("terms", help="the terms' vectors (embed_texts.py, .i8bin)")
    ap.add_argument("out", help="the query file (.i8bin)")
    ap.add_argument("--from", dest="start", type=int, default=0, help="seconds since 2023-09-01 00:00:00")
    ap.add_argument("--seconds", type=int, help="how long a stretch (default: to the end)")
    ap.add_argument("--first", type=int, help="at most this many searches")
    args = ap.parse_args()

    try:
        with open(args.terms + ".json") as f:
            meta = json.load(f)
    except FileNotFoundError:
        meta = {}
    if not meta.get("complete"):
        sys.exit(f"{args.terms}.json does not say complete: the vectors are not all written yet")
    with open(args.terms, "rb") as f:
        n, dim = struct.unpack("<II", f.read(8))
    vecs = np.memmap(args.terms, dtype=np.int8, mode="r", offset=8, shape=(n, dim))
    stream = np.load(args.stream, mmap_mode="r")
    t = stream["t"]
    lo = int(np.searchsorted(t, args.start, "left"))
    hi = len(t) if args.seconds is None else int(np.searchsorted(t, args.start + args.seconds, "left"))
    if args.first is not None:
        hi = min(hi, lo + args.first)
    picked = stream[lo:hi]
    if not len(picked):
        sys.exit("no search in that stretch")
    terms = np.asarray(picked["term"], dtype=np.int64)
    if terms.max() >= n:
        sys.exit(f"{args.stream} names term {terms.max()}, {args.terms} has {n}: not each other's")
    with open(args.out, "wb") as f:
        f.write(struct.pack("<II", len(terms), dim))
        np.ascontiguousarray(vecs[terms]).tofile(f)
    rec = {"stream": os.path.abspath(args.stream), "terms": os.path.abspath(args.terms),
           "searches": len(terms), "distinct": int(len(np.unique(terms))),
           "first_t": int(picked["t"][0]), "last_t": int(picked["t"][-1]),
           "from": args.start, "seconds": args.seconds, "first": args.first}
    with open(args.out + ".json", "w") as f:
        json.dump(rec, f, indent=2)
    print(f"{args.out}: {rec['searches']:,} searches ({rec['distinct']:,} distinct terms), "
          f"t {rec['first_t']} to {rec['last_t']} s")


if __name__ == "__main__":
    main()
