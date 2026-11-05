#!/usr/bin/env python3
"""Queries drawn from an index's own vectors without repetition: U11's worst case for lazy, a
query stream with no locality at all (internal/design/design.go, U11).

    python3 scripts/data/sample_queries.py sample --index DIR --count N --out QUERIES.fbin
        [--seed 1] [--layout bucket]
    python3 scripts/data/sample_queries.py check --gt GT.ibin --ids QUERIES.fbin.ids [--stride S]

sample draws N distinct vector IDs, uniformly, and writes their vectors in the order drawn: a run
sends them in that order and repeats none while the file lasts. It reads them from the index's
page file (vectors_<layout>.bin, each vector's page and slot from layout_<layout>.map), so the
base file is not needed, and writes the IDs it drew beside the queries (QUERIES.fbin.ids:
little-endian uint32, one per query, no header). Their ground truth comes from the same page
file:

    engine/build/fusion_gt --index DIR --queries QUERIES.fbin --nq M --stride S --k 10 --out GT.ibin

check then counts the queries among their own nearest neighbours, as each must be: the distance
to itself is 0. Only a base that holds a vector more than k times leaves one out, and a vector
held twice can come first with the smaller ID. It exits 1 when fewer than 99% are found.

Standard library only, so that it runs on whichever machine holds the index.
"""

import argparse
import array
import os
import random
import struct
import sys
from concurrent.futures import ThreadPoolExecutor

# fbin and its kin (big-ann-benchmarks): uint32 rows, uint32 dimensions, then the rows
EXT = {"float": ".fbin", "uint8": ".u8bin", "int8": ".i8bin"}


def u32():
    a = array.array("I")
    assert a.itemsize == 4, "array('I') is not 4 bytes here"
    return a


def read_meta(index):
    kv = {}
    with open(os.path.join(index, "meta.txt")) as f:
        for line in f:
            k, _, v = line.rstrip("\n").partition("=")
            kv[k] = v
    return kv


def read_layout(index, name):
    """(page_size, vec_bytes, vectors per page, page_of, slot_of) of layout_<name>.map."""
    with open(os.path.join(index, "layout_%s.map" % name), "rb") as f:
        page_size, vec_bytes, vpp, _ = struct.unpack("<4I", f.read(16))
        n, _pages = struct.unpack("<2Q", f.read(16))
        page_of, slot_of = u32(), array.array("H")
        page_of.fromfile(f, n)
        slot_of.fromfile(f, n)
    if sys.byteorder != "little":
        page_of.byteswap()
        slot_of.byteswap()
    return page_size, vec_bytes, vpp, page_of, slot_of


def sample(args):
    meta = read_meta(args.index)
    dtype, dim = meta.get("dtype"), int(meta["dim"])
    if dtype not in EXT:
        sys.exit("sample: the index's vectors are %r: want one of %s" % (dtype, ", ".join(EXT)))
    if not args.out.endswith(EXT[dtype]):
        sys.exit("sample: %s vectors go to a %s file, not %s" % (dtype, EXT[dtype], args.out))
    page_size, vb, _vpp, page_of, slot_of = read_layout(args.index, args.layout)
    n = len(page_of)
    if not 0 < args.count <= n:
        sys.exit("sample: --count %d, the index has %d vectors" % (args.count, n))
    ids = random.Random(args.seed).sample(range(n), args.count)
    pages = os.path.join(args.index, "vectors_%s.bin" % args.layout)
    fd = os.open(pages, os.O_RDONLY)

    def read(i):
        off = page_of[i] * page_size + slot_of[i] * vb
        b = os.pread(fd, vb, off)
        if len(b) != vb:
            raise OSError("%s: short read at %d" % (pages, off))
        return b

    tmp = args.out + ".partial"  # named once complete, as fusion_build's files are
    with open(tmp, "wb") as out, ThreadPoolExecutor(32) as ex:
        out.write(struct.pack("<II", args.count, dim))
        for k in range(0, len(ids), 1 << 16):
            for b in ex.map(read, ids[k:k + (1 << 16)]):
                out.write(b)
            print("  %d / %d" % (min(k + (1 << 16), len(ids)), len(ids)), file=sys.stderr)
    os.close(fd)
    drawn = u32()
    drawn.extend(ids)
    if sys.byteorder != "little":
        drawn.byteswap()
    with open(args.out + ".ids", "wb") as f:
        drawn.tofile(f)
    os.rename(tmp, args.out)
    print("%s: %d of the %d vectors (%d bytes each), drawn with seed %d; their IDs in %s.ids"
          % (args.out, args.count, n, vb, args.seed, args.out))


def check(args):
    with open(args.gt, "rb") as f:
        nq, k = struct.unpack("<II", f.read(8))
        gt = u32()
        gt.fromfile(f, nq * k)
    drawn = u32()
    with open(args.ids, "rb") as f:
        drawn.frombytes(f.read())
    if sys.byteorder != "little":
        gt.byteswap()
        drawn.byteswap()
    if (nq - 1) * args.stride >= len(drawn):
        sys.exit("check: %d rows of ground truth at stride %d need %d queries, %s has %d"
                 % (nq, args.stride, (nq - 1) * args.stride + 1, args.ids, len(drawn)))
    first = inside = 0
    for r in range(nq):
        me, row = drawn[r * args.stride], gt[r * k:(r + 1) * k]
        first += row[0] == me
        inside += me in row
    print("%s: %d queries (one in %d), each its own nearest neighbour: %d (%.2f%%); among its %d nearest: %d (%.2f%%)"
          % (args.gt, nq, args.stride, first, 100.0 * first / nq, k, inside, 100.0 * inside / nq))
    sys.exit(0 if inside >= 0.99 * nq else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="draw the queries from an index's page file")
    s.add_argument("--index", required=True, help="the index directory (meta.txt, layout_<layout>.map, vectors_<layout>.bin)")
    s.add_argument("--count", type=int, required=True, help="queries to draw, no two the same")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--layout", default="bucket")
    s.add_argument("--out", required=True, help="the queries, .fbin for float vectors")
    c = sub.add_parser("check", help="each query among its own nearest neighbours in a ground truth")
    c.add_argument("--gt", required=True)
    c.add_argument("--ids", required=True, help="the IDs sample drew (QUERIES.fbin.ids)")
    c.add_argument("--stride", type=int, default=1, help="the ground truth's stride (fusion_gt --stride)")
    args = ap.parse_args()
    (sample if args.cmd == "sample" else check)(args)


if __name__ == "__main__":
    main()
