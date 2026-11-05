#!/usr/bin/env python3
"""Queries drawn from an index's own vectors without repetition: U11's worst case for lazy, a
query stream with no locality at all (internal/design/design.go, U11).

    python3 scripts/data/sample_queries.py sample --index DIR --count N --out QUERIES.fbin
        [--seed 1] [--layout bucket]
    python3 scripts/data/sample_queries.py check --gt GT.ibin --ids QUERIES.fbin.ids [--stride S]
    python3 scripts/data/sample_queries.py join --index DIR --snapshot SNAPDIR --out DIR [--layout bucket]

join is for a machine that has the index's layout map but not its page file, such as a
cluster's control node, which runs on a warmup snapshot. Each node's raw.pages in the snapshot
is a sparse copy of the page file at the same offsets, and its presence bits (raw.pages.present)
mark the locations it holds. Together the nodes hold every vector. join writes the page file
again into DIR, from the first node that holds each location, with copies of the index's meta.txt
and layout map: DIR then serves as --index for sample and for fusion_gt. On a file system with
shared extents (XFS), the copies take no space.

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
import glob
import os
import random
import re
import shutil
import struct
import sys
import zlib
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


def read_layout(index, name, pages=False):
    """(page_size, vec_bytes, vectors per page, page_of, slot_of) of layout_<name>.map, and with
    pages its page count after them."""
    with open(os.path.join(index, "layout_%s.map" % name), "rb") as f:
        page_size, vec_bytes, vpp, _ = struct.unpack("<4I", f.read(16))
        n, num_pages = struct.unpack("<2Q", f.read(16))
        page_of, slot_of = u32(), array.array("H")
        page_of.fromfile(f, n)
        slot_of.fromfile(f, n)
    if sys.byteorder != "little":
        page_of.byteswap()
        slot_of.byteswap()
    return (page_size, vec_bytes, vpp, page_of, slot_of) + ((num_pages,) if pages else ())


def read_bits(path, page_size, vb, vpp):
    """A raw store's presence bits (RawStore::SaveBitmap): magic, version and geometry, pages,
    words, then a bit per location in 64-bit words, and an IEEE CRC-32 of all that."""
    with open(path, "rb") as f:
        b = f.read()
    if len(b) < 44 or b[:8] != b"RTRAWBIT":
        sys.exit("%s: not a raw store's presence bits" % path)
    if zlib.crc32(b[:-4]) != struct.unpack("<I", b[-4:])[0]:
        sys.exit("%s: checksum mismatch" % path)
    version, bvb, bvpp, bps = struct.unpack("<4I", b[8:24])
    _pages, nwords = struct.unpack("<2Q", b[24:40])
    if version != 1 or (bvb, bvpp, bps) != (vb, vpp, page_size) or len(b) != 40 + 8 * nwords + 4:
        sys.exit("%s: version %d, %d-byte vectors, %d a page of %d bytes, %d words in %d bytes: not this "
                 "layout's (%d-byte vectors, %d a page of %d)" % (path, version, bvb, bvpp, bps, nwords, len(b),
                                                                 vb, vpp, page_size))
    words = array.array("Q")
    assert words.itemsize == 8, "array('Q') is not 8 bytes here"
    words.frombytes(b[40:40 + 8 * nwords])
    if sys.byteorder != "little":
        words.byteswap()
    return words


def copy_range(src, dst, off, n):
    """Bytes [off, off + n) of src to the same place in dst: shared extents where the file system
    can, else a kernel copy, else through here."""
    while n > 0:
        try:
            k = os.copy_file_range(src, dst, n, off, off)
        except OSError:
            k = os.pwrite(dst, os.pread(src, min(n, 1 << 24), off), off)
        if k <= 0:
            raise OSError("copy stopped at byte %d" % off)
        off, n = off + k, n - k


def join(args):
    read_meta(args.index)  # an index directory
    page_size, vb, vpp, page_of, slot_of, num_pages = read_layout(args.index, args.layout, pages=True)
    nodes = sorted(glob.glob(os.path.join(args.snapshot, "n*", "raw.pages")),
                   key=lambda p: int(re.sub(r"\D", "", os.path.basename(os.path.dirname(p))) or 0))
    if not nodes:
        sys.exit("join: no n*/raw.pages under %s" % args.snapshot)
    nwords = (num_pages * vpp + 63) // 64
    have = array.array("Q", bytes(8 * nwords))
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "vectors_%s.bin" % args.layout)
    tmp = out + ".partial"
    dst = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(dst, num_pages * page_size)
    whole = vpp * vb == page_size  # consecutive locations are consecutive bytes, across pages too

    def flush(src, start, count):  # locations [start, start + count)
        while count:
            run = count if whole else min(count, vpp - start % vpp)  # within one page
            copy_range(src, dst, (start // vpp) * page_size + (start % vpp) * vb, run * vb)
            start, count = start + run, count - run

    for raw in nodes:
        words = read_bits(raw + ".present", page_size, vb, vpp)
        src = os.open(raw, os.O_RDONLY)
        took, start, count = 0, 0, 0  # the run of locations this node holds and no earlier one did
        for w in range(min(len(words), nwords)):
            new = words[w] & ~have[w]
            have[w] |= words[w]
            while new:
                lo = (new & -new).bit_length() - 1
                x = new >> lo
                ones = ((~x) & (x + 1)).bit_length() - 1  # the run of ones from bit lo
                loc = w * 64 + lo
                if count and loc == start + count:
                    count += ones
                else:
                    if count:
                        flush(src, start, count)
                    start, count = loc, ones
                took += ones
                new &= ~(((1 << ones) - 1) << lo)
        if count:
            flush(src, start, count)
        os.close(src)
        print("  %s: %d locations" % (raw, took), file=sys.stderr)
    missing = sum(1 for i in range(len(page_of))
                  if not have[(page_of[i] * vpp + slot_of[i]) >> 6] >> ((page_of[i] * vpp + slot_of[i]) & 63) & 1)
    if missing:
        os.close(dst)
        os.unlink(tmp)
        sys.exit("join: %d of the %d vectors are in none of %s's raw files" % (missing, len(page_of), args.snapshot))
    os.fsync(dst)
    os.close(dst)
    os.rename(tmp, out)
    for f in ("meta.txt", "layout_%s.map" % args.layout):
        if os.path.abspath(os.path.join(args.out, f)) != os.path.abspath(os.path.join(args.index, f)):
            shutil.copyfile(os.path.join(args.index, f), os.path.join(args.out, f))
    print("%s: the page file again, every one of the %d vectors, from %d raw files; %s serves as --index"
          % (out, len(page_of), len(nodes), args.out))


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
    j = sub.add_parser("join", help="the page file again, from a warmup snapshot's raw files")
    j.add_argument("--index", required=True, help="the index directory, with its layout map (meta.txt, layout_<layout>.map)")
    j.add_argument("--snapshot", required=True, help="the warmup snapshot (n1/raw.pages, n1/raw.pages.present, ...)")
    j.add_argument("--layout", default="bucket")
    j.add_argument("--out", required=True, help="the directory to write vectors_<layout>.bin to, with meta.txt and the map")
    args = ap.parse_args()
    {"sample": sample, "check": check, "join": join}[args.cmd](args)


if __name__ == "__main__":
    main()
