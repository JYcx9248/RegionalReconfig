#!/usr/bin/env python3
"""Download SPACEV-1B: big-ann-benchmarks' 1B base of Microsoft's SPACEV1B (100-d int8), its
queries and its ground truth.

    python3 scripts/data/fetch_spacev.py OUT                                      # everything
    python3 scripts/data/fetch_spacev.py OUT --hf-endpoint https://hf-mirror.com  # from China
    python3 scripts/data/fetch_spacev.py OUT --list                               # the plan only

What it fetches is fixed: spacev_files.json, next to this script, lists the files of a pinned
Hugging Face commit of SPTAG's release (jkhe/spacev1b: 1,402,020,720 vectors in 33 parts of
4,294,967,295 bytes, and its queries) and big-ann's ground truth and private queries, each with
its size and SHA-256.

big-ann's base (spacev1b_base.i8bin on its Azure storage) is the release's first 1B vectors byte
for byte, under a header that says 1B (rows 0, 123,456,789, 500,000,000 and 999,999,990
compared, 2026-11-23), and from China that storage gives about 23 KB/s a connection. So the base
is made from the release instead: its parts 1-24, joined, with the header rewritten. In OUT:

  spacev1b_base.i8bin          1,000,000,000 x 100 int8: big-ann's base (100 GB)
  spacev1b_extra.i8bin         the release's next 30,792,150 vectors (the rest of part 24): not in
                               the base, so queries that are not base vectors
  query.i8bin                  29,316 public queries (big-ann's query.i8bin = the release's query.bin)
  query_log.i8bin              94,162 queries of the release's query log
  public_query_gt100.bin       big-ann's top 100 of the public queries in the base
  private_query_30k.bin        big-ann's 30,000 private queries
  gt100_private_query_30k.bin  their top 100 in the base

The release's own ground truth (truth.bin) is in all 1.4B vectors, so it is not fetched.

Each file comes in ranges of --chunk-mb over --workers connections into NAME.part, the finished
ranges recorded in NAME.part.done once synced, so a rerun resumes where it stopped. A Hugging Face
connection that brings less than --min-rate KB/s over a minute is opened again: from China, one
connection to the same endpoint got 30 KB/s while another got 900. A file is checked against its
SHA-256 and only then renamed. Then the parts go into the base one after another, each removed
once it is in and synced (unless --keep-parts): the disk holds at most the parts and one more,
and the join resumes from the first part still there.

Standard library only; a proxy is taken from https_proxy, as usual.
"""

import argparse
import hashlib
import http.client
import json
import os
import queue
import shutil
import struct
import sys
import threading
import time
import urllib.error
import urllib.request

FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spacev_files.json")
HEADERS = {"User-Agent": "rtier-fetch-spacev/1"}
BLOCK = 1 << 16  # bytes a read asks for: small, so that a slow connection shows within its window
TIMEOUT = 120  # seconds without a byte before a connection counts as dead
ATTEMPTS = 30  # attempts in a row that bring no byte before a range gives up
WINDOW = 60  # seconds over which a connection's rate is judged
EXTRA = "spacev1b_extra.i8bin"
STOP = threading.Event()
_log_lock = threading.Lock()


def log(msg: str) -> None:
    with _log_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def size(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


class Slow(Exception):
    pass


class File:
    """A file fetched in ranges into dst.part; the finished ranges are listed in dst.part.done."""

    def __init__(self, url: str, dst: str, nbytes: int, sha256: str, min_rate: float):
        self.url, self.dst, self.nbytes, self.sha256, self.min_rate = url, dst, nbytes, sha256, min_rate
        self.name = os.path.basename(dst)
        self.part, self.record = dst + ".part", dst + ".part.done"
        self.lock = threading.Lock()
        self.chunk, self.done, self.fd = 0, set(), None
        self.tries, self.failed, self.finished = 0, None, False

    def in_place(self) -> bool:
        # Only a checked file is ever renamed to dst.
        return os.path.exists(self.dst) and os.path.getsize(self.dst) == self.nbytes

    def chunks(self) -> int:
        return (self.nbytes + self.chunk - 1) // self.chunk

    def span(self, i: int):
        return i * self.chunk, min((i + 1) * self.chunk, self.nbytes)

    def have(self) -> int:
        return self.nbytes if self.in_place() else sum(self.span(i)[1] - self.span(i)[0] for i in self.done)

    def load(self, chunk: int) -> None:
        """The record of an earlier run, if it is of this file (its chunk size wins), else none."""
        self.chunk, self.done = chunk, set()
        try:
            with open(self.record) as fh:
                r = json.load(fh)
            if r["size"] == self.nbytes and r["sha256"] == self.sha256 and os.path.exists(self.part):
                self.chunk, self.done = r["chunk"], set(r["done"])
        except (OSError, ValueError, KeyError):
            pass

    def open(self) -> None:
        if not self.done:
            for p in (self.part, self.record):
                if os.path.exists(p):
                    os.remove(p)
        os.makedirs(os.path.dirname(self.dst) or ".", exist_ok=True)
        self.fd = os.open(self.part, os.O_RDWR | os.O_CREAT, 0o644)
        if os.fstat(self.fd).st_size != self.nbytes:
            os.ftruncate(self.fd, self.nbytes)  # sparse: it takes room only as ranges arrive

    def finish(self, i: int) -> bool:
        """Records range i, once its bytes are on the device; True when it was the last one."""
        os.fdatasync(self.fd)
        with self.lock:
            self.done.add(i)
            tmp = self.record + ".tmp"
            with open(tmp, "w") as fh:
                json.dump({"size": self.nbytes, "sha256": self.sha256, "chunk": self.chunk,
                           "done": sorted(self.done)}, fh)
            os.replace(tmp, self.record)
            return len(self.done) == self.chunks()


class Progress:
    def __init__(self):
        self.lock = threading.Lock()
        self.arrived = 0
        self.slow = 0

    def add(self, n: int = 0, slow: int = 0) -> None:
        with self.lock:
            self.arrived += n
            self.slow += slow


def fetch_range(f: File, start: int, end: int, prog: Progress) -> None:
    """Writes bytes [start, end) of f at their offsets in f.part."""
    pos, failures = start, 0
    while pos < end:
        if STOP.is_set():
            raise RuntimeError("stopped")
        before, err = pos, "the connection closed early"
        req = urllib.request.Request(f.url, headers=dict(HEADERS, Range=f"bytes={pos}-{end - 1}"))
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                if r.status != 206 and not (r.status == 200 and pos == 0 and end == f.nbytes):
                    raise RuntimeError(f"{f.url}: HTTP {r.status} to a range request: the server ignores ranges")
                t_ck, pos_ck = time.monotonic(), pos
                while pos < end and not STOP.is_set():
                    b = r.read1(min(BLOCK, end - pos))
                    if not b:
                        break
                    os.pwrite(f.fd, b, pos)
                    pos += len(b)
                    prog.add(len(b))
                    now = time.monotonic()
                    if now - t_ck >= WINDOW:
                        rate = (pos - pos_ck) / (now - t_ck)
                        if f.min_rate and rate < f.min_rate and pos < end:
                            raise Slow(f"{rate / 1e3:.0f} KB/s")
                        t_ck, pos_ck = now, pos
        except Slow:
            prog.add(slow=1)
            continue  # another connection at once: that is the point
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code not in (408, 429):
                raise RuntimeError(f"{f.url}: HTTP {e.code} {e.reason}")
            err = f"HTTP {e.code}"
        except (OSError, http.client.HTTPException) as e:
            err = repr(e)
        if pos >= end:
            return
        if STOP.is_set():
            raise RuntimeError("stopped")
        # Only attempts that bring nothing count, so a link that keeps dropping still gets there.
        failures = 0 if pos > before else failures + 1
        if failures == ATTEMPTS:
            raise RuntimeError(f"{err}, {ATTEMPTS} times in a row without a byte")
        STOP.wait(min(60, 5 * failures))


def sha256_of(path: str):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while (b := fh.read(1 << 24)) and not STOP.is_set():
            h.update(b)
    return h.hexdigest()


def complete(f: File, tasks: queue.Queue) -> None:
    """f's last range is in: check it, then rename it, or start it over once."""
    os.close(f.fd)
    f.fd = None
    digest = sha256_of(f.part)
    if STOP.is_set():
        return
    if digest == f.sha256:
        os.replace(f.part, f.dst)
        os.remove(f.record)
        f.finished = True
        log(f"{f.name}: in place ({size(f.nbytes)}, SHA-256 checked)")
        return
    f.tries += 1
    f.done = set()
    if f.tries == 2:
        f.failed = "content does not match its SHA-256, twice"
        log(f"{f.name}: FAILED: {f.failed}")
        return
    log(f"{f.name}: content does not match its SHA-256; downloading it again")
    f.open()
    for i in range(f.chunks()):
        tasks.put((f, i))


def download(files: list, workers: int) -> list:
    """Fetches the ranges files still lack; returns the files that failed."""
    tasks = queue.Queue()  # (file, range), or (file, None): every range is in, check the file
    for f in files:
        f.open()
        missing = [i for i in range(f.chunks()) if i not in f.done]
        for i in missing or [None]:
            tasks.put((f, i))
    prog = Progress()
    total = sum(f.nbytes for f in files)
    have0 = sum(f.have() for f in files)

    def worker():
        while not STOP.is_set():
            try:
                f, i = tasks.get(timeout=1)
            except queue.Empty:
                if all(f.finished or f.failed for f in files):
                    return
                continue
            if f.failed:
                continue
            try:
                if i is None:
                    complete(f, tasks)
                    continue
                fetch_range(f, *f.span(i), prog)
                if f.finish(i):
                    complete(f, tasks)
            except Exception as e:  # noqa: BLE001 -- the file fails, the others go on
                if not STOP.is_set():
                    f.failed = str(e)
                    log(f"{f.name}: FAILED: {e}")

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()
    t0 = last_t = time.monotonic()
    last_b = 0
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=60 / len(threads))
            now = time.monotonic()
            if now - last_t >= 60:
                with prog.lock:
                    arrived, slow = prog.arrived, prog.slow
                rate = (arrived - last_b) / (now - last_t)
                left = max(0, total - have0 - arrived)
                eta = f", {left / rate / 3600:.1f} h to go at that rate" if rate > 0 else ""
                log(f"{size(have0 + arrived)} of {size(total)} ({100 * (have0 + arrived) / total:.0f}%), "
                    f"{rate / 1e6:.2f} MB/s over the last minute, {arrived / (now - t0) / 1e6:.2f} since the "
                    f"start{eta}; {sum(f.finished for f in files)} of {len(files)} files in place; "
                    f"{slow} slow connections replaced")
                last_t, last_b = now, arrived
    except KeyboardInterrupt:
        STOP.set()
        log("stopping; a rerun resumes from the finished ranges")
        sys.exit(130)
    return [f for f in files if not f.finished]


def copy(src: str, src_off: int, dst_fd: int, dst_off: int, n: int) -> None:
    with open(src, "rb") as fh:
        src_fd = fh.fileno()
        while n > 0:
            k = os.copy_file_range(src_fd, dst_fd, min(n, 1 << 30), src_off, dst_off)
            if k <= 0:
                raise RuntimeError(f"{src}: copy stopped with {n:,} bytes to go")
            src_off, dst_off, n = src_off + k, dst_off + k, n - k


def join_start(parts: list, base: str, part_size: int, end: int) -> int:
    """How many leading parts are already in base.part (each is removed only once it is in and
    synced, so they are the parts before the first one still in place)."""
    tmp = base + ".part"
    if not os.path.exists(tmp):
        return 0
    start = next((i for i, p in enumerate(parts) if p.in_place()), len(parts))
    if start and os.path.getsize(tmp) < min(start * part_size, end):
        sys.exit(f"{tmp} is shorter than the {start} parts already joined into it and removed: "
                 f"remove it and run again, which fetches those parts again")
    return start


def join(parts: list, start: int, part_size: int, rows: int, dim: int, base: str, keep: bool) -> None:
    """base = the release's bytes up to its row `rows`, under a header of `rows` rows; the rest of
    the part it ends in, whole rows, goes to EXTRA. File offsets in base are the release's."""
    tmp, end = base + ".part", 8 + rows * dim
    fd = os.open(tmp, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        os.ftruncate(fd, max(8, min(start * part_size, end)))  # drop a part that was only partly in
        os.pwrite(fd, struct.pack("<II", rows, dim), 0)
        for i in range(start, len(parts)):
            lo = i * part_size
            hi = min(lo + part_size, end)
            first = max(lo, 8)  # the release's own header is replaced
            copy(parts[i].dst, first - lo, fd, first, hi - first)
            os.fsync(fd)
            if hi < lo + part_size:
                n = (lo + part_size - hi) // dim
                ex = os.path.join(os.path.dirname(base), EXTRA)
                efd = os.open(ex + ".part", os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
                try:
                    os.pwrite(efd, struct.pack("<II", n, dim), 0)
                    copy(parts[i].dst, hi - lo, efd, 8, n * dim)
                    os.fsync(efd)
                finally:
                    os.close(efd)
                os.replace(ex + ".part", ex)
                log(f"{EXTRA}: {n:,} vectors after the base's, {size(8 + n * dim)}")
            if not keep:
                os.remove(parts[i].dst)
            log(f"{parts[i].name} joined: {size(hi)} of {size(end)}")
    finally:
        os.close(fd)
    os.replace(tmp, base)


def free_bytes(path: str) -> int:
    while not os.path.exists(path):
        path = os.path.dirname(path) or "."
    return shutil.disk_usage(path).free


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help="where the files go")
    ap.add_argument("--hf-endpoint", default="https://huggingface.co", help="e.g. https://hf-mirror.com")
    ap.add_argument("--workers", type=int, default=8, help="connections at a time (default 8)")
    ap.add_argument("--chunk-mb", type=int, default=64, help="MiB a range (default 64)")
    ap.add_argument("--min-rate", type=float, default=64,
                    help="KB/s over a minute below which a Hugging Face connection is opened again "
                         "(default 64; 0: never)")
    ap.add_argument("--keep-parts", action="store_true", help="keep the release's parts after the join")
    ap.add_argument("--skip-base", action="store_true", help="only the queries and the ground truth")
    ap.add_argument("--list", action="store_true", help="print what would be fetched, fetch nothing")
    args = ap.parse_args()

    with open(FILES) as fh:
        m = json.load(fh)
    hf, bg = m["hf"], m["bigann"]
    rows, dim = bg["base"]["rows"], bg["base"]["dim"]
    endpoint = args.hf_endpoint.rstrip("/")

    def hf_file(path: str, dst: str) -> File:
        e = next(x for x in hf["files"] if x["path"] == path)
        url = f"{endpoint}/datasets/{hf['repo']}/resolve/{hf['revision']}/{path}"
        return File(url, dst, e["size"], e["sha256"], args.min_rate * 1e3)

    release = [x for x in hf["files"] if x["path"].startswith("vectors.bin/")]
    part_size = release[0]["size"]
    nparts = (8 + rows * dim + part_size - 1) // part_size
    parts = [hf_file(f"vectors.bin/vectors_{i}.bin", os.path.join(args.out, "parts", f"vectors_{i}.bin"))
             for i in range(1, nparts + 1)]
    if any(p.nbytes != part_size for p in parts):
        sys.exit(f"{FILES}: the parts that hold the base are not all {part_size:,} bytes")
    small = [hf_file("query.bin", os.path.join(args.out, "query.i8bin")),
             hf_file("query_log.bin", os.path.join(args.out, "query_log.i8bin"))]
    small += [File(f"{bg['url']}/{x['path']}", os.path.join(args.out, x["path"]), x["size"], x["sha256"], 0)
              for x in bg["files"]]
    if args.workers < 1 or args.chunk_mb < 1:
        sys.exit("--workers and --chunk-mb must be at least 1")
    base = os.path.join(args.out, bg["base"]["path"])
    end = 8 + rows * dim
    base_done = os.path.exists(base) and os.path.getsize(base) == end
    start = len(parts) if base_done or args.skip_base else join_start(parts, base, part_size, end)

    chunk = args.chunk_mb << 20
    for f in small + parts:
        f.load(chunk)
        if f.in_place() and os.path.exists(f.record):  # a run stopped between the rename and this
            os.remove(f.record)
    pending = [f for f in small + parts[start:] if not f.in_place()]
    left = sum(f.nbytes - f.have() for f in pending)
    room = left + (0 if base_done or args.skip_base else part_size)  # the join holds one part twice
    free = free_bytes(args.out)
    log(f"{len(pending)} files to fetch: {size(left)} to go ({size(sum(f.have() for f in pending))} already "
        f"in .part files); the base {'is in place' if base_done else 'is not wanted' if args.skip_base else f'needs parts {start + 1}-{nparts}, {start} already joined'}; "
        f"{size(free)} free under {args.out}")
    if args.list:
        for f in pending:
            print(f"{f.nbytes:>14,}  {f.url}")
        return
    if room > free:
        sys.exit(f"not enough space under {args.out}: {size(room)} needed, {size(free)} free")

    failed = download(pending, args.workers)
    if failed:
        sys.exit(f"{len(failed)} files failed: {' '.join(f.name for f in failed)}; a rerun retries them")
    if not (base_done or args.skip_base):
        join(parts, start, part_size, rows, dim, base, args.keep_parts)
        if not args.keep_parts:
            try:
                os.rmdir(os.path.join(args.out, "parts"))
            except OSError:
                pass
    log(f"done: {base if not args.skip_base else 'the queries and the ground truth'} in place")


if __name__ == "__main__":
    main()
