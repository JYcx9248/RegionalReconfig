#!/usr/bin/env python3
"""Download the data of the Amazon experiments: Amazon Reviews 2023 (the corpus: reviews and item
metadata, per category) and AmazonQAC (the queries: Amazon's US search log of September 2023).

    python3 scripts/data/fetch_amazon.py OUT                                         # everything
    python3 scripts/data/fetch_amazon.py OUT --categories All_Beauty --qac-parts 1   # a pilot
    python3 scripts/data/fetch_amazon.py OUT --list                                  # the plan only

What it fetches is fixed: amazon_files.json, next to this script, lists the files of two pinned
Hugging Face commits with their sizes and SHA-256s. Each file is downloaded to NAME.part, resumed
where it stopped (HTTP Range), checked against that list, and only then renamed, so a rerun skips
the files that are in place and resumes the others.

  OUT/amazon-reviews-2023/raw/review_categories/<Category>.jsonl.gz      571.5M reviews
  OUT/amazon-reviews-2023/raw/meta_categories/meta_<Category>.jsonl.gz   48.2M items
  OUT/amazonqac/train/part-000NN-*.snappy.parquet                        395.6M searches, 20 shuffled parts
  OUT/amazonqac/test/part-00000-*.snappy.parquet                         20K, sampled from Oct 1-14

Amazon Reviews 2023 (33 categories and Unknown) comes gzipped from the McAuley Lab (--source ucsd,
the default: 102.7 GB) or uncompressed from Hugging Face (--source hf: 376.8 GB, as .jsonl). The
lab's files decompress to Hugging Face's byte for byte, so both are checked by the same SHA-256,
and a category already in place in either form is skipped. AmazonQAC (59.3 GB) is on Hugging Face
only. Where huggingface.co cannot be reached, a mirror serves the same commits: --hf-endpoint
https://hf-mirror.com. Only data files are fetched, never the dataset's loading script
(Amazon-Reviews-2023.py), and nothing downloaded is run.

Standard library only; a proxy is taken from https_proxy, as usual.
"""

import argparse
import gzip
import hashlib
import http.client
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "amazon_files.json")
HEADERS = {"User-Agent": "rtier-fetch-amazon/1"}
CHUNK = 1 << 22
ATTEMPTS = 30  # attempts in a row that bring no byte before a download gives up
TIMEOUT = 120  # seconds without a byte before a connection counts as dead
STOP = threading.Event()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def size(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


class File:
    def __init__(self, url: str, dst: str, wire: int, sha256: str, gz: bool):
        self.url, self.dst, self.wire, self.sha256, self.gz = url, dst, wire, sha256, gz
        self.name = os.path.basename(dst)

    def in_place(self) -> bool:
        # Only a checked file is ever renamed to dst.
        return os.path.exists(self.dst) and os.path.getsize(self.dst) == self.wire

    def on_disk(self) -> int:
        if self.in_place():
            return self.wire
        try:
            return os.path.getsize(self.dst + ".part")
        except OSError:
            return 0


def category(path: str) -> str:
    name = os.path.basename(path)[: -len(".jsonl")]
    return name[len("meta_"):] if name.startswith("meta_") else name


def plan(args, files: dict) -> list:
    """(file, the same content in the other form or None) for everything selected."""
    hf = args.hf_endpoint.rstrip("/")

    def hf_url(d, path):
        return f"{hf}/datasets/{d['repo']}/resolve/{d['revision']}/{path}"

    ar, qac = files["amazon-reviews-2023"], files["amazonqac"]
    cats = sorted({category(f["path"]) for f in ar["files"]})
    want = {"all": set(cats), "none": set()}.get(args.categories, {c.strip() for c in args.categories.split(",")})
    if want - set(cats):
        sys.exit(f"unknown categories {sorted(want - set(cats))}; they are: {' '.join(cats)}")
    out = []
    for f in ar["files"]:
        if category(f["path"]) in want:
            plain = os.path.join(args.out, "amazon-reviews-2023", f["path"])
            gz = File(f"{ar['gzip_url']}/{f['path']}.gz", plain + ".gz", f["gzip_size"], f["sha256"], True)
            raw = File(hf_url(ar, f["path"]), plain, f["size"], f["sha256"], False)
            out.append((gz, raw) if args.source == "ucsd" else (raw, gz))
    tests = [f for f in qac["files"] if f["path"].startswith("test/")]
    trains = sorted((f for f in qac["files"] if f["path"].startswith("train/")), key=lambda f: f["path"])
    for f in tests + trains[: args.qac_parts]:
        dst = os.path.join(args.out, "amazonqac", f["path"])
        out.append((File(hf_url(qac, f["path"]), dst, f["size"], f["sha256"], False), None))
    return out


def part_size(part: str) -> int:
    return os.path.getsize(part) if os.path.exists(part) else 0


def download(f: File, part: str) -> None:
    if part_size(part) > f.wire:  # not this file: start over
        os.remove(part)
    # Only attempts that bring nothing count, so a link that keeps dropping still gets there.
    failures = 0
    while (have := part_size(part)) < f.wire:
        headers = dict(HEADERS, **({"Range": f"bytes={have}-"} if have else {}))
        err = "the connection closed early"
        try:
            with urllib.request.urlopen(urllib.request.Request(f.url, headers=headers), timeout=TIMEOUT) as r:
                # A server that ignores the range sends the whole file: start over.
                with open(part, "ab" if have and r.status == 206 else "wb") as out:
                    while not STOP.is_set() and (b := r.read(CHUNK)):
                        out.write(b)
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code not in (408, 429):
                raise RuntimeError(f"{f.url}: HTTP {e.code} {e.reason}")
            err = f"HTTP {e.code}"
        except (OSError, http.client.HTTPException) as e:
            err = repr(e)
        if STOP.is_set():
            raise RuntimeError("stopped")
        now = part_size(part)
        if now > f.wire:
            os.remove(part)
            raise RuntimeError(f"{f.url} sent more than the {f.wire:,} bytes listed: not the same file")
        if now == f.wire:
            return
        failures = 0 if now > have else failures + 1
        if failures == ATTEMPTS:
            raise RuntimeError(f"{err}, {ATTEMPTS} times in a row without a byte")
        log(f"{f.name}: {err} at {size(now)} of {size(f.wire)}; resuming")
        STOP.wait(min(60, 5 * failures))


def sha256(path: str, gz: bool):
    """SHA-256 of the content (decompressed, for a .gz); None if it does not decompress."""
    h = hashlib.sha256()
    try:
        with (gzip.open if gz else open)(path, "rb") as f:
            while (b := f.read(CHUNK * 4)) and not STOP.is_set():
                h.update(b)
    except (OSError, EOFError, zlib.error):
        return None
    return h.hexdigest()


def fetch(f: File) -> None:
    os.makedirs(os.path.dirname(f.dst), exist_ok=True)
    part = f.dst + ".part"
    if 0 < part_size(part) <= f.wire:
        log(f"{f.name}: resuming at {size(part_size(part))} of {size(f.wire)}")
    for _ in range(2):
        download(f, part)
        digest = sha256(part, f.gz)
        if STOP.is_set():
            raise RuntimeError("stopped")
        if digest == f.sha256:
            os.replace(part, f.dst)
            return
        os.remove(part)
        log(f"{f.name}: content does not match its SHA-256; downloading it again")
    raise RuntimeError("content does not match its SHA-256, twice")


def free_bytes(path: str) -> int:
    while not os.path.exists(path):
        path = os.path.dirname(path) or "."
    return shutil.disk_usage(path).free


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help="where the datasets go (amazon-reviews-2023/, amazonqac/)")
    ap.add_argument("--categories", default="all",
                    help="comma-separated Amazon Reviews 2023 categories, all (default) or none")
    ap.add_argument("--qac-parts", type=int, default=20, choices=range(21), metavar="0..20",
                    help="AmazonQAC train parts (default: all 20). The log is shuffled across them: "
                         "each is a sample of the whole month, ~19.8M searches. The test part always "
                         "comes along")
    ap.add_argument("--source", choices=("ucsd", "hf"), default="ucsd",
                    help="Amazon Reviews 2023 gzipped from the McAuley Lab or uncompressed from Hugging Face")
    ap.add_argument("--hf-endpoint", default="https://huggingface.co", help="e.g. https://hf-mirror.com")
    ap.add_argument("--workers", type=int, default=4, help="files downloaded at a time")
    ap.add_argument("--list", action="store_true", help="print what would be fetched, fetch nothing")
    args = ap.parse_args()

    with open(FILES) as fh:
        selected = plan(args, json.load(fh))
    pending = [f for f, alt in selected if not (f.in_place() or (alt is not None and alt.in_place()))]
    total = sum(f.wire for f in pending)
    have = sum(f.on_disk() for f in pending)
    free = free_bytes(args.out)
    log(f"{len(selected)} files selected, {len(selected) - len(pending)} in place; {len(pending)} to fetch: "
        f"{size(total)}, of which {size(have)} already in .part files; {size(free)} free under {args.out}")
    if args.list:
        for f in pending:
            print(f"{f.wire:>14,}  {f.url}")
        return
    if total - have > free:
        sys.exit(f"not enough space under {args.out}: {size(total - have)} to fetch, {size(free)} free")

    failed = []
    # Largest first, so that what finishes last is small.
    pending.sort(key=lambda f: -f.wire)
    ex = ThreadPoolExecutor(args.workers)
    futs = {ex.submit(fetch, f): f for f in pending}
    last_t, last_b = time.monotonic(), have
    try:
        while futs:
            done, _ = wait(futs, timeout=60, return_when=FIRST_COMPLETED)
            for fu in done:
                f = futs.pop(fu)
                try:
                    fu.result()
                    log(f"{f.name}: in place ({size(f.wire)}, SHA-256 checked)")
                except Exception as e:
                    failed.append(f.name)
                    log(f"{f.name}: FAILED: {e}")
            now = time.monotonic()
            if now - last_t >= 60 or not futs:
                b = sum(f.on_disk() for f in pending)
                log(f"{size(b)} of {size(total)} ({100 * b / max(total, 1):.0f}%), "
                    f"{(b - last_b) / (now - last_t) / 1e6:.1f} MB/s; {len(pending) - len(futs)} of {len(pending)} files done")
                last_t, last_b = now, b
    except KeyboardInterrupt:
        STOP.set()
        log("stopping; a rerun resumes from the .part files")
        ex.shutdown(cancel_futures=True)
        sys.exit(130)
    ex.shutdown()
    if failed:
        sys.exit(f"{len(failed)} files failed: {' '.join(failed)}; a rerun retries them")
    log("all selected files in place")


if __name__ == "__main__":
    main()
