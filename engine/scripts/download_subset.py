#!/usr/bin/env python3
"""Download a prefix of a big-ann-benchmarks dataset plus its public queries.

The billion-scale base files are 100-400 GB, so only the first N vectors are fetched with
HTTP range requests; the header is rewritten to N. For N = 10M, 100M or 1B the official
ground truth is downloaded too; for other sizes compute it with fusion_gt.

  python3 scripts/download_subset.py --dataset bigann --n 10000000 --out data/
  python3 scripts/download_subset.py --dataset deep --n 1000000 --out data/

Datasets (as in the paper, Table 1):
  bigann    SIFT1B / BigANN, 128-dim uint8  (paper: SIFT1B)
  msspacev  SPACEV1B, 100-dim int8          (paper: SPACEV1B)
  deep      DEEP1B, 96-dim float32          (paper: DEEP1B)
"""
import argparse
import os
import struct
import sys
import time
import urllib.request

SUBSET = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/"
BIGANN = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/"
DEEP = "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/"
SPACEV = "https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/"

DATASETS = {
    "bigann": dict(base=BIGANN + "base.1B.u8bin", query=BIGANN + "query.public.10K.u8bin",
                   ext="u8bin", dim=128, esize=1,
                   gt={10**7: SUBSET + "GT_10M/bigann-10M", 10**8: SUBSET + "GT_100M/bigann-100M",
                       10**9: BIGANN + "GT.public.1B.ibin"}),
    "deep": dict(base=DEEP + "base.1B.fbin", query=DEEP + "query.public.10K.fbin",
                 ext="fbin", dim=96, esize=4,
                 gt={10**7: SUBSET + "GT_10M/deep-10M", 10**8: SUBSET + "GT_100M/deep-100M",
                     10**9: "https://storage.yandexcloud.net/yandex-research/ann-datasets/"
                            "deep_new_groundtruth.public.10K.bin"}),
    "msspacev": dict(base=SPACEV + "spacev1b_base.i8bin", query=SPACEV + "query.i8bin",
                     ext="i8bin", dim=100, esize=1,
                     gt={10**6: SPACEV + "msspacev-gt-1M", 10**7: SPACEV + "msspacev-gt-10M",
                         10**8: SPACEV + "msspacev-gt-100M", 10**9: SPACEV + "public_query_gt100.bin"}),
}


def fetch(url, start=None, end=None, retries=6, timeout=300):
    """Returns bytes [start, end] (inclusive) of url, or the whole file."""
    headers = {} if start is None else {"Range": f"bytes={start}-{end}"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if start is not None and len(data) != end - start + 1:
                raise IOError(f"expected {end - start + 1} bytes, got {len(data)}")
            return data
        except Exception as e:  # noqa: BLE001 -- retry on any network error
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  retrying in {wait}s after error: {e}", file=sys.stderr)
            time.sleep(wait)


def download_prefix(url, n, dim, esize, out_path, chunk=256 << 20):
    n_total, d = struct.unpack("<II", fetch(url, 0, 7))
    if d != dim:
        raise SystemExit(f"{url}: unexpected dimension {d} (expected {dim})")
    n = min(n, n_total)
    total = n * dim * esize
    tmp = out_path + ".part"
    done = os.path.getsize(tmp) - 8 if os.path.exists(tmp) else 0  # resume support
    mode = "ab" if done > 0 else "wb"
    with open(tmp, mode) as f:
        if done <= 0:
            done = 0
            f.write(struct.pack("<II", n, dim))
        t0 = time.time()
        while done < total:
            stop = min(total, done + chunk)
            f.write(fetch(url, 8 + done, 8 + stop - 1))
            done = stop
            rate = done / max(time.time() - t0, 1e-6) / 1e6
            print(f"  {done / 1e9:.2f} / {total / 1e9:.2f} GB ({rate:.0f} MB/s)", file=sys.stderr)
    os.replace(tmp, out_path)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--n", type=int, required=True, help="number of base vectors (prefix)")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    ds = DATASETS[args.dataset]
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.dataset}-{args.n // 10**6}M" if args.n % 10**6 == 0 else f"{args.dataset}-{args.n}"
    base_path = os.path.join(args.out, f"{tag}.{ds['ext']}")
    query_path = os.path.join(args.out, f"{args.dataset}-query.{ds['ext']}")

    if not os.path.exists(base_path):
        print(f"downloading first {args.n} vectors of {ds['base']}", file=sys.stderr)
        download_prefix(ds["base"], args.n, ds["dim"], ds["esize"], base_path)
    if not os.path.exists(query_path):
        print(f"downloading queries {ds['query']}", file=sys.stderr)
        with open(query_path, "wb") as f:
            f.write(fetch(ds["query"]))

    gt_url = ds["gt"].get(args.n)
    gt_path = os.path.join(args.out, f"{tag}-gt.ibin")
    if gt_url and not os.path.exists(gt_path):
        print(f"downloading ground truth {gt_url}", file=sys.stderr)
        with open(gt_path, "wb") as f:
            f.write(fetch(gt_url))
    print(f"base:    {base_path}")
    print(f"queries: {query_path}")
    if gt_url:
        print(f"gt:      {gt_path}")
    else:
        print("no official ground truth for this size; compute it with:")
        print(f"  build/fusion_gt --base {base_path} --queries {query_path} --out {gt_path} --k 100")


if __name__ == "__main__":
    main()
