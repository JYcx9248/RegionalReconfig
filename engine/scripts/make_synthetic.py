#!/usr/bin/env python3
"""Synthetic clustered datasets for smoke tests when the real benchmarks are not at hand.

Vectors come from a Gaussian mixture in a low-dimensional latent space, linearly embedded in
`dim` dimensions plus a little isotropic noise -- a crude stand-in for the low intrinsic
dimensionality of real descriptors, so PQ and graph navigation behave roughly as on real data.
Output uses the big-ann-benchmarks formats (.u8bin / .i8bin / .fbin).

  python3 scripts/make_synthetic.py --n 1000000 --nq 5000 --dim 128 --dtype uint8 --out data/syn
    -> data/syn-base.u8bin, data/syn-query.u8bin

This is NOT a substitute for SIFT1B/DEEP1B/SPACEV1B when reporting results.
"""
import argparse
import struct

import numpy as np

EXT = {"uint8": "u8bin", "int8": "i8bin", "float": "fbin"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--nq", type=int, default=10000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--dtype", choices=sorted(EXT), default="uint8")
    ap.add_argument("--latent", type=int, default=24, help="intrinsic dimensionality")
    ap.add_argument("--clusters", type=int, default=2000)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, help="output prefix")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    centers = rng.standard_normal((args.clusters, args.latent)).astype(np.float32) * 2.0
    scales = rng.uniform(0.3, 1.2, size=args.clusters).astype(np.float32)
    weights = rng.pareto(1.5, size=args.clusters) + 1.0  # skewed cluster sizes, as in real data
    weights /= weights.sum()
    w = (rng.standard_normal((args.latent, args.dim)) / np.sqrt(args.latent)).astype(np.float32)

    def sample(n):
        comp = rng.choice(args.clusters, size=n, p=weights)
        z = centers[comp] + rng.standard_normal((n, args.latent)).astype(np.float32) * scales[comp, None]
        return z @ w + args.noise * rng.standard_normal((n, args.dim)).astype(np.float32)

    # Fix the value mapping from a sample so base and queries share it.
    probe = sample(20000)
    lo, hi = np.percentile(probe, 0.1), np.percentile(probe, 99.9)
    amax = np.percentile(np.abs(probe), 99.9)

    def convert(x):
        if args.dtype == "uint8":
            return np.clip(np.rint((x - lo) / (hi - lo) * 255.0), 0, 255).astype(np.uint8)
        if args.dtype == "int8":
            return np.clip(np.rint(x / amax * 127.0), -127, 127).astype(np.int8)
        return x.astype(np.float32)

    for name, n in (("base", args.n), ("query", args.nq)):
        path = f"{args.out}-{name}.{EXT[args.dtype]}"
        with open(path, "wb") as f:
            f.write(struct.pack("<II", n, args.dim))
            done = 0
            while done < n:
                m = min(200000, n - done)
                f.write(convert(sample(m)).tobytes())
                done += m
        print(path)


if __name__ == "__main__":
    main()
