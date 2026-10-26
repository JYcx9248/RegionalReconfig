#!/usr/bin/env python3
"""Embed a text store (TEXTS.bin and TEXTS.off, made by amazon_texts.py) into int8 vectors:
OUT.i8bin -- u32 count, u32 dim, then count x dim int8 --, the base format fusion_build, fusion_gt
and the load generator read.

    python3 scripts/data/embed_texts.py $V/corpus-text  $V/corpus.i8bin  --gpu 4
    python3 scripts/data/embed_texts.py $V/queries-text $V/queries.i8bin --gpu 4 \\
        --task "..." --scale-from $V/corpus.i8bin.json

TEXTS is the store's path without its extension. The model is the Wikipedia corpus's,
Qwen3-Embedding-0.6B at the same revision, run as embed_queries.py runs it: documents as plain
text, queries as "Instruct: TASK\\nQuery:TEXT" (--task), last-token pooling, bfloat16. Its 1024-d
output is cut to --dim dimensions (768: the model is trained for it, Matryoshka) and normalized,
then quantized with one scale s for every vector, round(x * s) clipped to +-127: a single scale
keeps the ranking of the float vectors, where a scale per dimension would weigh the dimensions
apart. s is 127 over the 99.999th percentile of |x| in a random sample of the rows (--sample),
embedded first and kept in float16 next to the output (OUT.sample.npy; its rows in
OUT.sample-rows.npy) to measure what int8 costs. Queries take the corpus's s (--scale-from), so
that both are in one space; their sample only measures.

--gpu is the GPU's index as nvidia-smi shows it: CUDA_DEVICE_ORDER=PCI_BUS_ID is set here, since
on a machine with mixed GPUs (the lab server) CUDA's own order differs. --cpu runs in float32 on
the CPU, for small checks.

--check N, first on a new machine or model: N rows spread over the store, embedded here and as
embed_queries.py embeds them (the pipeline that gave the Wikipedia corpus's own vectors back); a
cosine above 0.99 for all of them says the two are the same embedding. It writes nothing.

The rows are embedded a chunk at a time (--chunk rows), each written at its place in OUT.i8bin,
which is made at full size first; a chunk done is listed in OUT.i8bin.done, so a rerun skips it
and redoes the one cut short. --shard i/n takes every n-th chunk: to share a run between GPUs,
start one process per GPU, all with the same n (shard 0 first: it sets the scale).
OUT.i8bin.json records how the vectors were made, and "complete" once every chunk is in: until
then OUT.i8bin holds rows of zeros, so nothing is to be built from it.

Needs numpy, torch and transformers (the lab server's venv).
"""

import argparse
import json
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

MODEL = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"  # the Wikipedia corpus's (its manifest.json)
PERCENTILE = 99.999
WINDOW = 32768  # rows tokenized at a time: the next window is tokenized while the GPU runs one


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Texts:
    """A text store: TEXTS.bin, UTF-8 back to back, and TEXTS.off, u64 offsets (N + 1)."""

    def __init__(self, path: str):
        self.off = np.memmap(path + ".off", dtype=np.uint64, mode="r")
        self.data = np.memmap(path + ".bin", dtype=np.uint8, mode="r")
        if len(self.off) < 1 or int(self.off[-1]) != len(self.data):
            sys.exit(f"{path}.off does not end at the size of {path}.bin: not a text store, or cut short")

    def __len__(self) -> int:
        return len(self.off) - 1

    def range(self, lo: int, hi: int) -> list:
        o = self.off[lo:hi + 1].astype(np.int64)
        blob = bytes(self.data[o[0]:o[-1]])
        return [blob[a - o[0]:b - o[0]].decode("utf-8") for a, b in zip(o[:-1], o[1:])]

    def rows(self, rows) -> list:
        return [bytes(self.data[int(self.off[r]):int(self.off[r + 1])]).decode("utf-8") for r in rows]


class Model:
    def __init__(self, args):
        # Before torch starts CUDA: the index the caller gives is nvidia-smi's.
        if not args.cpu:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        if args.cpu:
            self.dev, dtype, self.where = "cpu", torch.float32, "cpu (float32)"
        else:
            if not torch.cuda.is_available():
                sys.exit(f"no CUDA device as GPU {args.gpu}: check nvidia-smi's numbering, or use --cpu")
            self.dev = "cuda:0"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            free, total = torch.cuda.mem_get_info(0)
            self.where = (f"GPU {args.gpu} (nvidia-smi's numbering): {torch.cuda.get_device_name(0)}, "
                          f"{free / 2**30:.0f} of {total / 2**30:.0f} GiB free, {dtype}")
        # transformers 4.56 renamed torch_dtype to dtype (5.x still takes the old name, with a warning).
        major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
        kw = {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}
        self.tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        self.model = AutoModel.from_pretrained(args.model, revision=args.revision, **kw).to(self.dev).eval()
        if self.model.config.hidden_size < args.dim:
            sys.exit(f"--dim {args.dim}: the model's vectors have {self.model.config.hidden_size} dimensions")
        self.max_length = args.max_length
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.versions = {"torch": torch.__version__, "transformers": transformers.__version__}

    def tokenize(self, texts: list) -> list:
        # As embed_queries.py calls it, without the padding: what gets padded is a batch.
        return self.tok(texts, truncation=True, max_length=self.max_length)["input_ids"]

    def encode(self, ids: list) -> np.ndarray:
        """The last token's hidden state of each sequence, float32. Left padding puts every last
        token in the last column; RoPE sees relative positions only, so the shift changes nothing."""
        width = max(map(len, ids))
        tokens = np.full((len(ids), width), self.pad, dtype=np.int64)
        mask = np.zeros((len(ids), width), dtype=np.int64)
        for r, x in enumerate(ids):
            tokens[r, width - len(x):] = x
            mask[r, width - len(x):] = 1
        t = self.torch
        with t.inference_mode():
            h = self.model(input_ids=t.from_numpy(tokens).to(self.dev),
                           attention_mask=t.from_numpy(mask).to(self.dev)).last_hidden_state[:, -1]
        return h.float().cpu().numpy()


def prepare(model, texts: list, task):
    """Tokens of a window of texts, and their order by length, so that a batch pads little."""
    if task is not None:
        texts = [f"Instruct: {task}\nQuery:{t}" for t in texts]
    ids = model.tokenize(texts)
    lens = np.fromiter(map(len, ids), dtype=np.int64, count=len(ids))
    return ids, lens, np.argsort(lens, kind="stable")


def run(model, prepared, dim: int, batch_tokens: int) -> np.ndarray:
    """Unit vectors (float32, dim) of a prepared window, in the window's order."""
    ids, lens, order = prepared
    out = np.empty((len(ids), dim), dtype=np.float32)
    i = 0
    while i < len(order):
        j = i + 1  # sorted by length: the last one taken sets the batch's width
        while j < len(order) and (j - i + 1) * lens[order[j]] <= batch_tokens:
            j += 1
        b = order[i:j]
        x = model.encode([ids[k] for k in b])[:, :dim]
        out[b] = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        i = j
    return out


def quantize(x: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(np.rint(x * scale), -127, 127).astype(np.int8)


def sample_stats(x: np.ndarray, scale: float) -> dict:
    """What int8 costs on the sample: values clipped, and the cosine to each float vector."""
    q = quantize(x, scale).astype(np.float32)
    cos = (x * q).sum(axis=1) / np.maximum(np.linalg.norm(q, axis=1), 1e-12)
    return {"max_abs": float(np.abs(x).max()), "clipped": float((np.abs(x) * scale > 127.5).mean()),
            "cosine_min": float(cos.min()), "cosine_median": float(np.median(cos))}


def check(model, texts, args) -> None:
    """Rows spread over the store, embedded here and by embed_queries.py, whose pipeline gave the
    Wikipedia corpus's own vectors back (cosine 0.9994 at the least, on the lab server): the same
    model and settings put the two within a cosine of 1, batching and padding apart."""
    import types

    from transformers import AutoTokenizer

    import embed_queries

    rows = np.unique(np.linspace(0, len(texts) - 1, min(args.check, len(texts))).astype(np.int64))
    sample = texts.rows(rows)
    ours = run(model, prepare(model, sample, args.task), args.dim, args.batch_tokens)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, padding_side="left")
    theirs = embed_queries.embed(sample, tok, model.model, model.dev,
                                 types.SimpleNamespace(batch=16, max_length=args.max_length), args.task)[:, :args.dim]
    theirs /= np.linalg.norm(theirs, axis=1, keepdims=True)
    cos = (ours * theirs).sum(axis=1)
    log(f"{len(rows)} rows, cosine to embed_queries.py's vectors: min {cos.min():.4f}, median {np.median(cos):.5f}")
    if cos.min() <= 0.99:
        sys.exit(f"NOT embed_queries.py's embedding (lowest at rows {list(rows[np.argsort(cos)[:3]])}): "
                 "check the model, the revision and --max-length")
    log("same embedding as embed_queries.py")


def write_all(fd: int, data: bytes, offset: int) -> None:
    view = memoryview(data)
    while view:
        n = os.pwrite(fd, view, offset)
        view, offset = view[n:], offset + n


def done_chunks(path: str) -> set:
    try:
        with open(path) as f:
            return {int(line) for line in f if line.strip()}
    except FileNotFoundError:
        return set()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("texts", help="the text store, without .bin/.off")
    ap.add_argument("out", nargs="?", help="the vectors (.i8bin); none with --check")
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--gpu", type=int, help="the GPU, by nvidia-smi's index")
    where.add_argument("--cpu", action="store_true", help="on the CPU, float32: for small checks")
    ap.add_argument("--task", help="queries: the instruction (documents get none)")
    ap.add_argument("--scale-from", help="the int8 scale of another output's .json (the queries: the corpus's)")
    ap.add_argument("--dim", type=int, default=768, help="dimensions kept (Matryoshka), at most 1024")
    ap.add_argument("--max-length", type=int, default=512, help="tokens kept of a text")
    ap.add_argument("--batch-tokens", type=int, default=65536, help="padded tokens per batch")
    ap.add_argument("--chunk", type=int, default=1_000_000, help="rows written and listed done at a time")
    ap.add_argument("--sample", type=int, default=200_000, help="rows embedded first, kept in float16")
    ap.add_argument("--seed", type=int, default=0, help="of the sample")
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th chunk, from the i-th")
    ap.add_argument("--check", type=int, metavar="N",
                    help="embed N rows here and as embed_queries.py does, compare, write nothing")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    args = ap.parse_args()
    try:
        si, sn = (int(x) for x in args.shard.split("/"))
        assert 0 <= si < sn
    except (ValueError, AssertionError):
        sys.exit(f"--shard {args.shard}: want i/n with 0 <= i < n")

    texts = Texts(args.texts)
    n, dim = len(texts), args.dim
    if args.check:
        check(Model(args), texts, args)
        return
    if not args.out:
        sys.exit("give the output (.i8bin), or --check N")
    meta_path, done_path = args.out + ".json", args.out + ".done"
    # What decides the vectors: a rerun with any of it changed would mix two kinds in one file.
    want = {"texts": os.path.abspath(args.texts), "rows": n, "model": args.model, "revision": args.revision,
            "dim": dim, "max_length": args.max_length, "task": args.task, "chunk": args.chunk,
            "pooling": "last token", "int8": "round(x * scale), clipped to +-127; x cut to dim, then normalized"}
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        diff = [k for k in want if meta.get(k) != want[k]]
        if diff:
            sys.exit(f"{args.out} was begun with other settings ({', '.join(f'{k}: {meta.get(k)!r}' for k in diff)}): "
                     f"give those, or remove {args.out}, its .json and .done to start over")

    model = Model(args)
    log(f"{n:,} rows of {args.texts} on {model.where}")
    if meta is None:
        if si != 0 and not args.scale_from:
            sys.exit("shard 0 sets the scale: start it first")
        k = min(args.sample, n)
        rows = np.sort(np.random.default_rng(args.seed).choice(n, size=k, replace=False))
        started = time.monotonic()
        x = run(model, prepare(model, texts.rows(rows), args.task), dim, args.batch_tokens)
        if args.scale_from:
            with open(args.scale_from) as f:
                scale = float(json.load(f)["scale"])
        else:
            scale = 127.0 / float(np.percentile(np.abs(x), PERCENTILE))
        np.save(args.out + ".sample.npy", x.astype(np.float16))
        np.save(args.out + ".sample-rows.npy", rows.astype(np.uint32))
        stats = sample_stats(x, scale)
        meta = {**want, "scale": scale,
                "scale_from": os.path.abspath(args.scale_from) if args.scale_from else f"{PERCENTILE}th percentile of |x|",
                "sample": {"rows": k, "seed": args.seed, **stats},
                "begun": time.strftime("%Y-%m-%d %H:%M:%S"), "versions": model.versions, "complete": False}
        with open(meta_path + ".tmp", "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(meta_path + ".tmp", meta_path)
        log(f"scale {scale:.2f} from {k:,} rows in {time.monotonic() - started:.0f} s: max |x| {stats['max_abs']:.3f}, "
            f"{100 * stats['clipped']:.4f}% clipped, cosine to float min {stats['cosine_min']:.4f} "
            f"median {stats['cosine_median']:.5f}")
    scale = float(meta["scale"])

    size = 8 + n * dim
    if not os.path.exists(args.out):
        with open(args.out, "wb") as f:
            f.write(struct.pack("<II", n, dim))
            f.truncate(size)
    with open(args.out, "rb") as f:
        if struct.unpack("<II", f.read(8)) != (n, dim) or os.path.getsize(args.out) != size:
            sys.exit(f"{args.out} is not {n} x {dim}: remove it, its .json and .done to start over")

    chunks = (n + args.chunk - 1) // args.chunk
    done = done_chunks(done_path)
    mine = [c for c in range(chunks) if c % sn == si and c not in done]
    log(f"{chunks} chunks of {args.chunk:,} rows, {len(done)} done; shard {si}/{sn}: {len(mine)} to go")
    fd = os.open(args.out, os.O_WRONLY)
    windows = [(c, lo, min(lo + WINDOW, (c + 1) * args.chunk, n))
               for c in mine for lo in range(c * args.chunk, min((c + 1) * args.chunk, n), WINDOW)]
    started, rows_done, tokens = time.monotonic(), 0, 0
    try:
        with ThreadPoolExecutor(1) as pool:
            ahead = pool.submit(lambda w: prepare(model, texts.range(w[1], w[2]), args.task), windows[0]) if windows else None
            buf, cur = None, None
            for k, (c, lo, hi) in enumerate(windows):
                prepared = ahead.result()
                if k + 1 < len(windows):
                    ahead = pool.submit(lambda w: prepare(model, texts.range(w[1], w[2]), args.task), windows[k + 1])
                if c != cur:
                    cur, base = c, c * args.chunk
                    buf = np.empty((min(base + args.chunk, n) - base, dim), dtype=np.int8)
                buf[lo - base:hi - base] = quantize(run(model, prepared, dim, args.batch_tokens), scale)
                rows_done += hi - lo
                tokens += int(prepared[1].sum())
                if hi == min(base + args.chunk, n):  # the chunk's last window: written, then listed done
                    write_all(fd, buf.tobytes(), 8 + base * dim)
                    os.fsync(fd)
                    with open(done_path, "a") as f:
                        f.write(f"{c}\n")
                        f.flush()
                        os.fsync(f.fileno())
                    el = time.monotonic() - started
                    left = sum(min((x + 1) * args.chunk, n) - x * args.chunk for x in mine) - rows_done
                    log(f"chunk {c} done: {rows_done:,} rows in {el / 3600:.2f} h, {rows_done / el:,.0f} rows/s, "
                        f"{tokens / el:,.0f} tokens/s; {left:,} rows left, about {left / max(rows_done / el, 1e-9) / 3600:.1f} h")
    finally:
        os.close(fd)
    done = done_chunks(done_path)
    if len(done) == chunks and not meta.get("complete"):
        meta.update(complete=True, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(meta_path + ".tmp", "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(meta_path + ".tmp", meta_path)
    log(f"{args.out}: {len(done)} of {chunks} chunks" + (", complete" if len(done) == chunks else ""))


if __name__ == "__main__":
    main()
