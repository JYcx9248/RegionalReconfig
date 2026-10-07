#!/usr/bin/env python3
"""Embed QA questions as query vectors (.fbin: u32 count, u32 dim, float32 rows) for a corpus
embedded with a Qwen3-Embedding model, so that the load generator and fusion_gt can use them.

The corpus's manifest.json says how its vectors were made; this follows it. Wikipedia c1024:
Qwen3-Embedding-0.6B at revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3, plain text without a
document instruction or chat template, last-token pooling, unit length, computed in bfloat16
(vLLM) and stored in float16. Queries get the model card's instruction prefix
("Instruct: <task>\\nQuery:<question>"); the manifest does not record one, so take the
pipeline's own if it used another (--task). Questions come from any of:

  --nq DIR          Natural Questions parquet files (column question.text)
  --simpleqa CSV    SimpleQA (column problem)
  --musique JSONL   MuSiQue (field question)
  --frames TSV      FRAMES (column Prompt)

in that order; OUT.tsv lists them (row, source, text). --dry-run writes only the list.

--check-npy SHARD.npy --check-parquet SHARD.parquet embeds some of the shard's chunks the way
the corpus was embedded (no instruction) and compares them with the stored vectors: a cosine
above 0.99 everywhere means this pipeline reproduces the corpus's, so the queries it embeds
live in the same space. Row i of a shard is row i of its source parquet (the shard's .json:
source_parquet, rows == source_rows).

Needs torch and transformers (and pyarrow for --nq and --check-parquet):

    python3 scripts/data/embed_queries.py --nq .../natural_questions/default \\
        --simpleqa .../simple_qa_test_set.csv --out queries.fbin --revision COMMIT
    python3 scripts/data/embed_queries.py --check-npy .../shards/part-00000.embeddings.npy \\
        --check-parquet .../corpora/<c1024>/part-00000.parquet --revision COMMIT
"""

import argparse
import csv
import json
import os
import struct
import sys

import numpy as np

csv.field_size_limit(sys.maxsize)


def nq(d):
    import pyarrow.parquet as pq
    out = []
    for name in sorted(n for n in os.listdir(d) if n.endswith(".parquet")):
        for q in pq.read_table(os.path.join(d, name), columns=["question"]).column("question").to_pylist():
            out.append(q["text"] if isinstance(q, dict) else q)
    return out


def simpleqa(path):
    return [r["problem"] for r in csv.DictReader(open(path, newline="", encoding="utf-8"))]


def musique(path):
    return [json.loads(l)["question"] for l in open(path, encoding="utf-8") if l.strip()]


def frames(path):
    return [r["Prompt"] for r in csv.DictReader(open(path, newline="", encoding="utf-8"), delimiter="\t")]


def load(args):
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # The corpus was computed in bfloat16; on a CPU, float32.
    dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    # transformers 4.56 renamed torch_dtype to dtype (5.x still takes the old name, with a warning).
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    kw = {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, padding_side="left")
    model = AutoModel.from_pretrained(args.model, revision=args.revision, **kw).to(dev).eval()
    print(f"{args.model} at {args.revision or 'the default revision'} on {dev} ({dtype})")
    return tok, model, dev


def embed(texts, tok, model, dev, args, task):
    """Unit vectors (float32) of texts; with a task, as queries ("Instruct: ...\\nQuery:...")."""
    import torch
    import torch.nn.functional as F

    out = []
    with torch.no_grad():
        for s in range(0, len(texts), args.batch):
            batch = texts[s:s + args.batch]
            if task is not None:
                batch = [f"Instruct: {task}\nQuery:{t}" for t in batch]
            enc = tok(batch, padding=True, truncation=True, max_length=args.max_length,
                      return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state
            # Left padding: every sequence ends at the last position.
            out.append(F.normalize(h[:, -1].float(), p=2, dim=1).cpu().numpy())
            print(f"  {min(s + args.batch, len(texts))} of {len(texts)}", flush=True)
    return np.concatenate(out).astype(np.float32)


def check(args):
    import pyarrow.parquet as pq

    stored = np.load(args.check_npy, mmap_mode="r")
    pf = pq.ParquetFile(args.check_parquet)
    names = pf.schema_arrow.names
    col = args.check_column or ("text" if "text" in names else None)
    if col is None or col not in names:
        sys.exit(f"which column holds the chunk text? --check-column one of {names}")
    if pf.metadata.num_rows != stored.shape[0]:
        sys.exit(f"{args.check_parquet} has {pf.metadata.num_rows} rows, {args.check_npy} {stored.shape[0]}: "
                 "not each other's source")
    rows = np.linspace(0, stored.shape[0] - 1, min(args.check_rows, stored.shape[0])).astype(int)
    texts = pf.read(columns=[col]).column(col).to_pylist()
    tok, model, dev = load(args)
    ours = embed([texts[i] for i in rows], tok, model, dev, args, task=None)
    ref = np.asarray(stored[rows], dtype=np.float32)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    cos = (ours * ref).sum(axis=1)
    print(f"cosine to the stored vectors over {len(rows)} chunks: min {cos.min():.4f}, "
          f"median {np.median(cos):.4f}, max {cos.max():.4f}")
    if cos.min() > 0.99:
        print("same model and settings: queries embedded this way are in the corpus's space")
    else:
        worst = rows[np.argsort(cos)[:3]]
        sys.exit(f"NOT the corpus's embedding (lowest at rows {list(worst)}): check the model, revision, "
                 "--max-length and the text column")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nq")
    ap.add_argument("--simpleqa")
    ap.add_argument("--musique")
    ap.add_argument("--frames")
    ap.add_argument("--out", help="query vectors (.fbin); the list goes next to it (.tsv)")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--revision", default=None, help="the corpus model's commit (its manifest.json)")
    ap.add_argument("--task", default="Given a web search query, retrieve relevant passages that answer the query")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=8192, help="tokens (the corpus's manifest: 8192)")
    ap.add_argument("--dry-run", action="store_true", help="write the question list only")
    ap.add_argument("--check-npy", help="a corpus shard (.npy) to compare with")
    ap.add_argument("--check-parquet", help="its source parquet (the shard's .json: source_parquet)")
    ap.add_argument("--check-column", help="the parquet column with the chunk text (default: text)")
    ap.add_argument("--check-rows", type=int, default=64)
    args = ap.parse_args()

    if args.check_npy or args.check_parquet:
        if not (args.check_npy and args.check_parquet):
            sys.exit("--check-npy and --check-parquet go together")
        check(args)
        return
    if not args.out:
        sys.exit("--out is required")
    texts, sources = [], []
    for name, fn in (("nq", nq), ("simpleqa", simpleqa), ("musique", musique), ("frames", frames)):
        path = getattr(args, name)
        if path:
            got = [t.strip() for t in fn(path) if t and t.strip()]
            print(f"{name}: {len(got)} questions from {path}")
            texts += got
            sources += [name] * len(got)
    if not texts:
        sys.exit("no questions: give at least one of --nq --simpleqa --musique --frames")
    tsv = os.path.splitext(args.out)[0] + ".tsv"
    with open(tsv, "w", encoding="utf-8") as f:
        for i, (s, t) in enumerate(zip(sources, texts)):
            f.write(f"{i}\t{s}\t{' '.join(t.split())}\n")
    print(f"{tsv}: {len(texts)} questions")
    if args.dry_run:
        return
    tok, model, dev = load(args)
    v = embed(texts, tok, model, dev, args, task=args.task)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<II", *v.shape))
        v.tofile(f)
    print(f"{args.out}: {v.shape[0]} x {v.shape[1]} float32, norms "
          f"{np.linalg.norm(v, axis=1).min():.4f} .. {np.linalg.norm(v, axis=1).max():.4f}")


if __name__ == "__main__":
    main()
