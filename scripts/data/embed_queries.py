#!/usr/bin/env python3
"""Embed QA questions as query vectors (.fbin: u32 count, u32 dim, float32 rows) for a corpus
embedded with a Qwen3-Embedding model, so that the load generator and fusion_gt can use them.

Queries get the model's instruction prefix ("Instruct: <task>\\nQuery:<question>") and documents
do not, as the model card prescribes; last-token pooling with left padding, then unit length,
as the corpus vectors were made. Use the corpus's model and revision, and its query
instruction if its pipeline used another (--task). Questions come from any of:

  --nq DIR          Natural Questions parquet files (column question.text)
  --simpleqa CSV    SimpleQA (column problem)
  --musique JSONL   MuSiQue (field question)
  --frames TSV      FRAMES (column Prompt)

in that order; OUT.tsv lists them (row, source, text). --dry-run writes only the list.
Needs torch and transformers (and pyarrow for --nq):

    python3 scripts/data/embed_queries.py --nq .../natural_questions/default \\
        --simpleqa .../simple_qa_test_set.csv --out queries.fbin [--revision COMMIT]
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


def embed(texts, args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, padding_side="left")
    model = AutoModel.from_pretrained(args.model, revision=args.revision,
                                      torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), args.batch):
            batch = [f"Instruct: {args.task}\nQuery:{t}" for t in texts[s:s + args.batch]]
            enc = tok(batch, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state
            # Left padding: every sequence ends at the last position.
            v = F.normalize(h[:, -1].float(), p=2, dim=1)
            out.append(v.cpu().numpy())
            print(f"  {min(s + args.batch, len(texts))} of {len(texts)}", flush=True)
    return np.concatenate(out).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nq")
    ap.add_argument("--simpleqa")
    ap.add_argument("--musique")
    ap.add_argument("--frames")
    ap.add_argument("--out", required=True, help="query vectors (.fbin); the list goes next to it (.tsv)")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--revision", default=None, help="the corpus model's commit")
    ap.add_argument("--task", default="Given a web search query, retrieve relevant passages that answer the query")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dry-run", action="store_true", help="write the question list only")
    args = ap.parse_args()

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
    v = embed(texts, args)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<II", *v.shape))
        v.tofile(f)
    print(f"{args.out}: {v.shape[0]} x {v.shape[1]} float32, norms "
          f"{np.linalg.norm(v, axis=1).min():.4f} .. {np.linalg.norm(v, axis=1).max():.4f}")


if __name__ == "__main__":
    main()
