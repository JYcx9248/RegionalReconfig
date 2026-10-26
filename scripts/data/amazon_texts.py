#!/usr/bin/env python3
"""The texts of the Amazon experiments, ready to embed (scripts/data/embed_texts.py), from what
scripts/data/fetch_amazon.py downloaded into RAW:

    python3 scripts/data/amazon_texts.py corpus  RAW OUT [--categories All_Beauty,...]
    python3 scripts/data/amazon_texts.py queries RAW OUT [--qac-parts N] [--first N]

corpus: one document per review (its title and its text, a line each) and per item (its title,
features and description, a line each), category by category in name order, each category's
reviews before its items. A document is cut at --max-chars characters: the embedder keeps 512
tokens of it anyway. Empty documents are left out, and so is every text seen before, in any
category: exact duplicates ("Five Stars / Great", thousands of times) would be one point many
times over, all of them tied. Two passes over the files: a hash of each text first (128-bit
BLAKE2b), then the texts of the first occurrences. Writes

  OUT/corpus-text.bin, .off   the documents in row order, UTF-8 back to back; u64 offsets (N + 1)
  OUT/corpus-rows.npy         per row, where it comes from: (file, line), into the files that
                              corpus-text.json lists
  OUT/corpus-text.json        the inputs (and the Hugging Face revision they match), counts per
                              file, how the texts were made

Row i is vector i of the index built from the embeddings.

queries: AmazonQAC's train parts (the US log of September 2023: 395.6M searches, shuffled over 20
parts). What gets embedded is each distinct final search term (39.6M), as the log spells it; the
stream says which term was searched when:

  OUT/queries-text.bin, .off  the distinct terms, in order of first appearance in the parts
  OUT/queries-stream.npy      per search, sorted by time: (t: seconds since 2023-09-01 00:00:00,
                              as the log gives its times -- it states no time zone; term: its row)
  OUT/queries-text.json       the inputs and the counts

--first N keeps the first N searches of the stream and only the terms they use (a pilot: one
part holds some ten million distinct terms, an hour of embedding).

Needs numpy; pyarrow for queries (the lab server's venv has both).
"""

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from array import array

import numpy as np

FILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "amazon_files.json")
QAC_EPOCH = 1693526400  # 2023-09-01 00:00:00, the log's first day, as seconds since 1970


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def review_text(r: dict) -> str:
    return "\n".join(s for s in ((r.get("title") or "").strip(), (r.get("text") or "").strip()) if s)


def item_text(r: dict) -> str:
    parts = [r.get("title")] + list(r.get("features") or []) + list(r.get("description") or [])
    return "\n".join(s.strip() for s in parts if isinstance(s, str) and s.strip())


def corpus_inputs(raw: str, categories: list) -> list:
    """(kind, category, path) in row order; a file is .jsonl.gz or .jsonl, as it was fetched."""
    root = os.path.join(raw, "amazon-reviews-2023", "raw")
    out = []
    for c in categories:
        for kind, sub, name in (("review", "review_categories", f"{c}.jsonl"),
                                ("item", "meta_categories", f"meta_{c}.jsonl")):
            path = next((p for p in (os.path.join(root, sub, name + ".gz"), os.path.join(root, sub, name))
                         if os.path.exists(p)), None)
            if path is None:
                sys.exit(f"{os.path.join(root, sub, name)}(.gz) is missing: fetch it (fetch_amazon.py)")
            out.append((kind, c, path))
    return out


def documents(kind: str, path: str, max_chars: int):
    """The text of each line of a file, encoded (b"" for an empty one)."""
    text = review_text if kind == "review" else item_text
    with (gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")) as f:
        for line in f:
            yield text(json.loads(line))[:max_chars].encode("utf-8")


def corpus(args) -> None:
    with open(FILES) as fh:
        manifest = json.load(fh)["amazon-reviews-2023"]
    known = sorted({os.path.basename(f["path"])[: -len(".jsonl")].removeprefix("meta_") for f in manifest["files"]})
    cats = known if args.categories == "all" else [c.strip() for c in args.categories.split(",")]
    if set(cats) - set(known):
        sys.exit(f"unknown categories {sorted(set(cats) - set(known))}; they are: {' '.join(known)}")
    files = corpus_inputs(args.raw, sorted(cats))

    # Pass 1: a hash per line. Empty lines hash to zeros; they are dropped by `valid` below.
    hashes, valid = [], []
    for kind, cat, path in files:
        started = time.monotonic()
        h, v = bytearray(), bytearray()
        for b in documents(kind, path, args.max_chars):
            h += hashlib.blake2b(b, digest_size=16).digest() if b else bytes(16)
            v.append(1 if b else 0)
        hashes.append(np.frombuffer(h, dtype="V16"))
        valid.append(np.frombuffer(v, dtype=np.uint8).astype(bool))
        log(f"pass 1, {os.path.basename(path)}: {len(v):,} lines in {time.monotonic() - started:.0f} s")
    allh, allv = np.concatenate(hashes), np.concatenate(valid)
    del hashes, valid
    # np.unique sorts stably, so return_index gives each text's first occurrence in row order.
    _, first = np.unique(allh, return_index=True)
    del allh
    keep = np.zeros(len(allv), dtype=bool)
    keep[first] = True
    keep &= allv
    del first
    n = int(keep.sum())
    log(f"{len(keep):,} lines, {int((~allv).sum()):,} empty, {int(allv.sum()) - n:,} duplicates: {n:,} documents")

    # Pass 2: the kept texts, in row order.
    os.makedirs(args.out, exist_ok=True)
    lengths = np.empty(n, dtype=np.uint64)
    rows = np.empty(n, dtype=[("file", np.uint16), ("line", np.uint32)])
    row = g = 0
    stats = []
    with open(os.path.join(args.out, "corpus-text.bin"), "wb", buffering=1 << 24) as out:
        for fi, (kind, cat, path) in enumerate(files):
            lines = empty = 0
            kept, size = array("I"), array("Q")
            for li, b in enumerate(documents(kind, path, args.max_chars)):
                lines += 1
                empty += not b
                if g < len(keep) and keep[g]:
                    out.write(b)
                    kept.append(li)
                    size.append(len(b))
                g += 1
            k = len(kept)
            if row + k > n:
                break
            rows["file"][row:row + k] = fi
            rows["line"][row:row + k] = np.frombuffer(kept, dtype=np.uint32)
            lengths[row:row + k] = np.frombuffer(size, dtype=np.uint64)
            stats.append({"file": os.path.abspath(path), "kind": kind, "category": cat, "lines": lines,
                          "empty": empty, "duplicates": lines - empty - k, "rows": k, "first_row": row})
            row += k
            log(f"pass 2, {os.path.basename(path)}: rows {row - k:,} to {row:,}")
    if row != n or g != len(keep):
        sys.exit(f"the second pass read {g:,} lines and kept {row:,}, the first {len(keep):,} and {n:,}: "
                 "did the files change in between?")
    offsets = np.zeros(n + 1, dtype=np.uint64)
    np.cumsum(lengths, out=offsets[1:])
    offsets.tofile(os.path.join(args.out, "corpus-text.off"))
    np.save(os.path.join(args.out, "corpus-rows.npy"), rows)
    meta = {"made": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": n, "bytes": int(offsets[-1]),
            "source": {"repo": manifest["repo"], "revision": manifest["revision"],
                       "checked": "by fetch_amazon.py, against scripts/data/amazon_files.json"},
            "text": {"review": "title, then text, a line each", "item": "title, features, description, a line each",
                     "max_chars": args.max_chars, "empty": "left out",
                     "duplicates": "exact text (BLAKE2b-128): the first occurrence in row order kept"},
            "order": "categories by name; each category's reviews, then its items; lines in file order",
            "inputs": stats}
    with open(os.path.join(args.out, "corpus-text.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    log(f"{n:,} documents ({offsets[-1] / 1e9:.1f} GB of text) in {args.out}")


def queries(args) -> None:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    def seconds(col):
        """search_time as seconds since 1970: a timestamp, or a string as the card gives it."""
        if pa.types.is_timestamp(col.type):
            return pc.cast(pc.cast(col, pa.timestamp("s"), safe=False), pa.int64())
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return pc.cast(pc.strptime(col, format=fmt, unit="s"), pa.int64())
            except pa.ArrowInvalid:
                pass
        sys.exit(f"search_time is {col.type}, neither a timestamp nor YYYY-MM-DD HH:MM:SS")

    def usable(term):
        return pc.and_(pc.is_valid(term), pc.greater(pc.utf8_length(term), 0))

    d = os.path.join(args.raw, "amazonqac", "train")
    parts = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".parquet"))[: args.qac_parts]
    if not parts:
        sys.exit(f"no parquet parts in {d}: fetch them (fetch_amazon.py)")
    # Distinct terms in order of first appearance, per part and then over all of them; large
    # strings, as all the parts' distinct terms together pass 2 GB.
    distinct = []
    for p in parts:
        term = pq.read_table(p, columns=["final_search_term"]).column(0)
        distinct.append(pc.unique(pc.filter(term, usable(term))).cast(pa.large_string()))
    terms = pc.unique(pa.chunked_array(distinct))
    del distinct
    log(f"{len(terms):,} distinct terms in {len(parts)} parts")

    ts, ids, stats, dropped = [], [], [], 0
    for p in parts:
        t = pq.read_table(p, columns=["final_search_term", "search_time"])
        term, secs = t.column(0), seconds(t.column(1))
        ok = pc.and_(usable(term), pc.is_valid(secs))
        dropped += len(t) - pc.sum(ok).as_py()
        i = pc.index_in(pc.filter(term, ok).cast(pa.large_string()), value_set=terms)
        if i.null_count:
            sys.exit(f"{p}: {i.null_count} terms not among the distinct ones")
        s = pc.filter(secs, ok).to_numpy() - QAC_EPOCH
        if len(s) and (s.min() < 0 or s.max() >= 2 ** 32):
            sys.exit(f"{p}: search times before 2023-09-01 or far after: not this log?")
        ts.append(s.astype(np.uint32))
        ids.append(i.to_numpy().astype(np.uint32))
        stats.append({"file": os.path.abspath(p), "searches": len(t)})
        log(f"{os.path.basename(p)}: {len(t):,} searches")
    t, i = np.concatenate(ts), np.concatenate(ids)
    del ts, ids
    order = np.argsort(t, kind="stable")  # ties in the order of the parts and their rows
    stream = np.empty(len(t), dtype=[("t", np.uint32), ("term", np.uint32)])
    stream["t"], stream["term"] = t[order], i[order]
    del t, i, order
    if args.first:
        # A pilot's stream: its first searches, and only the terms they use, in the same order.
        stream = stream[: args.first]
        used, stream["term"] = np.unique(stream["term"], return_inverse=True)
        terms = terms.take(pa.array(used))
        log(f"the first {len(stream):,} searches, {len(terms):,} terms")

    # The terms' UTF-8 is already back to back in the array's data buffer.
    os.makedirs(args.out, exist_ok=True)
    offs = np.frombuffer(terms.buffers()[1], dtype=np.int64)[terms.offset: terms.offset + len(terms) + 1]
    start, end = int(offs[0]), int(offs[-1])
    with open(os.path.join(args.out, "queries-text.bin"), "wb") as out:
        out.write(memoryview(terms.buffers()[2])[start:end])
    (offs - start).astype(np.uint64).tofile(os.path.join(args.out, "queries-text.off"))
    np.save(os.path.join(args.out, "queries-stream.npy"), stream)
    meta = {"made": time.strftime("%Y-%m-%d %H:%M:%S"), "terms": len(terms), "searches": len(stream),
            "dropped": dropped, "first": args.first,
            "epoch": "2023-09-01 00:00:00, search_time as given (no time zone stated)",
            "first_t": int(stream["t"][0]) if len(stream) else None,
            "last_t": int(stream["t"][-1]) if len(stream) else None,
            "term_text": {"order": "first appearance in the parts, in name order", "spelling": "as in the log"},
            "inputs": stats}
    with open(os.path.join(args.out, "queries-text.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    log(f"{len(terms):,} terms, {len(stream):,} searches ({dropped:,} without a term or a time) in {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("corpus", "queries"))
    ap.add_argument("raw", help="the directory fetch_amazon.py downloaded into")
    ap.add_argument("out", help="where the text stores go")
    ap.add_argument("--categories", default="all", help="corpus: comma-separated categories, or all")
    ap.add_argument("--max-chars", type=int, default=8192, help="corpus: characters kept of a document")
    ap.add_argument("--qac-parts", type=int, default=20, help="queries: the first N train parts")
    ap.add_argument("--first", type=int, help="queries: keep the first N searches and their terms (a pilot)")
    args = ap.parse_args()
    corpus(args) if args.what == "corpus" else queries(args)


if __name__ == "__main__":
    main()
