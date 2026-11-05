#!/usr/bin/env python3
"""The Wikipedia update workload (U14): what a newer snapshot deletes from our base and what it
inserts, and the batches an update run applies.

    python3 scripts/data/wiki_diff.py diff --old-chunks DIR --old-articles DIR --old-embeddings DIR \\
        --new-chunks DIR --new-articles DIR --new-embeddings DIR [--ref-articles DIR] [--index DIR] --out DIR
    python3 scripts/data/wiki_diff.py batches --diff DIR --batches 4 --inserts-per-batch 0.05 [--seed 1] --out DIR

Needs numpy and pyarrow (the lab server's venv has both).

Our base's vector i is row i of the old embeddings (their shards in natural order, as
npy_to_fbin.py read them) and row i of the old chunk corpus. Every embedding shard's .json names its
source parquet and that file's SHA-256, which is checked for every shard, on both sides: the rows
line up by construction, not by assumption. With --index, the index's vector count and
dimension must match too.

Chunks are matched by the SHA-256 of their text, which the corpora store per chunk, as multisets:
an identical text in both snapshots is the same vector, whichever article holds it, so a renamed
article moves nothing. An old chunk without a match is deleted (its ID is its base row) and a new
one without a match is inserted (its row in the new embeddings). An edit is a delete and an
insert, and since the chunks are fixed token windows, every chunk after an edit changes. That
only measures edits when both snapshots chunk, extract and embed the same way: the chunking and
tokenizer must be equal and so must the extraction (refused otherwise), the embedding model's
revision, files and settings must be equal, and a sample of matched chunks must have the same
embedding in both (cosine > 0.99).

Articles, by title, group the changes: a batch applies an article's deletes and inserts
together, so that no epoch holds two versions of an article, or none. --ref-articles is the newer
snapshot's article selection applied to the old dump (peichun's u1 2025-08): our base selects
articles by an older rule, so an article in only one of the two snapshots can be a difference
of selection rather than of time. Such an article is added only if the reference lacks it too,
and removed only if the reference has it; otherwise it is left alone (its old chunks stay, its
new ones are not inserted) and counted apart. An article in both snapshots takes the newer
version in full.

diff writes, into --out:
  diff.npz    del_ids (u32 base IDs) and del_group, ins_rows (i64 rows of the new embeddings) and
              ins_group, sorted by group; group_kind (0 edited, 1 added, 2 removed); the
              selection differences left alone (sel_del_ids, sel_ins_rows); new_offsets (the
              first row of each new embedding shard)
  groups.tsv  group, kind, deletes, inserts, title
  diff.json   the counts, the checks, the inputs (paths, manifest hashes), how long it took

batches shuffles the groups (--seed) and hands them out in that order: each batch takes groups
until its inserts reach --inserts-per-batch (a fraction of the base below 1, a count otherwise),
and an article's deletes come with it. It writes, into --out:
  batch-K.fbin          the inserts, float32 unit rows as npy_to_fbin.py wrote the base, in the
                        new snapshot's row order (rtier-updater -inserts)
  batch-K-deletes.bin   the IDs to delete, the engine's u32 array (rtier-updater -deletes)
  batches.json          per batch: the counts and the IDs its inserts will get -- rtier-updater
                        hands out IDs in file order after those of earlier batches, so batch K's
                        first is the base count plus the inserts before it, if the batches run in
                        order and none fails
"""

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time

import numpy as np

KINDS = ("edited", "added", "removed")
CHUNK = 1 << 16  # rows converted at a time by batches


def natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fail(msg):
    sys.exit("wiki_diff: " + msg)


# ---------------------------------------------------------------------------------------------
# inputs

def corpus_shards(d):
    """A chunk corpus' shards in natural order: [(parquet path, chunks, parquet sha256)]."""
    s = os.path.join(d, "shards")
    names = sorted((n for n in os.listdir(s) if n.endswith(".parquet")), key=natural_key)
    if not names:
        fail("no shards/*.parquet in %s" % d)
    out = []
    for n in names:
        j = load_json(os.path.join(s, n[:-len(".parquet")] + ".json"))
        out.append((os.path.join(s, n), int(j["chunks"]), j["parquet_sha256"]))
    return out


def embedding_shards(d):
    """An embedding set's shards in natural order: [(npy path, rows, source parquet, its sha256)]."""
    s = os.path.join(d, "shards")
    names = sorted((n for n in os.listdir(s) if n.endswith(".embeddings.npy")), key=natural_key)
    if not names:
        fail("no shards/*.embeddings.npy in %s" % d)
    out = []
    for n in names:
        j = load_json(os.path.join(s, n[:-len(".embeddings.npy")] + ".json"))
        if j["rows"] != j["source_rows"]:
            fail("%s: %d rows of %d source rows" % (n, j["rows"], j["source_rows"]))
        out.append((os.path.join(s, n), int(j["rows"]), j["source_parquet"], j["source_parquet_sha256"]))
    return out


def aligned(corpus, emb, label):
    """Checks that embedding shard i embeds corpus shard i, row for row: (first row of each
    shard + the total, dim)."""
    if len(corpus) != len(emb):
        fail("%s: %d chunk shards, %d embedding shards" % (label, len(corpus), len(emb)))
    offsets, dim = [0], None
    for (path, chunks, sha), (npy, rows, src, src_sha) in zip(corpus, emb):
        if os.path.basename(path) != src or sha != src_sha or chunks != rows:
            fail("%s: %s embeds %s (%s, %d rows), not %s (%s, %d chunks)"
                 % (label, os.path.basename(npy), src, src_sha[:12], rows, os.path.basename(path), sha[:12], chunks))
        a = np.load(npy, mmap_mode="r")
        if a.ndim != 2 or a.shape[0] != rows or (dim is not None and a.shape[1] != dim):
            fail("%s: %s has shape %s" % (label, npy, a.shape))
        dim = a.shape[1]
        offsets.append(offsets[-1] + rows)
    return np.array(offsets, dtype=np.int64), dim


def read_chunks(shards, label):
    """Every chunk's article_id and the first 16 bytes of its text's SHA-256, in corpus order:
    (article_id i64, key u64, check u64). Two different texts share a key with probability about
    n^2 / 2^65 (1e-5 at our size); a matched pair must agree on the check as well."""
    import pyarrow.parquet as pq
    aids, keys, checks = [], [], []
    for i, (path, chunks, _) in enumerate(shards):
        t = pq.read_table(path, columns=["article_id", "text_sha256"])
        if t.num_rows != chunks:
            fail("%s: %s has %d rows, its .json says %d" % (label, path, t.num_rows, chunks))
        h = t.column("text_sha256").combine_chunks()
        if h.null_count or h.type.byte_width != 32:
            fail("%s: %s has %d missing or non-SHA-256 text hashes" % (label, path, h.null_count))
        b = np.frombuffer(h.buffers()[1], dtype=np.uint8)[h.offset * 32:(h.offset + len(h)) * 32].reshape(-1, 32)
        aids.append(t.column("article_id").to_numpy().astype(np.int64))
        keys.append(np.ascontiguousarray(b[:, :8]).view("<u8").ravel())
        checks.append(np.ascontiguousarray(b[:, 8:16]).view("<u8").ravel())
        if (i + 1) % 50 == 0 or i + 1 == len(shards):
            log("  %s: %d of %d shards read" % (label, i + 1, len(shards)))
    return np.concatenate(aids), np.concatenate(keys), np.concatenate(checks)


def read_articles(d, label):
    """Every article's article_id (ascending) and title (an Arrow string array), whatever its
    status: an article whose text went empty still exists."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    s = os.path.join(d, "shards")
    names = sorted((n for n in os.listdir(s) if n.endswith(".parquet")), key=natural_key)
    ids, titles = [], []
    for n in names:
        t = pq.read_table(os.path.join(s, n), columns=["article_id", "title"])
        ids.append(t.column("article_id").to_numpy().astype(np.int64))
        titles.append(t.column("title").combine_chunks())
    ids = np.concatenate(ids)
    titles = pa.concat_arrays(titles)
    if len(ids) > 1 and not (np.diff(ids) > 0).all():
        fail("%s: article IDs are not ascending" % label)
    dup = len(titles) - pc.count_distinct(titles).as_py()
    if dup:
        log("  %s: %d titles occur more than once; their articles are grouped together" % (label, dup))
    return ids, titles


def titles_of(article_ids, ids, titles, label):
    import pyarrow as pa
    rows = np.searchsorted(ids, article_ids)
    if len(article_ids) and (rows.max() >= len(ids) or (ids[rows] != article_ids).any()):
        fail("%s: chunks name articles the article corpus lacks" % label)
    return titles.take(pa.array(rows))


def manifest(d):
    return load_json(os.path.join(d, "manifest.json"))


def check_settings(a, args):
    """Refuses inputs that would measure a difference of processing as edits. a: the manifests."""
    oc, nc = a["old_chunks"]["configuration"], a["new_chunks"]["configuration"]
    for k in ("chunking", "tokenizer"):
        if oc.get(k) != nc.get(k):
            fail("the snapshots are chunked differently (%s): %s vs %s" % (k, oc.get(k), nc.get(k)))
    oa, na = a["old_articles"]["configuration"], a["new_articles"]["configuration"]
    if oa.get("extraction") != na.get("extraction"):
        fail("the snapshots are extracted differently: %s vs %s" % (oa.get("extraction"), na.get("extraction")))
    # Each input is the one its successor was made from.
    for chunks, arts, side in (("old_chunks", "old_articles", "old"), ("new_chunks", "new_articles", "new")):
        if a[chunks]["configuration"]["source"]["config_hash"] != a[arts]["config_hash"]:
            fail("%s: the chunk corpus was not made from that article corpus" % side)
    for emb, chunks, side in (("old_embeddings", "old_chunks", "old"), ("new_embeddings", "new_chunks", "new")):
        if a[emb]["configuration"]["source"]["chunk_config_hash"] != a[chunks]["config_hash"]:
            fail("%s: the embeddings were not made from that chunk corpus" % side)
    om, nm = a["old_embeddings"]["configuration"]["model"], a["new_embeddings"]["configuration"]["model"]
    for k in ("revision", "pooling", "normalize", "document_instruction", "chat_template", "input_format",
              "max_length", "local_fingerprint"):
        if om.get(k) != nm.get(k):
            fail("the snapshots are embedded differently (model.%s): %s vs %s" % (k, om.get(k), nm.get(k)))
    # A full-width prefix changes nothing, and no chunk reaches max_length to be truncated.
    for k in ("dimension_reduction", "output_dimension", "truncate_overlength"):
        if om.get(k) != nm.get(k):
            log("  embedding model.%s: %s vs %s (the cosine check below decides)" % (k, om.get(k), nm.get(k)))
    if args.ref_articles:
        ra = a["ref_articles"]["configuration"]
        if ra.get("extraction") != oa.get("extraction") or ra["source"]["zim_name"] != oa["source"]["zim_name"]:
            fail("--ref-articles is not the old dump (%s) extracted the same way" % oa["source"]["zim_name"])


# ---------------------------------------------------------------------------------------------
# matching

def run_rank(s):
    """For a sorted array: each element's position within its run of equal values."""
    n = len(s)
    if not n:
        return np.zeros(0, dtype=np.int64)
    idx = np.arange(n)
    start = np.r_[True, s[1:] != s[:-1]]
    return idx - np.maximum.accumulate(np.where(start, idx, 0))


def multiset_match(ka, kb):
    """Pairs the r-th occurrence of each key in ka (in index order) with its r-th occurrence in kb:
    (indices into ka, indices into kb)."""
    oa = np.argsort(ka, kind="stable")
    ob = np.argsort(kb, kind="stable")
    sa, sb = ka[oa], kb[ob]
    ra = run_rank(sa)
    lo = np.searchsorted(sb, sa, "left")
    hi = np.searchsorted(sb, sa, "right")
    ok = ra < hi - lo
    return oa[ok], ob[lo[ok] + ra[ok]]


def classify(d_in_new, d_in_ref, i_in_old, i_in_ref):
    """Which unmatched chunks are changes of time: (real deletes, real inserts) as boolean masks.
    d_*: per unmatched old chunk, whether its article's title is in the new snapshot / the
    reference; i_*: per unmatched new chunk, whether it is in the old snapshot / the reference
    (None: no reference, every change counts)."""
    if d_in_ref is None:
        return np.ones(len(d_in_new), bool), np.ones(len(i_in_old), bool)
    return d_in_new | d_in_ref, i_in_old | ~i_in_ref


def read_rows(paths, offsets, rows):
    """The given global rows of an embedding set, float32, in the order given."""
    out = None
    shard = np.searchsorted(offsets, rows, "right") - 1
    for s in np.unique(shard):
        a = np.load(paths[s], mmap_mode="r")
        sel = np.flatnonzero(shard == s)
        x = np.asarray(a[rows[sel] - offsets[s]], dtype=np.float32)
        if out is None:
            out = np.empty((len(rows), x.shape[1]), dtype=np.float32)
        out[sel] = x
    return out


def quantiles(v):
    if not len(v):
        return None
    q = np.quantile(v, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    return {k: round(float(x), 4) for k, x in zip(("min", "p25", "p50", "p75", "p90", "p99", "max"), q)} | \
        {"mean": round(float(np.mean(v)), 4)}


# ---------------------------------------------------------------------------------------------
# diff

def diff(args):
    import pyarrow as pa
    import pyarrow.compute as pc
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    names = ("old_chunks", "old_articles", "old_embeddings", "new_chunks", "new_articles", "new_embeddings")
    dirs = {k: getattr(args, k) for k in names}
    if args.ref_articles:
        dirs["ref_articles"] = args.ref_articles
    mans = {k: manifest(d) for k, d in dirs.items()}
    check_settings(mans, args)

    log("checking that the embeddings line up with the chunks")
    old_c, new_c = corpus_shards(args.old_chunks), corpus_shards(args.new_chunks)
    old_e, new_e = embedding_shards(args.old_embeddings), embedding_shards(args.new_embeddings)
    old_off, dim = aligned(old_c, old_e, "old")
    new_off, new_dim = aligned(new_c, new_e, "new")
    n_old, n_new = int(old_off[-1]), int(new_off[-1])
    if new_dim != dim:
        fail("dimensions differ: %d vs %d" % (dim, new_dim))
    if args.index:
        meta = dict(line.strip().split("=", 1) for line in open(os.path.join(args.index, "meta.txt")) if "=" in line)
        if int(meta["num_vectors"]) != n_old or int(meta["dim"]) != dim:
            fail("the index has %s vectors of %s dimensions, the old embeddings %d of %d"
                 % (meta["num_vectors"], meta["dim"], n_old, dim))
    log("old: %d chunks in %d shards; new: %d chunks in %d shards; %d dimensions"
        % (n_old, len(old_c), n_new, len(new_c), dim))

    log("reading the chunks' text hashes")
    old_aid, old_key, old_chk = read_chunks(old_c, "old")
    new_aid, new_key, new_chk = read_chunks(new_c, "new")

    log("matching")
    pi, pj = multiset_match(old_key, new_key)
    bad = int((old_chk[pi] != new_chk[pj]).sum())
    if bad:
        fail("%d matched chunks differ beyond the first 8 bytes of their hash: match on more bytes" % bad)
    old_matched = np.zeros(n_old, bool)
    old_matched[pi] = True
    new_matched = np.zeros(n_new, bool)
    new_matched[pj] = True
    d_all = np.flatnonzero(~old_matched)  # base IDs
    i_all = np.flatnonzero(~new_matched)  # rows of the new embeddings
    log("%d matched; %d old chunks unmatched, %d new ones" % (len(pi), len(d_all), len(i_all)))

    log("checking the embeddings of %d matched chunks" % args.check_pairs)
    rng = np.random.default_rng(args.seed)
    pick = np.sort(rng.choice(len(pi), min(args.check_pairs, len(pi)), replace=False))
    a = read_rows([p for p, _, _, _ in old_e], old_off, pi[pick])
    b = read_rows([p for p, _, _, _ in new_e], new_off, pj[pick])
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    log("cosine of matched chunks: min %.5f, median %.5f" % (cos.min(), np.median(cos)))
    if cos.min() < 0.99:
        fail("matched chunks embed differently (min cosine %.4f): rows or models do not line up" % cos.min())

    log("reading the articles' titles")
    old_ids, old_titles = read_articles(args.old_articles, "old")
    new_ids, new_titles = read_articles(args.new_articles, "new")
    ref_titles = read_articles(args.ref_articles, "reference")[1] if args.ref_articles else None
    d_titles = titles_of(old_aid[d_all], old_ids, old_titles, "old")
    i_titles = titles_of(new_aid[i_all], new_ids, new_titles, "new")
    d_in_new = pc.is_in(d_titles, value_set=new_titles).to_numpy(zero_copy_only=False)
    i_in_old = pc.is_in(i_titles, value_set=old_titles).to_numpy(zero_copy_only=False)
    d_in_ref = pc.is_in(d_titles, value_set=ref_titles).to_numpy(zero_copy_only=False) if ref_titles is not None else None
    i_in_ref = pc.is_in(i_titles, value_set=ref_titles).to_numpy(zero_copy_only=False) if ref_titles is not None else None
    d_real, i_real = classify(d_in_new, d_in_ref, i_in_old, i_in_ref)

    log("grouping the changes by article")
    real_titles = pa.concat_arrays([d_titles.filter(pa.array(d_real)), i_titles.filter(pa.array(i_real))])
    groups = pc.unique(real_titles)
    gidx = pc.index_in(real_titles, value_set=groups).to_numpy(zero_copy_only=False).astype(np.int32)
    nd = int(d_real.sum())
    del_ids, del_group = d_all[d_real].astype(np.uint32), gidx[:nd]
    ins_rows, ins_group = i_all[i_real].astype(np.int64), gidx[nd:]
    g_in_old = pc.is_in(groups, value_set=old_titles).to_numpy(zero_copy_only=False)
    g_in_new = pc.is_in(groups, value_set=new_titles).to_numpy(zero_copy_only=False)
    kind = np.where(g_in_old & g_in_new, 0, np.where(g_in_new, 1, 2)).astype(np.int8)
    g_del = np.bincount(del_group, minlength=len(groups))
    g_ins = np.bincount(ins_group, minlength=len(groups))
    o = np.lexsort((del_ids, del_group))
    del_ids, del_group = del_ids[o], del_group[o]
    o = np.lexsort((ins_rows, ins_group))
    ins_rows, ins_group = ins_rows[o], ins_group[o]
    np.savez(os.path.join(args.out, "diff.npz"), del_ids=del_ids, del_group=del_group, ins_rows=ins_rows,
             ins_group=ins_group, group_kind=kind, sel_del_ids=d_all[~d_real].astype(np.uint32),
             sel_ins_rows=i_all[~i_real].astype(np.int64), new_offsets=new_off)
    with open(os.path.join(args.out, "groups.tsv"), "w") as f:
        f.write("group\tkind\tdeletes\tinserts\ttitle\n")
        for g, t in enumerate(groups.to_pylist()):
            f.write("%d\t%s\t%d\t%d\t%s\n" % (g, KINDS[kind[g]], g_del[g], g_ins[g], t.replace("\t", " ").replace("\n", " ")))

    # How much of an edited article changes: its deletes against all its chunks in the base.
    edited_d = d_real & d_in_new
    ua, ca = np.unique(old_aid, return_counts=True)
    ue, ce = np.unique(old_aid[d_all[edited_d]], return_counts=True)
    share = ce / ca[np.searchsorted(ua, ue)]
    n_art_old, n_art_new = len(old_titles), len(new_titles)
    common = int(pc.sum(pc.is_in(old_titles, value_set=new_titles)).as_py() or 0)
    sel_d_titles = pc.unique(d_titles.filter(pa.array(~d_real)))
    sel_i_titles = pc.unique(i_titles.filter(pa.array(~i_real)))
    pct = lambda x: round(100.0 * x / n_old, 3)
    summary = {
        "base": {"vectors": n_old, "dim": dim, "articles": n_art_old},
        "new": {"chunks": n_new, "articles": n_art_new, "titles_in_both": common},
        "chunks": {
            "matched": len(pi),
            "deleted": len(del_ids), "deleted_pct_of_base": pct(len(del_ids)),
            "inserted": len(ins_rows), "inserted_pct_of_base": pct(len(ins_rows)),
            "left_alone": {"old_chunks_kept": int((~d_real).sum()), "new_chunks_skipped": int((~i_real).sum())},
        },
        "articles": {
            **{k: int((kind == i).sum()) for i, k in enumerate(KINDS)},
            "left_alone": {"old_only": len(sel_d_titles), "new_only": len(sel_i_titles)},
            "edited_share_of_chunks_deleted": quantiles(share),
            "inserts_per_group": quantiles(g_ins), "deletes_per_group": quantiles(g_del),
        },
        "checks": {"shards_aligned": True, "matched_cosine": quantiles(cos), "pairs": int(len(cos))},
        "inputs": {k: {"path": os.path.abspath(d), "config_hash": mans[k].get("config_hash")} for k, d in dirs.items()},
        "index": os.path.abspath(args.index) if args.index else None,
        "new_embedding_shards": [os.path.abspath(p) for p, _, _, _ in new_e],
        "seconds": round(time.time() - t_start, 1),
        "script_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "argv": sys.argv,
    }
    with open(os.path.join(args.out, "diff.json"), "w") as f:
        json.dump(summary, f, indent=2)
    c, ar = summary["chunks"], summary["articles"]
    log("base %d chunks in %d articles; new %d chunks in %d articles (%d titles in both)"
        % (n_old, n_art_old, n_new, n_art_new, common))
    log("matched %d chunks; delete %d (%.2f%% of the base), insert %d (%.2f%%)"
        % (len(pi), len(del_ids), c["deleted_pct_of_base"], len(ins_rows), c["inserted_pct_of_base"]))
    log("articles: %d edited, %d added, %d removed; left alone: %d old-only, %d new-only articles "
        "(%d old chunks kept, %d new chunks skipped)"
        % (ar["edited"], ar["added"], ar["removed"], len(sel_d_titles), len(sel_i_titles),
           c["left_alone"]["old_chunks_kept"], c["left_alone"]["new_chunks_skipped"]))
    if ar["edited_share_of_chunks_deleted"]:
        log("an edited article loses %.0f%% of its chunks (median; mean %.0f%%)"
            % (100 * ar["edited_share_of_chunks_deleted"]["p50"], 100 * ar["edited_share_of_chunks_deleted"]["mean"]))
    log("wrote %s/diff.npz, groups.tsv, diff.json in %.0f s" % (args.out, time.time() - t_start))


# ---------------------------------------------------------------------------------------------
# batches

def write_u32(path, v):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(v)))
        np.asarray(v, dtype="<u4").tofile(f)


def batches(args):
    t_start = time.time()
    meta = load_json(os.path.join(args.diff, "diff.json"))
    z = np.load(os.path.join(args.diff, "diff.npz"))
    del_ids, del_group, ins_rows, ins_group = z["del_ids"], z["del_group"], z["ins_rows"], z["ins_group"]
    kind, new_off = z["group_kind"], z["new_offsets"]
    paths = meta["new_embedding_shards"]
    n_base, dim = meta["base"]["vectors"], meta["base"]["dim"]
    ng = len(kind)
    g_ins = np.bincount(ins_group, minlength=ng)
    g_del = np.bincount(del_group, minlength=ng)
    target = int(round(args.inserts_per_batch * n_base)) if args.inserts_per_batch < 1 else int(args.inserts_per_batch)
    if target <= 0:
        fail("--inserts-per-batch gives %d inserts" % target)
    if target * args.batches > len(ins_rows):
        fail("the diff inserts %d vectors: %d batches of %d need %d. At most %d batches of that size, or "
             "--inserts-per-batch %.4f for %d batches"
             % (len(ins_rows), args.batches, target, target * args.batches, len(ins_rows) // target,
                len(ins_rows) / args.batches / n_base, args.batches))
    order = np.random.default_rng(args.seed).permutation(ng)
    before = np.cumsum(g_ins[order]) - g_ins[order]  # inserts of the groups ahead in the order
    of = np.full(ng, -1, dtype=np.int64)
    of[order] = before // target  # a group belongs to the batch whose target it starts in
    of[of >= args.batches] = -1
    os.makedirs(args.out, exist_ok=True)
    out, first_id = [], n_base
    for k in range(args.batches):
        rows = np.sort(ins_rows[of[ins_group] == k])
        ids = np.sort(del_ids[of[del_group] == k])
        gk = np.flatnonzero(of == k)
        name = "batch-%d" % (k + 1)
        fbin = os.path.join(args.out, name + ".fbin")
        norms = [np.inf, 0.0]
        with open(fbin, "wb") as f:
            f.write(struct.pack("<II", len(rows), dim))
            for s in range(0, len(rows), CHUNK):
                x = read_rows(paths, new_off, rows[s:s + CHUNK])
                nrm = np.linalg.norm(x, axis=1)  # as npy_to_fbin.py normalized the base
                if (nrm == 0).any():
                    fail("a zero vector among the inserts")
                norms = [min(norms[0], float(nrm.min())), max(norms[1], float(nrm.max()))]
                x /= nrm[:, None]
                x.tofile(f)
        write_u32(os.path.join(args.out, name + "-deletes.bin"), ids)
        out.append({
            "batch": k + 1, "inserts": int(len(rows)), "deletes": int(len(ids)),
            "inserts_pct_of_base": round(100.0 * len(rows) / n_base, 3),
            "deletes_pct_of_base": round(100.0 * len(ids) / n_base, 3),
            "articles": {kn: int((kind[gk] == i).sum()) for i, kn in enumerate(KINDS)},
            "expected_first_id": int(first_id), "input_norms": [round(norms[0], 5), round(norms[1], 5)],
            "files": {"inserts": fbin, "deletes": os.path.join(args.out, name + "-deletes.bin")},
        })
        first_id += len(rows)
        log("%s: %d inserts (%.2f%% of the base), %d deletes (%.2f%%), %d articles"
            % (name, len(rows), out[-1]["inserts_pct_of_base"], len(ids), out[-1]["deletes_pct_of_base"], len(gk)))
    with open(os.path.join(args.out, "batches.json"), "w") as f:
        json.dump({"diff": os.path.abspath(args.diff), "seed": args.seed, "target_inserts": target,
                   "base_vectors": n_base, "batches": out, "unused_groups": int((of < 0).sum()),
                   "ids_after_all": int(first_id), "argv": sys.argv}, f, indent=2)
    log("wrote %d batches to %s in %.0f s; IDs after them: %d (Headroom at least %.3f)"
        % (args.batches, args.out, time.time() - t_start, first_id, first_id / n_base))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diff", help="match the snapshots' chunks and group the changes by article")
    for side, what in (("old", "our base's snapshot"), ("new", "the newer snapshot")):
        d.add_argument("--%s-chunks" % side, required=True, help="%s: chunk corpus (shards/*.parquet)" % what)
        d.add_argument("--%s-articles" % side, required=True, help="%s: article corpus (titles)" % what)
        d.add_argument("--%s-embeddings" % side, required=True, help="%s: embeddings (shards/*.npy)" % what)
    d.add_argument("--ref-articles", help="the newer snapshot's article selection on the old dump")
    d.add_argument("--index", help="our index: its vector count and dimension must match the old embeddings")
    d.add_argument("--check-pairs", type=int, default=256, help="matched chunks whose embeddings are compared")
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--out", required=True)
    b = sub.add_parser("batches", help="cut the diff into update batches")
    b.add_argument("--diff", required=True, help="the diff's output directory")
    b.add_argument("--batches", type=int, required=True)
    b.add_argument("--inserts-per-batch", type=float, required=True,
                   help="inserts per batch: a fraction of the base below 1, a count otherwise")
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--out", required=True)
    args = ap.parse_args()
    diff(args) if args.cmd == "diff" else batches(args)


if __name__ == "__main__":
    main()
