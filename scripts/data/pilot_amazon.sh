#!/usr/bin/env bash
# The pilot of the Amazon experiments: every data step of the full run on one category and the
# first 20,000 searches of one AmazonQAC part, in under an hour, then the int8 vectors through the
# engine once (it has not built on int8 data before). What it measures sets the full run: the
# embedding's tokens/s on this GPU (the full run's time), the duplicates, what int8 costs, and the
# recall the engine reaches on these vectors.
#
#   scripts/data/pilot_amazon.sh GPU RAW OUT [CATEGORY]
#
#   GPU       the GPU, by nvidia-smi's index (an idle one)
#   RAW       where fetch_amazon.py downloads ($A): the pilot's files are the full run's too
#   OUT       the pilot's own directory (e.g. $A/pilot): texts, vectors, index, logs
#   CATEGORY  All_Beauty by default (0.70M reviews, 0.11M items)
#
# Python is $P (the venv's: torch, transformers, pyarrow), else python3. A step whose output is
# there is skipped, so a rerun goes on where one stopped. OUT/pilot.log has everything,
# OUT/summary.txt the numbers.
set -euo pipefail
if [ $# -lt 3 ]; then sed -n '2,18p' "$0"; exit 2; fi
GPU=$1 RAW=$2 OUT=$3 CAT=${4:-All_Beauty}
PY=${P:-python3}
R=$(cd "$(dirname "$0")/../.." && pwd)
D=$R/scripts/data E=$R/engine/build
# The full run's instruction for the queries; the pilot uses it so that what it measures holds.
TASK="Given a product search query, retrieve relevant product information and customer reviews"
THREADS=32  # CPU threads for the build, ground truth and search: the server is shared
mkdir -p "$OUT"
exec > >(tee -a "$OUT/pilot.log") 2>&1
say() { echo; echo "[$(date +%T)] == $*"; }

say "0. GPU $GPU (nvidia-smi's numbering), $CAT, into $OUT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv -i "$GPU" || true

say "1. download: $CAT and AmazonQAC's first part"
python3 "$D/fetch_amazon.py" "$RAW" --categories "$CAT" --qac-parts 1

say "2. texts"
[ -f "$OUT/corpus-text.json" ] || "$PY" "$D/amazon_texts.py" corpus "$RAW" "$OUT" --categories "$CAT"
[ -f "$OUT/queries-text.json" ] || "$PY" "$D/amazon_texts.py" queries "$RAW" "$OUT" --qac-parts 1 --first 20000

say "3. embedding on GPU $GPU"
[ -f "$OUT/corpus.i8bin.json" ] || "$PY" "$D/embed_texts.py" "$OUT/corpus-text" --check 256 --gpu "$GPU"
"$PY" "$D/embed_texts.py" "$OUT/corpus-text" "$OUT/corpus.i8bin" --gpu "$GPU" | tee -a "$OUT/embed-corpus.log"
"$PY" "$D/embed_texts.py" "$OUT/queries-text" "$OUT/queries.i8bin" --gpu "$GPU" --task "$TASK" \
    --scale-from "$OUT/corpus.i8bin.json" | tee -a "$OUT/embed-queries.log"

say "4. the query file: the 20,000 searches in time order"
[ -f "$OUT/queries-20k.i8bin" ] || "$PY" "$D/query_file.py" "$OUT/queries-stream.npy" "$OUT/queries.i8bin" "$OUT/queries-20k.i8bin"

say "5. the engine on int8: build (1% heads, 64-byte PQ codes), ground truth, recall"
if [ ! -f "$OUT/index/meta.txt" ]; then
  start=$(date +%s)
  "$E/fusion_build" --base "$OUT/corpus.i8bin" --out "$OUT/index" --centroid-ratio 0.01 --pq-m 64 --threads $THREADS
  echo "built in $(( $(date +%s) - start )) s" | tee "$OUT/build-time.txt"
fi
[ -f "$OUT/gt.ibin" ] || "$E/fusion_gt" --base "$OUT/corpus.i8bin" --queries "$OUT/queries-20k.i8bin" \
    --out "$OUT/gt.ibin" --nq 2000 --k 100 --threads $THREADS
rm -f "$OUT/search.csv"
"$E/fusion_search" --index "$OUT/index" --queries "$OUT/queries-20k.i8bin" --gt "$OUT/gt.ibin" --nq 2000 --k 10 \
    --nprobe 16,32,64,128 --rerank 100,200 --backend cpu --no-heuristic --threads 8 --csv "$OUT/search.csv"

say "6. summary"
"$PY" - "$OUT" <<'EOF' | tee "$OUT/summary.txt"
import json, os, re, sys

d = sys.argv[1]
c = json.load(open(f"{d}/corpus-text.json"))
lines, empty, dup = (sum(x[k] for x in c["inputs"]) for k in ("lines", "empty", "duplicates"))
print(f"corpus: {lines:,} lines, {empty:,} empty, {dup:,} duplicates ({100 * dup / max(lines, 1):.1f}%): "
      f"{c['rows']:,} documents, {c['bytes'] / 1e6:,.0f} MB of text")
m = json.load(open(f"{d}/corpus.i8bin.json"))
s = m["sample"]
print(f"int8: scale {m['scale']:.1f} (max |x| {s['max_abs']:.3f}), {100 * s['clipped']:.4f}% of the values clipped, "
      f"cosine to float min {s['cosine_min']:.4f} median {s['cosine_median']:.5f} over {s['rows']:,} sampled rows")


def rate(log):
    hits = [l for l in open(log) if "tokens/s" in l]
    if not hits:
        return None
    rows, tokens = re.search(r"([\d,]+) rows/s, ([\d,]+) tokens/s", hits[-1]).groups()
    return int(rows.replace(",", "")), int(tokens.replace(",", ""))


cr, qr = rate(f"{d}/embed-corpus.log"), rate(f"{d}/embed-queries.log")
if cr:
    print(f"embedding, corpus: {cr[0]:,} documents/s, {cr[1]:,} tokens/s on this GPU; the whole corpus, "
          f"about 44B tokens (an estimate), would take {44e9 / cr[1] / 86400:.1f} days on it")
if qr:
    print(f"embedding, queries: {qr[0]:,} terms/s; all 39.6M terms would take {39.6e6 / qr[0] / 3600:.1f} h")
q = json.load(open(f"{d}/queries-text.json"))
print(f"queries: {q['searches']:,} searches, {q['terms']:,} distinct terms, t {q['first_t']} to {q['last_t']} s")
meta = dict(l.strip().split("=", 1) for l in open(f"{d}/index/meta.txt") if "=" in l)
built = open(f"{d}/build-time.txt").read().strip() if os.path.exists(f"{d}/build-time.txt") else ""
print(f"index: {int(meta['num_vectors']):,} vectors, {int(meta['num_heads']):,} heads, "
      f"{float(meta['avg_replicas']):.2f} postings per vector; {built}")
for f in sorted(os.listdir(f"{d}/index")):
    print(f"  {f}: {os.path.getsize(os.path.join(d, 'index', f)) / 1e6:,.1f} MB")
print("recall@10 (fusion_search, CPU, the first 2,000 queries):")
print(open(f"{d}/search.csv").read().rstrip())
EOF
say "done: $OUT/summary.txt"
