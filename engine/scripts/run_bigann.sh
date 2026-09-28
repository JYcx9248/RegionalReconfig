#!/usr/bin/env bash
# End-to-end run on a machine with a GPU:
#   build -> unit tests -> download a dataset prefix -> ground truth -> index ->
#   GPU self-test -> parameter sweep -> thread scaling -> ablations (paper Figure 12)
#
#   scripts/run_bigann.sh                         # SIFT (BigANN) 10M, defaults below
#   DATASET=deep N=100000000 scripts/run_bigann.sh
#   CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=70" scripts/run_bigann.sh   # e.g. V100
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=${DATASET:-bigann}      # bigann | deep | msspacev
N=${N:-10000000}                # base vectors (multiple of 1M)
DATA=${DATA:-data}
BUILD=${BUILD:-build}
NPROBE=${NPROBE:-16,32,64,128}  # m
RERANK=${RERANK:-50,100,200}    # n
THREADS=${THREADS:-1,2,4,8,16,32,64}
SWEEP_THREADS=${SWEEP_THREADS:-16}
CSV=${CSV:-results-$DATASET.csv}

cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release ${CMAKE_ARGS:-}
cmake --build "$BUILD" -j
"$BUILD"/fusion_tests

python3 scripts/download_subset.py --dataset "$DATASET" --n "$N" --out "$DATA"
case "$DATASET" in
  bigann) EXT=u8bin ;;
  deep) EXT=fbin ;;
  msspacev) EXT=i8bin ;;
  *) echo "unknown dataset $DATASET" >&2; exit 1 ;;
esac
TAG="$DATASET-$((N / 1000000))M"
BASE="$DATA/$TAG.$EXT"
QUERY="$DATA/$DATASET-query.$EXT"
GT="$DATA/$TAG-gt.ibin"
[ -f "$GT" ] || "$BUILD"/fusion_gt --base "$BASE" --queries "$QUERY" --out "$GT" --k 100

INDEX=${INDEX:-idx-$TAG}
"$BUILD"/fusion_build --base "$BASE" --out "$INDEX" --layout both
"$BUILD"/fusion_selftest --index "$INDEX" --queries "$QUERY"   # GPU vs CPU filtering

S=("$BUILD"/fusion_search --index "$INDEX" --queries "$QUERY" --gt "$GT" --csv "$CSV")
echo "== parameter sweep (Figure 9/10 style)"
"${S[@]}" --nprobe "$NPROBE" --rerank "$RERANK" --threads "$SWEEP_THREADS"
echo "== thread scaling (Figure 11 style)"
"${S[@]}" --nprobe 64 --rerank 100 --threads "$THREADS"
echo "== ablations (Figure 12)"
A=(--nprobe 64 --rerank 100 --threads "$SWEEP_THREADS")
echo "-- MI (CPU)";          "${S[@]}" "${A[@]}" --backend cpu --layout id --no-heuristic --no-io-dedup
echo "-- MI (GPU)";          "${S[@]}" "${A[@]}" --backend gpu --layout id --no-heuristic --no-io-dedup
echo "-- MI (GPU) + HR";     "${S[@]}" "${A[@]}" --backend gpu --layout id --no-io-dedup
echo "-- FusionANNS (all)";  "${S[@]}" "${A[@]}" --backend gpu --layout bucket
echo "results appended to $CSV"
