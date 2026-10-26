#!/usr/bin/env bash
# Waits until a GPU is idle, then runs a command: for a shared machine whose GPUs free up at
# times nobody announces (the lab server). Idle means no compute process on the GPU, less than
# -m MiB of its memory in use and under 10% utilization, at two checks a minute apart, since one
# gap between someone's jobs is not a free GPU. It checks every -i seconds (600). The command
# then runs in the foreground, and its exit status is the script's.
#
#   scripts/when_gpu_idle.sh [-i SECONDS] [-m MIB] GPU COMMAND [ARGS...]
#
# GPU is nvidia-smi's index, the numbering embed_texts.py's --gpu takes too. Run it in tmux: it
# waits as long as it takes.
set -euo pipefail
interval=600 maxmem=1024
while getopts "i:m:" o; do
  case $o in i) interval=$OPTARG ;; m) maxmem=$OPTARG ;; *) exit 2 ;; esac
done
shift $((OPTIND - 1))
if [ $# -lt 2 ]; then sed -n '2,11p' "$0"; exit 2; fi
gpu=$1
shift
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu") || { echo "no GPU $gpu (nvidia-smi's index)"; exit 2; }

idle() {
  local line mem util apps
  # A failed query counts as busy: the next check tries again.
  line=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu" 2>/dev/null) || return 1
  read -r mem util <<<"${line//,/ }"
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "$uuid" || true)
  echo "[$(date +%T)] GPU $gpu: $mem MiB in use, $util% busy, $apps processes"
  [ "$apps" -eq 0 ] && [ "$mem" -lt "$maxmem" ] && [ "$util" -lt 10 ]
}

while true; do
  if idle; then
    sleep 60
    if idle; then break; fi
  fi
  sleep "$interval"
done
echo "[$(date +%T)] GPU $gpu is idle: $*"
exec "$@"
