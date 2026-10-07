#!/bin/bash
# One experiment run on a clean node state, as the results in results/*-v2 and later were made:
#
#   scripts/run_one.sh CONFIG PROTOCOL OUT_DIR ['{"Placement": "even-reversible-blocks", ...}']
#
# The config's "Protocol" is set to PROTOCOL and the optional JSON object overrides top-level
# keys. Before the run: the nodes' directories under WorkDir are removed (the partitions and
# the index stay), the page cache is dropped, and with swap on vm.swappiness is set to 1 --
# at the default 60 the kernel swaps node memory out under page-cache pressure, and a run that
# swaps in during its load is not comparable. System memory, swap and swap-ins are logged
# every 5 s to OUT_DIR/../mem-<protocol>.log; the last line printed says how many pages were
# swapped in. Dropping the page cache and setting swappiness need root; without it (a shared
# server) only the run's own files leave the page cache (scripts/evict_cache.py: WorkDir,
# IndexDir, Queries, GroundTruth) and swappiness stays as it is -- check the swap-ins.
set -u
cd "$(dirname "$0")/.."
CFG=$1; P=$2; OUT=$3; OVR=${4:-"{}"}
TMP=$(mktemp --suffix=.json)
python3 - "$CFG" "$P" "$OVR" "$TMP" <<'PY'
import json, sys
cfg, protocol, overrides, out = sys.argv[1:]
c = json.load(open(cfg))
c["Protocol"] = protocol
c.update(json.loads(overrides))
json.dump(c, open(out, "w"), indent=2)
PY
W=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['WorkDir'])" "$TMP")
for d in "$W"/n[0-9]*; do [ -d "$d" ] && rm -rf "$d"; done
rm -rf "$OUT"; mkdir -p "$(dirname "$OUT")"
MEM=$(dirname "$OUT")/mem-$P.log; : > "$MEM"
if [ "$(id -u)" = 0 ]; then
  if [ -n "$(swapon --noheadings 2>/dev/null)" ] && [ "$(cat /proc/sys/vm/swappiness)" != 1 ]; then
    sysctl -q -w vm.swappiness=1 && echo "vm.swappiness set to 1"
  fi
  sync; echo 3 > /proc/sys/vm/drop_caches
else
  [ -n "$(swapon --noheadings 2>/dev/null)" ] && echo "not root: vm.swappiness stays at $(cat /proc/sys/vm/swappiness)"
  mapfile -t OWN < <(python3 -c "import json,sys; c=json.load(open(sys.argv[1]))
print('\n'.join(p for p in (c['WorkDir'], c.get('IndexDir'), c.get('Queries'), c.get('GroundTruth')) if p))" "$TMP")
  python3 scripts/evict_cache.py "${OWN[@]}"
fi
echo "== $P -> $OUT $(date +%T)"
python3 scripts/run_local.py "$TMP" "$OUT" > "$OUT.log" 2>&1 &
RL=$!
while kill -0 $RL 2>/dev/null; do
  echo "$(cut -d' ' -f1 /proc/uptime) $(awk '/MemAvailable/{a=$2} /SwapTotal/{t=$2} /SwapFree/{f=$2} END{print int(a/1024), int((t-f)/1024)}' /proc/meminfo) $(awk '/^pswpin/{print $2}' /proc/vmstat)" >> "$MEM"
  sleep 5
done
wait $RL; rc=$?
mv "$OUT.log" "$OUT/run_local.log" 2>/dev/null
rm -f "$TMP"
# A calibration run writes calibration.json (and calib-<rate>.csv), not latency.csv.
RES=latency.csv; grep -q '"Calibrate"' "$OUT/experiment.json" 2>/dev/null && RES=calibration.json
echo "exit $rc $(date +%T), $RES: $(test -f "$OUT/$RES" && echo yes || echo NO), min available $(sort -k2 -n "$MEM" | head -1 | awk '{print $2}') MB, swap-ins $(( $(tail -1 "$MEM" | awk '{print $4}') - $(head -1 "$MEM" | awk '{print $4}') )) pages"
exit $rc
