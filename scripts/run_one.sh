#!/bin/bash
# One experiment run on a clean node state, as the results in results/*-v2 and later were made:
#
#   scripts/run_one.sh CONFIG.yaml PROTOCOL OUT_DIR ['{cluster: {placement: even-reversible-blocks}}']
#
# The run's configuration is a YAML file built on CONFIG (base: CONFIG) that sets
# cluster.protocol to PROTOCOL and, with the optional YAML (or JSON) mapping, overrides any other
# key. Before the run: the nodes' directories under the work directory are removed (the
# partitions and the index stay), the page cache is dropped, and with swap on vm.swappiness is
# set to 1 -- at the default 60 the kernel swaps node memory out under page-cache pressure, and a
# run that swaps in during its load is not comparable. System memory, swap and swap-ins are
# logged every 5 s to OUT_DIR/../mem-<protocol>.log; the last line printed says how many pages
# were swapped in. Dropping the page cache and setting swappiness need root; without it (a shared
# server) only the run's own files leave the page cache (scripts/evict_cache.py: the work
# directory, the index, the queries, the ground truth) and swappiness stays as it is -- check the
# swap-ins.
set -u
cd "$(dirname "$0")/.."
CFG=$1; P=$2; OUT=$3; OVR=${4:-"{}"}
TMP=$(mktemp --suffix=.yaml)
python3 - "$CFG" "$P" "$OVR" "$TMP" <<'PY'
import os, sys, yaml
cfg, protocol, overrides, out = sys.argv[1:]
sys.path.insert(0, "scripts")
import experiment_config as ec
c = ec.merge({"base": os.path.abspath(cfg), "cluster": {"protocol": protocol}}, yaml.safe_load(overrides) or {})
open(out, "w").write(ec.dump(c))
ec.load(out)  # every key known
PY
[ $? = 0 ] || exit 1
FLAT=$(mktemp --suffix=.json)
python3 -c "import json,sys; sys.path.insert(0, 'scripts'); import experiment_config as ec
json.dump(ec.flatten(ec.load(sys.argv[1])), open(sys.argv[2], 'w'))" "$TMP" "$FLAT"
W=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('WorkDir', '/tmp/rtier-run'))" "$FLAT")
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
print('\n'.join(p for p in (c.get('WorkDir'), c.get('IndexDir'), c.get('Queries'), c.get('GroundTruth')) if p))" "$FLAT")
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
rm -f "$TMP" "$FLAT"
# A calibration run writes calibration.json (and calib-<rate>.csv), not latency.csv.
RES=latency.csv; grep -q '"Calibrate"' "$OUT/experiment.json" 2>/dev/null && RES=calibration.json
echo "exit $rc $(date +%T), $RES: $(test -f "$OUT/$RES" && echo yes || echo NO), min available $(sort -k2 -n "$MEM" | head -1 | awk '{print $2}') MB, swap-ins $(( $(tail -1 "$MEM" | awk '{print $4}') - $(head -1 "$MEM" | awk '{print $4}') )) pages"
exit $rc
