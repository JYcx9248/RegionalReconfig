#!/usr/bin/env bash
# Single-machine network emulation for rtier: one network namespace per node, joined by a
# bridge, with each node's link shaped by tc. Needs root.
#
#   sudo scripts/emulation/netns.sh up 4      # namespaces rtier-n1..n4 at 10.10.0.11..14
#   sudo scripts/emulation/netns.sh down 4
#
# The host side of the bridge is 10.10.0.1: run the controller there so the nodes can reach
# it (scripts/run_local.py does this when "Emulation": {"Netns": true}).
#
# What this emulates and what it does not (from the single-machine plan):
#   * network: delay and jitter (tc netem) and bandwidth (tc tbf) per node      (this script)
#   * CPU:     pin each node with `systemd-run --scope -p AllowedCPUs=... -p MemoryMax=...`
#   * SSD:     one null_blk or a separate NVMe namespace per node; O_DIRECT is on by default
#   * GPU:     CUDA MPS (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE) or MIG slices per node
#   Keep ratios (bandwidth : SSD throughput : compute), not absolute scale.
#
# TODO(experiment): DELAY / JITTER / RATE are placeholders; set them per experiment.
set -euo pipefail

CMD=${1:-}
N=${2:-3}
BR=rtierbr0
DELAY=${DELAY:-100us}     # TODO(experiment): one-way delay per link
JITTER=${JITTER:-10us}
RATE=${RATE:-10gbit}      # TODO(experiment): per-node bandwidth

shape() {  # $1 = namespace
  local ns=$1
  if ip netns exec "$ns" tc qdisc add dev eth0 root handle 1: netem delay "$DELAY" "$JITTER" 2>/dev/null; then
    ip netns exec "$ns" tc qdisc add dev eth0 parent 1: handle 2: tbf rate "$RATE" burst 256kb latency 10ms
  else
    echo "warning: tc netem is not available in this kernel; $ns gets bandwidth shaping only" >&2
    ip netns exec "$ns" tc qdisc add dev eth0 root handle 2: tbf rate "$RATE" burst 256kb latency 10ms
  fi
}

up() {
  ip link add "$BR" type bridge 2>/dev/null || true
  ip addr add 10.10.0.1/24 dev "$BR" 2>/dev/null || true
  ip link set "$BR" up
  for i in $(seq 1 "$N"); do
    ns=rtier-n$i
    ip netns add "$ns"
    ip link add "veth-n$i" type veth peer name eth0 netns "$ns"
    ip link set "veth-n$i" master "$BR" up
    ip -n "$ns" addr add "10.10.0.$((10 + i))/24" dev eth0
    ip -n "$ns" link set eth0 up
    ip -n "$ns" link set lo up
    shape "$ns"
  done
  echo "created $N namespaces on $BR (10.10.0.11-$((10 + N))), delay $DELAY, rate $RATE"
}

down() {
  for i in $(seq 1 "$N"); do
    ip netns del "rtier-n$i" 2>/dev/null || true
  done
  ip link del "$BR" 2>/dev/null || true
}

case "$CMD" in
  up) up ;;
  down) down ;;
  *) echo "usage: $0 up|down N" >&2; exit 2 ;;
esac
