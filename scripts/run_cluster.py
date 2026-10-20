#!/usr/bin/env python3
"""Run one rtier experiment on a cluster: the data nodes on other machines, started over SSH.

Koala's scripts/runExperiment.py, the way it deploys a cluster, with run_local.py's stages and
load (run_local.py is the single-machine case):

  1. check every host: SSH without a prompt, rsync, writable directories, an engine build its
     CPU and GPUs can run; stop what an earlier run with this work directory left there
     (Koala's cleanUp before a run)
  2. build the index, the ground truth and the partitions on this machine, the root
     (run_local.py's stages)
  3. copy what the nodes read to every host, at this machine's paths (Koala's syncRepo):
     rtier_node and rtier-agent, the index, the partitions, each rebuild's next generation. The
     index's page file goes only where a node of the initial deployment runs: nothing but the
     bootstrap reads it, and the other nodes get raw vectors from their peers. A next
     generation goes without it too. A host that sees a directory at its path already (this
     machine, a shared file system) gets no copy of it.
  4. start the controller here, then each node in order over SSH -- its rtier_node, then its
     agent -- and wait for the agent's registration before the next: nodes n1, n2, ... get IDs
     1, 2, ..., and the first cluster.initial_nodes form the deployment (Koala starts its
     workers in order)
  5. run the load here and trigger the reconfigurations (run_local.py's)
  6. the results arrive here: metrics.jsonl from the controller and, through the agents, each
     node's stats log (stats-<node>.jsonl) and samples of each node, its agent and its machine
     (cgroups.jsonl), all stamped by the controller when they arrive (Koala's metric
     collector): one clock, the one clock.json and the load generator use, so the hosts' clocks
     need no synchronization. Every process's log too (node-n1.log, agent-n1.log, ...).
  7. stop everything. A remote process runs with its SSH session's input open and is stopped
     when that closes: when this script stops it, and when this script dies however it does.

    python3 scripts/run_cluster.py configs/experiment.X.yaml results/run1

The configuration is run_local.py's (see its doc) with a hosts section, one host per node:

    hosts:
      nodes:                            # n1 runs on the first, n2 on the second, ...
        - 10.0.0.2                      # the SSH destination and the address, the same
        - {ssh: gpu3, addr: 10.0.0.3}   # an SSH destination (a ~/.ssh/config alias, say) and
                                        # the address the other machines reach it at
      root: 10.0.0.1                    # the address the nodes reach this machine at
                                        # (default: the one it reaches the first host from)
      user: chenxu                      # SSH login (default: SSH's)
      ssh_options: [-p, 22222]          # more ssh options

cluster.nodes defaults to the number of hosts. A host may run several nodes, each with its own
directory (WORK_DIR/n1, ...), sharing the machine. This machine may be one of the hosts: as in
Koala its processes start over SSH too, to itself.

Every host has this machine's paths: the repository's binaries, the work directory, the index,
the next generations -- the epoch table names a generation by its directories, and each node
opens them there. An engine built with -march=native (FUSION_NATIVE, the default) runs only on
CPUs with every instruction set extension of the one it was built on, and a CUDA build only on
GPUs its architectures cover: step 1 stops the run rather than let a node die with SIGILL later.
Build for every host's hardware then (cmake -DFUSION_NATIVE=OFF, -DCMAKE_CUDA_ARCHITECTURES=...).

cluster.gpus applies on each node's own machine: node i gets the i-th entry (in turns, as in
run_local.py), named by nvidia-smi index, UUID or PCI bus id there. cluster.resources and
cluster.emulation emulate several machines on one (run_local.py) and are refused here, except
cluster.resources.read_iops_per_node, the data node's own read budget.

Without an SSH server to try it on, PATH=$PWD/scripts/testing/fake_ssh:$PATH runs every remote
command on this machine, as the remote login shell would.
"""

import concurrent.futures
import glob
import json
import os
import re
import shlex
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_local as rl  # noqa: E402
from rtier_ctl import Controller  # noqa: E402

log = rl.log

# Runs "$@" with the SSH session's input kept on fd 3 and stops it when that input closes: when
# this script closes it, or when this script or its ssh dies. A background job of a
# non-interactive shell starts with SIGINT ignored, so it is SIGTERM (rtier_node and the agents
# take both), then SIGKILL after 10 s. The watcher's output goes nowhere: the session ends with
# the process, and a process that exits by itself shows as its ssh exiting.
WRAPPER = ('exec 3<&0; "$@" </dev/null 3<&- & p=$!; '
           '( read -r _ <&3; kill -TERM $p; i=0; '
           'while kill -0 $p && [ $i -lt 100 ]; do sleep 0.1; i=$((i+1)); done; '
           'kill -KILL $p ) >/dev/null 2>&1 & '
           'wait $p')


class Host:
    """A machine nodes run on, reached over SSH."""

    def __init__(self, ssh: str, addr: str, user, options: list):
        self.ssh, self.addr = ssh, addr
        self.dest = f"{user}@{ssh}" if user else ssh
        # BatchMode: fail rather than wait for a password. Keepalives: a run is hours of an
        # idle session, which a NAT on the way may otherwise drop -- and that stops the node.
        self.ssh_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30"] + list(options)
        self.rsync_ok = False
        self.bad_gpus = {}  # UUID -> compute capability, of GPUs the CUDA build has no code for

    def argv(self, script: str) -> list:
        """ssh running script with sh on the host."""
        return self.ssh_cmd + [self.dest, "sh -c " + shlex.quote(script)]

    def session(self, cmd: list) -> list:
        """ssh running cmd under WRAPPER."""
        return self.ssh_cmd + [self.dest, " ".join(["sh", "-c", shlex.quote(WRAPPER), "rtier"] +
                                                   [shlex.quote(str(a)) for a in cmd])]

    def run(self, script: str, what: str, input=None, timeout=60) -> str:
        r = subprocess.run(self.argv(script), input=input, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"{what} failed ({r.returncode}): {r.stderr.strip()[-2000:]}")
        return r.stdout

    def rsync(self, src: str, dst_dir: str, logf, exclude=(), delete=False) -> None:
        """src (a file, or a directory's contents with a trailing /) into dst_dir on the host."""
        # --copy-links: an index may link its files from elsewhere (its page file, say).
        cmd = ["rsync", "-a", "--copy-links", "--partial", "-e", " ".join(shlex.quote(a) for a in self.ssh_cmd)]
        cmd += (["--delete"] if delete else []) + [f"--exclude={e}" for e in exclude]
        cmd += [src, f"{self.dest}:{dst_dir.rstrip('/')}/"]
        logf.write("$ " + " ".join(cmd) + "\n")
        logf.flush()
        r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise RuntimeError(f"rsync {src} to {self.dest} failed ({r.returncode}, see {logf.name})")


def hosts_of(cfg: dict) -> list:
    """The host of each node, n1 first."""
    h = cfg.get("Hosts") or {}
    entries = h.get("Nodes") or []
    if not isinstance(entries, list) or not entries:
        sys.exit("hosts.nodes: the host of each node, n1 first (a destination, or {ssh: DEST, addr: ADDRESS})")
    user, opts = h.get("User"), [str(o) for o in h.get("SSHOptions") or []]
    out, seen = [], {}
    for i, e in enumerate(entries):
        if isinstance(e, str):
            ssh, addr = e, e
        elif isinstance(e, dict) and e.get("ssh") and set(e) <= {"ssh", "addr"}:
            ssh, addr = e["ssh"], e.get("addr", e["ssh"])
        else:
            sys.exit(f"hosts.nodes[{i}]: {e!r}: want a destination, or {{ssh: DEST, addr: ADDRESS}}")
        key = (ssh, addr)
        if key not in seen:  # the nodes of one host share its Host
            seen[key] = Host(ssh, str(addr), user, opts)
        out.append(seen[key])
    return out


def root_addr(cfg: dict, hosts: list) -> str:
    """The address the nodes reach this machine at: hosts.root, or the source address of this
    machine's route to the first host it reaches other than over loopback (127.0.0.1 when every
    host is this machine)."""
    if (cfg.get("Hosts") or {}).get("Root"):
        return cfg["Hosts"]["Root"]
    for h in dict.fromkeys(hosts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((h.addr, 9))  # no packet: a UDP connect only picks the route
                addr = s.getsockname()[0]
        except OSError as e:
            sys.exit(f"hosts.root: not given, and no route to {h.addr} to find it ({e}): set it")
        if not addr.startswith("127."):
            return addr
    return "127.0.0.1"


def check_cluster_config(cfg: dict) -> None:
    res = cfg.get("NodeResources") or {}
    names = {flat: k for k, (flat, _) in rl.experiment_config.SCHEMA["cluster"][1]["resources"][1].items()}
    emulated = sorted(names.get(k, k) for k in res if k != "ReadIOPSPerNode")
    if emulated:
        sys.exit(f"cluster.resources ({', '.join(emulated)}) emulates a machine per node on one (run_local.py); "
                 "on a cluster each node has its hosts.nodes entry. read_iops_per_node, the data node's own "
                 "read budget, is the one kept")
    if (cfg.get("Emulation") or {}).get("Netns"):
        sys.exit("cluster.emulation emulates a network on one machine (run_local.py): not on a cluster")


# ------------------------------------------------------------------- 1. hosts

# The instruction set extensions among /proc/cpuinfo's x86 flags that a compiler emits code for
# without being asked (vectorization, bit manipulation): what -march=native makes a binary need.
# The other flags (hypervisor, constant_tsc, rdrand, ...) name nothing it executes unasked, and
# AMD's sse4a and fma4 nothing GCC emits that way. On ARM, every one of Features counts.
ISA = re.compile(r"^(sse(?!4a)|ssse|avx|fma(?!4)|f16c|bmi|adx|aes|pclmul|sha_ni|vaes|vpclmul|gfni|amx|"
                 r"popcnt|movbe|abm|cx16)")


def cpu_features(line: str) -> set:
    key, _, vals = line.partition(":")
    f = set(vals.split())
    return f if key.strip() == "Features" else {x for x in f if ISA.match(x)}


def cmake_cache(ebin: str) -> dict:
    out = {}
    try:
        for line in open(os.path.join(ebin, "CMakeCache.txt")):
            m = re.match(r"^([A-Za-z_]+):[A-Z]+=(.*)$", line.strip())
            if m:
                out[m.group(1)] = m.group(2)
    except OSError:
        pass
    return out


def cuda_arches(ebin: str, cache: dict):
    """The CUDA architectures the engine build has code for, as (number, kind) with kind real,
    virtual (PTX, compiled for newer GPUs at load time) or both; None when it covers them all
    or the build does not say."""
    v = cache.get("CMAKE_CUDA_ARCHITECTURES", "")
    if v == "native":
        v = ""
        for f in glob.glob(os.path.join(ebin, "CMakeFiles", "*", "CMakeCUDACompiler.cmake")):
            m = re.search(r'CMAKE_CUDA_ARCHITECTURES_NATIVE "([^"]*)"', open(f).read())
            if m:
                v = m.group(1)
    if not v or v.startswith("all"):
        return None
    out = []
    for a in v.split(";"):
        num, _, kind = a.partition("-")
        if num.isdigit():
            out.append((int(num), kind or "both"))
    return out or None


def runs_on(arches, cc: int) -> bool:
    """A cubin runs on its major version from its minor on; PTX on every GPU from its version on."""
    return any((k in ("real", "both") and n // 10 == cc // 10 and n <= cc) or (k in ("virtual", "both") and n <= cc)
               for n, k in arches)


def preflight(hosts: list, nodes: int, dirs: list, work: str, ebin: str, cfg: dict) -> dict:
    """Step 1: every host reachable, its directories made and writable, rsync there, the engine
    runnable on its CPU and GPUs, what an earlier run of this work directory left stopped, and
    this run's node directories empty of an earlier stats log. Returns each host's GPU table
    (nvidia-smi), for cluster.gpus."""
    cache = cmake_cache(ebin)
    native = cache.get("FUSION_NATIVE", "").upper() in ("ON", "TRUE", "1")
    mine = set()
    if native:
        line = next((l for l in open("/proc/cpuinfo") if l.startswith(("flags", "Features"))), "")
        mine = cpu_features(line)
    gpu = cfg.get("Backend") == "gpu"
    arches = cuda_arches(ebin, cache) if gpu else None
    # What an earlier run of this work directory left: its nodes' processes (their command lines
    # name their files there). Bracketed characters keep the pattern from matching the shell
    # that runs pkill with it.
    w = "".join(f"[{c}]" if c in ".*+?()|{}$" else ("\\" + c if c in "[]^\\" else c) for c in work)
    stale = f"{w}/agent-n[0-9]+[.]json|{w}/n[0-9]+/raw[.]pages"
    node_dirs = {}
    for i, h in enumerate(hosts[:nodes], 1):
        node_dirs.setdefault(h, []).append(os.path.join(work, f"n{i}"))

    def check(h: Host) -> list:
        # The directories copies go to are made by rsync, in their parents (on this machine, the
        # stages make them).
        made = list(dict.fromkeys([work] + node_dirs[h] + sorted({os.path.dirname(d) for d in dirs})))
        script = "\n".join([
            f"pkill -TERM -f -- {shlex.quote(stale)} && sleep 2; pkill -KILL -f -- {shlex.quote(stale)}",
            *(f"mkdir -p {shlex.quote(d)} 2>/dev/null && [ -w {shlex.quote(d)} ] || echo nodir={d}"
              for d in made),
            *(f"rm -f {shlex.quote(os.path.join(d, 'stats.jsonl'))}" for d in node_dirs[h]),
            "command -v rsync >/dev/null 2>&1 && echo rsync=yes",
            "grep -m1 -E '^(flags|Features)' /proc/cpuinfo | sed 's/^/cpu=/'",
            "command -v nvidia-smi >/dev/null 2>&1 && "
            "nvidia-smi --query-gpu=uuid,compute_cap --format=csv,noheader 2>/dev/null | sed 's/^/cc=/'",
            "true"])
        try:
            out = h.run(script, "ssh", timeout=120)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            return [f"{h.dest}: {e} (SSH must log in without a prompt: keys, and the host in known_hosts)"]
        kv = [l.split("=", 1) for l in out.splitlines() if "=" in l]
        problems = [f"{h.dest}: {v} is not writable" for k, v in kv if k == "nodir"]
        h.rsync_ok = any(k == "rsync" for k, _ in kv)
        if native:
            cpu = next((v for k, v in kv if k == "cpu"), "")
            lack = sorted(mine - cpu_features(cpu)) if cpu else []
            if lack:
                problems.append(f"{h.dest}: its CPU lacks {', '.join(lack)}, which {ebin} was compiled for "
                                "(FUSION_NATIVE, -march=native here): build the engine for every host's CPU, "
                                "cmake -DFUSION_NATIVE=OFF")
        if arches:
            for k, v in kv:
                if k == "cc" and "," in v:
                    uuid, cc = (x.strip() for x in v.split(",", 1))
                    if cc.replace(".", "").isdigit() and not runs_on(arches, int(cc.replace(".", ""))):
                        h.bad_gpus[uuid] = cc
        return problems

    problems = []
    distinct = list(dict.fromkeys(hosts[:nodes]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(distinct))) as ex:
        for p in ex.map(check, distinct):
            problems += p
    if problems:
        sys.exit("hosts:\n  " + "\n  ".join(problems))
    tables = {}
    if gpu:
        for h in distinct:
            tables[h] = rl.gpu_table(h.ssh_cmd + [h.dest])
            if not tables[h]:
                log(f"{h.dest}: nvidia-smi lists no GPU")
    return tables


def check_gpus(hosts: list, visible: dict, tables: dict, ebin: str) -> None:
    """Each node's GPU (or, without cluster.gpus, every GPU of its host) must be one the CUDA build
    has code for."""
    bad = []
    for i, h in enumerate(hosts, 1):
        b = h.bad_gpus
        if not b:
            continue
        uuids = [visible[i]] if i in visible else list(b)
        for u in uuids:
            if u in b:
                name = next((r["name"] for r in tables.get(h, []) if r["uuid"] == u), u)
                bad.append(f"n{i} on {h.dest}: {name} (compute capability {b[u]})")
    if bad:
        sys.exit(f"{ebin} has no CUDA code for: " + "; ".join(bad) +
                 ": build it for these GPUs too (-DCMAKE_CUDA_ARCHITECTURES)")


# ------------------------------------------------------------------- 3. copies

def sync(hosts: list, initial: int, files: list, index: str, dirs: list, out_dir: str) -> None:
    """Step 3: the copies, at the same paths, to every host in parallel. A directory a host already
    sees at that path -- this machine itself, under any name, or a file system it shares -- is
    left alone: a marker file put in it here shows there (Koala's syncRepo skips the local
    machine by its addresses). rsync copies what changed only, so a host keeps the index from one
    run to the next. The index's page file goes only where a node of the initial deployment runs
    (it bootstraps from it)."""
    bootstrap = set(hosts[:initial])
    distinct = list(dict.fromkeys(hosts))
    srcs = list(dict.fromkeys([os.path.dirname(f) for f in files] + [index] + dirs))
    marker = f".rtier-same-{os.urandom(8).hex()}"
    marked = []
    for d in srcs:
        try:  # a directory that takes no marker here (read-only) is copied
            open(os.path.join(d, marker), "w").close()
            marked.append(d)
        except OSError:
            pass
    try:
        def shared(h: Host) -> set:
            out = h.run("\n".join(f"[ -e {shlex.quote(os.path.join(d, marker))} ] && echo {shlex.quote(d)}"
                                   for d in srcs) + "\ntrue", f"{h.dest}: looking for this machine's directories")
            return set(out.splitlines())
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(distinct))) as ex:
            same = dict(zip(distinct, ex.map(shared, distinct)))
    finally:
        for d in marked:
            os.remove(os.path.join(d, marker))

    def copy(h: Host) -> None:
        with open(os.path.join(out_dir, f"sync-{h.ssh}.log"), "w") as logf:
            for f in files:
                if os.path.dirname(f) not in same[h]:
                    h.rsync(f, os.path.dirname(f), logf)
            if index not in same[h]:
                h.rsync(index + "/", index, logf, exclude=() if h in bootstrap else ("vectors_*",), delete=True)
            for d in dirs:
                if d not in same[h]:
                    h.rsync(d + "/", d, logf, exclude=("vectors_*",), delete=True)

    far = [h for h in distinct if not set(srcs) <= same[h]]
    for h in distinct:
        if h not in far:
            log(f"{h.dest} has this machine's files already: nothing copied")
    if not far:
        return
    missing = [h.dest for h in far if not h.rsync_ok]
    if missing:
        sys.exit(f"no rsync on {', '.join(missing)}: install it there")
    log(f"copying the binaries, the index and the partitions to {', '.join(h.dest for h in far)}")
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, len(far))) as ex:
        for f in [ex.submit(copy, h) for h in far]:
            f.result()
    log(f"copied in {time.monotonic() - started:.0f} s")


# ------------------------------------------------------------------- 4. processes

class Procs(rl.Procs):
    """run_local.py's processes, with remote ones: a remote process runs under ssh with the
    session's input open (WRAPPER), and closing it stops the process."""

    def start(self, name, cmd, prefix=(), wait_line=None, host=None):
        if host is None:
            return super().start(name, cmd, prefix=prefix, wait_line=wait_line)
        out = open(os.path.join(self.logdir, name + ".log"), "w")
        p = subprocess.Popen(host.session(list(prefix) + cmd), stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE if wait_line else out, stderr=out, text=True)
        self.procs.append((name, p))
        if not wait_line:
            return None
        while True:  # a login shell may print something first
            line = p.stdout.readline()
            if not line or line.startswith(wait_line):
                break
            out.write(line)
        if not line:
            p.wait()
            raise RuntimeError(f"{name} did not start on {host.dest} ({p.returncode}; see {out.name})")
        threading.Thread(target=lambda: shutil.copyfileobj(p.stdout, out), daemon=True).start()
        return line.strip()

    def stop(self):
        for name, p in reversed(self.procs):
            if p.poll() is None:
                if p.stdin is not None:
                    try:
                        p.stdin.close()
                    except OSError:
                        pass
                else:
                    p.send_signal(signal.SIGINT)
        for name, p in self.procs:
            try:
                p.wait(timeout=20)  # the remote side escalates to SIGKILL after 10 s
            except subprocess.TimeoutExpired:
                p.kill()


def connect(api_addr: str, procs: Procs, timeout: float = 30) -> Controller:
    deadline = time.monotonic() + timeout
    while True:
        procs.check()
        try:
            return Controller(api_addr, timeout=30)
        except OSError:
            if time.monotonic() > deadline:
                raise RuntimeError(f"the controller's API {api_addr} did not come up (see controller.log)")
            time.sleep(0.2)


def wait_registered(api: Controller, name: str, procs: Procs, timeout: float = 120) -> None:
    """Until the controller lists the node: the next one starts only then, so IDs follow the order."""
    deadline = time.monotonic() + timeout
    while not any(n["name"] == name for n in api.call("Status").get("nodes") or []):
        procs.check()
        if time.monotonic() > deadline:
            raise RuntimeError(f"{name} did not register within {timeout:.0f} s (see agent-{name}.log)")
        time.sleep(0.2)


def main(cfg_path: str, out_dir: str) -> None:
    ecfg, cfg, reconfigs = rl.load_config(cfg_path)
    check_cluster_config(cfg)
    hosts = hosts_of(cfg)
    nodes = cfg.get("Nodes", len(hosts))
    if nodes > len(hosts):
        sys.exit(f"cluster.nodes is {nodes}, hosts.nodes names {len(hosts)} hosts: one per node")
    if nodes < len(hosts):
        log(f"hosts.nodes names {len(hosts)} hosts, cluster.nodes is {nodes}: the first {nodes}")
    hosts = hosts[:nodes]
    initial = cfg.get("InitialNodes", 1)
    work, ebin, gbin = rl.paths_of(cfg)
    root = root_addr(cfg, hosts)
    index = os.path.abspath(cfg.get("IndexDir") or os.path.join(work, "index"))
    parts = os.path.join(work, "partitions")
    gens = [d for rc in reconfigs if rc.get("rebuild")
            for d in (rc["request"]["source"]["index_dir"], rc["request"]["source"]["parts_dir"])]
    for d in gens:
        if not os.path.isdir(d):
            sys.exit(f"reconfigurations: rebuild directory {d} does not exist")
    rl.record(ecfg, cfg, cfg_path, out_dir)

    # 1. hosts, before the stages: a build can take hours
    log(f"{nodes} nodes on {len(set(hosts))} hosts, the controller and the load here ({root})")
    tables = preflight(hosts, nodes, [work, index, parts, gbin, ebin] + gens, work, ebin, cfg)
    visible = rl.node_gpus(cfg, nodes, lambda i: tables.get(hosts[i - 1], []))
    check_gpus(hosts, visible, tables, ebin)

    # 2. index, ground truth, partitions; 3. copies
    index, gt, parts = rl.stages(ecfg, cfg)
    sync(hosts, initial, [os.path.join(gbin, "rtier-agent"), os.path.join(ebin, "rtier_node")],
         index, [parts] + gens, out_dir)
    node_args = rl.node_args_of(cfg, index, parts)

    ctl_addr = cfg.get("ControllerAddr", f"{root}:7100")
    api_addr = cfg.get("APIAddr", f"{root}:7101")
    procs = Procs(out_dir)
    results = []
    try:
        # 4. controller here, then the nodes in order
        ctl = rl.controller_config(cfg, ctl_addr, api_addr, index, parts, work, out_dir)
        # The files the controller makes for the nodes (a rebuild's segments) go out from here.
        ctl["bulk_listen"] = f"{root}:0"
        ctl_cfg = os.path.join(work, "controller.json")
        json.dump(ctl, open(ctl_cfg, "w"), indent=2)
        procs.start("controller", [f"{gbin}/rtier-controller", "-config", ctl_cfg])
        api = connect(api_addr, procs)
        for i, h in enumerate(hosts, 1):
            name, node_dir = f"n{i}", os.path.join(work, f"n{i}")
            # Listening everywhere, known by the host's address: the one the others reach.
            ready = procs.start(f"node-{name}", rl.node_command(cfg, ebin, index, parts, "0.0.0.0:0", node_dir, node_args),
                                prefix=rl.gpu_env(visible, i), wait_line="READY tcp=", host=h)
            agent = rl.agent_config(cfg, name, ctl_addr, "0.0.0.0", f"{h.addr}:{ready.split('=')[1]}", node_dir, parts)
            # The agent samples its node, itself and its machine for cgroups.jsonl here.
            agent.update(advertise_host=h.addr, sample_interval="1s")
            agent_cfg = os.path.join(work, f"agent-{name}.json")
            h.run(f"cat > {shlex.quote(agent_cfg)}", f"{h.dest}: writing {agent_cfg}", input=json.dumps(agent, indent=2))
            procs.start(f"agent-{name}", [f"{gbin}/rtier-agent", "-config", agent_cfg],
                        prefix=rl.gpu_env(visible, i), host=h)
            wait_registered(api, name, procs)
            log(f"{name} on {h.dest} ({h.addr}) registered")
        api.close()

        # 5. initial deployment, load, timed reconfigurations
        rl.drive(cfg, reconfigs, api_addr, gbin, gt, out_dir, procs, results, [])
    finally:
        # 7. clean up, then 6. results
        procs.stop()
        rl.finish(out_dir)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: run_cluster.py EXPERIMENT.yaml OUT_DIR")
    main(sys.argv[1], sys.argv[2])
