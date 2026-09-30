#!/usr/bin/env python3
"""Run one rtier experiment on a single machine.

Adapted from Koala's scripts/runExperiment.py (config JSON -> deploy -> load -> timed
reconfigurations -> collect results -> clean up), without Kafka and SSH:

  1. build the index (fusion_build) and ground truth (fusion_gt) unless given
  2. split the posting lists into partitions (rtier_segment) with the given assignment
  3. start the controller, then one rtier_node + rtier-agent per node
  4. wait for the initial deployment, start the open-loop load generator
  5. trigger the configured reconfigurations (rtier_ctl.py -> Rescale)
  6. collect metrics.jsonl (+ SQLite in Koala's layout), loadgen CSV, logs, timeline.svg; each
     node's stats log (stats-<node>.jsonl) and, with "NodeResources", cgroups.jsonl, both on
     the monotonic clock of clock.json
  7. stop everything

Two things a capacity-driven scale-out run needs:

  "Calibrate": {"Rates": [50, 100, 200, 400], "SecondsPerRate": 15} runs the load generator
      once per rate and reports where the cluster stops keeping up. That number sets the rate
      of the real run -- adding a node shows nothing under a load one node already serves.
      A config with "Calibrate" measures capacity and stops: no reconfiguration in that run.
  "Load": {"Rate": 200, "RateSteps": "30:600"} raises the offered load during the run (here to
      600 q/s at 30 s), which is the situation a scale-out answers.

"Load": {"Users": 8, "UserSteps": "30:24,70:8"} replaces the arrival rate with closed-loop
users: 8 concurrent users, 24 from 30 s, 8 again from 70 s, each sending its next query when
the last one returned.

"Load": {"WarmupSeconds": 15} marks the first 15 s of load as warm-up: scripts/compare_runs.py
leaves those queries out of its figures and numbers.

"Load": {"DrainSeconds": 300} (the default) is how long a run waits after the load ended for
the answers still due: an open-loop run that ends with a backlog drains it without new arrivals.

"Load": {"RetryForSeconds": 30} makes the clients wait for their answers: a query refused as
unavailable (an entry that left, admission paused by stop-and-copy) is retried for up to 30 s
after its scheduled time, and the wait counts in its latency. Without it a refused query is
retried three times and then fails, so it is missing from the mean latency; comparing
protocols by their mean latency needs it (scripts/compare_runs.py).

    python3 scripts/run_local.py configs/experiment.example.json results/run1

Open design questions this script depends on (see `bin/rtier-client design`):
  U2  the list -> partition assignment is an input ("Assignment"); for plumbing tests only,
      scripts/testing/make_range_assignment.py writes a placeholder.
  U5  queries need a strategy ("Strategy"); both production strategies are placeholders, so
      queries fail with "undecided" until one is implemented.
  U11 "Arrivals": "semdn" is a placeholder; "poisson" works.

With "Emulation": {"Netns": true, "Delay": "100us", "Rate": "10gbit"} (run as root), every
node runs in its own network namespace (scripts/emulation/netns.sh) and the controller
listens on the bridge address 10.10.0.1. Delay and rate values are per-experiment choices.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtier_ctl import Controller  # noqa: E402
from plot_run import read_latency, summarize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, **kw):
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def rate_steps(v) -> str:
    """"RateSteps" as the load generator's -rate-steps: "30:400,90:800", a list of "30:400"
    strings, or a list of [30, 400] pairs."""
    if not v:
        return ""
    if isinstance(v, str):
        return v
    return ",".join(s if isinstance(s, str) else f"{s[0]}:{s[1]}" for s in v)


class Procs:
    def __init__(self, logdir: str):
        self.logdir = logdir
        self.procs = []

    def start(self, name, cmd, prefix=(), wait_line=None):
        out = open(os.path.join(self.logdir, name + ".log"), "w")
        p = subprocess.Popen(list(prefix) + cmd, stdout=subprocess.PIPE if wait_line else out,
                             stderr=out, text=True)
        self.procs.append((name, p))
        if wait_line:
            line = p.stdout.readline()
            if not line.startswith(wait_line):
                raise RuntimeError(f"{name} did not start: {line!r} (see {out.name})")
            threading.Thread(target=lambda: shutil.copyfileobj(p.stdout, out), daemon=True).start()
            return line.strip()
        return None

    def check(self):
        for name, p in self.procs:
            if p.poll() is not None:
                raise RuntimeError(f"{name} exited with {p.returncode}")

    def stop(self):
        for name, p in reversed(self.procs):
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for name, p in self.procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


def isolation(cfg: dict, i: int, work: str, nodes: int) -> list:
    """The systemd-run prefix that gives node i (1-based) its own share of the machine.

    On one machine every node reads the same SSD and runs on the same cores, so adding a node
    adds no capacity: the cluster stops at what the disk serves (the two-phase RERANK reads
    ~150 pages per query). "NodeResources" emulates a node per machine instead -- each node,
    with its agent, gets its own CPUs and its own read IOPS on the disk of WorkDir (cgroup v2
    cpuset and io controllers), so the total stays below what the disk serves:

      "NodeResources": {"CPUsPerNode": 4, "IOReadIOPSMax": 40000, "MemoryMax": "2G"}

    Node i gets CPUs [FirstCPU + (i-1)*CPUsPerNode, ...); the controller and the load
    generator get the CPUs after the last node's (see rest_cpus)."""
    res = cfg.get("NodeResources")
    if not res:
        return []
    if not shutil.which("systemd-run"):
        raise RuntimeError('"NodeResources" needs systemd-run (cgroup v2 cpuset and io controllers)')
    props = []
    if res.get("CPUsPerNode"):
        c, first = res["CPUsPerNode"], res.get("FirstCPU", 0)
        if first + nodes * c > (os.cpu_count() or 0):
            raise RuntimeError(f"{nodes} nodes x {c} CPUs from CPU {first} do not fit in {os.cpu_count()} CPUs")
        lo = first + (i - 1) * c
        props += ["-p", f"AllowedCPUs={lo}-{lo + c - 1}"]
    for key in ("IOReadIOPSMax", "IOReadBandwidthMax"):
        if res.get(key):
            dev = subprocess.run(["df", "--output=source", work], capture_output=True, text=True,
                                 check=True).stdout.split()[-1]
            props += ["-p", f"{key}={dev} {res[key]}"]
    if res.get("MemoryMax"):
        props += ["-p", f"MemoryMax={res['MemoryMax']}"]
    return ["systemd-run", "--quiet", "--scope", "--collect"] + props


def data_disk(path: str) -> str:
    """MAJ:MIN of the disk that holds path, as the cgroup io.stat names it (a partition's
    I/O is accounted to its disk)."""
    dev = os.stat(path).st_dev
    mm = f"{os.major(dev)}:{os.minor(dev)}"
    if os.path.exists(f"/sys/dev/block/{mm}/partition"):
        mm = open(f"/sys/dev/block/{mm}/../dev").read().strip()
    return mm


class CgroupSampler:
    """With "NodeResources", every node and every agent runs in its own cgroup ("n1",
    "n1-agent", ...): samples each one's CPU time, I/O on the data disk, dirty and writeback page
    cache and pressure stall times once a second into cgroups.jsonl, stamped with
    time.monotonic() (clock.json has the load's start on that clock). Counters are cumulative;
    take differences between samples."""

    def __init__(self, path: str, disk: str):
        self.out = open(path, "w")
        self.disk = disk
        self.nodes = []  # (name, pid)
        self.stop_ = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def add(self, name: str, pid: int) -> None:
        # The cgroup is looked up at every sample: right after Popen, systemd-run may not have
        # moved the process into its scope yet (it would read as run_local's own cgroup).
        self.nodes.append((name, pid))

    @staticmethod
    def cgroup(pid: int):
        try:
            for line in open(f"/proc/{pid}/cgroup"):
                if line.startswith("0::"):
                    return "/sys/fs/cgroup" + line[3:].strip()
        except OSError:
            pass
        return None

    @staticmethod
    def keyed(path: str) -> dict:
        try:
            fields = [l.split() for l in open(path)]
        except OSError:
            return {}
        return {f[0]: int(f[1]) for f in fields if len(f) == 2 and f[1].isdigit()}

    def sample(self, name: str, cg: str) -> dict:
        r = {"t": round(time.monotonic(), 3), "node": name}
        r["cpu_usec"] = self.keyed(f"{cg}/cpu.stat").get("usage_usec", 0)
        io = {}
        try:
            for l in open(f"{cg}/io.stat"):
                f = l.split()
                if f and f[0] == self.disk:
                    io = {k: int(v) for k, v in (x.split("=") for x in f[1:])}
        except OSError:
            pass
        r["io"] = {k: io.get(k, 0) for k in ("rbytes", "wbytes", "rios", "wios")}
        mem = self.keyed(f"{cg}/memory.stat")
        r["mem"] = {k: mem.get(k, 0) for k in ("anon", "file", "file_dirty", "file_writeback")}
        psi = {}
        for res in ("cpu", "io", "memory"):
            try:
                for l in open(f"{cg}/{res}.pressure"):
                    kind, *kv = l.split()
                    psi[f"{res}_{kind}_us"] = int(dict(x.split("=") for x in kv)["total"])
            except (OSError, KeyError):
                pass
        r["psi"] = psi
        return r

    def loop(self) -> None:
        while not self.stop_.wait(1.0 - time.monotonic() % 1.0):
            mine = self.cgroup(os.getpid())
            for name, pid in self.nodes:
                cg = self.cgroup(pid)
                if cg and cg != mine:
                    self.out.write(json.dumps(self.sample(name, cg)) + "\n")
            self.out.flush()

    def start(self) -> None:
        if self.nodes:
            self.thread.start()

    def stop(self) -> None:
        self.stop_.set()
        if self.thread.is_alive():
            self.thread.join()
        self.out.close()


def rest_cpus(cfg: dict, nodes: int) -> list:
    """The prefix that keeps the controller and the load generator off the nodes' CPUs."""
    res = cfg.get("NodeResources") or {}
    if not res.get("CPUsPerNode"):
        return []
    lo = res.get("FirstCPU", 0) + nodes * res["CPUsPerNode"]
    hi = (os.cpu_count() or 1) - 1
    if lo > hi:
        return []
    return ["systemd-run", "--quiet", "--scope", "--collect", "-p", f"AllowedCPUs={lo}-{hi}"]


def check_fresh(gbin: str) -> None:
    """Refuses to run Go binaries older than the Go sources: `go test` (and so `make test`
    before it built bin/ too) compiles the code under test without touching bin/, and an
    experiment on a stale agent measures code that is no longer there."""
    built = min((os.path.getmtime(os.path.join(gbin, f)) for f in os.listdir(gbin)), default=0) \
        if os.path.isdir(gbin) else 0
    for top in ("cmd", "internal"):
        for d, _, files in os.walk(os.path.join(ROOT, top)):
            for f in files:
                if f.endswith(".go") and not f.endswith("_test.go") and os.path.getmtime(os.path.join(d, f)) > built:
                    sys.exit(f"{os.path.join(d, f)} is newer than the binaries in {gbin}: run `make` first")


def main(cfg_path: str, out_dir: str) -> None:
    cfg = json.load(open(cfg_path))
    work = os.path.abspath(cfg.get("WorkDir", "/tmp/rtier-run"))
    ebin = os.path.join(ROOT, cfg.get("EngineBin", "engine/build"))
    gbin = os.path.join(ROOT, cfg.get("GoBin", "bin"))
    check_fresh(gbin)
    os.makedirs(work, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(cfg_path, os.path.join(out_dir, "experiment.json"))

    # 1. index and ground truth
    index = cfg.get("IndexDir") or os.path.join(work, "index")
    if not os.path.exists(os.path.join(index, "meta.txt")):
        run([f"{ebin}/fusion_build", "--base", cfg["Base"], "--out", index] + cfg.get("BuildArgs", []))
    gt = cfg.get("GroundTruth", "")
    if gt and not os.path.exists(gt):
        run([f"{ebin}/fusion_gt", "--base", cfg["Base"], "--queries", cfg["Queries"], "--out", gt, "--k", "100"])

    # 2. partitions (the assignment is an input: U2)
    parts = os.path.join(work, "partitions")
    if not os.path.exists(os.path.join(parts, "manifest.txt")):
        run([f"{ebin}/rtier_segment", "--index", index, "--assign", cfg["Assignment"], "--out", parts])

    emu = cfg.get("Emulation", {})
    netns = bool(emu.get("Netns"))
    nodes = cfg.get("Nodes", 1)
    ctl_host = "10.10.0.1" if netns else "127.0.0.1"  # bridge address in netns mode
    ctl_addr = cfg.get("ControllerAddr", f"{ctl_host}:7100")
    api_addr = cfg.get("APIAddr", f"{ctl_host}:7101")
    netns_sh = os.path.join(ROOT, "scripts/emulation/netns.sh")

    procs = Procs(out_dir)
    results = []
    sampler = CgroupSampler(os.path.join(out_dir, "cgroups.jsonl"), data_disk(work)) \
        if cfg.get("NodeResources") else None
    try:
        if netns:
            env = dict(os.environ, DELAY=emu.get("Delay", "100us"), RATE=emu.get("Rate", "10gbit"))
            subprocess.run([netns_sh, "down", str(nodes)], env=env)
            run([netns_sh, "up", str(nodes)], env=env)

        # 3. controller, then nodes (spare nodes register idle and join at scale-out)
        ctl_cfg = os.path.join(work, "controller.json")
        json.dump({
            "listen": ctl_addr,
            "api_listen": api_addr,
            "index_dir": index, "partitions_dir": parts,
            "initial_nodes": cfg.get("InitialNodes", 1),
            "protocol": cfg.get("Protocol", "lazy"),
            "raw_priority": cfg.get("RawPriority", "background"),
            "placement": cfg.get("Placement", "even"),
            "metrics_path": os.path.join(out_dir, "metrics.jsonl"),
        }, open(ctl_cfg, "w"), indent=2)
        procs.start("controller", [f"{gbin}/rtier-controller", "-config", ctl_cfg], prefix=rest_cpus(cfg, nodes))
        time.sleep(0.5)

        for i in range(1, nodes + 1):
            name = f"n{i}"
            prefix = isolation(cfg, i, work, nodes) + (["ip", "netns", "exec", f"rtier-{name}"] if netns else [])
            host = f"10.10.0.{10 + i}" if netns else "127.0.0.1"
            # Each node keeps its raw vectors in its own sparse file (a subset of the index's
            # page file); the ones it gets by migration arrive after the flip (U9).
            os.makedirs(os.path.join(work, name), exist_ok=True)
            ready = procs.start(f"node-{name}", [f"{ebin}/rtier_node", "--index", index, "--partitions", parts,
                                                 "--listen", f"{host}:0", "--backend", cfg.get("Backend", "cpu"),
                                                 "--raw-file", os.path.join(work, name, "raw.pages"),
                                                 "--stats-file", os.path.join(out_dir, f"stats-{name}.jsonl")]
                                + cfg.get("NodeArgs", []), prefix=prefix, wait_line="READY tcp=")
            if sampler:
                sampler.add(name, procs.procs[-1][1].pid)
            node_addr = f"{host}:{ready.split('=')[1]}"
            agent_cfg = os.path.join(work, f"agent-{name}.json")
            json.dump({
                "name": name, "controller": ctl_addr,
                "query_listen": f"{host}:0", "bulk_listen": f"{host}:0", "node_addr": node_addr,
                "work_dir": os.path.join(work, name), "partitions_dir": parts,
                "strategy": cfg.get("Strategy", ""),
                "transfer_rate_bytes_per_sec": cfg.get("TransferRateBytesPerSec", 0),
                "metrics_interval": cfg.get("MetricsInterval", "1s"),
            }, open(agent_cfg, "w"), indent=2)
            procs.start(f"agent-{name}", [f"{gbin}/rtier-agent", "-config", agent_cfg], prefix=prefix)
            if sampler:
                sampler.add(f"{name}-agent", procs.procs[-1][1].pid)

        if sampler:
            sampler.start()

        # 4. initial deployment, then load
        api = Controller(api_addr)
        # Bootstrapping copies every partition's PQ codes and raw vectors from the index: minutes
        # on a 10M-vector index ("ReadyTimeoutSeconds").
        deploy_start = time.monotonic()
        st = api.call("WaitReady", {"timeout_seconds": cfg.get("ReadyTimeoutSeconds", 600)})
        log(f"epoch {st['table']['epoch']} ready on {len(st['table']['placement']['owners'])} partitions "
            f"after {time.monotonic() - deploy_start:.0f} s")
        load = cfg.get("Load", {})
        dur = load.get("DurationSeconds", 60)

        def loadgen(rate, seconds, out_csv, steps=""):
            cmd = [f"{gbin}/rtier-loadgen", "-api", api_addr, "-queries", cfg["Queries"],
                   "-rate", str(rate), "-duration", f"{seconds}s", "-k", str(load.get("K", 10)),
                   "-nprobe", str(load.get("NProbe", 64)), "-n", str(load.get("N", 200)),
                   "-arrivals", load.get("Arrivals", "poisson"), "-out", out_csv]
            if steps:
                cmd += ["-rate-steps", steps]
            if load.get("Users"):  # closed loop: concurrent users instead of an arrival rate
                cmd += ["-users", str(load["Users"])]
                if load.get("UserSteps"):
                    cmd += ["-users-steps", rate_steps(load["UserSteps"])]
            if load.get("RetryForSeconds"):
                cmd += ["-retry-for", f"{load['RetryForSeconds']}s"]
            if gt:
                cmd += ["-gt", gt]
            return cmd

        # 4b. capacity first: how much this cluster serves before it falls behind. Scale-out is
        # driven by load, so every later run's rate is set relative to this number.
        calib = cfg.get("Calibrate")
        if calib:
            table = []
            for rate in calib.get("Rates", [50, 100, 200, 400]):
                path = os.path.join(out_dir, f"calib-{rate:g}.csv")
                with open(os.path.join(out_dir, "calibration.log"), "a") as logf:
                    run(rest_cpus(cfg, nodes) + loadgen(rate, calib.get("SecondsPerRate", 15), path),
                        stdout=logf, stderr=logf)
                s = summarize(read_latency(path))
                s["offered_per_s"] = rate
                table.append(s)
                log(f"offered {rate:g}/s: answered {s['answered_per_s']}/s, mean {s['mean_us']} us, "
                    f"p50 {s['p50_us']} us, p99 {s['p99_us']} us, {s['status']}")
            json.dump(table, open(os.path.join(out_dir, "calibration.json"), "w"), indent=2)
            kept = [s["offered_per_s"] for s in table
                    if s["answered_per_s"] >= 0.95 * s["offered_per_s"] and s["status"].get("ok") == s["queries"]]
            log(f"saturation: served up to {max(kept) if kept else 0}/s without falling behind; "
                f"set the reconfiguration run's rate from this")
            return  # a calibration run measures capacity only: no reconfiguration in it

        procs.start("loadgen", loadgen(load.get("Rate", 100), dur,
                                       os.path.join(out_dir, "latency.csv"),
                                       rate_steps(load.get("RateSteps"))), prefix=rest_cpus(cfg, nodes))

        # 5. timed reconfigurations (Koala: "Reconfigurations" with TriggerTimeSeconds)
        # Timed with the monotonic clock: a wall clock can jump (WSL2 resyncs it by tens of
        # seconds), and the load generator's times are monotonic too.
        t0 = time.monotonic()
        json.dump({"load_start": round(t0, 3)}, open(os.path.join(out_dir, "clock.json"), "w"))
        for rc in sorted(cfg.get("Reconfigurations", []), key=lambda r: r["TriggerTimeSeconds"]):
            while time.monotonic() - t0 < rc["TriggerTimeSeconds"]:
                procs.check()
                time.sleep(0.2)
            log(f"rescale to {rc['DataNodes']} data nodes")
            started = time.monotonic()
            reply = api.call("Rescale", {"data_nodes": rc["DataNodes"]})
            results.append({"data_nodes": rc["DataNodes"], "trigger_seconds": round(started - t0, 3),
                            "elapsed_seconds": round(time.monotonic() - started, 3), "reply": reply})
            log(json.dumps(results[-1]))
        json.dump(results, open(os.path.join(out_dir, "reconfigurations.json"), "w"), indent=2)

        # The load generator stops sending at the duration and exits once every query it sent is
        # answered, writing latency.csv only then: an open-loop run that ends with a backlog
        # needs the time to drain it, or its queries are lost ("DrainSeconds").
        drain = load.get("DrainSeconds", 300)
        while time.monotonic() - t0 < dur + drain and procs.procs[-1][1].poll() is None:
            time.sleep(0.5)
        if procs.procs[-1][1].poll() is None:
            log(f"load generator still waiting for answers {drain} s after the load ended: stopped, "
                "no latency.csv")
    finally:
        # 7. clean up, then 6. results
        procs.stop()
        if sampler:
            sampler.stop()
        if netns:
            subprocess.run([netns_sh, "down", str(nodes)])
        m = os.path.join(out_dir, "metrics.jsonl")
        if os.path.exists(m):
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts/metrics_to_sqlite.py"), m,
                            os.path.join(out_dir, "metricCollector.db")])
        if os.path.exists(os.path.join(out_dir, "latency.csv")):
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts/plot_run.py"), out_dir])
        log(f"results in {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: run_local.py EXPERIMENT.json OUT_DIR")
    main(sys.argv[1], sys.argv[2])
