#!/usr/bin/env python3
"""Run one rtier experiment on a single machine.

Adapted from Koala's scripts/runExperiment.py (config JSON -> deploy -> load -> timed
reconfigurations -> collect results -> clean up), without Kafka and SSH:

  1. build the index (fusion_build) and ground truth (fusion_gt) unless given
  2. split the posting lists into partitions (rtier_segment) with the given assignment
  3. start the controller, then one rtier_node + rtier-agent per node
  4. wait for the initial deployment, start the open-loop load generator
  5. trigger the configured reconfigurations (rtier_ctl.py -> Rescale)
  6. collect metrics.jsonl (+ SQLite in Koala's layout), loadgen CSV, logs, timeline.svg
  7. stop everything

Two things a capacity-driven scale-out run needs:

  "Calibrate": {"Rates": [50, 100, 200, 400], "SecondsPerRate": 15} runs the load generator
      once per rate and reports where the cluster stops keeping up. That number sets the rate
      of the real run -- adding a node shows nothing under a load one node already serves.
      A config with "Calibrate" measures capacity and stops: no reconfiguration in that run.
  "Load": {"Rate": 200, "RateSteps": "30:600"} raises the offered load during the run (here to
      600 q/s at 30 s), which is the situation a scale-out answers.

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


def main(cfg_path: str, out_dir: str) -> None:
    cfg = json.load(open(cfg_path))
    work = os.path.abspath(cfg.get("WorkDir", "/tmp/rtier-run"))
    ebin = os.path.join(ROOT, cfg.get("EngineBin", "engine/build"))
    gbin = os.path.join(ROOT, cfg.get("GoBin", "bin"))
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
        procs.start("controller", [f"{gbin}/rtier-controller", "-config", ctl_cfg])
        time.sleep(0.5)

        for i in range(1, nodes + 1):
            name = f"n{i}"
            prefix = ["ip", "netns", "exec", f"rtier-{name}"] if netns else []
            host = f"10.10.0.{10 + i}" if netns else "127.0.0.1"
            # Each node keeps its raw vectors in its own sparse file (a subset of the index's
            # page file); the ones it gets by migration arrive after the flip (U9).
            os.makedirs(os.path.join(work, name), exist_ok=True)
            ready = procs.start(f"node-{name}", [f"{ebin}/rtier_node", "--index", index, "--partitions", parts,
                                                 "--listen", f"{host}:0", "--backend", cfg.get("Backend", "cpu"),
                                                 "--raw-file", os.path.join(work, name, "raw.pages")]
                                + cfg.get("NodeArgs", []), prefix=prefix, wait_line="READY tcp=")
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

        # 4. initial deployment, then load
        api = Controller(api_addr)
        st = api.call("WaitReady", {"timeout_seconds": 600})
        log(f"epoch {st['table']['epoch']} ready on {len(st['table']['placement']['owners'])} partitions")
        load = cfg.get("Load", {})
        dur = load.get("DurationSeconds", 60)

        def loadgen(rate, seconds, out_csv, steps=""):
            cmd = [f"{gbin}/rtier-loadgen", "-api", api_addr, "-queries", cfg["Queries"],
                   "-rate", str(rate), "-duration", f"{seconds}s", "-k", str(load.get("K", 10)),
                   "-nprobe", str(load.get("NProbe", 64)), "-n", str(load.get("N", 200)),
                   "-arrivals", load.get("Arrivals", "poisson"), "-out", out_csv]
            if steps:
                cmd += ["-rate-steps", steps]
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
                    run(loadgen(rate, calib.get("SecondsPerRate", 15), path), stdout=logf, stderr=logf)
                s = summarize(read_latency(path))
                s["offered_per_s"] = rate
                table.append(s)
                log(f"offered {rate:g}/s: answered {s['answered_per_s']}/s, p50 {s['p50_us']} us, "
                    f"p99 {s['p99_us']} us, {s['status']}")
            json.dump(table, open(os.path.join(out_dir, "calibration.json"), "w"), indent=2)
            kept = [s["offered_per_s"] for s in table
                    if s["answered_per_s"] >= 0.95 * s["offered_per_s"] and s["status"].get("ok") == s["queries"]]
            log(f"saturation: served up to {max(kept) if kept else 0}/s without falling behind; "
                f"set the reconfiguration run's rate from this")
            return  # a calibration run measures capacity only: no reconfiguration in it

        procs.start("loadgen", loadgen(load.get("Rate", 100), dur,
                                       os.path.join(out_dir, "latency.csv"),
                                       rate_steps(load.get("RateSteps"))))

        # 5. timed reconfigurations (Koala: "Reconfigurations" with TriggerTimeSeconds)
        t0 = time.time()
        results = []
        for rc in sorted(cfg.get("Reconfigurations", []), key=lambda r: r["TriggerTimeSeconds"]):
            while time.time() - t0 < rc["TriggerTimeSeconds"]:
                procs.check()
                time.sleep(0.2)
            log(f"rescale to {rc['DataNodes']} data nodes")
            started = time.time()
            reply = api.call("Rescale", {"data_nodes": rc["DataNodes"]})
            results.append({"data_nodes": rc["DataNodes"], "trigger_seconds": round(started - t0, 3),
                            "elapsed_seconds": round(time.time() - started, 3), "reply": reply})
            log(json.dumps(results[-1]))
        json.dump(results, open(os.path.join(out_dir, "reconfigurations.json"), "w"), indent=2)

        while time.time() - t0 < dur + 5 and procs.procs[-1][1].poll() is None:
            time.sleep(0.5)
    finally:
        # 7. clean up, then 6. results
        procs.stop()
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
