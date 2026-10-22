"""Experiment configurations: one JSON file per experiment (configs/experiment.*.json), read by
run_local.py and run_cluster.py.

As in Koala, the system has one configuration file, config.yaml at the repository root, which every
process reads at startup; an experiment is a JSON file, and the runner rewrites config.yaml from it
before the run. An experiment file is complete on its own: what it leaves out takes the runner's
defaults (and, for config.yaml's keys, the system's). Every key is checked against SCHEMA, so a
misspelled one stops the run; config.yaml's keys are allowed too, at the top level, and go there --
all but RUNNER_KEYS, which the runners set from the run itself.

    {
      "Comment": "what the experiment is for",
      "Type": "warmup",                      # optional: a warmup run (run_local.py)
      "WorkDir", "IndexDir", "SnapshotDir", "EngineBin", "GoBin",
      "Base", "Queries", "GroundTruth", "GroundTruthStride",
      "Assignment": FILE | "Partition": {"Generator": "gp", "Partitions": 64, "KaMinPar": PATH, ...},
      "BuildArgs": ["--pq-m", "64"],         # fusion_build's flags
      "Nodes", "InitialNodes", "Protocol", "Placement", "StageTimeout", ...,   # config.yaml's keys too
      "Backend", "GPUs", "NodeArgs", "PQCapacity", ...,
      "NodeResources": {...}, "Emulation": {...},
      "Hosts": {"Nodes": [HOST, {"SSH": DEST, "Addr": ADDRESS}, ...], ...},   # run_cluster.py only
      "Load": {"Rate", "RateSteps", "Users", "DurationSeconds", ...},
      "Calibrate": {"Rates", "SecondsPerRate"},
      "Updates": {"Headroom", "ListHeadroom", "MaxListLen"},
      "Reconfigurations": [{"TriggerTimeSeconds": 90, "DataNodes": 4}, {..., "Plan": P},
                           {..., "Rebuild": {"Index": DIR, "Partitions": DIR}}]
    }

Paths are relative to the repository root, as run_local.py runs from there, except a plan file and
a rebuild's directories, which are relative to the experiment file (load makes them absolute). A
results directory keeps the configuration as it ran (experiment.json, the form the analysis scripts
read) and the config.yaml the processes read.

    python3 scripts/experiment_config.py CONFIG.json     # checked, then printed
"""

import json
import os
import sys

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("config.yaml is YAML: install PyYAML (pip install --user pyyaml)")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEM_CONFIG = os.path.join(ROOT, "config.yaml")

# Key -> None (a value), a nested schema (a mapping; of a list's items for a list), or "opaque".
SCHEMA = {
    "Comment": None, "Type": None,
    "WorkDir": None, "IndexDir": None, "SnapshotDir": None, "EngineBin": None, "GoBin": None,
    "Base": None, "Queries": None, "GroundTruth": None, "GroundTruthStride": None,
    "Assignment": None,
    "Partition": {"Generator": None, "Partitions": None, "Points": None, "GraphSource": None,
                  "Weights": None, "KaMinPar": None, "Threads": None},
    "BuildArgs": None,
    "Nodes": None, "ReadyTimeoutSeconds": None, "PQCapacity": None, "PQHeadroom": None,
    "Backend": None, "GPUs": None, "NodeArgs": None,
    "NodeResources": {"CPUsPerNode": None, "FirstCPU": None, "MemoryMax": None, "IOReadIOPSMax": None,
                      "IOReadBandwidthMax": None, "ReadIOPSPerNode": None},
    "Emulation": {"Netns": None, "Delay": None, "Rate": None},
    "Hosts": {"Nodes": "opaque", "Root": None, "User": None, "SSHOptions": None},
    "Load": {"K": None, "NProbe": None, "N": None, "Arrivals": None, "Rate": None, "RateSteps": None,
             "Users": None, "UserSteps": None, "RetryForSeconds": None, "DurationSeconds": None,
             "WarmupSeconds": None, "DrainSeconds": None},
    "Calibrate": {"Rates": None, "SecondsPerRate": None},
    "Updates": {"Headroom": None, "ListHeadroom": None, "MaxListLen": None},
    "Reconfigurations": {
        "TriggerTimeSeconds": None, "DataNodes": None, "Plan": "opaque",
        "Rebuild": {"Index": None, "Partitions": None, "Spares": None, "Placement": None, "Seed": None,
                    "Warmup": None, "WarmupConcurrency": None, "StopTheWorld": None}},
}


# config.yaml's keys that the runners set from the run, and an experiment does not.
RUNNER_KEYS = {
    "PartitionsDir": "the partitions the stages make, WorkDir/partitions",
    "MetricsPath": "metrics.jsonl in the results directory",
    "StatsDir": "the results directory",
    "ControllerWorkDir": "WorkDir/controller",
    "ControllerBulkAddr": "run_cluster.py's, on this machine",
    "AllowUpdates": 'on with "Updates"',
}


def system_keys() -> set:
    """config.yaml's keys: an experiment may set any of them, by name."""
    try:
        with open(SYSTEM_CONFIG) as f:
            return set(yaml.safe_load(f) or {})
    except OSError as e:
        sys.exit(f"{SYSTEM_CONFIG}: {e} (bin/rtier-controller -defaults writes one)")


def merge(base, over):
    """over on top of base: mappings merge, anything else replaces, None removes."""
    if not isinstance(base, dict) or not isinstance(over, dict):
        return over
    out = dict(base)
    for k, v in over.items():
        if v is None:
            out.pop(k, None)
        elif k in out:
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def check(cfg, schema=None, where=""):
    """Every key must be in the schema (at the top level, or be one of config.yaml's)."""
    top = schema is None
    if top:
        schema = SCHEMA
    if not isinstance(cfg, dict):
        raise ValueError("%s: want a mapping, got %r" % (where or "configuration", cfg))
    system = system_keys() if top else set()
    for k, v in cfg.items():
        if top and k in RUNNER_KEYS:
            raise ValueError("%s: the runner sets it (%s)" % (k, RUNNER_KEYS[k]))
        if k not in schema:
            if k in system:
                continue
            known = sorted((set(schema) | system) - set(RUNNER_KEYS))
            raise ValueError("%s%s: unknown key (known: %s)" % (where, k, ", ".join(known)))
        sub = schema[k]
        if isinstance(sub, dict) and v is not None:
            if isinstance(v, list):
                for i, item in enumerate(v):
                    check(item, sub, "%s%s[%d]." % (where, k, i))
            else:
                check(v, sub, "%s%s." % (where, k))


def _absolute(cfg, d):
    """A plan file and a rebuild's directories are relative to the file that names them."""
    for rc in cfg.get("Reconfigurations") or []:
        if isinstance(rc, dict):
            if isinstance(rc.get("Plan"), str):
                rc["Plan"] = os.path.join(d, rc["Plan"])
            rb = rc.get("Rebuild")
            if isinstance(rb, dict):
                for k in ("Index", "Partitions"):
                    if isinstance(rb.get(k), str):
                        rb[k] = os.path.join(d, rb[k])


def load(path):
    """The experiment at path, checked, its relative plan and rebuild paths made absolute."""
    with open(path) as f:
        try:
            cfg = json.load(f)
            check(cfg)
        except ValueError as e:
            raise ValueError("%s: %s" % (path, e)) from None
    _absolute(cfg, os.path.dirname(os.path.abspath(path)))
    return cfg


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: experiment_config.py CONFIG.json")
    try:
        print(json.dumps(load(sys.argv[1]), indent=2))
    except (OSError, ValueError) as e:
        sys.exit(str(e))
