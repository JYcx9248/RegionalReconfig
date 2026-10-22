"""Experiment configuration: YAML files of a base and overrides, read by run_local.py and
run_cluster.py.

An experiment file names the file it builds on with `base:` (a path relative to it) and gives
only what differs; the base may have a base of its own. Files merge key by key: a mapping merges
with the base's, anything else (a number, a string, a list) replaces it, and `null` removes the
base's value. Every key is checked against SCHEMA, so a misspelled one stops the run.

Sections, in the order of the pipeline:

    base: base/wiki-c1024.yaml
    type: warmup                                        # optional: a warmup run (run_local.py)
    paths:      {work_dir, engine_bin, go_bin, snapshot_dir}
    dataset:    {vectors, queries, ground_truth, ground_truth_stride}   # fusion_gt makes it if missing
    build:      {index_dir, pq_m, threads, ...}         # fusion_build's flags, BUILD_FLAGS; extra_args
    partition:  {generator: file | gp | spatial | range, partitions, file, gp: {...}}
    cluster:    {nodes, initial_nodes, protocol, placement, strategy, backend, gpus, node_args,
                 resources: {...}, raw: {...}, emulation: {...}, ...}
    hosts:      {nodes: [HOST, {ssh, addr}, ...], root, user, ssh_options}   # run_cluster.py only
    load:      {k, nprobe, n, arrivals, rate, rate_steps, users, user_steps, duration_seconds, ...}
    calibrate:  {rates, seconds_per_rate}
    updates:    {headroom, list_headroom, max_list_len}
    reconfigurations: [{at: 90, data_nodes: 4}, {at: 90, plan: P}, {at: 90, rebuild: {...}}]

Paths are relative to the repository root, as run_local.py runs from there, except a plan file
and a rebuild's directories, which are relative to the experiment file.

flatten() turns a configuration into the flat form run_local.py's steps and the analysis scripts
read (experiment.json in a results directory: "WorkDir", "Load": {"Rate"}, ...).

    python3 scripts/experiment_config.py CONFIG.yaml     # the merged configuration, then flat
"""

import json
import os
import sys

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("experiment configurations are YAML: install PyYAML (pip install --user pyyaml)")

# fusion_build's flags, as build: keys (a flag without a value is a boolean).
BUILD_FLAGS = {
    "max_n": "--max-n", "dtype": "--dtype", "centroid_ratio": "--centroid-ratio", "branch": "--branch",
    "balance": "--balance", "hnsw_m": "--hnsw-m", "hnsw_efc": "--hnsw-efc", "replicas": "--replicas",
    "closure_eps": "--closure-eps", "rng": "--rng", "rng_factor": "--rng-factor",
    "assign_candidates": "--assign-candidates", "pq_m": "--pq-m", "pq_train": "--pq-train",
    "layout": "--layout", "threads": "--threads", "seed": "--seed",
}
BOOL_FLAGS = {"rng"}

# YAML key -> (flat key, nested schema or None). A nested schema of a list applies to its items.
SCHEMA = {
    "base": (None, None),
    "type": ("Type", None),
    "paths": (None, {
        "work_dir": ("WorkDir", None), "engine_bin": ("EngineBin", None), "go_bin": ("GoBin", None),
        "snapshot_dir": ("SnapshotDir", None)}),
    "dataset": (None, {
        "vectors": ("Base", None), "queries": ("Queries", None), "ground_truth": ("GroundTruth", None),
        "ground_truth_stride": ("GroundTruthStride", None)}),
    "build": (None, dict({"index_dir": ("IndexDir", None), "extra_args": (None, None)},
                         **{k: (None, None) for k in BUILD_FLAGS})),
    "partition": (None, {
        "generator": (None, None), "file": ("Assignment", None), "partitions": (None, None),
        "gp": (None, {"points": (None, None), "graph_source": (None, None), "weights": (None, None),
                      "kaminpar": (None, None), "threads": (None, None)})}),
    "cluster": (None, {
        "nodes": ("Nodes", None), "initial_nodes": ("InitialNodes", None), "protocol": ("Protocol", None),
        "placement": ("Placement", None), "strategy": ("Strategy", None), "raw_priority": ("RawPriority", None),
        "transfer_rate_bytes_per_sec": ("TransferRateBytesPerSec", None),
        "metrics_interval": ("MetricsInterval", None), "ready_timeout_seconds": ("ReadyTimeoutSeconds", None),
        "stage_timeout": ("StageTimeout", None), "node_conns": ("NodeConns", None),
        "pq_capacity": ("PQCapacity", None), "pq_headroom": ("PQHeadroom", None),
        "controller_addr": ("ControllerAddr", None), "api_addr": ("APIAddr", None), "backend": ("Backend", None),
        "gpus": ("GPUs", None), "node_args": ("NodeArgs", None), "query_sample": ("QuerySample", None),
        "raw": (None, {"streams": ("RawStreams", None), "batch_bytes": ("RawBatchBytes", None),
                       "path": ("RawPath", None)}),
        "resources": ("NodeResources", {
            "cpus_per_node": ("CPUsPerNode", None), "first_cpu": ("FirstCPU", None),
            "memory_max": ("MemoryMax", None), "io_read_iops_max": ("IOReadIOPSMax", None),
            "io_read_bandwidth_max": ("IOReadBandwidthMax", None), "read_iops_per_node": ("ReadIOPSPerNode", None)}),
        "emulation": ("Emulation", {"netns": ("Netns", None), "delay": ("Delay", None), "rate": ("Rate", None)})}),
    "hosts": ("Hosts", {
        "nodes": ("Nodes", "opaque"), "root": ("Root", None), "user": ("User", None),
        "ssh_options": ("SSHOptions", None)}),
    "load": ("Load", {
        "k": ("K", None), "nprobe": ("NProbe", None), "n": ("N", None), "arrivals": ("Arrivals", None),
        "rate": ("Rate", None), "rate_steps": ("RateSteps", None), "users": ("Users", None),
        "user_steps": ("UserSteps", None), "retry_for_seconds": ("RetryForSeconds", None),
        "duration_seconds": ("DurationSeconds", None), "warmup_seconds": ("WarmupSeconds", None),
        "drain_seconds": ("DrainSeconds", None)}),
    "calibrate": ("Calibrate", {"rates": ("Rates", None), "seconds_per_rate": ("SecondsPerRate", None)}),
    "updates": ("Updates", {
        "headroom": ("Headroom", None),
        "list_headroom": ("ListHeadroom", None), "max_list_len": ("MaxListLen", None)}),
    "reconfigurations": ("Reconfigurations", {
        "at": ("TriggerTimeSeconds", None), "data_nodes": ("DataNodes", None), "plan": ("Plan", "opaque"),
        "rebuild": ("Rebuild", {
            "index": ("Index", None), "partitions": ("Partitions", None), "spares": ("Spares", None),
            "placement": ("Placement", None), "seed": ("Seed", None), "warmup": ("Warmup", None),
            "warmup_concurrency": ("WarmupConcurrency", None), "stop_the_world": ("StopTheWorld", None)})}),
}
SECTIONS = list(SCHEMA)  # the order sections are written in


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


def _relative_to(doc, d):
    """A plan file and a rebuild's directories are relative to the file that names them."""
    for rc in doc.get("reconfigurations") or []:
        if isinstance(rc, dict):
            if isinstance(rc.get("plan"), str):
                rc["plan"] = os.path.join(d, rc["plan"])
            rb = rc.get("rebuild")
            if isinstance(rb, dict):
                for k in ("index", "partitions"):
                    if isinstance(rb.get(k), str):
                        rb[k] = os.path.join(d, rb[k])


def _read(path, seen):
    path = os.path.abspath(path)
    if path in seen:
        raise ValueError("%s: a base cycle (%s)" % (path, " -> ".join(seen + [path])))
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        raise ValueError("%s: an experiment configuration is a mapping" % path)
    _relative_to(doc, os.path.dirname(path))
    b = doc.pop("base", None)
    if b is None:
        return doc
    return merge(_read(os.path.join(os.path.dirname(path), b), seen + [path]), doc)


def check(cfg, schema=SCHEMA, where=""):
    """Every key must be in the schema."""
    if not isinstance(cfg, dict):
        raise ValueError("%s: want a mapping, got %r" % (where or "configuration", cfg))
    for k, v in cfg.items():
        if k not in schema:
            raise ValueError("%s%s: unknown key (known: %s)" % (where, k, ", ".join(sorted(schema))))
        sub = schema[k][1]
        if isinstance(sub, dict) and v is not None:
            if isinstance(v, list):
                for i, item in enumerate(v):
                    check(item, sub, "%s%s[%d]." % (where, k, i))
            else:
                check(v, sub, "%s%s." % (where, k))


def load(path):
    """The configuration path describes: its bases merged under it, checked against SCHEMA."""
    cfg = _read(path, [])
    check(cfg)
    return cfg


def build_args(build):
    """fusion_build's flags from a build section."""
    args = []
    for k, flag in BUILD_FLAGS.items():
        v = (build or {}).get(k)
        if v is None:
            continue
        if k in BOOL_FLAGS:
            if v:
                args.append(flag)
            continue
        args += [flag, str(v)]
    return args + [str(a) for a in (build or {}).get("extra_args", [])]


def _flat(cfg, schema, out):
    for k, v in cfg.items():
        flat, sub = schema[k]
        if sub == "opaque":
            out[flat] = v
        elif isinstance(sub, dict):
            if isinstance(v, list):
                out[flat] = [_flat(item, sub, {}) for item in v]
            elif flat:
                out[flat] = _flat(v, sub, {})
            else:
                _flat(v, sub, out)
        elif flat:
            out[flat] = v
    return out


def flatten(cfg):
    """The flat form of a configuration that run_local.py's steps and the analysis scripts read."""
    c = {k: v for k, v in cfg.items() if k not in ("build", "partition")}
    out = _flat(c, SCHEMA, {})
    if "NodeArgs" in out:  # command-line arguments: numbers may be written as numbers
        out["NodeArgs"] = [str(a) for a in out["NodeArgs"]]
    b = cfg.get("build") or {}
    if b.get("index_dir") is not None:
        out["IndexDir"] = b["index_dir"]
    args = build_args(b)
    if args:
        out["BuildArgs"] = args
    p = cfg.get("partition") or {}
    if p.get("generator", "file") == "file" and p.get("file") is not None:
        out["Assignment"] = p["file"]
    return out


def ordered(cfg, schema=SCHEMA):
    """cfg with its keys in the schema's order, at every level."""
    if not isinstance(cfg, dict):
        return cfg
    out = {}
    for k in list(schema) + [k for k in cfg if k not in schema]:
        if k not in cfg:
            continue
        v, sub = cfg[k], schema.get(k, (None, None))[1]
        if isinstance(sub, dict):
            v = [ordered(x, sub) for x in v] if isinstance(v, list) else ordered(v, sub)
        out[k] = v
    return out


def dump(cfg):
    """A configuration as YAML, keys in the schema's order (sections in pipeline order), short
    lists on one line."""
    class Dumper(yaml.SafeDumper):
        pass

    def lists(d, v):
        flow = all(not isinstance(x, (dict, list)) for x in v)
        return d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=flow)
    Dumper.add_representer(list, lists)
    return yaml.dump(ordered(cfg), Dumper=Dumper, sort_keys=False, default_flow_style=False, width=100,
                     allow_unicode=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: experiment_config.py CONFIG.yaml")
    c = load(sys.argv[1])
    print(dump(c))
    print("# flat:")
    print(json.dumps(flatten(c), indent=2))
