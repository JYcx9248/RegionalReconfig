#!/usr/bin/env python3
"""Converts experiment configurations from the flat JSON form to YAML, a base and overrides
(scripts/experiment_config.py).

    python3 scripts/convert_configs.py [--families NAME=PREFIX,...] [--keep] configs/experiment.*.json

Files with the same dataset form a family. Its base, configs/base/NAME.yaml, holds what every one
of them has in common, and each experiment file, configs/experiment.X.yaml, holds the rest, with
`base:` naming the family's base (a family of one has no base). A "Comment" becomes a YAML comment
at the top of the file. Every new file is read back and must give exactly the old configuration
before any file is written; then the JSON files are removed (--keep keeps them).
"""

import argparse
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_config as ec  # noqa: E402


def inverse(schema, prefix=()):
    """Flat key -> (YAML path, sub-schema), for one level of the flat form."""
    m = {}
    for k, (flat, sub) in schema.items():
        path = prefix + (k,)
        if isinstance(sub, dict) and flat is None:
            m.update(inverse(sub, path))
        elif flat:
            m[flat] = (path, sub)
    return m


def typed(s):
    for t in (int, float):
        try:
            return t(s)
        except ValueError:
            pass
    return s


def build_section(args):
    """fusion_build's flags as a build section."""
    flags = {f: k for k, f in ec.BUILD_FLAGS.items()}
    out, extra, i = {}, [], 0
    while i < len(args):
        a = args[i]
        k = flags.get(a)
        if k in ec.BOOL_FLAGS:
            out[k], i = True, i + 1
        elif k and i + 1 < len(args):
            out[k], i = typed(args[i + 1]), i + 2
        else:
            extra.append(a)
            i += 1
    if extra:
        out["extra_args"] = extra
    return out


def put(out, path, v):
    for k in path[:-1]:
        out = out.setdefault(k, {})
    out[path[-1]] = v


def unflatten(d, schema=ec.SCHEMA, top=True):
    inv = inverse(schema)
    out = {}
    for fk, v in d.items():
        if top and fk == "BuildArgs":
            for k, x in build_section(v).items():
                put(out, ("build", k), x)
            continue
        if fk not in inv:
            raise ValueError("unknown key %r" % fk)
        path, sub = inv[fk]
        if isinstance(sub, dict):
            v = [unflatten(x, sub, False) for x in v] if isinstance(v, list) else unflatten(v, sub, False)
        if top and fk == "NodeArgs":
            v = [typed(x) if isinstance(x, str) else x for x in v]
        if top and fk == "Assignment":
            put(out, ("partition", "generator"), "file")
        put(out, path, v)
    return out


def common(dicts):
    out = {}
    for k in dicts[0]:
        vals = [d.get(k, KeyError) for d in dicts]
        if any(v is KeyError for v in vals):
            continue
        if all(isinstance(v, dict) for v in vals):
            c = common(vals)
            if c:
                out[k] = c
        elif all(v == vals[0] for v in vals):
            out[k] = vals[0]
    return out


def diff(d, base):
    out = {}
    for k, v in d.items():
        if k not in base:
            out[k] = v
        elif isinstance(v, dict) and isinstance(base[k], dict):
            sub = diff(v, base[k])
            if sub:
                out[k] = sub
        elif v != base[k]:
            out[k] = v
    return out


def normalized(flat):
    """The flat form, with fusion_build's flags as a mapping (their order does not matter)."""
    f = dict(flat)
    f.pop("Comment", None)
    if "BuildArgs" in f:
        f["BuildArgs"] = build_section(f["BuildArgs"])
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--families", default="wiki-c1024=wiki-c1024,bigann=bigann,local=small,local=syn",
                    help="NAME=PREFIX,...: a file whose dataset's name starts with PREFIX is in family NAME")
    ap.add_argument("--keep", action="store_true", help="keep the JSON files")
    a = ap.parse_args()
    rules = [r.split("=", 1) for r in a.families.split(",")]
    docs, fams = {}, {}
    for path in a.files:
        flat = json.load(open(path))
        cfg = unflatten({k: v for k, v in flat.items() if k != "Comment"})
        name = os.path.basename((cfg.get("dataset") or {}).get("vectors", ""))
        fam = next((n for n, p in rules if name.startswith(p)), None)
        docs[path] = (flat, cfg)
        fams.setdefault(fam, []).append(path)
    out_dir = os.path.dirname(os.path.abspath(a.files[0]))
    writes = {}
    for fam, paths in sorted(fams.items(), key=lambda x: str(x[0])):
        base = common([docs[p][1] for p in paths]) if fam and len(paths) > 1 else {}
        bpath = os.path.join(out_dir, "base", fam + ".yaml") if base else None
        if base:
            head = ("# Base of the %s experiments (%d files): the settings every one of them shared when\n"
                    "# they were converted from JSON. An experiment file overrides any of them.\n" % (fam, len(paths)))
            writes[bpath] = head + ec.dump(base)
        for p in paths:
            flat, cfg = docs[p]
            mine = diff(cfg, base)
            if base:
                mine = dict({"base": os.path.relpath(bpath, out_dir)}, **mine)
            text = ""
            if flat.get("Comment"):
                text = "".join("# " + l + "\n" for l in textwrap.wrap(flat["Comment"], 96)) + "\n"
            yp = os.path.splitext(p)[0] + ".yaml"
            writes[yp] = text + ec.dump(mine)
    # Read every new file back (from a scratch copy) before writing anything.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for path, text in writes.items():
            q = os.path.join(tmp, os.path.relpath(path, out_dir))
            os.makedirs(os.path.dirname(q), exist_ok=True)
            open(q, "w").write(text)
        for p, (flat, _) in docs.items():
            got = ec.flatten(ec.load(os.path.join(tmp, os.path.relpath(os.path.splitext(p)[0] + ".yaml", out_dir))))
            if normalized(got) != normalized(flat):
                sys.exit("%s: the YAML does not give the same configuration:\n%s\n%s" %
                         (p, json.dumps(normalized(flat), sort_keys=True), json.dumps(normalized(got), sort_keys=True)))
    for path, text in writes.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(text)
    if not a.keep:
        for p in docs:
            os.remove(p)
    for fam, paths in sorted(fams.items(), key=lambda x: str(x[0])):
        print("%-12s %2d files%s" % (fam or "(no family)", len(paths),
                                     ", base configs/base/%s.yaml" % fam if fam and len(paths) > 1 else ""))
    print("%d files written, each read back equal to its JSON%s" % (len(writes), "" if a.keep else "; JSON removed"))


if __name__ == "__main__":
    main()
