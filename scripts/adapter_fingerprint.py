#!/usr/bin/env python3
"""Deterministic fingerprint of a LoRA adapter's weights.

Why this exists: work package 3 ran on adapters held on one machine while the
published numbers came from adapters on another. We recorded the safetensors
file md5 and called the provenance "consistent". That was an assumption. A file
md5 is the wrong instrument — two files can hold identical weights and differ in
header ordering or metadata, and a file that differs in one byte tells you
nothing about *which* tensor changed.

This hashes the tensor CONTENT, per tensor, so the comparison is about the model
rather than the container:

  * per tensor: name, dtype, shape, sha256 of its raw little-endian bytes;
  * a combined digest over the sorted per-tensor records;
  * the adapter_config fields that change what the weights mean (r, alpha,
    target modules, base model), hashed separately.

Two adapters with the same `weights_digest` are the same weights. Two with the
same `weights_digest` but different `config_digest` are the same tensors applied
differently, which is a real difference and is reported separately rather than
folded into one number.

Usage:
  python3 scripts/adapter_fingerprint.py FINGERPRINT <adapter-dir> [-o out.json]
  python3 scripts/adapter_fingerprint.py COMPARE a.json b.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

CONFIG_KEYS = ("base_model_name_or_path", "peft_type", "r", "lora_alpha",
               "lora_dropout", "target_modules", "bias", "use_rslora", "use_dora",
               "fan_in_fan_out", "modules_to_save", "init_lora_weights")


def _tensor_records(path: Path) -> list[dict]:
    """Per-tensor content hashes, read straight from the safetensors container.

    Parsed here rather than via torch so the fingerprint needs no GPU, no torch
    and no dtype conversion -- a conversion would make the hash depend on the
    reading library rather than on the file.
    """
    raw = path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + n])
    base = 8 + n
    out = []
    for name, spec in sorted(header.items()):
        if name == "__metadata__":
            continue
        a, b = spec["data_offsets"]
        out.append({
            "name": name,
            "dtype": spec["dtype"],
            "shape": spec["shape"],
            "n_bytes": b - a,
            "sha256": hashlib.sha256(raw[base + a : base + b]).hexdigest(),
        })
    return out


def fingerprint(adapter_dir: Path) -> dict:
    st = adapter_dir / "adapter_model.safetensors"
    if not st.exists():
        raise SystemExit(f"no adapter_model.safetensors in {adapter_dir}")
    recs = _tensor_records(st)
    if not recs:
        raise SystemExit(f"{st} contains no tensors")
    h = hashlib.sha256()
    for r in recs:
        h.update(f"{r['name']}|{r['dtype']}|{r['shape']}|{r['sha256']}\n".encode())

    cfg_path = adapter_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    picked = {}
    for k in CONFIG_KEYS:
        v = cfg.get(k)
        picked[k] = sorted(v) if isinstance(v, list) else v
    ch = hashlib.sha256(json.dumps(picked, sort_keys=True).encode()).hexdigest()

    return {
        "adapter_dir": str(adapter_dir),
        "n_tensors": len(recs),
        "total_bytes": sum(r["n_bytes"] for r in recs),
        "weights_digest": h.hexdigest(),
        "config_digest": ch,
        "config": picked,
        "tensors": recs,
    }


def compare(a: dict, b: dict) -> int:
    same_w = a["weights_digest"] == b["weights_digest"]
    same_c = a["config_digest"] == b["config_digest"]
    print(f"A: {a['adapter_dir']}  ({a['n_tensors']} tensors)")
    print(f"B: {b['adapter_dir']}  ({b['n_tensors']} tensors)")
    print(f"\nweights_digest  {'MATCH' if same_w else 'DIFFER'}")
    print(f"  A {a['weights_digest']}\n  B {b['weights_digest']}")
    print(f"config_digest   {'MATCH' if same_c else 'DIFFER'}")
    if not same_c:
        for k in CONFIG_KEYS:
            if a["config"].get(k) != b["config"].get(k):
                print(f"    {k}: {a['config'].get(k)!r} vs {b['config'].get(k)!r}")
    if not same_w:
        ta = {t["name"]: t for t in a["tensors"]}
        tb = {t["name"]: t for t in b["tensors"]}
        only_a, only_b = sorted(set(ta) - set(tb)), sorted(set(tb) - set(ta))
        diff = [n for n in sorted(set(ta) & set(tb)) if ta[n]["sha256"] != tb[n]["sha256"]]
        print(f"\n  tensors only in A: {len(only_a)}   only in B: {len(only_b)}"
              f"   differing: {len(diff)} of {len(set(ta) & set(tb))} shared")
        for n in diff[:8]:
            print(f"    {n}")
        if len(diff) > 8:
            print(f"    ... and {len(diff) - 8} more")
    print("\nVERDICT: " + (
        "identical weights and configuration" if same_w and same_c else
        "identical weights, DIFFERENT configuration -- same tensors, applied differently"
        if same_w else
        "DIFFERENT weights -- these are not the same adapter"))
    return 0 if (same_w and same_c) else 1


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    mode = sys.argv[1].upper()
    if mode == "FINGERPRINT":
        fp = fingerprint(Path(sys.argv[2]))
        text = json.dumps(fp, indent=2)
        if "-o" in sys.argv:
            out = Path(sys.argv[sys.argv.index("-o") + 1])
            out.write_text(text)
            print(f"{fp['weights_digest'][:16]}...  {fp['n_tensors']} tensors  -> {out}")
        else:
            print(text)
        return 0
    if mode == "COMPARE":
        return compare(json.loads(Path(sys.argv[2]).read_text()),
                       json.loads(Path(sys.argv[3]).read_text()))
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
