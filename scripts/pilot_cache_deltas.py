#!/usr/bin/env python3
"""Copy the layer-26 token deltas of the pilot examples into one small file.

The token-delta cache stores pickled numpy arrays, so every chunk that holds a
needed example must be read in full (about 1.1 GB each). Reading them once on
a CPU node keeps that I/O out of the GPU jobs. Values are copied unchanged
(float16, as extracted); rows, example ids and positions are checked.

Usage:
  python scripts/pilot_cache_deltas.py --extract-dir D --layer 26 --populations P --out O.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--populations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pops = json.loads(args.populations.read_text())
    keep = sorted(set(pops["malicious_receivers"]) | set(pops["matched_benign_receivers"])
                  | {int(i) for v in pops["donor_policy_by_dept"].values() for i in v})
    per_chunk = int(json.loads((args.extract_dir / "extract_summary.json").read_text())["chunk_examples"])
    man = pd.read_csv(args.extract_dir / "chunk_manifest.csv")
    man = man[man["layer"] == args.layer].set_index("chunk_id")
    scores = pd.read_parquet(args.extract_dir / "example_scores.parquet").set_index("example_idx")

    parts, idx_parts, pos_parts, sources = [], [], [], []
    for cid in sorted({i // per_chunk for i in keep}):
        path = args.extract_dir / f"layer_{args.layer}" / Path(man.loc[cid, "path"]).name
        obj = torch.load(path, map_location="cpu", weights_only=False)
        ex = np.asarray(obj["example_idx"], dtype=np.int64)
        pos = np.asarray(obj["position"], dtype=np.int32)
        for i in (k for k in keep if k // per_chunk == cid):
            lo, hi = np.searchsorted(ex, i, "left"), np.searchsorted(ex, i, "right")
            if hi <= lo:
                raise SystemExit(f"example {i} absent from {path.name}")
            if not np.array_equal(pos[lo:hi], np.arange(hi - lo)):
                raise SystemExit(f"example {i}: positions are not 0..n-1")
            if hi - lo != int(scores.loc[i, "n_tokens"]):
                raise SystemExit(f"example {i}: {hi - lo} rows but n_tokens={scores.loc[i, 'n_tokens']}")
            parts.append(np.asarray(obj["delta"][lo:hi], dtype=np.float16))
            idx_parts.append(ex[lo:hi]); pos_parts.append(pos[lo:hi])
        sources.append({"chunk": path.name, "bytes": path.stat().st_size})
        del obj
        print(f"[cache] chunk {cid} done", flush=True)

    delta = np.concatenate(parts); idx = np.concatenate(idx_parts); pos = np.concatenate(pos_parts)
    np.savez(args.out, delta=delta, example_idx=idx, position=pos)
    meta = {"extract_dir": str(args.extract_dir), "layer": args.layer, "n_examples": len(keep),
            "n_rows": int(len(idx)), "d_model": int(delta.shape[1]), "dtype": "float16 (as extracted)",
            "populations_sha256": hashlib.sha256(args.populations.read_bytes()).hexdigest(),
            "out_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(), "source_chunks": sources}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: meta[k] for k in ("n_examples", "n_rows", "out_sha256")}), flush=True)


if __name__ == "__main__":
    main()
