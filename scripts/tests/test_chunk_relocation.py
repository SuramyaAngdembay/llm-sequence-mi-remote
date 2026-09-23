#!/usr/bin/env python3
"""Regression test for the relocated-cache defect behind jobs 20861864/5.

A token-delta cache's chunk_manifest.csv records ABSOLUTE chunk paths from
wherever the cache was built. A byte-identical copy therefore still points at
the original location. Three failure modes, only one of which is loud:

  1. original unreadable  -> PermissionError at startup (what happened);
  2. original READABLE    -> the job silently reads a different cache than the
                             one it was pointed at (the dangerous one);
  3. a chunk missing      -> silently skipped, so examples silently vanish.

Usage: python3 scripts/tests/test_chunk_relocation.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_token_delta_sae_causal import _select_token_chunks  # noqa: E402

FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def make_cache(root: Path, recorded_root: str, n_chunks: int = 3, layer: int = 26) -> list[Path]:
    """A cache under `root` whose manifest records paths under `recorded_root`."""
    ld = root / f"layer_{layer}"
    ld.mkdir(parents=True)
    (root / "extract_summary.json").write_text(json.dumps({"chunk_examples": 2}))
    rows = ["layer,chunk_id,path,rows,d_model,unit"]
    for c in range(n_chunks):
        (ld / f"chunk_{c:05d}.pt").write_bytes(b"real")
        rows.append(f"{layer},{c},{recorded_root}/layer_{layer}/chunk_{c:05d}.pt,2,8,token")
    (root / "chunk_manifest.csv").write_text("\n".join(rows) + "\n")
    return sorted(ld.glob("chunk_*.pt"))


def main() -> int:
    print("chunk resolution for a relocated cache\n")
    keep = np.array([0, 1, 2, 3], dtype=np.int64)          # needs chunks 0 and 1

    with tempfile.TemporaryDirectory() as t:
        t = Path(t)

        print("1. manifest points at a location that does not exist (the job failure)")
        cache = t / "staged"
        globbed = make_cache(cache, "/nonexistent/original/cache")
        try:
            sel = _select_token_chunks(cache, 26, globbed, keep)
            check("selection succeeds from the copy", True)
            check("exactly the two needed chunks", [p.name for p in sel] == ["chunk_00000.pt", "chunk_00001.pt"],
                  str([p.name for p in sel]))
            check("every selected file is inside the given directory",
                  all(str(p).startswith(str(cache)) for p in sel))
        except Exception as e:
            check("selection succeeds from the copy", False, f"{type(e).__name__}: {e}")

        print("\n2. manifest points at a READABLE decoy (the silent-redirect case)")
        decoy = t / "decoy"
        make_cache(decoy, str(decoy))                       # a real, readable other cache
        cache2 = t / "staged2"
        globbed2 = make_cache(cache2, str(decoy))           # its manifest points at the decoy
        try:
            sel2 = _select_token_chunks(cache2, 26, globbed2, keep)
            check("reads the directory it was GIVEN, not the one the manifest names",
                  all(str(p).startswith(str(cache2)) for p in sel2),
                  "decoy leaked: " + str([str(p) for p in sel2 if not str(p).startswith(str(cache2))]))
        except Exception as e:
            check("reads the directory it was GIVEN, not the one the manifest names", False,
                  f"{type(e).__name__}: {e}")

        print("\n3. a needed chunk is missing from the copy")
        cache3 = t / "partial"
        globbed3 = make_cache(cache3, "/nonexistent/original/cache")
        (cache3 / "layer_26" / "chunk_00001.pt").unlink()
        globbed3 = sorted((cache3 / "layer_26").glob("chunk_*.pt"))
        try:
            sel3 = _select_token_chunks(cache3, 26, globbed3, keep)
            check("a missing needed chunk raises instead of silently dropping examples", False,
                  f"returned {len(sel3)} chunk(s) of 2 needed")
        except FileNotFoundError as e:
            check("a missing needed chunk raises instead of silently dropping examples",
                  "partial cache" in str(e))

        print("\n4. backward compatibility: relative manifest paths")
        cache4 = t / "relative"
        ld = cache4 / "layer_26"; ld.mkdir(parents=True)
        (cache4 / "extract_summary.json").write_text(json.dumps({"chunk_examples": 2}))
        rows = ["layer,chunk_id,path,rows,d_model,unit"]
        for c in range(3):
            (ld / f"chunk_{c:05d}.pt").write_bytes(b"real")
            rows.append(f"26,{c},layer_26/chunk_{c:05d}.pt,2,8,token")
        (cache4 / "chunk_manifest.csv").write_text("\n".join(rows) + "\n")
        sel4 = _select_token_chunks(cache4, 26, sorted(ld.glob("chunk_*.pt")), keep)
        check("relative paths still resolve to the same two chunks",
              [p.name for p in sel4] == ["chunk_00000.pt", "chunk_00001.pt"])

    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}): " + "; ".join(FAIL))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
