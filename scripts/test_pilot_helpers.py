#!/usr/bin/env python3
"""Checks for the pure helpers of the 2026-09-25 pilots (numpy only, no GPU).

  python3 scripts/test_pilot_helpers.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_intervention_specificity import (edit_stats, edited_codes, load_delta_cache, shift_orig,  # noqa: E402
                                            shift_random, shift_recon, shift_residual)
from pilot_identity_context import swap_profile  # noqa: E402

rng = np.random.default_rng(0)
D, L, T = 16, 12, 9
W = rng.standard_normal((D, L)).astype(np.float32)
x_mean = rng.standard_normal(D).astype(np.float32)
x_std = rng.uniform(0.5, 2.0, D).astype(np.float32)
decode = lambda z: (np.asarray(z, dtype=np.float32) @ W.T) * x_std + x_mean

Z = np.zeros((T, L), dtype=np.float32)
Z[1, 2] = 1.5; Z[1, 5] = 0.7          # both listed features active
Z[3, 5] = 2.0                         # only feature 5 active
Z[4, 7] = 0.9                         # an unlisted feature only
Z[6, 2] = 0.4; Z[6, 9] = 1.1
delta = rng.standard_normal((T, D)).astype(np.float32)
feats, proto = [2, 5], np.array([1.0, 3.0], dtype=np.float32)
failures = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        failures.append(name)


zu = edited_codes(Z, feats, proto, "union")
check("union: every listed coordinate set at union rows", np.allclose(zu[[1, 3, 6]][:, feats], proto))
check("union: other rows unchanged", np.array_equal(zu[[0, 2, 4, 5, 7, 8]], Z[[0, 2, 4, 5, 7, 8]]))
check("union: unlisted coordinates unchanged", np.array_equal(np.delete(zu, feats, axis=1), np.delete(Z, feats, axis=1)))
zo = edited_codes(Z, feats, proto, "own")
check("own: writes only into active coordinates", zo[3, 2] == 0 and zo[3, 5] == 3.0 and zo[6, 5] == 0 and zo[6, 2] == 1.0)

for mode, zp in (("union", zu), ("own", zo)):
    sh = shift_residual(Z, feats, proto, decode, mode, D)
    expect = ((zp - Z) @ W.T) * x_std
    check(f"residual ({mode}) = W(z'-z)*x_std", np.allclose(sh, expect, atol=1e-5))
    check(f"residual ({mode}) exactly zero on unchanged rows", np.all(sh[(zp == Z).all(axis=1)] == 0))

rp = shift_residual(Z, feats, proto, decode, "union", D)
check("orig = residual(union) + reconstruction error", np.allclose(shift_orig(Z, delta, feats, proto, decode),
                                                                 rp + (decode(Z) - delta), atol=1e-5))
check("recon = dec(z) - delta", np.allclose(shift_recon(Z, delta, feats, decode), decode(Z) - delta, atol=1e-6))
Zn = np.zeros_like(Z); Zn[2, 7] = 1.0
check("orig and recon are zero when no listed feature is active",
      not shift_orig(Zn, delta, feats, proto, decode).any() and not shift_recon(Zn, delta, feats, decode).any())

singles = sum(shift_residual(Z, [f], proto[[j]], decode, "own", D) for j, f in enumerate(feats))
check("own-support joint edit = sum of single-feature edits (hidden state is additive)",
      np.allclose(shift_residual(Z, feats, proto, decode, "own", D), singles, atol=1e-5))

dirs = rng.standard_normal((5, D)).astype(np.float32)
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
for name, rnd in (("isotropic", shift_random(rp, np.random.default_rng(1))),
                  ("dictionary", shift_random(rp, np.random.default_rng(1), dirs))):
    check(f"random ({name}): per-token norms match", np.allclose(np.linalg.norm(rnd, axis=1), np.linalg.norm(rp, axis=1), atol=1e-5))
    check(f"random ({name}): same edited positions", np.array_equal(np.linalg.norm(rnd, axis=1) > 0, np.linalg.norm(rp, axis=1) > 0))
rnd = shift_random(rp, np.random.default_rng(1), dirs)
rows = np.flatnonzero(np.linalg.norm(rp, axis=1) > 0)
unit = rnd[rows] / np.linalg.norm(rnd[rows], axis=1, keepdims=True)
check("random (dictionary): each row is a dictionary direction", np.allclose(np.abs(unit @ dirs.T).max(axis=1), 1.0, atol=1e-5))
cos = np.sum(shift_random(rp, np.random.default_rng(1))[rows] * rp[rows], axis=1) / (np.linalg.norm(rp[rows], axis=1) ** 2)
check("random (isotropic): not aligned with the edit", np.abs(cos).max() < 0.99)

st = edit_stats(rp, ["SES", "DAY", "PSY", "SES", "SES", "SES", "SES", "SESCOUNT", "SES"])   # edited rows: 1, 3, 6
check("edit_stats: counts and classes", st["n_edited"] == 3 and st["edited_SES"] == 2 and st["edited_DAY"] == 1
      and st["n_tokens"] == T and abs(st["norm_l2_total"] - float(np.linalg.norm(rp))) < 1e-4)

recv = "DAY week=7 project=na role=39 dept=15 team=11\nPSY O=1 C=2 E=3 A=4 N=5\nSESSIONS total=1 kept=1\nSES idx=0 pc=0"
part = "DAY week=40 project=p2 role=12 dept=15 team=3\nPSY O=9 C=9 E=9 A=9 N=9\nSESSIONS total=2 kept=2\nSES idx=0 pc=1"
new = swap_profile(recv, part)
check("swap: partner profile with the receiver's week, sessions kept",
      new.split("\n")[0] == "DAY week=7 project=p2 role=12 dept=15 team=3" and new.split("\n")[1] == part.split("\n")[1]
      and new.split("\n")[2:] == recv.split("\n")[2:])
check("swap: list of profile lines accepted", swap_profile(recv, part.split("\n")[:2]) == new)

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "c.npz"
    idx = np.array([3, 3, 3, 8, 8, 11], dtype=np.int64)
    np.savez(path, delta=np.arange(12, dtype=np.float16).reshape(6, 2), example_idx=idx, position=np.array([0, 1, 2, 0, 1, 0]))
    xs, ids = load_delta_cache(path, {3, 11})
    check("delta cache: filters examples, keeps order and values", ids.tolist() == [3, 3, 3, 11]
          and xs.dtype == np.float32 and xs[-1].tolist() == [10.0, 11.0])

print(f"{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
