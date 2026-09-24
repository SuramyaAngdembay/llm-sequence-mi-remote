#!/usr/bin/env python3
"""Row-by-row comparison of two intervention runs of the same design.

Built for the environment gate in docs/PREREGISTRATION_CERT_PACKAGE4.md
(addendum item 9): the collaborator-environment smoke must reproduce the
owner-environment smoke closely enough that environment differences cannot
explain the full run's selected-minus-control contrast. Computing both arms in
one run does not guarantee that environment effects cancel.

Rows are matched on (context_mode, feature_set, donor_type, receiver,
donor, alpha). Reported: match counts, per-field absolute differences, and the
selected-minus-control contrast (mean of receiver-user means of per-receiver
all-donor means) in each run on the matched rows only.

Gate (declared 2026-09-24, before either smoke's collaborator counterpart ran):
  PASS if every matched per-row |delta difference| <= --row-tol (1e-3) and the
  contrast differs by <= --contrast-tol (1e-4) in every view checked.

Stdlib only.
Usage:
  python3 scripts/compare_intervention_runs.py <run-a> <run-b> [--views full,behavior_only,profile_only]
  python3 scripts/compare_intervention_runs.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

KEY = ("context_mode", "feature_set", "donor_type", "receiver_example_id", "donor_example_id", "alpha")


def load(run: Path) -> dict:
    rows = csv.DictReader((run / "token_delta_sae_causal_candidate_rows.csv").open())
    out = {}
    for r in rows:
        k = tuple(r[c] if c != "alpha" else f"{float(r[c]):.6g}" for c in KEY)
        if k in out:
            raise SystemExit(f"duplicate key {k} in {run}")
        out[k] = r
    return out


def contrast(rows: dict, keys, view: str) -> float:
    sets = sorted({k[1] for k in keys})
    ctrl = next(s for s in sets if s.startswith("control"))
    top = next(s for s in sets if s != ctrl)
    acc = defaultdict(list)
    for k in keys:
        acc[(k[1], k[3])].append(float(rows[k][f"delta_{view}"]))
    recv = sorted({k[3] for k in keys})
    by_user = defaultdict(list)
    for rid in recv:
        if (top, rid) in acc and (ctrl, rid) in acc:
            by_user[rid.split(":")[0]].append(st.fmean(acc[(top, rid)]) - st.fmean(acc[(ctrl, rid)]))
    return st.fmean(st.fmean(v) for v in by_user.values()) if by_user else float("nan")


def compare(a: dict, b: dict, views, row_tol: float, contrast_tol: float) -> dict:
    shared = sorted(set(a) & set(b))
    res = {"matched": len(shared), "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a)),
           "fields": {}, "contrasts": {}, "pass": True, "reasons": []}
    if not shared:
        res["pass"] = False
        res["reasons"].append("no matched rows")
        return res
    for field in ["base_full", "patched_full"] + [f"delta_{v}" for v in views]:
        if field not in a[shared[0]] or field not in b[shared[0]]:
            continue
        diffs = [abs(float(a[k][field]) - float(b[k][field])) for k in shared]
        res["fields"][field] = {"max_abs_diff": max(diffs), "mean_abs_diff": st.fmean(diffs)}
        if field.startswith("delta_") and max(diffs) > row_tol:
            res["pass"] = False
            res["reasons"].append(f"{field}: max per-row difference {max(diffs):.3e} > {row_tol}")
    for v in views:
        ca, cb = contrast(a, shared, v), contrast(b, shared, v)
        res["contrasts"][v] = {"run_a": ca, "run_b": cb, "difference": cb - ca}
        if abs(cb - ca) > contrast_tol:
            res["pass"] = False
            res["reasons"].append(f"{v}: contrast differs by {cb - ca:+.3e} (> {contrast_tol})")
    return res


def self_test() -> int:
    def write(td: Path, name: str, shift: float) -> Path:
        d = td / name
        d.mkdir()
        with (d / "token_delta_sae_causal_candidate_rows.csv").open("w", newline="") as fh:
            w = None
            for u in range(3):
                for fs, eff in (("top5", -0.01), ("control5_active", -0.004)):
                    for dn in range(2):
                        row = {"context_mode": "team", "feature_set": fs, "donor_type": "benign",
                               "receiver_example_id": f"U{u}:1", "donor_example_id": f"D{dn}", "alpha": "1.0",
                               "base_full": "1.0", "patched_full": str(1.0 + eff),
                               "delta_full": str(eff + (shift if fs == "top5" else 0.0))}
                        if w is None:
                            w = csv.DictWriter(fh, fieldnames=list(row))
                            w.writeheader()
                        w.writerow(row)
        return d
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        a, same, far = write(td, "a", 0.0), write(td, "b", 5e-5), write(td, "c", 5e-3)
        ok = compare(load(a), load(same), ["full"], 1e-3, 1e-4)
        bad = compare(load(a), load(far), ["full"], 1e-3, 1e-4)
    checks = {
        "all rows match": ok["matched"] == 12 and ok["only_a"] == 0 and ok["only_b"] == 0,
        "a 5e-5 shift in one arm passes and is measured":
            ok["pass"] and abs(ok["contrasts"]["full"]["difference"] - 5e-5) < 1e-12,
        "a 5e-3 shift fails both tolerances": (not bad["pass"]) and len(bad["reasons"]) == 2,
    }
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", nargs="?", type=Path)
    ap.add_argument("run_b", nargs="?", type=Path)
    ap.add_argument("--views", default="full,behavior_only,profile_only")
    ap.add_argument("--row-tol", type=float, default=1e-3)
    ap.add_argument("--contrast-tol", type=float, default=1e-4)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not (args.run_a and args.run_b):
        ap.error("two run directories are required")
    res = compare(load(args.run_a), load(args.run_b), args.views.split(","), args.row_tol, args.contrast_tol)
    print(f"matched rows {res['matched']}  only in A {res['only_a']}  only in B {res['only_b']}")
    for f, d in res["fields"].items():
        print(f"  {f:<22} max |diff| {d['max_abs_diff']:.3e}  mean |diff| {d['mean_abs_diff']:.3e}")
    for v, c in res["contrasts"].items():
        print(f"  contrast {v:<14} A {c['run_a']:+.6f}  B {c['run_b']:+.6f}  B-A {c['difference']:+.3e}")
    print("GATE", "PASS" if res["pass"] else "FAIL", *res["reasons"], sep="\n  " if res["reasons"] else " ")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
