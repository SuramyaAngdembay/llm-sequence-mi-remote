#!/usr/bin/env python3
"""Per-unit-norm sensitivity for the patching estimand.

Reviewer concern: top-set edits are 3-10x larger than active-control edits,
so RA(S)-RA(C) compares interventions of different absolute size. This
sensitivity recomputes the estimand on per-unit-norm effects: each best-row
NLL change is divided by its edit-component Frobenius norm before forming
the donor-type contrast RA and the top-minus-control difference. Dividing by
norm assumes a linear response and therefore favors the smaller control
edits; a contrast that survives it cannot be a pure magnitude artifact,
while one that does not survive is size-uncalibrated (not thereby refuted).

Inputs are intervention_norms_rows.csv files from intervention_norm_report.py.
Uncertainty: user-level cluster bootstrap over receiver users (malicious
users as resampling units), matching the paper's primary convention.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.rows_csv.open()))
    best = {}
    for r in rows:
        key = (r["context_mode"], r["feature_set"], r["donor_type"], r["receiver_example_id"])
        if key not in best or float(r["delta"]) < float(best[key]["delta"]):
            best[key] = r

    report = {}
    rng = np.random.default_rng(args.seed)
    for ctx in sorted({r["context_mode"] for r in rows}):
        per_user = defaultdict(list)  # user -> list of (raw_contrast, norm_contrast)
        recvs = {k[3] for k in best if k[0] == ctx}
        for rec in recvs:
            vals = {}
            ok = True
            for fs in ["top5", "control5_active"]:
                for dt in ["benign", "anomalous"]:
                    b = best.get((ctx, fs, dt, rec))
                    if b is None or float(b["edit_norm_fro"]) <= 0:
                        ok = False
                        break
                    vals[(fs, dt)] = (float(b["delta"]), float(b["edit_norm_fro"]))
                if not ok:
                    break
            if not ok:
                continue
            ra = {}
            ra_n = {}
            for fs in ["top5", "control5_active"]:
                (da, na), (db, nb) = vals[(fs, "anomalous")], vals[(fs, "benign")]
                ra[fs] = da - db
                ra_n[fs] = da / na - db / nb
            per_user[rec.split(":")[0]].append(
                (ra["top5"] - ra["control5_active"], ra_n["top5"] - ra_n["control5_active"])
            )
        users = sorted(per_user)
        if not users:
            continue
        raw_all = np.array([v[0] for u in users for v in per_user[u]])
        nrm_all = np.array([v[1] for u in users for v in per_user[u]])
        user_arrays = [np.array(per_user[u]) for u in users]
        nu = len(users)
        boot_raw = np.empty(args.draws)
        boot_nrm = np.empty(args.draws)
        for i in range(args.draws):
            pick = rng.integers(0, nu, size=nu)
            stacked = np.concatenate([user_arrays[j] for j in pick], axis=0)
            boot_raw[i] = stacked[:, 0].mean()
            boot_nrm[i] = stacked[:, 1].mean()
        report[ctx] = {
            "n_receivers": int(len(raw_all)), "n_users": nu,
            "raw_contrast": float(raw_all.mean()),
            "raw_ci": [float(np.percentile(boot_raw, 2.5)), float(np.percentile(boot_raw, 97.5))],
            "per_unit_norm_contrast": float(nrm_all.mean()),
            "per_unit_norm_ci": [float(np.percentile(boot_nrm, 2.5)), float(np.percentile(boot_nrm, 97.5))],
        }
        print(ctx, json.dumps(report[ctx]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
