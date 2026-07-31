#!/usr/bin/env python3
"""User-level cluster-bootstrap CIs for the masking ablation's ROC changes.

Joins the original and masked example_scores parquets on example_id, then for
each bootstrap draw resamples benign users and positive users independently
with replacement (stratified user-level cluster bootstrap) and recomputes
day-level and user-level ROC-AUC for every arm under the SAME resampled
population, yielding paired draws for delta-ROC (variant minus original).

ROC under user multiplicities is computed as a weighted Mann-Whitney U:
each day carries its user's resample multiplicity as a weight; ties get 0.5.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def weighted_roc_prep(scores_ben: np.ndarray, scores_pos: np.ndarray):
    order = np.argsort(scores_ben, kind="mergesort")
    s_sorted = scores_ben[order]
    lo = np.searchsorted(s_sorted, scores_pos, side="left")
    hi = np.searchsorted(s_sorted, scores_pos, side="right")
    return order, lo, hi


def weighted_roc_batch(w_ben_sorted: np.ndarray, w_pos: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """w_ben_sorted: draws x n_ben (already in sorted-score order); w_pos: draws x n_pos."""
    csum = np.cumsum(w_ben_sorted, axis=1)
    total_ben = csum[:, -1]
    zeros = np.zeros((csum.shape[0], 1))
    padded = np.concatenate([zeros, csum], axis=1)
    w_lt = padded[:, lo]
    w_le = padded[:, hi]
    contrib = w_lt + 0.5 * (w_le - w_lt)
    num = (w_pos * contrib).sum(axis=1)
    den = total_ben * w_pos.sum(axis=1)
    return num / den


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--variant", action="append", required=True,
                    help="name=path, repeatable")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = pd.read_parquet(args.original)[["example_id", "user_id", "y", "adapted_nll"]]
    arms: Dict[str, pd.DataFrame] = {"original": base}
    for spec in args.variant:
        name, path = spec.split("=", 1)
        arms[name] = pd.read_parquet(Path(path))[["example_id", "user_id", "y", "adapted_nll"]]
    common = None
    for df in arms.values():
        ids = set(df["example_id"])
        common = ids if common is None else common & ids
    for name in arms:
        df = arms[name]
        arms[name] = df[df["example_id"].isin(common)].sort_values("example_id").reset_index(drop=True)
    ref = arms["original"]
    assert all((arms[n]["example_id"].values == ref["example_id"].values).all() for n in arms)

    users = ref["user_id"].values
    y = ref["y"].values.astype(int)
    # Strata follow the USER label (max y over the user's days), matching
    # compare_masked_scores.metrics: a malicious user's benign days carry that
    # user's multiplicity and fold into the user's single max-aggregated entry.
    ulab = ref.groupby("user_id")["y"].max()
    mal_users = np.array(sorted(ulab[ulab == 1].index))
    ben_users = np.array(sorted(ulab[ulab == 0].index))
    nb, npos = len(ben_users), len(mal_users)
    uidx = {u: ("mal", i) for i, u in enumerate(mal_users)}
    uidx.update({u: ("ben", i) for i, u in enumerate(ben_users)})
    ben_mask = y == 0
    pos_mask = y == 1
    # Each day maps to (stratum, index-in-stratum) of its user.
    def day_map(day_users):
        strat = np.array([0 if uidx[u][0] == "mal" else 1 for u in day_users])
        idx = np.array([uidx[u][1] for u in day_users])
        return strat, idx
    ben_day_strat, ben_day_idx = day_map(users[ben_mask])
    pos_day_strat, pos_day_idx = day_map(users[pos_mask])

    day_prep = {}
    user_prep = {}
    for name, df in arms.items():
        s = df["adapted_nll"].values
        order, lo, hi = weighted_roc_prep(s[ben_mask], s[pos_mask])
        day_prep[name] = (order, lo, hi)
        umax = df.groupby("user_id")["adapted_nll"].max()
        ub = umax.loc[ben_users].values
        up = umax.loc[mal_users].values
        uorder, ulo, uhi = weighted_roc_prep(ub, up)
        user_prep[name] = (uorder, ulo, uhi)

    rng = np.random.default_rng(args.seed)
    day_draws: Dict[str, List[np.ndarray]] = {n: [] for n in arms}
    user_draws: Dict[str, List[np.ndarray]] = {n: [] for n in arms}
    done = 0
    while done < args.draws:
        m = min(args.chunk, args.draws - done)
        mult_ben = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=m).astype(np.float64)
        mult_pos = rng.multinomial(npos, np.full(npos, 1.0 / npos), size=m).astype(np.float64)
        def day_weights(strat, idx):
            w = np.empty((m, len(idx)))
            mal_cols = strat == 0
            w[:, mal_cols] = mult_pos[:, idx[mal_cols]]
            w[:, ~mal_cols] = mult_ben[:, idx[~mal_cols]]
            return w
        w_ben_day = day_weights(ben_day_strat, ben_day_idx)
        w_pos_day = day_weights(pos_day_strat, pos_day_idx)
        for name in arms:
            order, lo, hi = day_prep[name]
            day_draws[name].append(weighted_roc_batch(w_ben_day[:, order], w_pos_day, lo, hi))
            uorder, ulo, uhi = user_prep[name]
            user_draws[name].append(weighted_roc_batch(mult_ben[:, uorder], mult_pos, ulo, uhi))
        done += m
        print(f"[boot] {done}/{args.draws}", flush=True)

    def point_roc(name: str, level: str) -> float:
        if level == "day":
            order, lo, hi = day_prep[name]
            w_b = np.ones((1, len(order)))
            w_p = np.ones((1, len(pos_day_idx)))
            return float(weighted_roc_batch(w_b, w_p, lo, hi)[0])
        uorder, ulo, uhi = user_prep[name]
        return float(weighted_roc_batch(np.ones((1, nb)), np.ones((1, npos)), ulo, uhi)[0])

    report = {"n_days": int(len(y)), "n_pos_days": int(pos_mask.sum()),
              "n_benign_users": nb, "n_pos_users": npos, "draws": args.draws}
    for level, draws in [("day", day_draws), ("user", user_draws)]:
        orig = np.concatenate(draws["original"])
        for name in arms:
            vals = np.concatenate(draws[name])
            entry = {"point": point_roc(name, level),
                     "ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}
            if name != "original":
                d = vals - orig
                entry["delta_point"] = point_roc(name, level) - point_roc("original", level)
                entry["delta_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            report[f"{level}_roc_{name}"] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
