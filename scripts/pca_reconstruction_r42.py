#!/usr/bin/env python3
"""PCA reconstruction-error detector on CERT r4.2 day-level features (thesis Chapter II/III), 2026-09-28.

Modernized from InsiderThreatDetection/r4.2/pca_based_detection.ipynb (cell 8) on Magnolia:
  * no GMM: the anomaly score is the reconstruction error itself, e = ||x - x_hat||^2 in standardized units;
  * labels: a day is malicious if the extraction's `insider` column is > 0 (the notebook used == 1, which kept
    only scenario 1 and silently dropped the scenario-2 and scenario-3 days);
  * identifiers and time indices (user, day, week, starttime, endtime) are not used as features;
  * user-disjoint split: the same users as the language-model experiments (train 851 benign users;
    held-out 79 benign users; 70 insiders), so scaler and PCA see only benign training users;
  * evaluation: day-level AUC and user-level AUC (a user's score is their worst day), with 95% intervals
    that resample users; all features, and behavior only (profile columns removed), as in Chapter IV;
  * columns that are constant in training are zero after centering, lie in no principal direction and keep
    their raw units, so any nonzero value on an evaluation day enters the error unscaled; their share of the
    error is reported, and variants without them are run as a sensitivity check;
  * one full SVD per variant; q follows sklearn's rule for a variance threshold (checked against sklearn's own
    truncation), and the threshold's effect is reported at 0.80, 0.90, 0.95 and 0.99.

  python pca_reconstruction_r42.py --day-csv dayr4.2.csv --user-map sessionr4.2_user_map.csv \
      --splits r42_user_splits.csv --out-dir OUT [--pve 0.95]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ID_COLS = ["starttime", "endtime", "user", "day", "week", "insider"]
PROFILE_COLS = ["role", "b_unit", "f_unit", "dept", "team", "ITAdmin", "O", "C", "E", "A", "N"]


def rec_mean_const(full: PCA, const: np.ndarray, q: int) -> np.ndarray:
    """Reconstruction of the training-constant columns. A direction with positive variance is zero on a column that
    never varies (that column's covariance row is zero), so the first q directions give x_hat = mean = 0 there."""
    assert np.allclose(full.components_[:q][:, const], 0, atol=1e-8) and np.allclose(full.mean_[const], 0)
    assert full.explained_variance_[q - 1] > 0
    return full.mean_[const]


def user_auc(scores_by_user: dict, mal: list, ben: list) -> float:
    """Fraction of (malicious, benign) user pairs ranked correctly; ties count 1/2 (the thesis's eq. for AUC)."""
    m = np.array([scores_by_user[u] for u in mal]); b = np.array([scores_by_user[u] for u in ben])
    return float(((m[:, None] > b[None, :]).sum() + 0.5 * (m[:, None] == b[None, :]).sum()) / (len(m) * len(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-csv", required=True); ap.add_argument("--user-map", required=True)
    ap.add_argument("--splits", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pve", type=float, default=0.95); ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    t0, c0 = time.time(), time.process_time()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.day_csv)
    um = pd.read_csv(a.user_map); code2id = dict(zip(um.user_code.astype(int), um.user_id.astype(str)))
    sp = pd.read_csv(a.splits); split = dict(zip(sp.user_id, sp.split)); lm_pool = set(sp.user_id[sp.lm_pool_insider])
    df["user_id"] = df["user"].astype(int).map(code2id)
    assert df.user_id.notna().all(), "unmapped user codes"
    df["split"] = df.user_id.map(split)
    df["y"] = (df["insider"] > 0).astype(int)
    ins_users = set(df.user_id[df.y == 1])
    # consistency checks between the day table, the user map and the language-model split
    checks = {"rows": int(len(df)), "users": int(df.user_id.nunique()), "malicious_days": int(df.y.sum()),
              "malicious_days_by_scenario": {int(k): int(v) for k, v in df.loc[df.y == 1, "insider"].value_counts().sort_index().items()},
              "insider_users": len(ins_users), "insider_users_all_in_eval_split": bool(all(split.get(u) == "eval" for u in ins_users)),
              "eval_split_users_all_insiders": bool({u for u, s in split.items() if s == "eval"} == ins_users),
              "users_without_split": int(df.split.isna().sum())}
    print(json.dumps(checks), flush=True)
    assert checks["insider_users_all_in_eval_split"] and checks["eval_split_users_all_insiders"]
    feats_all = [c for c in df.columns if c not in ID_COLS + ["user_id", "split", "y"]]
    train = df[df.split == "train"]; assert train.y.sum() == 0
    varying = [c for c in feats_all if train[c].nunique() > 1]      # constant in training: zero after centering
    variants = {"all features": feats_all, "behavior only": [c for c in feats_all if c not in PROFILE_COLS],
                "all features, varying in training": varying,
                "behavior only, varying in training": [c for c in varying if c not in PROFILE_COLS]}
    ben_users = sorted(df.user_id[df.split == "val"].unique())
    mal_all = sorted(ins_users); mal_lm = sorted(ins_users & lm_pool)
    ev = df[df.split.isin(["val", "eval"])].copy()
    rng = np.random.default_rng(a.seed)
    res = {"checks": checks, "protocol": {"pve_threshold": a.pve, "train_users": int(train.user_id.nunique()), "train_days": int(len(train)),
                                          "heldout_benign_users": len(ben_users), "insiders": len(mal_all), "lm_pool_insiders": len(mal_lm),
                                          "bootstrap": a.boot, "seed": a.seed}, "variants": {}}
    for name, cols in variants.items():
        scaler = StandardScaler().fit(train[cols].to_numpy(np.float64))
        Xtr = scaler.transform(train[cols].to_numpy(np.float64))
        Xev = scaler.transform(ev[cols].to_numpy(np.float64))
        full = PCA(svd_solver="full", random_state=a.seed).fit(Xtr)     # all directions; q is chosen below
        cum = np.cumsum(full.explained_variance_ratio_)
        q_of = lambda t: int(np.searchsorted(cum, t, side="right") + 1)  # sklearn's rule for a float n_components

        def recon_error(X, q):
            Z = (X - full.mean_) @ full.components_[:q].T               # scores, eq. (projection)
            return ((X - (Z @ full.components_[:q] + full.mean_)) ** 2).sum(axis=1)   # ||x - x_hat||^2

        q = q_of(a.pve)
        ref = PCA(n_components=a.pve, svd_solver="full", random_state=a.seed).fit(Xtr)  # the library's own truncation
        assert ref.n_components_ == q
        ev["e"] = recon_error(Xev, q)
        assert np.allclose(ev.e, ((Xev - ref.inverse_transform(ref.transform(Xev))) ** 2).sum(axis=1), rtol=1e-6, atol=1e-6)
        e_tr = recon_error(Xtr, q)
        umax = ev.groupby("user_id").e.max().to_dict()
        grp = {"heldout_benign_days": ev.split == "val", "malicious_days": ev.y == 1,
               "insider_benign_days": (ev.split == "eval") & (ev.y == 0)}
        v = {"p": len(cols), "p_varying_in_train": int((scaler.var_ > 0).sum()), "q": q, "pve": float(cum[q - 1]),
             "mse_train_days": float(e_tr.mean()), "median_e_train_days": float(np.median(e_tr))}
        # how much of each day's error comes from columns that never vary in training (no scale, no direction)
        const = scaler.var_ == 0
        if const.any():
            e_const = ((Xev[:, const] - rec_mean_const(full, const, q)) ** 2).sum(axis=1)
            nz = (Xev[:, const] != 0)
            v["train_constant_columns_nonzero_in_eval"] = {c: {"days": int(nz[:, j].sum()), "malicious_days": int(nz[ev.y.to_numpy() == 1, j].sum())}
                                                           for j, c in enumerate(np.array(cols)[const]) if nz[:, j].any()}
            v["share_of_error_from_train_constant_columns"] = {g: float(e_const[m.to_numpy()].sum() / ev.e[m].sum())
                                                               for g, m in (("heldout_benign_days", ev.split == "val"), ("malicious_days", ev.y == 1),
                                                                            ("insider_benign_days", (ev.split == "eval") & (ev.y == 0)))}
        for g, m in grp.items():
            v["mse_" + g] = float(ev.e[m].mean()); v["median_e_" + g] = float(ev.e[m].median())
            v["share_above_train_q99_" + g] = float((ev.e[m] > np.quantile(e_tr, 0.99)).mean())
        for pool, mal in (("all 70 insiders", mal_all), ("language-model pool (60 insiders)", mal_lm)):
            sub = ev[ev.user_id.isin(set(mal) | set(ben_users))]
            day_auc = float(roc_auc_score(sub.y, sub.e))
            u_auc = user_auc(umax, mal, ben_users)
            assert abs(u_auc - roc_auc_score([1] * len(mal) + [0] * len(ben_users), [umax[u] for u in mal + ben_users])) < 1e-12
            bs_u, bs_d = [], []
            groups = {u: g for u, g in sub.groupby("user_id")[["y", "e"]]}
            for _ in range(a.boot):
                mb = rng.choice(mal, len(mal)); bb = rng.choice(ben_users, len(ben_users))
                bs_u.append(user_auc(umax, list(mb), list(bb)))
            for _ in range(min(a.boot, 2000)):          # day-level AUC with users resampled (cluster bootstrap)
                us = list(rng.choice(mal, len(mal))) + list(rng.choice(ben_users, len(ben_users)))
                g = pd.concat([groups[u] for u in us])
                bs_d.append(roc_auc_score(g.y, g.e))
            v[pool] = {"user_auc": u_auc, "user_auc_ci95": [float(np.quantile(bs_u, 0.025)), float(np.quantile(bs_u, 0.975))],
                       "day_auc": day_auc, "day_auc_ci95": [float(np.quantile(bs_d, 0.025)), float(np.quantile(bs_d, 0.975))],
                       "malicious_users": len(mal), "benign_users": len(ben_users), "days": int(len(sub)), "malicious_days": int(sub.y.sum())}
        # sensitivity to the variance threshold (point estimates only)
        v["threshold_sensitivity"] = {}
        for t in (0.80, 0.90, 0.95, 0.99):
            qt = q_of(t); et = pd.Series(recon_error(Xev, qt), index=ev.index); um = et.groupby(ev.user_id).max().to_dict()
            v["threshold_sensitivity"][str(t)] = {"q": qt, "pve": float(cum[qt - 1]),
                **{pool: {"user_auc": user_auc(um, mal, ben_users),
                          "day_auc": float(roc_auc_score(ev.y[ev.user_id.isin(set(mal) | set(ben_users))],
                                                         et[ev.user_id.isin(set(mal) | set(ben_users))]))}
                   for pool, mal in (("all 70 insiders", mal_all), ("language-model pool (60 insiders)", mal_lm))}}
        res["variants"][name] = v
        print(name, json.dumps(v), flush=True)
    res["cpu_s"] = time.process_time() - c0; res["wall_s"] = time.time() - t0
    (out / "pca_reconstruction_r42.json").write_text(json.dumps(res, indent=1) + "\n")
    print("PCA_RECON_DONE", flush=True)


if __name__ == "__main__":
    main()
