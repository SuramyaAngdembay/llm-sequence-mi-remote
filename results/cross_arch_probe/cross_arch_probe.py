#!/usr/bin/env python3
"""Cross-architecture attribution probe on CERT (classical detectors).

Question: is r6.2 profile capture a dataset property or an LM-adaptation
property? Train classical one-class detectors (IsolationForest, OneClassSVM,
PCA-reconstruction) benign-only on per-user-day features WITH profile columns
included, then measure (a) permutation-importance mass on profile vs
behavioral columns (classical analog of top-5 token attribution) and (b) a
no-profile retrain ablation, under user-disjoint evaluation. Run on both
r6.2 and r4.2; the population-structure account predicts profile reliance on
r6.2 and predominantly behavioral reliance on r4.2.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROFILE_COLS = ["project","role","b_unit","f_unit","dept","team","ITAdmin","O","C","E","A","N"]
EXCLUDE = {"user_id","day_index","starttime","sessionid","week","insider","y"}

def load_days(input_dir):
    shards = sorted(glob.glob(os.path.join(input_dir, "session*_shard_*.csv.gz")))
    frames = [pd.read_csv(s, compression="gzip") for s in shards]
    df = pd.concat(frames, ignore_index=True)
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in num if c not in EXCLUDE]
    prof = [c for c in feat_cols if c in PROFILE_COLS]
    behav = [c for c in feat_cols if c not in PROFILE_COLS]
    agg = {c: "first" for c in prof}
    agg.update({c: "mean" for c in behav})
    days = df.groupby(["user_id","day_index"], as_index=False).agg(agg)
    nses = df.groupby(["user_id","day_index"]).size().rename("n_sessions").reset_index()
    days = days.merge(nses, on=["user_id","day_index"])
    behav = behav + ["n_sessions"]
    labels = pd.read_parquet(os.path.join(input_dir, "labels_daily.parquet"))
    labels = labels[["user_id","day_index","y"]].drop_duplicates()
    days = days.merge(labels, on=["user_id","day_index"], how="left")
    days["y"] = days["y"].fillna(0).astype(int)
    return days, prof, behav

def roc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s)+1)
    r = pd.DataFrame({"s": s, "r": r}).groupby("s")["r"].transform("mean").to_numpy()
    npos, nneg = int((y==1).sum()), int((y==0).sum())
    return float((r[y==1].sum() - npos*(npos+1)/2) / (npos*nneg)) if npos and nneg else float("nan")

def fit_models(Xtr, rng):
    m = {}
    m["IsolationForest"] = IsolationForest(n_estimators=300, random_state=42).fit(Xtr)
    sub = Xtr[rng.choice(len(Xtr), min(6000, len(Xtr)), replace=False)]
    m["OneClassSVM"] = OneClassSVM(nu=0.05, gamma="scale").fit(sub)
    m["PCA"] = PCA(n_components=min(10, Xtr.shape[1]-1)).fit(Xtr)
    return m

def scores(m, name, X):
    if name == "PCA":
        rec = m.inverse_transform(m.transform(X)); return np.linalg.norm(X-rec, axis=1)
    return -m.score_samples(X) if name == "IsolationForest" else -m.decision_function(X).ravel()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    days, prof, behav = load_days(args.input_dir)
    feat = prof + behav
    print(f"[{args.tag}] days={len(days)} users={days.user_id.nunique()} "
          f"profile_cols={len(prof)} behav_cols={len(behav)} positives={int(days.y.sum())}", flush=True)

    pos_users = set(days.loc[days.y==1, "user_id"])
    ben_users = sorted(set(days.user_id) - pos_users)
    rng.shuffle(ben_users)
    holdout = set(ben_users[:max(1, len(ben_users)//10)])
    train_users = set(ben_users) - holdout
    tr = days[days.user_id.isin(train_users)]
    ev = days[days.y.eq(1) | days.user_id.isin(holdout)]  # unseen-vs-unseen
    y = ev.y.to_numpy()

    report = {"tag": args.tag, "n_train_days": len(tr), "n_eval": len(ev),
              "n_pos": int(y.sum()), "n_holdout_users": len(holdout),
              "profile_cols": prof, "n_behav_cols": len(behav), "models": {}}

    for variant, cols in [("with_profile", feat), ("no_profile", behav)]:
        sc = StandardScaler().fit(tr[cols])
        Xtr, Xev = sc.transform(tr[cols]), sc.transform(ev[cols])
        models = fit_models(Xtr, rng)
        for name, m in models.items():
            base = scores(m, name, Xev)
            entry = report["models"].setdefault(name, {})
            entry[f"roc_{variant}"] = round(roc(y, base), 4)
            if variant == "with_profile":
                # permutation importance on discrimination (ROC drop), grouped
                imp = {}
                for j, c in enumerate(cols):
                    Xp = Xev.copy(); Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
                    imp[c] = max(0.0, entry["roc_with_profile"] - roc(y, scores(m, name, Xp)))
                tot = sum(imp.values()) or 1e-9
                pmass = sum(v for c, v in imp.items() if c in prof) / tot
                top5 = sorted(imp, key=imp.get, reverse=True)[:5]
                entry["profile_importance_mass"] = round(pmass, 3)
                entry["top5_features"] = top5
                entry["top5_profile_count"] = sum(1 for c in top5 if c in prof)
                print(f"[{args.tag}] {name}: ROC={entry['roc_with_profile']} "
                      f"profile_mass={pmass:.3f} top5={top5}", flush=True)
        print(f"[{args.tag}] variant={variant} done", flush=True)

    for name, e in report["models"].items():
        e["delta_roc_no_profile"] = round(e["roc_no_profile"] - e["roc_with_profile"], 4)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "profile_cols"}, indent=2), flush=True)

if __name__ == "__main__":
    main()
