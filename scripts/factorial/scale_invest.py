import pandas as pd, os
pd.set_option("display.width", 200)
MODES = ["full","no_psy","no_profile","shuffle_profile"]
for scale, P in [("3B","/anvil/scratch/x-bbhusal1/p1"), ("8B","/anvil/scratch/x-bbhusal1/p1_8b")]:
    print(f"\n################ {scale} ################")
    rows = []
    for m in MODES:
        f = f"{P}/fold_{m}/fold_aligned_detector_rows.csv"
        if not os.path.exists(f):
            print("missing", f); continue
        d = pd.read_csv(f); d = d[d.score_name == "adapted_nll"].copy(); d["mode"] = m; rows.append(d)
    df = pd.concat(rows)
    print("-- mean over 4 folds: day_roc / user_roc / heldout_user_rank (of n_test_users) --")
    g = df.groupby("mode")[["day_roc_auc","user_roc_auc","heldout_user_rank","n_test_users"]].mean().round(3)
    print(g.reindex(MODES).to_string())
    print("-- per held-out malicious user: user_roc_auc --")
    print(df.pivot(index="heldout_pos_user", columns="mode", values="user_roc_auc").reindex(columns=MODES).round(3).to_string())
    print("-- per held-out malicious user: day_roc_auc --")
    print(df.pivot(index="heldout_pos_user", columns="mode", values="day_roc_auc").reindex(columns=MODES).round(3).to_string())
    print("-- per held-out malicious user: rank of that user among test users (1 = most anomalous) --")
    print(df.pivot(index="heldout_pos_user", columns="mode", values="heldout_user_rank").reindex(columns=MODES).to_string())
    # what splits exist in the scores parquet (to see if a 'seen' benign pool exists for a seen-vs-unseen gap)
    sp = pd.read_parquet(f"{P}/deltas_full/example_scores.parquet", columns=["split","y","user_id"])
    print("-- scores parquet: rows by split x y --")
    print(sp.groupby(["split","y"]).size().to_string())
    print("   users by split:", sp.groupby("split").user_id.nunique().to_dict())
