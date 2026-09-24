#!/usr/bin/env python3
"""Regression checks for the feature-ranking gates.

The defect (found by the 2026-09-24 review, root-caused the same day): the
TWOS token-class re-runs pointed --frontier-dir at a benign-only SAE frontier.
That frontier ranked features on its own benign-only training rows, so every
malicious-minus-benign gap was NaN, and sorting NaN gaps returned features in
id order. Both runs therefore patched "top5 = [0, 1, 2, 3, 4]" against the
five most active features, with nothing raising.

These checks pin:
  1. the actual failure: a benign-only ranking population yields NaN gaps,
     and selection now refuses it instead of returning [0, 1, 2, 3, 4];
  2. the saved artifact shape (8,192 rows, no finite gap) is refused on load,
     and the fixed benign-only frontier writes no ranking file at all;
  3. a valid fixture with a known ordering selects exactly the known sets;
  4. a control set can satisfy its stated threshold rule while firing six
     times as often as the selected set, and the report says so;
  5. population, disjointness and manifest checks;
  6. the real reselect_token_sae_features.py CLI end to end: it refuses a
     discovery population with no positive rows, and on a valid one writes a
     finite ranking and a complete manifest.

Usage (CPU only; needs torch, pandas, pyarrow):
    python scripts/tests/test_feature_ranking_gates.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from sae_core import (  # noqa: E402
    RankingUndefinedError,
    TopKSAE,
    add_active_control_feature_sets,
    check_disjoint_users,
    choose_feature_sets,
    evaluate_features,
    feature_set_criteria,
    load_ranking,
    validate_ranking,
    validate_ranking_population,
)

FAIL: list[str] = []
CPU = torch.device("cpu")


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def raises(fn, exc=RankingUndefinedError) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception as other:  # the wrong exception is a failure too
        print(f"      unexpected {type(other).__name__}: {other}")
        return False
    return False


def identity_sae(d: int) -> TopKSAE:
    """z = relu(x): activations equal the (non-negative) inputs exactly."""
    model = TopKSAE(d, d, d)
    with torch.no_grad():
        model.encoder.weight.copy_(torch.eye(d))
        model.encoder.bias.zero_()
        model.decoder.weight.copy_(torch.eye(d))
    return model


def ranking_frame(n: int, *, gaps=None, active=None) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    gaps = rng.normal(0, 0.01, n) if gaps is None else np.asarray(gaps, dtype=float)
    active = rng.uniform(0.001, 0.02, n) if active is None else np.asarray(active, dtype=float)
    neg = np.full(n, 0.05)
    return pd.DataFrame({
        "feature_id": np.arange(n), "row_gap": gaps, "row_support_gap": gaps / 10,
        "row_mean_pos": neg + gaps, "row_mean_neg": neg, "row_active_frac": active,
    })


print("\n1. the actual failure: ranking on a benign-only population")
rng = np.random.default_rng(1)
x = rng.uniform(0.0, 1.0, size=(16, 8)).astype(np.float32)
y = np.zeros(16, dtype=np.int64)
fd, st = evaluate_features(identity_sae(8), x, y, device=CPU, batch_size=4)
check("row_gap is NaN for every feature (undefined, not zero)", bool(fd["row_gap"].isna().all()))
check("row_mean_pos is NaN, not replaced by zero", bool(fd["row_mean_pos"].isna().all()))
check("the stats record zero positive rows", st["n_pos_rows"] == 0 and st["n_neg_rows"] == 16)
old = [int(i) for i in fd.sort_values("row_gap", ascending=False).head(5)["feature_id"]]
check("without a gate, sorting NaN gaps returns [0, 1, 2, 3, 4] (the TWOS selection)", old == [0, 1, 2, 3, 4], str(old))
check("choose_feature_sets now refuses it", raises(lambda: choose_feature_sets(fd)))
check("add_active_control_feature_sets refuses it", raises(lambda: add_active_control_feature_sets({}, fd, min_active_frac=0.002)))
check("the population gate refuses a benign-only population", raises(lambda: validate_ranking_population(y)))

print("\n2. the saved artifact shape, and the fixed benign-only frontier")
art = ranking_frame(8192)
for col in ("row_gap", "row_support_gap", "row_mean_pos"):
    art[col] = np.nan
check("an 8,192-row file with no finite gap is refused", raises(lambda: validate_ranking(art)))
with tempfile.TemporaryDirectory() as td:
    cfg = Path(td) / "layer_24" / "m04_k08"
    cfg.mkdir(parents=True)
    art.to_csv(cfg / "delta_sae_top_features.csv", index=False)
    check("load_ranking refuses the saved NaN ranking file", raises(lambda: load_ranking(cfg)))
    import train_delta_sae_frontier as frontier  # noqa: E402
    frontier.write_benign_only_outputs(cfg, fd, st)
    check("the fixed frontier removes the stale ranking file", not (cfg / "delta_sae_top_features.csv").exists())
    check("it writes the marker and benign activity statistics",
          (cfg / frontier.RANKING_MARKER_FILE).exists() and (cfg / frontier.BENIGN_ACTIVITY_FILE).exists())
    check("a consumer pointed at it fails instead of selecting", raises(lambda: load_ranking(cfg)))
    mixed = dict(st, n_pos_rows=3)
    check("benign-only outputs refuse a population with positives",
          raises(lambda: frontier.write_benign_only_outputs(cfg, fd, mixed), ValueError))

print("\n3. a valid fixture with a known ordering")
gap_by_id = {5: 0.9, 2: 0.8, 7: 0.6, 0: 0.4, 3: 0.2, 4: 0.01, 6: -0.05, 1: 0.0}
base = np.full(8, 0.1, dtype=np.float32)
base[1] = 0.0                                     # feature 1 never fires
pos_row = base + np.array([gap_by_id[i] for i in range(8)], dtype=np.float32)
xv = np.vstack([np.tile(pos_row, (4, 1)), np.tile(base, (4, 1))]).astype(np.float32)
yv = np.array([1, 1, 1, 1, 0, 0, 0, 0])
fv, sv = evaluate_features(identity_sae(8), xv, yv, device=CPU, batch_size=3)
got = dict(zip(fv["feature_id"].astype(int), fv["row_gap"]))
check("every gap equals its constructed value",
      all(abs(got[i] - g) < 1e-6 for i, g in gap_by_id.items()), str({i: round(v, 4) for i, v in got.items()}))
sets = choose_feature_sets(fv)
check("top5 is the known order [5, 2, 7, 0, 3]", sets["top5"] == [5, 2, 7, 0, 3], str(sets["top5"]))
check("control1 is the lowest-|gap| active non-top feature [4]", sets["control1"] == [4], str(sets["control1"]))
sets = add_active_control_feature_sets(sets, fv, min_active_frac=0.5, sizes=(1, 2, 3))
check("control2_active is [4, 6]", sets.get("control2_active") == [4, 6], str(sets.get("control2_active")))
check("control3_active is not formed: only two features are eligible", "control3_active" not in sets)
crit = feature_set_criteria(fv, sets, control_sets=("control2_active",), min_active_frac=0.5)
c2 = crit["sets"]["control2_active"]
check("the report confirms the top set is the top of the gap ranking", crit["sets"]["top5"]["gap_rank_check"])
check("control2_active matches its stated rule and threshold", c2["matches_stated_rule"] and c2["meets_min_active"])

print("\n4. a threshold rule is not activity matching")
xa = np.zeros((10, 4), dtype=np.float32)
xa[0, 0] = xa[0, 1] = 1.0                          # selected: fire on one positive row
xa[[0, 1, 2, 5, 6, 7], 2] = 0.5                    # controls: fire on 3 positive + 3 benign rows
xa[[0, 1, 2, 5, 6, 7], 3] = 0.5
ya = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
fa, _ = evaluate_features(identity_sae(4), xa, ya, device=CPU, batch_size=4)
ca = feature_set_criteria(fa, {"top5": [0, 1], "control2_active": [2, 3]}, control_sets=("control2_active",),
                          min_active_frac=0.05)["sets"]["control2_active"]
check("the controls satisfy the stated threshold rule", ca["matches_stated_rule"] and ca["meets_min_active"])
check("yet they fire six times as often as the selected set",
      abs(ca["activity_ratio_to_top"] - 6.0) < 1e-9, f"ratio={ca['activity_ratio_to_top']}")
check("the report labels the criterion threshold-only", ca["activity_criterion"] == "threshold_only")

print("\n5. population, disjointness and manifest checks")
check("both classes present passes", validate_ranking_population([0, 1, 0]) == (1, 2))
check("positives only is refused", raises(lambda: validate_ranking_population([1, 1])))
check("disjoint discovery/evaluation users pass", check_disjoint_users(["U1", "U2"], ["U3"]) is None)
check("overlapping users are refused", raises(lambda: check_disjoint_users(["U1", "U2"], ["U2", "U4"])))
import eval_token_delta_sae_causal as causal  # noqa: E402
rf = ranking_frame(40, gaps=np.linspace(0.05, -0.01, 40), active=np.full(40, 0.01))
fs = add_active_control_feature_sets(choose_feature_sets(rf), rf, min_active_frac=0.002)
with tempfile.TemporaryDirectory() as td:
    cfg = Path(td)
    rf.to_csv(cfg / "delta_sae_top_features.csv", index=False)
    (cfg / "reselect_summary.json").write_text(json.dumps({"discovery_users": ["U1", "U2"]}))
    kw = dict(top_sets=["top5"], control_set="control5_active", min_active_frac=0.002)
    m_in = causal.build_selection_manifest(cfg, rf, fs, receiver_user_file=None, receiver_users=None, **kw)
    check("receivers = all positives is recorded as in-sample", m_in["held_out_from_selection"] is False)
    m_out = causal.build_selection_manifest(cfg, rf, fs, receiver_user_file=Path("conf.txt"),
                                            receiver_users={"U3", "U4"}, **kw)
    check("disjoint receivers are recorded as held out", m_out["held_out_from_selection"] is True)
    check("the manifest carries the ranking hash and the chosen ids",
          len(m_out["ranking_file_sha256"]) == 64 and m_out["top_sets"]["top5"] == fs["top5"])
    check("receivers that overlap discovery users are refused",
          raises(lambda: causal.build_selection_manifest(cfg, rf, fs, receiver_user_file=Path("conf.txt"),
                                                        receiver_users={"U2", "U9"}, **kw)))

print("\n6. reselect_token_sae_features.py end to end")
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    ext, src = td / "extract", td / "frontier" / "layer_24" / "m01_k08"
    (ext / "layer_24").mkdir(parents=True)
    src.mkdir(parents=True)
    d = 8
    users = [f"U{i}" for i in range(6)]
    rows, meta, rng = [], [], np.random.default_rng(3)
    for ex in range(36):
        u = users[ex % 6]
        yl = int(u in {"U0", "U1", "U2"} and (ex // 6) % 2 == 0)   # three attack users, 3 positive examples each
        meta.append({"example_idx": ex, "user_id": u, "y": yl})
        v = rng.uniform(0, 0.2, size=(5, d)).astype(np.float32)
        if yl:
            v[:, 6] += 1.0                                    # feature 6 marks attacks
        rows.append(v)
    pd.DataFrame(meta).to_parquet(ext / "example_scores.parquet", index=False)
    torch.save({"delta": np.vstack(rows), "example_idx": np.repeat(np.arange(36), 5),
                "position": np.tile(np.arange(5), 36)}, ext / "layer_24" / "chunk_00000.pt")
    model = identity_sae(d)
    torch.save({"state_dict": model.state_dict(), "layer": 24, "unit": "token", "d_in": d, "d_latent": d,
                "k": d, "x_mean": np.zeros((1, d), np.float32), "x_std": np.ones((1, d), np.float32)},
               src / "delta_sae_model.pt")
    (td / "no_pos.txt").write_text("U3\nU4\n")
    (td / "disc.txt").write_text("U0\nU1\n")
    base_cmd = [sys.executable, str(SCRIPTS / "reselect_token_sae_features.py"), "--extract-dir", str(ext),
                "--data-dir", str(td), "--frontier-dir", str(td / "frontier"), "--layer", "24",
                "--latent-mult", "1", "--k", "8", "--benign-sample-prob", "1.0", "--device", "cpu",
                "--active-control-min-frac", "0.0"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(SCRIPTS)}
    bad = subprocess.run(base_cmd + ["--out-frontier-dir", str(td / "bad"), "--discovery-user-file", str(td / "no_pos.txt")],
                         capture_output=True, text=True, env=env)
    check("a discovery population with no positive rows is refused",
          bad.returncode != 0 and "RankingUndefinedError" in bad.stderr, bad.stderr.strip().splitlines()[-1] if bad.stderr else "")
    check("and no ranking file is left behind",
          not (td / "bad" / "layer_24" / "m01_k08" / "delta_sae_top_features.csv").exists())
    good = subprocess.run(base_cmd + ["--out-frontier-dir", str(td / "good"), "--discovery-user-file", str(td / "disc.txt"),
                                      "--exclude-benign-user-file", str(td / "no_pos.txt")],
                          capture_output=True, text=True, env=env)
    check("a valid discovery population succeeds", good.returncode == 0, good.stderr.strip()[-300:])
    if good.returncode == 0:
        out = td / "good" / "layer_24" / "m01_k08"
        rk = load_ranking(out)
        summ = json.loads((out / "reselect_summary.json").read_text())
        check("the written ranking is finite and puts the attack feature first",
              int(rk.sort_values("row_gap", ascending=False)["feature_id"].iloc[0]) == 6)
        check("the manifest records discovery users with positive rows",
              summ["n_discovery_users_with_positive_rows"] == 2 and summ["discovery_users"] == ["U0", "U1"])
        check("the manifest records the excluded users' benign rows as dropped",
              summ["row_stats"]["n_benign_rows_dropped_excluded_users"] == 60, str(summ["row_stats"]))
        check("the manifest carries hashes and the control criteria",
              len(summ["ranking_file_sha256"]) == 64 and "control1" in summ["feature_set_criteria"]["sets"])

print()
if FAIL:
    print(f"{len(FAIL)} check(s) FAILED: {FAIL}")
    sys.exit(1)
print("all checks passed")
