#!/usr/bin/env python3
"""Build the per-user-day numeric matrix that a time-series detector consumes.

Why this exists: the paper's Discussion predicts that a sequence model which
shares the sequence-prediction setting but not the pretrained language
representation "should fall between the tabular detectors and the LM in profile
reliance". Testing that needs a NON-language detector fed the *same*
information as both, so the comparison isolates architecture/objective rather
than input.

This script therefore reproduces `results/cross_arch_probe/cross_arch_probe.py:
load_days` exactly — profile columns take the first value per user-day,
behavioural columns take the mean, plus `n_sessions` — and additionally

  * keeps rows ordered per user by `day_index`, so each user is a multivariate
    time series over their days rather than a bag of rows;
  * marks every channel as PROFILE or BEHAV, which is what makes the
    token-class score decomposition transfer: for a reconstruction detector the
    score is a mean over channels, so

        s_full = (N_P * s_P + N_B * s_B) / (N_P + N_B)

    is the same identity used for token classes, and a "behaviour-channel-only"
    score is the same exact conditional, not a subtraction of means;
  * assigns splits with the LM branch's own deterministic user hash
    (`stable_hash_frac`, val_frac 0.10, positive users reserved for eval), so
    the populations line up with the language-model results instead of using a
    fresh random split.

Profile channels are constant within a user by construction, which is precisely
the property that makes them an identity shortcut for a model that is rewarded
for reconstructing them.

Output: an .npz with X (float32, rows x channels), the channel names and their
classes, and per-row user_id / day_index / y / split; plus a JSON manifest with
verification counts.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

# PROFILE channels: the LM's DAY/PSY block minus `week` (which is temporal, not
# a static attribute). Identical to cross_arch_probe.py's PROFILE_COLS.
PROFILE_COLS = ["project", "role", "b_unit", "f_unit", "dept", "team", "ITAdmin", "O", "C", "E", "A", "N"]

# BEHAV channels: exactly the fields the LM's SES lines carry
# (`build_session_jsonl.SESSION_COLS`), so the time-series detector is given the
# same information as the language model rather than a superset.
SESSION_COLS = [
    "pc", "isworkhour", "isafterhour", "isweekend", "isweekendafterhour",
    "duration", "n_concurrent_sessions", "start_with", "end_with",
    "ses_start", "ses_end", "n_allact", "n_logon", "n_usb", "usb_mean_usb_dur",
    "n_file", "file_n-to_usb1", "file_n-from_usb1", "file_n-file_act3",
    "file_n-disk1", "file_n_exef", "n_email", "email_n_recvmail",
    "email_n_send_mail", "email_mean_n_atts", "email_mean_e_att_comp",
    "n_http", "http_n_jobf", "http_n_cloudf", "http_n_leakf", "http_n_hackf",
]

# Never a feature. `user` is the raw LC-DAL integer user code: it survives the
# rename to `user_id` and is numeric, so a naive numeric selection would hand
# the model an explicit identity channel that the LM never sees — which would
# decide the experiment by construction. `day`/`day_index`, the timestamps,
# `week`, `insider` and the label are bookkeeping.
EXCLUDE = {
    "user", "user_id", "day", "day_index", "starttime", "endtime",
    "sessionid", "week", "insider", "y", "n_days",
}


def stable_hash_frac(text: str) -> float:
    """The LM branch's split hash (`remote_common.stable_hash_frac`)."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16 ** 16 - 1)


def assign_split(user_id: str, positive_users: set, val_frac: float) -> str:
    if user_id in positive_users:
        return "eval"
    return "val" if stable_hash_frac(user_id) < val_frac else "train"


def sanitize(df: pd.DataFrame, user_map_path: str) -> pd.DataFrame:
    """Replicates `build_session_jsonl.sanitize_frame`: map the integer LC-DAL
    user code to the CERT user id and rename `day` -> `day_index`."""
    if "user" not in df.columns or "day" not in df.columns:
        raise SystemExit("expected raw LC-DAL shards with 'user' and 'day' columns")
    umap = pd.read_csv(user_map_path)
    if not {"user_code", "user_id"}.issubset(umap.columns):
        raise SystemExit(f"{user_map_path} missing user_code/user_id")
    m = {int(r.user_code): str(r.user_id) for r in umap.itertuples(index=False)}
    df = df.copy()
    df["user_id"] = df["user"].map(m)
    missing = int(df["user_id"].isna().sum())
    if missing:
        raise SystemExit(f"failed to map {missing} LC-DAL user codes")
    return df.rename(columns={"day": "day_index"})


def load_days(input_dir: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    shards = sorted(glob.glob(os.path.join(input_dir, "session*_shard_*.csv.gz")))
    if not shards:
        raise SystemExit(f"no session shards under {input_dir}")
    frames = [pd.read_csv(s, compression="gzip", low_memory=False) for s in shards]
    df = pd.concat(frames, ignore_index=True)
    umaps = sorted(glob.glob(os.path.join(input_dir, "session*_user_map.csv")))
    if not umaps:
        raise SystemExit(f"no session*_user_map.csv under {input_dir}")
    df = sanitize(df, umaps[0])
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    prof = [c for c in PROFILE_COLS if c in num and c not in EXCLUDE]
    behav = [c for c in SESSION_COLS if c in num and c not in EXCLUDE]
    dropped = sorted(set(num) - set(prof) - set(behav) - EXCLUDE)
    if dropped:
        print(f"[note] numeric columns present but not in the LM field set, excluded: {dropped}")
    agg = {c: "first" for c in prof}
    agg.update({c: "mean" for c in behav})
    days = df.groupby(["user_id", "day_index"], as_index=False).agg(agg)
    nses = df.groupby(["user_id", "day_index"]).size().rename("n_sessions").reset_index()
    days = days.merge(nses, on=["user_id", "day_index"])
    behav = behav + ["n_sessions"]
    labels = pd.read_parquet(os.path.join(input_dir, "labels_daily.parquet"))
    labels = labels[["user_id", "day_index", "y"]].drop_duplicates()
    days = days.merge(labels, on=["user_id", "day_index"], how="left")
    days["y"] = days["y"].fillna(0).astype(int)
    # Split assignment must use the ANSWER-KEY positive users, exactly as
    # `build_session_jsonl` does, not the users who still have a positive day
    # after the merge to the matched session domain. On r6.2 the answer key has
    # 5 positive users but only 4 of them retain a positive user-day in that
    # domain, so deriving the set post-merge would move the fifth (PLJ1771)
    # into train/val and silently desynchronise the populations from the LM
    # branch. Verified: with the answer-key set, train/val/eval reproduce the
    # LM's 1,251,225 / 140,711 rows and 3,590 / 405 users.
    answer_key_positive_users = set(labels.loc[labels["y"] > 0, "user_id"].astype(str))
    return days, prof, behav, answer_key_positive_users


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--val-frac", type=float, default=0.10)
    args = ap.parse_args()

    days, prof, behav, pos_users = load_days(args.input_dir)
    days["user_id"] = days["user_id"].astype(str)
    days = days.sort_values(["user_id", "day_index"]).reset_index(drop=True)

    days["split"] = [assign_split(u, pos_users, args.val_frac) for u in days["user_id"]]
    matched_pos_users = set(days.loc[days["y"] == 1, "user_id"])

    channels = prof + behav
    classes = ["PROFILE"] * len(prof) + ["BEHAV"] * len(behav)
    X = days[channels].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # profile channels must be constant within a user; verify rather than assume
    const_frac = []
    for i, c in enumerate(channels):
        if classes[i] != "PROFILE":
            continue
        g = days.groupby("user_id")[c].nunique()
        const_frac.append(float((g <= 1).mean()))
    profile_constant_frac = float(np.mean(const_frac)) if const_frac else float("nan")

    np.savez_compressed(
        args.out,
        X=X,
        channels=np.array(channels),
        channel_classes=np.array(classes),
        user_id=days["user_id"].to_numpy(),
        day_index=days["day_index"].to_numpy(),
        y=days["y"].to_numpy(dtype=np.int64),
        split=days["split"].to_numpy(),
    )

    manifest = {
        "tag": args.tag,
        "input_dir": args.input_dir,
        "rows": int(len(days)),
        "users": int(days["user_id"].nunique()),
        "positive_rows": int(days["y"].sum()),
        "answer_key_positive_users": int(len(pos_users)),
        "positive_users_with_a_matched_positive_day": int(len(matched_pos_users)),
        "eval_users_without_a_matched_positive_day": sorted(pos_users - matched_pos_users),
        "n_profile_channels": len(prof),
        "n_behav_channels": len(behav),
        "profile_channels": prof,
        "split_rows": days["split"].value_counts().to_dict(),
        "split_users": days.groupby("split")["user_id"].nunique().to_dict(),
        "profile_channel_constant_within_user_frac": profile_constant_frac,
        "aggregation": "profile=first, behavioural=mean, plus n_sessions",
        "channel_policy": "PROFILE = cross_arch_probe PROFILE_COLS (the LM DAY/PSY block minus week); BEHAV = build_session_jsonl.SESSION_COLS, so the detector sees the same fields as the language model",
        "excluded": sorted(EXCLUDE),
        "split_rule": "LM branch's stable_hash_frac; positive users reserved for eval; val_frac="
        + str(args.val_frac),
    }
    with open(os.path.splitext(args.out)[0] + "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2)[:1800])
    print(f"\nwrote {args.out}  X={X.shape}")


if __name__ == "__main__":
    main()
