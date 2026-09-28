#!/usr/bin/env python3
"""Check the day alignment between labels_daily.parquet and the LC-DAL session day index (2026-09-28).

The session extraction's `day` column and labels_daily's `day_index` may count days from different origins.
This script compares, per insider scenario, the session-days that the extraction itself flags as malicious
(`insider` > 0 on the session row) with the labelled days, under label shifts of -7..+7 days, and reports the
malicious user-day population that a join would produce with and without the best shift.

  python scripts/check_label_alignment.py --shards 'DIR/sessionr4.2_shard_*.csv.gz' --user-map MAP --labels LABELS \
      --out OUT.json
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from datetime import date, timedelta

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True); ap.add_argument("--user-map", required=True)
    ap.add_argument("--labels", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    um = pd.read_csv(a.user_map); code2id = dict(zip(um.user_code.astype(int), um.user_id.astype(str)))
    days, flagged = defaultdict(set), defaultdict(set); scen = {}
    for f in sorted(glob.glob(a.shards)):
        for ch in pd.read_csv(f, usecols=["user", "day", "insider", "starttime"], chunksize=500_000):
            ch.columns = [c.strip() for c in ch.columns]
            for u, d, s in zip(ch.user.astype(int), ch.day.astype(int), ch.insider.astype(float)):
                uid = code2id[u]; days[uid].add(d)
                if s > 0:
                    flagged[uid].add(d); scen[uid] = int(s)
    lab = pd.read_parquet(a.labels); lab = lab[lab.y > 0]
    sess = {(u, d) for u, ds in days.items() for d in ds}
    F = {(u, d) for u, ds in flagged.items() for d in ds}
    first_day = min(min(ds) for ds in days.values())
    res = {"session_day_range": [first_day, max(max(ds) for ds in days.values())], "labelled_malicious_user_days": int(len(lab)),
           "labelled_insiders": int(lab.user_id.nunique()), "flagged_session_days": len(F), "flagged_insiders": len(flagged),
           "flagged_by_scenario": {str(s): {"insiders": sum(1 for u in flagged if scen[u] == s),
                                            "session_days": sum(len(flagged[u]) for u in flagged if scen[u] == s)} for s in sorted(set(scen.values()))},
           "shift": {}}
    for delta in range(-7, 8):
        Ls = {(u, int(d) + delta) for u, d in zip(lab.user_id, lab.day_index)}
        by = {}
        for s in sorted(set(scen.values())):
            Fs = {(u, d) for u, d in F if scen[u] == s}
            by[str(s)] = len(Fs & Ls)
        hit = Ls & sess
        res["shift"][str(delta)] = {"flagged_days_matched": len(F & Ls), "matched_by_scenario": by,
                                    "joined_malicious_user_days": len(hit), "joined_malicious_insiders": len({u for u, _ in hit})}
    best = max(res["shift"], key=lambda k: res["shift"][k]["flagged_days_matched"])
    old = {(u, int(d)) for u, d in zip(lab.user_id, lab.day_index)} & sess
    new = {(u, int(d) + int(best)) for u, d in zip(lab.user_id, lab.day_index)} & sess
    res.update({"best_shift": int(best), "currently_joined_days_that_stay_malicious_after_best_shift": len(old & new),
                "currently_joined_days": len(old), "true_days_missing_from_current_join": len(new - old)})
    print(json.dumps(res, indent=1))
    open(a.out, "w").write(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
