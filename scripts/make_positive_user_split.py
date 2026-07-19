#!/usr/bin/env python3
"""Deterministic discovery/confirmation split of positive (malicious) users.

Used to separate mechanism discovery (feature ranking, config choice) from
confirmation (causal/necessity effect estimation) at the malicious-user level.

Reads example_metadata.csv from a session JSONL build directory, takes the
positive users in the example domain, and splits them by seeded shuffle.
With --louo it instead writes one confirmation file per held-out user
(training/discovery file = the other users), for small-population datasets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--discovery-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--louo", action="store_true")
    args = ap.parse_args()

    meta = pd.read_csv(args.data_dir / "example_metadata.csv")
    pos_users = sorted(meta.loc[meta["y"] == 1, "user_id"].astype(str).unique())
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "data_dir": str(args.data_dir),
        "n_positive_users": len(pos_users),
        "seed": args.seed,
        "mode": "louo" if args.louo else "split",
    }

    if args.louo:
        for user in pos_users:
            rest = [u for u in pos_users if u != user]
            (out_dir / f"louo_{user}_discovery_users.txt").write_text("\n".join(rest) + "\n")
            (out_dir / f"louo_{user}_confirmation_users.txt").write_text(user + "\n")
        summary["folds"] = pos_users
    else:
        rng = np.random.default_rng(args.seed)
        shuffled = list(pos_users)
        rng.shuffle(shuffled)
        n_disc = max(1, int(round(len(shuffled) * args.discovery_frac)))
        discovery = sorted(shuffled[:n_disc])
        confirmation = sorted(shuffled[n_disc:])
        (out_dir / "discovery_users.txt").write_text("\n".join(discovery) + "\n")
        (out_dir / "confirmation_users.txt").write_text("\n".join(confirmation) + "\n")
        summary["n_discovery"] = len(discovery)
        summary["n_confirmation"] = len(confirmation)
        # positive-day counts per side, for the record
        day_counts = meta.loc[meta["y"] == 1].groupby(meta["user_id"].astype(str)).size()
        summary["discovery_positive_days"] = int(day_counts.reindex(discovery).fillna(0).sum())
        summary["confirmation_positive_days"] = int(day_counts.reindex(confirmation).fillna(0).sum())

    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
