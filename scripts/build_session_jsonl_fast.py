#!/usr/bin/env python3
"""Fast, byte-faithful drop-in for build_session_jsonl.py.

Produces IDENTICAL output to build_session_jsonl.py but replaces the slow
per-group pandas `groupby(...)` iteration (which materializes ~1.39M sub-DataFrames)
with a single linear pass over the sorted frame using column numpy arrays.

Fidelity is guaranteed by REUSING the original serialization/splitting helpers
(serialize_text, assign_split, sanitize_frame, resolve_user_map, DAY_CONTEXT_COLS,
SESSION_COLS) and by replicating the original's itertuples() column-rename behavior:
columns whose names are not valid Python identifiers (file_n-to_usb1, file_n-from_usb1,
file_n-file_act3, file_n-disk1) are dropped from the per-session dicts, exactly as the
original does. See build_session_jsonl.py.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

from remote_common import dump_json, ensure_dir, read_session_shards, load_user_map, write_jsonl
from build_session_jsonl import (
    DAY_CONTEXT_COLS,
    SESSION_COLS,
    assign_split,
    resolve_user_map,
    sanitize_frame,
    serialize_text,
)


def _effective_session_cols(present_cols: list[str]) -> list[str]:
    """Mimic pandas DataFrame.itertuples() name handling: it builds a namedtuple
    with rename=True, so invalid identifiers / dups / leading-underscore names get
    renamed to positional `_N` and are therefore dropped by the original's
    `if col in row_dict` check. Keep only the names that survive unchanged."""
    nt = collections.namedtuple("S", present_cols, rename=True)
    return [c for c, f in zip(present_cols, nt._fields) if c == f]


def build_examples_fast(df: pd.DataFrame, labels: pd.DataFrame, val_frac: float, max_sessions: int):
    labels = labels[["user_id", "day_index", "y"]].drop_duplicates()
    label_map = {(str(r.user_id), int(r.day_index)): int(r.y) for r in labels.itertuples(index=False)}
    positive_users = set(labels.loc[labels["y"] > 0, "user_id"].astype(str))

    df = df.sort_values(["user_id", "day_index", "starttime", "sessionid"]).reset_index(drop=True)
    n = len(df)
    if n == 0:
        return [], pd.DataFrame([])

    present_context = [c for c in DAY_CONTEXT_COLS if c in df.columns]
    present_session = [c for c in SESSION_COLS if c in df.columns]
    eff_session = _effective_session_cols(present_session)

    uid = df["user_id"].to_numpy()
    day = df["day_index"].to_numpy()
    ctx_arrays = {c: df[c].to_numpy() for c in present_context}
    sess_arrays = {c: df[c].to_numpy() for c in eff_session}

    # contiguous-group boundaries on the sorted frame (same grouping as groupby(sort=False))
    same = (uid[1:] == uid[:-1]) & (day[1:] == day[:-1])
    starts = np.concatenate(([0], np.flatnonzero(~same) + 1, [n]))

    rows: list[dict] = []
    meta_rows: list[dict] = []
    for gi in range(len(starts) - 1):
        s = int(starts[gi]); e = int(starts[gi + 1])
        user_id = str(uid[s]); day_index = int(day[s])
        n_total = e - s
        kept = min(n_total, max_sessions)
        y = int(label_map.get((user_id, day_index), 0))
        split = assign_split(user_id, positive_users, val_frac=val_frac)
        context = {c: ctx_arrays[c][s] for c in present_context}
        sessions = [{c: sess_arrays[c][i] for c in eff_session} for i in range(s, s + kept)]
        text = serialize_text(context, sessions, total_sessions=int(n_total))
        example_id = f"{user_id}:{day_index}"
        rows.append({
            "example_id": example_id,
            "user_id": user_id,
            "day_index": day_index,
            "split": split,
            "y": y,
            "n_sessions_total": int(n_total),
            "n_sessions_kept": int(kept),
            "context": context,
            "sessions": sessions,
            "text": text,
        })
        meta_rows.append({
            "example_id": example_id,
            "user_id": user_id,
            "day_index": day_index,
            "split": split,
            "y": y,
            "n_sessions_total": int(n_total),
            "n_sessions_kept": int(kept),
            "text_chars": int(len(text)),
        })
    return rows, pd.DataFrame(meta_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--user-map", type=Path, default=None)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--max-sessions", type=int, default=24)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    labels_path = args.labels or (args.input_dir / "labels_daily.parquet")
    user_map_path = resolve_user_map(args.input_dir, args.user_map)

    user_map = load_user_map(user_map_path)
    print(f"[build_fast] loaded user map: {len(user_map)} entries", flush=True)
    raw_df = read_session_shards(args.input_dir)
    print(f"[build_fast] loaded raw sessions: {len(raw_df)} rows", flush=True)
    raw_df = sanitize_frame(raw_df, user_map)
    print(f"[build_fast] sanitized: {raw_df['user_id'].nunique()} users, {raw_df[['user_id','day_index']].drop_duplicates().shape[0]} user-days", flush=True)
    labels = pd.read_parquet(labels_path)
    print(f"[build_fast] loaded labels: {len(labels)} rows", flush=True)

    examples, meta_df = build_examples_fast(raw_df, labels, val_frac=args.val_frac, max_sessions=args.max_sessions)
    print(f"[build_fast] built examples: {len(meta_df)}", flush=True)

    train_rows = [r for r in examples if r["split"] == "train"]
    val_rows = [r for r in examples if r["split"] == "val"]
    eval_rows = [r for r in examples if r["split"] in {"val", "eval"}]

    write_jsonl(out_dir / "all.jsonl", examples)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)
    write_jsonl(out_dir / "eval.jsonl", eval_rows)
    meta_df.to_parquet(out_dir / "example_metadata.parquet", index=False)
    meta_df.to_csv(out_dir / "example_metadata.csv", index=False)

    split_counts = meta_df.groupby("split").size().to_dict()
    pos_counts = meta_df.groupby("split")["y"].sum().to_dict()
    summary = {
        "input_dir": str(args.input_dir),
        "user_map": str(user_map_path),
        "labels": str(labels_path),
        "out_dir": str(out_dir),
        "n_examples": int(len(meta_df)),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "positive_counts": {k: int(v) for k, v in pos_counts.items()},
        "max_sessions": int(args.max_sessions),
        "val_frac": float(args.val_frac),
        "builder": "build_session_jsonl_fast",
    }
    dump_json(out_dir / "build_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
