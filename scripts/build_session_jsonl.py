#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from remote_common import dump_json, ensure_dir, fmt_num, load_user_map, read_session_shards, stable_hash_frac, write_jsonl


DAY_CONTEXT_COLS = ["week", "project", "role", "b_unit", "f_unit", "dept", "team", "ITAdmin", "O", "C", "E", "A", "N"]
SESSION_COLS = [
    "pc",
    "isworkhour",
    "isafterhour",
    "isweekend",
    "isweekendafterhour",
    "duration",
    "n_concurrent_sessions",
    "start_with",
    "end_with",
    "ses_start",
    "ses_end",
    "n_allact",
    "n_logon",
    "n_usb",
    "usb_mean_usb_dur",
    "n_file",
    "file_n-to_usb1",
    "file_n-from_usb1",
    "file_n-file_act3",
    "file_n-disk1",
    "file_n_exef",
    "n_email",
    "email_n_recvmail",
    "email_n_send_mail",
    "email_mean_n_atts",
    "email_mean_e_att_comp",
    "n_http",
    "http_n_jobf",
    "http_n_cloudf",
    "http_n_leakf",
    "http_n_hackf",
]


def resolve_user_map(input_dir: Path, user_map_path: Path | None) -> Path:
    if user_map_path and user_map_path.exists():
        return user_map_path
    candidates = sorted(input_dir.glob("session*_user_map.csv"))
    if len(candidates) == 1:
        return candidates[0]
    candidate = input_dir / "sessionr6.2_user_map.csv"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "Missing session*_user_map.csv. Transfer this mapping alongside the session shards."
    )


def sanitize_frame(df: pd.DataFrame, user_map: Dict[int, str]) -> pd.DataFrame:
    if "user" not in df.columns or "day" not in df.columns:
        raise ValueError("Expected raw LC-DAL session shards with 'user' and 'day' columns")
    df = df.copy()
    df["user_id"] = df["user"].map(user_map)
    missing = int(df["user_id"].isna().sum())
    if missing:
        raise ValueError(f"Failed to map {missing} LC-DAL user codes")
    df = df.rename(columns={"day": "day_index"})
    return df


def assign_split(user_id: str, positive_users: set[str], val_frac: float) -> str:
    if user_id in positive_users:
        return "eval"
    return "val" if stable_hash_frac(user_id) < val_frac else "train"


def serialize_text(context: Dict[str, Any], sessions: List[Dict[str, Any]], total_sessions: int) -> str:
    org_bits = [
        f"week={fmt_num(context.get('week'))}",
        f"project={fmt_num(context.get('project'))}",
        f"role={fmt_num(context.get('role'))}",
        f"b_unit={fmt_num(context.get('b_unit'))}",
        f"f_unit={fmt_num(context.get('f_unit'))}",
        f"dept={fmt_num(context.get('dept'))}",
        f"team={fmt_num(context.get('team'))}",
        f"itadmin={fmt_num(context.get('ITAdmin'))}",
    ]
    psy_bits = [f"O={fmt_num(context.get('O'))}", f"C={fmt_num(context.get('C'))}", f"E={fmt_num(context.get('E'))}", f"A={fmt_num(context.get('A'))}", f"N={fmt_num(context.get('N'))}"]
    lines = [
        "DAY " + " ".join(org_bits),
        "PSY " + " ".join(psy_bits),
        f"SESSIONS total={total_sessions} kept={len(sessions)}",
    ]
    for idx, sess in enumerate(sessions):
        sess_bits = [f"idx={idx}"] + [f"{k}={fmt_num(v)}" for k, v in sess.items()]
        lines.append("SES " + " ".join(sess_bits))
    return "\n".join(lines)


def build_examples(df: pd.DataFrame, labels: pd.DataFrame, val_frac: float, max_sessions: int) -> tuple[list[Dict[str, Any]], pd.DataFrame]:
    labels = labels[["user_id", "day_index", "y"]].drop_duplicates()
    label_map = {(str(r.user_id), int(r.day_index)): int(r.y) for r in labels.itertuples(index=False)}
    positive_users = set(labels.loc[labels["y"] > 0, "user_id"].astype(str))

    df = df.sort_values(["user_id", "day_index", "starttime", "sessionid"]).reset_index(drop=True)
    rows: list[Dict[str, Any]] = []
    meta_rows: list[Dict[str, Any]] = []
    grouped = df.groupby(["user_id", "day_index"], sort=False)

    for (user_id, day_index), group in grouped:
        user_id = str(user_id)
        day_index = int(day_index)
        y = int(label_map.get((user_id, day_index), 0))
        split = assign_split(user_id, positive_users, val_frac=val_frac)
        first = group.iloc[0]
        context = {c: first[c] for c in DAY_CONTEXT_COLS if c in group.columns}
        sessions: list[Dict[str, Any]] = []
        for row in group.head(max_sessions).itertuples(index=False):
            sess = {}
            row_dict = row._asdict()
            for col in SESSION_COLS:
                if col in row_dict:
                    sess[col] = row_dict[col]
            sessions.append(sess)
        text = serialize_text(context, sessions, total_sessions=int(len(group)))
        example_id = f"{user_id}:{day_index}"
        rec = {
            "example_id": example_id,
            "user_id": user_id,
            "day_index": day_index,
            "split": split,
            "y": y,
            "n_sessions_total": int(len(group)),
            "n_sessions_kept": int(len(sessions)),
            "context": context,
            "sessions": sessions,
            "text": text,
        }
        rows.append(rec)
        meta_rows.append(
            {
                "example_id": example_id,
                "user_id": user_id,
                "day_index": day_index,
                "split": split,
                "y": y,
                "n_sessions_total": int(len(group)),
                "n_sessions_kept": int(len(sessions)),
                "text_chars": int(len(text)),
            }
        )
    return rows, pd.DataFrame(meta_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True, help="Directory containing session*_shard_*.csv.gz and labels_daily.parquet")
    ap.add_argument("--user-map", type=Path, default=None, help="Path to session*_user_map.csv")
    ap.add_argument("--labels", type=Path, default=None, help="Path to labels_daily.parquet; defaults to input-dir/labels_daily.parquet")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--max-sessions", type=int, default=24)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    labels_path = args.labels or (args.input_dir / "labels_daily.parquet")
    user_map_path = resolve_user_map(args.input_dir, args.user_map)

    user_map = load_user_map(user_map_path)
    print(f"[build_session_jsonl] loaded user map: {len(user_map)} entries", flush=True)
    raw_df = read_session_shards(args.input_dir)
    print(f"[build_session_jsonl] loaded raw sessions: {len(raw_df)} rows", flush=True)
    raw_df = sanitize_frame(raw_df, user_map)
    print(f"[build_session_jsonl] sanitized sessions: {raw_df['user_id'].nunique()} users, {raw_df[['user_id', 'day_index']].drop_duplicates().shape[0]} user-days", flush=True)
    labels = pd.read_parquet(labels_path)
    print(f"[build_session_jsonl] loaded labels: {len(labels)} rows", flush=True)

    examples, meta_df = build_examples(raw_df, labels, val_frac=args.val_frac, max_sessions=args.max_sessions)
    print(f"[build_session_jsonl] built examples: {len(meta_df)}", flush=True)

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
    }
    dump_json(out_dir / "build_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
