#!/usr/bin/env python3
"""Prepare a sharded transfer package for the remote Anvil branch.

This script streams the raw LC-DAL session CSV into compressed shards without
loading the entire file into memory. Every shard preserves the CSV header.
It also copies the label parquet and writes a manifest + sha256 checksums.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_SESSION_CSV = Path(
    "/homes/01/srangdembay/InsiderThreatDetection/r6.2/lcdal-r62-full/extract_stage/r6.2/ExtractedData/sessionr6.2.csv"
)
DEFAULT_LABELS = Path(
    "/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-semantic-daily-ldap/labels_daily.parquet"
)
DEFAULT_USER_MAP = Path(
    "/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-approach/benchmarks/oneclass_unsupervised_r62/results_r62_lcdal_session_features_clean/sessionr6.2_user_map.csv"
)
DEFAULT_OUT = Path(
    "/homes/01/srangdembay/InsiderThreatDetection/r6.2/llm-sequence-mi-remote/artifacts/transfer_package"
)
DEFAULTS_BY_DATASET = {
    "r6.2": {
        "session_csv": DEFAULT_SESSION_CSV,
        "labels": DEFAULT_LABELS,
        "user_map": DEFAULT_USER_MAP,
        "out_dir": DEFAULT_OUT,
        "source_root": Path("/homes/01/srangdembay/InsiderThreatDetection/r6.2"),
    },
    "r4.2": {
        "session_csv": Path("/homes/01/srangdembay/insider_threat/r4.2/ExtractedData/sessionr4.2.csv"),
        "labels": Path("/homes/01/srangdembay/InsiderThreatDetection/r4.2/labels_daily.parquet"),
        "user_map": None,
        "out_dir": Path("/homes/01/srangdembay/InsiderThreatDetection/r6.2/llm-sequence-mi-remote/artifacts/transfer_package_r42"),
        "source_root": Path("/homes/01/srangdembay/insider_threat/r4.2"),
    },
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def iter_csv_rows(path: Path) -> Iterable[list[str]]:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            yield row


def write_gzip_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with gzip.open(path, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def build_shards(session_csv: Path, out_dir: Path, rows_per_shard: int, dataset_tag: str) -> list[dict]:
    rows_iter = iter_csv_rows(session_csv)
    try:
        header = next(rows_iter)
    except StopIteration as exc:
        raise RuntimeError(f"Empty CSV: {session_csv}") from exc

    shard_rows: list[list[str]] = []
    shard_idx = 0
    total_rows = 0
    shard_meta: list[dict] = []

    for row in rows_iter:
        shard_rows.append(row)
        total_rows += 1
        if len(shard_rows) >= rows_per_shard:
            shard_path = out_dir / f"session{dataset_tag}_shard_{shard_idx:03d}.csv.gz"
            write_gzip_csv(shard_path, header, shard_rows)
            shard_meta.append(
                {
                    "path": shard_path.name,
                    "rows": len(shard_rows),
                    "sha256": sha256_file(shard_path),
                    "bytes": shard_path.stat().st_size,
                }
            )
            shard_rows = []
            shard_idx += 1

    if shard_rows:
        shard_path = out_dir / f"session{dataset_tag}_shard_{shard_idx:03d}.csv.gz"
        write_gzip_csv(shard_path, header, shard_rows)
        shard_meta.append(
            {
                "path": shard_path.name,
                "rows": len(shard_rows),
                "sha256": sha256_file(shard_path),
                "bytes": shard_path.stat().st_size,
            }
        )

    return shard_meta


def generate_r42_user_map(source_root: Path, dataset_tag: str, out_dir: Path) -> Path:
    ldap_dir = source_root / "LDAP"
    if not ldap_dir.exists():
        raise FileNotFoundError(ldap_dir)
    user_order: list[str] = []
    seen: set[str] = set()
    allfiles = [f for f in os.listdir(ldap_dir) if (ldap_dir / f).is_file()]
    for fname in allfiles:
        fpath = ldap_dir / fname
        df = pd.read_csv(fpath)
        if "user_id" not in df.columns:
            raise ValueError(f"{fpath} missing user_id column")
        for user_id in df["user_id"].astype(str).tolist():
            if user_id not in seen:
                seen.add(user_id)
                user_order.append(user_id)
    if not user_order:
        raise RuntimeError(f"No users found in LDAP dir: {ldap_dir}")
    user_map_df = pd.DataFrame(
        {
            "user_code": list(range(len(user_order))),
            "user_id": user_order,
        }
    )
    target = out_dir / f"session{dataset_tag}_user_map.csv"
    user_map_df.to_csv(target, index=False)
    return target


def resolve_defaults(dataset_tag: str) -> dict:
    if dataset_tag not in DEFAULTS_BY_DATASET:
        raise ValueError(f"Unsupported dataset_tag: {dataset_tag}")
    return DEFAULTS_BY_DATASET[dataset_tag]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-tag", type=str, default="r6.2", choices=sorted(DEFAULTS_BY_DATASET.keys()))
    ap.add_argument("--session-csv", type=Path, default=None)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--user-map", type=Path, default=None)
    ap.add_argument("--source-root", type=Path, default=None, help="Dataset source tree used for r4.2 user-map regeneration")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--rows-per-shard", type=int, default=250_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    defaults = resolve_defaults(args.dataset_tag)
    session_csv = args.session_csv or defaults["session_csv"]
    labels = args.labels or defaults["labels"]
    user_map = args.user_map or defaults["user_map"]
    source_root = args.source_root or defaults["source_root"]
    out_dir = args.out_dir or defaults["out_dir"]

    if not session_csv.exists():
        raise FileNotFoundError(session_csv)
    if not labels.exists():
        raise FileNotFoundError(labels)

    if out_dir.exists():
        if not args.force and any(out_dir.iterdir()):
            raise RuntimeError(f"Output dir is not empty: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    if user_map is None:
        if args.dataset_tag == "r4.2":
            user_map = generate_r42_user_map(source_root, args.dataset_tag, out_dir)
        else:
            raise FileNotFoundError("User map must be provided for this dataset")
    if not user_map.exists():
        raise FileNotFoundError(user_map)

    shard_meta = build_shards(session_csv, out_dir, args.rows_per_shard, args.dataset_tag)

    labels_target = out_dir / labels.name
    shutil.copy2(labels, labels_target)
    labels_sha = sha256_file(labels_target)
    user_map_target = out_dir / user_map.name
    if user_map.resolve() != user_map_target.resolve():
        shutil.copy2(user_map, user_map_target)
    user_map_sha = sha256_file(user_map_target)

    manifest = {
        "dataset_tag": args.dataset_tag,
        "session_csv_source": str(session_csv),
        "labels_source": str(labels),
        "user_map_source": str(user_map),
        "source_root": str(source_root),
        "rows_per_shard": args.rows_per_shard,
        "compression": "gzip",
        "num_shards": len(shard_meta),
        "shards": shard_meta,
        "labels": {
            "path": labels_target.name,
            "sha256": labels_sha,
            "bytes": labels_target.stat().st_size,
        },
        "user_map": {
            "path": user_map_target.name,
            "sha256": user_map_sha,
            "bytes": user_map_target.stat().st_size,
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    checksum_entries = []
    for item in shard_meta:
        checksum_entries.append(f"{item['sha256']}  {item['path']}")
    checksum_entries.append(f"{labels_sha}  {labels_target.name}")
    checksum_entries.append(f"{user_map_sha}  {user_map_target.name}")
    checksum_entries.append(f"{sha256_file(manifest_path)}  {manifest_path.name}")
    (out_dir / "sha256sums.txt").write_text("\n".join(checksum_entries) + "\n")

    print(json.dumps({
        "dataset_tag": args.dataset_tag,
        "out_dir": str(out_dir),
        "num_shards": len(shard_meta),
        "rows_per_shard": args.rows_per_shard,
        "labels": labels_target.name,
        "user_map": user_map_target.name,
    }, indent=2))


if __name__ == "__main__":
    main()
