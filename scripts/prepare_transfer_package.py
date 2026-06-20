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


DEFAULT_SESSION_CSV = Path(
    "/homes/01/srangdembay/InsiderThreatDetection/r6.2/lcdal-r62-full/extract_stage/r6.2/ExtractedData/sessionr6.2.csv"
)
DEFAULT_LABELS = Path(
    "/homes/01/srangdembay/InsiderThreatDetection/r6.2/ctmc-semantic-daily-ldap/labels_daily.parquet"
)
DEFAULT_OUT = Path(
    "/homes/01/srangdembay/InsiderThreatDetection/r6.2/llm-sequence-mi-remote/artifacts/transfer_package"
)


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


def build_shards(session_csv: Path, out_dir: Path, rows_per_shard: int) -> list[dict]:
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
            shard_path = out_dir / f"sessionr6.2_shard_{shard_idx:03d}.csv.gz"
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
        shard_path = out_dir / f"sessionr6.2_shard_{shard_idx:03d}.csv.gz"
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-csv", type=Path, default=DEFAULT_SESSION_CSV)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--rows-per-shard", type=int, default=250_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.session_csv.exists():
        raise FileNotFoundError(args.session_csv)
    if not args.labels.exists():
        raise FileNotFoundError(args.labels)

    out_dir = args.out_dir
    if out_dir.exists():
        if not args.force and any(out_dir.iterdir()):
            raise RuntimeError(f"Output dir is not empty: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    shard_meta = build_shards(args.session_csv, out_dir, args.rows_per_shard)

    labels_target = out_dir / args.labels.name
    shutil.copy2(args.labels, labels_target)
    labels_sha = sha256_file(labels_target)

    manifest = {
        "session_csv_source": str(args.session_csv),
        "labels_source": str(args.labels),
        "rows_per_shard": args.rows_per_shard,
        "compression": "gzip",
        "num_shards": len(shard_meta),
        "shards": shard_meta,
        "labels": {
            "path": labels_target.name,
            "sha256": labels_sha,
            "bytes": labels_target.stat().st_size,
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    checksum_entries = []
    for item in shard_meta:
        checksum_entries.append(f"{item['sha256']}  {item['path']}")
    checksum_entries.append(f"{labels_sha}  {labels_target.name}")
    checksum_entries.append(f"{sha256_file(manifest_path)}  {manifest_path.name}")
    (out_dir / "sha256sums.txt").write_text("\n".join(checksum_entries) + "\n")

    print(json.dumps({
        "out_dir": str(out_dir),
        "num_shards": len(shard_meta),
        "rows_per_shard": args.rows_per_shard,
        "labels": labels_target.name,
    }, indent=2))


if __name__ == "__main__":
    main()
