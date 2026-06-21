from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import numpy as np
import pandas as pd


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load config files") from exc
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Config at {path} is not a mapping")
    return obj


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_hash_frac(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    value = int(digest[:16], 16)
    return value / float(16**16 - 1)


def fmt_num(x: Any) -> str:
    if x is None:
        return "na"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if xf.is_integer():
        return str(int(xf))
    return f"{xf:.3f}".rstrip("0").rstrip(".")


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=True) + "\n")
            n += 1
    return n


def load_user_map(path: Path) -> Dict[int, str]:
    df = pd.read_csv(path)
    required = {"user_code", "user_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing columns {required}")
    return {int(r.user_code): str(r.user_id) for r in df.itertuples(index=False)}


def read_session_shards(input_dir: Path) -> pd.DataFrame:
    shard_paths = sorted(input_dir.glob("sessionr6.2_shard_*.csv.gz"))
    if not shard_paths:
        raise FileNotFoundError(f"No session shards found in {input_dir}")
    dfs = [pd.read_csv(p, compression="gzip", low_memory=False) for p in shard_paths]
    return pd.concat(dfs, ignore_index=True)


def dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(obj), indent=2) + "\n", encoding="utf-8")


def maybe_read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())
