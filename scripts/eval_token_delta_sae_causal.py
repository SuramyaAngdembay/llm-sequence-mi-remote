#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from remote_common import dump_json, ensure_dir, load_yaml, read_jsonl
from sae_core import TopKSAE, add_active_control_feature_sets, choose_feature_sets


CONTEXT_MODE_COLS: Dict[str, List[str]] = {
    "team": ["team"],
    "role": ["role"],
    "project": ["project"],
    "project_role": ["project", "role"],
    "dept": ["dept"],
    "dept_role": ["dept", "role"],
}


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def table_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def per_example_nll(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    loss_batch_size: int = 0,
) -> torch.Tensor:
    """Compute per-example NLL while bounding the fp32 cross-entropy workspace."""
    batch_size = int(logits.shape[0])
    chunk_size = batch_size if int(loss_batch_size) <= 0 else min(batch_size, int(loss_batch_size))
    out: List[torch.Tensor] = []
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        shift_logits = logits[start:end, :-1, :].float().contiguous()
        shift_labels = input_ids[start:end, 1:].contiguous()
        shift_mask = attention_mask[start:end, 1:].contiguous().float()
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.size())
        denom = shift_mask.sum(dim=1).clamp_min(1.0)
        out.append((token_loss * shift_mask).sum(dim=1) / denom)
        del shift_logits, shift_labels, shift_mask, token_loss, denom
        if logits.device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(out, dim=0)


def per_example_nll_from_hidden(
    lm_head: Any,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    max_logit_elements: int,
) -> torch.Tensor:
    """Compute exact per-example NLL while chunking the LM head over token positions."""
    batch_size, seq_len, _ = hidden_states.shape
    if seq_len <= 1:
        return torch.zeros((batch_size,), device=hidden_states.device, dtype=torch.float32)

    vocab_size = int(getattr(lm_head, "out_features", 0) or getattr(getattr(lm_head, "weight", None), "shape", [0])[0])
    if vocab_size <= 0:
        raise ValueError("Could not infer lm_head vocab size for chunked NLL")
    token_chunk = max(1, int(max_logit_elements) // max(1, batch_size * vocab_size))
    token_chunk = min(token_chunk, seq_len - 1)

    losses = torch.zeros((batch_size,), device=hidden_states.device, dtype=torch.float32)
    denom = attention_mask[:, 1:].float().sum(dim=1).clamp_min(1.0)

    for pos_start in range(0, seq_len - 1, token_chunk):
        pos_end = min(pos_start + token_chunk, seq_len - 1)
        labels = input_ids[:, pos_start + 1 : pos_end + 1].contiguous()
        mask = attention_mask[:, pos_start + 1 : pos_end + 1].contiguous().float()
        if float(mask.sum().item()) == 0.0:
            continue
        logits = lm_head(hidden_states[:, pos_start:pos_end, :])
        token_loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        ).view(labels.size())
        losses += (token_loss * mask).sum(dim=1)
        del logits, labels, mask, token_loss
        if hidden_states.device.type == "cuda":
            torch.cuda.empty_cache()

    return losses / denom


def _load_pt_cpu(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, ValueError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def _select_token_chunks(
    extract_dir: Path,
    layer: int,
    chunk_paths: Sequence[Path],
    keep_arr: np.ndarray | None,
) -> List[Path]:
    if keep_arr is None:
        print(f"[token_load] layer={layer} keep_examples=all selected_chunks={len(chunk_paths)}/{len(chunk_paths)}")
        return list(chunk_paths)

    summary_path = extract_dir / "extract_summary.json"
    manifest_path = extract_dir / "chunk_manifest.csv"
    if not summary_path.exists() or not manifest_path.exists():
        print(
            f"[token_load] layer={layer} keep_examples={len(keep_arr)} selected_chunks={len(chunk_paths)}/{len(chunk_paths)} "
            "(manifest unavailable; full scan)"
        )
        return list(chunk_paths)

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        chunk_examples = int(summary["chunk_examples"])
        if chunk_examples <= 0:
            raise ValueError(f"invalid chunk_examples={chunk_examples}")
        manifest = pd.read_csv(manifest_path)
        layer_manifest = manifest.loc[manifest["layer"].astype(int) == int(layer)].copy()
        if layer_manifest.empty or "chunk_id" not in layer_manifest or "path" not in layer_manifest:
            raise ValueError(f"missing layer {layer} manifest rows")
    except Exception as exc:
        print(
            f"[token_load] layer={layer} keep_examples={len(keep_arr)} selected_chunks={len(chunk_paths)}/{len(chunk_paths)} "
            f"(manifest read failed: {exc}; full scan)"
        )
        return list(chunk_paths)

    needed_chunk_ids = set((keep_arr // chunk_examples).astype(np.int64).tolist())
    layer_manifest["chunk_id"] = layer_manifest["chunk_id"].astype(int)
    layer_manifest = layer_manifest.loc[layer_manifest["chunk_id"].isin(needed_chunk_ids)].sort_values("chunk_id")

    selected: List[Path] = []
    for text_path in layer_manifest["path"].tolist():
        path = Path(str(text_path))
        if not path.is_absolute():
            path = extract_dir / path
        if path.exists():
            selected.append(path)

    if not selected:
        print(
            f"[token_load] layer={layer} keep_examples={len(keep_arr)} selected_chunks=0/{len(chunk_paths)} "
            "(no manifest paths matched requested examples)"
        )
        return []

    print(
        f"[token_load] layer={layer} keep_examples={len(keep_arr)} "
        f"selected_chunks={len(selected)}/{len(chunk_paths)}"
    )
    return selected


def _selected_example_intervals(idx: np.ndarray, keep_arr: np.ndarray | None) -> List[Tuple[int, int]]:
    if keep_arr is None:
        return [(0, int(len(idx)))]
    if len(idx) == 0 or len(keep_arr) == 0:
        return []
    lo_ex = int(idx[0])
    hi_ex = int(idx[-1])
    left = int(np.searchsorted(keep_arr, lo_ex, side="left"))
    right = int(np.searchsorted(keep_arr, hi_ex, side="right"))
    intervals: List[Tuple[int, int]] = []
    for ex in keep_arr[left:right]:
        start = int(np.searchsorted(idx, int(ex), side="left"))
        end = int(np.searchsorted(idx, int(ex), side="right"))
        if end > start:
            intervals.append((start, end))
    return intervals


def _concat_selected_delta(delta: Any, intervals: Sequence[Tuple[int, int]], dtype: np.dtype[Any]) -> np.ndarray:
    if not intervals:
        shape = delta.shape if hasattr(delta, "shape") else np.asarray(delta).shape
        return np.empty((0, int(shape[1])), dtype=dtype)
    parts: List[np.ndarray] = []
    for start, end in intervals:
        if isinstance(delta, torch.Tensor):
            arr = delta[start:end].cpu().numpy()
            if arr.dtype != dtype:
                arr = arr.astype(dtype, copy=False)
        else:
            arr = np.asarray(delta[start:end], dtype=dtype)
        parts.append(arr)
    if len(parts) == 1:
        return np.asarray(parts[0], dtype=dtype)
    return np.concatenate(parts, axis=0).astype(dtype, copy=False)


def load_token_layer_vectors(
    extract_dir: Path,
    layer: int,
    *,
    keep_examples: set[int] | None = None,
    delta_dtype: str = "float32",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    scores = pd.read_parquet(extract_dir / "example_scores.parquet").sort_values("example_idx").reset_index(drop=True)
    layer_dir = extract_dir / f"layer_{layer}"
    chunk_paths = sorted(layer_dir.glob("chunk_*.pt"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunks found for layer {layer} in {layer_dir}")

    keep_arr = np.asarray(sorted(keep_examples), dtype=np.int64) if keep_examples else None
    chunk_paths = _select_token_chunks(extract_dir, layer, chunk_paths, keep_arr)
    if not chunk_paths:
        raise RuntimeError(f"No token chunks matched requested examples at layer {layer}")
    total_rows = 0
    d_model: int | None = None

    for path in chunk_paths:
        obj = _load_pt_cpu(path)
        if "position" not in obj or obj["position"] is None:
            raise RuntimeError("Token-level causal eval requires token-level extraction with position arrays.")
        idx = np.asarray(obj["example_idx"], dtype=np.int64)
        if d_model is None:
            delta_shape = obj["delta"].shape if hasattr(obj["delta"], "shape") else np.asarray(obj["delta"]).shape
            d_model = int(delta_shape[1])
        intervals = _selected_example_intervals(idx, keep_arr)
        total_rows += int(sum(end - start for start, end in intervals))
    if total_rows == 0 or d_model is None:
        raise RuntimeError(f"No token rows found for requested examples at layer {layer}")

    np_delta_dtype = np.float16 if delta_dtype == "float16" else np.float32
    x = np.empty((total_rows, d_model), dtype=np_delta_dtype)
    example_idx = np.empty(total_rows, dtype=np.int64)
    position = np.empty(total_rows, dtype=np.int32)

    cursor = 0
    for path in chunk_paths:
        obj = _load_pt_cpu(path)
        idx = np.asarray(obj["example_idx"], dtype=np.int64)
        intervals = _selected_example_intervals(idx, keep_arr)
        if not intervals:
            continue
        vecs = _concat_selected_delta(obj["delta"], intervals, np_delta_dtype)
        idx = np.concatenate([idx[start:end] for start, end in intervals]).astype(np.int64, copy=False)
        pos_arr = np.asarray(obj["position"], dtype=np.int32)
        pos = np.concatenate([pos_arr[start:end] for start, end in intervals]).astype(np.int32, copy=False)
        n_rows = int(len(idx))
        if n_rows == 0:
            continue
        x[cursor : cursor + n_rows] = vecs
        example_idx[cursor : cursor + n_rows] = idx
        position[cursor : cursor + n_rows] = pos
        cursor += n_rows

    if cursor != total_rows:
        raise RuntimeError(f"Expected {total_rows} token rows, loaded {cursor}")
    if len(example_idx) > 1:
        if np.any(example_idx[1:] < example_idx[:-1]):
            raise RuntimeError("Token chunks are not ordered by example_idx; causal eval requires ordered extraction chunks.")
        same_example = example_idx[1:] == example_idx[:-1]
        if np.any(position[1:][same_example] < position[:-1][same_example]):
            raise RuntimeError("Token chunks are not ordered by token position within example; causal eval requires ordered extraction chunks.")

    print(f"[token_load] layer={layer} token_rows={int(total_rows)} d_model={int(d_model)}")
    return x, example_idx, scores


def build_all_candidate_pairs(
    example_meta: pd.DataFrame,
    context_modes: Sequence[str],
    *,
    max_receivers: int,
    max_candidate_donors: int,
    seed: int,
    exclude_same_user: bool = False,
) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
    out: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    receiver_indices: np.ndarray | None = None
    if max_receivers > 0:
        positive_idx = example_meta.loc[example_meta["y"] == 1, "row_idx"].to_numpy(
            dtype=int
        )
        if len(positive_idx) > max_receivers:
            receiver_indices = np.sort(
                np.random.default_rng(seed).choice(
                    positive_idx, size=max_receivers, replace=False
                )
            )

    for mode_i, context_mode in enumerate(context_modes):
        for donor_type, donor_label in [("benign", 0), ("anomalous", 1)]:
            out[(context_mode, donor_type)] = build_candidate_pairs(
                example_meta,
                donor_label=donor_label,
                context_mode=context_mode,
                max_receivers=max_receivers,
                max_candidate_donors=max_candidate_donors,
                rng=np.random.default_rng(seed + 1009 * mode_i + donor_label),
                receiver_indices=receiver_indices,
                exclude_same_user=exclude_same_user,
            )
    return out


def examples_from_pairs(candidate_pairs: Dict[Tuple[str, str], List[Tuple[int, int]]]) -> set[int]:
    out: set[int] = set()
    for pairs in candidate_pairs.values():
        for recv_idx, donor_idx in pairs:
            out.add(int(recv_idx))
            out.add(int(donor_idx))
    return out


def flatten_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (ctx or {}).items():
        out[str(k)] = v
    return out


def load_eval_examples(data_dir: Path, scores: pd.DataFrame) -> pd.DataFrame:
    rows = list(read_jsonl(data_dir / "eval.jsonl"))
    if len(rows) != len(scores):
        raise ValueError(f"eval.jsonl rows ({len(rows)}) != example_scores rows ({len(scores)})")
    meta_rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(rows):
        row = {
            "example_idx": i,
            "example_id": ex["example_id"],
            "user_id": ex["user_id"],
            "day_index": int(ex["day_index"]),
            "split": ex["split"],
            "y": int(ex["y"]),
            "n_sessions_total": int(ex["n_sessions_total"]),
            "n_sessions_kept": int(ex["n_sessions_kept"]),
            "text": ex["text"],
        }
        row.update(flatten_context(ex.get("context", {})))
        meta_rows.append(row)
    meta = pd.DataFrame(meta_rows).sort_values("example_idx").reset_index(drop=True)
    if not np.array_equal(meta["example_idx"].to_numpy(), scores["example_idx"].to_numpy()):
        raise ValueError("JSONL/example_scores example_idx mismatch")
    if not np.array_equal(meta["example_id"].astype(str).to_numpy(), scores["example_id"].astype(str).to_numpy()):
        raise ValueError("JSONL/example_scores example_id mismatch")
    merged = pd.concat(
        [
            meta,
            scores.drop(columns=["example_id", "user_id", "day_index", "split", "y", "n_sessions_total", "n_sessions_kept"]),
        ],
        axis=1,
    )
    merged["row_idx"] = np.arange(len(merged), dtype=int)
    return merged


def resolve_context_cols(df: pd.DataFrame, mode: str) -> List[str]:
    cols = CONTEXT_MODE_COLS.get(mode)
    if cols is None:
        raise ValueError(f"Unsupported context mode: {mode}")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Context mode {mode} requires missing columns: {missing}")
    return cols


def required_context_cols(df: pd.DataFrame, modes: List[str]) -> List[str]:
    cols: List[str] = []
    for mode in modes:
        for col in resolve_context_cols(df, mode):
            if col not in cols:
                cols.append(col)
    return cols


def context_key(df: pd.DataFrame, mode: str) -> pd.Series:
    cols = resolve_context_cols(df, mode)
    parts = [df[c].fillna("UNK").astype(str) for c in cols]
    if len(parts) == 1:
        return parts[0]
    out = parts[0]
    for p in parts[1:]:
        out = out + "||" + p
    return out


def build_candidate_pairs(
    meta: pd.DataFrame,
    *,
    donor_label: int,
    context_mode: str,
    max_receivers: int,
    max_candidate_donors: int,
    rng: np.random.Generator,
    receiver_indices: np.ndarray | None = None,
    exclude_same_user: bool = False,
) -> List[Tuple[int, int]]:
    ctx_cols = resolve_context_cols(meta, context_mode)
    primary_col = ctx_cols[0]
    work = meta[["row_idx", "user_id", "y"] + ctx_cols].copy()
    work["ctx_key"] = context_key(work, context_mode)

    recv_df = work.loc[work["y"] == 1].copy()
    if receiver_indices is not None:
        recv_df = (
            recv_df.set_index("row_idx")
            .loc[np.asarray(receiver_indices, dtype=int)]
            .reset_index()
        )
    else:
        recv_indices = recv_df["row_idx"].to_numpy(dtype=int)
        if max_receivers > 0 and len(recv_indices) > max_receivers:
            chosen = np.sort(rng.choice(recv_indices, size=max_receivers, replace=False))
            recv_df = recv_df.set_index("row_idx").loc[chosen].reset_index()

    donor_df = work.loc[work["y"] == donor_label].copy()
    donor_all = np.sort(donor_df["row_idx"].to_numpy(dtype=int))
    donor_by_key = {
        str(key): np.sort(sub["row_idx"].to_numpy(dtype=int))
        for key, sub in donor_df.groupby("ctx_key", sort=False)
    }
    donor_by_primary = {
        str(key): np.sort(sub["row_idx"].to_numpy(dtype=int))
        for key, sub in donor_df.groupby(primary_col, sort=False)
    }
    donor_user_map = donor_df.set_index("row_idx")["user_id"].astype(str).to_dict()

    pairs: List[Tuple[int, int]] = []
    for _, row in recv_df.iterrows():
        recv_idx = int(row["row_idx"])
        recv_user = str(row["user_id"])
        pool = donor_by_key.get(str(row["ctx_key"]))
        if pool is None or len(pool) == 0:
            pool = donor_by_primary.get(str(row[primary_col]))
        if pool is None or len(pool) == 0:
            pool = donor_all
        if len(pool) > 0:
            mask = np.ones(len(pool), dtype=bool)
            if donor_label == 1:
                mask &= pool != recv_idx
            if exclude_same_user:
                mask &= np.asarray([donor_user_map.get(int(d), "") != recv_user for d in pool], dtype=bool)
            pool = pool[mask]
        if pool is None or len(pool) == 0:
            continue
        if max_candidate_donors > 0 and len(pool) > max_candidate_donors:
            selected = np.sort(rng.choice(pool, size=max_candidate_donors, replace=False))
        else:
            selected = pool
        pairs.extend((recv_idx, int(d)) for d in selected.tolist())
    return pairs


def get_base_causal_lm(model: Any) -> Any:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def resolve_transformer_layers(model: Any) -> Any:
    base = get_base_causal_lm(model)
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return base.model.layers
    if hasattr(base, "model") and hasattr(base.model, "model") and hasattr(base.model.model, "layers"):
        return base.model.model.layers
    raise AttributeError("Could not resolve transformer layers for the adapted model")


def get_layer_module(model: Any, hidden_state_layer: int) -> Any:
    if hidden_state_layer <= 0:
        raise ValueError("hidden_state_layer must be >= 1; 0 is embeddings")
    layers = resolve_transformer_layers(model)
    block_idx = hidden_state_layer - 1
    if block_idx >= len(layers):
        raise IndexError(f"Requested hidden_state_layer={hidden_state_layer} but model has only {len(layers)} blocks")
    return layers[block_idx]


def estimate_full_logits_gib(model: Any, input_ids: torch.Tensor) -> float:
    base = get_base_causal_lm(model)
    vocab_size = int(getattr(getattr(base, "config", None), "vocab_size", 0))
    if vocab_size <= 0 and hasattr(base, "get_output_embeddings"):
        emb = base.get_output_embeddings()
        vocab_size = int(getattr(emb, "out_features", 0) or getattr(getattr(emb, "weight", None), "shape", [0])[0])
    if vocab_size <= 0:
        return float("inf")
    dtype = next(model.parameters()).dtype
    dtype_bytes = 2 if dtype in {torch.float16, torch.bfloat16} else 4
    elements = int(input_ids.shape[0]) * int(input_ids.shape[1]) * vocab_size
    return float(elements * dtype_bytes / (1024**3))


@torch.no_grad()
def encode_sparse_vectors(
    sae_model: TopKSAE,
    x_norm: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    zs: List[np.ndarray] = []
    for start in range(0, len(x_norm), batch_size):
        xb = torch.from_numpy(x_norm[start : start + batch_size]).to(device)
        z = sae_model.encode_sparse(xb).cpu().numpy().astype(np.float32)
        zs.append(z)
        del xb, z
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(zs, axis=0)


def build_example_slices(example_idx: np.ndarray) -> Dict[int, slice]:
    out: Dict[int, slice] = {}
    if len(example_idx) == 0:
        return out
    starts = np.concatenate(([0], np.flatnonzero(example_idx[1:] != example_idx[:-1]) + 1, [len(example_idx)]))
    for i in range(len(starts) - 1):
        s = int(starts[i])
        e = int(starts[i + 1])
        out[int(example_idx[s])] = slice(s, e)
    return out


def normalize_token_block(
    token_block: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
) -> np.ndarray:
    return ((np.asarray(token_block, dtype=np.float32) - x_mean) / x_std).astype(np.float32, copy=False)


def build_sparse_cache_for_examples(
    sae_model: TopKSAE,
    x: np.ndarray,
    token_slices: Dict[int, slice],
    example_ids: Sequence[int],
    *,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Dict[int, np.ndarray]:
    cache: Dict[int, np.ndarray] = {}
    for example_id in example_ids:
        token_slice = token_slices.get(int(example_id))
        if token_slice is None:
            continue
        token_norm = normalize_token_block(x[token_slice], x_mean=x_mean, x_std=x_std)
        cache[int(example_id)] = encode_sparse_vectors(
            sae_model,
            token_norm,
            device=device,
            batch_size=batch_size,
        )
    return cache


def donor_feature_prototype(sparse_donor_tokens: np.ndarray, feature_ids: Sequence[int]) -> np.ndarray:
    if sparse_donor_tokens.size == 0:
        return np.zeros((len(feature_ids),), dtype=np.float32)
    feat = sparse_donor_tokens[:, feature_ids]
    active = feat.sum(axis=1) > 0.0
    if np.any(active):
        return feat[active].mean(axis=0).astype(np.float32)
    return feat.mean(axis=0).astype(np.float32)


@torch.no_grad()
def decode_sparse_tokens(
    sae_model: TopKSAE,
    sparse_tokens: np.ndarray,
    *,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    out: List[np.ndarray] = []
    x_mean_t = torch.from_numpy(x_mean).to(device)
    x_std_t = torch.from_numpy(x_std).to(device)
    for start in range(0, len(sparse_tokens), batch_size):
        z = torch.from_numpy(sparse_tokens[start : start + batch_size]).to(device)
        x_patch_norm = sae_model.decoder(z)
        x_patch = (x_patch_norm * x_std_t + x_mean_t).cpu().numpy().astype(np.float32)
        out.append(x_patch)
        del z, x_patch_norm
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del x_mean_t, x_std_t
    return np.concatenate(out, axis=0)


def build_token_patch_shift(
    sae_model: TopKSAE,
    recv_delta_tokens: np.ndarray,
    recv_sparse_tokens: np.ndarray,
    donor_sparse_tokens: np.ndarray,
    feature_ids: Sequence[int],
    alpha: float,
    *,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, int]:
    recv_delta_tokens = np.asarray(recv_delta_tokens, dtype=np.float32)
    if recv_delta_tokens.shape[0] != recv_sparse_tokens.shape[0]:
        raise ValueError("receiver token delta / sparse rows mismatch")
    if len(feature_ids) == 0:
        return np.zeros_like(recv_delta_tokens), 0

    active_mask = recv_sparse_tokens[:, feature_ids].sum(axis=1) > 0.0
    active_idx = np.flatnonzero(active_mask)
    if len(active_idx) == 0:
        return np.zeros_like(recv_delta_tokens), 0

    donor_proto = donor_feature_prototype(donor_sparse_tokens, feature_ids)
    sparse_cf = np.array(recv_sparse_tokens, copy=True)
    for feat_offset, feat_id in enumerate(feature_ids):
        sparse_cf[active_idx, feat_id] = (1.0 - alpha) * recv_sparse_tokens[active_idx, feat_id] + alpha * donor_proto[feat_offset]

    patched_delta = decode_sparse_tokens(
        sae_model,
        sparse_cf,
        x_mean=x_mean,
        x_std=x_std,
        device=device,
        batch_size=batch_size,
    )
    shift = patched_delta - recv_delta_tokens
    return shift.astype(np.float32), int(len(active_idx))


@torch.no_grad()
def score_with_token_patches(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    patch_shifts: Sequence[np.ndarray],
    *,
    layer: int,
    max_seq_len: int,
    batch_size: int,
    loss_batch_size: int = 0,
    full_logits_max_gib: float = 28.0,
    max_logit_elements: int = 536_870_912,
) -> np.ndarray:
    layer_module = get_layer_module(model, layer)
    device = next(model.parameters()).device
    out_scores: List[np.ndarray] = []
    chunked_notice_emitted = False

    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        batch_patch = list(patch_shifts[start : start + batch_size])
        tok = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        attn = tok["attention_mask"]

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if isinstance(output, tuple):
                hs = output[0]
                rest = output[1:]
            else:
                hs = output
                rest = None
            patch = torch.zeros_like(hs)
            for bi, mat in enumerate(batch_patch):
                valid = int(attn[bi].sum().item())
                take = min(valid, int(mat.shape[0]))
                if take > 0:
                    patch[bi, :take, :] = torch.from_numpy(mat[:take]).to(device=device, dtype=hs.dtype)
            hs = hs + patch
            if rest is None:
                return hs
            return (hs, *rest)

        handle = layer_module.register_forward_hook(hook)
        try:
            full_logits_gib = estimate_full_logits_gib(model, tok["input_ids"])
            if full_logits_gib <= float(full_logits_max_gib):
                out = model(**tok, return_dict=True)
                logits = out.logits
                nll_t = per_example_nll(logits, tok["input_ids"], tok["attention_mask"], loss_batch_size=loss_batch_size)
                del out, logits
            else:
                if not chunked_notice_emitted:
                    print(
                        "[scoring] using chunked logits "
                        f"estimated_full_logits_gib={full_logits_gib:.2f} "
                        f"threshold_gib={float(full_logits_max_gib):.2f} "
                        f"max_logit_elements={int(max_logit_elements)}",
                        flush=True,
                    )
                    chunked_notice_emitted = True
                base_model = get_base_causal_lm(model)
                if not hasattr(base_model, "model") or not hasattr(base_model, "get_output_embeddings"):
                    raise AttributeError("Chunked scoring requires a causal LM with .model and .get_output_embeddings()")
                out = base_model.model(
                    input_ids=tok["input_ids"],
                    attention_mask=tok["attention_mask"],
                    return_dict=True,
                    use_cache=False,
                )
                hidden_states = out.last_hidden_state
                nll_t = per_example_nll_from_hidden(
                    base_model.get_output_embeddings(),
                    hidden_states,
                    tok["input_ids"],
                    tok["attention_mask"],
                    max_logit_elements=int(max_logit_elements),
                )
                del out, hidden_states
        finally:
            handle.remove()
        nll = nll_t.cpu().numpy()
        out_scores.append(nll)
        del nll_t, nll, tok, attn, batch_patch, batch_texts
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    return np.concatenate(out_scores, axis=0)


def summarize_best(best_df: pd.DataFrame, top_sets: Sequence[str], control_set: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if best_df.empty:
        return pd.DataFrame()
    key_cols = ["layer", "latent_mult", "k", "context_mode"]
    for key, sub in best_df.groupby(key_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        layer, latent_mult, k, context_mode = key
        work = sub.pivot_table(
            index=["receiver_row_idx"],
            columns=["feature_set", "donor_type"],
            values="delta",
            aggfunc="first",
        )
        work.columns = [f"{a}_{b}" for a, b in work.columns]
        work = work.reset_index()
        for top_set in top_sets:
            needed = [f"{top_set}_benign", f"{top_set}_anomalous", f"{control_set}_benign", f"{control_set}_anomalous"]
            row = {
                "layer": int(layer),
                "latent_mult": int(latent_mult),
                "k": int(k),
                "context_mode": context_mode,
                "target": top_set,
                "n_receivers": int(len(work)),
            }
            for col in needed:
                if col not in work.columns:
                    work[col] = float("nan")
            complete = work.dropna(subset=needed)
            row["n_complete_receivers"] = int(len(complete))
            row["top_benign_mean_best_delta"] = float(complete[f"{top_set}_benign"].mean())
            row["top_anomalous_mean_best_delta"] = float(complete[f"{top_set}_anomalous"].mean())
            row["control_benign_mean_best_delta"] = float(complete[f"{control_set}_benign"].mean())
            row["control_anomalous_mean_best_delta"] = float(complete[f"{control_set}_anomalous"].mean())
            row["top_repair_advantage"] = float((complete[f"{top_set}_anomalous"] - complete[f"{top_set}_benign"]).mean())
            row["control_repair_advantage"] = float((complete[f"{control_set}_anomalous"] - complete[f"{control_set}_benign"]).mean())
            row["top_minus_control_advantage"] = float(
                (
                    (complete[f"{top_set}_anomalous"] - complete[f"{top_set}_benign"])
                    - (complete[f"{control_set}_anomalous"] - complete[f"{control_set}_benign"])
                ).mean()
            )
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["top_minus_control_advantage", "top_repair_advantage", "top_benign_mean_best_delta"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def write_report(
    summary_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    best_df: pd.DataFrame,
    out_path: Path,
    *,
    layer: int,
    latent_mult: int,
    k: int,
    control_set: str,
    exclude_same_user: bool,
) -> None:
    lines = [
        "# Token Delta SAE Causal Eval",
        "",
        f"Token-level model patching on adapter deltas at hidden-state layer `{layer}` with SAE config `latent_mult={latent_mult}, k={k}`.",
        "",
        "Intervention protocol:",
        "- receivers = positive eval examples only",
        "- donors = matched benign donors and same-class anomalous donor controls",
        f"- same-user donors excluded: `{bool(exclude_same_user)}`",
        "- feature sets = top sparse sets patched in token-level delta-SAE space, compared against the control set",
        "- only receiver token positions where the target sparse features are active are patched",
        "- patched token deltas move toward a donor token-feature prototype rather than a uniform sequence-wide broadcast",
        "- summary advantages are paired receiver-level contrasts over receivers with complete top/control and benign/anomalous donor support",
        "",
        f"Control comparison: `{control_set}`",
        "",
        "## Summary",
        "",
        table_text(summary_df),
        "",
        "## Selected Feature Sets",
        "",
        table_text(selected_df),
        "",
        "## Example Receiver-Level Best Repairs",
        "",
        table_text(best_df.head(40)),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--frontier-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--latent-mult", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--loss-batch-size", type=int, default=0)
    ap.add_argument("--sae-batch-size", type=int, default=2048)
    ap.add_argument("--patch-chunk-size", type=int, default=0)
    ap.add_argument("--full-logits-max-gib", type=float, default=28.0)
    ap.add_argument("--max-logit-elements", type=int, default=536_870_912)
    ap.add_argument("--token-delta-dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--context-modes", default="team,role,project_role,dept_role")
    ap.add_argument("--top-sets", default="top1,top3,top5")
    ap.add_argument("--control-set", default="control3")
    ap.add_argument("--active-control-min-frac", type=float, default=0.002)
    ap.add_argument("--active-control-sizes", default="1,3,5")
    ap.add_argument("--alphas", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--max-receivers", type=int, default=0)
    ap.add_argument("--max-candidate-donors", type=int, default=16)
    ap.add_argument("--repair-threshold", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-same-user-donors", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Missing runtime dependencies for token causal eval.") from exc

    context_modes = parse_csv_list(args.context_modes)
    top_sets = parse_csv_list(args.top_sets)
    active_control_sizes = [int(x) for x in parse_csv_list(args.active_control_sizes)]
    alphas = [float(x) for x in parse_csv_list(args.alphas)]
    control_set = str(args.control_set)

    out_dir = ensure_dir(args.out_dir)
    scores = pd.read_parquet(args.extract_dir / "example_scores.parquet").sort_values("example_idx").reset_index(drop=True)
    example_meta = load_eval_examples(args.data_dir, scores)
    _ = required_context_cols(example_meta, context_modes)
    candidate_pairs_by_key = build_all_candidate_pairs(
        example_meta,
        context_modes,
        max_receivers=args.max_receivers,
        max_candidate_donors=args.max_candidate_donors,
        seed=args.seed,
        exclude_same_user=bool(args.exclude_same_user_donors),
    )
    needed_examples = examples_from_pairs(candidate_pairs_by_key)
    x, token_example_idx, _ = load_token_layer_vectors(
        args.extract_dir,
        args.layer,
        keep_examples=needed_examples,
        delta_dtype=args.token_delta_dtype,
    )

    cfg_dir = args.frontier_dir / f"layer_{args.layer}" / f"m{args.latent_mult:02d}_k{args.k:02d}"
    model_bundle = torch.load(cfg_dir / "delta_sae_model.pt", map_location="cpu", weights_only=False)
    if str(model_bundle.get("unit", "")) not in {"", "token"}:
        raise ValueError(f"Frontier model bundle unit is {model_bundle.get('unit')}, expected token")
    feature_df = pd.read_csv(cfg_dir / "delta_sae_top_features.csv")
    feature_sets = choose_feature_sets(feature_df)
    feature_sets = add_active_control_feature_sets(
        feature_sets,
        feature_df,
        min_active_frac=float(args.active_control_min_frac),
        sizes=active_control_sizes,
    )
    if control_set not in feature_sets:
        available = ", ".join(sorted(feature_sets))
        raise ValueError(f"Requested control set {control_set} missing from token SAE feature sets. Available: {available}")
    for name in top_sets:
        if name not in feature_sets:
            raise ValueError(f"Requested top set {name} missing from token SAE feature sets")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae_model = TopKSAE(
        d_in=int(model_bundle["d_in"]),
        d_latent=int(model_bundle["d_latent"]),
        k=int(model_bundle["k"]),
    ).to(device)
    sae_model.load_state_dict(model_bundle["state_dict"])
    sae_model.eval()
    x_mean = np.asarray(model_bundle["x_mean"], dtype=np.float32)
    x_std = np.asarray(model_bundle["x_std"], dtype=np.float32)
    token_slices = build_example_slices(token_example_idx.astype(np.int64, copy=False))

    import torch as _torch

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=bool(cfg["quantization"]["load_in_4bit"]),
        bnb_4bit_quant_type=str(cfg["quantization"]["bnb_4bit_quant_type"]),
        bnb_4bit_compute_dtype=getattr(_torch, str(cfg["quantization"]["bnb_4bit_compute_dtype"])),
        bnb_4bit_use_double_quant=bool(cfg["quantization"]["bnb_4bit_use_double_quant"]),
    )
    model_name = cfg["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    adapted_backbone = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        torch_dtype=_torch.bfloat16 if bool(cfg["training"].get("bf16", True)) else _torch.float16,
        device_map="auto",
    )
    adapted_model = PeftModel.from_pretrained(adapted_backbone, args.adapter_dir)
    adapted_model.config.use_cache = False
    adapted_model.eval()

    selected_rows = [
        {
            "layer": int(args.layer),
            "latent_mult": int(args.latent_mult),
            "k": int(args.k),
            "feature_set": name,
            "n_features": int(len(ids)),
            "feature_ids": json.dumps([int(x) for x in ids]),
            "mean_row_gap": float(feature_df.set_index("feature_id").loc[ids, "row_gap"].mean()) if ids else float("nan"),
        }
        for name, ids in feature_sets.items()
        if name in set(top_sets + [control_set, "control1"])
    ]

    texts = example_meta["text"].tolist()
    base_scores = example_meta["adapted_nll"].to_numpy(dtype=np.float32)
    candidate_path = out_dir / "token_delta_sae_causal_candidate_rows.csv"
    candidate_fieldnames = [
        "layer",
        "latent_mult",
        "k",
        "context_mode",
        "feature_set",
        "donor_type",
        "receiver_row_idx",
        "donor_row_idx",
        "receiver_example_id",
        "donor_example_id",
        "alpha",
        "base_score",
        "patched_score",
        "delta",
        "n_selected_features",
        "n_active_receiver_tokens",
        "selected_features",
        "repair",
        "strong_repair",
    ]
    best_rows_by_key: Dict[Tuple[int, int, int, str, str, str, int], Dict[str, Any]] = {}
    candidate_row_count = 0
    empty_sparse = np.zeros((0, int(model_bundle["d_latent"])), dtype=np.float32)
    pair_batch_size = max(1, int(args.patch_chunk_size or args.batch_size))

    with candidate_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=candidate_fieldnames)
        writer.writeheader()

        for context_mode in context_modes:
            for donor_type in ["benign", "anomalous"]:
                pairs = candidate_pairs_by_key[(context_mode, donor_type)]
                if not pairs:
                    continue

                for start in range(0, len(pairs), pair_batch_size):
                    pair_batch = pairs[start : start + pair_batch_size]
                    recv_idx = np.asarray([p[0] for p in pair_batch], dtype=int)
                    donor_idx = np.asarray([p[1] for p in pair_batch], dtype=int)
                    receiver_texts = [texts[i] for i in recv_idx.tolist()]
                    base = base_scores[recv_idx]
                    batch_example_ids = sorted(set(recv_idx.tolist()) | set(donor_idx.tolist()))
                    sparse_cache = build_sparse_cache_for_examples(
                        sae_model,
                        x,
                        token_slices,
                        batch_example_ids,
                        x_mean=x_mean,
                        x_std=x_std,
                        device=device,
                        batch_size=args.sae_batch_size,
                    )

                    for feature_set_name in top_sets + [control_set]:
                        ids = feature_sets[feature_set_name]
                        if not ids:
                            continue
                        selected_json = json.dumps([int(fid) for fid in ids])
                        for alpha in alphas:
                            patch_list: List[np.ndarray] = []
                            active_counts: List[int] = []
                            for r_idx, d_idx in zip(recv_idx.tolist(), donor_idx.tolist()):
                                recv_slice = token_slices.get(int(r_idx))
                                donor_slice = token_slices.get(int(d_idx))
                                if recv_slice is None:
                                    raise KeyError(f"Missing token slice for receiver example_idx={r_idx}")
                                recv_delta_tokens = np.asarray(x[recv_slice], dtype=np.float32)
                                recv_sparse_tokens = sparse_cache[int(r_idx)]
                                donor_sparse_tokens = sparse_cache.get(int(d_idx), empty_sparse) if donor_slice is not None else empty_sparse
                                shift, n_active = build_token_patch_shift(
                                    sae_model,
                                    recv_delta_tokens,
                                    recv_sparse_tokens,
                                    donor_sparse_tokens,
                                    ids,
                                    alpha,
                                    x_mean=x_mean,
                                    x_std=x_std,
                                    device=device,
                                    batch_size=args.sae_batch_size,
                                )
                                patch_list.append(shift)
                                active_counts.append(n_active)

                            patched_scores = score_with_token_patches(
                                adapted_model,
                                tokenizer,
                                receiver_texts,
                                patch_list,
                                layer=args.layer,
                                max_seq_len=int(cfg["training"]["max_seq_len"]),
                                batch_size=args.batch_size,
                                loss_batch_size=args.loss_batch_size,
                                full_logits_max_gib=args.full_logits_max_gib,
                                max_logit_elements=args.max_logit_elements,
                            )
                            deltas = patched_scores - base
                            batch_rows: List[Dict[str, Any]] = []
                            for i in range(len(recv_idx)):
                                row = {
                                    "layer": int(args.layer),
                                    "latent_mult": int(args.latent_mult),
                                    "k": int(args.k),
                                    "context_mode": context_mode,
                                    "feature_set": feature_set_name,
                                    "donor_type": donor_type,
                                    "receiver_row_idx": int(recv_idx[i]),
                                    "donor_row_idx": int(donor_idx[i]),
                                    "receiver_example_id": str(example_meta.iloc[recv_idx[i]]["example_id"]),
                                    "donor_example_id": str(example_meta.iloc[donor_idx[i]]["example_id"]),
                                    "alpha": float(alpha),
                                    "base_score": float(base[i]),
                                    "patched_score": float(patched_scores[i]),
                                    "delta": float(deltas[i]),
                                    "n_selected_features": int(len(ids)),
                                    "n_active_receiver_tokens": int(active_counts[i]),
                                    "selected_features": selected_json,
                                    "repair": bool(deltas[i] < 0.0),
                                    "strong_repair": bool(deltas[i] <= -args.repair_threshold),
                                }
                                batch_rows.append(row)
                                key = (
                                    int(args.layer),
                                    int(args.latent_mult),
                                    int(args.k),
                                    str(context_mode),
                                    str(feature_set_name),
                                    str(donor_type),
                                    int(recv_idx[i]),
                                )
                                prev = best_rows_by_key.get(key)
                                if prev is None or float(row["delta"]) < float(prev["delta"]):
                                    best_rows_by_key[key] = dict(row)
                            writer.writerows(batch_rows)
                            candidate_row_count += len(batch_rows)

    best_df = pd.DataFrame(best_rows_by_key.values())
    if best_df.empty:
        best_df = pd.DataFrame()
        summary_df = pd.DataFrame()
    else:
        best_df = best_df.sort_values(
            ["context_mode", "feature_set", "donor_type", "receiver_row_idx", "delta"],
            ascending=[True, True, True, True, True],
        ).reset_index(drop=True)
        summary_df = summarize_best(best_df, top_sets=top_sets, control_set=control_set)

    selected_df = pd.DataFrame(selected_rows)
    best_df.to_csv(out_dir / "token_delta_sae_causal_best_rows.csv", index=False)
    summary_df.to_csv(out_dir / "token_delta_sae_causal_summary.csv", index=False)
    selected_df.to_csv(out_dir / "token_delta_sae_causal_selected_sets.csv", index=False)
    write_report(
        summary_df,
        selected_df,
        best_df,
        out_dir / "TOKEN_DELTA_SAE_CAUSAL_REPORT.md",
        layer=args.layer,
        latent_mult=args.latent_mult,
        k=args.k,
        control_set=control_set,
        exclude_same_user=bool(args.exclude_same_user_donors),
    )

    stats = {
        "layer": int(args.layer),
        "latent_mult": int(args.latent_mult),
        "k": int(args.k),
        "n_examples": int(len(example_meta)),
        "n_positive_receivers": int((example_meta["y"] == 1).sum()),
        "n_token_rows": int(len(token_example_idx)),
        "top_sets": top_sets,
        "control_set": control_set,
        "active_control_min_frac": float(args.active_control_min_frac),
        "active_control_sizes": active_control_sizes,
        "context_modes": context_modes,
        "alphas": alphas,
        "candidate_rows": int(candidate_row_count),
        "patch_chunk_size": int(pair_batch_size),
        "token_delta_dtype": str(args.token_delta_dtype),
    }
    dump_json(out_dir / "token_delta_sae_causal_summary.json", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
