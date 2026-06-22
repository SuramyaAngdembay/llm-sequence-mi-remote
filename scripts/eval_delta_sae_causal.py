#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from remote_common import dump_json, ensure_dir, load_yaml, read_jsonl
from sae_core import TopKSAE, choose_feature_sets


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


def per_example_nll(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()
    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.size())
    denom = shift_mask.sum(dim=1).clamp_min(1.0)
    return (token_loss * shift_mask).sum(dim=1) / denom


def load_layer_vectors(extract_dir: Path, layer: int) -> tuple[np.ndarray, pd.DataFrame]:
    scores = pd.read_parquet(extract_dir / "example_scores.parquet").sort_values("example_idx").reset_index(drop=True)
    layer_dir = extract_dir / f"layer_{layer}"
    chunk_paths = sorted(layer_dir.glob("chunk_*.pt"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunks found for layer {layer} in {layer_dir}")
    all_vecs: List[np.ndarray] = []
    all_idx: List[np.ndarray] = []
    for path in chunk_paths:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if "position" in obj and obj["position"] is not None:
            raise RuntimeError("Token-level extraction is not supported here. Use mean-pooled extraction.")
        all_vecs.append(np.asarray(obj["delta"], dtype=np.float32))
        all_idx.append(np.asarray(obj["example_idx"], dtype=np.int64))
    x = np.concatenate(all_vecs, axis=0)
    idx = np.concatenate(all_idx, axis=0)
    order = np.argsort(idx)
    x = x[order]
    idx = idx[order]
    if not np.array_equal(idx, scores["example_idx"].to_numpy(dtype=np.int64)):
        raise ValueError("Vector/example ordering mismatch")
    return x, scores


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
    merged = pd.concat([meta, scores.drop(columns=["example_id", "user_id", "day_index", "split", "y", "n_sessions_total", "n_sessions_kept"])], axis=1)
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
) -> List[Tuple[int, int]]:
    ctx_cols = resolve_context_cols(meta, context_mode)
    primary_col = ctx_cols[0]
    work = meta[["row_idx", "y"] + ctx_cols].copy()
    work["ctx_key"] = context_key(work, context_mode)

    recv_df = work.loc[work["y"] == 1].copy()
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

    pairs: List[Tuple[int, int]] = []
    for _, row in recv_df.iterrows():
        recv_idx = int(row["row_idx"])
        pool = donor_by_key.get(str(row["ctx_key"]))
        if pool is None or len(pool) == 0:
            pool = donor_by_primary.get(str(row[primary_col]))
        if pool is None or len(pool) == 0:
            pool = donor_all
        if donor_label == 1 and len(pool) > 1:
            pool = pool[pool != recv_idx]
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
    return np.concatenate(zs, axis=0)


@torch.no_grad()
def decode_patched_deltas(
    sae_model: TopKSAE,
    sparse_recv: np.ndarray,
    sparse_donor: np.ndarray,
    feature_ids: Sequence[int],
    alpha: float,
    *,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    sparse_cf = np.array(sparse_recv, copy=True)
    if feature_ids:
        idx = [int(x) for x in feature_ids]
        sparse_cf[:, idx] = (1.0 - alpha) * sparse_recv[:, idx] + alpha * sparse_donor[:, idx]
    out: List[np.ndarray] = []
    x_mean_t = torch.from_numpy(x_mean).to(device)
    x_std_t = torch.from_numpy(x_std).to(device)
    for start in range(0, len(sparse_cf), batch_size):
        z = torch.from_numpy(sparse_cf[start : start + batch_size]).to(device)
        x_patch_norm = sae_model.decoder(z)
        x_patch = (x_patch_norm * x_std_t + x_mean_t).cpu().numpy().astype(np.float32)
        out.append(x_patch)
    return np.concatenate(out, axis=0)


@torch.no_grad()
def score_with_patched_layer(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    patch_shift: np.ndarray,
    *,
    layer: int,
    max_seq_len: int,
    batch_size: int,
) -> np.ndarray:
    layer_module = get_layer_module(model, layer)
    device = next(model.parameters()).device
    out_scores: List[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        shift = torch.from_numpy(patch_shift[start : start + batch_size]).to(device)
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
            delta = shift.to(hs.dtype).unsqueeze(1) * attn.unsqueeze(-1).to(hs.dtype)
            hs = hs + delta
            if rest is None:
                return hs
            return (hs, *rest)

        handle = layer_module.register_forward_hook(hook)
        try:
            out = model(**tok, return_dict=True)
        finally:
            handle.remove()
        nll = per_example_nll(out.logits.float(), tok["input_ids"], tok["attention_mask"]).cpu().numpy()
        out_scores.append(nll)

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
            row = {
                "layer": int(layer),
                "latent_mult": int(latent_mult),
                "k": int(k),
                "context_mode": context_mode,
                "target": top_set,
                "n_receivers": int(len(work)),
            }
            for col in [f"{top_set}_benign", f"{top_set}_anomalous", f"{control_set}_benign", f"{control_set}_anomalous"]:
                if col not in work.columns:
                    work[col] = float("nan")
            row["top_benign_mean_best_delta"] = float(work[f"{top_set}_benign"].mean())
            row["top_anomalous_mean_best_delta"] = float(work[f"{top_set}_anomalous"].mean())
            row["control_benign_mean_best_delta"] = float(work[f"{control_set}_benign"].mean())
            row["control_anomalous_mean_best_delta"] = float(work[f"{control_set}_anomalous"].mean())
            row["top_repair_advantage"] = row["top_anomalous_mean_best_delta"] - row["top_benign_mean_best_delta"]
            row["control_repair_advantage"] = row["control_anomalous_mean_best_delta"] - row["control_benign_mean_best_delta"]
            row["top_minus_control_advantage"] = row["top_repair_advantage"] - row["control_repair_advantage"]
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
) -> None:
    lines = [
        "# Delta SAE Causal Eval",
        "",
        f"Model-level causal patching on mean-pooled adapter deltas at hidden-state layer `{layer}` with SAE config `latent_mult={latent_mult}, k={k}`.",
        "",
        "Intervention protocol:",
        f"- receivers = positive eval examples only",
        f"- donors = matched benign donors and same-class anomalous donor controls",
        f"- feature sets = top sparse sets patched in delta-SAE space, compared against `{control_set}`",
        "- patched pooled delta vectors are converted back to layer-space and broadcast across valid tokens at the chosen layer",
        "",
        "Important: this is a first runnable model-level patch test on **mean-pooled** deltas. It is stronger than proxy selectivity, but still coarser than token-level circuit patching.",
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
    ap.add_argument("--batch-size", type=int, default=8, help="receiver scoring batch size")
    ap.add_argument("--sae-batch-size", type=int, default=1024, help="delta encode/decode batch size")
    ap.add_argument("--context-modes", default="team,role,project_role,dept_role")
    ap.add_argument("--top-sets", default="top1,top3,top5")
    ap.add_argument("--control-set", default="control3")
    ap.add_argument("--alphas", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--max-receivers", type=int, default=0)
    ap.add_argument("--max-candidate-donors", type=int, default=16)
    ap.add_argument("--repair-threshold", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Missing runtime dependencies for causal eval. Install transformers, peft, bitsandbytes.") from exc

    context_modes = parse_csv_list(args.context_modes)
    top_sets = parse_csv_list(args.top_sets)
    alphas = [float(x) for x in parse_csv_list(args.alphas)]
    control_set = str(args.control_set)

    out_dir = ensure_dir(args.out_dir)
    x, scores = load_layer_vectors(args.extract_dir, args.layer)
    meta = load_eval_examples(args.data_dir, scores)
    _ = required_context_cols(meta, context_modes)

    cfg_dir = args.frontier_dir / f"layer_{args.layer}" / f"m{args.latent_mult:02d}_k{args.k:02d}"
    model_bundle = torch.load(cfg_dir / "delta_sae_model.pt", map_location="cpu", weights_only=False)
    feature_df = pd.read_csv(cfg_dir / "delta_sae_top_features.csv")
    feature_sets = choose_feature_sets(feature_df)
    if control_set not in feature_sets:
        raise ValueError(f"Requested control set {control_set} missing from SAE feature sets")
    for name in top_sets:
        if name not in feature_sets:
            raise ValueError(f"Requested top set {name} missing from SAE feature sets")

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
    x_norm = (x - x_mean) / x_std
    sparse_all = encode_sparse_vectors(sae_model, x_norm.astype(np.float32), device=device, batch_size=args.sae_batch_size)

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

    texts = meta["text"].tolist()
    base_scores = meta["adapted_nll"].to_numpy(dtype=np.float32)
    candidate_rows: List[Dict[str, Any]] = []

    for context_mode in context_modes:
        donor_specs = [("benign", 0), ("anomalous", 1)]
        candidate_pairs = {
            donor_type: build_candidate_pairs(
                meta,
                donor_label=donor_label,
                context_mode=context_mode,
                max_receivers=args.max_receivers,
                max_candidate_donors=args.max_candidate_donors,
                rng=np.random.default_rng(args.seed + donor_label),
            )
            for donor_type, donor_label in donor_specs
        }
        for donor_type, pairs in candidate_pairs.items():
            if not pairs:
                continue
            recv_idx = np.asarray([p[0] for p in pairs], dtype=int)
            donor_idx = np.asarray([p[1] for p in pairs], dtype=int)
            receiver_texts = [texts[i] for i in recv_idx.tolist()]
            recv_delta = x[recv_idx]
            sparse_recv = sparse_all[recv_idx]
            sparse_donor = sparse_all[donor_idx]
            base = base_scores[recv_idx]

            for feature_set_name in top_sets + [control_set]:
                ids = feature_sets[feature_set_name]
                if not ids:
                    continue
                for alpha in alphas:
                    patched_delta = decode_patched_deltas(
                        sae_model,
                        sparse_recv,
                        sparse_donor,
                        ids,
                        alpha,
                        x_mean=x_mean,
                        x_std=x_std,
                        device=device,
                        batch_size=args.sae_batch_size,
                    )
                    patch_shift = patched_delta - recv_delta
                    patched_scores = score_with_patched_layer(
                        adapted_model,
                        tokenizer,
                        receiver_texts,
                        patch_shift,
                        layer=args.layer,
                        max_seq_len=int(cfg["training"]["max_seq_len"]),
                        batch_size=args.batch_size,
                    )
                    deltas = patched_scores - base
                    for i in range(len(recv_idx)):
                        candidate_rows.append(
                            {
                                "layer": int(args.layer),
                                "latent_mult": int(args.latent_mult),
                                "k": int(args.k),
                                "context_mode": context_mode,
                                "feature_set": feature_set_name,
                                "donor_type": donor_type,
                                "receiver_row_idx": int(recv_idx[i]),
                                "donor_row_idx": int(donor_idx[i]),
                                "receiver_example_id": str(meta.iloc[recv_idx[i]]["example_id"]),
                                "donor_example_id": str(meta.iloc[donor_idx[i]]["example_id"]),
                                "alpha": float(alpha),
                                "base_score": float(base[i]),
                                "patched_score": float(patched_scores[i]),
                                "delta": float(deltas[i]),
                                "n_selected_features": int(len(ids)),
                                "selected_features": json.dumps([int(x) for x in ids]),
                                "repair": bool(deltas[i] < 0.0),
                                "strong_repair": bool(deltas[i] <= -args.repair_threshold),
                            }
                        )

    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df.to_csv(out_dir / "delta_sae_causal_candidate_rows.csv", index=False)

    if candidate_df.empty:
        best_df = pd.DataFrame()
        summary_df = pd.DataFrame()
    else:
        best_df = (
            candidate_df.sort_values(
                ["context_mode", "feature_set", "donor_type", "receiver_row_idx", "delta"],
                ascending=[True, True, True, True, True],
            )
            .groupby(
                ["layer", "latent_mult", "k", "context_mode", "feature_set", "donor_type", "receiver_row_idx"],
                as_index=False,
            )
            .first()
        )
        summary_df = summarize_best(best_df, top_sets=top_sets, control_set=control_set)

    selected_df = pd.DataFrame(selected_rows)
    best_df.to_csv(out_dir / "delta_sae_causal_best_rows.csv", index=False)
    summary_df.to_csv(out_dir / "delta_sae_causal_summary.csv", index=False)
    selected_df.to_csv(out_dir / "delta_sae_causal_selected_sets.csv", index=False)
    write_report(
        summary_df,
        selected_df,
        best_df,
        out_dir / "DELTA_SAE_CAUSAL_REPORT.md",
        layer=args.layer,
        latent_mult=args.latent_mult,
        k=args.k,
        control_set=control_set,
    )

    stats = {
        "layer": int(args.layer),
        "latent_mult": int(args.latent_mult),
        "k": int(args.k),
        "n_examples": int(len(meta)),
        "n_positive_receivers": int((meta["y"] == 1).sum()),
        "top_sets": top_sets,
        "control_set": control_set,
        "context_modes": context_modes,
        "alphas": alphas,
        "candidate_rows": int(len(candidate_df)),
    }
    dump_json(out_dir / "delta_sae_causal_summary.json", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
