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

from eval_token_delta_sae_causal import (
    build_example_slices,
    build_sparse_cache_for_examples,
    context_key,
    load_eval_examples,
    load_token_layer_vectors,
    normalize_token_block,
    parse_csv_list,
    required_context_cols,
    resolve_context_cols,
    score_with_token_patches,
    table_text,
)
from remote_common import dump_json, ensure_dir, load_yaml
from sae_core import TopKSAE, add_active_control_feature_sets, choose_feature_sets


def build_receiver_pairs(
    meta: pd.DataFrame,
    *,
    context_mode: str,
    max_pairs: int,
    rng: np.random.Generator,
    exclude_same_user: bool = False,
) -> List[Tuple[int, int, int]]:
    ctx_cols = resolve_context_cols(meta, context_mode)
    primary_col = ctx_cols[0]
    work = meta[["row_idx", "user_id", "y"] + ctx_cols].copy()
    work["ctx_key"] = context_key(work, context_mode)

    pos_df = work.loc[work["y"] == 1].copy()
    pos_indices = pos_df["row_idx"].to_numpy(dtype=int)
    if max_pairs > 0 and len(pos_indices) > max_pairs:
        chosen = np.sort(rng.choice(pos_indices, size=max_pairs, replace=False))
        pos_df = pos_df.set_index("row_idx").loc[chosen].reset_index()

    benign_df = work.loc[work["y"] == 0].copy()
    benign_all = np.sort(benign_df["row_idx"].to_numpy(dtype=int))
    benign_by_key = {
        str(key): np.sort(sub["row_idx"].to_numpy(dtype=int))
        for key, sub in benign_df.groupby("ctx_key", sort=False)
    }
    benign_by_primary = {
        str(key): np.sort(sub["row_idx"].to_numpy(dtype=int))
        for key, sub in benign_df.groupby(primary_col, sort=False)
    }
    benign_user_map = benign_df.set_index("row_idx")["user_id"].astype(str).to_dict()

    pairs: List[Tuple[int, int, int]] = []
    pair_idx = 0
    for _, row in pos_df.iterrows():
        pos_idx = int(row["row_idx"])
        pos_user = str(row["user_id"])
        pool = benign_by_key.get(str(row["ctx_key"]))
        if pool is None or len(pool) == 0:
            pool = benign_by_primary.get(str(row[primary_col]))
        if pool is None or len(pool) == 0:
            pool = benign_all
        if len(pool) > 0 and exclude_same_user:
            pool = pool[np.asarray([benign_user_map.get(int(d), "") != pos_user for d in pool], dtype=bool)]
        if pool is None or len(pool) == 0:
            continue
        benign_idx = int(rng.choice(pool))
        pairs.append((pair_idx, pos_idx, benign_idx))
        pair_idx += 1
    return pairs


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
    return np.concatenate(out, axis=0) if out else np.empty((0, x_mean.shape[1]), dtype=np.float32)


def build_token_ablation_shift(
    sae_model: TopKSAE,
    recv_delta_tokens: np.ndarray,
    recv_sparse_tokens: np.ndarray,
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

    sparse_cf = np.array(recv_sparse_tokens, copy=True)
    for feat_id in feature_ids:
        sparse_cf[active_idx, feat_id] = (1.0 - alpha) * recv_sparse_tokens[active_idx, feat_id]

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
            index=["pair_idx"],
            columns=["feature_set", "receiver_type"],
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
                "context_mode": str(context_mode),
                "target": str(top_set),
                "n_pairs": int(len(work)),
            }
            for col in [
                f"{top_set}_positive",
                f"{top_set}_benign",
                f"{control_set}_positive",
                f"{control_set}_benign",
            ]:
                if col not in work.columns:
                    work[col] = float("nan")
            row["top_positive_mean_best_delta"] = float(work[f"{top_set}_positive"].mean())
            row["top_benign_mean_best_delta"] = float(work[f"{top_set}_benign"].mean())
            row["control_positive_mean_best_delta"] = float(work[f"{control_set}_positive"].mean())
            row["control_benign_mean_best_delta"] = float(work[f"{control_set}_benign"].mean())
            row["top_necessity_advantage"] = (
                row["top_benign_mean_best_delta"] - row["top_positive_mean_best_delta"]
            )
            row["control_necessity_advantage"] = (
                row["control_benign_mean_best_delta"] - row["control_positive_mean_best_delta"]
            )
            row["top_minus_control_necessity"] = (
                row["top_necessity_advantage"] - row["control_necessity_advantage"]
            )
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["top_minus_control_necessity", "top_necessity_advantage", "top_positive_mean_best_delta"],
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
    exclude_same_user_matches: bool,
) -> None:
    lines = [
        "# Token Delta SAE Necessity Eval",
        "",
        f"Token-level feature ablation on adapter deltas at hidden-state layer `{layer}` with SAE config `latent_mult={latent_mult}, k={k}`.",
        "",
        "Intervention protocol:",
        "- receivers = paired positive and matched benign eval examples",
        f"- same-user benign matches excluded: `{bool(exclude_same_user_matches)}`",
        "- pairs are matched by requested context mode with fallback to broader benign pools",
        "- feature sets = top sparse sets ablated in token-level delta-SAE space, compared against the control set",
        "- only receiver token positions where the target sparse features are active are modified",
        "- ablation shrinks selected sparse feature activations toward zero by alpha",
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
        "## Example Receiver-Level Best Ablations",
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
    ap.add_argument("--control-set", default="control5_active")
    ap.add_argument("--active-control-min-frac", type=float, default=0.002)
    ap.add_argument("--active-control-sizes", default="1,3,5")
    ap.add_argument("--alphas", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--effect-threshold", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-same-user-matches", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("Missing runtime dependencies for token necessity eval.") from exc

    context_modes = parse_csv_list(args.context_modes)
    top_sets = parse_csv_list(args.top_sets)
    active_control_sizes = [int(x) for x in parse_csv_list(args.active_control_sizes)]
    alphas = [float(x) for x in parse_csv_list(args.alphas)]
    control_set = str(args.control_set)

    out_dir = ensure_dir(args.out_dir)
    scores = pd.read_parquet(args.extract_dir / "example_scores.parquet").sort_values("example_idx").reset_index(drop=True)
    example_meta = load_eval_examples(args.data_dir, scores)
    _ = required_context_cols(example_meta, context_modes)

    pairs_by_context: Dict[str, List[Tuple[int, int, int]]] = {}
    needed_examples: set[int] = set()
    for mode_i, context_mode in enumerate(context_modes):
        pairs = build_receiver_pairs(
            example_meta,
            context_mode=context_mode,
            max_pairs=args.max_pairs,
            rng=np.random.default_rng(args.seed + 1009 * mode_i),
            exclude_same_user=bool(args.exclude_same_user_matches),
        )
        pairs_by_context[context_mode] = pairs
        for pair_idx, pos_idx, benign_idx in pairs:
            del pair_idx
            needed_examples.add(int(pos_idx))
            needed_examples.add(int(benign_idx))

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
    candidate_path = out_dir / "token_delta_sae_necessity_candidate_rows.csv"
    candidate_fieldnames = [
        "layer",
        "latent_mult",
        "k",
        "context_mode",
        "feature_set",
        "receiver_type",
        "pair_idx",
        "receiver_row_idx",
        "matched_row_idx",
        "receiver_example_id",
        "matched_example_id",
        "alpha",
        "base_score",
        "patched_score",
        "delta",
        "n_selected_features",
        "n_active_receiver_tokens",
        "selected_features",
        "effect",
        "strong_effect",
    ]
    best_rows_by_key: Dict[Tuple[int, int, int, str, str, str, int], Dict[str, Any]] = {}
    candidate_row_count = 0
    pair_batch_size = max(1, int(args.patch_chunk_size or args.batch_size))

    with candidate_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=candidate_fieldnames)
        writer.writeheader()

        for context_mode in context_modes:
            pairs = pairs_by_context[context_mode]
            if not pairs:
                continue

            for start in range(0, len(pairs), pair_batch_size):
                pair_batch = pairs[start : start + pair_batch_size]
                batch_example_ids = sorted({int(x) for _, p, b in pair_batch for x in (p, b)})
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
                        scored_meta: List[Tuple[int, str, int, int, float, int]] = []
                        batch_texts: List[str] = []
                        batch_patches: List[np.ndarray] = []

                        for pair_idx, pos_idx, benign_idx in pair_batch:
                            for receiver_type, recv_idx, matched_idx in [
                                ("positive", int(pos_idx), int(benign_idx)),
                                ("benign", int(benign_idx), int(pos_idx)),
                            ]:
                                recv_slice = token_slices.get(int(recv_idx))
                                if recv_slice is None:
                                    raise KeyError(f"Missing token slice for receiver example_idx={recv_idx}")
                                recv_delta_tokens = np.asarray(x[recv_slice], dtype=np.float32)
                                recv_sparse_tokens = sparse_cache[int(recv_idx)]
                                shift, n_active = build_token_ablation_shift(
                                    sae_model,
                                    recv_delta_tokens,
                                    recv_sparse_tokens,
                                    ids,
                                    alpha,
                                    x_mean=x_mean,
                                    x_std=x_std,
                                    device=device,
                                    batch_size=args.sae_batch_size,
                                )
                                batch_texts.append(texts[int(recv_idx)])
                                batch_patches.append(shift)
                                scored_meta.append(
                                    (int(pair_idx), str(receiver_type), int(recv_idx), int(matched_idx), float(base_scores[int(recv_idx)]), int(n_active))
                                )

                        patched_scores = score_with_token_patches(
                            adapted_model,
                            tokenizer,
                            batch_texts,
                            batch_patches,
                            layer=args.layer,
                            max_seq_len=int(cfg["training"]["max_seq_len"]),
                            batch_size=args.batch_size,
                            loss_batch_size=args.loss_batch_size,
                            full_logits_max_gib=args.full_logits_max_gib,
                            max_logit_elements=args.max_logit_elements,
                        )

                        batch_rows: List[Dict[str, Any]] = []
                        for i, meta_row in enumerate(scored_meta):
                            pair_idx, receiver_type, recv_idx, matched_idx, base_score, n_active = meta_row
                            delta = float(patched_scores[i] - base_score)
                            row = {
                                "layer": int(args.layer),
                                "latent_mult": int(args.latent_mult),
                                "k": int(args.k),
                                "context_mode": str(context_mode),
                                "feature_set": str(feature_set_name),
                                "receiver_type": str(receiver_type),
                                "pair_idx": int(pair_idx),
                                "receiver_row_idx": int(recv_idx),
                                "matched_row_idx": int(matched_idx),
                                "receiver_example_id": str(example_meta.iloc[recv_idx]["example_id"]),
                                "matched_example_id": str(example_meta.iloc[matched_idx]["example_id"]),
                                "alpha": float(alpha),
                                "base_score": float(base_score),
                                "patched_score": float(patched_scores[i]),
                                "delta": delta,
                                "n_selected_features": int(len(ids)),
                                "n_active_receiver_tokens": int(n_active),
                                "selected_features": selected_json,
                                "effect": bool(delta < 0.0),
                                "strong_effect": bool(delta <= -args.effect_threshold),
                            }
                            batch_rows.append(row)
                            key = (
                                int(args.layer),
                                int(args.latent_mult),
                                int(args.k),
                                str(context_mode),
                                str(feature_set_name),
                                str(receiver_type),
                                int(pair_idx),
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
            ["context_mode", "feature_set", "receiver_type", "pair_idx", "delta"],
            ascending=[True, True, True, True, True],
        ).reset_index(drop=True)
        summary_df = summarize_best(best_df, top_sets=top_sets, control_set=control_set)

    selected_df = pd.DataFrame(selected_rows)
    best_df.to_csv(out_dir / "token_delta_sae_necessity_best_rows.csv", index=False)
    summary_df.to_csv(out_dir / "token_delta_sae_necessity_summary.csv", index=False)
    selected_df.to_csv(out_dir / "token_delta_sae_necessity_selected_sets.csv", index=False)
    write_report(
        summary_df,
        selected_df,
        best_df,
        out_dir / "TOKEN_DELTA_SAE_NECESSITY_REPORT.md",
        layer=args.layer,
        latent_mult=args.latent_mult,
        k=args.k,
        control_set=control_set,
        exclude_same_user_matches=bool(args.exclude_same_user_matches),
    )

    stats = {
        "layer": int(args.layer),
        "latent_mult": int(args.latent_mult),
        "k": int(args.k),
        "n_examples": int(len(example_meta)),
        "n_positive_receivers": int((example_meta["y"] == 1).sum()),
        "n_pairs_total": int(sum(len(v) for v in pairs_by_context.values())),
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
    dump_json(out_dir / "token_delta_sae_necessity_summary.json", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
