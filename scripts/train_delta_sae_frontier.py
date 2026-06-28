#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from remote_common import dump_json, ensure_dir, load_yaml
from sae_core import (
    TopKSAE,
    choose_feature_sets,
    decoder_overlap_stats,
    evaluate_features,
    support_overlap_stats_streaming,
    tensor_stats,
    train_sae,
)


def load_layer_vectors(
    extract_dir: Path,
    layer: int,
    *,
    max_rows: int = 0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    scores = pd.read_parquet(extract_dir / "example_scores.parquet")
    score_index = scores.sort_values("example_idx").reset_index(drop=True).set_index("example_idx", drop=False)
    layer_dir = extract_dir / f"layer_{layer}"
    chunk_paths = sorted(layer_dir.glob("chunk_*.pt"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunks found for layer {layer} in {layer_dir}")
    sample_prob = 1.0
    if max_rows > 0:
        manifest_path = extract_dir / "chunk_manifest.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
            total_rows = int(manifest.loc[manifest["layer"].astype(int) == layer, "rows"].sum())
            if total_rows > max_rows:
                sample_prob = min(1.0, (max_rows * 1.1) / max(total_rows, 1))
    rng = np.random.default_rng(seed)
    all_vecs: List[np.ndarray] = []
    all_idx: List[np.ndarray] = []
    all_pos: List[np.ndarray] = []
    has_position = False
    for path in chunk_paths:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        vecs = np.asarray(obj["delta"], dtype=np.float32)
        idx = np.asarray(obj["example_idx"], dtype=np.int64)
        pos = None
        if "position" in obj and obj["position"] is not None:
            has_position = True
            pos = np.asarray(obj["position"], dtype=np.int64)
        if sample_prob < 1.0:
            labels = score_index.loc[idx, "y"].to_numpy(dtype=np.int64)
            keep = (labels > 0) | (rng.random(len(idx)) < sample_prob)
            vecs = vecs[keep]
            idx = idx[keep]
            if pos is not None:
                pos = pos[keep]
        all_vecs.append(vecs)
        all_idx.append(idx)
        if pos is not None:
            all_pos.append(pos)
    x = np.concatenate(all_vecs, axis=0)
    example_idx = np.concatenate(all_idx, axis=0)
    del all_vecs, all_idx
    if x.shape[0] != len(example_idx):
        raise ValueError(f"Mismatched vector/index sizes for layer {layer}")
    if has_position:
        position = np.concatenate(all_pos, axis=0)
        del all_pos
        order = np.lexsort((position, example_idx))
    else:
        position = None
        order = np.argsort(example_idx)
    x = x[order]
    example_idx = example_idx[order]
    if position is not None:
        position = position[order]
        if not np.all(np.isin(np.unique(example_idx), score_index.index.to_numpy())):
            raise ValueError("Some token rows point at unknown example_idx values")
        token_meta = score_index.loc[example_idx].reset_index(drop=True).copy()
        token_meta["position"] = position.astype(np.int64)
        y = token_meta["y"].to_numpy(dtype=np.int64)
        return x, y, token_meta
    if sample_prob < 1.0:
        order = np.argsort(example_idx)
        x = x[order]
        example_idx = example_idx[order]
        sampled_meta = score_index.loc[example_idx].reset_index(drop=True)
        y = sampled_meta["y"].to_numpy(dtype=np.int64)
        return x, y, sampled_meta
    scores = score_index.reset_index(drop=True)
    if not np.array_equal(example_idx, scores["example_idx"].to_numpy()):
        raise ValueError("Example ordering mismatch between vectors and scores")
    y = scores["y"].to_numpy(dtype=np.int64)
    return x, y, scores.reset_index(drop=True)


def subsample_rows(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    *,
    max_rows: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if max_rows <= 0 or len(x) <= max_rows:
        return x, y, meta
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y > 0)
    neg_idx = np.flatnonzero(y <= 0)
    if len(pos_idx) >= max_rows:
        chosen = np.sort(rng.choice(pos_idx, size=max_rows, replace=False))
    else:
        need_neg = max_rows - len(pos_idx)
        if len(neg_idx) > need_neg:
            neg_take = np.sort(rng.choice(neg_idx, size=need_neg, replace=False))
        else:
            neg_take = neg_idx
        chosen = np.sort(np.concatenate([pos_idx, neg_take], axis=0))
    return x[chosen], y[chosen], meta.iloc[chosen].reset_index(drop=True)


def energy_selectivity_summary(
    model: TopKSAE,
    x: np.ndarray,
    y: np.ndarray,
    feature_sets: Dict[str, List[int]],
    *,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    ds = torch.utils.data.TensorDataset(torch.from_numpy(x))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    removed: Dict[str, List[np.ndarray]] = {k: [] for k in feature_sets}
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            recon, z = model(xb)
            for name, ids in feature_sets.items():
                z_patch = z.clone()
                z_patch[:, ids] = 0.0
                recon_patch = model.decoder(z_patch)
                removed_energy = ((recon - recon_patch) ** 2).mean(dim=1).cpu().numpy()
                removed[name].append(removed_energy)
    rows = []
    for name, pieces in removed.items():
        vals = np.concatenate(pieces, axis=0)
        for label, recv in [(1, "anomaly"), (0, "benign")]:
            mask = y == label
            rows.append(
                {
                    "intervention": name,
                    "receiver_name": recv,
                    "delta_mean": float(vals[mask].mean()) if mask.any() else float("nan"),
                    "n_rows": int(mask.sum()),
                }
            )
    raw = pd.DataFrame(rows)
    wide = raw.pivot_table(index=["intervention"], columns="receiver_name", values=["delta_mean", "n_rows"], aggfunc="mean")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["selectivity_proxy"] = wide["delta_mean_anomaly"] - wide["delta_mean_benign"]
    return wide


def write_report(summary_df: pd.DataFrame, path: Path) -> None:
    try:
        table = summary_df.to_markdown(index=False)
    except Exception:
        table = summary_df.to_string(index=False)
    lines = [
        "# Delta SAE Frontier",
        "",
        "This is the first runnable frontier on adapter deltas.",
        "",
        "Important: the top-vs-control numbers here are **proxy selectivity metrics** based on removed reconstruction energy inside the delta-SAE, not full model-level causal patching yet.",
        "",
        table,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--layers", default="", help="comma-separated subset of layers to train; default = all extracted layers")
    ap.add_argument("--max-rows", type=int, default=0, help="optional cap on rows per layer for large token-level runs")
    ap.add_argument("--latent-multipliers", default="", help="optional comma-separated latent multipliers")
    ap.add_argument("--topk", default="", help="optional comma-separated top-k values")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    summary = json.loads((args.extract_dir / "extract_summary.json").read_text())
    layers = [int(x) for x in summary["layers"]]
    if args.layers.strip():
        want = {int(x) for x in args.layers.split(",") if x.strip()}
        layers = [x for x in layers if x in want]
        if not layers:
            raise ValueError("No requested layers found in extract summary")
    if args.latent_multipliers.strip():
        latent_mults = [int(x.strip()) for x in args.latent_multipliers.split(",") if x.strip()]
    else:
        latent_mults = [int(x) for x in cfg["sweep"]["latent_multipliers"]]
    if args.topk.strip():
        ks = [int(x.strip()) for x in args.topk.split(",") if x.strip()]
    else:
        ks = [int(x) for x in cfg["sweep"]["topk"]]
    out_dir = ensure_dir(args.out_dir)
    device = torch.device(args.device)
    unit = str(summary.get("pool_unit", "mean"))

    rows: List[Dict[str, object]] = []
    for layer in layers:
        x, y, scores = load_layer_vectors(
            args.extract_dir,
            layer,
            max_rows=args.max_rows,
            seed=args.seed + layer,
        )
        x, y, scores = subsample_rows(x, y, scores, max_rows=args.max_rows, seed=args.seed + layer)
        mean, std = tensor_stats(x)
        x -= mean
        x /= std
        x_norm = x
        d_in = x_norm.shape[1]
        del scores
        for latent_mult in latent_mults:
            d_latent = d_in * latent_mult
            for k in ks:
                print(f"[frontier] layer={layer} d_in={d_in} d_latent={d_latent} k={k}", flush=True)
                model, train_stats = train_sae(
                    x_norm,
                    device=device,
                    d_latent=d_latent,
                    k=k,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                )
                feature_df, eval_stats = evaluate_features(
                    model,
                    x_norm,
                    y,
                    device=device,
                    batch_size=args.batch_size,
                )
                feature_sets = choose_feature_sets(feature_df)
                proxy = energy_selectivity_summary(
                    model,
                    x_norm,
                    y,
                    feature_sets,
                    device=device,
                    batch_size=args.batch_size,
                )
                proxy_map = {r["intervention"]: float(r["selectivity_proxy"]) for _, r in proxy.iterrows()}
                top_ids = [int(x) for x in feature_df.head(10)["feature_id"].tolist()]
                overlap = decoder_overlap_stats(model, top_ids)
                support = support_overlap_stats_streaming(
                    model,
                    x_norm,
                    top_ids,
                    device=device,
                    batch_size=args.batch_size,
                )

                cfg_dir = ensure_dir(out_dir / f"layer_{layer}" / f"m{latent_mult:02d}_k{k:02d}")
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "layer": layer,
                        "unit": unit,
                        "d_in": d_in,
                        "d_latent": d_latent,
                        "k": k,
                        "x_mean": mean,
                        "x_std": std,
                    },
                    cfg_dir / "delta_sae_model.pt",
                )
                feature_df.to_csv(cfg_dir / "delta_sae_top_features.csv", index=False)
                proxy.to_csv(cfg_dir / "delta_sae_proxy_selectivity.csv", index=False)
                row = {
                    "unit": unit,
                    "layer": layer,
                    "latent_mult": latent_mult,
                    "k": k,
                    "n_rows": int(len(x)),
                    "d_in": d_in,
                    "d_latent": d_latent,
                    **train_stats,
                    **eval_stats,
                    **overlap,
                    **support,
                    "top1_selectivity_proxy": proxy_map.get("top1", float("nan")),
                    "top3_selectivity_proxy": proxy_map.get("top3", float("nan")),
                    "top5_selectivity_proxy": proxy_map.get("top5", float("nan")),
                    "control1_selectivity_proxy": proxy_map.get("control1", float("nan")),
                    "control3_selectivity_proxy": proxy_map.get("control3", float("nan")),
                }
                row["top5_minus_control3_advantage_proxy"] = row["top5_selectivity_proxy"] - row["control3_selectivity_proxy"]
                rows.append(row)

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["unit", "layer", "top5_minus_control3_advantage_proxy", "top10_row_gap_mean", "recon_mse"],
            ascending=[True, True, False, False, True],
        ).reset_index(drop=True)
    summary_df.to_csv(out_dir / "delta_sae_frontier_summary.csv", index=False)
    write_report(summary_df, out_dir / "DELTA_SAE_FRONTIER_REPORT.md")
    dump_json(out_dir / "delta_sae_frontier_summary.json", {"layers": layers, "n_rows": int(len(summary_df))})
    print(f"[done] wrote {out_dir / 'delta_sae_frontier_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
