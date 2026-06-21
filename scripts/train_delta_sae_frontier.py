#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from remote_common import dump_json, ensure_dir, load_yaml
from sae_core import (
    TopKSAE,
    choose_feature_sets,
    decoder_overlap_stats,
    evaluate_features,
    support_overlap_stats,
    tensor_stats,
    train_sae,
)


def load_layer_vectors(extract_dir: Path, layer: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    scores = pd.read_parquet(extract_dir / "example_scores.parquet")
    layer_dir = extract_dir / f"layer_{layer}"
    chunk_paths = sorted(layer_dir.glob("chunk_*.pt"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunks found for layer {layer} in {layer_dir}")
    all_vecs: List[np.ndarray] = []
    all_idx: List[np.ndarray] = []
    for path in chunk_paths:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        vecs = np.asarray(obj["delta"], dtype=np.float32)
        idx = np.asarray(obj["example_idx"], dtype=np.int64)
        all_vecs.append(vecs)
        all_idx.append(idx)
    x = np.concatenate(all_vecs, axis=0)
    example_idx = np.concatenate(all_idx, axis=0)
    if x.shape[0] != len(example_idx):
        raise ValueError(f"Mismatched vector/index sizes for layer {layer}")
    if len(np.unique(example_idx)) != len(scores):
        grp = pd.DataFrame({"example_idx": example_idx})
        grp["row_id"] = np.arange(len(grp))
        # token-level chunks: mean-pool to example level
        raise RuntimeError(
            "Token-level extraction is not yet supported in train_delta_sae_frontier.py. Re-run extraction with --pool-unit mean."
        )
    scores = scores.sort_values("example_idx").reset_index(drop=True)
    order = np.argsort(example_idx)
    x = x[order]
    example_idx = example_idx[order]
    if not np.array_equal(example_idx, scores["example_idx"].to_numpy()):
        raise ValueError("Example ordering mismatch between vectors and scores")
    y = scores["y"].to_numpy(dtype=np.int64)
    return x, y, scores


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
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    summary = json.loads((args.extract_dir / "extract_summary.json").read_text())
    layers = [int(x) for x in summary["layers"]]
    latent_mults = [int(x) for x in cfg["sweep"]["latent_multipliers"]]
    ks = [int(x) for x in cfg["sweep"]["topk"]]
    out_dir = ensure_dir(args.out_dir)
    device = torch.device(args.device)

    rows: List[Dict[str, object]] = []
    for layer in layers:
        x, y, scores = load_layer_vectors(args.extract_dir, layer)
        mean, std = tensor_stats(x)
        x_norm = (x - mean) / std
        d_in = x.shape[1]
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
                feature_df, eval_stats, z_all = evaluate_features(
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
                support = support_overlap_stats(z_all, top_ids)

                cfg_dir = ensure_dir(out_dir / f"layer_{layer}" / f"m{latent_mult:02d}_k{k:02d}")
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "layer": layer,
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
                    "layer": layer,
                    "latent_mult": latent_mult,
                    "k": k,
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
            ["layer", "top5_minus_control3_advantage_proxy", "top10_row_gap_mean", "recon_mse"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)
    summary_df.to_csv(out_dir / "delta_sae_frontier_summary.csv", index=False)
    write_report(summary_df, out_dir / "DELTA_SAE_FRONTIER_REPORT.md")
    dump_json(out_dir / "delta_sae_frontier_summary.json", {"layers": layers, "n_rows": int(len(summary_df))})
    print(f"[done] wrote {out_dir / 'delta_sae_frontier_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
