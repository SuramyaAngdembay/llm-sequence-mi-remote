from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class TopKSAE(nn.Module):
    def __init__(self, d_in: int, d_latent: int, k: int):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_latent)
        self.decoder = nn.Linear(d_latent, d_in, bias=False)
        self.k = k

    def encode_sparse(self, x: torch.Tensor) -> torch.Tensor:
        pre = F.relu(self.encoder(x))
        if self.k >= pre.shape[-1]:
            return pre
        vals, idx = torch.topk(pre, self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, vals)
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode_sparse(x)
        recon = self.decoder(z)
        return recon, z


def tensor_stats(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def train_sae(
    x: np.ndarray,
    *,
    device: torch.device,
    d_latent: int,
    k: int,
    batch_size: int,
    epochs: int,
    lr: float,
) -> Tuple[TopKSAE, Dict[str, float]]:
    model = TopKSAE(x.shape[1], d_latent, k).to(device)
    dataset = TensorDataset(torch.from_numpy(x))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    last_loss = float("nan")
    for epoch in range(epochs):
        model.train()
        losses: List[float] = []
        for (xb,) in loader:
            xb = xb.to(device)
            recon, _ = model(xb)
            loss = F.mse_loss(recon, xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        last_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[sae-train] epoch={epoch+1:03d} mse={last_loss:.6f}", flush=True)
    return model, {"train_mse": last_loss}


@torch.no_grad()
def evaluate_features(
    model: TopKSAE,
    x: np.ndarray,
    row_y: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> Tuple[pd.DataFrame, Dict[str, float], np.ndarray]:
    model.eval()
    dataset = TensorDataset(torch.from_numpy(x))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    zs: List[np.ndarray] = []
    recons: List[np.ndarray] = []
    for (xb,) in loader:
        xb = xb.to(device)
        recon, z = model(xb)
        zs.append(z.cpu().numpy())
        recons.append(recon.cpu().numpy())
    z_all = np.concatenate(zs, axis=0)
    recon_all = np.concatenate(recons, axis=0)
    mse = float(np.mean((recon_all - x) ** 2))
    active_mask = z_all > 0.0
    active_frac = float(np.mean(active_mask))
    effective_l0 = float(np.mean(active_mask.sum(axis=1)))
    dead_frac = float(np.mean(np.max(z_all, axis=0) <= 0.0))
    row_pos = row_y > 0
    rows = []
    for feat_idx in range(z_all.shape[1]):
        feat = z_all[:, feat_idx]
        row_gap = float(feat[row_pos].mean() - feat[~row_pos].mean()) if row_pos.any() and (~row_pos).any() else float("nan")
        row_support_gap = float(np.mean(feat[row_pos] > 0.0) - np.mean(feat[~row_pos] > 0.0)) if row_pos.any() and (~row_pos).any() else float("nan")
        rows.append(
            {
                "feature_id": feat_idx,
                "row_gap": row_gap,
                "row_support_gap": row_support_gap,
                "row_mean_pos": float(feat[row_pos].mean()) if row_pos.any() else float("nan"),
                "row_mean_neg": float(feat[~row_pos].mean()) if (~row_pos).any() else float("nan"),
                "row_active_frac": float(np.mean(feat > 0.0)),
            }
        )
    feature_df = pd.DataFrame(rows).sort_values("row_gap", ascending=False).reset_index(drop=True)
    stats = {
        "recon_mse": mse,
        "active_frac": active_frac,
        "effective_l0": effective_l0,
        "dead_feature_frac": dead_frac,
        "top10_row_gap_mean": float(feature_df.head(10)["row_gap"].mean()) if not feature_df.empty else float("nan"),
        "top1_row_gap": float(feature_df.iloc[0]["row_gap"]) if not feature_df.empty else float("nan"),
    }
    return feature_df, stats, z_all


def _choose_low_gap_ids(
    feature_df: pd.DataFrame,
    *,
    n: int,
    exclude: Sequence[int] = (),
) -> List[int]:
    ranked = (
        feature_df.assign(abs_gap=feature_df["row_gap"].abs())
        .sort_values(["abs_gap", "row_active_frac", "feature_id"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    blocked = {int(x) for x in exclude}
    chosen: List[int] = []
    for fid in ranked["feature_id"].tolist():
        val = int(fid)
        if val in blocked:
            continue
        chosen.append(val)
        blocked.add(val)
        if len(chosen) >= n:
            break
    return chosen


def choose_feature_sets(feature_df: pd.DataFrame) -> Dict[str, List[int]]:
    top = feature_df.sort_values("row_gap", ascending=False).reset_index(drop=True)
    active = top[top["row_active_frac"] > 0.02].copy()
    if active.empty:
        active = top.copy()

    top1 = [int(top.iloc[0]["feature_id"])]
    top3 = [int(x) for x in top.head(3)["feature_id"].tolist()]
    top5 = [int(x) for x in top.head(5)["feature_id"].tolist()]
    exclude = top5

    control_source = active if len(active) >= 3 else top
    control1 = _choose_low_gap_ids(control_source, n=1, exclude=exclude)
    if not control1:
        control1 = _choose_low_gap_ids(top, n=1, exclude=())
    control3 = _choose_low_gap_ids(control_source, n=3, exclude=exclude)
    if len(control3) < 3:
        control3 = _choose_low_gap_ids(top, n=3, exclude=exclude)
    if len(control3) < 3:
        control3 = _choose_low_gap_ids(top, n=3, exclude=())

    return {
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "control1": control1,
        "control3": control3,
    }


def decoder_overlap_stats(model: TopKSAE, top_feature_ids: Sequence[int]) -> Dict[str, float]:
    weight = model.decoder.weight.detach().cpu().numpy().astype(np.float64)
    norms = np.linalg.norm(weight, axis=0, keepdims=True)
    norms[norms < 1e-9] = 1.0
    w = weight / norms
    sim = np.abs(w.T @ w)
    np.fill_diagonal(sim, np.nan)
    tri = sim[np.triu_indices(sim.shape[0], k=1)]
    tri = tri[~np.isnan(tri)]
    mean_abs_cos = float(np.mean(tri)) if tri.size else float("nan")
    max_abs_cos = float(np.max(tri)) if tri.size else float("nan")

    top_ids = [int(x) for x in top_feature_ids if int(x) < sim.shape[0]]
    top_sim = sim[np.ix_(top_ids, top_ids)] if top_ids else np.empty((0, 0))
    top_tri = top_sim[np.triu_indices(len(top_ids), k=1)] if len(top_ids) > 1 else np.asarray([])
    top_tri = top_tri[~np.isnan(top_tri)] if top_tri.size else top_tri
    return {
        "decoder_mean_abs_cos_offdiag": mean_abs_cos,
        "decoder_max_abs_cos_offdiag": max_abs_cos,
        "decoder_top_features_mean_abs_cos": float(np.mean(top_tri)) if top_tri.size else float("nan"),
        "decoder_top_features_max_abs_cos": float(np.max(top_tri)) if top_tri.size else float("nan"),
    }


def support_overlap_stats(z_all: np.ndarray, top_feature_ids: Sequence[int]) -> Dict[str, float]:
    top_ids = [int(x) for x in top_feature_ids if int(x) < z_all.shape[1]]
    if not top_ids:
        return {
            "top_feature_mean_pairwise_jaccard": float("nan"),
            "top_feature_mean_active_overlap": float("nan"),
        }
    active = z_all[:, top_ids] > 0.0
    jaccs: List[float] = []
    conds: List[float] = []
    for i in range(active.shape[1]):
        ai = active[:, i]
        for j in range(i + 1, active.shape[1]):
            aj = active[:, j]
            inter = np.logical_and(ai, aj).sum()
            union = np.logical_or(ai, aj).sum()
            jaccs.append(float(inter / union) if union else 0.0)
            conds.append(float(inter / max(ai.sum(), 1)))
    return {
        "top_feature_mean_pairwise_jaccard": float(np.mean(jaccs)) if jaccs else float("nan"),
        "top_feature_mean_active_overlap": float(np.mean(conds)) if conds else float("nan"),
    }
