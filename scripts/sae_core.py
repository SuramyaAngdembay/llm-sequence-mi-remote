from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class RankingUndefinedError(ValueError):
    """A feature ranking is undefined, or was computed on an invalid population.

    Raised instead of selecting features. Sorting a column of NaN gaps returns
    features in id order, so an undefined malicious-minus-benign statistic
    silently became "top5 = [0, 1, 2, 3, 4]" in the TWOS token-class re-runs
    (2026-09-19). Missing gaps are never replaced by zero here.
    """


# Every ranking file must carry these, all finite, before anything is selected.
RANKING_COLUMNS = ("row_gap", "row_mean_pos", "row_mean_neg", "row_active_frac")


def validate_ranking_population(
    row_y: Sequence[int] | np.ndarray,
    *,
    min_pos_rows: int = 1,
    min_neg_rows: int = 1,
    context: str = "",
) -> Tuple[int, int]:
    """Require both classes in the population a ranking is computed on.

    Fitting an SAE on benign rows only is legitimate. Ranking its features by
    malicious-minus-benign activation is not possible on that same population.
    """
    y = np.asarray(row_y)
    n_pos = int(np.count_nonzero(y > 0))
    n_neg = int(len(y) - n_pos)
    if n_pos < int(min_pos_rows) or n_neg < int(min_neg_rows):
        raise RankingUndefinedError(
            f"{context}a malicious-minus-benign ranking needs both classes in its population; "
            f"got {n_pos} positive rows (need >= {int(min_pos_rows)}) and {n_neg} benign rows "
            f"(need >= {int(min_neg_rows)}). Rank features on a discovery population that "
            "contains both classes (reselect_token_sae_features.py)."
        )
    return n_pos, n_neg


def load_ranking(cfg_dir: Path, *, validate: bool = True) -> pd.DataFrame:
    """Read a frontier config's feature ranking, refusing a benign-only frontier.

    A benign-only frontier writes RANKING_NOT_COMPUTED.json and no ranking
    file; pointing a consumer at it must fail, not select arbitrary features.
    """
    cfg_dir = Path(cfg_dir)
    ranking_path = cfg_dir / "delta_sae_top_features.csv"
    marker = cfg_dir / "RANKING_NOT_COMPUTED.json"
    if marker.exists() or not ranking_path.exists():
        detail = marker.read_text(encoding="utf-8").strip() if marker.exists() else "no ranking file"
        raise RankingUndefinedError(
            f"{cfg_dir} has no usable feature ranking ({detail}). Use a "
            "reselect_token_sae_features.py output ranked on a discovery population."
        )
    feature_df = pd.read_csv(ranking_path)
    if validate:
        validate_ranking(feature_df, context=f"{ranking_path}: ")
    return feature_df


def validate_ranking(feature_df: pd.DataFrame, *, context: str = "") -> None:
    """Refuse a ranking with missing columns, repeated ids or non-finite statistics."""
    missing = [c for c in ("feature_id",) + RANKING_COLUMNS if c not in feature_df.columns]
    if missing:
        raise RankingUndefinedError(f"{context}ranking is missing columns {missing}")
    if feature_df.empty:
        raise RankingUndefinedError(f"{context}ranking has no rows")
    if feature_df["feature_id"].duplicated().any():
        raise RankingUndefinedError(f"{context}ranking repeats feature ids")
    for col in RANKING_COLUMNS:
        vals = pd.to_numeric(feature_df[col], errors="coerce").to_numpy(dtype=np.float64)
        bad = int(np.count_nonzero(~np.isfinite(vals)))
        if bad:
            raise RankingUndefinedError(
                f"{context}{col} is non-finite for {bad} of {len(vals)} features, so the ranking "
                "is undefined. This happens when the ranking population lacks positive or "
                "benign rows. Refusing to select features: sorting an undefined column "
                "returns arbitrary ids."
            )


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
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    model.eval()
    row_y = np.asarray(row_y)
    if len(row_y) != len(x):
        raise ValueError(f"row_y length {len(row_y)} does not match x rows {len(x)}")
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(row_y.astype(np.int64, copy=False)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    d_latent = int(model.encoder.out_features)
    sum_pos = np.zeros(d_latent, dtype=np.float64)
    sum_neg = np.zeros(d_latent, dtype=np.float64)
    support_pos = np.zeros(d_latent, dtype=np.float64)
    support_neg = np.zeros(d_latent, dtype=np.float64)
    max_z = np.full(d_latent, -np.inf, dtype=np.float32)
    sqerr_sum = 0.0
    n_elements = 0
    active_total = 0.0
    row_active_total = 0.0
    n_pos = int(np.count_nonzero(row_y > 0))
    n_neg = int(len(row_y) - n_pos)

    for xb, yb in loader:
        xb = xb.to(device)
        recon, z = model(xb)
        sqerr_sum += float(torch.sum((recon - xb) ** 2).item())
        n_elements += int(xb.numel())

        z_cpu = z.detach().cpu().numpy()
        active_cpu = z_cpu > 0.0
        active_total += float(active_cpu.sum())
        row_active_total += float(active_cpu.sum(axis=1).sum())
        max_z = np.maximum(max_z, z_cpu.max(axis=0))

        pos_mask = yb.numpy() > 0
        if pos_mask.any():
            z_pos = z_cpu[pos_mask]
            sum_pos += z_pos.sum(axis=0, dtype=np.float64)
            support_pos += (z_pos > 0.0).sum(axis=0)
        if (~pos_mask).any():
            z_neg = z_cpu[~pos_mask]
            sum_neg += z_neg.sum(axis=0, dtype=np.float64)
            support_neg += (z_neg > 0.0).sum(axis=0)

    n_rows = int(len(row_y))
    mse = float(sqerr_sum / max(n_elements, 1))
    active_frac = float(active_total / max(n_rows * d_latent, 1))
    effective_l0 = float(row_active_total / max(n_rows, 1))
    dead_frac = float(np.mean(max_z <= 0.0))
    row_mean_pos = sum_pos / n_pos if n_pos else np.full(d_latent, np.nan, dtype=np.float64)
    row_mean_neg = sum_neg / n_neg if n_neg else np.full(d_latent, np.nan, dtype=np.float64)
    row_support_pos = support_pos / n_pos if n_pos else np.full(d_latent, np.nan, dtype=np.float64)
    row_support_neg = support_neg / n_neg if n_neg else np.full(d_latent, np.nan, dtype=np.float64)
    row_gap = row_mean_pos - row_mean_neg if n_pos and n_neg else np.full(d_latent, np.nan, dtype=np.float64)
    row_support_gap = row_support_pos - row_support_neg if n_pos and n_neg else np.full(d_latent, np.nan, dtype=np.float64)

    feature_df = pd.DataFrame(
        {
            "feature_id": np.arange(d_latent, dtype=np.int64),
            "row_gap": row_gap,
            "row_support_gap": row_support_gap,
            "row_mean_pos": row_mean_pos,
            "row_mean_neg": row_mean_neg,
            "row_active_frac": (support_pos + support_neg) / max(n_rows, 1),
        }
    )
    feature_df = feature_df.sort_values("row_gap", ascending=False).reset_index(drop=True)
    stats = {
        "n_pos_rows": n_pos,
        "n_neg_rows": n_neg,
        "recon_mse": mse,
        "active_frac": active_frac,
        "effective_l0": effective_l0,
        "dead_feature_frac": dead_frac,
        "top10_row_gap_mean": float(feature_df.head(10)["row_gap"].mean()) if not feature_df.empty else float("nan"),
        "top1_row_gap": float(feature_df.iloc[0]["row_gap"]) if not feature_df.empty else float("nan"),
    }
    return feature_df, stats


def _choose_low_gap_ids(
    feature_df: pd.DataFrame,
    *,
    n: int,
    exclude: Sequence[int] = (),
) -> List[int]:
    gaps = pd.to_numeric(feature_df["row_gap"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(gaps)):
        # With NaN gaps the |gap| key is undefined and the sort silently falls
        # through to activity: "low gap" controls become "most active" controls.
        raise RankingUndefinedError(
            "control selection needs a finite row_gap for every candidate; "
            f"{int(np.count_nonzero(~np.isfinite(gaps)))} of {len(gaps)} are non-finite"
        )
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


def _choose_active_low_gap_ids(
    feature_df: pd.DataFrame,
    *,
    n: int,
    min_active_frac: float,
    exclude: Sequence[int] = (),
) -> List[int]:
    source = feature_df[feature_df["row_active_frac"] >= float(min_active_frac)].copy()
    if source.empty:
        return []
    return _choose_low_gap_ids(source, n=n, exclude=exclude)


def choose_feature_sets(feature_df: pd.DataFrame) -> Dict[str, List[int]]:
    validate_ranking(feature_df, context="choose_feature_sets: ")
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


def add_active_control_feature_sets(
    feature_sets: Dict[str, List[int]],
    feature_df: pd.DataFrame,
    *,
    min_active_frac: float,
    sizes: Sequence[int] = (1, 3, 5),
) -> Dict[str, List[int]]:
    validate_ranking(feature_df, context="add_active_control_feature_sets: ")
    out = {name: list(ids) for name, ids in feature_sets.items()}
    exclude = set(out.get("top5", []))
    for size in sizes:
        ids = _choose_active_low_gap_ids(
            feature_df,
            n=int(size),
            min_active_frac=min_active_frac,
            exclude=tuple(exclude),
        )
        if len(ids) == int(size):
            out[f"control{int(size)}_active"] = ids
    return out


def feature_set_criteria(
    feature_df: pd.DataFrame,
    feature_sets: Dict[str, List[int]],
    *,
    top_set: str = "top5",
    control_sets: Sequence[str] = ("control5_active",),
    min_active_frac: float = 0.002,
) -> Dict[str, Any]:
    """Check each set against its stated selection rule, for the run manifest.

    `control{n}_active` is defined as the n lowest-|gap| features among those
    active on >= min_active_frac of ranking rows, excluding the top set. That
    rule is a threshold: it does not match the controls' activity to the
    selected features, so the activity ratio is reported rather than assumed.
    """
    validate_ranking(feature_df, context="feature_set_criteria: ")
    by_id = feature_df.set_index("feature_id")
    abs_gap_all = np.abs(pd.to_numeric(feature_df["row_gap"]).to_numpy(dtype=np.float64))

    def describe(ids: Sequence[int]) -> Dict[str, Any]:
        ids = [int(i) for i in ids]
        sub = by_id.loc[ids]
        gaps = [float(v) for v in sub["row_gap"]]
        act = [float(v) for v in sub["row_active_frac"]]
        return {
            "ids": ids,
            "row_gap": gaps,
            "row_active_frac": act,
            "mean_row_gap": float(np.mean(gaps)) if gaps else float("nan"),
            "mean_active_frac": float(np.mean(act)) if act else float("nan"),
        }

    if top_set not in feature_sets:
        raise KeyError(f"top set {top_set} not in feature sets {sorted(feature_sets)}")
    top = describe(feature_sets[top_set])
    top["gap_rank_check"] = sorted(top["ids"]) == sorted(
        int(x) for x in feature_df.sort_values("row_gap", ascending=False).head(len(top["ids"]))["feature_id"]
    )
    report: Dict[str, Any] = {
        "n_features": int(len(feature_df)),
        "min_active_frac": float(min_active_frac),
        "sets": {top_set: top},
    }
    for name in control_sets:
        if name not in feature_sets:
            continue
        d = describe(feature_sets[name])
        max_abs = float(np.max(np.abs(d["row_gap"]))) if d["ids"] else float("nan")
        d["max_abs_gap"] = max_abs
        d["frac_features_with_abs_gap_at_most_max"] = float(np.mean(abs_gap_all <= max_abs)) if d["ids"] else float("nan")
        d["meets_min_active"] = bool(all(a >= float(min_active_frac) for a in d["row_active_frac"]))
        if name.endswith("_active"):
            eligible = feature_df[
                (feature_df["row_active_frac"] >= float(min_active_frac))
                & (~feature_df["feature_id"].isin(feature_sets[top_set]))
            ]
            expected = _choose_low_gap_ids(eligible, n=len(d["ids"]))
            d["matches_stated_rule"] = sorted(expected) == sorted(d["ids"])
        d["activity_criterion"] = "threshold_only"
        d["activity_ratio_to_top"] = (
            d["mean_active_frac"] / top["mean_active_frac"] if top["mean_active_frac"] > 0 else float("nan")
        )
        report["sets"][name] = d
    return report


def check_disjoint_users(
    discovery_users: Sequence[str],
    evaluation_users: Sequence[str],
    *,
    context: str = "",
) -> None:
    """Raise if a claimed held-out evaluation shares users with feature discovery."""
    overlap = sorted(set(map(str, discovery_users)) & set(map(str, evaluation_users)))
    if overlap:
        raise RankingUndefinedError(
            f"{context}{len(overlap)} evaluation users were also used to select features "
            f"(first: {overlap[:5]}); the evaluation is not held out from selection"
        )


def decoder_overlap_stats(model: TopKSAE, top_feature_ids: Sequence[int], *, block_size: int = 512) -> Dict[str, float]:
    weight = model.decoder.weight.detach().cpu().numpy().astype(np.float32, copy=True)
    norms = np.linalg.norm(weight, axis=0, keepdims=True).astype(np.float32)
    norms[norms < 1e-9] = 1.0
    weight /= norms
    d_latent = int(weight.shape[1])
    sim_sum = 0.0
    sim_count = 0
    max_abs_cos = float("nan")
    for start in range(0, d_latent, block_size):
        end = min(start + block_size, d_latent)
        sims = np.abs(weight[:, start:end].T @ weight)
        block_max = float("nan")
        for local_idx, feat_idx in enumerate(range(start, end)):
            vals = sims[local_idx, feat_idx + 1 :]
            if vals.size:
                sim_sum += float(vals.sum(dtype=np.float64))
                sim_count += int(vals.size)
                local_max = float(vals.max())
                block_max = local_max if np.isnan(block_max) else max(block_max, local_max)
        if not np.isnan(block_max):
            max_abs_cos = block_max if np.isnan(max_abs_cos) else max(max_abs_cos, block_max)
    mean_abs_cos = float(sim_sum / sim_count) if sim_count else float("nan")

    top_ids = [int(x) for x in top_feature_ids if 0 <= int(x) < d_latent]
    top_w = weight[:, top_ids] if top_ids else np.empty((weight.shape[0], 0), dtype=np.float32)
    top_sim = np.abs(top_w.T @ top_w) if top_ids else np.empty((0, 0), dtype=np.float32)
    if top_sim.size:
        np.fill_diagonal(top_sim, np.nan)
    top_tri = top_sim[np.triu_indices(len(top_ids), k=1)] if len(top_ids) > 1 else np.asarray([])
    top_tri = top_tri[~np.isnan(top_tri)] if top_tri.size else top_tri
    return {
        "decoder_mean_abs_cos_offdiag": mean_abs_cos,
        "decoder_max_abs_cos_offdiag": max_abs_cos,
        "decoder_top_features_mean_abs_cos": float(np.mean(top_tri)) if top_tri.size else float("nan"),
        "decoder_top_features_max_abs_cos": float(np.max(top_tri)) if top_tri.size else float("nan"),
    }


@torch.no_grad()
def support_overlap_stats_streaming(
    model: TopKSAE,
    x: np.ndarray,
    top_feature_ids: Sequence[int],
    *,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    d_latent = int(model.encoder.out_features)
    top_ids = [int(x) for x in top_feature_ids if 0 <= int(x) < d_latent]
    if not top_ids:
        return {
            "top_feature_mean_pairwise_jaccard": float("nan"),
            "top_feature_mean_active_overlap": float("nan"),
        }
    model.eval()
    dataset = TensorDataset(torch.from_numpy(x))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    active_counts = np.zeros(len(top_ids), dtype=np.float64)
    intersections = np.zeros((len(top_ids), len(top_ids)), dtype=np.float64)
    for (xb,) in loader:
        xb = xb.to(device)
        _, z = model(xb)
        active = (z[:, top_ids] > 0.0).cpu().numpy().astype(np.float64, copy=False)
        active_counts += active.sum(axis=0)
        intersections += active.T @ active
    jaccs: List[float] = []
    conds: List[float] = []
    for i in range(len(top_ids)):
        ai = active_counts[i]
        for j in range(i + 1, len(top_ids)):
            aj = active_counts[j]
            inter = intersections[i, j]
            union = ai + aj - inter
            jaccs.append(float(inter / union) if union else 0.0)
            conds.append(float(inter / max(ai, 1.0)))
    return {
        "top_feature_mean_pairwise_jaccard": float(np.mean(jaccs)) if jaccs else float("nan"),
        "top_feature_mean_active_overlap": float(np.mean(conds)) if conds else float("nan"),
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
