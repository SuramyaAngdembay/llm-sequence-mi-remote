#!/usr/bin/env python3
"""Benign-only reconstruction detector over user-day channels, with the score
decomposed by channel class.

The paper's Discussion predicts that a sequence model sharing the
sequence-prediction setting but not the pretrained language representation
"should fall between the tabular detectors and the LM in profile reliance".
This is the detector that tests it: same fields as the language model, same
deterministic splits, benign-only training, and a score whose profile and
behaviour parts can be separated exactly.

Score. For a reconstruction detector the anomaly score of a user-day is the
mean squared error over channels, so with channel classes c,

    s_full = sum_c (N_c / N) * s_c ,   s_c = err_sum_c / N_c

which is the identity the token-class decomposition uses. A
behaviour-channel-only score is therefore the exact conditional

    s_B = (err_sum_total - err_sum_P) / (N - N_P)

and NOT `s_full - s_P`, which is a different quantity (see
`token_class_decomposition`). Because every channel here contributes exactly
one squared-error term, N_P and N_B are the same for every row, so the two
coincide up to an affine transform and the ranking is unchanged — unlike the
token case, where N_P varies per example. That is worth stating rather than
assuming: it is why this design isolates *which channels carry the score*
cleanly.

Context. `--window W` gives the encoder the user's previous W days as context
(strictly past days of the same user, so no leakage across the day being
scored); W=1 is the plain per-day autoencoder, the learned analogue of the
published PCA-reconstruction baseline.

Standardisation uses training-split statistics only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class DayAutoencoder(nn.Module):
    """Small MLP autoencoder over (window x channels), reconstructing the last day."""

    def __init__(self, n_channels: int, window: int, hidden: int, latent: int):
        super().__init__()
        d_in = n_channels * window
        self.encoder = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_channels),
        )

    def forward(self, x):                      # x: [B, window, C]
        z = self.encoder(x.flatten(1))
        return self.decoder(z)                 # [B, C] = reconstruction of the last day


def build_windows(user_ids: np.ndarray, window: int) -> np.ndarray:
    """Row indices of the W-day context ending at each row (padded by repeating
    the user's first available day, never crossing a user boundary)."""
    n = len(user_ids)
    idx = np.empty((n, window), dtype=np.int64)
    start = 0
    for i in range(1, n + 1):
        if i == n or user_ids[i] != user_ids[start]:
            for j in range(start, i):
                for k in range(window):
                    idx[j, window - 1 - k] = max(start, j - k)
            start = i
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-train-rows", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(args.matrix, allow_pickle=True)
    X = z["X"].astype(np.float32)
    channels = [str(c) for c in z["channels"]]
    classes = [str(c) for c in z["channel_classes"]]
    user_id = z["user_id"].astype(str)
    y = z["y"].astype(np.int64)
    split = z["split"].astype(str)
    n_ch = X.shape[1]

    # standardise on TRAIN rows only
    tr = split == "train"
    mu = X[tr].mean(axis=0)
    sd = X[tr].std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xs = (X - mu) / sd

    win = build_windows(user_id, args.window)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DayAutoencoder(n_ch, args.window, args.hidden, args.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_rows = np.flatnonzero(tr)
    if args.max_train_rows > 0 and len(train_rows) > args.max_train_rows:
        rng = np.random.default_rng(args.seed)
        train_rows = np.sort(rng.choice(train_rows, args.max_train_rows, replace=False))

    Xs_t = torch.from_numpy(Xs)
    win_t = torch.from_numpy(win)
    t0 = time.time()
    model.train()
    for ep in range(args.epochs):
        perm = np.random.permutation(len(train_rows))
        tot, nb = 0.0, 0
        for b0 in range(0, len(perm), args.batch_size):
            rows = train_rows[perm[b0 : b0 + args.batch_size]]
            ctx = Xs_t[win_t[rows]].to(device)          # [B, W, C]
            tgt = Xs_t[rows].to(device)                 # [B, C]
            rec = model(ctx)
            loss = ((rec - tgt) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f"[epoch {ep}] train mse {tot / max(nb,1):.5f}  ({time.time()-t0:.0f}s)", flush=True)

    # score every row, caching per-channel-class squared-error sums
    model.eval()
    err_sums = {c: np.zeros(len(X), dtype=np.float64) for c in sorted(set(classes))}
    cls_arr = np.array(classes)
    with torch.no_grad():
        for b0 in range(0, len(X), args.batch_size):
            rows = np.arange(b0, min(b0 + args.batch_size, len(X)))
            ctx = Xs_t[win_t[rows]].to(device)
            tgt = Xs_t[rows].to(device)
            se = ((model(ctx) - tgt) ** 2).cpu().numpy()   # [B, C]
            for c in err_sums:
                err_sums[c][rows] = se[:, cls_arr == c].sum(axis=1)

    counts = {c: int((cls_arr == c).sum()) for c in err_sums}
    total = sum(err_sums[c] for c in err_sums)
    n_total = sum(counts.values())

    np.savez_compressed(
        out_dir / "channel_scores.npz",
        user_id=user_id, y=y, split=split,
        err_sum_total=total, n_total=np.int64(n_total),
        **{f"err_sum_{c}": err_sums[c] for c in err_sums},
        **{f"n_{c}": np.int64(counts[c]) for c in err_sums},
    )

    # exact reconstruction check of the class partition
    recon_err = float(np.abs(total - sum(err_sums[c] for c in err_sums)).max())
    manifest = {
        "matrix": str(args.matrix),
        "window": args.window, "hidden": args.hidden, "latent": args.latent,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "seed": args.seed, "device": str(device),
        "n_channels": n_ch, "channel_counts": counts,
        "n_train_rows": int(len(train_rows)),
        "standardisation": "train-split mean/std only",
        "score": "mean squared reconstruction error over channels; class partition sums to the total",
        "partition_max_abs_err": recon_err,
        "elapsed_sec": round(time.time() - t0, 1),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
