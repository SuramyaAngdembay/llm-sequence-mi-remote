#!/usr/bin/env python3
"""Correctness gate for a saved numerical-detector checkpoint.

The cached `channel_scores.npz` is what every downstream table is built from.
If the checkpoint cannot reproduce it, none of those tables are reproducible.
These checks run before the scores are used for anything:

  T1  reload-and-rescore: the saved state_dict + saved mu/sd must reproduce the
      cached per-class error sums.
  T2  forecast leakage: in forecast mode no context column of a row may be that
      row itself, and no eligible row may have a context made entirely of the
      target (which is what the first-day padding used to produce).
  T3  user boundaries: no context window may reach into another user's rows.
  T4  alignment: `has_history` must be exactly "not the user's first row" in
      forecast mode, and unconditionally true for autoencoding.
  T5  context-shuffle control (forecast only): attach each target day to a
      DIFFERENT user's context. If detection survives intact, the score is not
      actually using history, and calling the model a sequence model would be
      unsupported. Reported, not thresholded.

Usage:
  python3 scripts/timeseries/check_checkpoint.py \
      --matrix userday_matrix.npz --run-dir out/fc7 --window 7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_recon_detector import DayAutoencoder, build_windows  # noqa: E402
from eval_metrics_core import roc, user_max  # noqa: E402

FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--window", type=int, default=0,
                    help="context window as passed to training; read from "
                         "train_manifest.json when omitted, so the gate cannot "
                         "be run against the wrong window by mistake")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.window <= 0:
        mf = args.run_dir / "train_manifest.json"
        if not mf.exists():
            raise SystemExit(f"--window omitted and {mf} is absent")
        args.window = int(json.loads(mf.read_text())["window"])
        print(f"(window {args.window} read from train_manifest.json)")

    mat = np.load(args.matrix, allow_pickle=True)
    X = mat["X"].astype(np.float32)
    user_id = mat["user_id"].astype(str)
    y = mat["y"].astype(int)
    split = mat["split"].astype(str)

    ck = torch.load(args.run_dir / "model.pt", map_location="cpu", weights_only=False)
    cached = np.load(args.run_dir / "channel_scores.npz", allow_pickle=True)
    objective = str(ck["objective"])
    classes = np.array(list(ck["channel_classes"]))
    mu = np.asarray(ck["mu"], dtype=np.float32)
    sd = np.asarray(ck["sd"], dtype=np.float32)
    Xs = (X - mu) / sd

    print(f"checkpoint gate: {args.run_dir}  objective={objective}  window={args.window}\n")

    win = build_windows(user_id, args.window)
    first_of_user = np.zeros(len(user_id), dtype=bool)
    first_of_user[np.unique(user_id, return_index=True)[1]] = True
    has_history = ~first_of_user
    if objective == "forecast":
        win = win[:, :-1]
    else:
        has_history[:] = True
    print(f"T1. reload-and-rescore ({len(X)} rows, ctx_len={win.shape[1]})")
    if win.shape[1] != int(ck["ctx_len"]):
        check("ctx_len matches the checkpoint", False, f"{win.shape[1]} vs {ck['ctx_len']}")
        return 1
    check("ctx_len matches the checkpoint", True, str(win.shape[1]))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DayAutoencoder(int(ck["n_channels"]), int(ck["ctx_len"]),
                           int(ck["hidden"]), int(ck["latent"])).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    Xs_t = torch.from_numpy(Xs)
    win_t = torch.from_numpy(win)

    uniq = sorted(set(classes))
    err = {c: np.zeros(len(X), dtype=np.float64) for c in uniq}
    with torch.no_grad():
        for b0 in range(0, len(X), args.batch_size):
            rows = np.arange(b0, min(b0 + args.batch_size, len(X)))
            se = ((model(Xs_t[win_t[rows]].to(device)) - Xs_t[rows].to(device)) ** 2).cpu().numpy()
            for c in uniq:
                err[c][rows] = se[:, classes == c].sum(axis=1)
    worst = 0.0
    for c in uniq:
        d = float(np.abs(err[c] - cached[f"err_sum_{c}"]).max())
        worst = max(worst, d)
        print(f"     class {c:<8} max|Δ| = {d:.3e}")
    check("checkpoint reproduces the cached per-class error sums", worst < 1e-5, f"max {worst:.3e}")

    print("\nT2. forecast leakage")
    if objective == "forecast":
        self_in_ctx = (win == np.arange(len(X))[:, None]).any(axis=1)
        # The leakage condition is a row appearing in its OWN context. First
        # days do, because the padding repeats the earliest available row; the
        # fix excludes them. Two things must hold, and the second is the
        # stronger one: eligible rows must be clean, AND the exclusion flag
        # must cover exactly the leaking rows -- no fewer (a leak survives) and
        # no more (usable rows thrown away).
        check("no ELIGIBLE row appears in its own context",
              not bool((self_in_ctx & has_history).any()),
              f"{int((self_in_ctx & has_history).sum())} of "
              f"{int(has_history.sum())} eligible rows")
        check("excluded rows are EXACTLY the rows that contain themselves",
              np.array_equal(self_in_ctx, ~has_history),
              f"{int(self_in_ctx.sum())} self-containing, "
              f"{int((~has_history).sum())} excluded")
        # Descriptive, not a failure: a user's SECOND day has only one real
        # prior day, which the padding repeats across all context columns. The
        # context is degenerate but contains no target information, so these
        # rows are legitimately scorable with a weaker context than the name
        # "7-day" suggests.
        degenerate = (win == win[:, :1]).all(axis=1)
        print(f"     note: {int((degenerate & has_history).sum())} eligible rows have a "
              f"padding-collapsed context\n"
              f"           (a single real prior day repeated); leak-free, but their "
              f"effective history is 1 day")
    else:
        check("autoencoding: target is in the context by design (not a leak)",
              bool((win == np.arange(len(X))[:, None]).any(axis=1).all()))

    print("\nT3. user boundaries")
    ctx_user = user_id[win]
    crosses = (ctx_user != user_id[:, None]).any(axis=1)
    check("no context window crosses a user boundary", not crosses.any(), f"{int(crosses.sum())} rows")

    print("\nT4. alignment of the history flag")
    cached_hist = cached["has_history"].astype(bool) if "has_history" in cached.files \
        else np.ones(len(X), bool)
    check("cached has_history matches the recomputed flag", np.array_equal(cached_hist, has_history))
    if objective == "forecast":
        check("excluded rows are exactly the users' first rows",
              np.array_equal(~has_history, first_of_user), f"{int(first_of_user.sum())} users")
        pop = (split == "eval") | (split == "val")
        check("excluded-in-population count is reported",
              True, f"{int((pop & ~has_history).sum())} of {int(pop.sum())} population rows")

    print("\nT5. context-shuffle control")
    if objective != "forecast":
        print("     skipped: autoencoding has no history to shuffle")
    else:
        pop = (split == "eval") | (split == "val")
        elig = np.flatnonzero(pop & has_history)
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(elig))
        # guard: a donor must be a DIFFERENT user, else the shuffle is a no-op
        donor = elig[perm]
        bad = user_id[donor] == user_id[elig]
        for _ in range(20):
            if not bad.any():
                break
            donor[bad] = elig[rng.integers(0, len(elig), size=int(bad.sum()))]
            bad = user_id[donor] == user_id[elig]
        shuf = np.zeros(len(elig), dtype=np.float64)
        with torch.no_grad():
            for b0 in range(0, len(elig), args.batch_size):
                sl = slice(b0, min(b0 + args.batch_size, len(elig)))
                ctx = Xs_t[win_t[donor[sl]]].to(device)      # someone else's history
                tgt = Xs_t[elig[sl]].to(device)              # this row's target day
                shuf[sl] = ((model(ctx) - tgt) ** 2).sum(axis=1).cpu().numpy()
        true_full = sum(cached[f"err_sum_{c}"] for c in uniq)[elig]
        pos_users = sorted(set(user_id[elig][y[elig] == 1]))
        r_true, r_shuf = roc(y[elig], true_full), roc(y[elig], shuf)
        u_true = user_max(user_id, y, sum(cached[f"err_sum_{c}"] for c in uniq), elig)
        full_shuf = np.full(len(X), np.nan); full_shuf[elig] = shuf
        u_shuf = user_max(user_id, y, full_shuf, elig)
        print(f"     day  ROC  true {r_true:.4f}   shuffled-context {r_shuf:.4f}   "
              f"Δ {r_shuf - r_true:+.4f}")
        print(f"     user ROC  true {roc(u_true['y'].to_numpy(), u_true['score'].to_numpy()):.4f}   "
              f"shuffled-context {roc(u_shuf['y'].to_numpy(), u_shuf['score'].to_numpy()):.4f}")
        print(f"     ({len(elig)} eligible rows, {len(pos_users)} malicious users, "
              f"{int(bad.sum())} same-user donors left)")
        print("     Interpretation: a small Δ means the score is dominated by the target day,\n"
              "     not by the user's history. Reported, not thresholded.")

    rec = {"run_dir": str(args.run_dir), "objective": objective, "window": args.window,
           "rescore_max_abs_diff": worst, "failures": FAIL}
    (args.run_dir / "checkpoint_gate.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print()
    if FAIL:
        print(f"FAILED ({len(FAIL)}): " + "; ".join(FAIL))
        return 1
    print("all checkpoint gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
