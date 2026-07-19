#!/usr/bin/env python3
"""Generate the paper's data figures as vector PDFs.

Numbers are transcribed from the final tracked result bundles:
- results/qwen3_8b_token_causal/same_user_recovery/RESULTS.md
- results/qwen3_8b_token_necessity/same_user_recovery/RESULTS.md
- results/qwen3_8b_r42_token_causal/same_user_recovery/RESULTS.md
- results/qwen3_8b_r42_token_necessity/same_user_recovery/RESULTS.md
- results/*/detector_metrics_fold_aligned/FOLD_ALIGNED_DETECTOR_REPORT.md
- strict_compare *_no_same_user REMOTE_VS_LOCAL_DAYLEVEL_REPORT.md (local refs)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"

ACCENT = "#2F5FA5"      # session-LM / effect marks
MUTED = "#98A0AA"       # baseline marks
INK = "#1F2937"         # primary text
INK_2 = "#6B7280"       # secondary text
GRID = "#E5E7EB"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "axes.edgecolor": INK_2,
        "axes.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "axes.labelcolor": INK,
        "pdf.fonttype": 42,
        "figure.dpi": 150,
    }
)

# (label, estimate, lo, hi) — order top-to-bottom as displayed.
MECH = {
    ("r6.2", "Causal patching"): [
        ("role", 0.006848, 0.003362, 0.010790),
        ("dept x role", 0.006818, 0.003541, 0.010321),
        ("project x role", 0.004201, 0.001565, 0.006979),
    ],
    ("r6.2", "Necessity ablation"): [
        ("project x role", 0.065188, 0.055145, 0.075023),
        ("role", 0.062167, 0.050265, 0.073364),
        ("dept x role", 0.056603, 0.044996, 0.068990),
        ("team", 0.052234, 0.042054, 0.061397),
    ],
    ("r4.2", "Causal patching"): [
        ("team", 0.001418, 0.001139, 0.001690),
        ("role", 0.001112, 0.000826, 0.001391),
        ("dept x role", 0.001067, 0.000824, 0.001305),
        ("dept", 0.000982, 0.000751, 0.001215),
    ],
    ("r4.2", "Necessity ablation"): [
        ("dept x role", 0.002922, 0.001460, 0.004379),
        ("role", 0.002075, 0.000679, 0.003418),
        ("dept", 0.001155, -0.000242, 0.002536),
        ("team", 0.000662, -0.000880, 0.002236),
    ],
}

# Best local session-AE comparator day-level contrast on the same receivers.
LOCAL_REF = {"r6.2": 0.001133, "r4.2": 0.000909}

DETECTOR = {
    "r6.2": [
        ("Qwen3-8B session LM", 0.000754631, 0.953157),
        ("Deep SVDD", 0.0115455, 0.627919),
        ("GRU AE", 0.00572239, 0.765776),
        ("LSTM AE", 0.00206543, 0.767738),
        ("Isolation Forest", 0.000210794, 0.712505),
    ],
    "r4.2": [
        ("Qwen3-8B session LM", 0.0134474, 0.964124),
        ("Deep SVDD", 0.0337171, 0.742914),
        ("GRU AE", 0.0254413, 0.695754),
        ("LSTM AE", 0.0236354, 0.714125),
        ("Isolation Forest", 0.000254408, 0.714632),
    ],
}


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def mech_effects() -> None:
    order = [
        ("r6.2", "Causal patching"),
        ("r6.2", "Necessity ablation"),
        ("r4.2", "Causal patching"),
        ("r4.2", "Necessity ablation"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 3.5))
    for ax, key in zip(axes.flat, order):
        dataset, estimand = key
        rows = MECH[key]
        ys = range(len(rows), 0, -1)
        for y, (label, est, lo, hi) in zip(ys, rows):
            crosses = lo <= 0.0 <= hi
            est, lo, hi = est * 1e3, lo * 1e3, hi * 1e3
            ax.plot([lo, hi], [y, y], color=ACCENT, linewidth=1.4,
                    solid_capstyle="round", zorder=2)
            ax.plot(
                est, y,
                marker="o", markersize=4.6,
                markerfacecolor="white" if crosses else ACCENT,
                markeredgecolor=ACCENT, markeredgewidth=1.1, zorder=3,
            )
        ax.axvline(0.0, color=INK_2, linewidth=0.7, zorder=1)
        if estimand == "Causal patching":
            ax.axvline(LOCAL_REF[dataset] * 1e3, color=INK_2, linewidth=0.8,
                       linestyle=(0, (4, 2.5)), zorder=1)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_ylim(0.4, len(rows) + 0.6)
        ax.set_title(f"{dataset} — {estimand.lower()}", loc="left",
                     fontweight="bold", color=INK)
        style_axis(ax)
    axes[1][0].set_xlabel(r"top-vs-control contrast ($\times 10^{-3}$)")
    axes[1][1].set_xlabel(r"top-vs-control contrast ($\times 10^{-3}$)")
    fig.subplots_adjust(left=0.14, right=0.985, top=0.93, bottom=0.13,
                        hspace=0.75, wspace=0.42)
    fig.savefig(FIGS / "mech_effects.pdf")
    fig.savefig(FIGS / "mech_effects_preview.png", dpi=200)
    plt.close(fig)


def detector_dissociation() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5), sharey=True)
    # Per-panel label nudges (points): keep short labels clear of the marks.
    offsets = {
        ("r6.2", "Qwen3-8B session LM"): (0, 7),
        ("r6.2", "Deep SVDD"): (0, 7),
        ("r6.2", "GRU AE"): (16, 7),
        ("r6.2", "LSTM AE"): (-16, 7),
        ("r6.2", "Isolation Forest"): (2, 7),
        ("r4.2", "Qwen3-8B session LM"): (0, 7),
        ("r4.2", "Deep SVDD"): (-8, 7),
        ("r4.2", "GRU AE"): (-24, 2),
        ("r4.2", "LSTM AE"): (14, -10),
        ("r4.2", "Isolation Forest"): (6, 7),
    }
    for ax, dataset in zip(axes, ["r6.2", "r4.2"]):
        for name, pr, roc in DETECTOR[dataset]:
            remote = name.startswith("Qwen")
            ax.plot(
                pr, roc, marker="o",
                markersize=5.5 if remote else 4.5,
                markerfacecolor=ACCENT if remote else MUTED,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3,
            )
            dx, dy = offsets[(dataset, name)]
            ax.annotate(
                name, (pr, roc), textcoords="offset points", xytext=(dx, dy),
                ha="center", fontsize=6.8,
                color=INK if remote else INK_2,
                fontweight="bold" if remote else "normal",
            )
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 6e-2)
        ax.set_ylim(0.55, 1.03)
        ax.set_title(dataset, loc="left", fontweight="bold", color=INK)
        ax.set_xlabel("day-level PR-AUC (log scale)")
        style_axis(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
    axes[0].set_ylabel("day-level ROC-AUC")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.19, wspace=0.14)
    fig.savefig(FIGS / "detector_dissociation.pdf")
    fig.savefig(FIGS / "detector_dissociation_preview.png", dpi=200)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    mech_effects()
    detector_dissociation()
    print("Wrote:", FIGS / "mech_effects.pdf", "and", FIGS / "detector_dissociation.pdf")


if __name__ == "__main__":
    main()
