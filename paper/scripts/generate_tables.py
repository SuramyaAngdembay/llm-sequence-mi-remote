#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"


def write_table(path: Path, caption: str, label: str, colspec: str, header: list[str], rows: list[list[str]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def write_detector_table() -> None:
    rows = [
        ["r6.2", "Remote Qwen3-8B adapted NLL", fmt(0.000754631, 4), fmt(0.953157, 3), fmt(0.0537037, 3), fmt(24.0, 1)],
        ["r6.2", "Deep SVDD", fmt(0.0115455, 4), fmt(0.627919, 3), fmt(0.211455, 3), fmt(83.25, 1)],
        ["r6.2", "GRU AE", fmt(0.00572239, 4), fmt(0.765776, 3), fmt(0.0814103, 3), fmt(24.75, 1)],
        ["r6.2", "LSTM AE", fmt(0.00206543, 4), fmt(0.767738, 3), fmt(0.0574767, 3), fmt(24.25, 1)],
        ["r6.2", "Isolation Forest", fmt(0.000210794, 4), fmt(0.712505, 3), fmt(0.0126207, 3), fmt(153.0, 1)],
        ["r4.2", "Remote Qwen3-8B adapted NLL", fmt(0.0134474, 4), fmt(0.964124, 3), fmt(0.100838, 3), fmt(26.45, 1)],
        ["r4.2", "Deep SVDD", fmt(0.0337171, 4), fmt(0.742914, 3), fmt(0.381529, 3), fmt(53.4167, 1)],
        ["r4.2", "GRU AE", fmt(0.0254413, 4), fmt(0.695754, 3), fmt(0.12435, 3), fmt(86.5833, 1)],
        ["r4.2", "LSTM AE", fmt(0.0236354, 4), fmt(0.714125, 3), fmt(0.119657, 3), fmt(92.0833, 1)],
        ["r4.2", "Isolation Forest", fmt(0.000254408, 4), fmt(0.714632, 3), fmt(0.00794344, 3), fmt(186.417, 1)],
    ]
    write_table(
        TABLES / "cert_detector_comparison.tex",
        "Fold-aligned detector comparison on CERT. The remote Qwen3-8B branch has strong ROC and ranking behavior, but day PR-AUC remains below the stronger local Magnolia baselines on both datasets.",
        "tab:cert_detector",
        "llcccc",
        ["Dataset", "Method", "Day PR-AUC", "Day ROC-AUC", "User PR-AUC", "Held-out rank"],
        rows,
    )


def write_mech_table() -> None:
    rows = [
        [
            "r6.2",
            "Remote causal",
            "Same-user excluded",
            "role",
            "0.006848",
            "[0.003362, 0.010790]",
            "remote > local adaptive (0.001133)",
        ],
        [
            "r6.2",
            "Remote necessity",
            "Same-user excluded",
            "project\\_role",
            "0.065188",
            "[0.055145, 0.075023]",
            "strong positive necessity",
        ],
        [
            "r4.2",
            "Transferred causal",
            "Layer-18 r6.2 config",
            "multiple",
            "negative",
            "all audited configs < 0",
            "direct transfer fails",
        ],
        [
            "r4.2",
            "Remote native causal",
            "Same-user excluded",
            "positive across contexts",
            "0.001112*",
            "see compare report",
            "remote > local adaptive (0.000909)",
        ],
        [
            "r4.2",
            "Remote native necessity",
            "Same-user excluded",
            "dept\\_role",
            "0.002922",
            "[0.001460, 0.004379]",
            "role/dept\\_role strongest",
        ],
    ]
    write_table(
        TABLES / "cert_mechanistic_summary.tex",
        "Mechanistic summary after the same-user recovery audit. The asterisk marks the current regenerated compare-report row used for the r4.2 causal comparison while the no-same-user causal bundle is still being harmonized internally.",
        "tab:cert_mechanistic",
        "lllllll",
        ["Dataset", "Branch", "Protocol", "Context", "Effect", "95\\% CI", "Read"],
        rows,
    )


def write_claims_table() -> None:
    rows = [
        ["Benign-only QLoRA training is valid one-class training", "Supported"],
        ["r6.2 has strong remote causal structure", "Supported"],
        ["Direct token-mechanism transfer from r6.2 to r4.2 succeeds", "Rejected"],
        ["r4.2 has a native rediscovered remote mechanism", "Supported"],
        ["Remote detector is strongly superior on CERT", "Not supported"],
    ]
    write_table(
        TABLES / "claim_status.tex",
        "Current claim status after the detector split audit and same-user recovery reruns.",
        "tab:claim_status",
        "lp{5.5cm}",
        ["Claim", "Status"],
        rows,
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    write_detector_table()
    write_mech_table()
    write_claims_table()


if __name__ == "__main__":
    main()

