#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"


def write_table(path: Path, caption: str, label: str, colspec: str, header: list[str], rows: list[list[str]], size: str = r"\small", colsep: str | None = None) -> None:
    lines = [
        r"\begin{table}[t]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        size,
    ]
    if colsep:
        lines.append(rf"\setlength{{\tabcolsep}}{{{colsep}}}")
    lines += [
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        if row == ["---"]:
            lines.append(r"\midrule")
        else:
            lines.append(" & ".join(row) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def write_detector_table() -> None:
    rows = [
        ["r6.2", "Session LM (adapted NLL)", "0.0008", r"\textbf{0.953}", "0.054", r"\textbf{24.0}"],
        ["", "Deep SVDD", r"\textbf{0.0115}", "0.628", r"\textbf{0.211}", "83.2"],
        ["", "GRU AE", "0.0057", "0.766", "0.081", "24.8"],
        ["", "LSTM AE", "0.0021", "0.768", "0.057", "24.2"],
        ["", "Isolation Forest", "0.0002", "0.713", "0.013", "153.0"],
        ["---"],
        ["r4.2", "Session LM (adapted NLL)", "0.0134", r"\textbf{0.964}", "0.101", r"\textbf{26.4}"],
        ["", "Deep SVDD", r"\textbf{0.0337}", "0.743", r"\textbf{0.382}", "53.4"],
        ["", "GRU AE", "0.0254", "0.696", "0.124", "86.6"],
        ["", "LSTM AE", "0.0236", "0.714", "0.120", "92.1"],
        ["", "Isolation Forest", "0.0003", "0.715", "0.008", "186.4"],
    ]
    write_table(
        TABLES / "cert_detector_comparison.tex",
        "Fold-aligned detector comparison on CERT. Bold marks the best value per metric within each dataset (held-out rank: lower is better). The Qwen3-8B session LM shows strong ROC and ranking behavior, but its day-level PR-AUC remains below the stronger feature-based baselines.",
        "tab:cert_detector",
        "llcccc",
        ["Dataset", "Method", "Day PR-AUC", "Day ROC-AUC", "User PR-AUC", "Held-out rank"],
        rows,
        size=r"\footnotesize",
        colsep="4.5pt",
    )


def write_mech_table() -> None:
    rows = [
        ["r6.2", "Token-SAE causal", "role", "0.006848", "[0.003362, 0.010790]", "$6.0\\times$ comparator"],
        ["r6.2", "Token-SAE necessity", "project$\\times$role", "0.065188", "[0.055145, 0.075023]", "all contexts positive"],
        ["---"],
        ["r4.2", "Transferred causal", "multiple", "$<0$", "all audited configs $<0$", "direct transfer fails"],
        ["r4.2", "Native token-SAE causal", "team", "0.001418", "[0.001139, 0.001690]", "$1.6\\times$ comparator"],
        ["r4.2", "Native token-SAE necessity", "dept$\\times$role", "0.002922", "[0.001460, 0.004379]", "necessity partial"],
    ]
    write_table(
        TABLES / "cert_mechanistic_summary.tex",
        "Mechanistic summary (best context mode per estimand). All rows use the "
        "same-user-excluded protocol with active-control feature sets; the "
        "transferred row applies the r6.2 layer-18 configuration to r4.2 without "
        "re-fitting. Effects are paired complete-case top-versus-control "
        "contrasts with 95\\% bootstrap confidence intervals. Comparator ratios "
        "are relative to the best matched session-autoencoder contrast on the "
        "same receivers (r6.2: 0.001133; r4.2: 0.000909).",
        "tab:cert_mechanistic",
        "lllccl",
        ["Dataset", "Estimand", "Context", "Effect", "95\\% CI", "Note"],
        rows,
    )


def write_claims_table() -> None:
    rows = [
        ["Benign-only QLoRA training is valid one-class training", "Supported"],
        ["r6.2 has strong token-level causal structure", "Supported"],
        ["Direct token-mechanism transfer from r6.2 to r4.2 succeeds", "Rejected"],
        ["r4.2 has a native rediscovered token mechanism", "Supported"],
        ["The session LM detector is strongly superior on CERT", "Not supported"],
    ]
    write_table(
        TABLES / "claim_status.tex",
        "Summary of claims and their evidential status.",
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
