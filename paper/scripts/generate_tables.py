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
        ["", r"\emph{Session LM, user-disjoint benign}", "0.0005", "0.545", "0.013", "--"],
        ["---"],
        ["r4.2", "Session LM (adapted NLL)", "0.0134", r"\textbf{0.964}", "0.101", r"\textbf{26.4}"],
        ["", "Deep SVDD", r"\textbf{0.0337}", "0.743", r"\textbf{0.382}", "53.4"],
        ["", "GRU AE", "0.0254", "0.696", "0.124", "86.6"],
        ["", "LSTM AE", "0.0236", "0.714", "0.120", "92.1"],
        ["", "Isolation Forest", "0.0003", "0.715", "0.008", "186.4"],
        ["", r"\emph{Session LM, user-disjoint benign}", "0.1023", "0.668", "0.521", "--"],
    ]
    write_table(
        TABLES / "cert_detector_comparison.tex",
        "Detector comparison on CERT under two protocols. Fold-aligned rows follow the baselines' protocol, in which the session LM (trained once on ~90 percent of benign users) faces mostly training-seen benign test users while baselines exclude test users from training; bold marks the best fold-aligned value per metric (held-out rank: lower is better). The italicized user-disjoint rows restrict the LM's benign comparison population to never-trained validation users and are the fair comparison for the LM: its ranking advantage largely disappears (day ROC 0.953 to 0.545 on r6.2; 0.964 to 0.668 on r4.2), showing the fold-aligned strength is mostly a seen-versus-unseen-user effect. User-disjoint PR values are on a different benign population size and are not comparable to the fold-aligned column.",
        "tab:cert_detector",
        "llcccc",
        ["Dataset", "Method", "Day PR-AUC", "Day ROC-AUC", "User PR-AUC", "Held-out rank"],
        rows,
        size=r"\footnotesize",
        colsep="4.5pt",
    )


def write_mech_table() -> None:
    rows = [
        ["r6.2", "Token-SAE causal", "role", "0.006848", "[0.000092, 0.010000]", "4/4 users positive"],
        ["r6.2", "Token-SAE necessity", "project$\\times$role", "0.065188", "[0.026059, 0.082920]", "4/4 users positive"],
        ["---"],
        ["r4.2", "Transferred causal", "multiple", "$<0$", "all audited configs $<0$", "direct transfer fails"],
        ["r4.2", "Native token-SAE causal", "team", "0.001418", "[0.000967, 0.001863]", "all contexts positive"],
        ["r4.2", "Native token-SAE necessity", "dept$\\times$role", "0.002922", "[0.000911, 0.005005]", "necessity partial"],
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
        ["Fold-aligned detector strength reflects behavioral discrimination", "Rejected (seen-vs-unseen-user effect)"],
        ["r6.2 has a sparse profile-bound causal mechanism", "Supported descriptively; held-out replication concentrates on the dominant user"],
        ["r4.2 has a sparse behavioral-associated causal mechanism", "Supported; feature-level directional replication on held-out users"],
        ["Configuration-independent r4.2 confirmation", "Not established"],
        ["Literal feature transfer across benchmarks succeeds", "Rejected"],
        ["Transfer failure is explained by SAE seed non-identifiability", "Rejected (alignment controls)"],
        ["Positive-population size alone explains the profile/behavior dissociation", "Rejected (subsampling)"],
    ]
    write_table(
        TABLES / "claim_status.tex",
        "Audit claim map: what the evidence supports, rejects, and fails to support.",
        "tab:claim_status",
        "p{7.6cm}p{5.6cm}",
        ["Claim", "Status"],
        rows,
    )


def write_attribution_table() -> None:
    rows = [
        ["r6.2", "14358", "0.998", "0.000", "0.002", "13.4$\\times$", "psychometric values"],
        ["r6.2", "12848", "0.999", "0.000", "0.001", "13.4$\\times$", "psychometric values"],
        ["r6.2", "4196", "1.000", "0.000", "0.000", "13.4$\\times$", "psychometric values"],
        ["r6.2", "13580", "0.000", "0.999", "0.001", "$\\approx$0", "org header"],
        ["r6.2", "11292", "1.000", "0.000", "0.000", "13.4$\\times$", "psychometric values"],
        ["---"],
        ["r4.2", "4596", "0.001", "0.000", "0.999", "$\\approx$0", "session values"],
        ["r4.2", "3673", "0.000", "0.000", "1.000", "$\\approx$0", "session durations"],
        ["r4.2", "2302", "0.001", "0.998", "0.001", "$\\approx$0", "org header"],
        ["r4.2", "3455", "0.000", "0.000", "1.000", "$\\approx$0", "session values"],
        ["r4.2", "1268", "0.000", "0.000", "1.000", "$\\approx$0", "session values"],
    ]
    write_table(
        TABLES / "attribution.tex",
        "Token attribution of the top-5 causal features (positive examples). "
        "Columns are activation-mass fractions by serialization line class; PSY enrich "
        "is mass fraction over token share for the psychometric line. r6.2 features are "
        "profile-bound; four of five r4.2 features are behavioral (SES enrichment 1.33x "
        "over a 0.75 token share).",
        "tab:attribution",
        "llccccl",
        ["Bench", "Feature", "PSY mass", "DAY mass", "SES mass", "PSY enrich", "Top tokens"],
        rows,
        size=r"\footnotesize",
        colsep="4.5pt",
    )


def write_alignment_table() -> None:
    rows = [
        ["within r6.2 (3 seed pairs)", "0.881--0.929", "0.579--0.586"],
        ["within r4.2 (3 seed pairs)", "0.881--0.955", "0.635--0.645"],
        ["across benchmarks (both directions, 3 seeds/side)", "0.079--0.112", "0.075--0.093"],
    ]
    write_table(
        TABLES / "alignment.tex",
        "Decoder-space feature alignment. Best-match $|\\cos|$ of each source SAE's "
        "top-5 features into a target dictionary, versus the whole-dictionary median "
        "(chance). Within-benchmark cross-seed alignment is far above chance; "
        "cross-benchmark alignment is at chance.",
        "tab:alignment",
        "lcc",
        ["Comparison", "Top-5 best-match $|\\cos|$", "Empirical null baseline"],
        rows,
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    write_detector_table()
    write_mech_table()
    write_claims_table()
    write_attribution_table()
    write_alignment_table()


if __name__ == "__main__":
    main()
