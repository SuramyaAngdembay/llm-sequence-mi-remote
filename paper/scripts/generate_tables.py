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
        ["r6.2", "Session LM (adapted NLL)", "0.0008", r"\textbf{0.953}", "0.054", "0.970", r"\textbf{24.0}"],
        ["", "Deep SVDD", r"\textbf{0.0115}", "0.628", r"\textbf{0.211}", "--", "83.2"],
        ["", "GRU AE", "0.0057", "0.766", "0.081", "--", "24.8"],
        ["", "LSTM AE", "0.0021", "0.768", "0.057", "--", "24.2"],
        ["", "Isolation Forest", "0.0002", "0.713", "0.013", "--", "153.0"],
        ["", r"\emph{Session LM, user-disjoint benign}", "0.0005", "0.545", "0.013", "0.546", "--"],
        ["---"],
        ["r4.2", "Session LM (adapted NLL)", "0.0134", r"\textbf{0.964}", "0.101", "0.969", r"\textbf{26.4}"],
        ["", "Deep SVDD", r"\textbf{0.0337}", "0.743", r"\textbf{0.382}", "--", "53.4"],
        ["", "GRU AE", "0.0254", "0.696", "0.124", "--", "86.6"],
        ["", "LSTM AE", "0.0236", "0.714", "0.120", "--", "92.1"],
        ["", "Isolation Forest", "0.0003", "0.715", "0.008", "--", "186.4"],
        ["", r"\emph{Session LM, user-disjoint benign}", "0.1023", "0.668", "0.521", "0.565", "--"],
    ]
    write_table(
        TABLES / "cert_detector_comparison.tex",
        "Detector comparison on CERT under two protocols. Fold-aligned rows follow the baselines' protocol, in which the session LM (trained once on ~90 percent of benign users) faces mostly training-seen benign test users while baselines exclude test users from training; bold marks the best fold-aligned value per metric (held-out rank: lower is better). The italicized user-disjoint rows restrict the LM's benign comparison population to never-trained validation users and are the fairer generalization test for the LM: its ranking advantage largely disappears (day ROC 0.954 to 0.545 on r6.2 and 0.959 to 0.668 on r4.2 computed on the full evaluation pool; the fold-aligned column uses the baselines' per-fold test populations, hence its slightly different 0.953/0.964), showing the fold-aligned strength is mostly a seen-versus-unseen-user effect; under the user-disjoint protocol the LM's day ROC falls at or below every baseline's fold-aligned value. User-disjoint PR values are computed at a very different prevalence (60 malicious versus 79 benign users on r4.2) and are not comparable to the fold-aligned column; the informative cross-protocol comparison is the ROC collapse. Baseline user-level ROC is not reported by the fold-aligned benchmark harness.",
        "tab:cert_detector",
        "llccccc",
        ["Dataset", "Method", "Day PR-AUC", "Day ROC-AUC", "User PR-AUC", "User ROC-AUC", "Held-out rank"],
        rows,
        size=r"\footnotesize",
        colsep="3pt",
    )


def write_mech_table() -> None:
    rows = [
        ["r6.2", "Token-SAE causal", "role", "0.006848", "[0.000092, 0.009996]", "4/4 users positive"],
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
        "contrasts; intervals are the prespecified user-level cluster bootstrap "
        "(10,000 draws, malicious user as resampling unit; descriptive for "
        "r6.2's four clusters).",
        "tab:cert_mechanistic",
        "lllccl",
        ["Dataset", "Estimand", "Context", "Effect", "95\\% CI", "Note"],
        rows,
        colsep="5pt",
    )


def write_claims_table() -> None:
    rows = [
        ["Benign-only QLoRA training is valid one-class training", "Supported"],
        ["Fold-aligned detector strength reflects behavioral discrimination", "Rejected (seen-vs-unseen-user effect)"],
        ["r6.2 contains a profile-bound sparse feature family with causal relevance under the audited estimands", "Supported descriptively; held-out replication concentrates on the dominant user; patching size-uncalibrated (App.~A.5)"],
        ["r4.2 contains a behavior-associated sparse feature family supported by patching and ablation", "Supported; directional replication on held-out users; ablation replication persists under a benign-only dictionary, patch-repair replication does not"],
        ["Configuration-independent r4.2 confirmation", "Not established"],
        ["Literal feature transfer across benchmarks succeeds", "Rejected"],
        ["Transfer failure is explained by SAE seed non-identifiability", "Rejected (alignment controls)"],
        ["Positive-population size alone explains the feature-selection dissociation (conditional on the fixed r4.2 dictionary)", "Rejected at the selection stage (subsampling)"],
        ["The profile/behavior attribution dissociation requires positive-shaped dictionaries", "Rejected: benign-only SAE retrain reproduces it (r4.2 top-5 $\\geq$94\\% behavioral; r6.2 $\\geq$99\\% profile-bound in all LOUO folds)"],
    ]
    write_table(
        TABLES / "claim_status.tex",
        "Audit claim map: what the evidence supports, rejects, and fails to support.",
        "tab:claim_status",
        "p{7.6cm}p{5.6cm}",
        ["Claim", "Status"],
        rows,
    )


def write_dictionary_robustness_table() -> None:
    rows = [
        ["r6.2", "Full-pool", "Profile-bound (99.8--100\\%)", "Dominant-user concentrated", "User-specific (negative on dominant fold)"],
        ["r6.2", "Benign-only", "$\\geq$99\\% profile-bound, all folds", "Positive on one held-out fold", "Positive on dominant-user fold"],
        ["---"],
        ["r4.2", "Full-pool", "4/5 behavioral", "Directional; one context significant", "Partial"],
        ["r4.2", "Benign-only", "$\\geq$94\\% behavioral", "Does not replicate", "Replicates in all four contexts"],
    ]
    write_table(
        TABLES / "dictionary_robustness.tex",
        "Dictionary-independence summary. Attribution (what the selected features encode) reproduces exactly under SAE dictionaries retrained on benign rows only, while held-out replication of the intervention estimands is estimand- and dictionary-dependent.",
        "tab:dict_robustness",
        "llp{3.4cm}p{3.6cm}p{3.6cm}",
        ["Dataset", "Dictionary", "Attribution", "Held-out patching", "Held-out ablation"],
        rows,
        size=r"\footnotesize",
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
        colsep="3pt",
    )


def write_alignment_table() -> None:
    rows = [
        ["within r6.2 (3 seed pairs)", "0.881--0.929", "0.579--0.586"],
        ["within r4.2 (3 seed pairs)", "0.881--0.955", "0.635--0.645"],
        ["across benchmarks, native layers (18 vs 26)", "0.079--0.112", "0.075--0.093"],
        ["across benchmarks, matched layer 18", "0.264--0.434", "0.194--0.348"],
    ]
    write_table(
        TABLES / "alignment.tex",
        "Decoder-space feature alignment. Best-match $|\\cos|$ of each source SAE's "
        "top-5 features into a target dictionary, versus the whole-dictionary median "
        "(empirical null). Within-benchmark cross-seed alignment is far above "
        "the null; native-layer cross-benchmark alignment is indistinguishable "
        "from it, and matched-layer alignment remains far below cross-seed levels.",
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
    write_dictionary_robustness_table()
    write_attribution_table()
    write_alignment_table()


if __name__ == "__main__":
    main()
