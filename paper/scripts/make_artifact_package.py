#!/usr/bin/env python3
"""Build the anonymized code+results supplementary package (build/artifact_package.zip).

Layout mirrors the Sept-3 package: artifact_pkg/{README.md, code/, code/configs/, paper_gen/, results/}.
Text files are anonymized by substituting identifying strings with the placeholders named in the README.
"""
from __future__ import annotations
import io, re, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "build" / "artifact_package.zip"
TEXT_EXT = {".py", ".sh", ".sbatch", ".yaml", ".yml", ".md", ".txt", ".json", ".csv", ".tex", ".log", ".cfg", ".toml"}
MAX_BYTES = 2_000_000

CODE_FILES = sorted({
    "audit3b_select_attribute.py", "benchmark_qlora_throughput.py", "benchmark_qlora_vram.py",
    "benchmark_token_causal_vram.py", "bootstrap_token_delta_sae_causal.py", "bootstrap_token_delta_sae_necessity.py",
    "build_session_jsonl.py", "build_session_jsonl_fast.py", "cluster_bootstrap_token_delta_sae.py",
    "compare_masked_scores.py", "compare_remote_token_vs_local_session.py",
    "eval_delta_sae_causal.py", "eval_example_scores_detector_metrics.py", "eval_fold_aligned_detector_metrics.py",
    "eval_token_delta_sae_causal.py", "eval_token_delta_sae_necessity.py", "extract_adapter_deltas.py",
    "feature_token_attribution.py", "intervention_norm_report.py", "make_masked_session_jsonl.py",
    "make_positive_user_split.py", "make_swap_counterfactuals.py", "masking_bootstrap_ci.py",
    "norm_normalized_contrast.py", "population_subsample_attribution.py", "prepare_transfer_package.py",
    "rank_behavioral_features.py", "remote_common.py", "reselect_token_sae_features.py", "sae_core.py",
    "sae_seed_alignment.py", "score_adapter_examples.py", "select_matched_controls.py", "train_delta_sae_frontier.py",
    "train_qlora.py", "val_only_detector_sensitivity.py", "within_user_day_ranking.py",
})
CODE_DIRS = ["scripts/factorial", "scripts/lanl", "slurm/anvil_friend"]
RESULT_DIRS = [
    "behavioral_louo", "behavioral_ranking", "benign_dictionary", "cross_arch_probe", "feature_attribution",
    "masking_ablation", "matched_controls", "population_subsample_r42", "qwen3_8b_r42_token_causal",
    "qwen3_8b_r42_token_necessity", "qwen3_8b_r42_transfer", "qwen3_8b_token_causal", "qwen3_8b_token_frontier",
    "qwen3_8b_token_necessity", "qwen3b_pilot", "qwen3b_repaired", "sae_seed_stability", "swap_counterfactuals",
    "twos_replication", "valonly_detector", "within_user_ranking",
    "train_matched_factorial", "lanl_auth_replication",
]
RESULT_FILES = ["CLUSTER_BOOTSTRAP_2026-07-18_SUMMARY.md"]

# order matters: longer / more specific patterns first
SUBS = [
    (re.compile(r"https?://github\.com/[A-Za-z0-9_./-]+"), "<REPOSITORY_URL>"),
    (re.compile(r"/anvil/scratch/x-[a-z0-9]+"), "<SCRATCH>"),
    (re.compile(r"/home/x-[a-z0-9]+"), "<HOME>"),
    (re.compile(r"/Users/suramya"), "<HOME>"),
    (re.compile(r"anvil\.rcac\.purdue\.edu|magnolia\.usm\.edu"), "<CLUSTER>"),
    (re.compile(r"/anvil/scratch"), "<SCRATCH>"),
    (re.compile(r"\bMagnolia\b|\bmagnolia\b"), "<CLUSTER-B>"),
    (re.compile(r"\bAnvil\b"), "<CLUSTER-A>"),
    (re.compile(r"x-sangdembay|srangdembay|suramya", re.I), "<USER>"),
    (re.compile(r"x-bbhusal1|bhusal", re.I), "<COLLABORATOR>"),
    (re.compile(r"cis260991-gpu|tra250034-gpu|cis260991|tra250034"), "<ALLOCATION>"),
    (re.compile(r"llm-sequence-mi-remote|cert-insider-threat-qlora[a-z-]*"), "<REPOSITORY>"),
    (re.compile(r"Angdembay|Haiyan Tian|\bTian\b"), "<AUTHOR>"),
    (re.compile(r"usm\.edu"), "<INSTITUTION>"),
]

def anonymize(data: bytes) -> bytes:
    try:
        s = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    for pat, rep in SUBS:
        s = pat.sub(rep, s)
    return s.encode("utf-8")

def add(z: zipfile.ZipFile, src: Path, arc: str) -> None:
    data = src.read_bytes()
    if src.suffix.lower() in TEXT_EXT:
        data = anonymize(data)
    z.writestr("artifact_pkg/" + arc, data)

def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("artifact_pkg/README.md", anonymize((ROOT / "paper" / "scripts" / "artifact_readme.md").read_bytes()))
        for f in CODE_FILES:
            p = ROOT / "scripts" / f
            if p.exists():
                add(z, p, f"code/{f}"); n += 1
            else:
                print("missing code file:", f, file=sys.stderr)
        cap = ROOT / "results" / "cross_arch_probe" / "cross_arch_probe.py"
        if cap.exists():
            add(z, cap, "code/cross_arch_probe.py"); n += 1
        for d in CODE_DIRS:
            for p in sorted((ROOT / d).glob("*")):
                if p.is_file() and p.suffix.lower() in TEXT_EXT:
                    add(z, p, f"code/{Path(d).name}/{p.name}"); n += 1
        for p in sorted((ROOT / "configs").glob("*.y*ml")):
            add(z, p, f"code/configs/{p.name}"); n += 1
        for p in sorted((ROOT / "paper" / "scripts").glob("*")):
            if p.is_file() and p.suffix in {".py", ".sh"} and p.name != "make_artifact_package.py":
                add(z, p, f"paper_gen/{p.name}"); n += 1
        for f in RESULT_FILES:
            add(z, ROOT / "results" / f, f"results/{f}"); n += 1
        for d in RESULT_DIRS:
            base = ROOT / "results" / d
            if not base.exists():
                print("missing results dir:", d, file=sys.stderr); continue
            for p in sorted(base.rglob("*")):
                if p.is_file() and p.suffix.lower() in TEXT_EXT and p.stat().st_size <= MAX_BYTES:
                    add(z, p, f"results/{p.relative_to(ROOT / 'results')}"); n += 1
    print(f"wrote {OUT} ({n} files, {OUT.stat().st_size/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
