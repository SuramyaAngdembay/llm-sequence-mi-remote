# Qwen3-8B Token Causal Eval Results

This is the Anvil lightweight result bundle for the Qwen3-8B token-level
causal patching run requested after the token SAE frontier handoff.

## Completed Runs

All corrected full-suite jobs completed after commit `5feae1f`
(`Fix Qwen3 causal eval Slurm exports`). That fix mattered because the
comma-valued Slurm exports for context modes, target sets, and alphas were
previously being truncated.

| config | causal job | bootstrap job | elapsed | MaxRSS |
|---|---:|---:|---:|---:|
| `l18_m04_k04` | `18708729` | `18708730` | `01:18:10` | `198306404K` |
| `l18_m02_k04` | `18708731` | `18708732` | `01:14:15` | `126352084K` |
| `l18_m04_k08` | `18708733` | `18708734` | `01:21:00` | `198352232K` |

The B12/B16 throughput probes also completed:

| probe | job | elapsed | MaxRSS |
|---|---:|---:|---:|
| `l18_m04_k04_b12` | `18706862` | `00:12:05` | `20599244K` |
| `l18_m04_k04_b16` | `18706863` | `00:11:58` | `20741400K` |

## Evaluated Grid

- layer: `18`
- full-suite configs: `m04_k04`, `m02_k04`, `m04_k08`
- context modes: `team`, `role`, `project_role`, `dept_role`
- target sets: `top1`, `top3`, `top5`
- alphas: `0.25`, `0.5`, `0.75`, `1.0`
- full-suite batch size: `BATCH_SIZE=8`
- candidate rows per full causal run: `142160`
- token rows available: `1112836`

## Top Bootstrap Rows

The strongest row is `l18_m04_k04 / project_role / top5` with estimate
`0.042177` and 95% CI `[0.033429, 0.050793]`. There are `17` rows with
positive CI lower bounds across the three full-suite configs.

| config | context | target | estimate | 95% CI |
|---|---|---|---:|---:|
| `l18_m04_k04` | `project_role` | `top5` | `0.042177` | `[0.033429, 0.050793]` |
| `l18_m04_k04` | `role` | `top5` | `0.040864` | `[0.032626, 0.049304]` |
| `l18_m04_k04` | `dept_role` | `top5` | `0.040479` | `[0.032252, 0.048826]` |
| `l18_m04_k04` | `team` | `top1` | `0.034279` | `[0.027978, 0.040564]` |
| `l18_m04_k04` | `team` | `top5` | `0.024662` | `[0.019391, 0.030130]` |
| `l18_m04_k08` | `role` | `top5` | `0.018769` | `[0.013444, 0.024649]` |
| `l18_m04_k08` | `dept_role` | `top5` | `0.016588` | `[0.012042, 0.021516]` |

The `l18_m02_k04` config underperformed the other two in this causal run.

## Audited Read

After these lightweight results were committed, the missing
`token_delta_sae_causal_candidate_rows.csv` files for the strongest configs were
synced back to Magnolia and inspected locally.

That audit changes how the three configs should be interpreted:

- `l18_m04_k08` is the **clean headline result**
- `l18_m04_k04` is a strong **upper-bound result with a weak-control caveat**
- `l18_m02_k04` is a negative config

The key control audit finding is:

- `l18_m04_k04` `control3` rows are completely inert
  - mean active receiver tokens: `0.0` for both benign and anomalous donors
- `l18_m04_k08` `control3` rows are active and nontrivial
  - mean active receiver tokens: about `23.1`

So while `l18_m04_k04` shows the largest raw effect, its control is too weak to
use as the main headline. By contrast, `l18_m04_k08` stays clearly positive even
against an active control set:

| config | context | target | estimate | 95% CI |
|---|---|---|---:|---:|
| `l18_m04_k08` | `role` | `top5` | `0.018769` | `[0.013444, 0.024649]` |
| `l18_m04_k08` | `dept_role` | `top5` | `0.016588` | `[0.012042, 0.021516]` |
| `l18_m04_k08` | `project_role` | `top5` | `0.016075` | `[0.011553, 0.020986]` |

Strict matched day-level comparison against the local session baseline also
confirms that the remote `8B` token branch is ahead:

- local adaptive day-level best: `0.001133`
- local residual day-level best: `0.000654`
- remote `l18_m04_k08` best: `0.018769`

So the conservative conclusion is now:

- `Qwen3-8B` token-level patching is the strongest mechanistic branch so far
- `l18_m04_k08` is the audited headline result
- `l18_m04_k04` should be treated as an upper-bound variant because of the inert
  control

## Active-Control Rerun

The tightened active-control rerun completed on Anvil after the inert-control
audit. It directly reran the two strongest layer-18 configs with
`control5_active` instead of the weak `control3` comparison:

| config | causal job | bootstrap job | control |
|---|---:|---:|---|
| `l18_m04_k04_top5_control5_active` | `18728774` | `18728775` | `control5_active` |
| `l18_m04_k08_top5_control5_active` | `18728776` | `18728777` | `control5_active` |

Top active-control bootstrap rows:

| config | context | target | estimate | 95% CI |
|---|---|---|---:|---:|
| `l18_m04_k04_top5_control5_active` | `project_role` | `top5` | `0.041975` | `[0.033279, 0.050582]` |
| `l18_m04_k04_top5_control5_active` | `dept_role` | `top5` | `0.040670` | `[0.032455, 0.049084]` |
| `l18_m04_k04_top5_control5_active` | `role` | `top5` | `0.040655` | `[0.032362, 0.049114]` |
| `l18_m04_k08_top5_control5_active` | `role` | `top5` | `0.018653` | `[0.013396, 0.024391]` |
| `l18_m04_k08_top5_control5_active` | `dept_role` | `top5` | `0.016378` | `[0.012051, 0.021068]` |

Lightweight active-control files are committed under:

- `results/qwen3_8b_token_causal/active_control_v1/`

This preserves the conservative headline (`l18_m04_k08`) and shows that the
strong `l18_m04_k04` effect remains large when compared against active controls.

## Git Bundle

Committed lightweight files for each full config:

- `TOKEN_DELTA_SAE_CAUSAL_REPORT.md`
- `token_delta_sae_causal_summary.csv`
- `token_delta_sae_causal_summary.json`
- `token_delta_sae_causal_selected_sets.csv`
- `token_delta_sae_causal_best_rows.csv`
- `bootstrap/TOKEN_DELTA_SAE_BOOTSTRAP_REPORT.md`
- `bootstrap/token_delta_sae_bootstrap_summary.csv`

Committed lightweight files for each probe:

- `TOKEN_DELTA_SAE_CAUSAL_REPORT.md`
- `token_delta_sae_causal_summary.csv`
- `token_delta_sae_causal_summary.json`
- `token_delta_sae_causal_selected_sets.csv`
- `token_delta_sae_causal_best_rows.csv`

The large `token_delta_sae_causal_candidate_rows.csv` files are intentionally
not committed.

## Anvil Runtime Paths

- full causal root:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_mb12_gc_on_fresh`
- probe root:
  `/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_mb12_gc_on_fresh_probe`

The full causal root is about `73M`, mostly from candidate-row CSVs. The git
bundle is about `445K`.
