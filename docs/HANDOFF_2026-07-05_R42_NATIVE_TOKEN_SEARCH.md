# R4.2 Native Remote Token Search

Status: completed Anvil follow-up added on 2026-07-07.

## Completion Update

Anvil ran the first `r4.2`-native remote token-causal search from the existing
frontier rather than launching a broader new frontier immediately.

Tested full uncapped streamed causal configs:

- `l26_m02_k04`
- `l34_m04_k04`
- `l26_m04_k04`
- `l26_m04_k08`

Final successful launcher setting:

- `BATCH_SIZE=24`
- `PATCH_CHUNK_SIZE=24`
- `CAUSAL_MEM=480G`
- `TOKEN_DELTA_DTYPE=float32`

The attempted `BATCH_SIZE=32` rerun was faster but OOMed in the scoring loss
step. The completed `BATCH_SIZE=24` jobs all finished inside the 24h walltime.

Committed artifacts:

- `results/qwen3_8b_r42_token_causal/native_search_v3_bs24/`
- `results/qwen3_8b_r42_token_causal/vram_probe/`

Headline result:

| config | best context | target | estimate | CI low | CI high | read |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `l26_m02_k04` | `team` | `top5` | `0.001307` | `0.000960` | `0.001663` | positive |
| `l34_m04_k04` | `dept_role` | `top5` | `0.000119` | `-0.000038` | `0.000280` | weak/null |
| `l26_m04_k04` | `dept` | `top3` | `0.000054` | `-0.000149` | `0.000269` | null |
| `l26_m04_k08` | `role` | `top5` | `-0.000261` | `-0.000473` | `-0.000046` | negative |

Updated interpretation:

- direct `r6.2` token-mechanism transfer to `r4.2` failed
- `r4.2` is not remote-token-mechanism-free
- the native `r4.2` remote token mechanism is currently `layer 26, m02, k04`
- the best native remote estimate is competitive with the local `r4.2`
  adaptive comparator and close to the local residual comparator scale
- the effect remains much smaller than the positive `r6.2` remote token result

The full candidate-row CSVs were not committed because they are about
`413-419 MB` per config. They remain on Anvil under:

`/anvil/projects/x-cis230270/x-sangdembay/cert-qlora-MI/outputs/token_delta_sae_causal_qwen3_8b_r42_native_search_v3_bs24/`

If deeper audit is needed, start by rsyncing:

- `l26_m02_k04/token_delta_sae_causal_candidate_rows.csv`

## Original Planning Note

The rest of this note defines the `r4.2` remote mechanistic step after the
completed local-vs-remote comparator.

## Current State

We now have all of the following on the same `r4.2` problem:

- remote `Qwen3-8B` detector metrics:
  - detector transfer is positive
  - `adapted_nll` day PR-AUC is about `0.1788`
- remote full uncapped token-causal transfer runs:
  - all three transferred `r6.2`-style configs are negative
- local `r4.2` session-AE mechanistic comparator:
  - positive on the same `1309` receiver-days

The current strict day-level comparison is:

- best local adaptive day-level advantage: about `+0.000986`
- best local residual day-level advantage: about `+0.001380`
- best remote transferred token-causal advantage:
  - `l18_m04_k08`: about `-0.001132`
  - `l18_m04_k04`: about `-0.000941`
  - `l18_m02_k04`: about `-0.001047`

See:

- `results/qwen3_8b_r42_token_causal/strict_compare_local_session_daylevel_l18_m04_k08/`
- `results/qwen3_8b_r42_token_causal/strict_compare_local_session_daylevel_l18_m04_k04/`
- `results/qwen3_8b_r42_token_causal/strict_compare_local_session_daylevel_l18_m02_k04/`

## What This Means

The right read is now:

- `r4.2` does have causal structure
- the remote `r6.2` token mechanism does **not** transfer unchanged
- this is **not** just a detector collapse
- this is **not** just a streamed-evaluator artifact

So the next mainline question is:

- can the remote `Qwen3-8B` branch recover an `r4.2`-native token-causal
  mechanism if we search directly on `r4.2`, rather than forcing transferred
  `r6.2` settings?

## Main Recommendation

Do **not** spend the next cycle rerunning more transferred `r6.2` token
configs.

Do:

1. run an `r4.2`-native token-SAE frontier
2. pick the best `r4.2` configs from that frontier
3. run full uncapped token-causal eval on those `r4.2`-native configs
4. compare them directly against the local `r4.2` session-AE comparator

## Why This Is The Right Next Step

The local `r4.2` mechanistic outputs suggest that the operative mechanism is
not identical to the `r6.2` story.

Qualitatively, the current local `r4.2` strongest rows are more
`psychometric` / `org_context` centered than the strongest `r6.2`
USB-centered story.

That means a negative direct transfer result does **not** imply:

- no remote mechanism exists on `r4.2`

It may instead imply:

- the correct sparse token features live at different layers
- or require different latent width / `k`
- or the right `r4.2` token mechanism is semantically different

## R4.2-Native Frontier Proposal

Use the existing `r4.2` remote pipeline and rerun only the frontier/model
selection part, not the data prep.

Recommended search:

- layers: `14,18,22,26`
- latent multipliers: `2,4,8`
- top-k values: `4,8,16`

Keep fixed:

- token-level extraction
- benign-only QLoRA detector
- active-control capable causal eval
- full uncapped streamed evaluator
- same donor/receiver matching protocol

This is intentionally modest. The goal is not a giant architecture sweep. The
goal is to find whether `r4.2` has its own strong token-causal point.

## Existing Anvil Launcher Base

The current `r4.2` targeted remote pipeline launcher already exists:

- `scripts/submit_qwen3_8b_r42_targeted_pipeline_anvil.sh`

That is the right base to reuse. Override only the frontier-related parameters
needed for the `r4.2`-native search.

## Selection Rule

Do **not** choose the next `r4.2` causal configs by copied `r6.2` priors.

Choose them by:

1. frontier proxy selectivity
2. control activity quality
3. then full uncapped causal patching

If the first best row looks suspiciously dependent on inert or too-weak
controls, follow the same active-control audit pattern already used on `r6.2`.

## Success Criterion

The `r4.2` native remote search is successful only if at least one remote
token-causal config becomes:

- positive on `top_minus_control_advantage`
- under the full uncapped evaluator
- and competitive with the local `r4.2` comparator

Useful thresholds:

- beat `0.0` robustly first
- then compare to local adaptive `+0.000986`
- and local residual `+0.001380`

## What Not To Make Mainline Yet

There is older `r4.2` causal/transformer work on disk:

- native time-aware masked sequence
- CTMC transformer variants
- LSTM/transformer hybrid causal work

Those are useful side references and potential benchmark context, but they are
not the clean next mainline because they change the representation family and
causal object at the same time.

The cleanest next mechanistic test remains:

- same remote `Qwen3-8B` session-token branch
- but with an `r4.2`-native frontier instead of transferred `r6.2` settings

## Decision Rule After The Native Search

- if `r4.2`-native remote token-causal becomes positive:
  - conclude that direct mechanism transfer failed, but `r4.2` does support
    its own remote token-causal mechanism
- if `r4.2`-native remote token-causal stays negative:
  - conclude that the remote token branch is dataset-sensitive mechanistically,
    even when detector transfer remains positive

## Next Robustness Check

After the positive native search result, the next Anvil-side check is the
active-control audit on the current winner:

- `l26_m02_k04_top5_control5_active`

Launch with:

```bash
cd ~/cert-qlora-MI/llm-sequence-mi-remote
git pull origin main
bash scripts/submit_qwen3_8b_r42_native_active_control_anvil.sh
```

See:

- `docs/HANDOFF_2026-07-08_R42_NATIVE_ACTIVE_CONTROL.md`
