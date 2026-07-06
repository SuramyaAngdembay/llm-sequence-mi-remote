# R4.2 Native Remote Token Search

Status: this note defines the next `r4.2` remote mechanistic step after the
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

