# Full-pilot control mismatch: outcome-blind diagnosis and separate diagnostics

The full corrected pilot completed 316 receivers × 15 conditions. The original
all-controls gate failed and remains failed. One of the 1,693 active selected-arm
tokens failed for `randD_S`: receiver 369 (`AAM0658:298`, window-negative), position
95 (`SES`). No token, receiver or condition is dropped, and the 5% tolerance stays
unchanged. No losses were evaluated in choosing the following diagnostic plan.

## Exact arithmetic reproduction

Using the frozen SAE, donor prototype, RNG and saved live state reproduces every
recorded measurement for that receiver's `rpU_S` and `randD_S` exactly. CPU and GPU
BF16 cast/add outputs are also exactly equal. Requested norms were 2.56722558 and
2.56722562; realized norms were 2.57188134 and 2.40677250. The 6.41977% mismatch is
real and comes from the numerical implementation, not another label error.

The dominant control coordinate has hidden value 65 and BF16 spacing 0.5. Its
requested +2.21170974 is cast to +2.21875, then the hidden-state sum rounds to 67:
the actual change is +2.0. Its squared-norm reduction is 0.89166. Other coordinates
also round, so this coordinate alone is not an exact decomposition of total norm
error.

For the fixed control direction and nonnegative float32 scalar, each coordinate's
displacement magnitude is nondecreasing under scalar multiplication and BF16
cast/add rounding. Searching the ordered float32 bit patterns gives adjacent
scalars 1.020844817 and 1.020844936, with realized norms 2.42190613 and 2.84879436.
They bracket the target 2.57188134 and skip the entire allowed interval
[2.44328727, 2.70047540]. Thus simple scalar calibration cannot satisfy the existing
5% gate for this direction/state under the original arithmetic. This is a narrow
feasibility result, not a claim of mathematical novelty or universal impossibility.

## Endpoint dependencies, recorded before viewing scientific effect estimates

The frozen primary uses `rpU_S - randI_S` on attack days. The co-primary and
specificity equations use `rpU_S` and `rpU_C`, not `randD_S`. The secondary
`rpU_S - randD_S` uses attack days, whereas the failed token is window-negative.
The failure does enter window-negative `randD_S` condition diagnostics and any
new contrast using those observations. Audit every required comparison by day
type directly from saved token measurements; do not infer passing from this one
failure alone.

A separately named **post-run implementation diagnostic** may report every
originally specified estimate, with the same rows, user weights, 10,000 bootstrap
draws, seed, and endpoints. It must carry the failed original global gate, list
the per-endpoint dependencies, and mark `scientific_claim_cleared=false`. Local
checks do not retrospectively pass the frozen protocol. Selected-versus-control
features remain different policies, not equal-dose interventions. The population
remains exploratory and transductive.

The default scientific analyzer continues to reject the run. Its explicit
`--implementation-diagnostic` mode retains output, label, token-grid, arithmetic
and hash validation, rejects engineering-only cohorts, and adds the diagnostic
scope. No corrected intervention is claimed or rerun here. Changing direction,
precision, or rounding policy would be a separately specified experiment.

Evidence: `control_rounding_diagnostic_v1/diagnostic.json`,
`control_rounding_diagnostic_v1/scalar_feasibility.json`, and
`full_completion_check_v1.json` under
`/data/suramya/cert_review_20260928/causal_validation_20260929/`.
