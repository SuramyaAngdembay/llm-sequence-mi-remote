# User-Level Cluster Bootstrap Summary — 2026-07-18

Motivated by an external methodological critique: the receiver/pair-level
bootstrap treats same-user days as independent units. This recomputes the
uncertainty for all four final same-user-excluded bundles with the malicious
user as the resampling unit (job `19365690`, CPU `shared`, 10000 draws).

Script: `scripts/cluster_bootstrap_token_delta_sae.py`.
Per-bundle artifacts: `*/same_user_recovery/*/cluster_bootstrap/`.

Pooled estimates are unchanged (they match the tracked bootstrap estimates);
only the uncertainty statement changes.

## Outcome by bundle (top5 vs control5_active)

### r6.2 causal (4 users; day counts 46/18/5/1 — CDE1846 dominates)

| context | pooled | cluster 95% CI | users positive |
| --- | ---: | --- | --- |
| role | 0.006848 | [0.000092, 0.009996] | 4/4 |
| dept_role | 0.006818 | [-0.000230, 0.009955] | 2/4 |
| project_role | 0.004201 | [-0.000058, 0.005892] | 3/4 |

Read: the causal effect is driven primarily by CDE1846 (46/70 days;
per-user estimate ~0.010 vs ~0.0001–0.0008 for the others). `role` remains
positive under clustering and is positive for all four users, but the day-level
CI materially overstated precision. **Downgrade r6.2 causal from "robust" to
"positive but dominated by one heavily active malicious user."**

### r6.2 necessity

| context | pooled | cluster 95% CI | users positive |
| --- | ---: | --- | --- |
| project_role | 0.065188 | [0.026059, 0.082920] | 4/4 |
| role | 0.062167 | [0.016480, 0.082691] | 3/4 |
| dept_role | 0.056603 | [-0.001859, 0.077447] | 2/4 |
| team | 0.052234 | [-0.018468, 0.070531] | 2/4 |

Read: **necessity `project_role` is the strongest surviving r6.2 claim** —
positive for every malicious user (0.019 / 0.082 / 0.032 / 0.103) with a
cluster CI clearly excluding zero. `role` also survives.

### r4.2 native causal (49–54 users per context)

| context | pooled | cluster 95% CI | users positive |
| --- | ---: | --- | --- |
| team | 0.001418 | [0.000967, 0.001863] | 33/49 |
| role | 0.001112 | [0.000638, 0.001541] | 41/52 |
| dept_role | 0.001067 | [0.000656, 0.001467] | 40/50 |
| dept | 0.000982 | [0.000583, 0.001360] | 34/54 |

Read: **fully survives user-level clustering** — all four contexts positive
with cluster CIs excluding zero, and mean-of-user-means CIs excluding zero as
well. With 49–54 users, r4.2 is now arguably the more robust *causal*
population, though its effects remain smaller than r6.2 and the
selection-inference caveat (config found by search) still applies until a
discovery/confirmation split is run.

### r4.2 native necessity

| context | pooled | cluster 95% CI | users positive |
| --- | ---: | --- | --- |
| dept_role | 0.002922 | [0.000911, 0.005005] | 33/51 |
| role | 0.002075 | [0.000133, 0.004192] | 31/54 |
| dept | 0.001155 | [-0.000943, 0.003196] | 37/57 |
| team | 0.000662 | [-0.001316, 0.002761] | 31/59 |

Read: same qualitative story as the day-level analysis — `dept_role` and
`role` positive, `dept` and `team` cross zero.

## Paper implications

- Keep the r4.2 causal claim; its uncertainty statement should now cite the
  cluster CIs (they are the honest ones and still exclude zero).
- Recenter the r6.2 story on necessity (`project_role`, cross-user) and state
  the causal result with the single-user-dominance caveat and cluster CI.
- Report per-user counts (46/18/5/1) and per-user estimates in an appendix.
- The day-level CIs should not be presented as the headline uncertainty
  statement anywhere.
