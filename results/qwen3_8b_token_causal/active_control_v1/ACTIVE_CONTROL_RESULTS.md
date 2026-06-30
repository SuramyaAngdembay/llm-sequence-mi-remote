# Qwen3-8B Active-Control Causal Rerun Results

This bundle records the tightened active-control rerun for the strongest
`Qwen3-8B` token causal configs on `r6.2`.

## Jobs

| config | causal job | bootstrap job | state |
|---|---:|---:|---|
| `l18_m04_k04_top5_control5_active` | `18728774` | `18728775` | completed |
| `l18_m04_k08_top5_control5_active` | `18728776` | `18728777` | completed |

Both causal jobs completed with exit code `0:0` on Anvil AI nodes. Both
bootstrap jobs completed with exit code `0:0` on the shared partition.

## Protocol

- dataset: `r6.2`
- model branch: `Qwen/Qwen3-8B`
- layer: `18`
- configs: `latent_mult=4,k=4` and `latent_mult=4,k=8`
- target set: `top5`
- control set: `control5_active`
- active-control threshold: `active_control_min_frac=0.002`
- receivers: `70` positive eval examples
- token rows: `1,112,836`
- candidate rows per run: `71,080`

The active-control set was selected to avoid the inert-control failure found in
the earlier `control3` audit.

## Bootstrap Results

### `l18_m04_k04_top5_control5_active`

| context | estimate | 95% CI |
|---|---:|---:|
| `project_role` | `0.041975` | `[0.033279, 0.050582]` |
| `dept_role` | `0.040670` | `[0.032455, 0.049084]` |
| `role` | `0.040655` | `[0.032362, 0.049114]` |
| `team` | `0.022219` | `[0.017022, 0.027499]` |

### `l18_m04_k08_top5_control5_active`

| context | estimate | 95% CI |
|---|---:|---:|
| `role` | `0.018653` | `[0.013396, 0.024391]` |
| `dept_role` | `0.016378` | `[0.012051, 0.021068]` |
| `project_role` | `0.015664` | `[0.011314, 0.020448]` |
| `team` | `0.008150` | `[0.005584, 0.010818]` |

## Interpretation

The tightened active-control rerun preserves the main `Qwen3-8B` token causal
finding:

- `l18_m04_k08` remains the conservative headline config because it was already
  the clean audited result and stays positive against `control5_active`.
- `l18_m04_k04` remains very strong even after replacing the inert `control3`
  with `control5_active`, so the previous upper-bound caveat is substantially
  reduced.

The result supports carrying the same token-level causal protocol forward into
the `r4.2` transfer run.
