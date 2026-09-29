# Separate control feasibility experiment

This is a new exploratory control policy, declared after diagnosing the failed
full run and before sampling or scoring replacement candidates. Original outcomes
are available; the policy uses only requested edits, frozen hidden states and
numerical measurements. It does not change the original run or retroactively
pass its protocol. The diagnostic analysis and original failure remain reported.

- Keep every original dictionary-control draw whose realized token support and
  norm agree with its reference within the existing 5% rule.
- For a failed draw only, sample independent dictionary directions from the same
  active feature pool (activity >= .002, excluding selected and control sets).
  Keep the original requested norm. Use at most 32 candidates, accepting the
  first whose BF16 cast/add realization meets the same support/norm rule.
- RNG is fixed by run seed, receiver index, token position, condition and the
  `dictionary-control-bf16-rejection-v1` hash salt. Do not select by loss or labels.
  Log all attempts, including failures. No feasible candidate means failure.
- This conditions the control distribution on numerical feasibility. It is not
  the original uniform dictionary control. The reference intervention, dtype,
  dictionary, population and target-token scoring do not change.
- Apply the policy to the known failing receiver as a bounded feasibility pilot.
  The full token audit establishes that this is the only failed dictionary-control
  token. First reproduce zero, reference and original-control scoring from the
  saved run. Require exact view means/target counts and exact live states/recorded
  edit measurements. Then score the replacement; record it separately.
- Do not merge the replacement into the completed run or call the original
  scientific protocol passed. This pilot answers whether a physically matched
  replacement exists and can be scored reproducibly. Any larger use must carry
  the new control-policy provenance.

The pilot uses Aquaman GPU 0 with a ten-minute cap, including startup. It requires
no adapter or SAE fitting. The existing failure is a window-negative `randD_S`
token outside the frozen primary/co-primary/specificity equations and the
attack-day dictionary-control secondary. Their values cannot change in this
separate receiver-level feasibility experiment.
