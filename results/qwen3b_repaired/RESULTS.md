# 3B Serializer-Repaired + Adapter-Seed Rerun (Aquaman, 2026-08-06..12)

Design: Qwen2.5-3B QLoRA (NF4 r16 a32, 1 epoch), REPAIRED serializer
(--repair-hyphenated-cols: restores file_n-to_usb1/-from_usb1/-file_act3/
-disk1; r4.2's release contains only file_n-disk1 of the four), 2 adapter
seeds x 2 benchmarks; r4.2 full 287,827-example corpus, r6.2 capped at
300k of 1.25M. Token deltas at layers 12/18/24; benign-only dictionaries
(r4.2 m2k4, r6.2 m4k8); full-positive-pool selection; token attribution.

## Attribution (profile mass of top-5, by layer)
| arm | l12 | l18 | l24 | verdict |
| --- | --- | --- | --- | --- |
| r4.2 s42 | 4/5 behavioral | 5/5 behavioral | 5/5 behavioral | BEHAVIORAL |
| r4.2 s43 | (best l24) | 4/5 behavioral | 5/5 behavioral | BEHAVIORAL |
| r6.2 s42 | mixed | behavioral-lean | 3/5 @100% profile | PROFILE AT DEPTH |
| r6.2 s43 | (best l24) | 2/5 profile | 4/5 @100% profile | PROFILE AT DEPTH |

## Detector (day ROC, adapted NLL, positives vs never-trained benigns)
r4.2: s42 0.678, s43 0.718 (8B: 0.668). r6.2: s42 0.414 (weak adapter),
s43 0.738. Profile dominance strongest in the better-trained r6.2 seed ->
profile capture is acquired with training, not a serialization artifact.

VERDICT: the profile/behavior dissociation reproduces at 3B with the
repaired serializer across both adapter seeds; r6.2 profile capture
concentrates at the deepest extracted layer at this scale.
