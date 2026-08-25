# TWOS Real-Data Replication (Aquaman pilot, 2026-08-24/25)

Data: TWOS (Harilal et al., MIST@CCS 2017) - 24 real users, 5 days, real
IPIP-50 Big-Five per user, 12 staged masquerade sessions (+5 traitor
instances, held for companion work). Serialized to CERT-analogous
DAY/PSY/SES 10-min windows (mouse+keystroke+eventviewer; 18,048 windows,
138 positives, 12-user benign train pool; scripts/twos/twos_serialize.py).
Pipeline: Qwen2.5-3B benign-only QLoRA (3 epochs, 2 adapter seeds) ->
token deltas l12/18/24 -> benign-only SAE (m4 k8) -> full-pool selection
-> token attribution; headline interventions on seed 42.

## Attribution (profile mass of top-5)
Both seeds x both layers essentially 100% behavioral with real OCEAN in
every window: s42 l24 profile=[0,0,0,0,0]; s42 l18 worst 0.017;
s43 l18/l24 all 0.000. (Also held in the v1 2-source corpus: 5 clean
replications total.)

## Detector
Within-user ROC (identity fixed): mean 0.783 over 16 malicious users.
Masquerade-only pool ROC 0.780.

## Interventions (s42, l24, team-matched, same-user excluded, 138 recv)
necessity +0.0019 / patching -0.0018 -> the CERT estimand asymmetry's
4th independent instance.

## Reading
12/24 users malicious => identity useless as a separator; the population-
structure account predicts behavioral selection - observed. The real-data
corner of the dissociation is confirmed with REAL psychometrics present.
