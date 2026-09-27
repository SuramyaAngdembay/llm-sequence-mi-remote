| case | condition | τ (mean) | ideal rescue I2 | ideal rescue I1 | necessity | interaction | realization error |
|---|---|---|---|---|---|---|---|
| 1 | irrelevant, exact realization | 0.008 | -0.000 to +0.000 | -0.018 to +0.040 | -0.000 to +0.000 | +0.000 to +0.000 | 0.00 |
| 1 | irrelevant, crosstalk 0.15 | 0.008 | -0.001 to +0.004 | -0.036 to +0.062 | -0.001 to +0.001 | +0.000 to +0.000 | 0.07 |
| 1 | irrelevant, crosstalk 0.4 | 0.009 | -0.006 to +0.010 | -0.032 to +0.062 | -0.002 to +0.004 | +0.000 to +0.000 | 0.18 |
| 1 | irrelevant, execution noise 5% | 0.008 | -0.000 to +0.000 | -0.018 to +0.040 | -0.000 to +0.000 | +0.000 to +0.000 | 0.03 |
| 1 | irrelevant, edit applied at 10% | 0.008 | -0.000 to +0.000 | -0.018 to +0.040 | -0.000 to +0.000 | +0.000 to +0.000 | 0.90 |
| 1 | irrelevant, MLP readout (loose K) | 0.013 | -0.017 to +0.009 | -0.113 to +0.124 | -0.005 to +0.003 | +0.000 to +0.000 | 0.07 |
| 1 | irrelevant, protected row at cosine 0.97 | 0.025 | -1.233 to +0.361 | -1.341 to +0.380 | -0.075 to +0.122 | +0.000 to +0.000 | 0.97 |
| 1 | irrelevant, uniform scales | 0.008 | -0.000 to +0.000 | -0.000 to +0.000 | -0.000 to +0.000 | +0.000 to +0.000 | 0.00 |
| 2 | relevant, exact realization | 0.026 | +0.336 to +0.361 | +0.334 to +0.400 | +0.150 to +0.163 | +0.000 to +0.000 | 0.00 |
| 2 | relevant, crosstalk 0.15 | 0.027 | +0.334 to +0.395 | +0.333 to +0.416 | +0.150 to +0.172 | +0.000 to +0.000 | 0.07 |
| 2 | relevant, MLP readout (loose K) | 0.035 | +0.372 to +0.503 | +0.361 to +0.533 | +0.161 to +0.198 | +0.000 to +0.000 | 0.07 |
| 2 | relevant, protected row at cosine 0.97 | 0.072 | -0.285 to +1.302 | -0.392 to +1.322 | -0.010 to +0.649 | +0.000 to +0.000 | 0.97 |
| 3 | relevant, edit applied at 2% | 0.026 | +0.336 to +0.361 | +0.334 to +0.400 | +0.150 to +0.163 | +0.000 to +0.000 | 0.98 |
| 3 | weakly relevant, edit applied at 5% | 0.013 | +0.099 to +0.107 | +0.087 to +0.147 | +0.041 to +0.045 | +0.000 to +0.000 | 0.95 |
| 3 | relevant, execution noise 30% | 0.026 | +0.336 to +0.361 | +0.334 to +0.400 | +0.150 to +0.163 | +0.000 to +0.000 | 0.16 |
| 4 | decoder writes an encoder-invisible direction | 0.057 | +0.903 to +0.958 | -0.067 to +0.161 | +0.430 to +0.462 | +0.000 to +0.000 | 0.00 |
| 5 | AND with an invisible partner | 0.065 | -0.000 to +0.000 | -0.862 to +0.101 | +0.204 to +0.221 | +0.443 to +0.476 | 0.00 |
| 6 | insertion into a full TopK code (over-k) | 0.050 | undefined | undefined | n/a | n/a | 0.57 |
| 6 | deletion requiring a support change | 0.050 | +0.282 to +0.617 | +0.294 to +0.627 | n/a | +0.000 to +0.000 | 0.55 |
| 2/3 | relevant, steep MLP, execution noise 30% | 0.025 | -0.022 to +0.531 | -0.041 to +0.638 | +0.004 to +0.213 | +0.000 to +0.000 | 0.17 |

| condition | truth (I2) | P1 applied | P2 gated | P3 direct | P4a bound | P4b sampled K | P5 two lifts |
|---|---|---|---|---|---|---|---|
| C1a_irrelevant_exact | neg | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 |
| C1b_irrelevant_crosstalk | neg | 0.51, **FP 0.35** | 0.28 | 1.00 | 0.00 | 0.00 | 1.00 |
| C1c_irrelevant_crosstalk_hi | neg/non-neg | 0.17, **FP 0.60** | 0.00 | 0.85 | 0.00 | 0.00 | 0.83 |
| C1d_irrelevant_exec_noise | neg | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.96 |
| C1e_irrelevant_under | neg | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.96 |
| C1f_irrelevant_mlp | neg/non-neg | 0.83, **FP 0.20** | 0.28 | 0.92 | 0.00 | 0.00 | 0.72 |
| C1g_irrelevant_coherent | non-neg | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| C1h_irrelevant_uniform_sigma | neg | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C2a_relevant_exact | non-neg | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C2b_relevant_crosstalk | non-neg | 1.00 | 0.28 | 1.00 | 0.98 | 1.00 | 1.00 |
| C2c_relevant_mlp | non-neg | 1.00 | 0.28 | 1.00 | 0.00 | 1.00 | 1.00 |
| C2d_relevant_coherent | non-neg | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| C3a_relevant_under | non-neg | 0.00, **FN 1.00** | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| C3b_small_relevant_under | non-neg | 0.00, **FN 1.00** | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| C3c_relevant_exec_noise | non-neg | 1.00 | 0.00 | 1.00 | 0.00 | 0.88 | 1.00 |
| C4_many_to_one | non-neg | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 |
| C5_interaction | neg | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 |
| C6a_over_k | ill-posed | 0.00 | 0.00 | ill-posed | ill-posed | ill-posed | ill-posed |
| C6b_deletion | non-neg | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| C7_steep_exec_noise | non-neg | 0.91 | 0.00 | 0.85 | 0.00 | 0.00 | 0.85 |
