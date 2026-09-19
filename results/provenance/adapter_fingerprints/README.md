# Adapter provenance baselines

The self-audit against arXiv:2509.08713 found that work package 3's results were
produced with adapters whose identity to the Anvil adapters was **assumed**. We
had recorded a safetensors **file** md5, which is the wrong instrument: two files
can hold identical weights and differ in header ordering or metadata, and a
one-byte difference tells you nothing about which tensor moved.

These are content fingerprints — per-tensor sha256 over raw bytes, plus a
combined digest, plus a separate digest over the configuration fields that
change what the weights mean.

| adapter | weights digest | config digest | tensors |
|---|---|---|---|
| Aquaman `cert-data/r62_adapter` | `5a6c2ffa1ffa8ea6…` | see JSON | 504 |
| Aquaman `cert-data/r42_adapter` | `49bbbd4b0bf0cfe7…` | see JSON | 504 |

The tool was checked before these were trusted: identical on a repeated run,
**differing on all 504 shared tensors** between r4.2 and r6.2, and able to
separate a configuration-only difference from a weight difference.

## The open question this exists to settle

When Anvil returns, run the same command against
`checkpoints/qwen3_8b_session_qlora_r42_ddp_mb22_gc_on/adapter` and the r6.2
equivalent, then:

```
python3 scripts/adapter_fingerprint.py COMPARE \
  results/provenance/adapter_fingerprints/aquaman_r42.json anvil_r42.json
```

Exit status 0 means the package 3 numbers were produced on the published
adapter. **Until that runs, package 3's provenance is "assumed", not
"verified", and must be labelled that way anywhere it is cited.**
