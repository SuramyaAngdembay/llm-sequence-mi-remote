# Success Criteria

This branch is not being judged by raw detector quality alone.

## Current Local Baseline To Beat

Primary local MI baseline:

- base LC-DAL session autoencoder branch

What it already achieves:

- strong detector quality
- strongest family-level mechanism around `usb_activity`
- adaptive repair that beats residual baseline at family level

## Minimum Scientific Return Needed

The remote LLM branch is worthwhile only if it gives at least one of:

1. cleaner sparse units than session SAE
2. stronger sparse repair/patchability than session SAE
3. a tighter necessary-and-sufficient circuit than session family-level `usb_activity`
4. a more convincing representation-first story for sequence-native MI

## Failure Modes

This branch is not a success if it only gives:

- slightly better detection
- prettier activations
- sparse features without patchability
- distributed features that are no cleaner than current session AE

## Main Comparison Questions

1. Does delta-SAE beat current session SAE on top-vs-control ablation?
2. Do grounded delta features patch/repair better than current session SAE sparse units?
3. Does the LLM sequence branch produce a cleaner localized mechanism than family-level `usb_activity`?
4. If not, does it at least justify a broader claim about sequence-native distributed mechanisms?

