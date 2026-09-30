#!/usr/bin/env python3
"""Read trusted local research bundles and audit their saved fold memberships.

No inference, fitting, or writes to the experiment directories. The official
insider list is supplied independently of the historical day-label join.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--insiders", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    insiders = json.loads(args.insiders.read_text())
    result = {"scope": "actual saved local-AE bundles; not a sequence-model refit",
              "insiders_sha256": hashlib.sha256(args.insiders.read_bytes()).hexdigest(),
              "datasets": {}}
    for dataset in ("r42", "r62"):
        official = set(insiders[dataset])
        root = args.root / f"results_{dataset}_session_lcdal_autoencoder_mech_clean"
        rows = []
        for path in sorted(root.glob("*/fold*/model_bundle.pt")):
            bundle = torch.load(path, map_location="cpu", weights_only=False)
            fold = bundle["fold"]
            train, test = set(fold["oneclass_train_users"]), set(fold["test_users"])
            heldout = fold["heldout_pos_user"]
            rows.append({
                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "fold": fold, "method": bundle["method"], "args": bundle["args"],
                "train_official_insiders": sorted(train & official),
                "test_other_official_insiders": sorted((test & official) - {heldout}),
                "train_test_overlap": sorted(train & test),
                "scaler_feature_count": len(bundle["scaler_mean"]),
                "saved_test_latents": (path.parent / "test_latents.parquet").exists(),
                "safe_original_training_membership": not bool(train & official),
            })
        if not rows:
            raise RuntimeError(f"No bundles found for {dataset} at {root}")
        result["datasets"][dataset] = {
            "bundles": len(rows), "folds": len({x["fold"]["fold"] for x in rows}),
            "contaminated_bundles": sum(bool(x["train_official_insiders"]) for x in rows),
            "test_contaminated_bundles": sum(bool(x["test_other_official_insiders"]) for x in rows),
            "rows": rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {a: b for a, b in v.items() if a != "rows"}
                      for k, v in result["datasets"].items()}))


if __name__ == "__main__":
    main()
