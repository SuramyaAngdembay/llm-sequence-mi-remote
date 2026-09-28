#!/usr/bin/env python3
"""Coordinates of thesis Figure 3.1: the user-level ROC curve of the PCA detector (2026-09-28).

Reads the user scores saved by pca_reconstruction_r42.py, keeps the corners of the step curve, checks that the
area under them equals the saved user AUC, and prints a pgfplots coordinate list and the marked threshold.

  python scripts/thesis_roc_corners.py results/pca_reconstruction_r42_2026_09_28/pca_reconstruction_r42.json
"""
import json
import sys

import numpy as np

VARIANT = "all features, varying in training"
POOL = "language-model pool (60 insiders)"

d = json.load(open(sys.argv[1])); U = d["users"]; s = d["variants"][VARIANT]["user_max_error"]
mal = np.array([s[u] for u in U["lm_pool_insiders"]]); ben = np.array([s[u] for u in U["heldout_benign"]])
assert len(np.unique(np.concatenate([mal, ben]))) == len(mal) + len(ben), "tied scores: the step construction assumes none"
pts = [(0.0, 0.0)] + [((ben >= t).mean(), (mal >= t).mean()) for t in np.sort(np.concatenate([mal, ben]))[::-1]]
corners = [pts[0]] + [b for a, b, c in zip(pts, pts[1:], pts[2:]) if not (a[0] == b[0] == c[0] or a[1] == b[1] == c[1])] + [pts[-1]]
area = sum((x1 - x0) * y0 for (x0, y0), (x1, _) in zip(corners, corners[1:]))
assert abs(area - d["variants"][VARIANT][POOL]["user_auc"]) < 1e-9
half = next(p for p in pts if p[1] >= 0.5)
print(f"% {len(corners)} corners, area {area:.4f}; half the insiders: {round(half[1] * len(mal))} of {len(mal)}, "
      f"{round(half[0] * len(ben))} of {len(ben)} benign at ({half[0]:.4f},{half[1]:.4f})")
print(" ".join(f"({x:.4f},{y:.4f})" for x, y in corners))
