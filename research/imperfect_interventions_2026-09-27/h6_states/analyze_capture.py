#!/usr/bin/env python3
"""Part 2 analysis of the captured H6 states (CPU only; no behavioural outcomes).

  python3 analyze_capture.py CAPTURE_DIR OUT_JSON

Target definitions (per aligned session token, codes from arithmetic path A unless stated):
  full-code target     tau = z_R with the selected coordinates replaced by z_O,S; every other coordinate held.
                       supp(tau) = supp(z_R) - supp_S(z_R) + supp_S(z_O), built directly from the full TopK codes.
  selected-only target only the selected coordinates are specified; the others are free. Its specified support is
                       supp_S(z_O), which never exceeds k, so over-k cannot arise; encoder feasibility is a separate question.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
meta = json.loads((d / "h6_capture_meta.json").read_text())
recv = json.loads((d / "h6_capture_receivers.json").read_text())
t = np.load(d / "h6_capture_tokens.npz")
S = meta["selected"]; k = meta["k"]
n = len(t["recv"])


def supp(idx, val):
    return {int(i) for i, v in zip(idx, val) if v > 0}


def code_dict(idx, val):
    return {int(i): float(v) for i, v in zip(idx, val) if v > 0}


def qs(x, ps=(0.5, 0.9, 0.99, 1.0)):
    x = np.asarray(x, dtype=float)
    return {f"q{int(p * 100)}": float(np.quantile(x, p)) for p in ps} if x.size else None


res = {"source": str(d), "gpu": meta.get("gpu"), "sae_sha256": meta["sae_sha256"], "k": k, "sae_width": meta["sae_width"],
       "saved_code_check_vs_h6": meta["saved_code_check_vs_h6"], "session_tokens": n}
edited = t["edited_sel"].astype(bool)
res["edited_selected_tokens"] = int(edited.sum())

# ---- supports and targets, both arithmetic paths
for path in ("A", "B"):
    rows = []
    for j in np.flatnonzero(edited):
        zR = code_dict(t[f"R_{path}_idx"][j], t[f"R_{path}_val"][j]); zO = code_dict(t[f"O_{path}_idx"][j], t[f"O_{path}_val"][j])
        sR, sO = set(zR), set(zO)
        selR, selO = sR & set(S), sO & set(S)
        target = (sR - selR) | selO
        req = np.array([zO.get(f, 0.0) - zR.get(f, 0.0) for f in S])
        rows.append({"tok": int(j), "recv": int(t["recv"][j]), "n_active_R": len(sR), "n_active_O": len(sO), "sel_R": len(selR), "sel_O": len(selO),
                     "target_size": len(target), "formula_size": len(sR) - len(selR) + len(selO), "req_sq": float((req ** 2).sum()),
                     "involves_7693": bool(abs(req[S.index(7693)]) > 0) if 7693 in S else False})
    assert all(r["target_size"] == r["formula_size"] for r in rows)
    tot = sum(r["req_sq"] for r in rows)
    over = [r for r in rows if r["target_size"] > k]
    users = {r["recv"]: recv[r["recv"]]["user"] for r in rows}
    kinds = {r["recv"]: recv[r["recv"]]["kind"] for r in rows}
    res[f"path_{path}"] = {
        "edited_tokens_with_nonzero_request": sum(1 for r in rows if r["req_sq"] > 0),
        "swapped_active_count_distribution_on_edited_tokens": dict(sorted(Counter(r["n_active_R"] for r in rows).items())),
        "original_active_count_distribution_on_edited_tokens": dict(sorted(Counter(r["n_active_O"] for r in rows).items())),
        "selected_support_change_distribution": dict(sorted(Counter(r["sel_O"] - r["sel_R"] for r in rows).items())),
        "full_code_target_size_distribution": dict(sorted(Counter(r["target_size"] for r in rows).items())),
        "over_k_tokens": len(over), "over_k_share_of_edited_tokens": len(over) / len(rows),
        "over_k_receivers": len({r["recv"] for r in over}), "over_k_users": len({users[r["recv"]] for r in over}),
        "over_k_by_kind_tokens": dict(Counter(kinds[r["recv"]] for r in over)),
        "over_k_share_of_requested_squared_change": sum(r["req_sq"] for r in over) / tot,
        "over_k_tokens_involving_7693": sum(1 for r in over if r["involves_7693"]),
        "edited_receivers": len({r["recv"] for r in rows}), "edited_users": len(set(users.values())),
        "selected_only_target_over_k_tokens": sum(1 for r in rows if r["sel_O"] > k),
        "under_k_full_code_targets": sum(1 for r in rows if r["target_size"] < k),
    }
    if path == "A":
        res["_rows_A"] = rows

# ---- arithmetic paths on the same states
def path_compare(prefix, mask):
    same, rel = [], []
    for j in np.flatnonzero(mask):
        a = code_dict(t[f"{prefix}_A_idx"][j], t[f"{prefix}_A_val"][j]); b = code_dict(t[f"{prefix}_B_idx"][j], t[f"{prefix}_B_val"][j])
        same.append(set(a) == set(b))
        keys = set(a) | set(b)
        diff = np.sqrt(sum((a.get(q, 0.0) - b.get(q, 0.0)) ** 2 for q in keys)); na = np.sqrt(sum(v * v for v in a.values()))
        rel.append(diff / na if na > 0 else 0.0)
    return {"tokens": int(mask.sum()), "support_identical_share": float(np.mean(same)) if same else None,
            "support_changed_tokens": int(len(same) - sum(same)), "rel_code_diff": qs(rel)}


allm = np.ones(n, dtype=bool)
res["arithmetic_A_vs_B"] = {"R_all_session": path_compare("R", allm), "R_edited": path_compare("R", edited),
                            "O_all_session": path_compare("O", allm), "O_edited": path_compare("O", edited),
                            "patched_edited": path_compare("P", edited)}
ra = {r["tok"]: r["target_size"] > k for r in res["_rows_A"]}
rb = {}
for j in np.flatnonzero(edited):
    zR = code_dict(t["R_B_idx"][j], t["R_B_val"][j]); zO = code_dict(t["O_B_idx"][j], t["O_B_val"][j])
    rb[int(j)] = (len(set(zR) - (set(zR) & set(S)) | (set(zO) & set(S))) > k)
res["arithmetic_A_vs_B"]["full_code_over_k_classification_disagreements"] = sum(ra[j] != rb[j] for j in ra)

# ---- zero-edit and vector execution error
res["zero_edit"] = {"identity_hook_equal_all_receivers": all(r["zero_identity_equal"] for r in recv),
                    "add_zero_equal_all_receivers": all(r["zero_addzero_equal"] for r in recv),
                    "add_zero_max_abs": max(r["zero_addzero_max_abs"] for r in recv)}
res["vector_execution_error_edited"] = {"rel_norm_err": qs(t["exec_rel"][edited]), "abs_norm_err": qs(t["exec_err"][edited]),
                                        "cosine_min": float(t["exec_cos"][edited].min()), "cosine_q01": float(np.quantile(t["exec_cos"][edited], 0.01)),
                                        "applied_equals_bf16_sum_share": float(t["exec_bf16_exact"][edited].mean()),
                                        "intended_edit_norm": qs(t["e_sel_norm"][edited])}

# ---- realization of the applied edit (target from path A, as in H6's verification), both paths
real = {}
for path in ("A", "B"):
    errs, errs_over, errs_rest, num, den, inserted_missing = [], [], [], 0.0, 0.0, 0
    over_set = {r["tok"] for r in res["_rows_A"] if r["target_size"] > k}
    for j in np.flatnonzero(edited):
        zR = code_dict(t["R_A_idx"][j], t["R_A_val"][j]); zO = code_dict(t["O_A_idx"][j], t["O_A_val"][j])
        zP = code_dict(t[f"P_{path}_idx"][j], t[f"P_{path}_val"][j])
        req = np.array([zO.get(f, 0.0) - zR.get(f, 0.0) for f in S])
        err = np.array([zP.get(f, 0.0) - zO.get(f, 0.0) for f in S])
        if np.linalg.norm(req) == 0:
            continue
        e_ = float(np.linalg.norm(err) / np.linalg.norm(req)); errs.append(e_)
        (errs_over if int(j) in over_set else errs_rest).append(e_)
        num += float(np.linalg.norm(err)); den += float(np.linalg.norm(req))
        if int(j) in over_set:
            newly = [f for f in S if zO.get(f, 0) > 0 and zR.get(f, 0) == 0]
            inserted_missing += all(zP.get(f, 0) == 0 for f in newly)
    real[f"realized_path_{path}"] = {"normalized_target_err": qs(errs), "mass_weighted_target_err": num / den,
                                     "over_k_tokens_normalized_err": qs(errs_over), "other_tokens_normalized_err": qs(errs_rest),
                                     "over_k_tokens_where_every_newly_requested_feature_stayed_inactive": inserted_missing}
res["realization_of_applied_edit"] = real
res.pop("_rows_A")
Path(sys.argv[2]).write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1))
