"""q22_rho_sigmaDL_dirlab.py -- 2026-09-30, read-only check for the Discussion ("Complementarity"): Spearman rho(sigma_p, TRE) of the
Smolders DL sigma (corrected wiring, training resolution: the per-landmark sigma of Table 3) on DIR-LAB Castillo, per engine, pooled
and per-case mean +- sd, side by side with sigma_h (body-local, Table 4 v2). Same per-landmark data as q22_table23_v2.py (dl_cases,
load_all_landmarks); Spearman as in Table 4 (pooled; per-case mean, ddof=0). Output: ./_out_q22_rho_sigmaDL/rho_sigmaDL_dirlab.json
Usage (venv): python q22_rho_sigmaDL_dirlab.py
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from scipy.stats import spearmanr
import q22_table23_v2 as T

def get(c):
    for a in ("sigma_raw", "sigma_mm", "sigma"):
        if hasattr(c, a): return np.asarray(getattr(c, a), float)
    raise AttributeError(dir(c))

out = {}
for eng, key in T.ENG_KEY.items():
    dl = sorted(T.dl_cases(key), key=T.natkey)
    h = sorted(T.load_all_landmarks(T.bl(T.ENGINE_PATHS[eng]["datasets"]["DIR-LAB"])), key=T.natkey)
    row = {}
    for arm, cs in (("sigma_DL", dl), ("sigma_h", h)):
        tre = np.concatenate([c.tre_mm for c in cs]); sg = np.concatenate([get(c) for c in cs])
        per = [spearmanr(get(c), c.tre_mm)[0] for c in cs]
        row[arm] = {"n_cases": len(cs), "n_landmarks": int(tre.size), "rho_pool": float(spearmanr(sg, tre)[0]),
                    "rho_case_mean": float(np.mean(per)), "rho_case_sd": float(np.std(per))}
    out[eng] = row
    print(f"{eng:12s} sigma_DL rho_pool {row['sigma_DL']['rho_pool']:+.3f} per-case {row['sigma_DL']['rho_case_mean']:+.3f}+-{row['sigma_DL']['rho_case_sd']:.3f} (n {row['sigma_DL']['n_landmarks']}) | "
          f"sigma_h rho_pool {row['sigma_h']['rho_pool']:+.3f} per-case {row['sigma_h']['rho_case_mean']:+.3f}+-{row['sigma_h']['rho_case_sd']:.3f} (n {row['sigma_h']['n_landmarks']})")
od = os.path.join(HERE, "_out_q22_rho_sigmaDL"); os.makedirs(od, exist_ok=True)
json.dump(out, open(os.path.join(od, "rho_sigmaDL_dirlab.json"), "w"), indent=2)
