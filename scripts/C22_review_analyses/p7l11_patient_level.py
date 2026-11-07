# -*- coding: utf-8 -*-
"""P7L11 (2026-10-03): what the per-patient (body-local) min-max normalisation does BETWEEN patients.
Same cells and inputs as q22_table23_v2.py (Table 2, q <= 10 mm; sigma_BL of 27/9, unscaled: f is common to the cell so it
does not change ranks or CVs). Per case: mean sigma_raw at the landmarks, RMS TRE. Per cell: CV across cases of each, and the
Spearman rank correlation between them (does a patient with larger error get a larger sigma?). Read-only. Output ./_out_p7l11/."""
from dirunc_paths import DATASETS, LIB  # data roots, see README
import os, sys, types, json, re
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = LIB; PS = LIB + "/validation"
sys.path.insert(0, SCRIPTS); sys.path.insert(0, PS)
if "validation" not in sys.modules:
    _v = types.ModuleType("validation"); _v.__path__ = [PS]; sys.modules["validation"] = _v
import numpy as np
from scipy.stats import spearmanr
from validation.build_v11_data import ENGINE_PATHS
from validation.calibrate import load_all_landmarks

DS_ROOT = f"{DATASETS}"; BL_ROOT = f"{DS_ROOT}/_sigma_BL_20260927"
OUT = os.path.join(HERE, "_out_p7l11"); os.makedirs(OUT, exist_ok=True)
DATASETS = ["Lung-DIR-QA", "Abdominal-DIR-QA", "Liver-DIR-QA", "DIR-LAB"]
ENGINES = ["SyNRA", "ConvexAdam", "uniGradICON", "Elastix"]
bl = lambda p: os.path.join(BL_ROOT, os.path.relpath(p, DS_ROOT)).replace("\\", "/")
cv = lambda x: float(np.std(x) / np.mean(x))

res = {}
for eng in ENGINES:
    for ds in DATASETS:
        cases = load_all_landmarks(bl(ENGINE_PATHS[eng]["datasets"][ds]))
        rows = [(c.case_id, float(np.mean(c.sigma_raw)), float(np.sqrt(np.mean(c.tre_mm ** 2)))) for c in cases]
        rows = [r for r in rows if r[2] <= 10.0]
        if len(rows) < 3:
            res[f"{ds}|{eng}"] = {"n": len(rows)}; continue
        s = np.array([r[1] for r in rows]); t = np.array([r[2] for r in rows])
        rho, p = spearmanr(s, t)
        res[f"{ds}|{eng}"] = {"n": len(rows), "cv_sigma_cases": cv(s), "cv_tre_cases": cv(t), "rho_cases": float(rho), "p": float(p)}
        print(f"{ds:17s} {eng:12s} n={len(rows):2d}  CV(sigma) {cv(s):.2f}  CV(RMS TRE) {cv(t):.2f}  rho {rho:+.2f} (p {p:.2f})", flush=True)

json.dump(res, open(os.path.join(OUT, "patient_level.json"), "w"), indent=1)
V = [v for v in res.values() if "rho_cases" in v]
for k in ("cv_sigma_cases", "cv_tre_cases", "rho_cases"):
    a = np.array([v[k] for v in V]); print(f"{k:15s} min {a.min():+.2f} median {np.median(a):+.2f} max {a.max():+.2f}  ({len(a)} cells)")
print("cells with p < 0.05:", sum(v["p"] < 0.05 for v in V), "/", len(V), "; rho > 0:", sum(v["rho_cases"] > 0 for v in V))
