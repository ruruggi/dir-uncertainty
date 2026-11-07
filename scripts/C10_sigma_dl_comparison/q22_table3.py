# -*- coding: utf-8 -*-
"""
q22_table3.py -- step 3 of the recalculation order (2026-09-28): Table 3 (tab:sigma-compare) rebuilt with
  sigma_h  = body-local heuristic sigma (conventions_check.sigma_body_local), <DATASETS>/_sigma_BL_20260927
  sigma_DL = Smolders L20S3-4 with the three corrections (TRE, input DVF, sigma->mm), ./_out_q22_sigmaDL (Tnew_IC_Snet)
Same calibration as the paper (table3_5fold_both.calibrate_block: fit_c_grid, bin 0.25 mm, 5-fold seed 42).

Controls: (1) sigma_h from the submitted paths reproduces the current Table 3 (f, LECE global, 5-fold; tol 0.011 / 0.01);
(2) sigma_DL reproduces aggregate.json; (3) per case, the TRE of the two arms is identical (same DVF, same landmarks).
Output ./_out_q22_table3/: table3.json, Table3_rows_draft.tex (rows only). Read-only on inputs. Usage (venv): python q22_table3.py
"""
from dirunc_paths import DATASETS, LIB  # data roots, see README
import os, sys, types, json, glob
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = LIB; PS = LIB + "/validation"
sys.path.insert(0, SCRIPTS); sys.path.insert(0, PS)
if "validation" not in sys.modules:
    _v = types.ModuleType("validation"); _v.__path__ = [PS]; sys.modules["validation"] = _v
import numpy as np
from table3_5fold_both import calibrate_block, KNOWN_GLOBAL, TOL
from validation.build_v11_data import ENGINE_PATHS
from validation.calibrate import load_all_landmarks, CaseLandmarks

BL = f"{DATASETS}/_sigma_BL_20260927/Castillo-DIR-LAB"
DL = os.path.join(HERE, "_out_q22_sigmaDL")
OUT = os.path.join(HERE, "_out_q22_table3"); os.makedirs(OUT, exist_ok=True)
ENG = [("SyNRA", "synra"), ("ConvexAdam", "convexadam"), ("uniGradICON", "unigradicon"), ("Elastix", "elastix")]
# current Table 3 (Tables/Table3.tex): f global, 5-fold mean, 5-fold sd for sigma_h and sigma_DL
TEX = {"sigma_h": {"SyNRA": (7.08, 3.09, 1.53), "ConvexAdam": (2.11, 0.22, 0.09), "uniGradICON": (5.07, 1.34, 1.00), "Elastix": (3.60, 0.40, 0.24)},
       "sigma_DL": {"SyNRA": (3.94, 2.88, 1.21), "ConvexAdam": (2.97, 1.50, 0.73), "uniGradICON": (3.27, 1.77, 0.55), "Elastix": (3.04, 1.56, 0.57)}}


def check(tag, got, ref, tol):
    ok = abs(got - ref) <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag:52s} got {got:.4f} ref {ref:.3f}")
    return ok


def dl_cases(key, lab_key):
    out = []
    for f in sorted(glob.glob(os.path.join(DL, f"{lab_key}_case*.npz")), key=lambda p: int(p.split("case")[-1][:-4])):
        r = np.load(f); cid = "case" + f.split("case")[-1][:-4]
        tre = np.asarray(r["tre_new"], np.float64).ravel(); sig = np.asarray(r[key], np.float64).ravel()
        keep = np.isfinite(tre) & np.isfinite(sig) & (sig > 1e-9) & (tre >= 0)
        out.append(CaseLandmarks(case_id=cid, tre_mm=tre[keep], sigma_raw=sig[keep]))
    return out


ok_all, res = True, {}
agg = json.load(open(os.path.join(DL, "aggregate.json")))["results"]
print("Controls")
for lab, k in ENG:
    old = calibrate_block(load_all_landmarks(ENGINE_PATHS[lab]["datasets"]["DIR-LAB"]))
    ok_all &= check(f"sigma_h submitted {lab} LECE global", old["lece_global"], KNOWN_GLOBAL["sigma_h"][lab], TOL)
    ok_all &= check(f"sigma_h submitted {lab} f", old["f_global"], TEX["sigma_h"][lab][0], 0.01)
    ok_all &= check(f"sigma_h submitted {lab} 5-fold mean", old["lece_5fold_mean"], TEX["sigma_h"][lab][1], 0.01)
    hc = load_all_landmarks(os.path.join(BL, f"Results_{k}_procrustes")); dc = dl_cases("sp_C_net", k)
    h = calibrate_block(hc); d = calibrate_block(dc)
    ok_all &= check(f"sigma_DL corrected {lab} = aggregate.json", d["lece_global"], agg[lab]["Tnew_IC_Snet"]["lece_global"], 1e-9)
    dmax = max(float(np.max(np.abs(a.tre_mm - b.tre_mm))) if a.tre_mm.size == b.tre_mm.size else np.inf
               for a, b in zip(sorted(hc, key=lambda c: c.case_id), sorted(dc, key=lambda c: c.case_id)))
    ok_all &= check(f"same TRE in both arms {lab} (max |d| mm)", dmax, 0.0, 1e-3)
    res[lab] = {"sigma_h_submitted": old, "sigma_h_BL": h, "sigma_DL_corrected": d}

print(f"\n  OVERALL CONTROLS: {'PASS' if ok_all else 'FAIL'}\n")
print(f"{'engine':12s} {'sigma':22s} {'f':>6s} {'LECE glob':>10s} {'5-fold':>16s}   (submitted: f, LECE, 5-fold)")
rows = []
for lab, _ in ENG:
    for name, key, sub in (("heuristic-residual sigma_h", "sigma_h_BL", "sigma_h"), ("DL Unsupervised L20S3-4", "sigma_DL_corrected", "sigma_DL")):
        b = res[lab][key]; t = TEX[sub][lab]
        print(f"{lab:12s} {name:26s} {b['f_global']:6.2f} {b['lece_global']:10.3f} {b['lece_5fold_mean']:8.2f} +- {b['lece_5fold_std']:4.2f}"
              f"   ({t[0]:.2f}, {KNOWN_GLOBAL[sub][lab]:.3f}, {t[1]:.2f} +- {t[2]:.2f})")
        sig = r"heuristic-residual $\sigma_{h}$" if sub == "sigma_h" else "DL Unsupervised L20S3-4        "
        rows.append(f"{lab + ' (this work)':23s} & {sig} & {b['f_global']:.2f} & {b['lece_global']:.3f} & "
                    f"$ {b['lece_5fold_mean']:.2f} \\pm {b['lece_5fold_std']:.2f}$ \\\\")
json.dump({"controls_pass": bool(ok_all), "results": res}, open(os.path.join(OUT, "table3.json"), "w"), indent=2)
open(os.path.join(OUT, "Table3_rows_draft.tex"), "w", encoding="utf-8").write("\n".join(rows) + "\n")
print(f"\n-> {OUT}")
