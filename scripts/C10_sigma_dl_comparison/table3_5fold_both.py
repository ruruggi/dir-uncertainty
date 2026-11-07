"""
table3_5fold_both.py — Table 3 (tab:sigma-compare) fairness column.

Reviewer concern: the global-f LECE in Table 3 is optimistic because the same
DIR-LAB landmarks are used to BOTH tune the calibration factor f and to
evaluate LECE. This script adds an honest 5-fold cross-validated LECE (fit f on
4/5 of the CASES, evaluate LECE on the held-out 1/5; mean +/- s.d. over the 5
folds) alongside the existing global-f LECE, for BOTH sigma predictors:

  - sigma_h  : heuristic+residual (this work). Per-landmark (TRE, sigma_raw)
               loaded from the per-case landmarks.npz produced by the DIR
               pipeline, via the canonical ENGINE_PATHS / load_all_landmarks.
  - sigma_DL : Smolders 2023 DL Unsupervised, model L20S3-4. Per-landmark
               (TRE, sigma_raw) are NOT persisted anywhere (the existing
               _castillo_smolders_dl_test/{aggregate,per_case}.json store only
               per-case summaries), so this script RE-RUNS the Smolders DL
               inference per (engine, case) reusing run_one() from
               smolders_dl_sigma_castillo.py to recover the per-landmark arrays
               needed to re-fit f per fold.

All 10 DIR-LAB Castillo cases, all 4 engines (SyNRA, ConvexAdam, uniGradICON,
Elastix). Calibration: fit_c_grid (log-spaced grid + refinement, minimises
LECE), bin_width=0.25 mm, KFold n_splits=5, shuffle, seed=42 — exactly as the
paper / validation.calibrate.

MANDATORY SELF-CHECK: the GLOBAL-f LECE computed here must reproduce the current
Table 3 values within ~0.01 mm. If not, the data wiring is wrong and the script
aborts before reporting any 5-fold numbers.

Output: _castillo_calibration/table3_5fold_both.json
"""
from __future__ import annotations

import os
import sys
import json
import tempfile
import shutil

import numpy as np

# UTF-8 console (Greek letters in diagnostics); fall back to file if it fails.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PAPER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_ROOT = os.path.normpath(os.path.join(PAPER_ROOT, "..", "DIR_Uncertainty_v2"))
sys.path.insert(0, CODE_ROOT)
sys.path.insert(0, os.path.join(CODE_ROOT, "dl_arch"))
sys.path.insert(0, PAPER_ROOT)

from validation.build_v11_data import ENGINE_PATHS
from validation.calibrate import (
    CaseLandmarks, load_all_landmarks, fit_c_grid, stack_cases, kfold_calibrate,
)
from validation.lece import compute_lece

BIN = 0.25
SEED = 42
N_SPLITS = 5

# Known current Table 3 GLOBAL-f LECE values (mm) for the self-check.
KNOWN_GLOBAL = {
    "sigma_h":  {"SyNRA": 0.402, "ConvexAdam": 0.126, "uniGradICON": 0.176, "Elastix": 0.188},
    "sigma_DL": {"SyNRA": 0.464, "ConvexAdam": 0.420, "uniGradICON": 0.710, "Elastix": 0.554},
}
TOL = 0.011  # ~0.01 mm tolerance for the self-check

# Engine label <-> Smolders engine_key map (smolders_dl_sigma_castillo.ENGINES)
ENGINE_KEYS = {
    "SyNRA": "synra",
    "ConvexAdam": "convexadam",
    "uniGradICON": "unigradicon",
    "Elastix": "elastix",
}


def calibrate_block(cases):
    """Return f_global, lece_global, lece_5fold_mean, lece_5fold_std, n_cases.

    Uses the canonical fit_c_grid (global) and kfold_calibrate (5-fold,
    seed=42, bin 0.25) so the numbers are bit-identical with the paper.
    """
    if not cases:
        return None
    tre, sig = stack_cases(cases)
    f_global, _ = fit_c_grid(tre, sig, bin_width=BIN)
    lece_global, _ = compute_lece(tre, f_global * sig, bin_width=BIN)

    n = len(cases)
    if n < N_SPLITS:
        kf_mean = float("nan")
        kf_std = float("nan")
        folds_lece = []
    else:
        folds = kfold_calibrate(cases, n_splits=N_SPLITS, seed=SEED, bin_width=BIN)
        folds_lece = [float(fl.lece_test_grid) for fl in folds]
        cv = np.array(folds_lece)
        kf_mean = float(cv.mean())
        kf_std = float(cv.std(ddof=1))
    return {
        "f_global": float(f_global),
        "lece_global": float(lece_global),
        "lece_5fold_mean": kf_mean,
        "lece_5fold_std": kf_std,
        "lece_5fold_per_fold": folds_lece,
        "n_cases": int(n),
        "n_landmarks": int(tre.size),
    }


def load_sigma_h_cases(engine_label):
    """Per-engine sigma_h CaseLandmarks for the 10 DIR-LAB Castillo cases."""
    path = ENGINE_PATHS[engine_label]["datasets"]["DIR-LAB"]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"sigma_h DIR-LAB path missing: {path}")
    return load_all_landmarks(path)  # full set, no LIM filter (Table 3 = all 10)


def load_sigma_dl_cases(engine_label, registerer, tmp_root):
    """Per-engine sigma_DL CaseLandmarks: RE-RUN Smolders DL inference per case
    to recover per-landmark (TRE, sigma_raw) arrays.

    Reuses run_one() / discover_cases from smolders_dl_sigma_castillo so the
    wiring (Procrustes pre-alignment, DVF loading, sigma sampling, TRE via DVF)
    is identical to the script that produced the published global LECE.
    """
    import smolders_dl_sigma_castillo as sm
    engine_key = ENGINE_KEYS[engine_label]
    cases_disc = sorted(sm.discover_cases(sm.CASTILLO_NIFTI),
                        key=lambda c: int(c[0].replace("case", "")))
    case_ids = [c[0] for c in cases_disc]
    out = []
    for cid in case_ids:
        tre, sigma_p, _ = sm.run_one(engine_key, cid, registerer, tmp_root)
        tre = np.asarray(tre, dtype=np.float64).ravel()
        sig = np.asarray(sigma_p, dtype=np.float64).ravel()
        # Same usable-landmark filter as load_all_landmarks (sigma>0, finite).
        keep = np.isfinite(tre) & np.isfinite(sig) & (sig > 1e-9) & (tre >= 0)
        out.append(CaseLandmarks(case_id=cid, tre_mm=tre[keep], sigma_raw=sig[keep]))
    return out


def main():
    out_dir = os.path.join(PAPER_ROOT, "_castillo_calibration")
    os.makedirs(out_dir, exist_ok=True)
    diag_lines = []

    def log(msg=""):
        print(msg)
        diag_lines.append(str(msg))

    log("=" * 78)
    log("Table 3 — global-f vs 5-fold CV LECE, sigma_h and sigma_DL, 4 engines")
    log(f"  bin_width={BIN} mm | KFold n_splits={N_SPLITS} shuffle seed={SEED}")
    log(f"  out_dir: {out_dir}")
    log("=" * 78)

    results = {}

    # ---- sigma_h (fast, from landmarks.npz) ----------------------------------
    log("\n[1/2] sigma_h (heuristic+residual) — loading per-case landmarks.npz")
    for eng in ["SyNRA", "ConvexAdam", "uniGradICON", "Elastix"]:
        cases = load_sigma_h_cases(eng)
        blk = calibrate_block(cases)
        results.setdefault(eng, {})["sigma_h"] = blk
        log(f"  {eng:<13} n_cases={blk['n_cases']:>2} n_lm={blk['n_landmarks']:>5} "
            f"f={blk['f_global']:.3f} LECE_global={blk['lece_global']:.4f} "
            f"5fold={blk['lece_5fold_mean']:.4f}+/-{blk['lece_5fold_std']:.4f}")

    # ---- sigma_DL (re-run Smolders DL inference) -----------------------------
    log("\n[2/2] sigma_DL (Smolders DL Unsupervised L20S3-4) — RE-RUNNING inference")
    log("       (per-landmark arrays not persisted; recomputed from DVF + model)")
    import smolders_dl_sigma_castillo as sm
    from dl.model_loader import load_registerer
    log(f"       loading registerer TrainedModels/{sm.MODEL_NAME} ...")
    registerer = load_registerer(model_name=sm.MODEL_NAME)
    tmp_root = tempfile.mkdtemp(prefix="t3_5fold_")
    try:
        for eng in ["SyNRA", "ConvexAdam", "uniGradICON", "Elastix"]:
            log(f"  --- {eng} ---")
            cases = load_sigma_dl_cases(eng, registerer, tmp_root)
            blk = calibrate_block(cases)
            results.setdefault(eng, {})["sigma_DL"] = blk
            log(f"  {eng:<13} n_cases={blk['n_cases']:>2} n_lm={blk['n_landmarks']:>5} "
                f"f={blk['f_global']:.3f} LECE_global={blk['lece_global']:.4f} "
                f"5fold={blk['lece_5fold_mean']:.4f}+/-{blk['lece_5fold_std']:.4f}")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # ---- MANDATORY SELF-CHECK ------------------------------------------------
    log("\n" + "=" * 78)
    log("SELF-CHECK: global-f LECE vs known Table 3 values (tol ~0.01 mm)")
    log("=" * 78)
    all_pass = True
    checks = []
    for pred in ("sigma_h", "sigma_DL"):
        for eng, known in KNOWN_GLOBAL[pred].items():
            got = results[eng][pred]["lece_global"]
            ok = abs(got - known) <= TOL
            all_pass = all_pass and ok
            tag = "PASS" if ok else "FAIL"
            log(f"  [{tag}] {pred:<8} {eng:<13} got={got:.4f}  known={known:.3f}  "
                f"d={got - known:+.4f}")
            checks.append({"predictor": pred, "engine": eng, "got": got,
                           "known": known, "diff": got - known, "pass": ok})

    verdict = "PASS" if all_pass else "FAIL"
    log(f"\n  OVERALL SELF-CHECK: {verdict}")

    payload = {
        "description": "Table 3 fairness column: global-f vs 5-fold CV LECE, "
                       "sigma_h and sigma_DL, 4 DIR engines, 10 DIR-LAB Castillo cases.",
        "bin_width_mm": BIN,
        "kfold": {"n_splits": N_SPLITS, "shuffle": True, "seed": SEED},
        "self_check": {"overall": verdict, "tolerance_mm": TOL, "checks": checks},
        "results": results,
    }
    json_path = os.path.join(out_dir, "table3_5fold_both.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log(f"\n  results -> {json_path}")

    # Always dump a UTF-8 diagnostics file (robust to console encoding issues).
    diag_path = os.path.join(out_dir, "table3_5fold_both_diag.txt")
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write("\n".join(diag_lines))

    if not all_pass:
        log("\n*** SELF-CHECK FAILED — data wiring is wrong. Do NOT trust the "
            "5-fold numbers above. Diagnose before reporting. ***")
        sys.exit(2)


if __name__ == "__main__":
    main()
