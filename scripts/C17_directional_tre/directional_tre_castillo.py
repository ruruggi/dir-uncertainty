"""
directional_tre_castillo.py -- R1.6 / Limitations (i): per-axis decomposition of the post-DIR landmark TRE on DIR-LAB Castillo.

Corrected copy (2026-09-30) of Scripts/_paper_scripts/directional_tre_castillo.py (submitted copy, left untouched).
Same computation and same numbers; changes:
  - validation fixed (audit C3, 2026-09-29): Table 1 reports the MEAN OF THE PER-CASE RMS TRE, not the RMS over the pooled
    landmarks, so the check against Table 1 now uses the per-case mean (the pooled RMS is reported separately);
  - self-checks added: residual norm == tre_mm for every landmark; world axis 2 tracks the slice axis (cranio-caudal) of the
    fixed CT; raw inhale-exhale motion is largest along that axis (respiratory motion);
  - output written next to this script (./_out_C17/directional_tre.json) instead of the development tree.
Input: the canonical per-landmark prediction saved by run_pair.py in Results_<engine>_procrustes/case*/landmarks.npz
(keys lm_fixed_voxel, lm_fixed_world, lm_moving_world, lm_moving_world_pred, tre_mm, spacing_fixed), i.e. the residual behind
Table 1 (no DVF re-interpolation). All 10 cases pooled, as in Table 1.
Verified 2026-09-30: CC/lateral 5.23 / 2.01 / 3.35 / 2.60 (SyNRA / ConvexAdam / uniGradICON / Elastix); SyNRA CC 5.63, lateral 1.08 mm.

Usage (venv):  python directional_tre_castillo.py
"""
from __future__ import annotations
from dirunc_paths import DATASETS  # data roots, see README
import os, sys, json
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

CASTILLO_BASE = f"{DATASETS}/Castillo-DIR-LAB"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out_C17")
ENGINES = [("synra", "SyNRA"), ("convexadam", "ConvexAdam"),
           ("unigradicon", "uniGradICON"), ("elastix", "Elastix")]
CASES = range(1, 11)
TABLE1 = {"SyNRA": 4.90, "ConvexAdam": 1.70, "uniGradICON": 3.39, "Elastix": 2.93}   # mean of per-case RMS TRE (mm)


def load(engine_key, c):
    return np.load(os.path.join(CASTILLO_BASE, f"Results_{engine_key}_procrustes", f"case{c}", "landmarks.npz"))


def check_axes(engine_key):
    """World axis 2 = slice axis of the fixed CT (cranio-caudal); raw respiratory motion largest along it."""
    raw = []
    for c in CASES:
        z = load(engine_key, c)
        sp = np.asarray(z["spacing_fixed"], float)
        assert int(np.argmax(sp)) == 2, f"case {c}: slice axis is not axis 2 (spacing {sp})"
        v = np.asarray(z["lm_fixed_voxel"], float)[:, 2] * sp[2]
        w = np.asarray(z["lm_fixed_world"], float)[:, 2]
        assert abs(np.corrcoef(v, w)[0, 1]) > 0.99, f"case {c}: world z does not track the slice axis"
        raw.append(np.asarray(z["lm_moving_world"], float) - np.asarray(z["lm_fixed_world"], float))
    raw_rms = np.sqrt(np.mean(np.concatenate(raw) ** 2, axis=0))
    assert int(np.argmax(raw_rms)) == 2, f"raw motion not cranio-caudally dominated: {raw_rms}"
    return raw_rms


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    raw_rms = check_axes("synra")
    print(f"axes PASS: raw inhale-exhale motion RMS (x, y, z) = {raw_rms.round(2)} mm")
    print(f"  {'engine':<13} {'lat(x)':>8} {'AP(y)':>8} {'CC(z)':>8} {'pooled':>8} {'case-mean':>10} {'Tab1':>6}  {'CC/lat':>7} {'CC/AP':>7}")
    rows = []
    for engine_key, label in ENGINES:
        resid, per_case = [], []
        for c in CASES:
            z = load(engine_key, c)
            r = np.asarray(z["lm_moving_world_pred"], np.float64) - np.asarray(z["lm_moving_world"], np.float64)
            tre = np.asarray(z["tre_mm"], np.float64).ravel()
            assert np.allclose(np.linalg.norm(r, axis=1), tre, atol=1e-3), f"{label} case {c}: residual norm != tre_mm"
            resid.append(r); per_case.append(float(np.sqrt(np.mean(tre ** 2))))
        r = np.concatenate(resid, axis=0)              # (N,3) world mm
        rms = np.sqrt(np.mean(r ** 2, axis=0))
        pooled = float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))
        case_mean = float(np.mean(per_case))
        assert abs(case_mean - TABLE1[label]) < 0.02, f"{label}: mean per-case RMS {case_mean:.3f} != Table 1 {TABLE1[label]}"
        row = {"engine": label, "n_landmarks": int(r.shape[0]),
               "rms_lateral_x_mm": float(rms[0]), "rms_AP_y_mm": float(rms[1]), "rms_CC_z_mm": float(rms[2]),
               "rms_pooled_mm": pooled, "rms_per_case_mm": per_case, "rms_case_mean_mm": case_mean,
               "table1_rmse_tre_mm": TABLE1[label],
               "anis_CC_over_lat": float(rms[2] / rms[0]), "anis_CC_over_AP": float(rms[2] / rms[1])}
        rows.append(row)
        print(f"  {label:<13} {rms[0]:>8.3f} {rms[1]:>8.3f} {rms[2]:>8.3f} {pooled:>8.3f} {case_mean:>10.3f} {TABLE1[label]:>6.2f}  "
              f"{row['anis_CC_over_lat']:>7.2f} {row['anis_CC_over_AP']:>7.2f}")
    with open(os.path.join(OUT_DIR, "directional_tre.json"), "w") as f:
        json.dump({"dataset": "DIR-LAB Castillo", "cases": list(CASES),
                   "axes": "x=lateral, y=AP, z=cranio-caudal (slice axis, 2.5 mm)",
                   "raw_motion_rms_xyz_mm": raw_rms.tolist(),
                   "validation": "residual norm == tre_mm per landmark; mean of per-case RMS == Table 1 (+-0.02 mm)",
                   "per_engine": rows}, f, indent=2)
    print(f"\nResults -> {os.path.join(OUT_DIR, 'directional_tre.json')}")


if __name__ == "__main__":
    main()
