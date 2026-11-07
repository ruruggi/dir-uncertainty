"""
rebuild_table2_bin025_noLIM.py — Q1.3 / Option B + bin-width harmonisation.

Regenerate Table 2 with BOTH revision changes:
  (1) NO LIM filter for any engine (full set; only the shared convergence threshold q);
  (2) LECE bin_width = 0.25 mm (the value stated in §2.5; the original table used 0.5).

Uses the canonical per-(engine,dataset) folder map ENGINE_PATHS and the canonical
calibration functions from build_v11_data / calibrate, only overriding bin_width and
forcing filter_lim=False.

Columns mirror Table 2: f(all), LECE global(all), LECE 5-fold(all), 5-fold(<10), 5-fold(<5).
Output: _synra_calibration_no_lim/table2_bin025_noLIM.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from validation.build_v11_data import ENGINE_PATHS
from validation.calibrate import load_all_landmarks, fit_c_grid, stack_cases, kfold_calibrate
from validation.lece import compute_lece

BIN = 0.25
DATASETS = ["Lung-DIR-QA", "Abdominal-DIR-QA", "Liver-DIR-QA", "DIR-LAB"]


def cal(cases):
    """global f, global LECE, 5-fold mean/std at BIN."""
    if not cases: return None
    n = min(5, len(cases))
    tre, sig = stack_cases(cases)
    f, _ = fit_c_grid(tre, sig, bin_width=BIN)
    lg, _ = compute_lece(tre, f * sig, bin_width=BIN)
    if n <= 1:
        return {"f": float(f), "global": float(lg), "kf_mean": float(lg), "kf_std": 0.0, "n": len(cases)}
    folds = kfold_calibrate(cases, n_splits=n, seed=42, bin_width=BIN)
    cv = np.array([fl.lece_test_grid for fl in folds])
    return {"f": float(f), "global": float(lg), "kf_mean": float(cv.mean()),
            "kf_std": float(cv.std(ddof=1)) if len(cv) > 1 else 0.0, "n": len(cases)}


def main():
    out = {}
    print(f"=== Table 2 regenerated: NO-LIM (full set) + bin_width={BIN} mm ===\n")
    print(f"{'Dataset':<18}{'Engine':<13}{'f':>7}{'global':>8}{'5fold(all)':>16}{'<=10 5f':>16}{'<=5 5f':>16}")
    for ds in DATASETS:
        out[ds] = {}
        for eng, cfg in ENGINE_PATHS.items():
            path = cfg["datasets"].get(ds)
            if not path or not os.path.isdir(path):
                continue
            cases = load_all_landmarks(path)           # full set, NO LIM filter
            rmse = {c.case_id: float(np.sqrt(np.mean(c.tre_mm ** 2))) for c in cases}
            res = {"all": cal(cases)}
            for thr in (10.0, 5.0):
                kept = [c for c in cases if rmse[c.case_id] <= thr]
                res[f"<={thr:.0f}"] = cal(kept)
            out[ds][eng] = res
            a = res["all"]; t10 = res["<=10"]; t5 = res["<=5"]
            def kf(r): return f"{r['kf_mean']:.2f}±{r['kf_std']:.2f}" if r else "N.A."
            print(f"{ds:<18}{eng:<13}{a['f']:>7.2f}{a['global']:>8.2f}{kf(a):>16}{kf(t10):>16}{kf(t5):>16}")
    odir = os.path.join(ROOT, "_synra_calibration_no_lim"); os.makedirs(odir, exist_ok=True)
    with open(os.path.join(odir, "table2_bin025_noLIM.json"), "w") as f:
        json.dump({"bin_width": BIN, "lim_filter": False, "data": out}, f, indent=2)
    print(f"\n-> {os.path.join(odir, 'table2_bin025_noLIM.json')}")


if __name__ == "__main__":
    main()
