# -*- coding: utf-8 -*-
"""
synra_table1_baseline.py  (PMB revision, Q1.3 / Table 1 under alpha)

Recompute SyNRA mean post-DIR RMSE TRE per QA/DIR-LAB dataset on the FULL set
(no LIM pre-filter), so SyNRA is reported on the same footing as the other 3
engines in Table 1. Reports, per dataset:
  - full set (all cases):        N, n_landmarks, mean post-DIR RMSE TRE
  - converged q<=10 mm subset:   N, n_landmarks, mean post-DIR RMSE TRE
Mean RMSE TRE = mean over cases of sqrt(mean(tre^2)) (per-case RMS, then averaged),
matching the "Mean post-DIR RMSE TRE" definition used for Table 1.
Pure analysis on consolidated landmarks.npz (no model inference).
"""
from __future__ import annotations
import os, sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from validation.compute_synra_no_lim import split_lim_no_lim, DATASETS, LIM_THR
from validation.calibrate import load_all_landmarks


def main():
    print("=== SyNRA Table-1 baseline: full set (no LIM) vs q<=10 subset ===")
    print(f"  (LIM pre-filter = pre-Procrustes RMS TRE > {LIM_THR} mm; now removed)\n")
    print(f"  {'dataset':<18} {'N_full':>6} {'RMSE_full':>10} {'N@q10':>6} {'RMSE@q10':>9} {'n_lm_full':>10}")
    for label, src in DATASETS.items():
        all_cases = load_all_landmarks(src)
        per_case_rms = {c.case_id: float(np.sqrt(np.mean(c.tre_mm ** 2))) for c in all_cases}
        n_lm_full = int(sum(c.tre_mm.size for c in all_cases))
        rms_full = float(np.mean([per_case_rms[c.case_id] for c in all_cases]))
        conv = [c for c in all_cases if per_case_rms[c.case_id] <= 10.0]
        rms_q10 = float(np.mean([per_case_rms[c.case_id] for c in conv])) if conv else float("nan")
        print(f"  {label:<18} {len(all_cases):>6} {rms_full:>10.2f} {len(conv):>6} {rms_q10:>9.2f} {n_lm_full:>10}")


if __name__ == "__main__":
    main()
