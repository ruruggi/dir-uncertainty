"""
compute_synra_no_lim.py — Subset of SyNRA calculations excluding the LIM cases
(Large Initial Misalignment, defined as pre-Procrustes RMS TRE > LIM_THR mm).

Output:
  - for each dataset, pooled post-SyNRA RMSE TRE on the no-LIM subset (Table 1)
  - for each dataset, calibration f + 5-fold LECE + global LECE + per TRE
    threshold {15, 10, 5} mm on the no-LIM subset (Table 2)
"""
from __future__ import annotations
from dirunc_paths import DATASETS  # data roots, see README
import os
import sys
import json
import shutil
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LIM_THR = 25.0  # pre-Procrustes RMS TRE > LIM_THR mm = LIM case (excluded)

DATASETS = {
    "Lung":    f"{DATASETS}/Results_SynRA/Lung",
    "Abdomen": f"{DATASETS}/Results_SynRA/Abdomen",
    "Liver":   f"{DATASETS}/Results_SynRA/Liver",
}


def split_lim_no_lim(src: str, lim_thr: float):
    """Returns (no_lim_cases: list, lim_cases: list).
    Each case: (case_id, summary, npz_landmarks).
    """
    no_lim, lim = [], []
    for sub in sorted(os.listdir(src)):
        p = os.path.join(src, sub, "summary.json")
        if not os.path.isfile(p): continue
        s = json.load(open(p))
        pre = s.get("procrustes", {}).get("initial_rmse_tre_pre_procrustes_mm")
        if pre is None: continue
        npz_path = os.path.join(src, sub, "landmarks.npz")
        if pre > lim_thr:
            lim.append((s["case_id"], s, npz_path))
        else:
            no_lim.append((s["case_id"], s, npz_path))
    return no_lim, lim


def pooled_rmse_tre(cases):
    """Pooled RMSE TRE across all landmarks of all cases (post-DIR)."""
    all_tre = []
    for _, _, npz_path in cases:
        z = np.load(npz_path)
        all_tre.append(z["tre_mm"].ravel())
    if not all_tre: return None, 0
    all_tre = np.concatenate(all_tre)
    return float(np.sqrt(np.mean(all_tre ** 2))), int(len(all_tre))


def main():
    # Print the summary table
    print(f"\n=== SyNRA no L.I.M. (LIM_THR = pre-Procrustes RMS TRE > {LIM_THR} mm) ===\n")
    print(f"{'Dataset':<10} {'Total':>5} {'no-LIM':>7} {'LIM':>5} "
          f"{'Pooled RMSE no-LIM (mm)':>26}  Excluded LIM cases")
    summary_csv = []
    no_lim_subsets = {}
    for label, src in DATASETS.items():
        no_lim, lim = split_lim_no_lim(src, LIM_THR)
        rmse_no_lim, n_lm_no_lim = pooled_rmse_tre(no_lim)
        excluded_ids = ",".join(c[0] for c in lim)
        print(f"{label:<10} {len(no_lim)+len(lim):>5} {len(no_lim):>7} {len(lim):>5} "
              f"{rmse_no_lim:>26.2f}  ({excluded_ids})")
        summary_csv.append([label, len(no_lim)+len(lim), len(no_lim), len(lim),
                            rmse_no_lim, excluded_ids])
        no_lim_subsets[label] = no_lim

    # ---- Calibration f + LECE on the no-LIM subset of each dataset ----
    print(f"\n=== Calibration f + LECE on no-LIM subsets ===")
    from validation.calibrate import (
        load_all_landmarks, stack_cases, fit_c_grid, kfold_calibrate,
    )
    from validation.lece import compute_lece, plot_reliability_diagram

    bin_width = 0.25
    seed = 42
    rep_dir = os.path.join(ROOT, "_synra_calibration_no_lim")
    os.makedirs(rep_dir, exist_ok=True)

    print(f"{'Dataset':<10} {'thr':<8} {'N':>4} {'n_lm':>7} {'f':>9} "
          f"{'LECE_global':>13} {'LECE_5fold':>15}")
    out_csv = []
    for label in DATASETS:
        no_lim_caseids = {c[0] for c in no_lim_subsets[label]}
        all_cases = load_all_landmarks(DATASETS[label])
        sub_cases = [c for c in all_cases if c.case_id in no_lim_caseids]
        if not sub_cases:
            print(f"{label}: empty no-LIM subset, skip"); continue

        # Per-case final RMSE
        case_rmse = {c.case_id: float(np.sqrt(np.mean(c.tre_mm ** 2)))
                     for c in sub_cases}

        # All thresholds: all + 15, 10, 5
        rows_for_label = []
        for thr_label, thr_val in [("all", None), ("<=15 mm", 15.0),
                                    ("<=10 mm", 10.0), ("<=5 mm", 5.0)]:
            if thr_val is None:
                kept = sub_cases
            else:
                kept = [c for c in sub_cases if case_rmse[c.case_id] <= thr_val]
            if not kept:
                rows_for_label.append([label, thr_label, 0, 0, "-", "-", "-"])
                print(f"{label:<10} {thr_label:<8} {0:>4} {0:>7} {'-':>9} "
                      f"{'-':>13} {'-':>15}")
                continue
            n_splits = min(5, len(kept))
            tre_all, sig_all = stack_cases(kept)
            S_g, lece_g = fit_c_grid(tre_all, sig_all, bin_width=bin_width)
            try:
                folds = kfold_calibrate(kept, n_splits=n_splits, seed=seed,
                                        bin_width=bin_width)
                lece_5f = np.array([f.lece_test_grid for f in folds])
                lece_5f_mean = float(lece_5f.mean())
                lece_5f_std = float(lece_5f.std(ddof=1)) if len(lece_5f) > 1 else 0.0
            except ValueError:
                lece_5f_mean = float("nan"); lece_5f_std = float("nan")
            rows_for_label.append([label, thr_label, len(kept), int(tre_all.size),
                                    f"{S_g:.3f}", f"{lece_g:.3f}",
                                    f"{lece_5f_mean:.2f}+-{lece_5f_std:.2f}"])
            print(f"{label:<10} {thr_label:<8} {len(kept):>4} {int(tre_all.size):>7} "
                  f"{S_g:>9.3f} {lece_g:>13.3f} {lece_5f_mean:>9.2f}+-{lece_5f_std:.2f}")
            # Reliability diagram for this threshold
            tre_arr, sig_arr = stack_cases(kept)
            lece, diag = compute_lece(tre_arr, S_g * sig_arr, bin_width=bin_width)
            thr_safe = thr_label.replace(" ", "").replace("<=", "le").replace("/", "_")
            out_png = os.path.join(rep_dir,
                                    f"reliability_{label}_SyNRA_noLIM_{thr_safe}.png")
            plot_reliability_diagram(
                diag, output_path=out_png,
                title=f"{label} no-LIM (thr={thr_label}, "
                      f"f={S_g:.2f}, LECE={lece:.2f} mm)",
                label="heuristic+residual SyNRA no-LIM (calibrated)",
            )
        out_csv.extend(rows_for_label)
        print()

    # CSV with all results
    csv_path = os.path.join(rep_dir, "synra_no_lim_summary.csv")
    with open(csv_path, "w") as f:
        f.write("dataset,threshold,N,n_landmarks,f_global,LECE_global_mm,LECE_5fold_mm\n")
        for row in out_csv:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
