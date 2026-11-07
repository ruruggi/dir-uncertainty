"""
build_v11_data.py — Aggregator for Table 1 + Table 2 (4 engines
× 4 datasets, homogeneous).

For each of the 16 (engine, dataset) combinations it computes:
    - n_cases, n_landmarks
    - TRE_initial mean (pre-Procrustes RMS TRE)
    - TRE_final mean (post-DIR RMS TRE)
    - global f + global LECE + 5-fold LECE per threshold {all, ≤15, ≤10, ≤5}
SyNRA uses the LIM filter (pre-Procrustes RMS TRE ≤ 25 mm); the other 3 engines
include all cases.

Output: a dict (no JSON file is saved).
"""
from __future__ import annotations

from dirunc_paths import DATASETS  # data roots, see README
import os
import sys
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from validation.calibrate import (
    load_all_landmarks, stack_cases,
    fit_c_grid, kfold_calibrate,
)
from validation.lece import compute_lece


LIM_THR = 25.0  # pre-Procrustes RMS TRE > 25 mm = LIM (filter for SyNRA)
_CASTILLO = f"{DATASETS}/Castillo-DIR-LAB"

# (engine_label, engine_key in Results dir, datasets dict)
ENGINE_PATHS = {
    "SyNRA": {
        "filter_lim": True,
        "datasets": {
            "Lung-DIR-QA":      f"{DATASETS}/Results_SynRA/Lung",
            "Abdominal-DIR-QA": f"{DATASETS}/Results_SynRA/Abdomen",
            "Liver-DIR-QA":     f"{DATASETS}/Results_SynRA/Liver",
            "DIR-LAB":          f"{_CASTILLO}/Results_synra_procrustes",
        },
    },
    "ConvexAdam": {
        "filter_lim": False,
        "datasets": {
            "Lung-DIR-QA":      f"{DATASETS}/Lung-DIR-QA/Results_adam_procrustes",
            "Abdominal-DIR-QA": f"{DATASETS}/AbdominalDIR-QA/Results_adam_procrustes",
            "Liver-DIR-QA":     f"{DATASETS}/Liver-DIR-QA/Results_adam_procrustes",
            "DIR-LAB":          f"{_CASTILLO}/Results_convexadam_procrustes",
        },
    },
    "uniGradICON": {
        "filter_lim": False,
        "datasets": {
            "Lung-DIR-QA":      f"{DATASETS}/Lung-DIR-QA/Results_unigradicon_procrustes",
            "Abdominal-DIR-QA": f"{DATASETS}/AbdominalDIR-QA/Annotated Image Pairs/Results_unigradicon_procrustes",
            "Liver-DIR-QA":     f"{DATASETS}/Liver-DIR-QA/Results_unigradicon_procrustes",
            "DIR-LAB":          f"{_CASTILLO}/Results_unigradicon_procrustes",
        },
    },
    "Elastix": {
        "filter_lim": False,
        "datasets": {
            "Lung-DIR-QA":      f"{DATASETS}/Lung-DIR-QA/Results_elastix_procrustes",
            "Abdominal-DIR-QA": f"{DATASETS}/AbdominalDIR-QA/Annotated Image Pairs/Results_elastix_procrustes",
            "Liver-DIR-QA":     f"{DATASETS}/Liver-DIR-QA/Results_elastix_procrustes",
            "DIR-LAB":          f"{_CASTILLO}/Results_elastix_procrustes",
        },
    },
}

DATASETS = ["Lung-DIR-QA", "Abdominal-DIR-QA", "Liver-DIR-QA", "DIR-LAB"]
DATASET_N_FULL = {"Lung-DIR-QA": 30, "Abdominal-DIR-QA": 30,
                    "Liver-DIR-QA": 30, "DIR-LAB": 10}


def _read_summary_json(out_root: str, case_id: str) -> dict:
    p = os.path.join(out_root, case_id, "summary.json")
    with open(p) as f:
        return json.load(f)


def _filter_cases_lim(out_root: str, lim_thr: float = LIM_THR) -> set:
    """Return set of case_id with pre-Procrustes RMS TRE <= lim_thr."""
    keep = set()
    for sub in sorted(os.listdir(out_root)):
        p = os.path.join(out_root, sub, "summary.json")
        if not os.path.isfile(p): continue
        try:
            s = json.load(open(p))
        except Exception:
            continue
        pre = s.get("procrustes", {}).get("initial_rmse_tre_pre_procrustes_mm")
        if pre is None or pre <= lim_thr:
            keep.add(s.get("case_id", sub))
    return keep


def _calibrate_subset(cases, bin_width: float = 0.25, n_splits: int = 5) -> dict:
    """Returns f_global, lece_global, lece_kfold_mean ± std.

    NB: default bin_width harmonised to 0.25 mm to match section 2.5,
    Table 3 and the figure builders."""
    if not cases:
        return None
    n = min(n_splits, len(cases))
    if n <= 1:
        # no kfold meaningful
        tre_all, sig_all = stack_cases(cases)
        f_global, _ = fit_c_grid(tre_all, sig_all, bin_width=bin_width)
        sig_cal = f_global * sig_all
        lece_g, _ = compute_lece(tre_all, sig_cal, bin_width=bin_width)
        return {"f_global": float(f_global),
                "lece_global": float(lece_g),
                "lece_kfold_mean": float(lece_g),
                "lece_kfold_std": 0.0,
                "n_cases_thr": len(cases)}
    folds = kfold_calibrate(cases, n_splits=n, seed=42, bin_width=bin_width)
    cv_grid_test = np.array([f.lece_test_grid for f in folds])
    tre_all, sig_all = stack_cases(cases)
    f_global, _ = fit_c_grid(tre_all, sig_all, bin_width=bin_width)
    sig_cal = f_global * sig_all
    lece_g, _ = compute_lece(tre_all, sig_cal, bin_width=bin_width)
    return {
        "f_global": float(f_global),
        "lece_global": float(lece_g),
        "lece_kfold_mean": float(cv_grid_test.mean()),
        "lece_kfold_std": float(cv_grid_test.std(ddof=1)) if len(cv_grid_test) > 1 else 0.0,
        "n_cases_thr": len(cases),
    }


def aggregate_engine_dataset(out_root: str, filter_lim: bool,
                                thresholds=(15.0, 10.0, 5.0)) -> dict:
    """Aggregate per (dataset, engine). Returns:
    {
        n_cases (post-LIM-filter),
        n_landmarks,
        tre_initial_mean, tre_final_mean,
        thresholds: { 'all'/'≤15'/'≤10'/'≤5': {f_global, lece_global,
                                                  lece_kfold_mean, lece_kfold_std}}
    }
    """
    cases_all = load_all_landmarks(out_root)
    if filter_lim:
        keep = _filter_cases_lim(out_root, LIM_THR)
        cases = [c for c in cases_all if c.case_id in keep]
    else:
        cases = cases_all

    if not cases:
        return None

    # Per case: TRE_final
    case_final_rmse = {c.case_id: float(np.sqrt(np.mean(c.tre_mm ** 2)))
                        for c in cases}

    # TRE_initial (pre-Procrustes) and n_landmarks
    initial_rmse_list = []
    n_landmarks_total = 0
    for c in cases:
        try:
            s = _read_summary_json(out_root, c.case_id)
        except Exception:
            continue
        pre = s.get("procrustes", {}).get("initial_rmse_tre_pre_procrustes_mm")
        if pre is None:
            pre = s.get("initial_rmse_tre_mm")
        if pre is not None:
            initial_rmse_list.append(float(pre))
        n_landmarks_total += int(s.get("n_landmarks", len(c.tre_mm)))

    final_rmse_list = list(case_final_rmse.values())

    thr_results = {"all": _calibrate_subset(cases)}
    for thr in thresholds:
        kept = [c for c in cases if case_final_rmse[c.case_id] <= thr]
        thr_results[f"<={thr:.0f}mm"] = _calibrate_subset(kept) if kept else None

    return {
        "n_cases": len(cases),
        "n_landmarks": n_landmarks_total,
        "tre_initial_mean_mm": float(np.mean(initial_rmse_list)) if initial_rmse_list else None,
        "tre_final_mean_mm": float(np.mean(final_rmse_list)),
        "thresholds": thr_results,
    }


def build_all() -> dict:
    """Aggregate over all 16 (engine, dataset) combinations. Returns a hierarchical dict."""
    out = {}
    for engine_label, cfg in ENGINE_PATHS.items():
        out[engine_label] = {}
        for ds_label, ds_path in cfg["datasets"].items():
            if not os.path.isdir(ds_path):
                print(f"  [warn] missing {ds_path}", file=sys.stderr)
                continue
            res = aggregate_engine_dataset(ds_path, filter_lim=cfg["filter_lim"])
            if res is None:
                continue
            out[engine_label][ds_label] = res
    return out


if __name__ == "__main__":
    import json
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data = build_all()
    # Print summary
    for engine, ds_map in data.items():
        print(f"\n=== {engine} ===")
        for ds, stats in ds_map.items():
            f_all = stats["thresholds"]["all"]["f_global"]
            lece_5fold = stats["thresholds"]["all"]["lece_kfold_mean"]
            lece_5fold_std = stats["thresholds"]["all"]["lece_kfold_std"]
            print(f"  {ds:<22}: n={stats['n_cases']:<3} "
                  f"TRE_init={stats['tre_initial_mean_mm']:.2f} mm "
                  f"TRE_final={stats['tre_final_mean_mm']:.2f} mm "
                  f"f={f_all:.2f} LECE5fold={lece_5fold:.2f}±{lece_5fold_std:.2f} mm")
