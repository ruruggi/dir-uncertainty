"""
validation/calibrate.py — Calibration of the global factor c for the
heuristic+residual model.

Works on the per-case `landmarks.npz` files produced by run_pair / run_batch
(`sigma_p_mm_raw` + `tre_mm`). Finds the single c > 0 such that
sigma_p = c * sigma_p_raw is well calibrated against TRE.

Strategies:
    - 5-fold CV over cases (primary, for generalisation)
    - LOOCV (sanity check)
    - Global (for the final figure and the c to deploy)

For each fit of c, two optimisers are compared:
    - GRID:  log-spaced grid + refinement, minimises LECE directly
             (Smolders 2023, Eq. 9). Robust, no assumptions.
    - MLE :  Maxwell-Boltzmann log-likelihood assuming an isotropic 3D Gaussian
             error with std c*sigma_h. Smooth in c, scipy.minimize_scalar.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold

if __package__ is None or __package__ == "":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    sys.path.insert(0, os.path.join(os.path.dirname(here), "dl_arch"))

from validation.lece import compute_lece, plot_reliability_diagram


# ----------------------------------------------------------------------
# Per-case data containers
# ----------------------------------------------------------------------
@dataclass
class CaseLandmarks:
    case_id: str
    tre_mm: np.ndarray         # (N,)
    sigma_raw: np.ndarray      # (N,)  dimensionless sigma in [0, 1]


def load_all_landmarks(out_root: str,
                       sigma_field: str = "sigma_p_mm_raw") -> List[CaseLandmarks]:
    """Load all landmarks.npz files produced by run_batch in out_root.

    Filters out landmarks with sigma_raw == 0 (outside the body mask) or
    non-finite values. Returns one entry per case_id present.
    """
    cases = []
    for sub in sorted(os.listdir(out_root)):
        npz_path = os.path.join(out_root, sub, "landmarks.npz")
        if not os.path.isfile(npz_path):
            continue
        z = np.load(npz_path, allow_pickle=True)
        if sigma_field not in z.files:
            # backward compat: old runs only have sigma_p_mm
            sigma_raw = z["sigma_p_mm"]
        else:
            sigma_raw = z[sigma_field]
        tre = np.asarray(z["tre_mm"], dtype=np.float64).ravel()
        sig = np.asarray(sigma_raw, dtype=np.float64).ravel()
        # Drop landmarks with sigma <= 0 (outside mask) or NaN
        keep = np.isfinite(tre) & np.isfinite(sig) & (sig > 1e-9) & (tre >= 0)
        if keep.sum() == 0:
            print(f"  [{sub}] skipped: no usable landmarks")
            continue
        cases.append(CaseLandmarks(case_id=str(sub),
                                   tre_mm=tre[keep],
                                   sigma_raw=sig[keep]))
    return cases


def stack_cases(cases: List[CaseLandmarks]) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate the landmarks of several cases into a single (TRE, sigma_raw)."""
    if not cases:
        return np.array([]), np.array([])
    tre = np.concatenate([c.tre_mm for c in cases])
    sig = np.concatenate([c.sigma_raw for c in cases])
    return tre, sig


# ----------------------------------------------------------------------
# Optimization: c that minimises LECE (grid) and that maximises MLE
# ----------------------------------------------------------------------
def fit_c_grid(tre: np.ndarray, sigma_raw: np.ndarray,
               c_min: float = 0.05, c_max: float = 200.0,
               n_grid: int = 200, refine_factor: float = 50,
               bin_width: float = 0.25,
               convergence_regime_max_tre: float = None) -> Tuple[float, float]:
    """Search for the c that minimises LECE via a log-spaced grid + refinement.

    Optional `convergence_regime_max_tre`: when not None, restrict the fit to
    landmark pairs with TRE <= this threshold (mm). Reproduces the
    "convergence-regime" filter of the paper §2.5 (Algorithm B.4 step 2) at
    the function level, instead of requiring the caller to pre-filter the
    input arrays.
    """
    if tre.size == 0:
        return 1.0, float("nan")
    if convergence_regime_max_tre is not None:
        keep = tre <= float(convergence_regime_max_tre)
        tre = tre[keep]
        sigma_raw = sigma_raw[keep]
        if tre.size == 0:
            return 1.0, float("nan")
    grid = np.geomspace(c_min, c_max, n_grid)
    best_c, best_l = grid[0], np.inf
    for c in grid:
        l, _ = compute_lece(tre, c * sigma_raw, bin_width=bin_width)
        if l < best_l:
            best_l = l
            best_c = c
    # refinement around best_c
    span = best_c / refine_factor + 0.001
    fine = np.linspace(max(c_min, best_c - 5 * span), best_c + 5 * span, 80)
    for c in fine:
        if c <= 0:
            continue
        l, _ = compute_lece(tre, c * sigma_raw, bin_width=bin_width)
        if l < best_l:
            best_l = l
            best_c = c
    return float(best_c), float(best_l)


def fit_c_mle(tre: np.ndarray, sigma_raw: np.ndarray) -> Tuple[float, float]:
    """Estimate c via MLE assuming an isotropic 3D Gaussian error:
        ||error|| ~ Maxwell-Boltzmann(scale = c * sigma_raw)
    log L(c) = sum_i [ log(r_i^2) - 3 log(c sigma_i) - r_i^2 / (2 (c sigma_i)^2) ]
    """
    if tre.size == 0:
        return 1.0, float("nan")
    r = tre[(tre > 0) & (sigma_raw > 0)]
    s = sigma_raw[(tre > 0) & (sigma_raw > 0)]

    def neg_log_lik(c):
        if c <= 0:
            return np.inf
        scale = c * s
        return float(np.sum(3 * np.log(scale) + r ** 2 / (2 * scale ** 2)))

    res = minimize_scalar(neg_log_lik, bounds=(1e-3, 1e4), method="bounded")
    if not res.success:
        return 1.0, float("nan")
    return float(res.x), float(res.fun)


# ----------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------
@dataclass
class FoldResult:
    fold: int
    train_cases: List[str]
    test_cases: List[str]
    c_grid: float
    c_mle: float
    lece_train_grid: float
    lece_test_grid: float
    lece_test_mle: float


def kfold_calibrate(cases: List[CaseLandmarks], n_splits: int = 5,
                    seed: int = 42, bin_width: float = 0.25
                    ) -> List[FoldResult]:
    if n_splits > len(cases):
        raise ValueError(f"n_splits={n_splits} > #cases={len(cases)}")
    rng = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    case_ids = [c.case_id for c in cases]
    results = []
    for k, (train_idx, test_idx) in enumerate(rng.split(case_ids)):
        train_cases = [cases[i] for i in train_idx]
        test_cases = [cases[i] for i in test_idx]
        tre_tr, sig_tr = stack_cases(train_cases)
        tre_te, sig_te = stack_cases(test_cases)

        c_grid, lece_train_grid = fit_c_grid(tre_tr, sig_tr, bin_width=bin_width)
        c_mle, _ = fit_c_mle(tre_tr, sig_tr)

        # apply to test
        lece_test_grid, _ = compute_lece(tre_te, c_grid * sig_te, bin_width=bin_width)
        lece_test_mle, _ = compute_lece(tre_te, c_mle * sig_te, bin_width=bin_width)

        results.append(FoldResult(
            fold=k,
            train_cases=[c.case_id for c in train_cases],
            test_cases=[c.case_id for c in test_cases],
            c_grid=c_grid, c_mle=c_mle,
            lece_train_grid=lece_train_grid,
            lece_test_grid=lece_test_grid,
            lece_test_mle=lece_test_mle,
        ))
    return results


def loocv_calibrate(cases: List[CaseLandmarks], bin_width: float = 0.25
                    ) -> List[FoldResult]:
    return kfold_calibrate(cases, n_splits=len(cases), seed=0, bin_width=bin_width)


# ----------------------------------------------------------------------
# Main reporting
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Calibrate the global factor c on the completed dataset "
                    "via 5-fold CV + LOOCV + global fit, and produce the "
                    "final reliability diagram."
    )
    parser.add_argument("--out_root", required=True,
                        help="Root folder produced by run_batch.py "
                        "(with one subfolder per case_id containing landmarks.npz).")
    parser.add_argument("--report_dir", default=None,
                        help="Output folder for the report (default: <out_root>/_calibration).")
    parser.add_argument("--bin_width", type=float, default=0.25,
                        help="LECE bin width in mm (default 0.25 from Smolders 2023).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_loocv", action="store_true",
                        help="Skip LOOCV (sanity check). Default: it is run.")
    parser.add_argument("--register", default=None,
                        help="District ID in config/c_calibration_registry.json "
                        "to update with the fitted c (e.g. 'abdomen_CT'). "
                        "If None, the registry is left untouched.")
    parser.add_argument("--registry_path", default=None,
                        help="Path to the registry JSON (default: <project>/config/c_calibration_registry.json).")
    parser.add_argument("--exclude", default=None,
                        help="Comma-separated list of case_ids to exclude "
                        "(e.g. non-converging DIR cases). Exclusions are "
                        "logged and end up in calibration_report.json.")
    parser.add_argument("--max_final_tre_mm", type=float, default=None,
                        help="Per-case RMSE TRE threshold (mm): cases with final RMSE "
                        "TRE > threshold are auto-excluded. If None (default), "
                        "no automatic exclusion.")
    args = parser.parse_args()

    if args.report_dir is None:
        args.report_dir = os.path.join(args.out_root, "_calibration")
    os.makedirs(args.report_dir, exist_ok=True)

    print("=" * 70)
    print("Calibration of c (heuristic+residual) on Abdominal-DIR-QA")
    print("=" * 70)
    print(f"  out_root  : {args.out_root}")
    print(f"  report_dir: {args.report_dir}")
    print(f"  bin_width : {args.bin_width} mm")
    print(f"  seed      : {args.seed}")
    print()

    cases_all = load_all_landmarks(args.out_root)
    if not cases_all:
        print("ERROR: no case with a valid landmarks.npz in", args.out_root)
        return

    # ---- Exclusions: manual (--exclude) and automatic (--max_final_tre_mm)
    excluded_manual = set()
    if args.exclude:
        excluded_manual = {s.strip() for s in args.exclude.split(",") if s.strip()}
    excluded_auto = set()
    if args.max_final_tre_mm is not None:
        for c in cases_all:
            rmse = float(np.sqrt(np.mean(c.tre_mm ** 2)))
            if rmse > float(args.max_final_tre_mm):
                excluded_auto.add(c.case_id)

    excluded = excluded_manual | excluded_auto
    cases = [c for c in cases_all if c.case_id not in excluded]
    if excluded:
        print(f"Excluded {len(excluded)} case(s) from calibration:")
        for cid in sorted(excluded):
            tag = []
            if cid in excluded_manual: tag.append("manual")
            if cid in excluded_auto:   tag.append(f"final_RMSE_TRE > {args.max_final_tre_mm} mm")
            print(f"  {cid}  ({', '.join(tag)})")
        print()
    if not cases:
        print("ERROR: all cases excluded, none available for calibration.")
        return

    print(f"Loaded {len(cases)} cases (after exclusion):")
    for c in cases:
        print(f"  {c.case_id:<10}  N={c.tre_mm.size}  "
              f"TRE median={np.median(c.tre_mm):.2f} mm  "
              f"sigma_raw median={np.median(c.sigma_raw):.3f}")

    # ---- 5-fold CV
    # NB: the FoldResult dataclass has the fields c_grid/c_mle for historical
    # reasons (they corresponded to the symbol c of Eq.10 in early versions of
    # the paper). The symbol in the paper has since been renamed S; the Python
    # variables keep the old name but the logs and JSON keys use S. Treat
    # c_grid/c_mle as internal aliases of S_grid/S_mle.
    print("\n--- 5-fold CV ---")
    folds = kfold_calibrate(cases, n_splits=5, seed=args.seed,
                            bin_width=args.bin_width)
    cv_grid_test = np.array([f.lece_test_grid for f in folds])
    cv_mle_test = np.array([f.lece_test_mle for f in folds])
    cv_S_grid = np.array([f.c_grid for f in folds])
    cv_S_mle = np.array([f.c_mle for f in folds])
    for f in folds:
        print(f"  fold {f.fold}: S_grid={f.c_grid:.3f} S_mle={f.c_mle:.3f}  "
              f"LECE_train={f.lece_train_grid:.3f}  LECE_test_grid={f.lece_test_grid:.3f}  "
              f"LECE_test_mle={f.lece_test_mle:.3f}  (test={f.test_cases})")
    print(f"  5-fold mean +/- std (grid) : LECE = {cv_grid_test.mean():.3f} +/- "
          f"{cv_grid_test.std(ddof=1):.3f} mm")
    print(f"  5-fold mean +/- std (mle)  : LECE = {cv_mle_test.mean():.3f} +/- "
          f"{cv_mle_test.std(ddof=1):.3f} mm")
    print(f"  S values (grid): {np.round(cv_S_grid, 3).tolist()}")
    print(f"  S values (mle) : {np.round(cv_S_mle, 3).tolist()}")

    # ---- LOOCV
    loocv = []
    if not args.no_loocv:
        print("\n--- LOOCV ---")
        loocv = loocv_calibrate(cases, bin_width=args.bin_width)
        loo_grid = np.array([f.lece_test_grid for f in loocv])
        loo_S_grid = np.array([f.c_grid for f in loocv])
        print(f"  LOOCV mean LECE (grid) = {loo_grid.mean():.3f} +/- "
              f"{loo_grid.std(ddof=1):.3f} mm  (over {len(loocv)} folds)")
        print(f"  LOOCV S range: [{loo_S_grid.min():.3f}, {loo_S_grid.max():.3f}], "
              f"median {np.median(loo_S_grid):.3f}")
        med_S = np.median(loo_S_grid)
        outlier_idx = int(np.argmax(np.abs(loo_S_grid - med_S)))
        print(f"  Most-deviating case: {loocv[outlier_idx].test_cases[0]} "
              f"(S={loo_S_grid[outlier_idx]:.3f})")

    # ---- Global fit
    print("\n--- Global fit (all retained cases) ---")
    tre_all, sig_all = stack_cases(cases)
    S_grid_full, lece_grid_full = fit_c_grid(tre_all, sig_all,
                                             bin_width=args.bin_width)
    S_mle_full, _ = fit_c_mle(tre_all, sig_all)
    lece_mle_full, _ = compute_lece(tre_all, S_mle_full * sig_all,
                                    bin_width=args.bin_width)
    print(f"  S_grid_full = {S_grid_full:.4f}, LECE_grid_full = {lece_grid_full:.3f} mm")
    print(f"  S_mle_full  = {S_mle_full:.4f},  LECE_mle_full  = {lece_mle_full:.3f} mm")

    # ---- Reliability diagram with S_grid_full
    print("\n--- Reliability diagram (Fig.4 style) ---")
    final_lece, final_diag = compute_lece(tre_all, S_grid_full * sig_all,
                                          bin_width=args.bin_width)
    plot_path = os.path.join(args.report_dir, "reliability_diagram_global.png")
    plot_reliability_diagram(
        final_diag,
        output_path=plot_path,
        title=f"Heuristic+residual on calibration set (N={len(cases)} cases, "
              f"{tre_all.size} landmarks, S={S_grid_full:.3f}, "
              f"LECE={final_lece:.3f} mm)",
        label="heuristic+residual (calibrated)",
    )
    print(f"  saved -> {plot_path}")

    # ---- JSON / CSV report
    # JSON report: the new `S_*` keys are canonical; for backward
    # compatibility with earlier reports and external tooling, the old
    # `c_*` keys are also kept as aliases.
    report = {
        "n_cases": len(cases),
        "n_cases_total_loaded": len(cases_all),
        "excluded_cases": sorted(excluded),
        "exclusion_manual": sorted(excluded_manual),
        "exclusion_auto_max_final_tre_mm": args.max_final_tre_mm,
        "exclusion_auto_cases": sorted(excluded_auto),
        "n_landmarks_total": int(tre_all.size),
        "bin_width_mm": args.bin_width,
        "seed": args.seed,
        "5fold_cv": {
            "folds": [asdict(f) for f in folds],
            "lece_test_grid_mean_mm": float(cv_grid_test.mean()),
            "lece_test_grid_std_mm": float(cv_grid_test.std(ddof=1)),
            "lece_test_mle_mean_mm": float(cv_mle_test.mean()),
            "lece_test_mle_std_mm": float(cv_mle_test.std(ddof=1)),
            "S_grid_per_fold": cv_S_grid.tolist(),
            "S_mle_per_fold": cv_S_mle.tolist(),
            "c_grid_per_fold": cv_S_grid.tolist(),  # deprecated alias
            "c_mle_per_fold": cv_S_mle.tolist(),    # deprecated alias
        },
        "loocv": {
            "folds": [asdict(f) for f in loocv],
            "lece_test_grid_mean_mm": float(np.mean([f.lece_test_grid for f in loocv])) if loocv else None,
            "lece_test_grid_std_mm": float(np.std([f.lece_test_grid for f in loocv], ddof=1)) if len(loocv) > 1 else None,
            "S_grid_min": float(min([f.c_grid for f in loocv])) if loocv else None,
            "S_grid_max": float(max([f.c_grid for f in loocv])) if loocv else None,
            "S_grid_median": float(np.median([f.c_grid for f in loocv])) if loocv else None,
            "c_grid_min": float(min([f.c_grid for f in loocv])) if loocv else None,         # alias
            "c_grid_max": float(max([f.c_grid for f in loocv])) if loocv else None,         # alias
            "c_grid_median": float(np.median([f.c_grid for f in loocv])) if loocv else None, # alias
        },
        "global": {
            "S_grid": float(S_grid_full),
            "lece_grid": float(lece_grid_full),
            "S_mle": float(S_mle_full),
            "lece_mle": float(lece_mle_full),
            "c_grid": float(S_grid_full),  # deprecated alias
            "c_mle": float(S_mle_full),    # deprecated alias
        },
        "recommended_S": float(S_grid_full),
        "recommended_c": float(S_grid_full),  # deprecated alias
    }
    json_path = os.path.join(args.report_dir, "calibration_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  report saved -> {json_path}")

    # CSV for the publication table
    csv_path = os.path.join(args.report_dir, "calibration_summary.csv")
    with open(csv_path, "w") as f:
        f.write("metric,value,units\n")
        f.write(f"n_cases,{len(cases)},\n")
        f.write(f"n_landmarks_total,{tre_all.size},\n")
        f.write(f"bin_width,{args.bin_width},mm\n")
        f.write(f"5fold_LECE_grid,{cv_grid_test.mean():.4f} +/- {cv_grid_test.std(ddof=1):.4f},mm\n")
        f.write(f"5fold_LECE_mle,{cv_mle_test.mean():.4f} +/- {cv_mle_test.std(ddof=1):.4f},mm\n")
        if loocv:
            f.write(f"loocv_LECE_grid,{np.mean([f.lece_test_grid for f in loocv]):.4f} "
                    f"+/- {np.std([f.lece_test_grid for f in loocv], ddof=1):.4f},mm\n")
        f.write(f"global_S_grid,{S_grid_full:.6f},(unitless)\n")
        f.write(f"global_S_mle,{S_mle_full:.6f},(unitless)\n")
        f.write(f"global_LECE_grid,{lece_grid_full:.4f},mm\n")
        f.write(f"global_LECE_mle,{lece_mle_full:.4f},mm\n")
        f.write(f"recommended_S,{S_grid_full:.6f},(unitless)\n")
    print(f"  csv saved    -> {csv_path}")

    print()
    print("=" * 70)
    print(f"RECOMMENDED for the pipeline: S = {S_grid_full:.4f}")
    print(f"  (5-fold LECE = {cv_grid_test.mean():.3f} +/- "
          f"{cv_grid_test.std(ddof=1):.3f} mm, "
          f"global LECE = {lece_grid_full:.3f} mm)")
    print("=" * 70)

    # ---- Update the registry if requested -----------------------------------
    if args.register:
        registry_path = args.registry_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "c_calibration_registry.json")
        try:
            updated = update_registry(
                registry_path=registry_path,
                district_id=args.register,
                c_value=S_grid_full,
                lece_cv_mean=float(cv_grid_test.mean()),
                lece_cv_std=float(cv_grid_test.std(ddof=1)),
                loocv_lece=(float(np.mean([f.lece_test_grid for f in loocv]))
                            if loocv else None),
                n_landmarks=int(tre_all.size),
                n_cases=len(cases),
                bin_width_mm=args.bin_width,
                fit_seed=args.seed,
            )
            if updated:
                print(f"\nRegistry updated: {registry_path} (district='{args.register}')")
            else:
                print(f"\n[WARNING] Registry NOT updated (district '{args.register}' "
                      f"not found in {registry_path})")
        except Exception as e:
            print(f"\n[ERROR] Registry update failed: {e}")


def update_registry(registry_path: str,
                    district_id: str,
                    c_value: float,
                    lece_cv_mean: float,
                    lece_cv_std: float,
                    loocv_lece,
                    n_landmarks: int,
                    n_cases: int,
                    bin_width_mm: float,
                    fit_seed: int) -> bool:
    """Update the district_id entry of the registry with the calibration results.

    Returns True if the entry was updated, False if district_id does not exist.
    """
    import datetime
    if not os.path.isfile(registry_path):
        raise FileNotFoundError(f"Registry not found: {registry_path}")
    with open(registry_path, "r") as f:
        reg = json.load(f)
    if "districts" not in reg or district_id not in reg["districts"]:
        return False
    entry = reg["districts"][district_id]
    # The canonical key is 'S'; 'c' is kept as a deprecated alias so as not
    # to break readers that still look for it.
    entry["S"] = float(c_value)
    entry["c"] = float(c_value)
    entry["lece_cv_mean_mm"] = float(lece_cv_mean)
    entry["lece_cv_std_mm"] = float(lece_cv_std)
    entry["loocv_lece_mm"] = float(loocv_lece) if loocv_lece is not None else None
    entry["n_landmarks"] = int(n_landmarks)
    entry["n_cases"] = int(n_cases)
    entry["bin_width_mm"] = float(bin_width_mm)
    entry["fit_seed"] = int(fit_seed)
    entry["fit_method"] = "grid_lece"
    entry["fit_date"] = datetime.date.today().isoformat()
    entry["status"] = "calibrated"
    # write atomically
    tmp_path = registry_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(reg, f, indent=2)
    os.replace(tmp_path, registry_path)
    return True


if __name__ == "__main__":
    main()
