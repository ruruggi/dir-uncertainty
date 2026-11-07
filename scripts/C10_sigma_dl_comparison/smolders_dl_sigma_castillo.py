"""
smolders_dl_sigma_castillo.py — Controlled A/B test, σ-method only.

For each of the 4 engines (SyNRA, ConvexAdam, uniGradICON, Elastix) on the 10
DIR-LAB Castillo patients, apply the SMOLDERS UNSUPERVISED MODEL (L20S3-4
in TrainedModels/) to the already computed DVF. Same DVF, Smolders DL σ vs our
heuristic+residual σ -> the LECE difference is attributable to the σ-method alone.

Aggregation: per engine, pool the (TRE, σ_p) of all 10 cases -> global
fit of f -> global LECE (matching the Smolders 2023 §3.2.1 methodology).

Output:
    _castillo_smolders_dl_test/per_case.json — per-landmark/case results
    _castillo_smolders_dl_test/aggregate.json — global LECE per engine
"""
from __future__ import annotations

from dirunc_paths import DATASETS  # data roots, see README
import os
import sys
import json
import time
import tempfile
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PAPER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# utils/, dl/, dl_arch/, core/ live in the sibling DIR_Uncertainty_v2 repo —
# the PAPER repo holds only the validation/ wrapper scripts that drive the
# cohort analyses. Resolve the code root explicitly.
CODE_ROOT = os.path.normpath(os.path.join(PAPER_ROOT, "..", "DIR_Uncertainty_v2"))
sys.path.insert(0, CODE_ROOT)
sys.path.insert(0, os.path.join(CODE_ROOT, "dl_arch"))
sys.path.insert(0, PAPER_ROOT)   # for validation.* sub-modules in this repo
ROOT = CODE_ROOT

from utils.scan import Scan
from utils.vector_field import VectorField
from validation.load_abdominal_dir_qa import load_pair, discover_cases
from validation.run_pair import _apply_procrustes_to_pair
from validation.lece import compute_lece
from validation.calibrate import fit_c_grid
from dl.model_loader import load_registerer

CASTILLO_NIFTI = f"{DATASETS}/Castillo-DIR-LAB/nifti"
CASTILLO_BASE = f"{DATASETS}/Castillo-DIR-LAB"
ENGINES = [
    ("synra",       "SyNRA"),
    ("convexadam",  "ConvexAdam"),
    ("unigradicon", "uniGradICON"),
    ("elastix",     "Elastix"),
]
MODEL_NAME = "L20S3-4"


def _sample_sigma_magnitude(sigma_arr_3xyz, lm_fixed_voxel):
    """Sample sigma magnitude (Euclidean over 3 components) at landmarks."""
    from scipy.ndimage import map_coordinates
    if hasattr(sigma_arr_3xyz, "cpu"):
        sigma_arr_3xyz = sigma_arr_3xyz.cpu().numpy()
    sigma_arr_3xyz = np.asarray(sigma_arr_3xyz, dtype=np.float32)
    if sigma_arr_3xyz.ndim != 4 or sigma_arr_3xyz.shape[0] != 3:
        raise ValueError(f"Expected (3, X, Y, Z), got {sigma_arr_3xyz.shape}")
    magnitude = np.sqrt(np.sum(sigma_arr_3xyz ** 2, axis=0))  # (X, Y, Z)
    coords = lm_fixed_voxel.T  # (3, N)
    sigma_p = map_coordinates(magnitude, coords, order=1,
                                mode="constant", cval=float(magnitude.mean()))
    return np.asarray(sigma_p, dtype=np.float64)


def _sample_tre_via_dvf(dvf_3xyz, pair_pre):
    """Compute per-landmark TRE: predict moving_world via DVF, compare with truth."""
    from scipy.ndimage import map_coordinates
    # DVF convention: (3, X, Y, Z) [dz, dy, dx]; obtain (X, Y, Z, 3) [dx, dy, dz]
    dvf_xyzv = np.transpose(dvf_3xyz[::-1], (1, 2, 3, 0))
    coords = pair_pre.lm_fixed_voxel.T
    disp = np.zeros_like(pair_pre.lm_fixed_voxel, dtype=np.float64)
    for c in range(3):
        disp[:, c] = map_coordinates(dvf_xyzv[..., c], coords, order=1,
                                       mode="constant", cval=0.0)
    lm_pred = pair_pre.lm_fixed_world + disp
    tre = np.linalg.norm(lm_pred - pair_pre.lm_moving_world, axis=1)
    return tre


def run_one(engine_key, case_id, registerer, tmp_root):
    pair_orig = load_pair(CASTILLO_NIFTI, case_id=case_id)
    pair_pre, _, _ = _apply_procrustes_to_pair(
        pair_orig, out_dir=os.path.join(tmp_root, f"{engine_key}_{case_id}"),
        save_artifacts=False,
    )
    vtw = np.identity(4)
    vtw[0, 0] = pair_pre.spacing_fixed[0]
    vtw[1, 1] = pair_pre.spacing_fixed[1]
    vtw[2, 2] = pair_pre.spacing_fixed[2]
    fixed_scan = Scan(array=pair_pre.fixed_arr[None, ...], vtwMatrix=vtw)
    moving_scan = Scan(array=pair_pre.moving_arr[None, ...], vtwMatrix=vtw)

    dvf_path = os.path.join(CASTILLO_BASE, f"Results_{engine_key}_procrustes",
                              case_id, "dvf_mu.npy")
    dvf_arr = np.load(dvf_path).astype(np.float32)
    dvf = VectorField(array=dvf_arr, vtwMatrix=vtw, inMM=True)

    t0 = time.perf_counter()
    _, prob = registerer.process(fixed_scan, moving_scan, dvf)
    inf_time = time.perf_counter() - t0
    sigma_arr = prob.getVectorFieldUncertainty()
    if hasattr(sigma_arr, "cpu"):
        sigma_arr = sigma_arr.cpu().numpy()
    sigma_arr = np.asarray(sigma_arr, dtype=np.float32)

    sigma_p = _sample_sigma_magnitude(sigma_arr, pair_pre.lm_fixed_voxel)
    tre = _sample_tre_via_dvf(dvf_arr, pair_pre)
    return tre, sigma_p, inf_time


def main():
    out_dir = os.path.join(ROOT, "_castillo_smolders_dl_test")
    os.makedirs(out_dir, exist_ok=True)
    tmp_root = tempfile.mkdtemp(prefix="smolders_test_")
    print(f"=== Smolders DL Unsupervised σ on Castillo (4 engines × 10 cases) ===")
    print(f"  model: TrainedModels/{MODEL_NAME}")
    print(f"  out_dir: {out_dir}\n")

    cases = sorted(discover_cases(CASTILLO_NIFTI),
                    key=lambda c: int(c[0].replace("case", "")))
    case_ids = [c[0] for c in cases]
    print(f"  cases: {case_ids}\n")

    print("Loading Smolders Unsupervised registerer...")
    registerer = load_registerer(model_name=MODEL_NAME)
    print("  done.\n")

    per_case = []
    pooled = {label: {"tre": [], "sigma_p": []} for _, label in ENGINES}

    for engine_key, engine_label in ENGINES:
        print(f"--- {engine_label} ---")
        for cid in case_ids:
            try:
                tre, sigma_p, inf_t = run_one(
                    engine_key, cid, registerer, tmp_root)
                pooled[engine_label]["tre"].append(tre)
                pooled[engine_label]["sigma_p"].append(sigma_p)
                rec = {"engine": engine_label, "case_id": cid,
                       "n_lm": int(len(tre)),
                       "rmse_tre_mm": float(np.sqrt(np.mean(tre ** 2))),
                       "sigma_p_smolders_mean": float(sigma_p.mean()),
                       "sigma_p_smolders_std": float(sigma_p.std(ddof=1)),
                       "inference_time_s": inf_t}
                per_case.append(rec)
                print(f"  {cid}: n={rec['n_lm']:<3}, "
                      f"RMSE_TRE={rec['rmse_tre_mm']:.2f}, "
                      f"σ_p_smolders mean={rec['sigma_p_smolders_mean']:.2f}±{rec['sigma_p_smolders_std']:.2f} "
                      f"({inf_t:.1f}s)")
            except Exception as e:
                import traceback; traceback.print_exc()
                per_case.append({"engine": engine_label, "case_id": cid,
                                  "error": str(e)})
                print(f"  {cid}: FAIL {e}")

    # Aggregate per engine: pool all (tre, sigma_p) and compute the global LECE
    print(f"\n=== Aggregate (Smolders methodology: pooled landmarks per engine) ===")
    aggregate = []
    for _, label in ENGINES:
        if not pooled[label]["tre"]:
            continue
        tre_all = np.concatenate(pooled[label]["tre"])
        sigma_all = np.concatenate(pooled[label]["sigma_p"])
        # Fit f globally on the pooled landmarks
        f_global, _ = fit_c_grid(tre_all, sigma_all, bin_width=0.25)
        sigma_cal = f_global * sigma_all
        lece, _ = compute_lece(tre_all, sigma_cal, bin_width=0.25)
        agg = {
            "engine": label,
            "n_landmarks_pooled": int(tre_all.size),
            "n_cases": len(pooled[label]["tre"]),
            "f_smolders_dl": float(f_global),
            "lece_smolders_dl_mm": float(lece),
        }
        aggregate.append(agg)
        print(f"  {label:<13}: n_lm={agg['n_landmarks_pooled']:>5} "
              f"f={agg['f_smolders_dl']:.2f} "
              f"LECE_smolders_dl={agg['lece_smolders_dl_mm']:.3f} mm")

    with open(os.path.join(out_dir, "per_case.json"), "w") as f:
        json.dump(per_case, f, indent=2, default=str)
    with open(os.path.join(out_dir, "aggregate.json"), "w") as f:
        json.dump(aggregate, f, indent=2)

    # Cleanup tmp
    import shutil; shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"\nResults in: {out_dir}")


if __name__ == "__main__":
    main()
