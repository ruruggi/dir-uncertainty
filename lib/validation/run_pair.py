"""
validation/run_pair.py — Full pipeline on one Abdominal-DIR-QA pair.

Steps:
    1. Load fixed/moving + landmarks from case_dir (.mat or .nii)
    2. Pre-resample moving onto the fixed grid (for heuristic and propagation)
    3. Compute the DVF μ with ANTs SyN (core.registration.AntsSyNRegisterer)
    4. Compute σ with runner.compute_heuristic_uncertainty (heuristic+residual)
    5. Compute TRE, sigma_p, LECE per landmark
    6. Save: DVF as DICOM DSR, sigma as .npy, landmarks as .npz, plot as .png
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import numpy as np

# Allow running as a standalone script
if __package__ is None:
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    sys.path.insert(0, os.path.join(os.path.dirname(here), "dl_arch"))

import torch

from utils.scan import Scan
from utils.vector_field import VectorField
from core.registration import AntsSyNRegisterer
from core.dvf_dicom import write_dvf_dicom
from core.rigid_align import procrustes_rigid, matrix_to_params, save_rigid_json

from validation.load_abdominal_dir_qa import load_pair, AbdominalDIRQAPair
from validation.lece import (
    evaluate_landmarks,
    compute_lece,
    plot_reliability_diagram,
)


# Reuse the existing heuristic implementation
from runner import compute_heuristic_uncertainty


# ----------------------------------------------------------------------
# Procrustes prealign: transforms the moving image (and its landmarks) towards
# the fixed one via Kabsch SVD, then continues the pipeline as if fixed/moving
# were aligned. SyNRA acts on the deformable residual.
# ----------------------------------------------------------------------
def _save_itk_rigid_mat(path: str, M_xyz: np.ndarray, center_xyz) -> None:
    """Save M_xyz as an ITK Affine transform (.mat). ANTs convention:
    apply_transforms with whichtoinvert=[True] applies the inverse (because
    M_xyz is moving->fixed; ANTs expects it as fixed->moving)."""
    import ants
    R = M_xyz[:3, :3]
    c = np.asarray(center_xyz, dtype=np.float64)
    # ITK: y = R*(x - c) + c + t  =>  M_xyz[:3,3] = c - R*c + t  =>  t = M_xyz[:3,3] - c + R*c
    t = M_xyz[:3, 3] - c + R @ c

    tx = ants.create_ants_transform(
        transform_type="AffineTransform", precision="float", dimension=3,
    )
    params = np.concatenate([R.flatten().astype(np.float32), t.astype(np.float32)])
    tx.set_parameters(params)
    tx.set_fixed_parameters(c.astype(np.float32))
    ants.write_transform(tx, path)


def _apply_procrustes_to_pair(pair: AbdominalDIRQAPair, out_dir: str,
                              save_artifacts: bool = True
                              ) -> tuple:
    """Compute M = Kabsch SVD(lm_fixed_world, lm_moving_world), apply it to the
    moving image (resampling onto the fixed grid via ANTs) and to the moving
    landmarks. Returns (pair_prealigned, M_xyz, center_xyz).

    pair_prealigned has:
        - moving_arr resampled onto the fixed grid (post-rigid)
        - moving_affine = fixed_affine
        - spacing_moving = spacing_fixed
        - lm_moving_world  = M @ lm_moving_world_original
        - lm_moving_voxel  = inv(fixed_affine) @ lm_moving_world (in fixed grid)
        - lm_fixed_*   unchanged
    """
    import ants
    from dataclasses import replace

    if pair.lm_fixed_world is None or pair.lm_moving_world is None:
        raise ValueError("Procrustes prealign requires lm_fixed_world and "
                         "lm_moving_world (annotated landmark pair).")

    M_xyz = procrustes_rigid(pair.lm_fixed_world, pair.lm_moving_world)
    center_xyz = tuple(pair.lm_fixed_world.mean(axis=0).tolist())

    # 1) Save rigid.mat + reference 6-DOF JSON (for reproducibility)
    rigid_mat_path = os.path.join(out_dir, "rigid.mat")
    if save_artifacts:
        os.makedirs(out_dir, exist_ok=True)
        _save_itk_rigid_mat(rigid_mat_path, M_xyz, center_xyz)
        try:
            params_pro = matrix_to_params(M_xyz, center_world_mm=center_xyz)
            ref_json = os.path.join(out_dir, "rigid_align_procrustes.json")
            params_for_save = dict(params_pro)
            params_for_save["center_world_mm"] = list(center_xyz)
            save_rigid_json(ref_json, params_for_save,
                            fixed_path="(in-process)", moving_path="(in-process)")
        except Exception:
            pass
    else:
        # Even without persistent saving, ANTs needs the .mat file on disk.
        os.makedirs(out_dir, exist_ok=True)
        _save_itk_rigid_mat(rigid_mat_path, M_xyz, center_xyz)

    # 2) Build fixed/moving ANTsImage with correct origin/direction
    def _origin_dir(A, spacing):
        if A is None:
            return None, None
        A = np.asarray(A, dtype=np.float64)
        origin = tuple(float(v) for v in A[:3, 3])
        s = np.array(spacing, dtype=np.float64)
        R = A[:3, :3] / s[None, :]
        return origin, R

    f_o, f_d = _origin_dir(pair.fixed_affine, pair.spacing_fixed)
    m_o, m_d = _origin_dir(pair.moving_affine, pair.spacing_moving)
    f_kwargs = {"spacing": pair.spacing_fixed}
    if f_o is not None: f_kwargs["origin"] = f_o
    if f_d is not None: f_kwargs["direction"] = f_d
    m_kwargs = {"spacing": pair.spacing_moving}
    if m_o is not None: m_kwargs["origin"] = m_o
    if m_d is not None: m_kwargs["direction"] = m_d

    fixed_a = ants.from_numpy(pair.fixed_arr.astype(np.float32), **f_kwargs)
    moving_a = ants.from_numpy(pair.moving_arr.astype(np.float32), **m_kwargs)

    moving_prealigned_a = ants.apply_transforms(
        fixed=fixed_a, moving=moving_a,
        transformlist=[rigid_mat_path],
        whichtoinvert=[True],
        interpolator="linear",
    )
    moving_prealigned_arr = moving_prealigned_a.numpy().astype(np.float32, copy=False)

    # 3) Transform the moving landmarks via M (world -> world)
    homo = np.column_stack([pair.lm_moving_world,
                            np.ones(len(pair.lm_moving_world))])
    lm_moving_world_pre = (M_xyz @ homo.T).T[:, :3]

    # 4) Voxel coordinates in the fixed grid
    inv_aff = np.linalg.inv(pair.fixed_affine) if pair.fixed_affine is not None \
        else np.eye(4) / np.array([*pair.spacing_fixed, 1.0])
    homo_w = np.column_stack([lm_moving_world_pre,
                              np.ones(len(lm_moving_world_pre))])
    lm_moving_voxel_pre = (inv_aff @ homo_w.T).T[:, :3]

    pair_pre = replace(
        pair,
        moving_arr=moving_prealigned_arr,
        spacing_moving=pair.spacing_fixed,
        moving_affine=pair.fixed_affine,
        lm_moving_world=lm_moving_world_pre,
        lm_moving_voxel=lm_moving_voxel_pre,
    )
    return pair_pre, M_xyz, center_xyz


# ----------------------------------------------------------------------
# Utility: resample moving onto the fixed grid via ANTs
# ----------------------------------------------------------------------
def _resample_moving_to_fixed(
    fixed_arr: np.ndarray,
    moving_arr: np.ndarray,
    spacing_fixed,
    spacing_moving,
    fixed_affine=None,
    moving_affine=None,
) -> np.ndarray:
    """Wrapper around ants.resample_image_to_target. Returns moving_arr_aligned
    with shape == fixed_arr.shape, using linear interpolation and the identity
    as transform.

    If fixed_affine / moving_affine are given, origin and direction matrix
    are propagated to `ants.from_numpy` (needed for NIfTI datasets with
    X-flip or non-zero origin, e.g. Lung/Liver-DIR-QA). Without them, ANTs
    assumes origin (0,0,0) + identity direction, introducing offsets of a few
    mm on Lung and tens of mm on Liver.
    """
    import ants

    def _origin_dir(A, spacing):
        if A is None:
            return None, None
        A = np.asarray(A, dtype=np.float64)
        origin = tuple(float(v) for v in A[:3, 3])
        s = np.array(spacing, dtype=np.float64)
        R = A[:3, :3] / s[None, :]
        return origin, R

    f_o, f_d = _origin_dir(fixed_affine, spacing_fixed)
    m_o, m_d = _origin_dir(moving_affine, spacing_moving)
    f_kwargs = {"spacing": spacing_fixed}
    if f_o is not None: f_kwargs["origin"] = f_o
    if f_d is not None: f_kwargs["direction"] = f_d
    m_kwargs = {"spacing": spacing_moving}
    if m_o is not None: m_kwargs["origin"] = m_o
    if m_d is not None: m_kwargs["direction"] = m_d

    fixed_a = ants.from_numpy(fixed_arr.astype(np.float32), **f_kwargs)
    moving_a = ants.from_numpy(moving_arr.astype(np.float32), **m_kwargs)
    moving_resampled = ants.resample_image_to_target(
        moving_a, fixed_a, interp_type="linear"
    )
    return moving_resampled.numpy().astype(np.float32, copy=False)


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def run_one(
    case_dir: str,
    out_dir: str,
    case_id: str = None,
    type_of_transform: str = "SyN",
    reg_iterations=(100, 70, 50, 20),
    syn_metric: str = "CC",
    syn_sampling: int = 4,
    flow_sigma: float = 3.0,
    grad_step: float = 0.25,
    heuristic_w: float = 0.7,
    heuristic_jac_clip: float = 0.95,
    heuristic_ratio_clip: float = 0.95,
    heuristic_use_minmax: bool = True,
    S_scaling: float = 1.0,           # scale factor from calibration (post-batch)
    prealign: str = "none",            # "none" | "procrustes"
    save_dicom: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> dict:
    """Run the full pipeline on one pair. Returns a dict of results."""
    os.makedirs(out_dir, exist_ok=True)

    def log(msg):
        if verbose:
            print(msg, flush=True)

    t0 = time.time()
    log(f"[{case_id or os.path.basename(case_dir)}] === START ===")
    log(f"  case_dir: {case_dir}")
    log(f"  out_dir : {out_dir}")

    # --- 1. Load --------------------------------------------------------------
    pair: AbdominalDIRQAPair = load_pair(case_dir, case_id=case_id)
    log(f"  loaded fixed {pair.fixed_arr.shape} / moving {pair.moving_arr.shape}, "
        f"{pair.n_landmarks} landmarks")

    # --- 1b. Procrustes prealign (optional) ---------------------------------
    # If prealign='procrustes', compute Kabsch SVD on the world landmarks and apply
    # the rigid transform to the moving image (resampling onto the fixed grid). All
    # downstream steps (resample, SyN, heuristic, eval) work on the prealigned
    # moving image; the reported initial TRE is the post-Procrustes one (residual).
    rigid_M = None
    initial_tre_raw = None
    if prealign == "procrustes":
        log("  Procrustes prealign (Kabsch SVD on annotated landmarks)...")
        # TRE pre-Procrustes (for logging/diagnosis: comparing with the post-prealign
        # value shows the effectiveness of the rigid step)
        initial_tre_raw = float(np.sqrt(np.mean(
            ((pair.lm_fixed_world - pair.lm_moving_world) ** 2).sum(axis=1)
        )))
        log(f"    initial RMSE TRE (pre-Procrustes): {initial_tre_raw:.2f} mm")
        pair, rigid_M, _center = _apply_procrustes_to_pair(pair, out_dir,
                                                           save_artifacts=True)
        post_pre_tre = float(np.sqrt(np.mean(
            ((pair.lm_fixed_world - pair.lm_moving_world) ** 2).sum(axis=1)
        )))
        log(f"    rigid-only RMSE TRE (post-Procrustes): {post_pre_tre:.2f} mm")
    elif prealign != "none":
        raise ValueError(f"prealign='{prealign}' not supported (use 'none' or 'procrustes').")

    # --- 2. Resample moving onto the fixed grid (for heuristic) --------------
    log("  resampling moving -> fixed grid via ANTs identity...")
    moving_aligned = _resample_moving_to_fixed(
        pair.fixed_arr, pair.moving_arr,
        pair.spacing_fixed, pair.spacing_moving,
        fixed_affine=getattr(pair, "fixed_affine", None),
        moving_affine=getattr(pair, "moving_affine", None),
    )
    log(f"    moving_aligned shape: {moving_aligned.shape}")

    # --- 3. Wrap in Scan and ANTs SyN registration ----------------------------
    vtw = np.identity(4)
    vtw[0, 0] = pair.spacing_fixed[0]
    vtw[1, 1] = pair.spacing_fixed[1]
    vtw[2, 2] = pair.spacing_fixed[2]
    fixed_scan = Scan(array=pair.fixed_arr[None, ...], vtwMatrix=vtw)
    vtw_m = vtw.copy()
    vtw_m[0, 0] = pair.spacing_moving[0]
    vtw_m[1, 1] = pair.spacing_moving[1]
    vtw_m[2, 2] = pair.spacing_moving[2]
    moving_scan = Scan(array=pair.moving_arr[None, ...], vtwMatrix=vtw_m)

    log(f"  ANTs registration ({type_of_transform})...")
    t_reg = time.time()
    registerer = AntsSyNRegisterer(
        type_of_transform=type_of_transform,
        reg_iterations=reg_iterations,
        syn_metric=syn_metric,
        syn_sampling=syn_sampling,
        flow_sigma=flow_sigma,
        grad_step=grad_step,
        verbose=False,
    )
    vf: VectorField = registerer.register(
        fixed_scan, moving_scan,
        progress_callback=(lambda m, p: log(f"    [{p:.2f}] {m}")) if verbose else None,
    )
    reg_time = time.time() - t_reg
    log(f"    registration done in {reg_time:.1f}s")

    dvf_mm = vf.getPixelArray()
    if hasattr(dvf_mm, "cpu"):
        dvf_mm = dvf_mm.cpu().numpy()
    log(f"    DVF range: [{dvf_mm.min():.2f}, {dvf_mm.max():.2f}] mm")

    # --- 4. Heuristic σ -------------------------------------------------------
    log("  computing heuristic+residual sigma...")
    t_h = time.time()
    fixed_t = torch.from_numpy(pair.fixed_arr[None, ...].astype(np.float32))
    moving_t = torch.from_numpy(moving_aligned[None, ...].astype(np.float32))
    dvf_t = torch.from_numpy(np.ascontiguousarray(dvf_mm.astype(np.float32)))
    if device == "cuda" and torch.cuda.is_available():
        fixed_t = fixed_t.cuda()
        moving_t = moving_t.cuda()
        dvf_t = dvf_t.cuda()

    # voxel_spacing in (sz, sy, sx) order, as required by the signature of
    # compute_heuristic_uncertainty (see runner.py:91, where dvf_vox[0]/=sz,
    # dvf_vox[1]/=sy, dvf_vox[2]/=sx).
    sx, sy, sz = pair.spacing_fixed
    sigma_1ch = compute_heuristic_uncertainty(
        fixed_ct=fixed_t,
        moving_ct=moving_t,
        dvf_mm=dvf_t,
        voxel_spacing=(sz, sy, sx),
        w_res_grad=heuristic_w,
        w_jac=1.0 - heuristic_w,
        jac_clip_percentile=heuristic_jac_clip,
        ratio_clip_percentile=heuristic_ratio_clip,
        use_minmax_norm=heuristic_use_minmax,
        global_scale=1.0,
        suppression_alpha=1.0,
        debug=False,
    )
    sigma_np_raw = sigma_1ch.squeeze().detach().cpu().numpy().astype(np.float32)
    h_time = time.time() - t_h
    log(f"    sigma raw shape {sigma_np_raw.shape}, range "
        f"[{sigma_np_raw.min():.3f}, {sigma_np_raw.max():.3f}], computed in {h_time:.1f}s")

    # Optional calibration via a global factor c (post-batch).
    # Default c=1.0 -> no rescaling (sigma stays in [0,1], dimensionless).
    # Once the batch is complete and calibrate.py has produced c, the run can
    # be repeated with S_scaling=c to obtain sigma in mm, comparable with TRE.
    if S_scaling != 1.0:
        log(f"    applying calibration scale S = {S_scaling:.4f}")
        sigma_np = (sigma_np_raw * float(S_scaling)).astype(np.float32)
    else:
        sigma_np = sigma_np_raw

    # --- 5. Eval per-landmark + LECE ------------------------------------------
    log("  per-landmark eval + LECE...")
    eval_res = evaluate_landmarks(
        dvf_mm=dvf_mm,
        sigma_mm=sigma_np,
        spacing_fixed=pair.spacing_fixed,
        lm_fixed_voxel=pair.lm_fixed_voxel,
        lm_moving_world_true=pair.lm_moving_world,
        case_id=pair.case_id,
        fixed_affine=getattr(pair, "fixed_affine", None),
    )
    initial_tre = float(np.sqrt(np.mean(((pair.lm_fixed_world - pair.lm_moving_world) ** 2).sum(axis=1))))
    final_tre = float(np.sqrt(np.mean(eval_res.tre_mm ** 2)))
    median_tre = float(np.median(eval_res.tre_mm))
    log(f"    initial RMSE TRE: {initial_tre:.2f} mm")
    log(f"    final   RMSE TRE: {final_tre:.2f} mm")
    log(f"    median  TRE     : {median_tre:.2f} mm")
    log(f"    sigma_p median  : {float(np.median(eval_res.sigma_p_mm)):.2f} mm")

    lece, diag = compute_lece(eval_res.tre_mm, eval_res.sigma_p_mm, bin_width=0.25)
    log(f"    LECE = {lece:.3f} mm")

    # --- 6. Save artefacts ----------------------------------------------------
    # Always save the RAW σ map (in [0,1]); if S_scaling != 1, also save the calibrated one.
    np.save(os.path.join(out_dir, "sigma_raw.npy"), sigma_np_raw)
    if S_scaling != 1.0:
        np.save(os.path.join(out_dir, "sigma_calibrated.npy"), sigma_np)
    np.save(os.path.join(out_dir, "dvf_mu.npy"), dvf_mm.astype(np.float32))

    # Also save the RAW sigma_p at the landmark, separate from the scaled one,
    # so that calibrate.py can work on (tre_mm, sigma_p_mm_raw) alone
    # without rerunning ANTs.
    sigma_p_raw = eval_res.sigma_p_mm if S_scaling == 1.0 else \
        eval_res.sigma_p_mm / float(S_scaling)

    np.savez(
        os.path.join(out_dir, "landmarks.npz"),
        case_id=pair.case_id,
        lm_fixed_voxel=pair.lm_fixed_voxel,
        lm_moving_voxel=pair.lm_moving_voxel,
        lm_fixed_world=pair.lm_fixed_world,
        lm_moving_world=pair.lm_moving_world,
        lm_moving_world_pred=eval_res.lm_moving_world_pred,
        tre_mm=eval_res.tre_mm,
        sigma_p_mm=eval_res.sigma_p_mm,            # after S_scaling, if applied
        sigma_p_mm_raw=sigma_p_raw,                # before scaling (for calibrate.py)
        S_scaling_applied=float(S_scaling),
        spacing_fixed=np.array(pair.spacing_fixed),
        spacing_moving=np.array(pair.spacing_moving),
    )

    if save_dicom:
        dvf_dicom_path = os.path.join(out_dir, "dvf_mu.dcm")
        try:
            write_dvf_dicom(vf, dvf_dicom_path,
                            method_metadata=registerer.get_metadata())
            log(f"    DVF saved to {dvf_dicom_path} "
                f"({os.path.getsize(dvf_dicom_path) / 1024 / 1024:.1f} MB)")
        except Exception as e:
            log(f"    [WARNING] DICOM save failed: {e}")

    plot_path = os.path.join(out_dir, "reliability_diagram.png")
    plot_reliability_diagram(
        diag, output_path=plot_path,
        title=f"Abdominal-DIR-QA — {pair.case_id} (LECE = {lece:.3f} mm)",
        label="heuristic+residual",
    )
    log(f"    plot saved to {plot_path}")

    summary = {
        "case_id": pair.case_id,
        "case_dir": case_dir,
        "n_landmarks": pair.n_landmarks,
        "initial_rmse_tre_mm": initial_tre,
        "final_rmse_tre_mm": final_tre,
        "median_tre_mm": median_tre,
        "lece_mm": lece,
        "S_scaling_applied": float(S_scaling),
        "registration_time_s": reg_time,
        "heuristic_time_s": h_time,
        "total_time_s": time.time() - t0,
        "prealign": prealign,
        "registration_metadata": registerer.get_metadata(),
        "heuristic_params": {
            "w_res_grad": heuristic_w,
            "w_jac": 1.0 - heuristic_w,
            "jac_clip_percentile": heuristic_jac_clip,
            "ratio_clip_percentile": heuristic_ratio_clip,
            "use_minmax_norm": heuristic_use_minmax,
        },
    }
    if rigid_M is not None:
        summary["procrustes"] = {
            "initial_rmse_tre_pre_procrustes_mm": initial_tre_raw,
            "M_xyz_4x4": rigid_M.tolist(),
        }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"  summary.json saved.")
    log(f"[{pair.case_id}] === DONE in {summary['total_time_s']:.1f}s ===")
    return summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_dir", required=True,
                        help="Folder containing the pair (.mat or .nii)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--case_id", default=None)
    parser.add_argument("--type_of_transform", default="SyN")
    parser.add_argument("--no_dicom", action="store_true",
                        help="Disable saving the DVF as DICOM")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced iterations (40,20,10) for a quick smoke test")
    parser.add_argument("--S_scaling", "--c_scaling", type=float, default=1.0,
                        dest="S_scaling",
                        help="Global calibration factor for sigma "
                        "(default 1.0 = no calibration, raw sigma in [0,1]). "
                        "--c_scaling is a deprecated alias of --S_scaling.")
    parser.add_argument("--prealign", choices=["none", "procrustes"],
                        default="none",
                        help="Rigid pre-alignment before SyN. "
                        "'procrustes' = Kabsch SVD on the world landmarks (closed-form, "
                        "auto-batchable; requires an annotated dataset). 'none' = no "
                        "prealign (default).")
    args = parser.parse_args()

    iters = (40, 20, 10) if args.quick else (100, 70, 50, 20)
    run_one(
        case_dir=args.case_dir,
        out_dir=args.out_dir,
        case_id=args.case_id,
        type_of_transform=args.type_of_transform,
        reg_iterations=iters,
        S_scaling=args.S_scaling,
        prealign=args.prealign,
        save_dicom=not args.no_dicom,
        verbose=True,
    )


if __name__ == "__main__":
    _main()
