"""
validation/run_batch_convexadam.py — Batch ConvexAdam on Abdominal-DIR-QA.

Sequential single-GPU version of the SyNRA batch: since each ConvexAdam
case takes 5-15 s on a GV100, process-level parallelism makes no
sense (spawn overhead > registration time). The batch runs
in a single process, on a single configurable GPU.

Per-case pipeline:
    1. Load (load_pair) + resampling of moving onto the fixed grid (ANTs identity)
    2. DVF μ via ConvexAdamRegisterer (MIND-SSC + coupled convex + Adam IO)
    3. Heuristic σ via runner.compute_heuristic_uncertainty (default w=0.7)
    4. Landmark eval (TRE + sigma_p) + LECE (Smolders Eq. 9, bin 0.25 mm)
    5. Saving: dvf_mu.dcm (DICOM DSR), sigma_raw.npy, landmarks.npz,
       reliability_diagram.png, summary.json — same schema as run_pair.py
       so that calibrate.py can process both batches.

Operational notes:
    - The output goes to a DIFFERENT directory from the SyNRA batch, so
      the two per-case results coexist and calibrate.py can be launched
      separately with --engine ants_syn / convex_adam.
    - Default device cuda:1 to avoid contention with any heuristic steps
      of the SyNRA batch running on cuda:0.
    - Resume: skips cases with summary.json + n_landmarks > 0 (like run_batch).

Example:
    venv/Scripts/python.exe validation/run_batch_convexadam.py \
        --root_dir "<DATASETS>/AbdominalDIR-QA/Annotated Image Pairs/nifti_format" \
        --out_root "<DATASETS>/AbdominalDIR-QA/Annotated Image Pairs/Results_convexadam" \
        --device cuda:1
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import numpy as np

# CUDA allocator: limits the maximum block size of the caching allocator
# to reduce fragmentation on large volumes (Lung-DIR-QA case18+ at 144M
# voxels would otherwise saturate a 32 GB GV100). max_split_size_mb is
# compatible with PyTorch 2.0 (expandable_segments requires 2.1+). It must
# be set BEFORE importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

if __package__ is None or __package__ == "":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    sys.path.insert(0, os.path.join(os.path.dirname(here), "dl_arch"))

import torch

from utils.scan import Scan
from utils.vector_field import VectorField
from core.registration import ConvexAdamRegisterer
from core.dvf_dicom import write_dvf_dicom

from validation.load_abdominal_dir_qa import discover_cases, load_pair, AbdominalDIRQAPair
from validation.lece import evaluate_landmarks, compute_lece, plot_reliability_diagram
from runner import compute_heuristic_uncertainty
from validation.run_pair import _apply_procrustes_to_pair  # Procrustes helper


def _resample_moving_to_fixed(fixed_arr, moving_arr,
                              spacing_fixed, spacing_moving,
                              fixed_affine=None, moving_affine=None):
    """Identity-resample moving onto the fixed grid via ANTs.

    If fixed_affine / moving_affine are provided (NIfTI datasets with
    non-zero origin or non-identity direction, e.g. Liver/Lung-DIR-QA), the
    origin and direction matrix are propagated to `ants.from_numpy` to
    resample correctly in world space. Without this, ANTs would
    assume origin (0,0,0) and identity direction, introducing offsets of
    a few mm on typical NIfTI datasets (irrelevant on Abdominal-DIR-QA,
    which already has a trivial affine, critical on Lung/Liver-DIR-QA).
    """
    import ants

    def _origin_direction_from_affine(A, spacing):
        if A is None:
            return None, None
        A = np.asarray(A, dtype=np.float64)
        origin = tuple(float(v) for v in A[:3, 3])
        spacing_arr = np.array(spacing, dtype=np.float64)
        # R = A[:3,:3] / spacing (per column). For orthogonal affines it is the
        # direction matrix (rotation only).
        R = A[:3, :3] / spacing_arr[None, :]
        return origin, R

    f_origin, f_direction = _origin_direction_from_affine(fixed_affine, spacing_fixed)
    m_origin, m_direction = _origin_direction_from_affine(moving_affine, spacing_moving)

    f_kwargs = {"spacing": spacing_fixed}
    if f_origin is not None: f_kwargs["origin"] = f_origin
    if f_direction is not None: f_kwargs["direction"] = f_direction
    fixed_a = ants.from_numpy(fixed_arr.astype(np.float32), **f_kwargs)

    m_kwargs = {"spacing": spacing_moving}
    if m_origin is not None: m_kwargs["origin"] = m_origin
    if m_direction is not None: m_kwargs["direction"] = m_direction
    moving_a = ants.from_numpy(moving_arr.astype(np.float32), **m_kwargs)

    moving_resampled = ants.resample_image_to_target(
        moving_a, fixed_a, interp_type="linear"
    )
    return moving_resampled.numpy().astype(np.float32, copy=False)


def _build_scans(pair: AbdominalDIRQAPair, moving_aligned: np.ndarray):
    vtw = np.identity(4)
    vtw[0, 0] = pair.spacing_fixed[0]
    vtw[1, 1] = pair.spacing_fixed[1]
    vtw[2, 2] = pair.spacing_fixed[2]
    fixed_scan = Scan(array=pair.fixed_arr[None, ...], vtwMatrix=vtw)
    moving_scan = Scan(array=moving_aligned[None, ...], vtwMatrix=vtw)
    return fixed_scan, moving_scan


def run_one_convexadam(
    case_id: str,
    case_dir: str,
    out_dir: str,
    device: str = "cuda:1",
    heuristic_w: float = 0.7,
    heuristic_jac_clip: float = 0.95,
    heuristic_ratio_clip: float = 0.95,
    heuristic_use_minmax: bool = True,
    S_scaling: float = 1.0,
    prealign: str = "none",
    save_dicom: bool = True,
    cvx_kwargs: dict = None,
    similarity_metric: str = "mind_ssc",
    similarity_kwargs: dict = None,
    verbose: bool = True,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    def log(msg):
        if verbose:
            print(msg, flush=True)

    t0 = time.time()
    log(f"[{case_id}] === START ===")
    log(f"  case_dir: {case_dir}")
    log(f"  out_dir : {out_dir}")

    # --- 1. Load --------------------------------------------------------------
    pair = load_pair(case_dir, case_id=case_id)
    log(f"  loaded fixed {pair.fixed_arr.shape} / moving {pair.moving_arr.shape}, "
        f"{pair.n_landmarks} landmarks")

    # --- 1b. Procrustes prealign (optional) ----------------------------------
    rigid_M = None
    initial_tre_raw = None
    if prealign == "procrustes":
        log("  Procrustes prealign (Kabsch SVD on annotated landmarks)...")
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

    # --- 2. Resample moving -> fixed grid ------------------------------------
    log("  resampling moving -> fixed grid via ANTs identity...")
    moving_aligned = _resample_moving_to_fixed(
        pair.fixed_arr, pair.moving_arr,
        pair.spacing_fixed, pair.spacing_moving,
        fixed_affine=getattr(pair, "fixed_affine", None),
        moving_affine=getattr(pair, "moving_affine", None),
    )
    log(f"    moving_aligned shape: {moving_aligned.shape}")

    # --- 3. ConvexAdam register ----------------------------------------------
    fixed_scan, moving_scan = _build_scans(pair, moving_aligned)

    log(f"  ConvexAdam registration on {device}...")
    t_reg = time.time()
    cvx_kwargs = cvx_kwargs or {}
    registerer = ConvexAdamRegisterer(device=device, verbose=False, **cvx_kwargs)
    vf = registerer.register(
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
    if device.startswith("cuda") and torch.cuda.is_available():
        # For the heuristic step we use the same device as for ConvexAdam
        dev = torch.device(device)
        fixed_t = fixed_t.to(dev)
        moving_t = moving_t.to(dev)
        dvf_t = dvf_t.to(dev)

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
        similarity_metric=similarity_metric,
        similarity_kwargs=similarity_kwargs,
    )
    sigma_np_raw = sigma_1ch.squeeze().detach().cpu().numpy().astype(np.float32)
    h_time = time.time() - t_h
    log(f"    sigma raw shape {sigma_np_raw.shape}, range "
        f"[{sigma_np_raw.min():.3f}, {sigma_np_raw.max():.3f}], computed in {h_time:.1f}s")

    if S_scaling != 1.0:
        log(f"    applying calibration scale S = {S_scaling:.4f}")
        sigma_np = (sigma_np_raw * float(S_scaling)).astype(np.float32)
    else:
        sigma_np = sigma_np_raw

    # --- 5. Landmark eval + LECE ----------------------------------------------
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

    # --- 6. Save artefacts (same schema as run_pair.py) ---------------------
    np.save(os.path.join(out_dir, "sigma_raw.npy"), sigma_np_raw)
    if S_scaling != 1.0:
        np.save(os.path.join(out_dir, "sigma_calibrated.npy"), sigma_np)
    np.save(os.path.join(out_dir, "dvf_mu.npy"), dvf_mm.astype(np.float32))

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
        sigma_p_mm=eval_res.sigma_p_mm,
        sigma_p_mm_raw=sigma_p_raw,
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
        title=f"Abdominal-DIR-QA — {pair.case_id} (ConvexAdam, LECE = {lece:.3f} mm)",
        label="heuristic+residual",
    )
    log(f"    plot saved to {plot_path}")

    summary = {
        "case_id": pair.case_id,
        "case_dir": case_dir,
        "engine": "convex_adam",
        "device": device,
        "n_landmarks": int(pair.n_landmarks),
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
            "similarity_metric": similarity_metric,
            "similarity_kwargs": dict(similarity_kwargs or {}),
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


def _is_done(case_out: str) -> bool:
    summary_path = os.path.join(case_out, "summary.json")
    landmarks_path = os.path.join(case_out, "landmarks.npz")
    if not (os.path.isfile(summary_path) and os.path.isfile(landmarks_path)):
        return False
    try:
        with open(summary_path) as f:
            s = json.load(f)
        return s.get("n_landmarks", 0) > 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run ConvexAdam + heuristic+residual on all cases of the "
                    "Abdominal-DIR-QA dataset, sequential single-GPU."
    )
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--cases", default=None,
                        help="Comma-separated list of case_ids. Default: all.")
    parser.add_argument("--device", default="cuda:1",
                        help="Torch device for ConvexAdam + heuristic (default cuda:1, "
                        "chosen to avoid competing with the ANTs batch on cuda:0).")
    parser.add_argument("--S_scaling", "--c_scaling", type=float, default=1.0,
                        dest="S_scaling",
                        help="Calibration factor (default 1.0). "
                        "--c_scaling is a deprecated alias.")
    parser.add_argument("--no_dicom", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prealign", choices=["none", "procrustes"],
                        default="none",
                        help="Rigid pre-alignment. 'procrustes' = Kabsch SVD "
                        "on the world landmarks (auto-batch, requires an annotated dataset). "
                        "'none' = no prealign.")
    # ConvexAdam params (defaults Heinrich/Hansen 2022 Learn2Reg)
    parser.add_argument("--lambda_weight", type=float, default=1.25)
    parser.add_argument("--grid_sp", type=int, default=6)
    parser.add_argument("--disp_hw", type=int, default=4)
    parser.add_argument("--selected_niter", type=int, default=80)
    parser.add_argument("--selected_smooth", type=int, default=0)
    parser.add_argument("--grid_sp_adam", type=int, default=2)
    parser.add_argument("--no_ic", action="store_true",
                        help="Disable inverse-consistency.")
    # Similarity metric for the residual term R (Eq. 10)
    parser.add_argument("--similarity_metric", default="mind_ssc",
                        choices=["abs_diff", "mind_ssc", "lncc"],
                        help="Metric for the residual term of Eq. 10. "
                        "Default 'mind_ssc' (Modality Independent Neighbourhood "
                        "Descriptor, Heinrich 2012/2013): 12-channel structural "
                        "residual, robust to wrong correspondences with chance "
                        "intensity matches; calibrated on 3 CT datasets with "
                        "LECE ≤ 1.3 mm. Opt-in: 'abs_diff' (legacy |I_m - I_f|, "
                        "CT/sCT only) or 'lncc' (Local NCC, same-sequence MR).")
    parser.add_argument("--mind_ssc_radius", type=int, default=1,
                        help="MIND-SSC patch radius (default 1 -> patch 3^3).")
    parser.add_argument("--mind_ssc_dilation", type=int, default=2,
                        help="MIND-SSC self-similarity context dilation (default 2 vox).")
    parser.add_argument("--lncc_win", type=int, default=5,
                        help="LNCC window size (cubic, default 5 vox).")
    args = parser.parse_args()

    cvx_kwargs = {
        "lambda_weight": args.lambda_weight,
        "grid_sp": args.grid_sp,
        "disp_hw": args.disp_hw,
        "selected_niter": args.selected_niter,
        "selected_smooth": args.selected_smooth,
        "grid_sp_adam": args.grid_sp_adam,
        "ic": not args.no_ic,
    }

    # kwargs for the chosen similarity metric
    if args.similarity_metric == "mind_ssc":
        sim_kwargs = {"radius": args.mind_ssc_radius,
                      "dilation": args.mind_ssc_dilation}
    elif args.similarity_metric == "lncc":
        sim_kwargs = {"win": args.lncc_win}
    else:
        sim_kwargs = {}

    all_cases = discover_cases(args.root_dir)
    if args.cases:
        wanted = set(c.strip() for c in args.cases.split(","))
        all_cases = [c for c in all_cases if c[0] in wanted]

    todo = []
    skipped = []
    for case_id, case_dir in all_cases:
        case_out = os.path.join(args.out_root, case_id)
        if not args.force and _is_done(case_out):
            skipped.append(case_id)
            continue
        todo.append((case_id, case_dir))

    print("=" * 70)
    print(f"Abdominal-DIR-QA batch run — ConvexAdam (sequential)")
    print("=" * 70)
    print(f"  root_dir   : {args.root_dir}")
    print(f"  out_root   : {args.out_root}")
    print(f"  device     : {args.device}")
    print(f"  cvx params : {cvx_kwargs}")
    print(f"  total cases: {len(all_cases)}")
    print(f"  skipped    : {len(skipped)}")
    print(f"  todo       : {len(todo)} ({[c[0] for c in todo[:5]]}{'...' if len(todo) > 5 else ''})")
    print(f"  S_scaling  : {args.S_scaling}")
    print(f"  similarity : {args.similarity_metric} (kwargs={sim_kwargs})")
    print(f"  save_dicom : {not args.no_dicom}")
    print()

    if not todo:
        print("Nothing to do.")
        return

    os.makedirs(args.out_root, exist_ok=True)
    t0 = time.time()
    completed = 0
    failed = []

    for case_id, case_dir in todo:
        case_out = os.path.join(args.out_root, case_id)
        try:
            s = run_one_convexadam(
                case_id=case_id,
                case_dir=case_dir,
                out_dir=case_out,
                device=args.device,
                S_scaling=args.S_scaling,
                prealign=args.prealign,
                save_dicom=not args.no_dicom,
                cvx_kwargs=cvx_kwargs,
                similarity_metric=args.similarity_metric,
                similarity_kwargs=sim_kwargs,
                verbose=True,
            )
            # Free the residual GPU memory between cases: without this call
            # fragmentation causes OOM after ~15-20 cases on Lung-DIR-QA volumes
            # (~70M voxels) even on a 32 GB GV100.
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            completed += 1
            elapsed = time.time() - t0
            avg = elapsed / completed
            eta = avg * (len(todo) - completed)
            print(f"[{completed}/{len(todo)}] OK {case_id:<10} "
                  f"(initial RMSE TRE {s['initial_rmse_tre_mm']:.2f} -> "
                  f"final {s['final_rmse_tre_mm']:.2f} mm in "
                  f"{s['total_time_s']:.0f}s, LECE {s['lece_mm']:.3f}) | "
                  f"elapsed {elapsed/60:.1f}min, ETA {eta/60:.1f}min",
                  flush=True)
        except Exception as e:
            import traceback
            completed += 1
            failed.append((case_id, str(e)))
            print(f"[{completed}/{len(todo)}] FAIL {case_id}: {e}", flush=True)
            print(traceback.format_exc(), flush=True)

    total_time = time.time() - t0
    print()
    print("=" * 70)
    print(f"Batch finished in {total_time/60:.1f} min")
    print(f"  completed: {completed - len(failed)}")
    print(f"  failed    : {len(failed)}")
    for cid, err in failed:
        print(f"    {cid}: {err}")

    aggregate_path = os.path.join(args.out_root, "batch_summary.json")
    aggregate = {
        "root_dir": args.root_dir,
        "out_root": args.out_root,
        "engine": "convex_adam",
        "device": args.device,
        "n_cases_total": len(all_cases),
        "n_cases_done": completed - len(failed) + len(skipped),
        "n_cases_failed": len(failed),
        "failed_cases": [f[0] for f in failed],
        "cvx_params": cvx_kwargs,
        "S_scaling": args.S_scaling,
        "elapsed_minutes": total_time / 60,
    }
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nbatch_summary saved -> {aggregate_path}")


if __name__ == "__main__":
    main()
