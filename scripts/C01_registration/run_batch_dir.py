"""
validation/run_batch_dir.py — Parametrised batch DIR for the expanded validation set.

Engines supported via --engine:
    unigradicon : Tian 2024 MICCAI — DL foundation model pretrained, GPU
    elastix     : Klein 2010 IEEE TMI — B-spline parametric, CPU

Per-case pipeline (identical to run_batch_convexadam.py):
    1. load_pair (fixed + moving + annotated landmarks)
    2. (opt) Procrustes 6-DOF prealign (Kabsch SVD on landmarks)
    3. Resampling of moving onto the fixed grid via ANTs identity (redundant
       after Procrustes but guarantees the exact same shape for the Registerer)
    4. DIR via core.registration.{UniGradICONRegisterer, ElastixRegisterer}
    5. Heuristic+residual σ via runner.compute_heuristic_uncertainty
    6. Per-landmark TRE + σ_p + LECE (Smolders Eq. 9, bin 0.25 mm)
    7. Save artifacts (schema shared with run_batch / run_batch_convexadam):
        summary.json + landmarks.npz + sigma_raw.npy + dvf_mu.npy +
        reliability_diagram.png + (opt) dvf_mu.dcm

Resume policy: skip cases with summary.json + n_landmarks > 0 (like the other
batches). Aggregate batch_summary.json in --out_root.

Examples:
    venv/Scripts/python.exe validation/run_batch_dir.py \\
        --engine unigradicon \\
        --root_dir "<DATASETS>/Liver-DIR-QA/NIfTI" \\
        --out_root "<DATASETS>/Liver-DIR-QA/Results_unigradicon_procrustes" \\
        --device cuda:0 --prealign procrustes

    venv/Scripts/python.exe validation/run_batch_dir.py \\
        --engine elastix \\
        --root_dir "<DATASETS>/Liver-DIR-QA/NIfTI" \\
        --out_root "<DATASETS>/Liver-DIR-QA/Results_elastix_procrustes" \\
        --prealign procrustes
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")

if __package__ is None or __package__ == "":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    sys.path.insert(0, os.path.join(os.path.dirname(here), "dl_arch"))

import torch

from utils.scan import Scan
from core.registration import (
    BaseRegisterer, UniGradICONRegisterer, ElastixRegisterer,
)
from core.dvf_dicom import write_dvf_dicom

from validation.load_abdominal_dir_qa import discover_cases, load_pair
from validation.lece import evaluate_landmarks, compute_lece, plot_reliability_diagram
from runner import compute_heuristic_uncertainty
from validation.run_pair import _apply_procrustes_to_pair
from validation.run_batch_convexadam import (
    _resample_moving_to_fixed, _build_scans, _is_done,
)


def run_one_dir(
    case_id: str,
    case_dir: str,
    out_dir: str,
    engine_name: str,
    registerer: BaseRegisterer,
    *,
    heuristic_device: str = "cuda:0",
    heuristic_w: float = 0.7,
    heuristic_jac_clip: float = 0.95,
    heuristic_ratio_clip: float = 0.95,
    heuristic_use_minmax: bool = True,
    S_scaling: float = 1.0,
    prealign: str = "procrustes",
    save_dicom: bool = False,
    similarity_metric: str = "mind_ssc",
    similarity_kwargs: dict = None,
    verbose: bool = True,
) -> dict:
    """Run the complete DIR pipeline for a single case and save the artifacts.

    Engine-agnostic: the `registerer` (UniGradICONRegisterer, ElastixRegisterer
    or any BaseRegisterer) is called to compute the DVF μ. Everything
    else (Procrustes, resampling, heuristic σ, LECE) is identical to
    run_one_convexadam.
    """
    os.makedirs(out_dir, exist_ok=True)

    def log(msg):
        if verbose:
            print(msg, flush=True)

    t0 = time.time()
    log(f"[{case_id}] === START ({engine_name}) ===")

    # 1. Load
    pair = load_pair(case_dir, case_id=case_id)
    log(f"  loaded fixed {pair.fixed_arr.shape} / moving {pair.moving_arr.shape}, "
        f"{pair.n_landmarks} landmarks")

    # 1b. Procrustes prealign (optional)
    rigid_M = None
    initial_tre_raw = None
    if prealign == "procrustes":
        initial_tre_raw = float(np.sqrt(np.mean(
            ((pair.lm_fixed_world - pair.lm_moving_world) ** 2).sum(axis=1))))
        log(f"  Procrustes prealign (initial RMSE TRE: {initial_tre_raw:.2f} mm)")
        pair, rigid_M, _ = _apply_procrustes_to_pair(pair, out_dir, save_artifacts=True)
        post_pre_tre = float(np.sqrt(np.mean(
            ((pair.lm_fixed_world - pair.lm_moving_world) ** 2).sum(axis=1))))
        log(f"    rigid-only RMSE TRE (post-Procrustes): {post_pre_tre:.2f} mm")
    elif prealign != "none":
        raise ValueError(f"prealign='{prealign}' not supported")

    # 2. Resample moving -> fixed grid (idempotent after Procrustes)
    log("  resampling moving -> fixed grid...")
    moving_aligned = _resample_moving_to_fixed(
        pair.fixed_arr, pair.moving_arr,
        pair.spacing_fixed, pair.spacing_moving,
        fixed_affine=getattr(pair, "fixed_affine", None),
        moving_affine=getattr(pair, "moving_affine", None),
    )

    # 3. DIR registration
    fixed_scan, moving_scan = _build_scans(pair, moving_aligned)
    log(f"  {engine_name} registration...")
    t_reg = time.time()
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

    # 4. Heuristic σ (always via runner.compute_heuristic_uncertainty)
    log("  computing heuristic+residual sigma...")
    t_h = time.time()
    fixed_t = torch.from_numpy(pair.fixed_arr[None, ...].astype(np.float32))
    moving_t = torch.from_numpy(moving_aligned[None, ...].astype(np.float32))
    dvf_t = torch.from_numpy(np.ascontiguousarray(dvf_mm.astype(np.float32)))
    if heuristic_device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(heuristic_device)
        fixed_t = fixed_t.to(dev); moving_t = moving_t.to(dev); dvf_t = dvf_t.to(dev)

    sx, sy, sz = pair.spacing_fixed
    sigma_1ch = compute_heuristic_uncertainty(
        fixed_ct=fixed_t, moving_ct=moving_t, dvf_mm=dvf_t,
        voxel_spacing=(sz, sy, sx),
        w_res_grad=heuristic_w, w_jac=1.0 - heuristic_w,
        jac_clip_percentile=heuristic_jac_clip,
        ratio_clip_percentile=heuristic_ratio_clip,
        use_minmax_norm=heuristic_use_minmax,
        global_scale=1.0, suppression_alpha=1.0, debug=False,
        similarity_metric=similarity_metric,
        similarity_kwargs=similarity_kwargs,
    )
    sigma_np_raw = sigma_1ch.squeeze().detach().cpu().numpy().astype(np.float32)
    h_time = time.time() - t_h
    log(f"    sigma raw range [{sigma_np_raw.min():.3f}, {sigma_np_raw.max():.3f}] "
        f"in {h_time:.1f}s")

    sigma_np = (sigma_np_raw * float(S_scaling)).astype(np.float32) \
        if S_scaling != 1.0 else sigma_np_raw

    # 5. Landmark eval + LECE
    eval_res = evaluate_landmarks(
        dvf_mm=dvf_mm, sigma_mm=sigma_np,
        spacing_fixed=pair.spacing_fixed,
        lm_fixed_voxel=pair.lm_fixed_voxel,
        lm_moving_world_true=pair.lm_moving_world,
        case_id=pair.case_id,
        fixed_affine=getattr(pair, "fixed_affine", None),
    )
    initial_tre = float(np.sqrt(np.mean(((pair.lm_fixed_world - pair.lm_moving_world) ** 2).sum(axis=1))))
    final_tre = float(np.sqrt(np.mean(eval_res.tre_mm ** 2)))
    median_tre = float(np.median(eval_res.tre_mm))
    log(f"    initial RMSE TRE: {initial_tre:.2f} mm -> final {final_tre:.2f} mm "
        f"(median {median_tre:.2f})")

    lece, diag = compute_lece(eval_res.tre_mm, eval_res.sigma_p_mm, bin_width=0.25)
    log(f"    LECE = {lece:.3f} mm")

    # 6. Save artefacts (shared schema)
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
        try:
            dvf_dicom_path = os.path.join(out_dir, "dvf_mu.dcm")
            write_dvf_dicom(vf, dvf_dicom_path,
                             method_metadata=registerer.get_metadata())
            log(f"    DVF saved to {dvf_dicom_path}")
        except Exception as e:
            log(f"    [WARN] DICOM save failed: {e}")

    plot_path = os.path.join(out_dir, "reliability_diagram.png")
    dataset_name = os.path.basename(os.path.normpath(case_dir))
    plot_reliability_diagram(
        diag, output_path=plot_path,
        title=f"{dataset_name} — {pair.case_id} ({engine_name}, LECE = {lece:.3f} mm)",
        label="heuristic+residual",
    )

    summary = {
        "case_id": pair.case_id,
        "case_dir": case_dir,
        "engine": engine_name,
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
    log(f"[{pair.case_id}] === DONE in {summary['total_time_s']:.1f}s ===")
    return summary


def _build_registerer(args) -> BaseRegisterer:
    if args.engine == "unigradicon":
        return UniGradICONRegisterer(
            model_name=args.model_name,
            finetune_steps=args.finetune_steps,
            io_sim=args.io_sim,
            device=args.device,
            verbose=False,
        )
    elif args.engine == "elastix":
        return ElastixRegisterer(
            parameter_files=args.parameter_files,
            log_to_console=False,
            verbose=False,
        )
    else:
        raise ValueError(f"Engine '{args.engine}' not supported")


def main():
    parser = argparse.ArgumentParser(
        description="Parametrised batch DIR (uniGradICON / Elastix) "
                    "for f calibration + LECE.")
    parser.add_argument("--engine", required=True,
                        choices=["unigradicon", "elastix"])
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--cases", default=None,
                        help="Comma-separated subset (default: all)")
    parser.add_argument("--device", default="cuda:0",
                        help="Device for uniGradICON and for the heuristic σ (Elastix is CPU-only)")
    parser.add_argument("--prealign", choices=["none", "procrustes"],
                        default="procrustes")
    parser.add_argument("--no_dicom", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--S_scaling", type=float, default=1.0)
    # uniGradICON-specific
    parser.add_argument("--model_name", default="unigradicon",
                        choices=["unigradicon", "multigradicon"])
    parser.add_argument("--finetune_steps", type=int, default=50)
    parser.add_argument("--io_sim", default="lncc",
                        choices=["lncc", "lncc2", "mind"])
    # Elastix-specific
    parser.add_argument("--parameter_files", nargs="*", default=None,
                        help="Elastix parameter file(s); default = bspline default")
    # Heuristic σ shared
    parser.add_argument("--similarity_metric", default="mind_ssc",
                        choices=["abs_diff", "mind_ssc", "lncc"])
    parser.add_argument("--mind_ssc_radius", type=int, default=1)
    parser.add_argument("--mind_ssc_dilation", type=int, default=2)
    parser.add_argument("--lncc_win", type=int, default=5)
    args = parser.parse_args()

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
    print(f"Batch DIR run — engine = {args.engine}")
    print("=" * 70)
    print(f"  root_dir   : {args.root_dir}")
    print(f"  out_root   : {args.out_root}")
    print(f"  total cases: {len(all_cases)}")
    print(f"  skipped    : {len(skipped)}")
    print(f"  todo       : {len(todo)}")
    print(f"  prealign   : {args.prealign}")
    print(f"  similarity : {args.similarity_metric} (kwargs={sim_kwargs})")
    if args.engine == "unigradicon":
        print(f"  ug params  : model={args.model_name}, finetune={args.finetune_steps}, "
              f"io_sim={args.io_sim}, device={args.device}")
    elif args.engine == "elastix":
        print(f"  elx params : files={args.parameter_files or ['default_bspline']}")
    print()

    if not todo:
        print("Nothing to do.")
        return

    os.makedirs(args.out_root, exist_ok=True)
    registerer = _build_registerer(args)
    t0 = time.time()
    completed = 0
    failed = []

    for case_id, case_dir in todo:
        case_out = os.path.join(args.out_root, case_id)
        try:
            s = run_one_dir(
                case_id=case_id, case_dir=case_dir, out_dir=case_out,
                engine_name=args.engine, registerer=registerer,
                heuristic_device=args.device,
                S_scaling=args.S_scaling, prealign=args.prealign,
                save_dicom=not args.no_dicom,
                similarity_metric=args.similarity_metric,
                similarity_kwargs=sim_kwargs, verbose=True,
            )
            if torch.cuda.is_available():
                import gc
                gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            completed += 1
            elapsed = time.time() - t0
            avg = elapsed / completed
            eta = avg * (len(todo) - completed)
            print(f"[{completed}/{len(todo)}] OK {case_id:<10} "
                  f"(initial {s['initial_rmse_tre_mm']:.2f} -> "
                  f"final {s['final_rmse_tre_mm']:.2f} mm in "
                  f"{s['total_time_s']:.0f}s, LECE {s['lece_mm']:.3f}) | "
                  f"elapsed {elapsed/60:.1f}min, ETA {eta/60:.1f}min", flush=True)
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

    aggregate = {
        "root_dir": args.root_dir,
        "out_root": args.out_root,
        "engine": args.engine,
        "device": args.device,
        "n_cases_total": len(all_cases),
        "n_cases_done": completed - len(failed) + len(skipped),
        "n_cases_failed": len(failed),
        "failed_cases": [f[0] for f in failed],
        "registerer_metadata": registerer.get_metadata(),
        "S_scaling": args.S_scaling,
        "elapsed_minutes": total_time / 60,
    }
    aggregate_path = os.path.join(args.out_root, "batch_summary.json")
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nbatch_summary saved -> {aggregate_path}")


if __name__ == "__main__":
    main()
