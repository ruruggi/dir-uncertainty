#!/usr/bin/env python3
"""
runner.py — CLI pipeline orchestrator for DIR_Uncertainty_v2.
Heuristic: R = 1 - alpha * residual * norm(||grad_I_m||) + Jacobian.
"""

import argparse
import os
import sys
import struct
import json
import tempfile
import shutil
import pathlib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from utils.scan import Scan
from utils.vector_field import VectorField
from utils.conversions import toNumpy, toTorch
from utils.interpolate import resampleScan
from core.grf import adaptive_ell_map
from core.sampling import DVFSampler
from core.dose_propagation import DosePropagator
# Lazy import: load_registerer/extract_* are used only by the CLI runner
# (--method smolders). validation.run_pair imports only
# compute_heuristic_uncertainty and does not require `dl/`. The import is
# moved inside the functions that use it so as not to break importing on
# systems without dl/ (e.g. CPU-only clusters without dl_arch).
from core.jacobian import jacobian_determinant
from utils.warp import warp_torch


# ----------------------------------------------------------------------
# Normalization helpers (unchanged)
# ----------------------------------------------------------------------
def zscore_normalize(x, subsample_max=500000):
    numel = x.numel()
    if numel > subsample_max:
        device = x.device
        indices = torch.randperm(numel, device=device)[:subsample_max]
        x_sample = x.flatten()[indices].cpu()
        mean = x_sample.mean().to(device)
        std = x_sample.std().to(device)
    else:
        mean = x.mean()
        std = x.std()
    if std < 1e-6:
        print(f"  [WARNING] zscore_normalize: std={std:.2e} < 1e-6, returning zeros")
        return torch.zeros_like(x)
    return (x - mean) / std


def minmax_normalize(x, eps=1e-8):
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min < eps:
        return torch.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def gradient_magnitude_central(img):
    gz = (img[2:] - img[:-2]) / 2.0
    gy = (img[:, 2:] - img[:, :-2]) / 2.0
    gx = (img[:, :, 2:] - img[:, :, :-2]) / 2.0
    gz = F.pad(gz, (0,0,0,0,1,1))
    gy = F.pad(gy, (0,0,1,1,0,0))
    gx = F.pad(gx, (1,1,0,0,0,0))
    return torch.sqrt(gz**2 + gy**2 + gx**2 + 1e-8)


GRADIENT_FUNC = gradient_magnitude_central


def compute_optimal_scale(heuristic_map, reference_map, method='mse'):
    mask = (heuristic_map > 0) & (reference_map > 0)
    h = heuristic_map[mask]
    r = reference_map[mask]
    if h.size == 0 or r.size == 0:
        return 1.0
    if method == 'mse':
        scale = np.sum(h * r) / (np.sum(h * h) + 1e-8)
    elif method == 'mean':
        scale = np.mean(r) / (np.mean(h) + 1e-8)
    elif method == 'median':
        scale = np.median(r) / (np.median(h) + 1e-8)
    else:
        scale = 1.0
    return scale


def _estimate_body_mask(ct_zyx_np, threshold: float = -500.0) -> "np.ndarray":
    """Estimate BODY mask from CT array (Z, Y, X) in HU.

    Steps: HU threshold → 3D largest connected component → per-slice
    binary_fill_holes (faster than 3D fill and preserves anatomy).

    Returns
    -------
    np.ndarray (bool), shape == ct_zyx_np.shape
    """
    import scipy.ndimage as ndi
    mask = ct_zyx_np > threshold
    labels, num = ndi.label(mask)
    if num > 0:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        if sizes.max() > 0:
            mask = labels == sizes.argmax()
    for z in range(mask.shape[0]):
        mask[z] = ndi.binary_fill_holes(mask[z])
    return mask.astype(bool)


def compute_heuristic_uncertainty(
    fixed_ct: torch.Tensor,
    moving_ct: torch.Tensor,
    dvf_mm: torch.Tensor,
    voxel_spacing: tuple,
    w_res_grad: float = 0.7,
    w_jac: float = 0.3,
    global_scale: float = 1.0,
    jac_clip_percentile: float = 0.95,
    ratio_clip_percentile=None,
    use_minmax_norm: bool = True,
    suppression_alpha: float = 1.0,
    debug: bool = False,
    similarity_metric: str = "mind_ssc",
    similarity_kwargs: dict = None,
    body_mask: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """
    Heuristic+Residual DVF uncertainty estimator.

    similarity_metric : str
        Metric for the residual term. Default 'mind_ssc' (Heinrich 2012/2013):
        12-channel structural descriptor, robust to wrong correspondences with
        chance intensity matches, valid cross-modal (CT, MR, MR-CT). Adopted
        as the default; it reduces LECE by an order of magnitude vs 'abs_diff'
        in the partial DIR failure regime.
        Opt-in options:
            'abs_diff' : |I_m(x+phi) - I_f|, the original metric, kept for
                         comparative experiments. CT/sCT only.
            'lncc'     : Local NCC, invariant to linear transformations (same-sequence MR).
        Implementations in core/similarity_metrics.py.
    similarity_kwargs : dict
        kwargs passed to the metric (e.g. {'radius': 1, 'dilation': 2} for MIND,
        {'win': 5} for LNCC). See core/similarity_metrics.py.
    """
    if body_mask is None:
        raise ValueError(
            "compute_heuristic_uncertainty: body_mask must be provided. "
            "Compute via _estimate_body_mask(fixed_ct) before calling, or pass "
            "an explicit mask. Without it, body-local min-max normalisation "
            "degenerates to global min-max and σ is non-zero outside the patient body."
        )
    device = dvf_mm.device
    fixed_ct = fixed_ct.to(device)
    moving_ct = moving_ct.to(device)
    sz, sy, sx = voxel_spacing

    dvf_vox = dvf_mm.clone()
    dvf_vox[0] /= sz
    dvf_vox[1] /= sy
    dvf_vox[2] /= sx

    if moving_ct.dim() == 3:
        moving_ct = moving_ct.unsqueeze(0)
    warped_moving = warp_torch(moving_ct.squeeze(0), dvf_vox).unsqueeze(0)

    # --- Residual term: dispatcher on the similarity metric -----------------
    # 'abs_diff' = original behaviour (back-compat). The alternatives
    # cover the MRI/cross-modal extension.
    if similarity_metric == "abs_diff":
        residual = torch.abs(warped_moving - fixed_ct)
    else:
        from core.similarity_metrics import compute_residual
        kw = dict(similarity_kwargs or {})
        residual = compute_residual(fixed_ct, warped_moving,
                                    metric=similarity_metric, **kw)
    # Both branches now produce a 4D tensor (1, X, Y, Z), matching the abs_diff
    # inline computation. (Pre-fix: an extra unsqueeze on the non-abs_diff path
    # made the residual 5D, propagating a spurious dim through sigma_1ch and
    # crashing sigma_1ch.repeat(3, 1, 1, 1) in PASS 1 with mind_ssc default.)

    moving_original = moving_ct.squeeze(0)
    grad_mag_moving_orig = GRADIENT_FUNC(moving_original)
    
    grad_min = grad_mag_moving_orig.min()
    grad_max = grad_mag_moving_orig.max()
    if grad_max - grad_min > 1e-8:
        grad_norm = (grad_mag_moving_orig - grad_min) / (grad_max - grad_min)
    else:
        grad_norm = torch.zeros_like(grad_mag_moving_orig)
    
    R = 1.0 - suppression_alpha * residual.squeeze(0) * grad_norm
    R = torch.clamp(R, min=0.0, max=1.0)

    if debug:
        print(f"\n[DEBUG Heuristic]")
        print(f"  residual: min={residual.min().item():.4f}, max={residual.max().item():.4f}, mean={residual.mean().item():.4f}")
        print(f"  grad_moving_orig: min={grad_mag_moving_orig.min().item():.4f}, max={grad_mag_moving_orig.max().item():.4f}")
        print(f"  grad_norm: min={grad_norm.min().item():.4f}, max={grad_norm.max().item():.4f}, mean={grad_norm.mean().item():.4f}")
        print(f"  suppression_alpha: {suppression_alpha}")
        print(f"  R (1 - alpha * residual * grad_norm): min={R.min().item():.4f}, max={R.max().item():.4f}, mean={R.mean().item():.4f}")

    numel = R.numel()
    sample_size = min(numel, 500000)
    indices = torch.randperm(numel, device=device)[:sample_size]
    R_sample = R.flatten()[indices]
    if ratio_clip_percentile is not None:     # IV: no clip of R by default (Alg. 1)
        clip_val = torch.quantile(R_sample, ratio_clip_percentile)
        R = torch.clamp(R, max=clip_val)
    
    if debug:
        if ratio_clip_percentile is not None:
            print(f"  R clipped at {ratio_clip_percentile*100:.0f}%: max={clip_val.item():.4f}")

    # Body mask preparation: outside-body voxels are excluded from minmax
    # dynamic-range estimation AND zeroed in the final σ. This makes the
    # heuristic σ a physically consistent field (zero where there is no patient).
    bm = None
    if body_mask is not None:
        bm = body_mask.to(device=R.device, dtype=torch.bool)
        # Broadcast bm to R shape (R is (Z,Y,X) or (1,Z,Y,X)); match by squeeze
        if bm.shape != R.shape:
            bm_squeezed = bm.squeeze()
            if bm_squeezed.shape == R.squeeze().shape:
                bm = bm_squeezed.reshape(R.shape)

    def _bodylocal_minmax(t, mask, eps=1e-8):
        """Min-max scaling using ONLY voxels where mask==True, output zeroed outside."""
        if mask is None:
            return minmax_normalize(t)
        vals = t[mask]
        if vals.numel() == 0:
            return torch.zeros_like(t)
        tmin = vals.min()
        tmax = vals.max()
        denom = (tmax - tmin) if (tmax - tmin) > eps else torch.tensor(1.0, device=t.device)
        out = (t - tmin) / denom
        out = torch.clamp(out, min=0.0, max=1.0)
        return torch.where(mask, out, torch.zeros_like(out))

    if use_minmax_norm:
        R_norm = _bodylocal_minmax(R, bm)
        if debug:
            print(f"  R_norm (minmax{', body-local' if bm is not None else ''}): "
                  f"min={R_norm.min().item():.4f}, max={R_norm.max().item():.4f}, "
                  f"mean={R_norm.mean().item():.4f}")
    else:
        R_norm = zscore_normalize(R)
        if bm is not None:
            R_norm = torch.where(bm, R_norm, torch.zeros_like(R_norm))
        if debug:
            print(f"  R_norm (zscore): min={R_norm.min().item():.4f}, "
                  f"max={R_norm.max().item():.4f}, mean={R_norm.mean().item():.4f}, "
                  f"std={R_norm.std().item():.4f}")

    if w_jac > 0:
        det_J = jacobian_determinant(dvf_vox)
        jac_dev = torch.abs(det_J - 1.0)

        numel_jac = jac_dev.numel()
        sample_size_jac = min(numel_jac, 500000)
        indices_jac = torch.randperm(numel_jac, device=device)[:sample_size_jac]
        jac_sample = jac_dev.flatten()[indices_jac].cpu()
        jac_thr = torch.quantile(jac_sample, jac_clip_percentile).to(device)
        jac_dev = torch.clamp(jac_dev, max=jac_thr)

        if debug:
            print(f"  jac_dev (clipped): min={jac_dev.min().item():.4f}, "
                  f"max={jac_dev.max().item():.4f}, mean={jac_dev.mean().item():.4f}")

        jac_norm = _bodylocal_minmax(jac_dev, bm)

        if debug:
            print(f"  jac_norm ({'body-local' if bm is not None else 'global'} minmax): "
                  f"min={jac_norm.min().item():.4f}, max={jac_norm.max().item():.4f}, "
                  f"mean={jac_norm.mean().item():.4f}")
    else:
        jac_norm = torch.zeros_like(R_norm)
        if debug:
            print(f"  w_jac = 0 -> jac_norm = zero")

    sigma = w_res_grad * R_norm + w_jac * jac_norm
    sigma = torch.clamp(sigma, min=0.0) * global_scale

    # Final masking: σ = 0 outside body (no patient → no registration uncertainty)
    if bm is not None:
        sigma = torch.where(bm, sigma, torch.zeros_like(sigma))

    if debug:
        print(f"  final sigma: min={sigma.min().item():.4f}, "
              f"max={sigma.max().item():.4f}, mean={sigma.mean().item():.4f}")
        print(f"  [DEBUG] -- end --\n")

    return sigma.unsqueeze(0)


def safe_save_npy(path, array):
    path = pathlib.Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npy', delete=False) as tmp:
        tmp_path = tmp.name
    np.save(tmp_path, array)
    shutil.move(tmp_path, path)


def create_identity_bdf(reference_dvf_path, output_path):
    ref_dvf = VectorField(reference_dvf_path)
    z, y, x = ref_dvf.getShape()
    spacing = list(ref_dvf.getVoxelSpacing())
    with open(output_path, 'wb') as f:
        for s in [x, y, z]:
            f.write(struct.pack('i', int(s)))
        for v in spacing:
            f.write(struct.pack('f', float(v)))
        np.zeros(int(z * y * x * 3), dtype=np.float32).tofile(f)
    print(f"  Identity DVF: shape=({x},{y},{z}), spacing={spacing}")
    return output_path


def reorient_dvf_to_target(dvf_tensor: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    if dvf_tensor.dim() != 4 or dvf_tensor.shape[0] != 3:
        raise ValueError(f"Expected 4D tensor (3, Z, Y, X), got shape {dvf_tensor.shape}")
    current_shape = dvf_tensor.shape[1:]
    if current_shape == target_shape:
        return dvf_tensor
    from itertools import permutations
    for perm in permutations([0, 1, 2]):
        permuted = dvf_tensor.permute(0, perm[0]+1, perm[1]+1, perm[2]+1)
        if permuted.shape[1:] == target_shape:
            print(f"  DVF reoriented with permutation {perm}: {current_shape} -> {target_shape}")
            return permuted
    dvf_tensor = dvf_tensor.unsqueeze(0)
    resampled = F.interpolate(dvf_tensor, size=target_shape, mode='trilinear', align_corners=True)
    resampled = resampled.squeeze(0)
    print(f"  DVF resampled from {current_shape} to {target_shape}")
    return resampled


def _apply_predef_rigid(fixed_scan, dvf_path: str) -> None:
    """IGRT/SBRT case: if the DVF DICOM carries a PreDeformationMatrix (rigid
    pCT->sCT pre-alignment from the commercial DIR), apply its inverse to
    ``fixed_scan``'s voxelToWorldMatrix so the pCT is brought into the
    DVF/sCT frame (metadata-only; pixel data unchanged).

    No-op when the DVF DICOM has no PreDeformationMatrix or the matrix is
    identity. utils/dicom_vf.py reads but does NOT add the rigid component
    to the elastic VF (line ~124), so the pCT-side alignment is left for
    callers — this helper centralises that alignment for PASS 1/PASS 4.
    """
    import pydicom as _pyd
    try:
        _dcm = _pyd.dcmread(dvf_path, stop_before_pixels=True)
        _predef = None
        for _reg in getattr(_dcm, "DeformableRegistrationSequence", []):
            if hasattr(_reg, "PreDeformationMatrixRegistrationSequence"):
                _predef = np.array(
                    _reg.PreDeformationMatrixRegistrationSequence[0]
                        .FrameOfReferenceTransformationMatrix
                ).reshape(4, 4)
                break
        if _predef is None:
            return
        if np.allclose(_predef, np.eye(4), atol=1e-6):
            return
        _inv = np.linalg.inv(_predef)
        _new_v2w = _inv @ fixed_scan.getVoxelToWorldMatrix()
        print(f"  Applying inv(PreDeformationMatrix) to fixed_scan v2w "
              f"(rigid pCT->sCT, translation Z={_predef[2, 3]:.1f} mm)")
        fixed_scan._setVoxelToWorldMatrix(_new_v2w)
    except Exception as _e:
        print(f"  [WARN] PreDeformationMatrix probe failed: {_e}")


def _build_subprocess_argv_for_fraction(args, fixed_path: str, frac: dict, subdir: str) -> list:
    """Build the argv for a runner.py invocation on a single fraction.

    Propagates all the sampling/uncertainty parameters of ``args``, overriding
    fixed/moving/dvf/moving_dose/out_dir with the per-fraction values.
    """
    py = sys.executable
    runner_path = os.path.abspath(__file__)
    argv = [
        py, runner_path,
        "--mode", "all",
        "--fixed", fixed_path,
        "--moving", frac["moving"],
        "--dvf", frac["dvf"],
        "--moving_dose", frac.get("moving_dose", ""),
        "--out_dir", subdir,
        "--model", args.model,
        "--n_samples", str(args.n_samples),
        "--l0_x", str(args.l0_x),
        "--l0_y", str(args.l0_y),
        "--l0_z", str(args.l0_z),
        "--alpha", str(args.alpha),
        "--theta", str(args.theta),
        "--variability_gain", str(args.variability_gain),
        "--kernel_size", str(args.kernel_size),
        "--max_retries", str(args.max_retries),
        "--uncertainty_method", args.uncertainty_method,
        "--heuristic_w", str(args.heuristic_w),
        "--heuristic_jac_clip", str(args.heuristic_jac_clip),
        "--heuristic_ratio_clip", str(args.heuristic_ratio_clip),
        "--heuristic_scale", str(args.heuristic_scale),
        "--heuristic_suppression_alpha", str(args.heuristic_suppression_alpha),
        "--heuristic_scale_method", args.heuristic_scale_method,
        "--heuristic_calibration_S", str(args.heuristic_calibration_S),
    ]
    # Boolean / optional flags
    if args.heuristic_use_minmax:
        argv.append("--heuristic_use_minmax")
    else:
        argv.append("--no-heuristic_use_minmax")
    if getattr(args, "heuristic_body_mask", True):
        argv.append("--heuristic_body_mask")
    else:
        argv.append("--no-heuristic_body_mask")
    argv.extend(["--heuristic_body_threshold",
                 str(getattr(args, "heuristic_body_threshold", -500.0))])
    if getattr(args, "noise_seed", None) is not None:
        argv.extend(["--noise_seed", str(args.noise_seed)])
    if args.heuristic_scale_from_smolders:
        argv.extend(["--heuristic_scale_from_smolders", args.heuristic_scale_from_smolders])
    if args.heuristic_debug:
        argv.append("--heuristic_debug")
    if args.fixed_rtstruct:
        argv.extend(["--fixed_rtstruct", args.fixed_rtstruct])
    return argv


def _run_dose_accumulation(args) -> None:
    """Multi-fraction dose accumulation (Iter.1, CLI-only).

    For each fraction, runs a subprocess with ``--mode all`` in a dedicated
    subfolder ``<out_dir>/<frac_id>``. When all fractions have produced
    ``dose_mean.npy`` + ``dose_std.npy``, accumulates with
    ``core.dose_accumulation.DoseAccumulator`` assuming inter-fraction
    independence (conservative model) and saves ``dose_accumulated.npy``,
    ``dose_accumulated_std.npy``, ``accumulation_metadata.json`` in ``out_dir``.

    Cache: if ``<frac_id>/dose_mean.npy`` and ``dose_std.npy`` exist, the fraction
    is considered already processed and the run goes straight to accumulation.
    To force recomputation, delete the subfolder manually.
    """
    import subprocess
    from core.dose_accumulation import load_fraction_outputs  # only for back-compat cache check
    from core.dose_accumulation_fc import SampleEnsembleDoseAccumulator

    print("=" * 60)
    print("DIR Uncertainty Pipeline v2 — Multi-Fraction Dose Accumulation")
    print("=" * 60)

    # Parse accumulation_mode: subset of {"nc","fc"}.
    modes = [m.strip().lower() for m in (args.accumulation_mode or "nc").split(",") if m.strip()]
    modes = [m for m in modes if m in ("nc", "fc")]
    if not modes:
        modes = ["nc"]
    needs_fc = "fc" in modes
    needs_nc = "nc" in modes

    # FC needs shared epsilons -> a deterministic --noise_seed is required. If
    # the user did not provide one, a default is set and stored in
    # accumulation_metadata so that the run is reproducible.
    if needs_fc and args.noise_seed is None:
        args.noise_seed = 12345
        print(f"  [FC] noise_seed not provided, using deterministic default = {args.noise_seed}")

    print(f"  Accumulation modes: {modes}")

    with open(args.fractions, "r") as f:
        spec = json.load(f)

    fixed_path = spec.get("fixed", "") or args.fixed
    if not fixed_path:
        raise ValueError("fractions.json must contain 'fixed' (plan CT dir) "
                         "or --fixed must be passed on the CLI.")
    fractions = spec.get("fractions", [])
    if not fractions:
        raise ValueError("fractions.json must contain a non-empty 'fractions' list.")

    # Adaptive RT extrapolation: CLI override > JSON. 0 / None = not active.
    n_total = args.n_total if args.n_total else int(spec.get("n_total", 0) or 0)
    m_delivered = args.m_delivered if args.m_delivered else int(spec.get("m_delivered", 0) or 0)
    if m_delivered <= 0:
        m_delivered = len(fractions)
    if n_total <= 0:
        n_total = m_delivered  # no extrapolation
    if n_total < m_delivered:
        raise ValueError(f"n_total ({n_total}) < m_delivered ({m_delivered}): "
                         "invalid configuration (cannot deliver more "
                         "fractions than planned).")

    print(f"Fixed (plan CT): {fixed_path}")
    print(f"Fractions: {len(fractions)} (m_delivered={m_delivered}, n_total={n_total}"
          + (f", scale n/m={n_total/m_delivered:.4f}" if n_total != m_delivered else ", no extrapolation")
          + ")")
    for i, frac in enumerate(fractions, 1):
        fid = frac.get("id") or f"frac_{i:02d}"
        w = float(frac.get("weight", 1.0))
        print(f"  [{i}/{len(fractions)}] id={fid}  weight={w}  moving={frac.get('moving','')}")

    # PASS 1-4 for each fraction (subprocess, cache-aware).
    # A cache hit even for the _nc files requires dose_ensemble.npy for FC; hence
    # the cache check must be sensitive to the requested mode.
    for i, frac in enumerate(fractions, 1):
        fid = frac.get("id") or f"frac_{i:02d}"
        subdir = os.path.join(args.out_dir, fid)
        os.makedirs(subdir, exist_ok=True)

        # Both NC and FC consume dose_ensemble.npy (K, Z, Y, X). NC applies a
        # per-fraction permutation to it to break the correlation of the
        # shared noise; FC preserves the order. Hence the dose_ensemble.npy file
        # is needed in both cases.
        cached_ens_file = os.path.exists(os.path.join(subdir, "dose_ensemble.npy"))
        # FC additionally requires that dose_ensemble was generated with the
        # same noise_seed as the current run (correct shared epsilon).
        cached_seed = None
        meta_path = os.path.join(subdir, "pass3_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as mf:
                    cached_seed = json.load(mf).get("noise_seed")
            except Exception:
                cached_seed = None
        cached_fc_seed_ok = (cached_seed == args.noise_seed)
        nc_ok = (not needs_nc) or cached_ens_file
        fc_ok = (not needs_fc) or (cached_ens_file and cached_fc_seed_ok)
        if nc_ok and fc_ok:
            print(f"\n[{i}/{len(fractions)}] {fid}: cache hit ("
                  f"{'NC' if needs_nc else ''}{'+FC' if needs_fc else ''}), skipping PASS 1-4.")
            continue
        if needs_fc and cached_ens_file and not cached_fc_seed_ok:
            print(f"\n[{i}/{len(fractions)}] {fid}: cache invalid for FC "
                  f"(noise_seed cached={cached_seed} != requested={args.noise_seed}), "
                  f"recomputing PASS 1-4.")

        for required_key in ("moving", "dvf"):
            if not frac.get(required_key):
                raise ValueError(f"Fraction '{fid}': missing field '{required_key}'.")

        cmd = _build_subprocess_argv_for_fraction(args, fixed_path, frac, subdir)
        print(f"\n[{i}/{len(fractions)}] {fid}: starting PASS 1-4 in {subdir}")
        print(f"  cmd: {' '.join(cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            raise RuntimeError(f"Fraction '{fid}': runner.py PASS 1-4 exited with code {rc}.")

    # Accumulation: one pass for each requested mode.
    print("\n" + "=" * 60)
    print("ACCUMULATION")
    print("=" * 60)
    combined_meta = {"modes": modes, "noise_seed": args.noise_seed,
                     "n_total": n_total, "m_delivered": m_delivered, "fractions": []}

    # Sample-level accumulation for BOTH modes (Smolders 2023 §2.7.3 and
    # Fig. 8). The only difference between FC and NC is whether the k axis of
    # the fraction is permuted: without permutation the same shared epsilon_k
    # add up coherently (FC); with per-fraction permutation the k-vs-k
    # pairings become effectively independent (NC).
    first_subdir = os.path.join(args.out_dir,
                                 fractions[0].get("id") or "frac_01")
    ens0_path = os.path.join(first_subdir, "dose_ensemble.npy")
    if not os.path.exists(ens0_path):
        raise RuntimeError(f"Missing {ens0_path}; rerun PASS 1-4.")
    ens0 = np.load(ens0_path, mmap_mode="r")
    K = int(ens0.shape[0])
    fixed_shape = tuple(int(s) for s in ens0.shape[1:])
    print(f"  K={K} samples, fixed_shape={fixed_shape}")

    def _run_accum(mode_label, get_permute_seed):
        """Run sample-level accumulation for one mode (label = 'fc'|'nc').
        get_permute_seed(i) -> int | None for fraction i (1-based).
        Saves dose_accumulated_<mode>.npy, _<mode>_std.npy and
        dose_acc_per_sample_<mode>.npy (memmap K-samples).
        Returns the result dict of finalize()."""
        print(f"\n--- {mode_label.upper()} accumulation (sample-level) ---")
        with SampleEnsembleDoseAccumulator(fixed_shape, n_samples=K,
                                            out_dir=args.out_dir,
                                            n_total=n_total,
                                            m_delivered=m_delivered,
                                            correlation_mode=mode_label.upper()
                                            ) as acc:
            for i, frac in enumerate(fractions, 1):
                fid = frac.get("id") or f"frac_{i:02d}"
                subdir = os.path.join(args.out_dir, fid)
                weight = float(frac.get("weight", 1.0))
                ens_path = os.path.join(subdir, "dose_ensemble.npy")
                if not os.path.exists(ens_path):
                    raise RuntimeError(f"Fraction '{fid}': dose_ensemble.npy "
                                       f"missing in {subdir}.")
                p_seed = get_permute_seed(i)
                acc.add_per_sample(ens_path, weight=weight, frac_id=fid,
                                    permute_seed=p_seed)
                print(f"  [{i}/{len(fractions)}] {mode_label.upper()} {fid}: "
                      f"w={weight}  permute_seed={p_seed}")
            res = acc.finalize()
            suf = mode_label.lower()
            np.save(os.path.join(args.out_dir, f"dose_accumulated_{suf}.npy"),
                    res["mean"])
            np.save(os.path.join(args.out_dir, f"dose_accumulated_{suf}_std.npy"),
                    res["std"])
            if "predicted_mean" in res:
                np.save(os.path.join(args.out_dir, f"dose_predicted_full_{suf}.npy"),
                        res["predicted_mean"])
                np.save(os.path.join(args.out_dir, f"dose_predicted_full_{suf}_std.npy"),
                        res["predicted_std"])
            total = res["metadata"]["total"]
            print(f"[OK] {mode_label.upper()}: "
                  f"mean=[{total['mean_min']:.3f}, {total['mean_max']:.3f}]  "
                  f"std=[{total['std_min']:.4f}, {total['std_max']:.4f}]")
            return res

    nc_result = None
    fc_result = None
    if needs_nc:
        # NC: per-fraction permutation (breaks the correlation). seed=i is stable
        # and produces distinct permutations for each fraction.
        nc_result = _run_accum("nc", lambda i: int(i))
        combined_meta["nc"] = nc_result["metadata"]
    if needs_fc:
        # FC: no permutation -> the running sum pairs the same shared epsilon_k
        # across fractions.
        fc_result = _run_accum("fc", lambda i: None)
        combined_meta["fc"] = fc_result["metadata"]

    with open(os.path.join(args.out_dir, "accumulation_metadata.json"), "w") as f:
        json.dump(combined_meta, f, indent=2)
    print(f"\n[OK] Combined metadata -> {os.path.join(args.out_dir, 'accumulation_metadata.json')}")
    print("\n" + "=" * 60)
    print("[DONE] dose_accumulation")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    # --fixed/--moving/--dvf are required ONLY for the single-fraction modes.
    # For --mode dose_accumulation they are taken from the --fractions JSON; the
    # conditional validation is done after parse_args().
    parser.add_argument("--fixed", default="")
    parser.add_argument("--moving", default="")
    parser.add_argument("--dvf", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", default="L20S3-4")
    # Dose associated with the MOVING image (RTDOSE on the moving CT). PASS 4
    # deforms it onto the fixed grid via DVF pull-back. Renamed from
    # --dose; the old name is kept as a deprecated alias for CLI compatibility.
    parser.add_argument("--moving_dose", "--dose", default="", dest="moving_dose")
    parser.add_argument("--fixed_rtstruct", default="")
    parser.add_argument("--moving_rtstruct", default="")
    parser.add_argument("--mode", default="all",
                        choices=["uncertainty", "sampling", "dose", "all", "dose_accumulation"])
    # Multi-fraction: JSON with the list of fractions.
    # Schema: {"fixed": "<plan_CT_dir>", "n_total": <int>, "m_delivered": <int>,
    # "fractions": [{"id", "moving", "dvf", "moving_dose", "weight"}, ...]}.
    # See core/dose_accumulation.py.
    parser.add_argument("--fractions", default="",
                        help="Path to fractions.json (required if --mode=dose_accumulation).")
    # Adaptive RT extrapolation: expected end-of-treatment dose = (n/m)*D_acc.
    # Overrides the corresponding field in fractions.json if both are present.
    # 0 = not specified (use the JSON value or disable extrapolation).
    parser.add_argument("--n_total", type=int, default=0,
                        help="Total number of planned fractions (adaptive RT extrapolation). "
                        "Overrides fractions.json. 0 = not specified.")
    parser.add_argument("--m_delivered", type=int, default=0,
                        help="Number of fractions already delivered (adaptive RT extrapolation). "
                        "Default: len(fractions) in the JSON. 0 = not specified.")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--l0_z", type=float, default=19.5)
    parser.add_argument("--l0_y", type=float, default=13.0)
    parser.add_argument("--l0_x", type=float, default=13.0)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--variability_gain", type=float, default=1.0)
    parser.add_argument("--kernel_size", type=int, default=61)
    parser.add_argument("--max_retries", type=int, default=20)
    # Multi-fraction Smolders 2023 §2.7.3: FC vs NC. If --noise_seed is set,
    # PASS 3 uses a deterministic shared_white_noise (so that fractions with the
    # same seed share epsilon -> correct inter-fraction FC correlation).
    parser.add_argument("--noise_seed", type=int, default=None,
                        help="Deterministic seed for the white noise of PASS 3 "
                        "(FC mode: use the same seed for all fractions "
                        "of the treatment). Default: random (NC).")
    # Dose accumulation mode (only --mode=dose_accumulation):
    parser.add_argument("--accumulation_mode", type=str, default="nc",
                        help="Comma-separated list from {nc, fc}. "
                        "'nc' uses core.dose_accumulation.DoseAccumulator (sample-free, "
                        "Var_acc = sum w_i^2 * Var_i). 'fc' uses "
                        "core.dose_accumulation_fc.SampleEnsembleDoseAccumulator "
                        "(sample-level, requires --noise_seed for the generation "
                        "of the shared epsilon). 'nc,fc' computes both and saves "
                        "with suffix _nc/_fc.")
    parser.add_argument("--uncertainty_method", default="smolders", choices=["smolders", "heuristic"])
    
    parser.add_argument("--heuristic_w", type=float, default=0.7,
                        help="Weight of the residual term in Eq. (10); the Jacobian weight is 1 - w.")
    parser.add_argument("--heuristic_jac_clip", type=float, default=0.95,
                        help="Clip percentile for |det J - 1| outlier filter "
                             "(default 0.95, i.e. 95-percentile).")
    parser.add_argument("--heuristic_ratio_clip", type=float, default=None,
                        help="Clip percentile for the data-fidelity term R "
                             "(default: no clip).")
    parser.add_argument("--heuristic_scale", type=float, default=1.0)
    parser.add_argument("--heuristic_suppression_alpha", type=float, default=1.0)
    parser.add_argument("--heuristic_use_minmax", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heuristic_scale_from_smolders", type=str, default="",
                        help="DEPRECATED — use --heuristic_calibration_S. "
                        "If set, scales sigma_h via RMS against a precomputed Smolders map.")
    parser.add_argument("--heuristic_scale_method", type=str, default="mse", choices=["mse", "mean", "median"])
    parser.add_argument("--heuristic_calibration_S", "--heuristic_calibration_c",
                        type=float, default=1.0, dest="heuristic_calibration_S",
                        help="Global calibration factor S (per-district, from "
                        "config/c_calibration_registry.json). Default 1.0 = no calibration. "
                        "--heuristic_calibration_c is a deprecated alias.")
    parser.add_argument("--heuristic_debug", action="store_true")
    parser.add_argument("--heuristic_body_mask",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Restrict heuristic σ to a CT-derived BODY mask and "
                             "compute minmax dynamic range using only intra-body voxels. "
                             "Outside body, σ = 0 (no patient, no registration "
                             "uncertainty). Default: enabled.")
    parser.add_argument("--heuristic_body_threshold", type=float, default=-500.0,
                        help="HU threshold for BODY mask extraction "
                             "(standard RT convention: -500). Default: -500.")
    args = parser.parse_args()

    args.out_dir = args.out_dir.strip()
    os.makedirs(args.out_dir, exist_ok=True)

    # Conditional validation of the required-by-mode args.
    if args.mode == "dose_accumulation":
        if not args.fractions:
            parser.error("--mode dose_accumulation requires --fractions <path>.")
        # Dispatch and exit (single-fraction PASSes are launched as
        # subprocesses by _run_dose_accumulation, one per fraction).
        _run_dose_accumulation(args)
        return
    else:
        missing = [n for n, v in [("--fixed", args.fixed),
                                  ("--moving", args.moving),
                                  ("--dvf", args.dvf)] if not v]
        if missing:
            parser.error(f"Arguments required for --mode {args.mode}: {', '.join(missing)}")

    print("=" * 60)
    print("DIR Uncertainty Pipeline v2")
    print("=" * 60)
    print(f"Mode: {args.mode}, Samples: {args.n_samples}")
    print(f"l0: ({args.l0_x}, {args.l0_y}, {args.l0_z}) mm, alpha={args.alpha}, gain={args.variability_gain}")
    print(f"Uncertainty method: {args.uncertainty_method}")
    if args.uncertainty_method == "heuristic":
        print(f"  w (residual): {args.heuristic_w}")
        print(f"  1 - w (Jacobian): {1.0 - args.heuristic_w}")
        print(f"  Jacobian clip percentile: {args.heuristic_jac_clip}")
        print(f"  R clip percentile: {args.heuristic_ratio_clip}")
        print(f"  Suppression alpha: {args.heuristic_suppression_alpha}")
        print(f"  R = 1 - alpha * residual * norm(||grad_I_m||)")
        print(f"  Use min-max normalization: {args.heuristic_use_minmax}")
        print(f"  Global scale: {args.heuristic_scale}")
        if args.heuristic_scale_from_smolders:
            print(f"  Optimal scaling from Smolders: {args.heuristic_scale_from_smolders}")
            print(f"  Scaling method: {args.heuristic_scale_method}")
        print(f"  Debug mode: {args.heuristic_debug}")

    registerer = None
    if args.mode in ["uncertainty", "all"] and args.uncertainty_method == "smolders":
        from dl.model_loader import load_registerer
        registerer = load_registerer(args.model)

    if args.mode in ["uncertainty", "all"]:
        print(f"\n{'=' * 60}\nPASS 1: DIR Uncertainty ({args.uncertainty_method})\n{'=' * 60}")

        fixed_scan = Scan(args.fixed)
        moving_scan = Scan(args.moving)
        fixed_dvf = VectorField(args.dvf)
        fixed_dvf.toMM()

        _apply_predef_rigid(fixed_scan, args.dvf)

        target_shape = fixed_scan.getShape()
        spacing = fixed_scan.getVoxelSpacing()
        desired_shape = target_shape[::-1]
        print(f"  Target shape (Z,Y,X): {target_shape}")

        # IGRT/SBRT case: sCT (moving) may be cropped FOV vs pCT (fixed). warp_torch
        # downstream uses moving.shape as output grid, so shape parity is required.
        # Resample moving to fixed grid (world-coord aware via resampleScan, fills
        # outside-FOV with HU=-1000 = air). No-op if shapes already match.
        if moving_scan.getShape() != target_shape:
            # NOTE: resampleScan is imported at module level (line 24). A redundant
            # `from utils.interpolate import resampleScan` here would force Python
            # to treat resampleScan as a local in main(), causing UnboundLocalError
            # on any later call (e.g. line ~1125) if this branch is not entered.
            print(f"  Resampling moving CT to fixed grid: "
                  f"{moving_scan.getShape()} -> {target_shape}")
            moving_scan = resampleScan(
                moving_scan, fixed_scan.getVoxelToWorldMatrix(),
                target_shape, defaultPixelValue=-1000.0,
            )
            # resampleScan leaves pixel data on CUDA. The Smolders DL path
            # downstream calls np.concatenate on the scan which triggers
            # tensor.numpy() — fails on CUDA tensors. Bring back to CPU/numpy.
            moving_scan.toNumpy()

        if args.uncertainty_method == "smolders":
            from dl.model_loader import extract_mu_sigma, extract_uncertainty_magnitude
            _, prob_dvf = registerer.process(fixed_scan, moving_scan, fixed_dvf)
            prob_dvf.toMM()
            unc_mag = extract_uncertainty_magnitude(prob_dvf)
            mu, sigma = extract_mu_sigma(prob_dvf)
            print(f"  Smolders uncertainty shape: {unc_mag.shape}")
            # mu/sigma live on the grid of the input bdf (postProcess does _setPixelArray
            # with the pixel array of the fixedVectorField). Realign them to the CT, so
            # that dvf_magnitude.npy (and the following steps) use the correct resolution.
            if mu.dim() == 5 and mu.shape[0] == 1:
                mu = mu.squeeze(0)
            if sigma.dim() == 5 and sigma.shape[0] == 1:
                sigma = sigma.squeeze(0)
            if mu.shape[-3:] != target_shape:
                print(f"  Aligning mu/sigma resolution: {tuple(mu.shape[-3:])} -> {target_shape}")
                mu = F.interpolate(mu.unsqueeze(0), size=target_shape, mode='trilinear', align_corners=True).squeeze(0)
                sigma = F.interpolate(sigma.unsqueeze(0), size=target_shape, mode='trilinear', align_corners=True).squeeze(0)
        else:
            dvf_resampled = resampleScan(fixed_dvf, fixed_scan.getVoxelToWorldMatrix(), target_shape)
            mu = toTorch(dvf_resampled.getPixelArray()).float()
            if mu.dim() == 4 and mu.shape[0] == 1:
                mu = mu.squeeze(0)
            mu = reorient_dvf_to_target(mu, target_shape)
            if torch.cuda.is_available():
                mu = mu.cuda()
            
            ct_tensor = toTorch(fixed_scan.getPixelArray()).float()
            if ct_tensor.dim() == 3:
                ct_tensor = ct_tensor.unsqueeze(0)
            elif ct_tensor.dim() == 4 and ct_tensor.shape[0] != 1:
                ct_tensor = ct_tensor[0:1]
            ct_tensor = ct_tensor.to(mu.device)
            
            moving_ct = toTorch(moving_scan.getPixelArray()).float()
            if moving_ct.dim() == 3:
                moving_ct = moving_ct.unsqueeze(0)
            elif moving_ct.dim() == 4 and moving_ct.shape[0] != 1:
                moving_ct = moving_ct[0:1]
            moving_ct = moving_ct.to(mu.device)

            # Body mask from fixed CT (Z,Y,X order matches dvf/sigma downstream).
            body_mask_t = None
            if args.heuristic_body_mask:
                ct_np = ct_tensor.squeeze().detach().cpu().numpy()
                if ct_np.ndim == 4:
                    ct_np = ct_np[0]
                # ct_tensor shape from fixed_scan is (X, Y, Z); want (Z, Y, X) for
                # consistency with dvf_vox (Z, Y, X). Transpose then mask.
                if ct_np.shape == (target_shape[2], target_shape[1], target_shape[0]):
                    pass  # already (Z,Y,X)
                elif ct_np.shape == target_shape:
                    # target_shape is (X,Y,Z) per fixed_scan.getShape()
                    ct_np = np.transpose(ct_np, (2, 1, 0))
                body_mask_np = _estimate_body_mask(ct_np,
                                                    threshold=args.heuristic_body_threshold)
                # The function expects mask in same (X,Y,Z) order as sigma below
                # (which lives in fixed-scan coord). compute_heuristic_uncertainty
                # broadcasts to R shape regardless, so pass as torch in (Z,Y,X).
                body_mask_t = torch.from_numpy(body_mask_np).to(mu.device)
                # mu / R are in shape (3, X, Y, Z) and (X, Y, Z) per upstream
                # convention. Transpose body_mask back to match.
                if body_mask_t.shape != target_shape:
                    if body_mask_t.shape[::-1] == target_shape:
                        body_mask_t = body_mask_t.permute(2, 1, 0).contiguous()
                print(f"  Body mask: {int(body_mask_t.sum().item())} / "
                      f"{body_mask_t.numel()} voxels "
                      f"({100.0*body_mask_t.float().mean().item():.1f}%)")
                safe_save_npy(os.path.join(args.out_dir, "body_mask.npy"),
                              body_mask_t.cpu().numpy().astype(np.uint8))

            sigma_1ch = compute_heuristic_uncertainty(
                fixed_ct=ct_tensor,
                moving_ct=moving_ct,
                dvf_mm=mu,
                voxel_spacing=tuple(spacing),
                w_res_grad=args.heuristic_w,
                w_jac=1.0 - args.heuristic_w,
                global_scale=args.heuristic_scale,
                jac_clip_percentile=args.heuristic_jac_clip,
                ratio_clip_percentile=args.heuristic_ratio_clip,
                use_minmax_norm=args.heuristic_use_minmax,
                suppression_alpha=args.heuristic_suppression_alpha,
                debug=args.heuristic_debug,
                body_mask=body_mask_t,
            )
            sigma = sigma_1ch.repeat(3, 1, 1, 1)
            unc_mag = sigma_1ch.squeeze(0).cpu().numpy()
            
            print(f"  [DEBUG] unc_mag shape before anything else: {unc_mag.shape}")
            
            # ============================================================
            # ORIENTATION FOR VISUALISATION — unc_mag only.
            # mu and sigma MUST stay in target_shape (X,Y,Z) because
            # PASS 3 compares `mu_dvf.shape[-3:]` with `fixed_image.getShape()`
            # (= target_shape). If transposed to (Z,Y,X), PASS 3 passes them to
            # F.interpolate, which stretches the axes without permuting -> 90°
            # rotation + mirror visible on the mean/ensemble DVF.
            # The (X,Y,Z)→(Z,Y,X) conversion for the visualisation files
            # (dvf_mean_mag.npy, dvf_ensemble_mag.npy) is handled by
            # `_standardize_vol` on the GUI side.
            # ============================================================
            if unc_mag.ndim == 3 and unc_mag.shape != desired_shape:
                if unc_mag.shape[::-1] == desired_shape:
                    unc_mag = np.transpose(unc_mag, (2, 1, 0))
                    print(f"  [DEBUG] unc_mag transposed (2,1,0): {unc_mag.shape}")
            
            # ============================================================
            # SCALING FROM SMOLDERS
            # ============================================================
            if args.heuristic_scale_from_smolders and os.path.exists(args.heuristic_scale_from_smolders):
                try:
                    smolders_ref = np.load(args.heuristic_scale_from_smolders)
                    print(f"  [DEBUG] Smolders shape: {smolders_ref.shape}")
                    print(f"  [DEBUG] Heuristic shape (after orientation): {unc_mag.shape}")
                    
                    if smolders_ref.shape == unc_mag.shape:
                        optimal_scale = compute_optimal_scale(unc_mag, smolders_ref, method=args.heuristic_scale_method)
                        print(f"  Optimal scale from Smolders ({args.heuristic_scale_method}): {optimal_scale:.6f}")
                        if abs(optimal_scale - 1.0) > 0.001:
                            unc_mag = unc_mag * optimal_scale
                            sigma = sigma * optimal_scale
                            print(f"  Applied optimal scaling (new max: {unc_mag.max():.4f})")
                    else:
                        print(f"  [WARNING] Shape mismatch")
                except Exception as e:
                    print(f"  [WARNING] Scaling failed: {e}")

        # Global calibration c (per-district post-fit, see
        # config/c_calibration_registry.json and validation/calibrate.py).
        # Default args.heuristic_calibration_S = 1.0 -> no-op (back-compat).
        if abs(args.heuristic_calibration_S - 1.0) > 1e-9:
            print(f"  Applying district calibration S = {args.heuristic_calibration_S:.4f} "
                  f"to sigma and unc_mag")
            sigma = sigma * float(args.heuristic_calibration_S)
            if isinstance(unc_mag, np.ndarray):
                unc_mag = unc_mag * float(args.heuristic_calibration_S)
            else:
                unc_mag = unc_mag * float(args.heuristic_calibration_S)

        safe_save_npy(os.path.join(args.out_dir, "mu_dvf.npy"), mu.cpu().numpy())
        safe_save_npy(os.path.join(args.out_dir, "sigma_dvf.npy"), sigma.cpu().numpy())
        safe_save_npy(os.path.join(args.out_dir, "uncertainty_dir.npy"), unc_mag)

        dvf_mag = torch.norm(mu, dim=0).cpu().numpy()
        if dvf_mag.ndim == 3 and dvf_mag.shape != desired_shape:
            if dvf_mag.shape[::-1] == desired_shape:
                dvf_mag = np.transpose(dvf_mag, (2, 1, 0))
        safe_save_npy(os.path.join(args.out_dir, "dvf_magnitude.npy"), dvf_mag)

        print(f"  [OK] PASS 1 — sigma shape: {sigma.shape}, uncertainty shape: {unc_mag.shape}")
        torch.cuda.empty_cache()

    # ================================================================
    # PASS 2: Baseline Uncertainty (always Smolders on identity)
    # ================================================================
    if args.mode in ["uncertainty", "all"]:
        print(f"\n{'=' * 60}\nPASS 2: Baseline Uncertainty (Smolders on identity DVF)\n{'=' * 60}")

        identity_path = os.path.join(args.out_dir, "_temp_identity.bdf")
        create_identity_bdf(args.dvf, identity_path)

        moving_as_fixed = Scan(args.moving)
        moving_as_moving = Scan(args.moving)
        identity_dvf = VectorField(identity_path)
        identity_dvf.toVoxels()

        if registerer is None:
            from dl.model_loader import load_registerer
            registerer = load_registerer(args.model)
        from dl.model_loader import extract_uncertainty_magnitude
        _, prob_baseline = registerer.process(moving_as_fixed, moving_as_moving, identity_dvf)
        prob_baseline.toMM()

        unc_baseline = extract_uncertainty_magnitude(prob_baseline)
        safe_save_npy(os.path.join(args.out_dir, "uncertainty_baseline.npy"), unc_baseline)

        unc_dir_path = os.path.join(args.out_dir, "uncertainty_dir.npy")
        if os.path.exists(unc_dir_path):
            unc_dir = np.load(unc_dir_path)
            if unc_dir.shape == unc_baseline.shape:
                diff = np.clip(unc_dir - unc_baseline, 0, None)
                safe_save_npy(os.path.join(args.out_dir, "uncertainty_diff.npy"), diff)

        if os.path.exists(identity_path):
            os.remove(identity_path)
        torch.cuda.empty_cache()
        print("  [OK] PASS 2")

    # ================================================================
    # PASS 3: DVF Ensemble Sampling
    # ================================================================
    if args.mode in ["sampling", "all"]:
        print(f"\n{'=' * 60}\nPASS 3: DVF Ensemble ({args.n_samples} samples)\n{'=' * 60}")

        torch.cuda.empty_cache()
        fixed_image = Scan(args.fixed)
        _apply_predef_rigid(fixed_image, args.dvf)
        target_shape = fixed_image.getShape()
        spacing = fixed_image.getVoxelSpacing()

        mu_path = os.path.join(args.out_dir, "mu_dvf.npy")
        sigma_path = os.path.join(args.out_dir, "sigma_dvf.npy")
        if not os.path.exists(mu_path) or not os.path.exists(sigma_path):
            print("  ERROR: mu_dvf.npy or sigma_dvf.npy not found. Run PASS 1 first.")
            return
        mu_dvf = torch.from_numpy(np.load(mu_path)).float()
        sigma_dvf = torch.from_numpy(np.load(sigma_path)).float()

        if mu_dvf.shape[-3:] != target_shape:
            print(f"  Aligning resolution: {mu_dvf.shape[-3:]} -> {target_shape}")
            mu_dvf = F.interpolate(mu_dvf.unsqueeze(0), size=target_shape, mode='trilinear', align_corners=True).squeeze(0)
            sigma_dvf = F.interpolate(sigma_dvf.unsqueeze(0), size=target_shape, mode='trilinear', align_corners=True).squeeze(0)
            safe_save_npy(mu_path, mu_dvf)
            safe_save_npy(sigma_path, sigma_dvf)

        # Pipeline default for θ: --theta 0 (CLI default, GUI default, see
        # gui/app.py "theta": tk.DoubleVar(value=0.0) with label "θ (HU,
        # 0=auto)") triggers median(|∇I_f|) of the fixed CT. The signature
        # default 50.0 in core.grf.adaptive_ell_map is a convenience fallback
        # for bare imports and is NEVER reached on this code path.
        theta = args.theta
        if theta == 0.0:
            ct_arr = toNumpy(fixed_image.getPixelArray())
            if ct_arr.ndim == 4:
                ct_arr = ct_arr[0]
            gz, gy, gx = np.gradient(ct_arr.astype(np.float32))
            grad_mag = np.sqrt(gz**2 + gy**2 + gx**2)
            theta = float(np.median(grad_mag))
            print(f"  Auto theta = {theta:.2f} HU (median |grad I_f|)")

        ct_tensor = toTorch(fixed_image.getPixelArray()).float()
        if ct_tensor.dim() == 3:
            ct_tensor = ct_tensor.unsqueeze(0)
        elif ct_tensor.dim() == 4 and ct_tensor.shape[0] != 1:
            ct_tensor = ct_tensor[0:1]

        ell_map = adaptive_ell_map(
            ct_tensor,
            spacing=tuple(spacing),
            l0=(args.l0_z, args.l0_y, args.l0_x),
            alpha=args.alpha,
            theta=theta,
        )
        print(f"  ell_map range: [{ell_map.min():.2f}, {ell_map.max():.2f}] voxels")

        effective_gain = args.variability_gain
        folding_threshold = 0.0001
        print(f"  variability_gain = {effective_gain}, folding_threshold = {folding_threshold}")

        min_dim = min(target_shape)
        effective_kernel = args.kernel_size
        if effective_kernel > min_dim:
            effective_kernel = min_dim if min_dim % 2 == 1 else min_dim - 1
            effective_kernel = max(effective_kernel, 3)
            print(f"  Kernel clamped to {effective_kernel}")

        # GPU exclusion: comma-separated list in env var DVF_EXCLUDE_GPUS.
        # Default empty (all CUDA devices used). To reproduce the workstation
        # default of the original development setup, set DVF_EXCLUDE_GPUS=2.
        _env = os.environ.get("DVF_EXCLUDE_GPUS", "").strip()
        exclude_gpus = [int(x) for x in _env.split(",") if x.strip().isdigit()]
        sampler = DVFSampler(
            n_samples=args.n_samples,
            kernel_size=effective_kernel,
            variability_gain=effective_gain,
            max_retries=args.max_retries,
            folding_threshold=folding_threshold,
            exclude_gpus=exclude_gpus,
        )
        # FC mode: generate N deterministic white-noise tensors from the
        # shared seed; all fractions of the treatment receive the same
        # epsilon_k (Smolders 2023 §2.7.3).
        shared_white_noise = None
        if args.noise_seed is not None:
            Z, Y, X = mu_dvf.shape[-3:]
            gen = torch.Generator().manual_seed(int(args.noise_seed))
            shared_white_noise = DVFSampler.draw_shared_noise(
                args.n_samples, (Z, Y, X), generator=gen)
            print(f"  [FC] noise_seed={args.noise_seed} -> {args.n_samples} shared eps "
                  f"of shape (3, {Z}, {Y}, {X})")
        results = sampler.generate(mu_dvf, sigma_dvf, ell_map, args.out_dir,
                                    shared_white_noise=shared_white_noise)

        np.save(os.path.join(args.out_dir, "ell_map.npy"), ell_map.cpu().numpy())

        metadata = {
            "n_samples": args.n_samples,
            "l0": [args.l0_x, args.l0_y, args.l0_z],
            "alpha": args.alpha,
            "theta": theta,
            "variability_gain": effective_gain,
            "kernel_size": effective_kernel,
            "max_retries": args.max_retries,
            "folding_threshold": folding_threshold,
            "mean_folding_rate": results["folding_stats"]["mean_folding"],
            "uncertainty_method": args.uncertainty_method,
            "spacing": list(spacing),
            # Marker for FC mode: deterministic noise_seed, None for NC.
            "noise_seed": args.noise_seed,
            "heuristic_params": {
                "w_res_grad": args.heuristic_w,
                "w_jac": 1.0 - args.heuristic_w,
                "scale": args.heuristic_scale,
                "jac_clip": args.heuristic_jac_clip,
                "ratio_clip": args.heuristic_ratio_clip,
                "use_minmax": args.heuristic_use_minmax,
                "suppression_alpha": args.heuristic_suppression_alpha,
            } if args.uncertainty_method == "heuristic" else None,
        }
        with open(os.path.join(args.out_dir, "pass3_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"  [OK] PASS 3 — {args.n_samples} samples generated")
        torch.cuda.empty_cache()

    # ================================================================
    # PASS 4: Dose Deformation
    # ================================================================
    if args.mode in ["dose", "all"] and args.moving_dose and os.path.exists(args.moving_dose):
        print(f"\n{'=' * 60}\nPASS 4: Dose Deformation\n{'=' * 60}")

        fixed_image = Scan(args.fixed)
        _apply_predef_rigid(fixed_image, args.dvf)
        spacing = fixed_image.getVoxelSpacing()
        dose_scan = Scan(args.moving_dose)
        dose_aligned = resampleScan(dose_scan, fixed_image.getVoxelToWorldMatrix(), fixed_image.getShape())
        dose_volume = toNumpy(dose_aligned.getPixelArray())
        if dose_volume.ndim == 4:
            dose_volume = dose_volume[0]
        dose_zyx = np.transpose(dose_volume, (2, 1, 0)).astype(np.float32)

        ensemble_path = os.path.join(args.out_dir, "dvf_ensemble.npy")
        if not os.path.exists(ensemble_path):
            print("  ERROR: DVF ensemble not found. Run PASS 3 first.")
            return

        propagator = DosePropagator(output_dir=args.out_dir)
        propagator.propagate(dose_zyx, ensemble_path, tuple(spacing))
        print("  [OK] PASS 4")

    print(f"\n{'=' * 60}\n[DONE]\n{'=' * 60}")


if __name__ == "__main__":
    main()