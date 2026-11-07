# -*- coding: utf-8 -*-
"""
conventions_check.py -- automatic checks of the array conventions of the study (user, 2026-09-27), each validated by a
known-bad (negative) and a known-good (positive) control. Every script that feeds a table or figure must pass them
before its numbers are used.

  check_warp        the warp used for the residual must register: mean |I_m o phi - I_f| in the body <= 0.7 x no warp
  check_kernel_axes the correlation lengths of K, in mm, must put l0_Z on the cranio-caudal axis (ratio CC/lateral = 1.5)
  check_tre         any TRE computation must reproduce the stored per-landmark TRE (max |d| < 1e-3 mm)
  check_body_independence  a sigma function must not depend on voxels outside the body (user premise): perturbing
                    fixed/moving/DVF outside the body must leave sigma inside unchanged (max |d| < 1e-4)

Conventions (lece.py, runner.py production): images (Z,Y,X); DVF (3,Z,Y,X) with channels [dz,dy,dx] in mm; voxel
spacing (sz,sy,sx); warp_torch and jacobian_determinant apply channel i along array axis i. The May batches pass
(X,Y,Z) arrays with the same channel order -> they fail check_warp (known-bad control).

sigma_body_local: reference heuristic sigma computed on the body only (candidate, not yet approved): outside-body
image voxels set to -1000 HU (fixed: its body; moving: its own body), DVF extended outside the body by its nearest
body value (no step at the skin), grad-norm, clip quantiles (exact, deterministic) and min-max on body voxels, sigma = 0
outside. Same formula as runner.compute_heuristic_uncertainty (w = 0.7, R = 1 - a*residual*grad_norm, MIND-SSC r1 d2).

Self-test (python conventions_check.py, project venv): runs every check with its controls on Castillo case 5 / Elastix
and on the Figure 3 run, and prints PASS/FAIL. Read-only.
"""
from dirunc_paths import DATASETS, LIB, LUNG_SBRT  # data roots, see README
import os, sys, types, json
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = LIB; PS = LIB + "/validation"
sys.path.insert(0, SCRIPTS); sys.path.insert(0, PS)
if "validation" not in sys.modules:
    _v = types.ModuleType("validation"); _v.__path__ = [PS]; sys.modules["validation"] = _v
import numpy as np, torch
from scipy import ndimage as ndi
from utils.warp import warp_torch
from core.jacobian import jacobian_determinant


def _res(a, b, m):
    return float((a - b).abs()[m].mean())


def warp_zyx(mov_zyx, dvf_zyx_mm, spacing_zyx):
    """Production convention: (Z,Y,X) image, DVF (3,Z,Y,X) [dz,dy,dx] mm -> voxels -> warp_torch."""
    d = dvf_zyx_mm.clone()
    for c in range(3):
        d[c] = d[c] / spacing_zyx[c]
    return warp_torch(mov_zyx, d)


def check_warp(fix, mov, warped, body):
    """fix, mov, warped: tensors of the same layout; body: bool tensor. Returns (ok, residual ratio)."""
    r = _res(warped, fix, body) / _res(mov, fix, body)
    return r <= 0.7, r


def check_kernel_axes(ell_vox, spacing_per_axis, cc_axis, expected=1.5, tol=0.05):
    """ell_vox (3, ...): channel c = length along array axis c, in voxels (core.grf). Returns (ok, CC/lateral in mm)."""
    mm = [float(np.median(np.asarray(ell_vox[c]))) * spacing_per_axis[c] for c in range(3)]
    lat = np.mean([mm[c] for c in range(3) if c != cc_axis])
    r = mm[cc_axis] / lat
    return abs(r - expected) < tol, r


def check_tre(tre, tre_ref, tol=1e-3):
    d = float(np.max(np.abs(np.asarray(tre) - np.asarray(tre_ref))))
    return d < tol, d


def check_body_independence(sigma_fn, fix, mov, dvf, body, body_mov, seed=0, tol=1e-4):
    """sigma_fn(fix, mov, dvf, body, body_mov) -> sigma (Z,Y,X) tensor. Perturb everything outside the bodies."""
    g = torch.Generator(device=fix.device); g.manual_seed(seed)
    out_f, out_m = ~body, ~body_mov
    f2 = torch.where(out_f, torch.rand(fix.shape, generator=g, device=fix.device) * 1500 - 1000, fix)
    m2 = torch.where(out_m, torch.rand(mov.shape, generator=g, device=fix.device) * 1500 - 1000, mov)
    d2 = torch.where(out_f.unsqueeze(0), torch.randn(dvf.shape, generator=g, device=fix.device) * 5, dvf)
    s1 = sigma_fn(fix, mov, dvf, body, body_mov); s2 = sigma_fn(f2, m2, d2, body, body_mov)
    d = float((s1 - s2).abs()[body].max())
    return d < tol, d


def _quantile(v, q):
    k = max(1, int(round(q * v.numel())))
    return torch.kthvalue(v.flatten().float().cpu(), k).values.to(v.device)


def _minmax(t, m):
    lo, hi = t[m].min(), t[m].max()
    return torch.where(m, ((t - lo) / (hi - lo if hi - lo > 1e-8 else 1.0)).clamp(0, 1), torch.zeros_like(t))


def sigma_body_local(fix, mov, dvf_mm, body, body_mov, spacing_zyx, w=0.7, clip_r=None, clip_j=0.95, alpha=1.0,
                     metric="mind_ssc", metric_kw=None):
    """Reference sigma on the body only; tensors (Z,Y,X), DVF (3,Z,Y,X) [dz,dy,dx] mm. See module docstring."""
    from runner import GRADIENT_FUNC
    from core.similarity_metrics import compute_residual
    f = torch.where(body, fix, torch.full_like(fix, -1000.0)); m = torch.where(body_mov, mov, torch.full_like(mov, -1000.0))
    idx = ndi.distance_transform_edt(~body.cpu().numpy(), return_distances=False, return_indices=True)   # gather in RAM
    dvf = torch.from_numpy(np.ascontiguousarray(dvf_mm.cpu().numpy()[:, idx[0], idx[1], idx[2]])).to(dvf_mm.device); del idx
    d = dvf.clone()
    for c in range(3):
        d[c] = d[c] / spacing_zyx[c]
    wm = warp_torch(m, d)
    res = compute_residual(f.unsqueeze(0), wm.unsqueeze(0), metric=metric, **(metric_kw or {"radius": 1, "dilation": 2})).squeeze(0)
    gm = GRADIENT_FUNC(m); gn = _minmax(gm, body_mov)
    R = (1.0 - alpha * res * gn).clamp(0, 1)
    if clip_r is not None:                    # IV: no clip of R (Alg. 1); clip_r=0.99 reproduces the III sigma_BL
        R = torch.minimum(R, _quantile(R[body], clip_r))
    J = (jacobian_determinant(d) - 1.0).abs()
    J = torch.minimum(J, _quantile(J[body], clip_j))
    s = w * _minmax(R, body) + (1 - w) * _minmax(J, body)
    return torch.where(body, s, torch.zeros_like(s))


if __name__ == "__main__":
    from runner import compute_heuristic_uncertainty, _estimate_body_mask
    from validation.load_abdominal_dir_qa import load_pair
    from validation.run_pair import _apply_procrustes_to_pair, _resample_moving_to_fixed
    from lece import evaluate_landmarks
    from core.grf import adaptive_ell_map
    dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
    D = rf"{DATASETS}/Castillo-DIR-LAB"; R_ = os.path.join(D, "Results_elastix_procrustes", "case5")
    pair = load_pair(os.path.join(D, "nifti"), case_id="case5")
    pair, _, _ = _apply_procrustes_to_pair(pair, os.path.join(HERE, "_tmp_diag"), save_artifacts=False)
    mov = _resample_moving_to_fixed(pair.fixed_arr, pair.moving_arr, pair.spacing_fixed, pair.spacing_moving,
                                    fixed_affine=getattr(pair, "fixed_affine", None), moving_affine=getattr(pair, "moving_affine", None))
    sx, sy, sz = pair.spacing_fixed; sp = (sz, sy, sx)
    fz = np.ascontiguousarray(pair.fixed_arr.astype(np.float32).transpose(2, 1, 0)); mz = np.ascontiguousarray(mov.astype(np.float32).transpose(2, 1, 0))
    bz = _estimate_body_mask(fz); bmz = _estimate_body_mask(mz)
    dvf_xyz = np.load(os.path.join(R_, "dvf_mu.npy")).astype(np.float32)
    F = torch.from_numpy(fz).to(dev); M = torch.from_numpy(mz).to(dev); B = torch.from_numpy(bz).to(dev); BM = torch.from_numpy(bmz).to(dev)
    DZ = torch.from_numpy(np.ascontiguousarray(dvf_xyz.transpose(0, 3, 2, 1))).to(dev)
    out = []
    def rep(name, expect, ok, val):
        out.append((name, expect, ok, val)); print(f"{'OK ' if ok == (expect == 'PASS') else 'XX '} {name:62s} expected {expect}: {'PASS' if ok else 'FAIL'}  ({val:.4g})", flush=True)

    # 1. warp: May batch convention (X,Y,Z arrays, same channels) vs production (Z,Y,X)
    Fx = F.permute(2, 1, 0).contiguous(); Mx = M.permute(2, 1, 0).contiguous(); Bx = B.permute(2, 1, 0)
    dv = torch.from_numpy(dvf_xyz).to(dev).clone(); dv[0] /= sz; dv[1] /= sy; dv[2] /= sx
    rep("warp, May batch call (X,Y,Z)", "FAIL", *check_warp(Fx, Mx, warp_torch(Mx, dv), Bx))
    rep("warp, production convention (Z,Y,X)", "PASS", *check_warp(F, M, warp_zyx(M, DZ, sp), B))
    # 2. kernel axes: q21 test (Z,Y,X) vs the Figure 3 production run (X,Y,Z)
    ell = adaptive_ell_map(F.unsqueeze(0), sp, l0=(15.0, 10.0, 10.0), alpha=0.5, theta=30.0).cpu().numpy()
    rep("K axes, adaptive_ell_map on (Z,Y,X), q21 test", "PASS", *check_kernel_axes(ell, sp, cc_axis=0))
    d3 = rf"{LUNG_SBRT}/Results100_isoalign/Heuristic/frac_01"
    e3 = np.load(d3 + "/ell_map.npy", mmap_mode="r"); sp3 = json.load(open(d3 + "/pass3_metadata.json"))["spacing"]
    rep("K axes, Figure 3 production ell_map (array X,Y,Z)", "FAIL", *check_kernel_axes(e3[:, ::4, ::4, ::2], sp3, cc_axis=2))
    # 3. TRE: lece.evaluate_landmarks vs the sigma_DL helper of Table 3
    lm = np.load(os.path.join(R_, "landmarks.npz"))
    ev = evaluate_landmarks(dvf_mm=dvf_xyz, sigma_mm=np.ones(dvf_xyz.shape[1:], np.float32), spacing_fixed=pair.spacing_fixed,
                            lm_fixed_voxel=pair.lm_fixed_voxel, lm_moving_world_true=pair.lm_moving_world, case_id="case5",
                            fixed_affine=getattr(pair, "fixed_affine", None))
    rep("TRE, lece.evaluate_landmarks", "PASS", *check_tre(ev.tre_mm, lm["tre_mm"]))
    import smolders_dl_sigma_castillo as smo
    rep("TRE, smolders_dl_sigma_castillo._sample_tre_via_dvf (Table 3 sigma_DL)", "FAIL", *check_tre(smo._sample_tre_via_dvf(dvf_xyz, pair), lm["tre_mm"]))
    # 4. body independence: submitted runner with body mask (S3) vs the body-local reference
    kw = dict(w_res_grad=0.7, w_jac=0.3, ratio_clip_percentile=0.99, jac_clip_percentile=0.95, use_minmax_norm=True, global_scale=1.0,
              suppression_alpha=1.0, similarity_metric="mind_ssc", similarity_kwargs={"radius": 1, "dilation": 2})
    def s3(f, m, d, b, bm):
        torch.manual_seed(0)
        return compute_heuristic_uncertainty(fixed_ct=f.unsqueeze(0), moving_ct=m.unsqueeze(0), dvf_mm=d, voxel_spacing=sp, body_mask=b, **kw).squeeze()
    rep("body independence, runner.compute_heuristic_uncertainty (S3)", "FAIL?", *check_body_independence(s3, F, M, DZ, B, BM))
    rep("body independence, sigma_body_local (reference)", "PASS", *check_body_independence(lambda f, m, d, b, bm: sigma_body_local(f, m, d, b, bm, sp), F, M, DZ, B, BM))
    print("controls behaving as expected:", sum(ok == (e == "PASS") for _, e, ok, _ in out if e in ("PASS", "FAIL")), "/", sum(e in ("PASS", "FAIL") for _, e, _, _ in out))
