# -*- coding: utf-8 -*-
"""
q22_rho_components_liver.py -- read-only diagnostic (2026-09-28): why rho(sigma_BL, TRE) < 0 for uniGradICON on Liver-DIR-QA.
Recomputes the components of conventions_check.sigma_body_local at the landmarks -- MIND-SSC residual, suppression factor
M = norm|grad I_m|, R~ (clipped, min-max), d~ (|det J - 1| clipped, min-max) -- for one engine on the 30 Liver cases, and checks
that w R~ + (1-w) d~ reproduces the stored sigma_BL landmarks (control). Output ./_out_q22_rho_liver/<engine>.npz + summary line.
Usage (venv): python q22_rho_components_liver.py <engine: unigradicon|elastix|adam> [cuda:0]
"""
from dirunc_paths import DATASETS, LIB  # data roots, see README
import os, sys, types, json
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = LIB; PS = LIB + "/validation"
sys.path.insert(0, SCRIPTS); sys.path.insert(0, PS); sys.path.insert(0, HERE)
v = types.ModuleType("validation"); v.__path__ = [PS]; sys.modules["validation"] = v
import numpy as np, torch
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from runner import _estimate_body_mask, GRADIENT_FUNC
from core.similarity_metrics import compute_residual
from core.jacobian import jacobian_determinant
from utils.warp import warp_torch
from conventions_check import _quantile, _minmax
from validation.load_abdominal_dir_qa import load_pair
from validation.run_pair import _apply_procrustes_to_pair, _resample_moving_to_fixed
from lece import predict_landmark_sigma

eng = sys.argv[1]; dev = torch.device(sys.argv[2] if len(sys.argv) > 2 else "cuda:0")
SRC = f"{DATASETS}/Liver-DIR-QA/Results_{eng}_procrustes"
BL = f"{DATASETS}/_sigma_BL_20260927/Liver-DIR-QA/Results_{eng}_procrustes"
OUT = os.path.join(HERE, "_out_q22_rho_liver"); os.makedirs(OUT, exist_ok=True)
w, clip_r, clip_j = 0.7, 0.99, 0.95
rec = {k: [] for k in ("case", "tre", "sigma_bl", "res", "M", "R", "d", "Rn", "dn", "recon")}
for c in sorted(os.listdir(SRC), key=lambda s: (len(s), s)):
    d0 = os.path.join(SRC, c)
    if not os.path.isfile(os.path.join(d0, "dvf_mu.npy")):
        continue
    S = json.load(open(os.path.join(d0, "summary.json")))
    pair = load_pair(S["case_dir"], case_id=S["case_id"])
    pair, _, _ = _apply_procrustes_to_pair(pair, os.path.join(OUT, "_tmp"), save_artifacts=False)
    mov = _resample_moving_to_fixed(pair.fixed_arr, pair.moving_arr, pair.spacing_fixed, pair.spacing_moving,
                                    fixed_affine=getattr(pair, "fixed_affine", None), moving_affine=getattr(pair, "moving_affine", None))
    sx, sy, sz = pair.spacing_fixed; sp = (sz, sy, sx)
    fz = np.ascontiguousarray(pair.fixed_arr.astype(np.float32).transpose(2, 1, 0)); mz = np.ascontiguousarray(mov.astype(np.float32).transpose(2, 1, 0))
    body = torch.from_numpy(_estimate_body_mask(fz)).to(dev); bm = torch.from_numpy(_estimate_body_mask(mz)).to(dev)
    fix = torch.from_numpy(fz).to(dev); mv = torch.from_numpy(mz).to(dev)
    dvf_mm = torch.from_numpy(np.ascontiguousarray(np.load(os.path.join(d0, "dvf_mu.npy")).astype(np.float32).transpose(0, 3, 2, 1))).to(dev)
    # --- identical to conventions_check.sigma_body_local, keeping the intermediate fields ---
    f = torch.where(body, fix, torch.full_like(fix, -1000.0)); m = torch.where(bm, mv, torch.full_like(mv, -1000.0))
    idx = ndi.distance_transform_edt(~body.cpu().numpy(), return_distances=False, return_indices=True)
    dvf = torch.from_numpy(np.ascontiguousarray(dvf_mm.cpu().numpy()[:, idx[0], idx[1], idx[2]])).to(dev); del idx
    dd = dvf.clone()
    for k in range(3):
        dd[k] = dd[k] / sp[k]
    res = compute_residual(f.unsqueeze(0), warp_torch(m, dd).unsqueeze(0), metric="mind_ssc", radius=1, dilation=2).squeeze(0)
    gn = _minmax(GRADIENT_FUNC(m), bm)
    R = (1.0 - res * gn).clamp(0, 1); R = torch.minimum(R, _quantile(R[body], clip_r))
    J = (jacobian_determinant(dd) - 1.0).abs(); J = torch.minimum(J, _quantile(J[body], clip_j))
    Rn, dn = _minmax(R, body), _minmax(J, body)
    s = torch.where(body, w * Rn + (1 - w) * dn, torch.zeros_like(Rn))
    lm = np.load(os.path.join(d0, "landmarks.npz")); lmb = np.load(os.path.join(BL, c, "landmarks.npz"))
    at = lambda t: predict_landmark_sigma(np.ascontiguousarray(t.squeeze().cpu().numpy().transpose(2, 1, 0)).astype(np.float32), lm["lm_fixed_voxel"])
    recon = at(s)
    assert np.max(np.abs(recon - lmb["sigma_p_mm_raw"])) < 1e-4, f"{c}: recomposition != stored sigma_BL"
    for k, t in (("res", res), ("M", gn), ("R", R), ("d", J), ("Rn", Rn), ("dn", dn)):
        rec[k].append(at(t))
    rec["recon"].append(recon); rec["sigma_bl"].append(lmb["sigma_p_mm_raw"]); rec["tre"].append(lm["tre_mm"]); rec["case"].append(np.full(len(recon), int(c[4:])))
    print(f"  {eng} {c}: n {len(recon)}  rho(sigma,TRE) {spearmanr(recon, lm['tre_mm'])[0]:+.3f}", flush=True)
    del f, m, dvf, dd, res, gn, R, J, Rn, dn, s, fix, mv, dvf_mm; torch.cuda.empty_cache()
A = {k: np.concatenate(vv) for k, vv in rec.items()}
np.savez(os.path.join(OUT, f"{eng}.npz"), **A)
pc = lambda x: np.nanmean([spearmanr(A[x][A["case"] == i], A["tre"][A["case"] == i])[0] for i in np.unique(A["case"])])
print(f"[{eng}] control recon == sigma_BL: PASS | rho with TRE (pool / per-case mean): " +
      " ; ".join(f"{k} {spearmanr(A[k], A['tre'])[0]:+.3f}/{pc(k):+.3f}" for k in ("sigma_bl", "Rn", "dn", "res", "M")), flush=True)
