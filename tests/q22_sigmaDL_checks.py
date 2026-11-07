# -*- coding: utf-8 -*-
"""
q22_sigmaDL_checks.py -- step 2 of the recalculation order (audit N1): conventions of the sigma_DL arm (Table 3), read-only.

  1. check_tre   : TRE of smolders_dl_sigma_castillo._sample_tre_via_dvf (known-bad) vs lece.evaluate_landmarks (candidate fix),
                   both against the stored landmarks.npz tre_mm.
  2. check_warp  : channel convention of the DL network. Its SpatialTransformer (dl_arch/Model/STN.py, the warper of the
                   unsupervised training loss) adds flow channel i to array axis i. The network works on (X,Y,Z) arrays, so it
                   was trained with [dx,dy,dz]. dvf_mu.npy is (3,X,Y,Z) with channels [dz,dy,dx] (conventions_check.py) and is
                   passed as is by run_one(). Test: STN-warp the moving image with the stored DVF as is vs channel-reversed;
                   only the network's own convention registers (residual in the body <= 0.7 x no warp).

Castillo cases 1, 5, 8 (case 8 = worst N1 ratio) x Elastix / ConvexAdam. Usage (project venv): python q22_sigmaDL_checks.py [cuda:0]
"""
from dirunc_paths import DATASETS, LIB  # data roots, see README
import os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = LIB; PS = LIB + "/validation"
sys.path.insert(0, SCRIPTS); sys.path.insert(0, PS); sys.path.insert(0, HERE)
if "validation" not in sys.modules:
    _v = types.ModuleType("validation"); _v.__path__ = [PS]; sys.modules["validation"] = _v
import numpy as np, torch
from conventions_check import check_tre, check_warp
from validation.load_abdominal_dir_qa import load_pair
from validation.run_pair import _apply_procrustes_to_pair, _resample_moving_to_fixed
from runner import _estimate_body_mask
from lece import evaluate_landmarks
import smolders_dl_sigma_castillo as smo
sys.path.insert(0, os.path.join(SCRIPTS, "dl_arch"))
from Model.STN import SpatialTransformer

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
D = rf"{DATASETS}/Castillo-DIR-LAB"
for cid in ("case1", "case5", "case8"):
    pair = load_pair(os.path.join(D, "nifti"), case_id=cid)
    pair, _, _ = _apply_procrustes_to_pair(pair, os.path.join(HERE, "_tmp_diag"), save_artifacts=False)
    mov = _resample_moving_to_fixed(pair.fixed_arr, pair.moving_arr, pair.spacing_fixed, pair.spacing_moving,
                                    fixed_affine=getattr(pair, "fixed_affine", None), moving_affine=getattr(pair, "moving_affine", None))
    sx, sy, sz = pair.spacing_fixed
    F = torch.from_numpy(pair.fixed_arr.astype(np.float32)).to(dev)            # (X,Y,Z), as fed to the network
    M = torch.from_numpy(mov.astype(np.float32)).to(dev)
    B = torch.from_numpy(_estimate_body_mask(pair.fixed_arr.astype(np.float32).transpose(2, 1, 0)).transpose(2, 1, 0).copy()).to(dev)
    stn = SpatialTransformer(tuple(F.shape)).to(dev)
    for eng in ("elastix", "convexadam"):
        R_ = os.path.join(D, f"Results_{eng}_procrustes", cid)
        dvf = np.load(os.path.join(R_, "dvf_mu.npy")).astype(np.float32)       # (3,X,Y,Z), [dz,dy,dx] mm
        tre_ref = np.load(os.path.join(R_, "landmarks.npz"))["tre_mm"]
        ok0, d0 = check_tre(smo._sample_tre_via_dvf(dvf, pair), tre_ref)
        ev = evaluate_landmarks(dvf_mm=dvf, sigma_mm=np.ones(dvf.shape[1:], np.float32), spacing_fixed=pair.spacing_fixed,
                                lm_fixed_voxel=pair.lm_fixed_voxel, lm_moving_world_true=pair.lm_moving_world, case_id=cid,
                                fixed_affine=getattr(pair, "fixed_affine", None))
        ok1, d1 = check_tre(ev.tre_mm, tre_ref)
        asis = torch.from_numpy(dvf).to(dev).clone(); asis[0] /= sz; asis[1] /= sy; asis[2] /= sx       # voxels, channels as stored
        rev = torch.from_numpy(dvf[::-1].copy()).to(dev); rev[0] /= sx; rev[1] /= sy; rev[2] /= sz       # voxels, [dx,dy,dz]
        w = lambda v: stn(M[None, None], v[None])[0, 0]
        ok2, r2 = check_warp(F, M, w(asis), B); ok3, r3 = check_warp(F, M, w(rev), B)
        print(f"{cid} {eng:10s} TRE sample_tre_via_dvf max|d| {d0:7.3f} {'PASS' if ok0 else 'FAIL'} | evaluate_landmarks {d1:.1e} "
              f"{'PASS' if ok1 else 'FAIL'} | STN warp as stored {r2:.3f} {'PASS' if ok2 else 'FAIL'}, reversed [dx,dy,dz] {r3:.3f} "
              f"{'PASS' if ok3 else 'FAIL'}", flush=True)
