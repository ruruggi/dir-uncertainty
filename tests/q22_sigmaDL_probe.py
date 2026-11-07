# -*- coding: utf-8 -*-
"""q22_sigmaDL_probe.py -- read-only probe of the DVF fed to the Smolders DL network (fixedMean: the input DVF IS the mean flow,
VoxelMorph.correctMeanField, applied by the STN in network-grid voxels, channel i along array axis i).

On the network grid (after FixedMeanPatchRegisterer.preProcess) the moving image is warped by the network's own STN with four
versions of the input DVF; only the network's convention registers. Residual = mean |warped - fixed| where fixed > its 5th
percentile + 10 % of range (body proxy on the normalised network input), ratio to no warp.
  A  channels as stored [dz,dy,dx], mm           (current run_one)
  B  channels reversed [dx,dy,dz],  mm
  C  channels reversed [dx,dy,dz],  fixed voxels  (candidate fix; interpolateVectorField then rescales to network voxels)
  D  channels as stored,            fixed voxels
Castillo cases 1, 5, 8 x Elastix. Also prints the grids/spacings seen by postProcess (sigma units).
"""
from dirunc_paths import LIB, MODELS  # data roots, see README
import os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = LIB; PS = LIB + "/validation"
sys.path.insert(0, SCRIPTS); sys.path.insert(0, os.path.join(SCRIPTS, "dl_arch")); sys.path.insert(0, PS)
if "validation" not in sys.modules:
    _v = types.ModuleType("validation"); _v.__path__ = [PS]; sys.modules["validation"] = _v
import numpy as np, torch
import smolders_dl_sigma_castillo as smo
from utils.scan import Scan
from utils.vector_field import VectorField
from Registration.fixedMeanRegistration import FixedMeanPatchRegisterer
from Model.STN import SpatialTransformer

MODEL = os.path.join(rf"{MODELS}", smo.MODEL_NAME)   # not in the submitted snapshot
reg = FixedMeanPatchRegisterer(MODEL, patchSize=(256, 256, 96), voxelSpacing=(1.9531, 1.9531, 2.0))   # = dl.model_loader defaults
for cid in ("case1", "case5", "case8"):
    pair = smo.load_pair(smo.CASTILLO_NIFTI, case_id=cid)
    pair, _, _ = smo._apply_procrustes_to_pair(pair, os.path.join(HERE, "_tmp_diag"), save_artifacts=False)
    sp = np.asarray(pair.spacing_fixed, float); vtw = np.diag([*sp, 1.0])
    fs = Scan(array=pair.fixed_arr[None], vtwMatrix=vtw); ms = Scan(array=pair.moving_arr[None], vtwMatrix=vtw)
    dvf = np.load(os.path.join(smo.CASTILLO_BASE, "Results_elastix_procrustes", cid, "dvf_mu.npy")).astype(np.float32)
    rev = dvf[::-1].copy()
    variants = {"A stored/mm": dvf, "B reversed/mm": rev, "C reversed/voxels": rev / sp[:, None, None, None].astype(np.float32),
                "D stored/voxels": dvf / sp[::-1, None, None, None].astype(np.float32)}
    out = []
    for name, arr in variants.items():
        inp = reg.preProcess(fs, ms, VectorField(array=arr, vtwMatrix=vtw, inMM=True))
        fix, mov, flow = inp[:, 0:1], inp[:, 1:2], inp[:, 2:5]
        stn = SpatialTransformer(tuple(fix.shape[-3:])).to(fix.device)
        m = fix > (torch.quantile(fix.flatten()[::97], 0.05) + 0.1 * (fix.max() - fix.min()))
        r = float((stn(mov, flow) - fix).abs()[m].mean() / (mov - fix).abs()[m].mean())
        out.append(f"{name} {r:.3f}")
    print(cid, "grid", tuple(inp.shape[-3:]), "| fixed spacing", sp.round(3), "|", " ; ".join(out), flush=True)
