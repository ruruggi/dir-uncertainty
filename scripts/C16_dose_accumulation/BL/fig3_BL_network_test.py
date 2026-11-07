# -*- coding: utf-8 -*-
"""
fig3_BL_network_test.py -- STEP 8 (Figure 3), test 2 (2026-09-28): the DVF given to the Smolders DL network (L20S3-4,
FixedMeanPatchRegisterer) on the Figure 3 case. Read-only.

Input under test = the corrected mu of test 1 (fixed-point pull-back inverse of the raw DICOM field) in plan-CT voxels,
(3,X,Y,Z) with channel i along array axis i ([dx,dy,dz]), passed as q22_recompute_sigmaDL.infer does (variant C). After
reg.preProcess the network sees [fixed, moving, dvf] on its 1.95/1.95/2.0 mm grid; its SpatialTransformer (dl_arch/Model/STN.py)
adds flow channel i to axis i. Check: STN(moving, dvf) must register to fixed on that grid (residual in the body <= 0.7 x no
warp), and the unit scale 1 must beat 0.5 and 2 (a units error would move the optimum); channel-reversed input as negative control.
Then one full inference: sigma_DL in mm = stdev x network-grid spacing (q22 correction S), body statistics.
Usage (venv): python fig3_BL_network_test.py [frac] [cuda:0]
"""
from dirunc_paths import LIB, MODELS  # data roots, see README
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = LIB
sys.path.insert(0, HERE); sys.path.insert(0, SCRIPTS); sys.path.insert(0, os.path.join(SCRIPTS, "dl_arch"))
import numpy as np, torch
import torch.nn.functional as F
import fig3_BL_conventions_test as T
from runner import _estimate_body_mask
from utils.scan import Scan
from utils.vector_field import VectorField
from Registration.fixedMeanRegistration import FixedMeanPatchRegisterer
from Model.STN import SpatialTransformer

MODEL = os.path.join(rf"{MODELS}", "L20S3-4")   # = dl.model_loader default
INV_ITERS = 10


def to_np(a):
    return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)


def network_inputs(fixed, fx, mv, u_vox, dev):
    """Scans and VectorField as q22_recompute_sigmaDL.infer (variant C): plan grid, vtw = diag(spacing)."""
    sp = np.asarray(fixed.getVoxelSpacing(), float); vtw = np.diag([*sp, 1.0])
    fs = Scan(array=fx[None], vtwMatrix=vtw); ms = Scan(array=mv[None], vtwMatrix=vtw)
    return fs, ms, vtw, sp


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    dev = torch.device(sys.argv[2] if len(sys.argv) > 2 else "cuda:0"); torch.cuda.set_device(dev)
    fixed, fx, mv, _ = T.load_case(n)
    body = np.ascontiguousarray(_estimate_body_mask(fx.transpose(2, 1, 0)).transpose(2, 1, 0))
    vox_raw, _, _ = T.raw_dvf(n, fixed)
    u = T.pullback_inverse(vox_raw, INV_ITERS, dev).astype(np.float32)
    fs, ms, vtw, sp = network_inputs(fixed, fx, mv, u, dev)
    reg = FixedMeanPatchRegisterer(MODEL, patchSize=(256, 256, 96), voxelSpacing=(1.9531, 1.9531, 2.0))
    rep = {"frac": n, "stn_ratio": {}}
    for tag, arr in (("x1", u), ("x0.5", 0.5 * u), ("x2", 2 * u), ("reversed", np.ascontiguousarray(u[::-1]))):
        inp = reg.preProcess(fs, ms, VectorField(array=np.ascontiguousarray(arr), vtwMatrix=vtw, inMM=True))
        fN, mN, dN = inp[0, 0], inp[0, 1], inp[0, 2:5]
        if tag == "x1":
            bN = F.interpolate(torch.from_numpy(body.astype(np.float32))[None, None], size=tuple(fN.shape), mode="trilinear",
                               align_corners=True)[0, 0].to(fN.device) > 0.5
            stn = SpatialTransformer(tuple(fN.shape)).to(fN.device)
            rep["net_shape"] = list(fN.shape); none = float((mN - fN).abs()[bN].mean())
        w = stn(mN[None, None], dN[None])[0, 0]
        rep["stn_ratio"][tag] = float((w - fN).abs()[bN].mean()) / none
        print(f"  frac {n} network input {tag:8s}: STN residual ratio {rep['stn_ratio'][tag]:.3f} "
              f"{'PASS' if rep['stn_ratio'][tag] <= 0.7 else 'fail'}", flush=True)
    del inp, fN, mN, dN, w; torch.cuda.empty_cache()
    # full inference, corrected wiring (= q22_recompute_sigmaDL.infer, S = network-grid spacing)
    inp = reg.preProcess(fs, ms, VectorField(array=u, vtwMatrix=vtw, inMM=True))
    res = reg.resampleVectorField(reg.register(inp), fs.getShape())
    n_net = np.asarray(inp.shape[-3:], float); n_fix = np.asarray(fs.getShape(), float)
    s_net = (n_fix - 1) * sp / (n_net - 1)
    sig = to_np(res.stdev).astype(np.float32) * s_net.reshape(3, 1, 1, 1).astype(np.float32)     # (3,X,Y,Z) mm, [sx,sy,sz]
    mag = np.linalg.norm(sig, axis=0); b = body.astype(bool)
    rep["s_net"] = s_net.tolist()
    rep["sigmaDL_body"] = {"comp_mean_mm": [float(sig[c][b].mean()) for c in range(3)],
                           "mag_mean_mm": float(mag[b].mean()), "mag_p50": float(np.median(mag[b])), "mag_p99": float(np.percentile(mag[b], 99))}
    print(f"  sigma_DL (corrected) body: components x/y/z {', '.join(f'{v:.2f}' for v in rep['sigmaDL_body']['comp_mean_mm'])} mm, "
          f"|sigma| mean {rep['sigmaDL_body']['mag_mean_mm']:.2f} p50 {rep['sigmaDL_body']['mag_p50']:.2f} p99 {rep['sigmaDL_body']['mag_p99']:.2f}", flush=True)
    prod = os.path.join(T.DATA, "Results100_isoalign", "Smolders", f"frac_0{n}", "uncertainty_dir.npy")
    if os.path.exists(prod):
        pm = np.load(prod).transpose(2, 1, 0)                                       # production (Z,Y,X) -> (X,Y,Z)
        rep["sigmaDL_production_body_mag"] = {"mean": float(pm[b].mean()), "p50": float(np.median(pm[b])), "p99": float(np.percentile(pm[b], 99))}
        print(f"  sigma_DL production |sigma| body mean {pm[b].mean():.2f} p50 {np.median(pm[b]):.2f} p99 {np.percentile(pm[b], 99):.2f}")
    os.makedirs(T.OUT, exist_ok=True)
    np.save(os.path.join(T.OUT, f"sigmaDL_frac{n}.npy"), sig)
    json.dump(rep, open(os.path.join(T.OUT, f"network_frac{n}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
