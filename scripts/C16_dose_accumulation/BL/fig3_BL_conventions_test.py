# -*- coding: utf-8 -*-
"""
fig3_BL_conventions_test.py -- STEP 8 (Figure 3), test 1 (2026-09-28): which reading of the Velocity DICOM DVF registers the
fraction sCT onto the plan CT? Read-only; the submitted code (../) is imported, never modified.

Readings compared, all on the plan-CT grid (fixed v2w after inv(PreDeformationMatrix), as runner.py PASS 1):
  prod      runner.py PASS 1 heuristic path: VectorField(dcm).toMM() -> resampleScan -> mu (3,X,Y,Z); runner treats channel c
            as the displacement along array axis c (mm / spacing[c]) -- compute_heuristic_uncertainty, DosePropagator
  prod_rev  the same mu with the channel order reversed
  noMM      as prod but without .toMM() (utils.dicom_vf already returns mm; toMM() rescales anyway)
  raw       independent: VectorGridData (x,y,z) in mm read with pydicom, sampled trilinearly at the world position of every
            plan-CT voxel, converted to index displacements with inv(A) of the plan-CT v2w
Metric (conventions_check.check_warp): mean |I_m o phi - I_f| in the body / the same without warp; PASS <= 0.7.
Also: per-channel scale and correlation of prod against raw (tells channel order and any double mm scaling).
Usage (venv): python fig3_BL_conventions_test.py [frac 1|2|3] [cuda:0]
"""
from dirunc_paths import LIB, LUNG_SBRT  # data roots, see README
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = LIB
sys.path.insert(0, SCRIPTS)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np, torch, pydicom
import torch.nn.functional as F
from utils.scan import Scan
from utils.vector_field import VectorField
from utils.conversions import toTorch, toNumpy
from utils.interpolate import resampleScan
from utils.warp import warp_torch
from runner import _apply_predef_rigid, _estimate_body_mask, reorient_dvf_to_target

DATA = rf"{LUNG_SBRT}"
PLAN = os.path.join(DATA, "Offline_Plan_18Gyx3fx54Gy")
DVF = {1: os.path.join(DATA, "Fx_1", "reg_Fx1.dcm"),
       2: os.path.join(DATA, "Fx_2", "reg_Fx2.dcm"),
       3: os.path.join(DATA, "Fx_3", "reg_Fx3.dcm")}
OUT = os.path.join(HERE, "_out_fig3_BL", "tests"); os.makedirs(OUT, exist_ok=True)


def sq(a):
    a = toNumpy(a) if not isinstance(a, np.ndarray) else a
    return np.asarray(a).squeeze()


def load_case(n):
    """Plan CT, sCT on the plan grid, production mu (with and without toMM) -- runner.py PASS 1 lines 781-836."""
    fixed = Scan(PLAN); _apply_predef_rigid(fixed, DVF[n])
    moving = Scan(os.path.join(DATA, f"Fx_{n}"))
    shape = fixed.getShape()
    if moving.getShape() != shape:
        moving = resampleScan(moving, fixed.getVoxelToWorldMatrix(), shape, defaultPixelValue=-1000.0); moving.toNumpy()
    mus = {}
    for tag, to_mm in (("prod", True), ("noMM", False)):
        v = VectorField(DVF[n])
        if to_mm:
            v.toMM()
        r = resampleScan(v, fixed.getVoxelToWorldMatrix(), shape)
        mu = toTorch(r.getPixelArray()).float()
        mu = mu.squeeze(0) if mu.dim() == 5 else mu
        mus[tag] = reorient_dvf_to_target(mu, shape).cpu().numpy()
    return fixed, sq(fixed.getPixelArray()).astype(np.float32), sq(moving.getPixelArray()).astype(np.float32), mus


def raw_dvf(n, fixed):
    """(3,X,Y,Z) index displacement on the plan-CT grid from the DICOM grid, independent of utils.dicom_vf."""
    ds = pydicom.dcmread(DVF[n])
    regs = [r for r in ds.DeformableRegistrationSequence if hasattr(r, "DeformableRegistrationGridSequence")]
    assert len(regs) == 1, len(regs)
    g = regs[0].DeformableRegistrationGridSequence[0]
    nx, ny, nz = [int(v) for v in g.GridDimensions]
    vec = np.frombuffer(g.VectorGridData, np.float32).reshape(nz, ny, nx, 3)          # C order: z slowest, (x,y,z) last
    ori = np.asarray(g.ImageOrientationPatient, float); ipp = np.asarray(g.ImagePositionPatient, float)
    res = np.asarray(g.GridResolution, float)
    info = {"grid_dims_xyz": [nx, ny, nz], "grid_res": res.tolist(), "ipp": ipp.tolist(), "orientation": ori.tolist()}
    assert np.allclose(ori, [1, 0, 0, 0, 1, 0]), f"non-identity grid orientation {ori}"
    v2w = fixed.getVoxelToWorldMatrix(); X, Y, Z = fixed.getShape()
    ii, jj, kk = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing="ij")
    P = (v2w[:3, :3] @ np.stack([ii.ravel(), jj.ravel(), kk.ravel()]) + v2w[:3, 3:4])     # world of each plan voxel
    gi = (P - ipp[:, None]) / res[:, None]                                                   # DVF-grid index (x,y,z)
    sz_ = np.array([nx, ny, nz], float)[:, None]
    grid = torch.from_numpy((2 * gi / (sz_ - 1) - 1).T.reshape(X, Y, Z, 3).astype(np.float32))   # grid_sample (x,y,z)
    vol = torch.from_numpy(np.ascontiguousarray(vec.transpose(3, 0, 1, 2)))[None]           # (1,3,z,y,x)
    grid = grid.permute(2, 1, 0, 3)[None]                                                    # (1,Z,Y,X,3) output order
    d = F.grid_sample(vol, grid, mode="bilinear", align_corners=True, padding_mode="zeros")[0]   # (3,Z,Y,X) world mm
    d = d.permute(0, 3, 2, 1).numpy()                                                        # (3,X,Y,Z), comps (x,y,z)
    Ainv = np.linalg.inv(v2w[:3, :3])
    info["plan_v2w"] = np.asarray(v2w).tolist()
    return np.einsum("ij,j...->i...", Ainv, d).astype(np.float32), d.astype(np.float32), info


def pullback_inverse(v, iters, dev):
    """The DICOM grid maps y (sCT frame) -> y + d(y) (plan point). Pull-back on the plan grid: u(x) with u = -v(x + u),
    v = raw index displacement at plan voxel x; fixed-point iteration from u0 = -v (first order). (3,X,Y,Z) index units."""
    vt = torch.from_numpy(np.ascontiguousarray(v)).to(dev); u = -vt.clone()
    for _ in range(iters):
        u = -warp_torch(vt, u)
    return u.cpu().numpy()


def ratio(fix, mov, vox, body, dev):
    f = torch.from_numpy(fix).to(dev); m = torch.from_numpy(mov).to(dev); b = torch.from_numpy(body).to(dev)
    w = warp_torch(m, torch.from_numpy(np.ascontiguousarray(vox)).to(dev))
    return float((w - f).abs()[b].mean() / (m - f).abs()[b].mean())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    dev = torch.device(sys.argv[2] if len(sys.argv) > 2 else "cuda:0")
    fixed, fx, mv, mus = load_case(n)
    sp = np.asarray(fixed.getVoxelSpacing(), float)                                          # (sx,sy,sz), array (X,Y,Z)
    body = np.ascontiguousarray(_estimate_body_mask(fx.transpose(2, 1, 0)).transpose(2, 1, 0))
    vox_raw, d_raw, info = raw_dvf(n, fixed)
    rep = {"frac": n, "spacing_xyz": sp.tolist(), "shape_xyz": list(fx.shape), **info, "warp_ratio": {}, "prod_vs_raw": {}}
    cand = {"no_warp": np.zeros_like(vox_raw), "raw": vox_raw, "raw_neg": -vox_raw}
    for it in (1, 3, 10):
        cand[f"raw_inv{it}"] = pullback_inverse(vox_raw, it, dev)
    for tag, mu in mus.items():
        cand[tag] = mu / sp[:, None, None, None]
        cand[tag + "_rev"] = mu[::-1] / sp[:, None, None, None]
    for tag, vox in cand.items():
        rep["warp_ratio"][tag] = ratio(fx, mv, vox, body, dev)
        print(f"  frac {n}  {tag:10s} residual ratio {rep['warp_ratio'][tag]:.3f}  "
              f"{'PASS' if rep['warp_ratio'][tag] <= 0.7 else 'fail'}", flush=True)
    bm = body.astype(bool)
    for tag, mu in mus.items():
        for c in range(3):                                   # prod channel c against each raw world component (x,y,z)
            row = []
            for w in range(3):
                a, b = mu[c][bm], d_raw[w][bm]
                cc = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
                row.append({"corr": round(cc, 4), "scale": round(float((a * b).sum() / max((b * b).sum(), 1e-12)), 4)})
            rep["prod_vs_raw"][f"{tag}_ch{c}"] = row
            best = int(np.nanargmax([abs(r["corr"]) for r in row]))
            print(f"  {tag} ch{c} ~ raw {'xyz'[best]}: corr {row[best]['corr']:+.3f} scale {row[best]['scale']:.3f}", flush=True)
    for w in range(3):
        print(f"  raw {'xyz'[w]}: body mean |d| {np.abs(d_raw[w][bm]).mean():.2f} mm, p99 {np.percentile(np.abs(d_raw[w][bm]), 99):.2f}")
    json.dump(rep, open(os.path.join(OUT, f"dvf_conventions_frac{n}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
