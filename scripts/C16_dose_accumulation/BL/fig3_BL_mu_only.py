# -*- coding: utf-8 -*-
"""
fig3_BL_mu_only.py -- STEP 8 (Figure 3), N9 impact (2026-09-28): deterministic accumulated dose (no ensemble) of the three
realigned fraction doses (Fx_n/RD.*_isoalign.dcm), pulled back to the plan CT with
  none   no deformable warp (rigid placement only, as runner.py PASS 4 resampling)
  prod   the submitted reading of the Velocity field (runner.py: toMM, reversed channels applied along axis c, + direction)
  new    the corrected reading (fixed-point pull-back inverse of the raw field, fig3_BL_conventions_test.py)
PTV54 D95 / mean dose with the evaluation geometry of the published numbers (_lung_batch/fig_dvh_overlay.py: masks, plan dose,
400 bins on 0-60 Gy). Control: 'prod' must reproduce the deficit of the submitted pipeline (ensemble mean D95 42.28 Gy, Results100_isoalign
Heuristic) up to the small ensemble effect. Read-only; output _out_fig3_BL/tests/mu_only.json.
Usage (venv): python fig3_BL_mu_only.py [cuda:0]
"""
from dirunc_paths import LIB  # data roots, see README
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = LIB
sys.path.insert(0, HERE); sys.path.insert(0, SCRIPTS)
sys.path.insert(0, LIB); os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np, torch, pydicom
import fig3_BL_conventions_test as T
from utils.scan import Scan
from utils.interpolate import resampleScan
from utils.conversions import toNumpy
from utils.warp import warp_torch
from runner import _apply_predef_rigid
from fig_dvh_overlay import STRUCTURES, RS_PATH, RD_PLAN, load_pct_post_rigid, rasterize_roi, resample_plan_dose, dvh_cumulative
sys.path.insert(0, LIB)
from compare_d95_fx2fix import d_at_vol                                           # same D95 interpolation as the published numbers

RD = {1: "RD_Fx1_isoalign.dcm",
      2: "RD_Fx2_isoalign.dcm",
      3: "RD_Fx3_isoalign.dcm"}


def main():
    dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
    acc = {k: 0.0 for k in ("none", "prod", "new")}; per = {}
    for n in (1, 2, 3):
        fixed, fx, mv, mus = T.load_case(n)
        sp = np.asarray(fixed.getVoxelSpacing(), float)
        vox_raw, _, _ = T.raw_dvf(n, fixed)
        f2 = Scan(T.PLAN); _apply_predef_rigid(f2, T.DVF[n])                       # PASS 4 dose placement (runner.py 1140-1148)
        dose = resampleScan(Scan(os.path.join(T.DATA, f"Fx_{n}", RD[n])), f2.getVoxelToWorldMatrix(), f2.getShape())
        dv = toNumpy(dose.getPixelArray()); dv = (dv[0] if dv.ndim == 4 else dv).astype(np.float32)        # (X,Y,Z)
        D = torch.from_numpy(np.ascontiguousarray(dv)).to(dev)
        fields = {"none": np.zeros_like(vox_raw), "prod": mus["prod"] / sp[:, None, None, None],
                  "new": T.pullback_inverse(vox_raw, 10, dev)}
        for k, v in fields.items():
            w = warp_torch(D, torch.from_numpy(np.ascontiguousarray(v)).float().to(dev)).cpu().numpy()
            acc[k] = acc[k] + np.transpose(w, (2, 1, 0))                                                   # (Z,Y,X) as dose_accumulated_fc
        per[n] = {"warp_ratio_new": T.ratio(fx, mv, fields["new"], fx > -500, dev), "warp_ratio_prod": T.ratio(fx, mv, fields["prod"], fx > -500, dev)}
        print(f"  Fx_{n}: CT residual ratio (HU > -500) prod {per[n]['warp_ratio_prod']:.3f} new {per[n]['warp_ratio_new']:.3f}", flush=True)
        del D; torch.cuda.empty_cache()
    ct, v2w_post, inv_predef = load_pct_post_rigid(); nz, ny, nx = ct.shape
    rs = pydicom.dcmread(RS_PATH); edges = np.linspace(0, 60, 401); centers = 0.5 * (edges[:-1] + edges[1:])
    dose_plan = resample_plan_dose(RD_PLAN, ct, v2w_post, inv_predef)
    rep = {"per_fraction": per, "structures": {}}
    for name, _ in STRUCTURES:
        m = rasterize_roi(rs, name, ny, nx, nz, inv_predef, v2w_post)
        if m is None or int(m.sum()) == 0:
            continue
        r = {"D95_plan": d_at_vol(centers, dvh_cumulative(dose_plan, m, edges)), "mean_plan": float(dose_plan[m].mean())}
        for k, a in acc.items():
            assert a.shape == dose_plan.shape, (a.shape, dose_plan.shape)
            r[f"D95_{k}"] = d_at_vol(centers, dvh_cumulative(a, m, edges)); r[f"mean_{k}"] = float(a[m].mean())
        rep["structures"][name] = r
        print(f"  {name:>15s}: D95 plan {r['D95_plan']:6.2f} | none {r['D95_none']:6.2f} prod {r['D95_prod']:6.2f} new {r['D95_new']:6.2f}"
              f"  | mean plan {r['mean_plan']:6.2f} none {r['mean_none']:6.2f} prod {r['mean_prod']:6.2f} new {r['mean_new']:6.2f}", flush=True)
    json.dump(rep, open(os.path.join(T.OUT, "mu_only.json"), "w"), indent=2)
    np.save(os.path.join(T.OUT, "dose_mu_only_new.npy"), acc["new"].astype(np.float32))


if __name__ == "__main__":
    main()
