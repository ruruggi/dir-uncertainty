# -*- coding: utf-8 -*-
"""
q22_recompute_sigmaDL.py -- step 2 of the recalculation order (audit N1, 2026-09-28): sigma_DL of Table 3 (Smolders DL
Unsupervised L20S3-4) on the 10 DIR-LAB Castillo cases x 4 engines, with every combination of the three corrections, against
the submitted v2 wiring (smolders_dl_sigma_castillo.run_one + FixedMeanPatchRegisterer.process). Submitted code not modified.

Corrections (each verified in q22_sigmaDL_checks.py / q22_sigmaDL_probe.py):
  T  TRE        old: smo._sample_tre_via_dvf (ignores the affine; |d| up to 25 mm)  -> new: lece.evaluate_landmarks (= stored TRE)
  I  input DVF  A: dvf_mu.npy as is ([dz,dy,dx] mm, current)  -> C: [dx,dy,dz] in fixed voxels (the fixedMean network uses it as
                its mean flow in voxels, channel i along array axis i; only C registers with the network's STN)
  S  sigma->mm  v2: stdev x (resampling ratio)[::-1] (postProcess l.72)  -> net: stdev x network-grid spacing (stdev is in
                network-grid voxels: GaussianVectorField stdev is not rescaled by interpolateVectorField)
Per (case, engine): 2 inferences (A, C); sigma_p sampled as in smo._sample_sigma_magnitude for both scalings.

Outputs (resume-safe, one npz per engine/case) in ./_out_q22_sigmaDL/: <engine>_<case>.npz (tre_old, tre_new, sp_A_v2, sp_A_net,
sp_C_v2, sp_C_net, spacings), aggregate.json (8 combinations x 4 engines: f, LECE global, 5-fold as Table 3) and the control:
T=old, I=A, S=v2 must reproduce the published global LECE 0.464/0.420/0.710/0.554 (tol 0.011) and the first case must match
FixedMeanPatchRegisterer.process() exactly.
Usage (venv): python q22_recompute_sigmaDL.py [cuda:0]
"""
from dirunc_paths import LIB, MODELS  # data roots, see README
import os, sys, types, json, time, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = LIB; PS = LIB + "/validation"
sys.path.insert(0, SCRIPTS); sys.path.insert(0, os.path.join(SCRIPTS, "dl_arch")); sys.path.insert(0, PS)
if "validation" not in sys.modules:
    _v = types.ModuleType("validation"); _v.__path__ = [PS]; sys.modules["validation"] = _v
import numpy as np, torch
import smolders_dl_sigma_castillo as smo
from table3_5fold_both import calibrate_block, KNOWN_GLOBAL, TOL
from validation.calibrate import CaseLandmarks
from lece import evaluate_landmarks
from utils.scan import Scan
from utils.vector_field import VectorField
from Registration.fixedMeanRegistration import FixedMeanPatchRegisterer

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
torch.cuda.set_device(torch.device(DEV))
MODEL = os.path.join(rf"{MODELS}", smo.MODEL_NAME)   # not in the submitted snapshot
OUT = os.path.join(HERE, "_out_q22_sigmaDL"); os.makedirs(OUT, exist_ok=True)
LABEL = dict(smo.ENGINES)


def to_np(a):
    return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)


def infer(reg, fs, ms, arr, vtw):
    """preProcess + register + resample to the fixed grid, as FixedMeanPatchRegisterer.process. Returns stdev in network-grid
    voxels (3,X,Y,Z), the v2 postProcess factor (resampling ratio reversed) and the network-grid spacing."""
    inp = reg.preProcess(fs, ms, VectorField(array=arr, vtwMatrix=vtw, inMM=True))
    res = reg.resampleVectorField(reg.register(inp), fs.getShape())
    n_net = np.asarray(inp.shape[-3:], float); n_fix = np.asarray(fs.getShape(), float)
    s_net = (n_fix - 1) * np.asarray(fs.getVoxelSpacing(), float) / (n_net - 1)
    return to_np(res.stdev).astype(np.float32), np.asarray(res.getVoxelSpacing(), float)[::-1].copy(), s_net


def main():
    reg = FixedMeanPatchRegisterer(MODEL, patchSize=(256, 256, 96), voxelSpacing=(1.9531, 1.9531, 2.0))   # = dl.model_loader
    cases = [c[0] for c in sorted(smo.discover_cases(smo.CASTILLO_NIFTI), key=lambda c: int(c[0].replace("case", "")))]
    t0 = time.time(); equiv_done = os.path.exists(os.path.join(OUT, "equivalence.json"))
    for cid in cases:
        todo = [k for k, _ in smo.ENGINES if not os.path.exists(os.path.join(OUT, f"{k}_{cid}.npz"))]
        if not todo:
            continue
        pair = smo.load_pair(smo.CASTILLO_NIFTI, case_id=cid)
        pair, _, _ = smo._apply_procrustes_to_pair(pair, os.path.join(HERE, "_tmp_diag"), save_artifacts=False)
        sp = np.asarray(pair.spacing_fixed, float); vtw = np.diag([*sp, 1.0])
        fs = Scan(array=pair.fixed_arr[None], vtwMatrix=vtw); ms = Scan(array=pair.moving_arr[None], vtwMatrix=vtw)
        for k in todo:
            dvf = np.load(os.path.join(smo.CASTILLO_BASE, f"Results_{k}_procrustes", cid, "dvf_mu.npy")).astype(np.float32)
            tre_ref = np.load(os.path.join(smo.CASTILLO_BASE, f"Results_{k}_procrustes", cid, "landmarks.npz"))["tre_mm"]
            rec = {"tre_old": smo._sample_tre_via_dvf(dvf, pair),
                   "tre_new": evaluate_landmarks(dvf_mm=dvf, sigma_mm=np.ones(dvf.shape[1:], np.float32), spacing_fixed=pair.spacing_fixed,
                                                 lm_fixed_voxel=pair.lm_fixed_voxel, lm_moving_world_true=pair.lm_moving_world,
                                                 case_id=cid, fixed_affine=getattr(pair, "fixed_affine", None)).tre_mm,
                   "spacing_fixed": sp}
            assert np.max(np.abs(rec["tre_new"] - tre_ref)) < 1e-3, f"{k} {cid}: tre_new != stored TRE"
            variants = {"A": dvf, "C": (dvf[::-1] / sp[:, None, None, None]).astype(np.float32)}
            for v, arr in variants.items():
                raw, f_v2, s_net = infer(reg, fs, ms, arr, vtw)
                rec[f"sp_{v}_v2"] = smo._sample_sigma_magnitude(raw * f_v2.reshape(3, 1, 1, 1).astype(np.float32), pair.lm_fixed_voxel)
                rec[f"sp_{v}_net"] = smo._sample_sigma_magnitude(raw * s_net.reshape(3, 1, 1, 1).astype(np.float32), pair.lm_fixed_voxel)
                rec[f"f_v2_{v}"] = f_v2; rec["s_net"] = s_net
                if v == "A" and not equiv_done:   # control: our split pipeline == the submitted process() (v2 wiring)
                    _, prob = reg.process(fs, ms, VectorField(array=dvf, vtwMatrix=vtw, inMM=True))
                    ref = smo._sample_sigma_magnitude(prob.getVectorFieldUncertainty(), pair.lm_fixed_voxel)
                    d = float(np.max(np.abs(ref - rec["sp_A_v2"])))
                    json.dump({"case": cid, "engine": k, "max_abs_diff_sigma_p": d}, open(os.path.join(OUT, "equivalence.json"), "w"))
                    print(f"  [control] split pipeline vs FixedMeanPatchRegisterer.process: max|d sigma_p| = {d:.2e}", flush=True)
                    equiv_done = True
            np.savez(os.path.join(OUT, f"{k}_{cid}.npz"), **rec)
            print(f"  {cid} {LABEL[k]:12s} RMSE TRE old {np.sqrt(np.mean(rec['tre_old']**2)):6.2f} new {np.sqrt(np.mean(rec['tre_new']**2)):5.2f}"
                  f" | mean sigma_p A/v2 {rec['sp_A_v2'].mean():.2f} A/net {rec['sp_A_net'].mean():.2f} C/v2 {rec['sp_C_v2'].mean():.2f}"
                  f" C/net {rec['sp_C_net'].mean():.2f}  ({time.time() - t0:.0f}s)", flush=True)

    # ---- aggregate: 8 combinations, same calibration/LECE/5-fold as Table 3 --------------------------------------------------
    agg, control_ok = {}, True
    for k, lab in smo.ENGINES:
        recs = [(cid, np.load(os.path.join(OUT, f"{k}_{cid}.npz"))) for cid in cases]
        agg[lab] = {}
        for T, I, S in itertools.product(("old", "new"), ("A", "C"), ("v2", "net")):
            cl = []
            for cid, r in recs:
                tre = np.asarray(r[f"tre_{T}"], np.float64).ravel(); sig = np.asarray(r[f"sp_{I}_{S}"], np.float64).ravel()
                keep = np.isfinite(tre) & np.isfinite(sig) & (sig > 1e-9) & (tre >= 0)   # = table3_5fold_both
                cl.append(CaseLandmarks(case_id=cid, tre_mm=tre[keep], sigma_raw=sig[keep]))
            agg[lab][f"T{T}_I{I}_S{S}"] = calibrate_block(cl)
        got = agg[lab]["Told_IA_Sv2"]["lece_global"]; ok = abs(got - KNOWN_GLOBAL["sigma_DL"][lab]) <= TOL; control_ok &= ok
        print(f"  [control] {lab:12s} published {KNOWN_GLOBAL['sigma_DL'][lab]:.3f} reproduced {got:.4f} {'PASS' if ok else 'FAIL'}", flush=True)
    json.dump({"control_reproduces_published": bool(control_ok), "tol": TOL, "results": agg},
              open(os.path.join(OUT, "aggregate.json"), "w"), indent=2)
    print("\n  engine        combination      f      LECE global   5-fold mean+-sd")
    for lab in agg:
        for c, b in agg[lab].items():
            print(f"  {lab:12s}  {c:15s} {b['f_global']:6.3f}  {b['lece_global']:.4f}      {b['lece_5fold_mean']:.4f}+-{b['lece_5fold_std']:.4f}")
    print(f"[done] {time.time() - t0:.0f}s  control {'PASS' if control_ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
