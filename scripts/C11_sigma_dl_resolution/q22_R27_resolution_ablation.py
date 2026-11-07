# -*- coding: utf-8 -*-
"""
q22_R27_resolution_ablation.py -- item R27 (2026-09-28): the R1.5 native-vs-training-resolution ablation of the Smolders DL
sigma (validation/smolders_dl_resolution_ablation.py, June: dLECE native-resampled +0.121/-0.067/-0.219/-0.138) redone with the
three corrections of step 2 (TRE via lece.evaluate_landmarks; input DVF [dx,dy,dz] in voxels; sigma -> mm with the network-grid
spacing). The only change between the two arms is reg.voxelSpacing: TRAIN (1.9531,1.9531,2.0) vs the case's native spacing.
Control: old wiring at both resolutions reproduces resolution_ablation.json (tol 1e-3 mm).
Output ./_out_q22_R27/R27.json. Usage (venv): python q22_R27_resolution_ablation.py [cuda:0]
"""
from dirunc_paths import RESULTS  # data roots, see README
import os, sys, types, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from q22_recompute_sigmaDL import infer, MODEL, smo, evaluate_landmarks          # same path setup as step 2
import numpy as np, torch
from utils.scan import Scan
from Registration.fixedMeanRegistration import FixedMeanPatchRegisterer
from validation.calibrate import fit_c_grid
from validation.lece import compute_lece

TRAIN = (1.9531, 1.9531, 2.0)
REF = json.load(open(RESULTS + "/resolution_ablation.json"))
OUT = os.path.join(HERE, "_out_q22_R27"); os.makedirs(OUT, exist_ok=True)
reg = FixedMeanPatchRegisterer(MODEL, patchSize=(256, 256, 96), voxelSpacing=TRAIN)
cases = [c[0] for c in sorted(smo.discover_cases(smo.CASTILLO_NIFTI), key=lambda c: int(c[0].replace("case", "")))]
P = {lab: {k: [] for k in ("tre_old", "tre_new", "old_train", "old_nat", "new_train", "new_nat")} for _, lab in smo.ENGINES}
for cid in cases:
    pair = smo.load_pair(smo.CASTILLO_NIFTI, case_id=cid)
    pair, _, _ = smo._apply_procrustes_to_pair(pair, os.path.join(HERE, "_tmp_diag"), save_artifacts=False)
    sp = np.asarray(pair.spacing_fixed, float); vtw = np.diag([*sp, 1.0])
    fs = Scan(array=pair.fixed_arr[None], vtwMatrix=vtw); ms = Scan(array=pair.moving_arr[None], vtwMatrix=vtw)
    for k, lab in smo.ENGINES:
        dvf = np.load(os.path.join(smo.CASTILLO_BASE, f"Results_{k}_procrustes", cid, "dvf_mu.npy")).astype(np.float32)
        P[lab]["tre_old"].append(smo._sample_tre_via_dvf(dvf, pair))
        P[lab]["tre_new"].append(evaluate_landmarks(dvf_mm=dvf, sigma_mm=np.ones(dvf.shape[1:], np.float32), spacing_fixed=pair.spacing_fixed,
                                                    lm_fixed_voxel=pair.lm_fixed_voxel, lm_moving_world_true=pair.lm_moving_world,
                                                    case_id=cid, fixed_affine=getattr(pair, "fixed_affine", None)).tre_mm)
        dvf_C = (dvf[::-1] / sp[:, None, None, None]).astype(np.float32)
        for vs, tag in ((TRAIN, "train"), (tuple(float(x) for x in sp), "nat")):
            reg.voxelSpacing = vs
            raw, f_v2, _ = infer(reg, fs, ms, dvf, vtw)                    # old wiring: input A, sigma v2
            P[lab][f"old_{tag}"].append(smo._sample_sigma_magnitude(raw * f_v2.reshape(3, 1, 1, 1).astype(np.float32), pair.lm_fixed_voxel))
            raw, _, s_net = infer(reg, fs, ms, dvf_C, vtw)                 # corrected: input C, sigma x network-grid spacing
            P[lab][f"new_{tag}"].append(smo._sample_sigma_magnitude(raw * s_net.reshape(3, 1, 1, 1).astype(np.float32), pair.lm_fixed_voxel))
    print(f"  {cid} done", flush=True)
    torch.cuda.empty_cache()


def lece(t, s):
    k = np.isfinite(t) & np.isfinite(s) & (s > 1e-9) & (t >= 0); f, _ = fit_c_grid(t[k], s[k], bin_width=0.25)
    return float(f), float(compute_lece(t[k], f * s[k], bin_width=0.25)[0])


res, ok = {}, True
ref = {r["engine"]: r for r in REF["per_engine"]}
for _, lab in smo.ENGINES:
    c = {k: np.concatenate(v) for k, v in P[lab].items()}
    r = {}
    for arm, tre in (("old", "tre_old"), ("new", "tre_new")):
        for tag in ("train", "nat"):
            r[f"{arm}_{tag}_f"], r[f"{arm}_{tag}_lece"] = lece(c[tre], c[f"{arm}_{tag}"])
        r[f"{arm}_delta"] = r[f"{arm}_nat_lece"] - r[f"{arm}_train_lece"]
    okl = abs(r["old_train_lece"] - ref[lab]["lece_resampled_mm"]) < 1e-3 and abs(r["old_nat_lece"] - ref[lab]["lece_native_mm"]) < 1e-3
    ok &= okl; res[lab] = r
    print(f"{lab:12s} control old: train {r['old_train_lece']:.3f} (ref {ref[lab]['lece_resampled_mm']:.3f}) nat {r['old_nat_lece']:.3f} "
          f"(ref {ref[lab]['lece_native_mm']:.3f}) {'PASS' if okl else 'FAIL'} | NEW: train {r['new_train_lece']:.3f} (f {r['new_train_f']:.2f}) "
          f"nat {r['new_nat_lece']:.3f} (f {r['new_nat_f']:.2f}) delta {r['new_delta']:+.3f} (old {r['old_delta']:+.3f})", flush=True)
json.dump({"control_pass": bool(ok), "results": res}, open(os.path.join(OUT, "R27.json"), "w"), indent=2)
print(f"control {'PASS' if ok else 'FAIL'} -> {OUT}")
