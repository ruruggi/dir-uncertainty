# -*- coding: utf-8 -*-
"""
q22_step6_sensitivity.py -- step 6 of the recalculation order (2026-09-28): the analyses of _analysis_v28.py and _sensitivity_full.py
(Figure 4 = old Table 5 w-sweep, per-HU stratification, q_R / q_J / bin-width sweeps of Appendix C, z-score ablation of §3.4) redone
on the body-local sigma (conventions_check.sigma_body_local) on DIR-LAB Castillo, 4 engines x 10 cases. N4: the old scripts used their
own sigma (X,Y,Z arrays with [dz,dy,dx] channels, no Procrustes, whole-volume statistics, sagittal hole filling).

Per case the fields R = clip01(1 - residual * norm|grad I_m|) and J = |det J - 1| are computed ONCE exactly as sigma_body_local
(Procrustes, moving resampled to fixed, (Z,Y,X), DVF extended outside the body, MIND-SSC r1 d2, stats on the body); stored at the
landmarks together with their body statistics (min, max, exact quantiles, mean/sd). Any variant is then
  sigma = w * minmax_body(min(R, q_R-quantile)) + (1 - w) * minmax_body(min(J, q_J-quantile)).
Controls: at (w, q_R, q_J) = (0.7, 0.99, 0.95) sigma == stored sigma_BL landmarks (1e-4) and LECE == Table 3 sigma_h (new).
Output ./_out_q22_step6/ (cache.npz, step6.json). Usage (venv): python q22_step6_sensitivity.py [cuda:1]
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
from validation.calibrate import fit_c_grid
from lece import predict_landmark_sigma, compute_lece

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")
D = f"{DATASETS}/Castillo-DIR-LAB"; BL = f"{DATASETS}/_sigma_BL_20260927/Castillo-DIR-LAB"
ENG = {"SyNRA": "synra", "ConvexAdam": "convexadam", "uniGradICON": "unigradicon", "Elastix": "elastix"}
QR, QJ = [0.95, 0.97, 0.99], [0.90, 0.93, 0.95, 0.97, 0.99]
W = [round(0.1 * i, 1) for i in range(11)]; BINS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
HU_BANDS = {"lung": (-1024, -400), "soft": (-400, 400), "bone": (400, 3071)}
OUT = os.path.join(HERE, "_out_q22_step6"); os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "cache.npz")

if not os.path.isfile(CACHE):
    C = {}
    for c in range(1, 11):
        cid = f"case{c}"
        pair = load_pair(f"{D}/nifti", case_id=cid)
        pair, _, _ = _apply_procrustes_to_pair(pair, os.path.join(OUT, "_tmp"), save_artifacts=False)
        mov = _resample_moving_to_fixed(pair.fixed_arr, pair.moving_arr, pair.spacing_fixed, pair.spacing_moving,
                                        fixed_affine=getattr(pair, "fixed_affine", None), moving_affine=getattr(pair, "moving_affine", None))
        sx, sy, sz = pair.spacing_fixed; sp = (sz, sy, sx)
        fz = np.ascontiguousarray(pair.fixed_arr.astype(np.float32).transpose(2, 1, 0)); mz = np.ascontiguousarray(mov.astype(np.float32).transpose(2, 1, 0))
        body = torch.from_numpy(_estimate_body_mask(fz)).to(dev); bm = torch.from_numpy(_estimate_body_mask(mz)).to(dev)
        f = torch.where(body, torch.from_numpy(fz).to(dev), torch.tensor(-1000.0, device=dev))
        m = torch.where(bm, torch.from_numpy(mz).to(dev), torch.tensor(-1000.0, device=dev))
        gn = _minmax(GRADIENT_FUNC(m), bm)
        idx = ndi.distance_transform_edt(~body.cpu().numpy(), return_distances=False, return_indices=True)
        hu = predict_landmark_sigma(pair.fixed_arr.astype(np.float32), pair.lm_fixed_voxel)
        for lab, k in ENG.items():
            d0 = f"{D}/Results_{k}_procrustes/{cid}"
            dvf = np.load(f"{d0}/dvf_mu.npy").astype(np.float32).transpose(0, 3, 2, 1)
            dvf = torch.from_numpy(np.ascontiguousarray(dvf[:, idx[0], idx[1], idx[2]])).to(dev)
            for j in range(3):
                dvf[j] = dvf[j] / sp[j]
            res = compute_residual(f.unsqueeze(0), warp_torch(m, dvf).unsqueeze(0), metric="mind_ssc", radius=1, dilation=2).squeeze(0)
            R = (1.0 - res * gn).clamp(0, 1); J = (jacobian_determinant(dvf) - 1.0).abs()
            at = lambda t: predict_landmark_sigma(np.ascontiguousarray(t.cpu().numpy().transpose(2, 1, 0)).astype(np.float32), pair.lm_fixed_voxel)
            Rb, Jb = R[body], J[body]
            C[f"{lab}|{c}|R"] = at(R); C[f"{lab}|{c}|J"] = at(J); C[f"{lab}|{c}|hu"] = hu; C[f"{lab}|{c}|inb"] = at(body.float())
            C[f"{lab}|{c}|tre"] = np.load(f"{d0}/landmarks.npz")["tre_mm"]
            C[f"{lab}|{c}|sigBL"] = np.load(f"{BL}/Results_{k}_procrustes/{cid}/landmarks.npz")["sigma_p_mm_raw"]
            C[f"{lab}|{c}|Rstat"] = np.array([float(Rb.min()), float(Rb.max()), float(Rb.mean()), float(Rb.std())] + [float(_quantile(Rb, q)) for q in QR])
            C[f"{lab}|{c}|Jstat"] = np.array([float(Jb.min()), float(Jb.max())] + [float(_quantile(Jb, q)) for q in QJ])
            del dvf, res, R, J; torch.cuda.empty_cache()
        print(f"  case{c} cached", flush=True)
        del f, m, body, bm, gn, idx; torch.cuda.empty_cache()
    np.savez(CACHE, **C)
C = dict(np.load(CACHE))


def sig(lab, c, w=0.7, qr=0.99, qj=0.95, zscore=False):
    R, J = C[f"{lab}|{c}|R"], C[f"{lab}|{c}|J"]; rs, js = C[f"{lab}|{c}|Rstat"], C[f"{lab}|{c}|Jstat"]
    qv = js[2 + QJ.index(qj)]; lo, hi = min(js[0], qv), min(js[1], qv)
    Jn = np.clip((np.minimum(J, qv) - lo) / (hi - lo if hi - lo > 1e-8 else 1.0), 0, 1)
    inb = C[f"{lab}|{c}|inb"] > 0.5                # sigma = 0 outside the body
    if zscore:                                   # runner use_minmax_norm=False, body-local: R z-scored on the body, no clip
        return np.where(inb, np.maximum(w * (R - rs[2]) / rs[3] + (1 - w) * Jn, 0.0), 0.0)
    qv = rs[4 + QR.index(qr)]; lo, hi = min(rs[0], qv), min(rs[1], qv)
    Rn = np.clip((np.minimum(R, qv) - lo) / (hi - lo if hi - lo > 1e-8 else 1.0), 0, 1)
    return np.where(inb, w * Rn + (1 - w) * Jn, 0.0)


def pooled(lab, band=None, **kw):
    T, S = [], []
    for c in range(1, 11):
        t, s = C[f"{lab}|{c}|tre"], sig(lab, c, **kw)
        k = np.ones(t.size, bool) if band is None else (C[f"{lab}|{c}|hu"] >= band[0]) & (C[f"{lab}|{c}|hu"] < band[1])
        T.append(t[k]); S.append(s[k])
    t, s = np.concatenate(T), np.concatenate(S); k = np.isfinite(t) & np.isfinite(s) & (s > 1e-9)
    return t[k], s[k]


def metrics(t, s, bw=0.25):
    if t.size < 10:
        return {"n": int(t.size)}
    f, _ = fit_c_grid(t, s, bin_width=bw)
    return {"n": int(t.size), "f": float(f), "lece": float(compute_lece(t, f * s, bin_width=bw)[0]), "rho": float(spearmanr(s, t)[0]),
            "cv": float(s.std() / s.mean())}


T3 = json.load(open(os.path.join(HERE, "_out_q22_table3", "table3.json")))["results"]
Rj, ok = {"sweep_w": {}, "sweep_qR": {}, "sweep_qJ": {}, "bins": {}, "hu": {}, "zscore": {}}, True
print("Controls")
for lab in ENG:
    d = max(float(np.max(np.abs(sig(lab, c) - C[f"{lab}|{c}|sigBL"]))) for c in range(1, 11))
    m0 = metrics(*pooled(lab)); ref = T3[lab]["sigma_h_BL"]["lece_global"]
    okl = d < 1e-4 and abs(m0["lece"] - ref) < 1e-6; ok &= okl
    print(f"  [{'PASS' if okl else 'FAIL'}] {lab:12s} max|sigma - sigma_BL| {d:.1e}  LECE {m0['lece']:.4f} (Table 3 new {ref:.4f})")
for lab in ENG:
    Rj["sweep_w"][lab] = {str(w): metrics(*pooled(lab, w=w)) for w in W}
    Rj["sweep_qR"][lab] = {str(q): metrics(*pooled(lab, qr=q)) for q in QR}
    Rj["sweep_qJ"][lab] = {str(q): metrics(*pooled(lab, qj=q)) for q in QJ}
    Rj["bins"][lab] = {str(b): metrics(*pooled(lab), bw=b) for b in BINS}
    Rj["hu"][lab] = {n: metrics(*pooled(lab, band=b)) for n, b in HU_BANDS.items()}
    Rj["zscore"][lab] = {"default": metrics(*pooled(lab)), "zscore": metrics(*pooled(lab, zscore=True))}
json.dump({"controls_pass": bool(ok), **Rj}, open(os.path.join(OUT, "step6.json"), "w"), indent=2)

print("\nFigure 4 (w sweep): LECE | rho_pool   (SyNRA / ConvexAdam / uniGradICON / Elastix)")
for w in W:
    print(f"  w {w:.1f}  " + " ".join(f"{Rj['sweep_w'][l][str(w)]['lece']:6.3f}" for l in ENG) + "  |  " + " ".join(f"{Rj['sweep_w'][l][str(w)]['rho']:+.3f}" for l in ENG))
print("\nq_R sweep (w 0.7, q_J 0.95): LECE | rho");  [print(f"  q_R {q}  " + " ".join(f"{Rj['sweep_qR'][l][str(q)]['lece']:6.3f}" for l in ENG) + "  |  " + " ".join(f"{Rj['sweep_qR'][l][str(q)]['rho']:+.3f}" for l in ENG)) for q in QR]
print("q_J sweep (w 0.7, q_R 0.99): LECE | rho");  [print(f"  q_J {q}  " + " ".join(f"{Rj['sweep_qJ'][l][str(q)]['lece']:6.3f}" for l in ENG) + "  |  " + " ".join(f"{Rj['sweep_qJ'][l][str(q)]['rho']:+.3f}" for l in ENG)) for q in QJ]
print("bin width (default sigma): LECE");        [print(f"  bin {b:.2f} " + " ".join(f"{Rj['bins'][l][str(b)]['lece']:6.3f}" for l in ENG)) for b in BINS]
print("\nper-HU (default): n / LECE / rho")
for l in ENG:
    print(f"  {l:12s} " + " | ".join(f"{n}: {x['n']}" + (f" {x['lece']:.3f} {x['rho']:+.3f}" if 'lece' in x else " (too few)") for n, x in Rj["hu"][l].items()))
print("\nz-score ablation (body-local, R without clip): CV sigma_p and rho_pool, default -> z-score")
for l in ENG:
    a, b = Rj["zscore"][l]["default"], Rj["zscore"][l]["zscore"]
    print(f"  {l:12s} CV {100*a['cv']:.0f} % -> {100*b['cv']:.0f} %   rho {a['rho']:+.3f} -> {b['rho']:+.3f}   LECE {a['lece']:.3f} -> {b['lece']:.3f}")
print(f"\nOVERALL CONTROLS: {'PASS' if ok else 'FAIL'} -> {OUT}")
