# -*- coding: utf-8 -*-
"""
q22_recompute_sigma_BL.py -- recompute the heuristic sigma of the 13 (dataset, engine) cells that have a stored mu
(all except SyNRA on Lung/Abdominal/Liver), with the body-local version adopted by the user on 2026-09-27 (replaces S3):
  warp along the right axes (arrays (Z,Y,X), DVF (3,Z,Y,X) channels [dz,dy,dx], as runner.py in production),
  conventions_check.sigma_body_local: every statistic on the body only (fixed body = runner._estimate_body_mask(fixed),
  moving body = the same on the resampled moving), clip R/J = 0.99/0.95, w = 0.7, MIND-SSC r1 d2; same Procrustes
  pre-alignment and moving resampling as the May batches. Certified by conventions_check.py (body independence PASS).
The originals are never touched. Output mirrors the Results tree under OUT_ROOT:
  landmarks.npz  = the stored one with sigma_p_mm / sigma_p_mm_raw replaced by the S3 values (TRE etc. unchanged)
  summary.json   = the stored one plus a "sigma_S3" block; lece_mm recomputed at S = 1
  sigma_raw.npy  = the S3 volume, (X,Y,Z) as the stored one -- DIR-LAB only (needed by the Q2.1 test)
Resume-safe (a case with landmarks.npz in the output is skipped). Check per case: predict_landmark_sigma on the
stored sigma_raw reproduces the stored sigma_p (when sigma_raw exists).
Run with the project venv (needs ants).
"""
from dirunc_paths import DATASETS, LIB  # data roots, see README
import os, sys, types, json, re, time
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = LIB; PS = LIB + "/validation"
sys.path.insert(0, SCRIPTS); sys.path.insert(0, PS)
v = types.ModuleType("validation"); v.__path__ = [PS]; sys.modules["validation"] = v
import numpy as np, torch
from runner import _estimate_body_mask
import conventions_check as cc
from validation.load_abdominal_dir_qa import load_pair
from validation.run_pair import _apply_procrustes_to_pair, _resample_moving_to_fixed
from lece import predict_landmark_sigma, compute_lece

DS_ROOT = f"{DATASETS}"
OUT_ROOT = f"{DATASETS}/_sigma_BL_20260927"
dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None          # optional dataset filter
KW = dict(w_res_grad=0.7, w_jac=0.3, ratio_clip_percentile=0.99, jac_clip_percentile=0.95, use_minmax_norm=True,
          global_scale=1.0, suppression_alpha=1.0, similarity_metric="mind_ssc", similarity_kwargs={"radius": 1, "dilation": 2})

src = open(os.path.join(PS, "build_v11_data.py"), encoding="utf-8").read()
cells = []                                                          # (dataset, results root)
for ds, r in re.findall(r'"([^"]+)":\s+f?"([^"]+)"', src):
    r = r.replace("{_CASTILLO}", f"{DS_ROOT}/Castillo-DIR-LAB")
    if os.path.isdir(r) and any(os.path.isfile(os.path.join(r, c, "dvf_mu.npy")) for c in os.listdir(r)):
        cells.append((ds, r))
assert len(cells) == 13, f"expected 13 cells with a stored mu, found {len(cells)}"
if ONLY:
    cells = [c for c in cells if c[0] in ONLY]
jobs = {}                                                           # (dataset, case_id) -> [results case dirs]
for ds, r in cells:
    for c in sorted(os.listdir(r)):
        d = os.path.join(r, c)
        if all(os.path.isfile(os.path.join(d, f)) for f in ("dvf_mu.npy", "landmarks.npz", "summary.json")):
            jobs.setdefault((ds, c), []).append(d)
print(f"{len(cells)} cells, {sum(len(v) for v in jobs.values())} case x engine pairs", flush=True)

t0 = time.time()
for (ds, c), dirs in jobs.items():
    todo = [d for d in dirs if not os.path.isfile(os.path.join(OUT_ROOT, os.path.relpath(d, DS_ROOT), "landmarks.npz"))]
    if not todo:
        continue
    S = json.load(open(os.path.join(todo[0], "summary.json")))
    pair = load_pair(S["case_dir"], case_id=S["case_id"])
    pair, _, _ = _apply_procrustes_to_pair(pair, os.path.join(HERE, "_tmp_diag"), save_artifacts=False)
    mov = _resample_moving_to_fixed(pair.fixed_arr, pair.moving_arr, pair.spacing_fixed, pair.spacing_moving,
                                    fixed_affine=getattr(pair, "fixed_affine", None), moving_affine=getattr(pair, "moving_affine", None))
    sx, sy, sz = pair.spacing_fixed
    fz = np.ascontiguousarray(pair.fixed_arr.astype(np.float32).transpose(2, 1, 0))
    mz = np.ascontiguousarray(mov.astype(np.float32).transpose(2, 1, 0))
    body = _estimate_body_mask(fz); body_m = _estimate_body_mask(mz)
    f_t = torch.from_numpy(fz).to(dev); m_t = torch.from_numpy(mz).to(dev); b_t = torch.from_numpy(body).to(dev); bm_t = torch.from_numpy(body_m).to(dev)
    for d in todo:
        lm = dict(np.load(os.path.join(d, "landmarks.npz"), allow_pickle=True)); Sj = json.load(open(os.path.join(d, "summary.json")))
        if os.path.isfile(os.path.join(d, "sigma_raw.npy")):
            chk = np.abs(predict_landmark_sigma(np.load(os.path.join(d, "sigma_raw.npy")), lm["lm_fixed_voxel"]) - lm["sigma_p_mm_raw"]).max()
            assert chk < 1e-5, f"{d}: stored sigma_p not reproduced ({chk})"
        dz = torch.from_numpy(np.ascontiguousarray(np.load(os.path.join(d, "dvf_mu.npy")).astype(np.float32).transpose(0, 3, 2, 1))).to(dev)
        s = cc.sigma_body_local(f_t, m_t, dz, b_t, bm_t, (sz, sy, sx))
        s_xyz = np.ascontiguousarray(s.squeeze().cpu().numpy().transpose(2, 1, 0)).astype(np.float32)
        sp = predict_landmark_sigma(s_xyz, lm["lm_fixed_voxel"])
        od = os.path.join(OUT_ROOT, os.path.relpath(d, DS_ROOT)); os.makedirs(od, exist_ok=True)
        if ds == "DIR-LAB":
            np.save(os.path.join(od, "sigma_raw.npy"), s_xyz)
        sp_old = lm["sigma_p_mm_raw"].copy()
        lm["sigma_p_mm"] = sp * float(lm.get("S_scaling_applied", 1.0)); lm["sigma_p_mm_raw"] = sp
        lece, _ = compute_lece(lm["tre_mm"], sp, bin_width=0.25)
        Sj["lece_mm_May"] = Sj.get("lece_mm"); Sj["lece_mm"] = float(lece)
        Sj["sigma_BL"] = {"date": "2026-09-27", "script": "q22_recompute_sigma_BL.py", "function": "conventions_check.sigma_body_local", "axes": "(Z,Y,X) + DVF (3,Z,Y,X) [dz,dy,dx]",
                          "body_mask": "runner._estimate_body_mask(fixed)", "body_fraction": float(body.mean()), **{k: v for k, v in KW.items()}}
        json.dump(Sj, open(os.path.join(od, "summary.json"), "w"), indent=2, default=str)
        np.savez(os.path.join(od, "landmarks.npz"), **lm)                  # written last: marks the case as done
        print(f"  {ds:17s} {os.path.relpath(d, DS_ROOT)[:60]:60s} body {body.mean()*100:4.1f}%  sigma_p mean {sp.mean():.3f} "
              f"(May {sp_old.mean():.3f})  ({time.time() - t0:.0f}s)", flush=True)
        del dz, s
    del f_t, m_t, b_t, bm_t; torch.cuda.empty_cache()
print(f"[done] {time.time() - t0:.0f}s")
