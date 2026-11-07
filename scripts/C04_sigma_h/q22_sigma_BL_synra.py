# -*- coding: utf-8 -*-
"""
q22_sigma_BL_synra.py -- body-local heuristic sigma (conventions_check.sigma_body_local, adopted 2026-09-27) for the three SyNRA
cells without a local mu (Lung / Abdominal / Liver). Same computation as q22_recompute_sigma_BL.py (Procrustes, moving resampled
to fixed, arrays (Z,Y,X), DVF (3,Z,Y,X) [dz,dy,dx] mm, fixed/moving body = runner._estimate_body_mask), made portable so it can
run on the cluster mqscimoca (CPU) where the mu of those registrations are stored.

  python q22_sigma_BL_synra.py --device cpu --threads 24 --out OUT --cell Lung=RESULTS_ROOT [--cell ...]
         [--data Lung=DATASET_ROOT ...]  (override of summary.json case_dir)  [--cases case1,case2]  [--part i/n]
Output OUT/<cell>/<case>/: landmarks.npz (stored one, sigma_p_mm(_raw) replaced by sigma_BL; TRE etc. unchanged) and summary.json
(stored one + "sigma_BL" block, lece_mm recomputed at S = 1, lece_mm_May = previous value). Inputs never modified. Resume-safe.
Check per case: predict_landmark_sigma on the stored sigma_raw.npy reproduces the stored sigma_p (layout check).
"""
from dirunc_paths import LIB  # data roots, see README
import os, sys, types, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = LIB; PS = LIB + "/validation"
sys.path.insert(0, SCRIPTS); sys.path.insert(0, PS); sys.path.insert(0, HERE)
v = types.ModuleType("validation"); v.__path__ = [PS]; sys.modules["validation"] = v
ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=0); ap.add_argument("--out", required=True)
ap.add_argument("--cell", action="append", required=True); ap.add_argument("--data", action="append", default=[])
ap.add_argument("--cases", default=None); ap.add_argument("--part", default="1/1")
a = ap.parse_args()
import numpy as np, torch
if a.threads:
    torch.set_num_threads(a.threads)
from runner import _estimate_body_mask
import conventions_check as cc
from validation.load_abdominal_dir_qa import load_pair
from validation.run_pair import _apply_procrustes_to_pair, _resample_moving_to_fixed
from lece import predict_landmark_sigma, compute_lece

KW = dict(w_res_grad=0.7, w_jac=0.3, ratio_clip_percentile=0.99, jac_clip_percentile=0.95, use_minmax_norm=True,
          global_scale=1.0, suppression_alpha=1.0, similarity_metric="mind_ssc", similarity_kwargs={"radius": 1, "dilation": 2})
dev = torch.device(a.device)
cells = dict(c.split("=", 1) for c in a.cell); data = dict(d.split("=", 1) for d in a.data)
ip, n_parts = (int(x) for x in a.part.split("/"))
jobs = []
for cell, root in cells.items():
    for c in sorted(os.listdir(root), key=lambda s: (len(s), s)):
        d = os.path.join(root, c)
        if (a.cases is None or c in a.cases.split(",")) and all(os.path.isfile(os.path.join(d, f)) for f in ("dvf_mu.npy", "landmarks.npz", "summary.json")):
            jobs.append((cell, c, d))
jobs = jobs[ip - 1::n_parts]
print(f"{len(jobs)} cases (part {ip}/{n_parts}), device {dev}, torch threads {torch.get_num_threads()}", flush=True)
t0 = time.time()
for cell, c, d in jobs:
    od = os.path.join(a.out, cell, c)
    if os.path.isfile(os.path.join(od, "landmarks.npz")):
        continue
    t1 = time.time()
    S = json.load(open(os.path.join(d, "summary.json")))
    pair = load_pair(data.get(cell, S["case_dir"]), case_id=S["case_id"])
    pair, _, _ = _apply_procrustes_to_pair(pair, os.path.join(a.out, "_tmp_diag", cell, c), save_artifacts=False)
    mov = _resample_moving_to_fixed(pair.fixed_arr, pair.moving_arr, pair.spacing_fixed, pair.spacing_moving,
                                    fixed_affine=getattr(pair, "fixed_affine", None), moving_affine=getattr(pair, "moving_affine", None))
    sx, sy, sz = pair.spacing_fixed
    fz = np.ascontiguousarray(pair.fixed_arr.astype(np.float32).transpose(2, 1, 0))
    mz = np.ascontiguousarray(mov.astype(np.float32).transpose(2, 1, 0))
    body = _estimate_body_mask(fz); body_m = _estimate_body_mask(mz)
    lm = dict(np.load(os.path.join(d, "landmarks.npz"), allow_pickle=True))
    if os.path.isfile(os.path.join(d, "sigma_raw.npy")):
        chk = np.abs(predict_landmark_sigma(np.load(os.path.join(d, "sigma_raw.npy")), lm["lm_fixed_voxel"]) - lm["sigma_p_mm_raw"]).max()
        assert chk < 1e-5, f"{d}: stored sigma_p not reproduced ({chk})"
    f_t = torch.from_numpy(fz).to(dev); m_t = torch.from_numpy(mz).to(dev); b_t = torch.from_numpy(body).to(dev); bm_t = torch.from_numpy(body_m).to(dev)
    dz = torch.from_numpy(np.ascontiguousarray(np.load(os.path.join(d, "dvf_mu.npy")).astype(np.float32).transpose(0, 3, 2, 1))).to(dev)
    s = cc.sigma_body_local(f_t, m_t, dz, b_t, bm_t, (sz, sy, sx))
    s_xyz = np.ascontiguousarray(s.squeeze().cpu().numpy().transpose(2, 1, 0)).astype(np.float32)
    sp = predict_landmark_sigma(s_xyz, lm["lm_fixed_voxel"])
    os.makedirs(od, exist_ok=True)
    Sj = json.load(open(os.path.join(d, "summary.json"))); sp_old = lm["sigma_p_mm_raw"].copy()
    lm["sigma_p_mm"] = sp * float(lm.get("S_scaling_applied", 1.0)); lm["sigma_p_mm_raw"] = sp
    lece, _ = compute_lece(lm["tre_mm"], sp, bin_width=0.25)
    Sj["lece_mm_May"] = Sj.get("lece_mm"); Sj["lece_mm"] = float(lece)
    Sj["sigma_BL"] = {"date": time.strftime("%Y-%m-%d"), "script": "q22_sigma_BL_synra.py", "function": "conventions_check.sigma_body_local",
                      "device": str(dev), "torch": torch.__version__, "axes": "(Z,Y,X) + DVF (3,Z,Y,X) [dz,dy,dx]",
                      "body_mask": "runner._estimate_body_mask(fixed)", "body_fraction": float(body.mean()), **KW}
    json.dump(Sj, open(os.path.join(od, "summary.json"), "w"), indent=2, default=str)
    np.savez(os.path.join(od, "landmarks.npz"), **lm)                  # written last: marks the case as done
    print(f"  {cell:8s} {c:7s} body {body.mean()*100:4.1f}%  sigma_p mean {sp.mean():.4f} (stored {sp_old.mean():.4f})  "
          f"case {time.time() - t1:.0f}s  total {time.time() - t0:.0f}s", flush=True)
    del f_t, m_t, b_t, bm_t, dz, s
print(f"[done] {time.time() - t0:.0f}s", flush=True)
