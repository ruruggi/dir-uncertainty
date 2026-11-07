# -*- coding: utf-8 -*-
"""
fig3_BL_splithalf.py -- STEP 8, check of option c2 (user, 2026-09-28): is the SD difference K aniso - K iso of the accumulated dose
(+-0.8 Gy at the cranio-caudal faces of the PTV) real, or sampling noise of an SD estimated from N = 100 samples (~7 %)?
Split-half test: SD of the accumulated FC dose from samples 0-49 (A) and 50-99 (B), per arm (same eps_k in both arms).
  signal  d_A = SD_aniso(A) - SD_iso(A),  d_B = SD_aniso(B) - SD_iso(B)   -> must agree (correlation, sign, CC-face shell value)
  noise   n = SD_aniso(A) - SD_aniso(B) (same arm, different samples)     -> reference amplitude of the sampling noise
Region: PTV dilated 10 mm (isotropic, physical mm; 2026-09-29 A2) (voxels with SD > 0.5 Gy in either arm); shells 0-5 mm outside the PTV by face (CC vs lateral), as
fig3_BL_panel_c.py. Output: _out_fig3_BL/splithalf.json and Figures/_da_approvare/Figure3_BL_c2_splithalf.png. Read-only.
"""
from dirunc_paths import LIB, RESULTS_DATA  # data roots, see README
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LIB); sys.path.insert(0, LIB)
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np, pydicom
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fig_dvh_overlay import RS_PATH, load_pct_post_rigid, rasterize_roi

RES = os.environ.get("FIG3_RES", rf"{RESULTS_DATA}/Fig3_BL_20260929")   # 2026-09-29 D: env override; 2026-10-08: default = v2 (..._20260928 deleted)
OUTF = os.environ.get("FIG3_FIG_OUT", os.path.normpath(os.path.join(HERE, "..", "..", "Figures", "_da_approvare")))   # 2026-09-29 D
os.makedirs(OUTF, exist_ok=True); TAG = os.environ.get("FIG3_TAG", "")   # 2026-09-29 D: suffix of the json outputs
ct, v2w_post, inv_predef = load_pct_post_rigid(); nz, ny, nx = ct.shape
sp = json.load(open(os.path.join(RES, "Heuristic_aniso", "frac_01", "pass3_metadata.json")))["spacing"]; sz, sy, sx = sp[2], sp[1], sp[0]
ptv = rasterize_roi(pydicom.dcmread(RS_PATH), "PTV54", ny, nx, nz, inv_predef, v2w_post).astype(bool)
zc, yc, xc = [int(round(c)) for c in np.argwhere(ptv).mean(axis=0)]
roi = ndi.distance_transform_edt(~ptv, sampling=(sz, sy, sx)) <= 10.0   # 2026-09-29 A2: 10 mm isotropic in physical mm (was round(10/sy) voxels on every axis, i.e. 20 mm along Z)
bz, by, bx = [slice(int(i.min()), int(i.max()) + 1) for i in np.where(roi)]


def half_sd(arm):
    """SD over samples 0-49 and 50-99 inside the ROI box (population SD, as dose_accumulated_fc_std)."""
    per = np.load(os.path.join(RES, arm, "dose_acc_per_sample_fc.npy"), mmap_mode="r"); K = per.shape[0]; h = K // 2
    out = []
    for ks in (range(0, h), range(h, K)):
        s1 = 0.0; s2 = 0.0
        for k in ks:
            d = np.asarray(per[k][bz, by, bx], np.float64); s1 = s1 + d; s2 = s2 + d * d
        m = s1 / len(ks); out.append(np.sqrt(np.maximum(s2 / len(ks) - m * m, 0)).astype(np.float32))
    return out


A_an, B_an = half_sd("Heuristic_aniso"); A_is, B_is = half_sd(os.environ.get("FIG3_ISO_ARM", "Heuristic_iso_nom"))   # 2026-09-30: iso arm selectable; 2026-10-08: default = Heuristic_iso_nom, the arm of §3.6
dA, dB, noise = A_an - A_is, B_an - B_is, A_an - B_an
P = ptv[bz, by, bx]; R = roi[bz, by, bx] & ((A_an > 0.5) | (A_is > 0.5))
dist_out = ndi.distance_transform_edt(~P, sampling=(sz, sy, sx))
Z, Y, X = np.meshgrid((np.arange(nz)[bz] - zc) * sz, (np.arange(ny)[by] - yc) * sy, (np.arange(nx)[bx] - xc) * sx, indexing="ij")
cc = np.abs(Z) > np.maximum(np.abs(Y), np.abs(X)); shell = (~P) & (dist_out <= 5)
rep = {"n_per_half": 50, "roi_voxels": int(R.sum())}
rep["corr_dA_dB"] = float(np.corrcoef(dA[R], dB[R])[0, 1])
rep["sign_agreement_where_|d|>0.2Gy"] = float(np.mean(np.sign(dA[R & (np.abs(dA) > 0.2)]) == np.sign(dB[R & (np.abs(dA) > 0.2)])))
rep["rms_signal_mean_of_halves_Gy"] = float(np.sqrt((((dA + dB) / 2)[R] ** 2).mean()))
rep["rms_signal_half_difference_Gy"] = float(np.sqrt((((dA - dB) / 2)[R] ** 2).mean()))     # noise left in the paired difference
rep["rms_noise_same_arm_Gy"] = float(np.sqrt((noise[R] ** 2).mean()))
for face, fm in (("CC", cc), ("lateral", ~cc)):
    m = shell & fm
    rep[f"shell_out_{face}"] = {"dA_mean_Gy": float(dA[m].mean()), "dB_mean_Gy": float(dB[m].mean()),
                               "SD_aniso_A": float(A_an[m].mean()), "SD_iso_A": float(A_is[m].mean()),
                               "SD_aniso_B": float(B_an[m].mean()), "SD_iso_B": float(B_is[m].mean()),
                               "noise_same_arm_mean_Gy": float(noise[m].mean())}
json.dump(rep, open(os.path.join(HERE, "_out_fig3_BL", f"splithalf{TAG}.json"), "w"), indent=2)
for k, v in rep.items():
    print(f"  {k}: {v}")

# figure: sagittal through the PTV centroid (same orientation as fig3_BL_panel_c.py)
xs = xc - bx.start; ASPECT = sz / sy
sl = lambda a: a[:, :, xs][::-1]
lim = float(np.percentile(np.abs(np.concatenate([dA[R], dB[R]])), 99.5))
fig, axs = plt.subplots(1, 3, figsize=(7.09, 2.5), dpi=200, gridspec_kw={"wspace": 0.15})
for ax, (t, d) in zip(axs, [("SD aniso $-$ iso, samples 0–49", dA), ("SD aniso $-$ iso, samples 50–99", dB),
                              ("noise: SD aniso 0–49 $-$ 50–99", noise)]):
    im = ax.imshow(np.where(sl(roi[bz, by, bx]), sl(d), np.nan), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect=ASPECT, interpolation="bilinear")
    ax.contour(sl(P.astype(float)), levels=[0.5], colors="k", linewidths=0.9, linestyles="dashed"); ax.set_title(t, fontsize=8); ax.axis("off")
cb = fig.colorbar(im, ax=list(axs), fraction=0.02, pad=0.01); cb.set_label("Gy", fontsize=7.5); cb.ax.tick_params(labelsize=7)
fig.suptitle(f"Split-half check of option c2: corr(d$_A$, d$_B$) = {rep['corr_dA_dB']:.2f}", fontsize=8.5, y=1.02)
p = os.path.join(OUTF, "Figure3_BL_c2_splithalf.png"); tmp = p[:-4] + "_tmp.png"; fig.savefig(tmp, dpi=300, bbox_inches="tight"); os.replace(tmp, p)
print("saved", p)
