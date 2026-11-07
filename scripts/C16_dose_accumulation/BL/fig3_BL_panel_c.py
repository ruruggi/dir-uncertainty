# -*- coding: utf-8 -*-
"""
fig3_BL_panel_c.py -- Figure 3: the four candidate versions of row (c), rendered as strips
of the Figure 3 width, plus a sheet with the four options stacked, to choose one. Reads the step-8 runs (fig3_BL_runner.py);
evaluation geometry of the published numbers (fig_dvh_overlay.py). Read-only on the data.

  c1  accumulated FC dose (ensemble mean) with isodoses 50/80/95/100 % of 54 Gy, zoom on the PTV:  Heuristic | plan | Smolders
  c2  voxel SD of the accumulated dose (FC), Heuristic arms:  K anisotropic | K isotropic matched | difference
  c3  one accumulated sample minus the mean, same eps_k (FC):  Heuristic K aniso | Heuristic K iso | Smolders
      (k = the sample with the median RMS deviation inside PTV + 10 mm [isotropic, physical mm], Heuristic K aniso)
  c4  dose profiles through the PTV centroid, cranio-caudal and left-right: plan, mean and 95 % band, K aniso vs K iso
Sagittal views at the PTV-centroid x; aspect from the plan-grid spacing (the submitted builder used 2.0/0.91, the sCT pixel).
Metrics (JSON): SD of the accumulated dose in 0-5 mm shells inside / outside the PTV surface, by face (cranio-caudal vs lateral),
and the 1/e coherence length of the dose error e_k = D_k - mean along CC and LR, per arm.
Usage (venv): python fig3_BL_panel_c.py [--only c1,c4]
"""
from dirunc_paths import LIB, RESULTS_DATA  # data roots, see README
import os, sys, json, math, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LIB); sys.path.insert(0, LIB)
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np, pydicom
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from fig_dvh_overlay import RS_PATH, RD_PLAN, load_pct_post_rigid, rasterize_roi, resample_plan_dose

RES = os.environ.get("FIG3_RES", rf"{RESULTS_DATA}/Fig3_BL_20260929")   # env override; default = v2 run
OUTF = os.environ.get("FIG3_FIG_OUT", os.path.normpath(os.path.join(HERE, "..", "..", "Figures", "_da_approvare")))
os.makedirs(OUTF, exist_ok=True); TAG = os.environ.get("FIG3_TAG", "")   # suffix of the json outputs
ARMS = {"aniso": "Heuristic_aniso", "iso": os.environ.get("FIG3_ISO_ARM", "Heuristic_iso_nom"), "dl": "Smolders"}   # iso arm selectable (Heuristic_iso_nom = 11.45 mm); default = Heuristic_iso_nom, the arm of §3.6
RX = 54.0; ISO = [(0.50, ":"), (0.80, "--"), (0.95, "-"), (1.00, "-")]
ZOOM_MM = 30.0; DOSE_VMIN, DOSE_VMAX = 1.0, 58.0; CT_VMIN, CT_VMAX = -1000, 200

ap = argparse.ArgumentParser(); ap.add_argument("--only", default="c1,c2,c3,c4"); args = ap.parse_args()
ONLY = args.only.split(",")
os.makedirs(OUTF, exist_ok=True)

# ---------------------------------------------------------------- data
ct, v2w_post, inv_predef = load_pct_post_rigid(); nz, ny, nx = ct.shape
sp_xyz = json.load(open(os.path.join(RES, "Heuristic_aniso", "frac_01", "pass3_metadata.json")))["spacing"]
sz, sy, sx = sp_xyz[2], sp_xyz[1], sp_xyz[0]
rs = pydicom.dcmread(RS_PATH)
ptv = rasterize_roi(rs, "PTV54", ny, nx, nz, inv_predef, v2w_post).astype(bool)
zc, yc, xc = [int(round(c)) for c in np.argwhere(ptv).mean(axis=0)]
dose_plan = resample_plan_dose(RD_PLAN, ct, v2w_post, inv_predef)
have = {a: os.path.exists(os.path.join(RES, d, "dose_accumulated_fc_std.npy")) for a, d in ARMS.items()}
mean = {a: np.load(os.path.join(RES, d, "dose_accumulated_fc.npy")) for a, d in ARMS.items() if have[a]}
std = {a: np.load(os.path.join(RES, d, "dose_accumulated_fc_std.npy")) for a, d in ARMS.items() if have[a]}
per = {a: np.load(os.path.join(RES, d, "dose_acc_per_sample_fc.npy"), mmap_mode="r") for a, d in ARMS.items() if have[a]}
print(f"PTV centroid (z,y,x) = ({zc},{yc},{xc}); spacing z/y/x {sz:.3f}/{sy:.3f}/{sx:.3f}; arms available {[a for a in have if have[a]]}")

# sagittal zoom window around the PTV at x = xc
zz, yy = np.where(ptv[:, :, xc])
z0, z1 = max(0, zz.min() - int(ZOOM_MM / sz)), min(nz, zz.max() + int(ZOOM_MM / sz) + 1)
y0, y1 = max(0, yy.min() - int(ZOOM_MM / sy)), min(ny, yy.max() + int(ZOOM_MM / sy) + 1)
ASPECT = sz / sy
sag = lambda a: np.asarray(a[z0:z1, y0:y1, xc], np.float32)[::-1]
ct_s, ptv_s = sag(ct), sag(ptv.astype(np.float32))
EXT = None


def base(ax, title):
    ax.imshow(ct_s, cmap="gray", vmin=CT_VMIN, vmax=CT_VMAX, aspect=ASPECT, interpolation="bilinear")
    ax.set_title(title, fontsize=8.5); ax.axis("off")


def ptv_line(ax, color="yellow"):
    ax.contour(ptv_s, levels=[0.5], colors=color, linewidths=1.1, linestyles="dashed")


def cbar(fig, im, ax_list, label):
    cb = fig.colorbar(im, ax=ax_list, fraction=0.025, pad=0.01); cb.set_label(label, fontsize=7.5); cb.ax.tick_params(labelsize=7)


# k with the median RMS deviation inside PTV + 10 mm (Heuristic K aniso); same k for every arm (shared eps_k, FC)
# Dilation of 10 mm isotropic in physical mm (EDT with the (sz,sy,sx) sampling of the (Z,Y,X) array); before it was
# binary_dilation with iterations = round(10/sy) voxels on EVERY axis (= 2x the intended margin along Z, spacing 2 mm; rhombus shape)
roi = (ndi.distance_transform_edt(~ptv, sampling=(sz, sy, sx)) <= 10.0) & (np.abs(np.arange(nz)[:, None, None] - zc) * sz <= 40)
zr, yr, xr = [slice(int(i.min()), int(i.max()) + 1) for i in np.where(roi)]
rms = np.array([float(np.sqrt(((np.asarray(per["aniso"][k][zr, yr, xr]) - mean["aniso"][zr, yr, xr])[roi[zr, yr, xr]] ** 2).mean()))
                for k in range(per["aniso"].shape[0])])
K_SEL = int(np.argsort(rms)[len(rms) // 2])
print(f"c3 sample k = {K_SEL} (median RMS deviation in PTV+10 mm, K aniso: {rms[K_SEL]:.2f} Gy; range {rms.min():.2f}-{rms.max():.2f})")


def strip(fig_w=7.09, h=2.35):
    return plt.figure(figsize=(fig_w, h), dpi=200)


def draw_c1(fig):
    axs = fig.subplots(1, 3); im = None
    for ax, (t, d) in zip(axs, [("Heuristic-FC accumulated", mean["aniso"]), ("Original planned dose", dose_plan),
                                 ("Smolders-FC accumulated", mean["dl"])]):
        base(ax, t); s = sag(d)
        im = ax.imshow(np.where(s >= DOSE_VMIN, s, np.nan), cmap="jet", vmin=DOSE_VMIN, vmax=DOSE_VMAX, alpha=0.5,
                       aspect=ASPECT, interpolation="bilinear")
        for frac, ls in ISO:
            ax.contour(s, levels=[frac * RX], colors="black" if frac == 1.0 else "white", linewidths=0.8, linestyles=ls)
        ptv_line(ax, "magenta")
    cbar(fig, im, list(axs), "Dose (Gy)")
    return [Line2D([0], [0], color="black" if f == 1.0 else "white", ls=ls, lw=1.2, label=f"{int(f * 100)} %") for f, ls in ISO] + \
           [Line2D([0], [0], color="magenta", ls="--", lw=1.2, label="PTV54")]


def draw_c2(fig):
    axs = fig.subplots(1, 3, gridspec_kw={"wspace": 0.45})
    vmax = float(np.nanpercentile(sag(std["aniso"]), 99.5))
    for ax, (t, d) in zip(axs[:2], [("SD, $K$ anisotropic", std["aniso"]), ("SD, $K$ isotropic (matched)", std["iso"])]):
        base(ax, t); im = ax.imshow(sag(d), cmap="magma", vmin=0, vmax=vmax, alpha=0.75, aspect=ASPECT, interpolation="bilinear"); ptv_line(ax)
    dif = sag(std["aniso"] - std["iso"]); lim = float(np.nanpercentile(np.abs(dif), 99.5)) or 1e-3
    base(axs[2], "SD difference (aniso $-$ iso)")
    im2 = axs[2].imshow(dif, cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.85, aspect=ASPECT, interpolation="bilinear"); ptv_line(axs[2], "black")
    for ax, im_, lab in ((axs[1], im, "SD (Gy)"), (axs[2], im2, "Gy")):               # colorbars in their own inset axes: no overlap
        cb = fig.colorbar(im_, cax=ax.inset_axes([1.04, 0.0, 0.05, 1.0])); cb.set_label(lab, fontsize=7.5); cb.ax.tick_params(labelsize=7)


def draw_c3(fig):
    axs = fig.subplots(1, 3)
    sl = {a: sag(np.asarray(per[a][K_SEL]) - mean[a]) for a in ("aniso", "iso", "dl")}
    lim = float(max(np.nanpercentile(np.abs(v), 99.5) for v in sl.values()))
    for ax, (a, t) in zip(axs, [("aniso", f"Sample {K_SEL} $-$ mean, $K$ aniso"), ("iso", f"Sample {K_SEL} $-$ mean, $K$ iso"),
                                 ("dl", f"Sample {K_SEL} $-$ mean, Smolders")]):
        base(ax, t); im = ax.imshow(sl[a], cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.85, aspect=ASPECT, interpolation="bilinear"); ptv_line(ax, "black")
    cbar(fig, im, list(axs), "Dose deviation (Gy)")


def profiles(a):
    cc = np.asarray(per[a][:, :, yc, xc], np.float32); lr = np.asarray(per[a][:, zc, yc, :], np.float32)
    return cc, lr


def draw_c4(fig):
    axs = fig.subplots(1, 2)
    zmm = (np.arange(nz) - zc) * sz; xmm = (np.arange(nx) - xc) * sx
    pc, pl = ptv[:, yc, xc], ptv[zc, yc, :]
    cols = {"aniso": "#c0392b", "iso": "#2471a3"}
    for ax, axis_mm, plan_line, inside, idx, t in [(axs[0], zmm, dose_plan[:, yc, xc], pc, 0, "Cranio-caudal profile"),
                                                   (axs[1], xmm, dose_plan[zc, yc, :], pl, 1, "Left-right profile")]:
        keep = np.abs(axis_mm) <= 45
        for a in ("aniso", "iso"):
            p = profiles(a)[idx]
            ax.fill_between(axis_mm[keep], np.percentile(p, 2.5, 0)[keep], np.percentile(p, 97.5, 0)[keep], color=cols[a], alpha=0.18, lw=0)
            ax.plot(axis_mm[keep], p.mean(0)[keep], color=cols[a], lw=1.3, ls="-" if a == "aniso" else (0, (3, 2)))   # iso dashed: the means overlap
        ax.plot(axis_mm[keep], plan_line[keep], color="k", lw=0.9, ls="--")
        if inside.any():
            e = axis_mm[inside]; ax.axvspan(e.min(), e.max(), color="gold", alpha=0.15, lw=0)
        ax.axhline(0.95 * RX, color="gray", ls=":", lw=0.7)
        ax.set_title(t, fontsize=8.5); ax.set_xlabel("Distance from PTV centroid (mm)", fontsize=7.5); ax.tick_params(labelsize=7)
        ax.set_ylim(0, 62); ax.grid(alpha=0.3)
    axs[0].set_ylabel("Dose (Gy)", fontsize=7.5)
    axs[1].legend(handles=[Line2D([0], [0], color=cols["aniso"], lw=1.3, label="$K$ aniso, mean"),
                           Line2D([0], [0], color=cols["iso"], lw=1.3, ls=(0, (3, 2)), label="$K$ iso, mean"),
                           Line2D([0], [0], color="k", lw=0.9, ls="--", label="planned"),
                           Patch(color="gray", alpha=0.3, label="95 % band"), Patch(color="gold", alpha=0.3, label="PTV")],
                  fontsize=6, loc="lower center", framealpha=0.9)


DRAW = {"c1": (draw_c1, "c1 — dose media accumulata con isodosi, zoom sull'interfaccia"),
        "c2": (draw_c2, "c2 — SD della dose accumulata: K aniso | K iso | differenza"),
        "c3": (draw_c3, "c3 — un campione meno la media, stesso ε: K aniso | K iso | Smolders"),
        "c4": (draw_c4, "c4 — profili di dose CC e LR attraverso il PTV, banda 95 %: K aniso vs K iso")}
NEED = {"c1": ("aniso", "dl"), "c2": ("aniso", "iso"), "c3": ("aniso", "iso", "dl"), "c4": ("aniso", "iso")}

done = []
for c in ONLY:
    if not all(have[a] for a in NEED[c]):
        print(f"[skip] {c}: missing arms {[a for a in NEED[c] if not have[a]]}"); continue
    fig = strip(); leg = DRAW[c][0](fig)
    if leg:
        lg = fig.legend(handles=leg, title="isodose (% of 54 Gy)", title_fontsize=6.5, fontsize=6.5, loc="lower center", ncol=5,
                        frameon=True, facecolor="#7f7f7f", edgecolor="none", bbox_to_anchor=(0.5, -0.06))
        lg.get_title().set_color("white")
        for t in lg.get_texts():
            t.set_color("white")
    fig.text(0.005, 0.97, "(c)", fontsize=14, ha="left", va="top")
    p = os.path.join(OUTF, f"Figure3_BL_panel_{c}.png"); tmp = p[:-4] + "_tmp.png"
    fig.savefig(tmp, dpi=300, bbox_inches="tight"); plt.close(fig); os.replace(tmp, p); done.append(c)   # Windows: target may be open in a viewer
    print(f"saved {p}")

# ---------------------------------------------------------------- metrics
def shell_metrics(sd):
    dist_in = ndi.distance_transform_edt(ptv, sampling=(sz, sy, sx)); dist_out = ndi.distance_transform_edt(~ptv, sampling=(sz, sy, sx))
    Z, Y, X = np.meshgrid((np.arange(nz) - zc) * sz, (np.arange(ny) - yc) * sy, (np.arange(nx) - xc) * sx, indexing="ij")
    cc_face = np.abs(Z) > np.maximum(np.abs(Y), np.abs(X))
    out = {}
    for sh, m in (("in_0_5mm", ptv & (dist_in <= 5)), ("out_0_5mm", (~ptv) & (dist_out <= 5))):
        for face, fm in (("CC", cc_face), ("lateral", ~cc_face)):
            v = sd[m & fm]; out[f"{sh}_{face}"] = {"SD_mean_Gy": float(v.mean()), "n": int(v.size)}
    return out


def coherence(a, nk=100):
    """1/e length (mm) of the ACF of e_k = D_k - mean along z (CC) and x (LR), over the PTV + 10 mm box, averaged over samples."""
    zs, ys, xs = zr, yr, xr; m = roi[zs, ys, xs].astype(np.float32); res = {}
    for ax_, s_ in ((0, sz), (2, sx)):
        n = m.shape[ax_]; oth = tuple(i for i in range(3) if i != ax_)
        Fm = np.fft.rfft(m, n=2 * n, axis=ax_); cnt = np.fft.irfft(np.abs(Fm) ** 2, n=2 * n, axis=ax_).sum(axis=oth)
        acc = 0
        for k in range(min(nk, per[a].shape[0])):
            e = (np.asarray(per[a][k][zs, ys, xs], np.float32) - mean[a][zs, ys, xs]); e = (e - e[m > 0].mean()) * m
            F = np.fft.rfft(e, n=2 * n, axis=ax_); acc = acc + np.fft.irfft(np.abs(F) ** 2, n=2 * n, axis=ax_).sum(axis=oth)
        ac = (acc / np.maximum(cnt, 1))[:n]; ac = ac / ac[0]; thr = 1 / math.e
        j = int(np.argmax(ac < thr)) if np.any(ac < thr) else n - 1
        res["CC" if ax_ == 0 else "LR"] = float(((j - 1) + (ac[j - 1] - thr) / (ac[j - 1] - ac[j])) * s_) if j > 0 else 0.0
    return res


met = {"k_sample_c3": K_SEL, "zoom_mm": ZOOM_MM, "aspect_sagittal": ASPECT, "arms": {}}
for a in ARMS:
    if have[a]:
        met["arms"][a] = {"shells": shell_metrics(std[a]), "coherence_length_mm": coherence(a)}
        s = met["arms"][a]
        print(f"  {ARMS[a]:16s} shell SD in/out CC {s['shells']['in_0_5mm_CC']['SD_mean_Gy']:.2f}/{s['shells']['out_0_5mm_CC']['SD_mean_Gy']:.2f}  "
              f"lateral {s['shells']['in_0_5mm_lateral']['SD_mean_Gy']:.2f}/{s['shells']['out_0_5mm_lateral']['SD_mean_Gy']:.2f} Gy | "
              f"coherence CC {s['coherence_length_mm']['CC']:.1f} LR {s['coherence_length_mm']['LR']:.1f} mm", flush=True)
json.dump(met, open(os.path.join(HERE, "_out_fig3_BL", f"panel_c_metrics{TAG}.json"), "w"), indent=2)

# ---------------------------------------------------------------- options sheet
if len(done) > 1:
    imgs = [plt.imread(os.path.join(OUTF, f"Figure3_BL_panel_{c}.png")) for c in done]
    fig = plt.figure(figsize=(7.5, 2.6 * len(imgs)), dpi=150)
    for i, (c, im) in enumerate(zip(done, imgs)):
        ax = fig.add_subplot(len(imgs), 1, i + 1); ax.imshow(im); ax.axis("off")
        ax.set_title("Opzione " + DRAW[c][1], fontsize=9, loc="left")
    p = os.path.join(OUTF, "Figure3_BL_panel_c_options.png"); fig.savefig(p, dpi=150, bbox_inches="tight"); print(f"saved {p}")
