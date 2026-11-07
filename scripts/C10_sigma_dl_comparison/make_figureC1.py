# -*- coding: utf-8 -*-
"""Figure C1 (Appendix C; Q2.4.d (i), 2026-10-02): sigma_h vs sigma_DL on the same DVF, DIR-LAB Castillo case 5, Elastix, three planes.
Derived from Replies_last/make_figureB.py (3/9 annex) with the CORRECTED inputs:
  sigma_h  = f' * sigma_BL  (body-local sigma of 27/9, Datasets/_sigma_BL_20260927; f' = Table 3 v2 Elastix f, f/sqrt3 convention R43)
  sigma_DL = |sigma_DL| / sqrt(3)  (RMS over components, per displacement component as Figure 3 row (a), fig3_BL_build.py l.88-90); unscaled
ct and |sigma_DL| are re-read from the 3/9 cache (network output, unaffected by the recomputations). Writes Figures_modificabili/FigureC1.{png,tiff} (approved 2026-10-02)"""
from dirunc_paths import DATASETS, LIB, RESULTS  # data roots, see README
import os, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PAPER = LIB
OUTDIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Figures_modificabili"))   # approved 2026-10-02: final output
CACHE = RESULTS + "/_sigma_cmp_case5_elastix.npz"  # re-plot only, from the cache

CASE, ENGINE = "case5", "elastix"
import json
HERE = os.path.dirname(os.path.abspath(__file__))
F_H = json.load(open(os.path.join(HERE, "_out_q22_table23_v2", "table23_v2.json")))["results"]["T3|DIR-LAB|Elastix|sigma_h"]["f"]
F_DL = 1.0 / np.sqrt(3.0)          # |sigma_DL| -> per component; no calibration factor (unscaled)
SIGMA_BL = rf"{DATASETS}/_sigma_BL_20260927/Castillo-DIR-LAB/Results_elastix_procrustes/case5/sigma_raw.npy"
SIGMA_VMAX = None                 # set below from the p99 of both maps

# ---------------------------------------------------------------- volumes ---
if not os.path.exists(CACHE):
    sys.path.insert(0, PAPER)
    # importing the validation module performs the sys.path wiring for
    # utils/, dl/, dl_arch/, core/ in the sibling code repo
    from validation import smolders_dl_sigma_castillo as S
    from utils.scan import Scan
    from utils.vector_field import VectorField

    print(f"Loading {CASE} / {ENGINE} ...")
    pair = S.load_pair(S.CASTILLO_NIFTI, case_id=CASE)
    pair_pre, _, _ = S._apply_procrustes_to_pair(
        pair, out_dir=os.path.join(OUTDIR, "_tmp_procrustes"), save_artifacts=False)
    sp_xyz = tuple(float(v) for v in pair_pre.spacing_fixed[:3])
    vtw = np.identity(4)
    for i in range(3):
        vtw[i, i] = sp_xyz[i]

    fixed = Scan(array=pair_pre.fixed_arr[None, ...], vtwMatrix=vtw)
    moving = Scan(array=pair_pre.moving_arr[None, ...], vtwMatrix=vtw)
    dvf_arr = np.load(os.path.join(S.CASTILLO_BASE, f"Results_{ENGINE}_procrustes",
                                   CASE, "dvf_mu.npy")).astype(np.float32)
    dvf = VectorField(array=dvf_arr, vtwMatrix=vtw, inMM=True)

    print(f"  running {S.MODEL_NAME} inference ...")
    reg = S.load_registerer(model_name=S.MODEL_NAME)
    _, prob = reg.process(fixed, moving, dvf)
    sig = prob.getVectorFieldUncertainty()
    if hasattr(sig, "cpu"):
        sig = sig.cpu().numpy()
    sig = np.asarray(sig, dtype=np.float32)              # (3, X, Y, Z)
    sigma_dl = np.sqrt((sig ** 2).sum(axis=0))           # (X, Y, Z)

    sigma_h = np.load(os.path.join(S.CASTILLO_BASE, f"Results_{ENGINE}_procrustes",
                                   CASE, "sigma_raw.npy")).astype(np.float32)
    ct = np.asarray(pair_pre.fixed_arr, dtype=np.float32)

    # -> (Z, Y, X), the orientation used by the correlation-map figure
    tr = lambda a: np.ascontiguousarray(a.transpose(2, 1, 0))
    np.savez_compressed(CACHE, ct=tr(ct), sigma_h=tr(sigma_h), sigma_dl=tr(sigma_dl),
                        spacing_zyx=np.array(sp_xyz[::-1]))
    print("  cached ->", CACHE)

z = np.load(CACHE)
ct, sigma_dl_raw = z["ct"], z["sigma_dl"]
sigma_h_raw = np.ascontiguousarray(np.load(SIGMA_BL).astype(np.float32).transpose(2, 1, 0)); assert sigma_h_raw.shape == ct.shape, (sigma_h_raw.shape, ct.shape)
sp = tuple(float(v) for v in z["spacing_zyx"])           # (sz, sy, sx)
sigma_h = F_H * sigma_h_raw                              # mm
sigma_dl = F_DL * sigma_dl_raw                           # mm
Z, Y, X = ct.shape
print(f"shape (Z,Y,X) = {(Z, Y, X)}  spacing (z,y,x) = {sp} mm")

# ------------------------------------------------------------------- masks ---
import scipy.ndimage as ndi
m = ct > -500.0
lab, n = ndi.label(m)
if n:
    s = np.bincount(lab.ravel()); s[0] = 0
    m = lab == s.argmax()
for k in range(Z):
    m[k] = ndi.binary_fill_holes(m[k])
body = m.astype(bool)
aer = body & (ct <= -500.0)
print(f"body {body.sum()} voxels; aerated inside body {100*aer.sum()/body.sum():.1f}%")

SIGMA_VMAX = float(np.ceil(max(np.percentile(sigma_h[body], 99), np.percentile(sigma_dl[body], 99)) * 2) / 2)
print("shared scale 0 -", SIGMA_VMAX, "mm")
# ----------------------------------------------------------------- numbers ---
from scipy.stats import pearsonr, spearmanr
a, b = sigma_h[body].astype(np.float64), sigma_dl[body].astype(np.float64)
print(f"\nsigma_h  (f={F_H}): mean {a.mean():.3f}  sd {a.std():.3f}  CV {a.std()/a.mean():.3f}  "
      f"p1 {np.percentile(a,1):.2f}  p99 {np.percentile(a,99):.2f} mm")
print(f"sigma_DL (f={F_DL}): mean {b.mean():.3f}  sd {b.std():.3f}  CV {b.std()/b.mean():.3f}  "
      f"p1 {np.percentile(b,1):.2f}  p99 {np.percentile(b,99):.2f} mm")
rng = np.random.default_rng(0)
idx = rng.choice(a.size, size=min(300_000, a.size), replace=False)
PR = pearsonr(a[idx], b[idx])[0]; SR = spearmanr(a[idx], b[idx])[0]
print(f"\nagreement between the two maps: Pearson r = {PR:+.3f}   Spearman rho = {SR:+.3f}")
for nm, v in (("sigma_h", sigma_h), ("sigma_DL", sigma_dl)):
    g = np.gradient(v.astype(np.float32))
    gm = np.sqrt(sum(x ** 2 for x in g))[body]
    print(f"  {nm}: mean|grad|/mean = {gm.mean()/v[body].mean():.4f} per voxel")
print("\nin the aerated lung vs the rest of the body:")
for nm, v in (("sigma_h", sigma_h), ("sigma_DL", sigma_dl)):
    print(f"  {nm}: lung {v[aer].mean():.3f} mm   soft {v[body & ~aer].mean():.3f} mm   "
          f"ratio {v[aer].mean()/v[body & ~aer].mean():.3f}")

# ------------------------------------------------------------------ figure ---
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# plane positions: through the centroid of the aerated volume
zc, yc, xc = [int(round(c)) for c in ndi.center_of_mass(aer)]
print(f"\northogonal planes at z={zc}, y={yc}, x={xc}")

ALPHA_OVL = 0.72
CMAP = "hot"


def planes(vol):
    """(axial, coronal, sagittal) slices, masked to the body, + their aspects."""
    # Orientation, established empirically on this data (not from the affine,
    # whose axis codes read S for increasing index and would give the opposite):
    # row 0 of the (Z,Y,X) array is the SUPERIOR end, and imshow defaults to
    # origin="upper", so the slices are taken WITHOUT reversing z -> apices up,
    # diaphragm and liver down, i.e. the radiological convention. The earlier
    # [::-1] inverted head and feet.
    ax_ = np.where(body[zc], vol[zc], np.nan)                # (Y, X)
    co_ = np.where(body[:, yc, :], vol[:, yc, :], np.nan)    # (Z, X)
    sa_ = np.where(body[:, :, xc], vol[:, :, xc], np.nan)    # (Z, Y)
    return [(ax_, sp[1] / sp[2]), (co_, sp[0] / sp[2]), (sa_, sp[0] / sp[1])]


ct_pl = planes(ct.astype(np.float32))
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7})
fig = plt.figure(figsize=(7.09, 4.55), dpi=200)
gs = fig.add_gridspec(2, 3, hspace=0.22, wspace=0.06,
                      left=0.02, right=0.885, top=0.93, bottom=0.02)

PLANE = ["axial", "coronal", "sagittal"]
ROWS = [(sigma_h, r"$\sigma_h$ (this work)", F_H, "abc"),
        (sigma_dl, r"$\sigma_{DL}$ (Smolders)", F_DL, "def")]
im = None
for row, (vol, lbl, f, tags) in enumerate(ROWS):
    pl = planes(vol)
    first = None
    for col in range(3):
        ax = fig.add_subplot(gs[row, col])
        if col == 0:
            first = ax
        ax.imshow(ct_pl[col][0], cmap="gray", vmin=-1000, vmax=200,
                  aspect=ct_pl[col][1], interpolation="bilinear")
        im = ax.imshow(pl[col][0], cmap=CMAP, vmin=0, vmax=SIGMA_VMAX,
                       aspect=pl[col][1], interpolation="nearest", alpha=ALPHA_OVL)
        ax.set_title(f"({tags[col]}) {lbl}, {PLANE[col]}" if col == 0
                     else f"({tags[col]}) {PLANE[col]}")
        ax.axis("off")
    v = vol[body]
    ftxt = f"$f$={f:.2f}" if lbl.startswith(r"$\sigma_h") else "unscaled"
    first.text(0.02, 0.02,
               f"mean {v.mean():.2f} mm  CV {v.std()/v.mean():.3f}  {ftxt}",
               transform=first.transAxes, fontsize=6.4, color="white",
               ha="left", va="bottom",
               bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.4))

cax = fig.add_axes([0.895, 0.15, 0.014, 0.70])
plt.colorbar(im, cax=cax).set_label(r"$\sigma$ (mm)", fontsize=8)

# 2026-09-03: panels (g) joint distribution and (h) marginals removed (author's decision);
# rho, CV and lung/soft ratio are printed above and quoted in the caption.

png = os.path.join(OUTDIR, "FigureC1.png")
fig.savefig(png, dpi=300, bbox_inches="tight"); fig.savefig(png[:-4] + "_master.png", dpi=800, bbox_inches="tight")
from PIL import Image; Image.open(png[:-4] + "_master.png").convert("RGB").save(png[:-4] + ".tiff", compression="tiff_lzw", dpi=(800, 800)); os.remove(png[:-4] + "_master.png")
# (1000-dpi master not needed for the annex PNG)
print("\n[out]", png)
