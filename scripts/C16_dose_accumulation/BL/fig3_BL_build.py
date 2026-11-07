"""fig3_BL_build.py -- Figure 3 from the corrected runs (fig3_BL_runner.py), row (c) = option c1.
Copy of _build_figure3_v15.py (md5 64bfb23d2371faa148e1ce88ef870d24, not modified). Changes, all marked BL:
  - inputs <RESULTS_DATA>/Fig3_BL_20260929/{Heuristic_aniso,Smolders}; outputs Figures/_da_approvare/Figure3_BL_c1.*
  - DVF channels: the step-8 files are (3,X,Y,Z) with channel c along axis c = [dx,dy,dz]; the submitted runs had [dz,dy,dx]
    (utils.dicom_vf reversal, N9) -> load_dvf reverses the channels so that vec[2] = dx, vec[1] = dy as the quiver code expects
  - scales from the data: sigma colour scale and the perturbation-arrow factor (sigma and delta are no longer attenuated, N8)
  - row (c): c1 = accumulated FC dose with isodoses 50/80/95/100 % of 54 Gy, zoom +-30 mm around the PTV, sagittal aspect from the
    plan-grid spacing (the submitted builder used 2.0/0.91, the sCT pixel: ~7 % vertical stretch)
  - no backup / no write into the submission folder
Original description follows.

Build Figure 3 v15 — 4 rows with post-fix Smolders data + restored μ DVF row.

Row (a)  axial   σ-maps            : Heuristic σ_h | Smolders σ_DL,  0–5 mm
Row (b)  axial   μ DVF (Option B)  : Heuristic ⟨μ⟩ | Original μ | Smolders ⟨μ⟩
                                     heatmap turbo + sparse white quiver
Row (c)  sagittal dose             : Heuristic-FC | Plan | Smolders-FC,  0–55 Gy
Row (d)  DVH                       : Heuristic-FC | Smolders-FC, 6 ROIs
         dual legend matching v13 (ROI in left panel, line-style in right panel)
"""
from dirunc_paths import LIB, RESULTS_DATA  # data roots, see README
import os, sys
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError: pass
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pydicom

DIR_V2 = LIB
sys.path.insert(0, DIR_V2); sys.path.insert(0, os.path.join(DIR_V2, '_lung_batch'))
from fig_dvh_overlay import (
    RS_PATH, RD_PLAN, STRUCTURES,
    load_pct_post_rigid, rasterize_roi, resample_plan_dose, dvh_cumulative,
)

# Override via env var to regenerate from an alternative run directory without
# duplicating the script. Default = original behaviour.
RES = os.environ.get('FIG3_RES', rf'{RESULTS_DATA}/Fig3_BL_20260929')                        # BL
_HERE = os.path.dirname(os.path.abspath(__file__))                                                  # BL
OUT_DIR = os.path.normpath(os.path.join(_HERE, '..', '..', 'Figures', '_da_approvare'))              # BL
OUT_PREVIEW = os.environ.get('FIG3_OUT', os.path.join(OUT_DIR, '_preview_Figure3_BL_c1.png'))       # BL
# Same figure on the in-house registration chain: the deterministic field of the
# central panel of (b) is then no longer the Velocity one, and the output files
# must not overwrite the existing Figure3 in the same folder.
MU_LABEL = os.environ.get('FIG3_MU_LABEL', 'Velocity')
BASENAME = os.environ.get('FIG3_BASENAME', 'Figure3_BL_c1')                                         # BL
# Names of the two arm subfolders: used to compare two f calibrations
# (e.g. Heuristic_f211) without junctions or copies on the filesystem.
ARM_H = os.environ.get('FIG3_ARM_H', 'Heuristic_aniso')                                             # BL
ARM_S = os.environ.get('FIG3_ARM_S', 'Smolders')
# Label of f in the row (a) title: changes with the calibration used.
F_LABEL = os.environ.get('FIG3_F_LABEL', '2.23')   # 2.23 = f of the v2 runs (fig3_BL_runner.py --f)

# ============================================================================
# 1) Anatomy, ROIs, plan dose
# ============================================================================
print('Loading anatomy ...')
ct, v2w_post, inv_predef = load_pct_post_rigid()
nz, ny, nx = ct.shape
rs = pydicom.dcmread(RS_PATH)
ptv = rasterize_roi(rs, 'PTV54', ny, nx, nz, inv_predef, v2w_post)
zc, yc, xc = [int(np.round(c)) for c in np.argwhere(ptv).mean(axis=0)]
print(f'  PTV centroid (z,y,x) = ({zc}, {yc}, {xc})')

masks = {}
for name, _ in STRUCTURES:
    m = rasterize_roi(rs, name, ny, nx, nz, inv_predef, v2w_post)
    if m is not None and m.sum() > 0:
        masks[name] = m
# BL (audit D2/N7): the submitted display mask ct > -500 excluded the aerated lung, so sigma looked absent in the lung parenchyma
# although it is computed there; use the body mask of the run (HU > -500, largest component, per-slice hole filling), (X,Y,Z) -> (Z,Y,X)
body = np.load(os.path.join(RES, ARM_H, 'frac_01', 'body_mask.npy')).astype(bool).transpose(2, 1, 0)
assert body.shape == ct.shape, (body.shape, ct.shape)
body_crop = (ct > -500)   # BL: crop windows computed exactly as in the submitted figure (same framing)

print('Resampling plan dose ...')
dose_plan = resample_plan_dose(RD_PLAN, ct, v2w_post, inv_predef)

# ============================================================================
# 2) σ-maps + dose + DVF + DVH cache
# ============================================================================
print('Loading σ-maps + accumulated FC dose ...')
sigma_heu = np.load(f'{RES}/{ARM_H}/frac_01/uncertainty_dir.npy')
sigma_smo = np.load(f'{RES}/{ARM_S}/frac_01/uncertainty_dir.npy')
# BL: uncertainty_dir of the Smolders arm is |sigma_DL| (norm of the 3 components, as runner.py); sigma_h is the
# per-component value G. Row (a) shows both per component: sigma_DL RMS over components = |sigma_DL| / sqrt(3).
sigma_smo = sigma_smo / np.sqrt(3.0)
dose_heu  = np.load(f'{RES}/{ARM_H}/dose_accumulated_fc.npy')
dose_smo  = np.load(f'{RES}/{ARM_S}/dose_accumulated_fc.npy')

print('Loading μ DVFs ...')
def load_dvf(path):
    m = np.load(path)
    if m.shape[1:] == (nx, ny, nz):
        m = np.transpose(m, (0, 3, 2, 1))   # (3, X, Y, Z) -> (3, Z, Y, X)
    return np.ascontiguousarray(m[::-1])    # BL: step-8 channels [dx,dy,dz] -> [dz,dy,dx] as the quiver code expects
mu_orig = load_dvf(f'{RES}/{ARM_H}/frac_01/mu_dvf.npy')
# row (b): median-distance ensemble samples (see _find_median_samples.py)
import json
meta_heu = json.load(open(f'{RES}/{ARM_H}/frac_01/dvf_sample_median.json'))
meta_smo = json.load(open(f'{RES}/{ARM_S}/frac_01/dvf_sample_median.json'))
sample_heu = load_dvf(f'{RES}/{ARM_H}/frac_01/dvf_sample_median.npy')
sample_smo = load_dvf(f'{RES}/{ARM_S}/frac_01/dvf_sample_median.npy')
print(f'  Heuristic median sample: k={meta_heu["k_median"]} '
      f'L2={meta_heu["l2_dist_at_median"]:.0f}')
print(f'  Smolders  median sample: k={meta_smo["k_median"]} '
      f'L2={meta_smo["l2_dist_at_median"]:.0f}')

def load_dvh(path):
    d = np.load(path, allow_pickle=True)
    return dict(centers=d['centers'], names=list(d['names']),
                colors=list(d['colors']), v_plan=d['v_plan'],
                v_mean=d['v_mean'], v_lo=d['v_lo'], v_hi=d['v_hi'],
                K=int(d['K']))
print('Loading DVH caches ...')
dvh_heu = load_dvh(f'{RES}/{ARM_H}/_dvh_cache.npz')
dvh_smo = load_dvh(f'{RES}/{ARM_S}/_dvh_cache.npz')

# ============================================================================
# 3) Crop windows
# ============================================================================
# Axial (rows a, b): right-lung + vertebra + ~3 cm mediastinum crop (as in
# _preview_Figure3_v14.png). The DVF, σ and dose-relevant action concentrates
# on the right hemithorax in this Lung SBRT case (PTV is right-lung); cropping
# discards the contralateral lung area which adds no information.
slc_body  = body_crop[zc]
midline_x = nx // 2
ys_body, xs_body = np.where(slc_body)
right_xs = xs_body[xs_body < midline_x]
right_ys = ys_body[xs_body < midline_x]
xa_lo = max(0, int(right_xs.min()) - 5)
xa_hi = min(nx - 1, midline_x + 30)        # 3 cm beyond midline (vertebra + left mediastinum)
_ya_lo_full = max(0, int(right_ys.min()) - 5)
_ya_hi_full = min(ny - 1, int(right_ys.max()) + 5)
# Crop the posterior third (treatment couch + sub-couch artefacts); keep the
# anterior two thirds of the antero-posterior extent.
ya_lo = _ya_lo_full
ya_hi = _ya_lo_full + int(2 / 3 * (_ya_hi_full - _ya_lo_full))

# Sagittal (row c): at PTV centroid x
sag_x = xc
SAG_ASPECT = 2.0 / 0.91   # vox_z / vox_xy
zs_body = np.where(body_crop[:, :, sag_x].any(axis=1))[0]
zs_lo = max(0, int(zs_body.min()) - 2); zs_hi = min(nz - 1, int(zs_body.max()) + 2)
ys_sag = np.where(body_crop[:, :, sag_x].any(axis=0))[0]
_ys_lo_full = max(0, int(ys_sag.min()) - 5)
_ys_hi_full = min(ny - 1, int(ys_sag.max()) + 5)
# Crop the posterior quarter of the AP extent on the sagittal view: the right
# edge of each sagittal panel shows the air pocket under the treatment couch.
ys_lo = _ys_lo_full
ys_hi = _ys_lo_full + int(3 / 4 * (_ys_hi_full - _ys_lo_full))

# ============================================================================
# 4) Helpers
# ============================================================================
CT_VMIN, CT_VMAX = -1000, 200

def axial(arr, mask=True):
    s = arr[zc, ya_lo:ya_hi, xa_lo:xa_hi].astype(np.float32)
    if mask:
        s = np.where(body[zc, ya_lo:ya_hi, xa_lo:xa_hi], s, np.nan)
    return s

def sag(arr, mask=True):
    s = arr[zs_lo:zs_hi, ys_lo:ys_hi, sag_x].astype(np.float32)
    if mask:
        s = np.where(body[zs_lo:zs_hi, ys_lo:ys_hi, sag_x], s, np.nan)
    return s[::-1, :]

# ============================================================================
# 5) Figure compose — PMB 180 x 230 mm
# ============================================================================
print('Composing figure ...')
fig = plt.figure(figsize=(7.09, 9.06), dpi=200)
gs = fig.add_gridspec(4, 6,
                      height_ratios=[1.0, 1.0, 0.85, 1.15],
                      hspace=0.16, wspace=0.06,
                      left=0.06, right=0.91, top=0.965, bottom=0.045)

ct_axial = axial(ct, mask=False)
ptv_axial = ptv[zc, ya_lo:ya_hi, xa_lo:xa_hi].astype(np.uint8)
ct_sag = sag(ct, mask=False)
ptv_sag = ptv[zs_lo:zs_hi, ys_lo:ys_hi, sag_x].astype(np.uint8)[::-1, :]

# ---- Row (a): σ-maps, 2 cols ---------------------------------------------
SIGMA_VMAX = float(np.ceil(max(np.percentile(sigma_heu[zc][body[zc]], 99.5),                        # BL: from the data
                               np.percentile(sigma_smo[zc][body[zc]], 99.5))))
print(f'  sigma colour scale 0-{SIGMA_VMAX:.0f} mm')
ax_a1 = fig.add_subplot(gs[0, 1:3])
ax_a2 = fig.add_subplot(gs[0, 3:5])
for ax, title, s_arr in [
    (ax_a1, rf'Heuristic  $\sigma_h$  (f={F_LABEL})',      axial(sigma_heu)),
    (ax_a2, r'Smolders  $\sigma_{DL}$  (L20S3-4)',        axial(sigma_smo))]:
    ax.imshow(ct_axial, cmap='gray', vmin=CT_VMIN, vmax=CT_VMAX, aspect='equal',
              interpolation='bilinear')
    im_sig = ax.imshow(s_arr, cmap='hot', vmin=0, vmax=SIGMA_VMAX,
                       alpha=0.65, aspect='equal', interpolation='nearest')
    if ptv_axial.any():
        ax.contour(ptv_axial, levels=[0.5], colors='cyan', linewidths=0.8)
    ax.set_title(title, fontsize=8.5)
    ax.axis('off')
cax_a = fig.add_axes([0.918, 0.785, 0.012, 0.165])
plt.colorbar(im_sig, cax=cax_a, label='σ (mm)', ticks=list(range(0, int(SIGMA_VMAX) + 1, 2 if SIGMA_VMAX > 5 else 1)))   # BL

# ---- Row (b) NEW: μ DVF Option B, 3 cols ---------------------------------
def mag3d(mu_arr):
    return np.sqrt(mu_arr[0]**2 + mu_arr[1]**2 + mu_arr[2]**2)

mag_orig = mag3d(mu_orig)
in_body  = mag_orig[zc][body[zc]]
VMAX_DVF = float(np.percentile(in_body, 99.5))
VMAX_DVF = round(VMAX_DVF + 0.5)
print(f'  μ DVF VMAX (body p99.5 ceil) = {VMAX_DVF} mm')

# Delta = sample - mean = σ⊙ε per-voxel. Mean ≈ Velocity for both methods so
# delta = sample - Velocity gives identical info; we use the proper mean.
mu_heu_mean = load_dvf(f'{RES}/{ARM_H}/frac_01/dvf_mean_vec.npy')
mu_smo_mean = load_dvf(f'{RES}/{ARM_S}/frac_01/dvf_mean_vec.npy')
delta_heu = sample_heu - mu_heu_mean
delta_smo = sample_smo - mu_smo_mean
mag_delta_heu = mag3d(delta_heu)
mag_delta_smo = mag3d(delta_smo)
print(f'  |delta_heu| mean in body = {mag_delta_heu[zc][body[zc]].mean():.3f} mm  '
      f'p99 = {np.percentile(mag_delta_heu[zc][body[zc]], 99):.3f} mm')
print(f'  |delta_smo| mean in body = {mag_delta_smo[zc][body[zc]].mean():.3f} mm  '
      f'p99 = {np.percentile(mag_delta_smo[zc][body[zc]], 99):.3f} mm')

STEP = 14
# BL: the same arrow magnification x4 in all three panels of row (b)
FACTOR_BULK = FACTOR_DELTA = 4.0
Y_grid, X_grid = np.mgrid[0:(ya_hi-ya_lo):STEP, 0:(xa_hi-xa_lo):STEP]
body_axial = body[zc, ya_lo:ya_hi, xa_lo:xa_hi]
Bs = body_axial[::STEP, ::STEP]

ax_b1 = fig.add_subplot(gs[1, 0:2])
ax_b2 = fig.add_subplot(gs[1, 2:4])
ax_b3 = fig.add_subplot(gs[1, 4:6])

def draw_dvf_panel(ax, title, vec_arr, factor):
    """Heatmap: always |μ_Velocity| (shared baseline). Quiver: vec_arr × factor."""
    ax.imshow(ct_axial, cmap='gray', vmin=CT_VMIN, vmax=CT_VMAX, aspect='equal',
              interpolation='bilinear')
    heat = axial(mag_orig)
    im_dvf = ax.imshow(heat, cmap='turbo', vmin=0, vmax=VMAX_DVF,
                       alpha=0.55, aspect='equal', interpolation='nearest')
    U = vec_arr[2, zc, ya_lo:ya_hi, xa_lo:xa_hi]   # dx
    V = vec_arr[1, zc, ya_lo:ya_hi, xa_lo:xa_hi]   # dy
    Us = U[::STEP, ::STEP] * factor
    Vs = V[::STEP, ::STEP] * factor
    ax.quiver(X_grid[Bs], Y_grid[Bs], Us[Bs], Vs[Bs],
              color='white', scale_units='xy', scale=1, width=0.004,
              alpha=1.0, edgecolor='black', linewidth=0.3,
              headwidth=4, headlength=5)
    if ptv_axial.any():
        ax.contour(ptv_axial, levels=[0.5], colors='cyan', linewidths=0.8)
    ax.set_title(title, fontsize=8.5)
    ax.axis('off')
    return im_dvf

_kh = meta_heu["k_median"]; _ks = meta_smo["k_median"]
im_dvf = draw_dvf_panel(ax_b1,
    rf'$\delta_h - \mu$' + '\n'
    + rf'({_kh}$^{{\mathrm{{th}}}}$ sample; arrows ×{FACTOR_DELTA:.0f})',
    delta_heu, FACTOR_DELTA)
draw_dvf_panel(ax_b2,
    rf'$\mu_{{\mathrm{{{MU_LABEL}}}}}$' + '\n'
    + rf'(arrows ×{FACTOR_BULK:.0f})',
    mu_orig, FACTOR_BULK)
draw_dvf_panel(ax_b3,
    rf'$\delta_{{DL}} - \mu$' + '\n'
    + rf'({_ks}$^{{\mathrm{{th}}}}$ sample; arrows ×{FACTOR_DELTA:.0f})',
    delta_smo, FACTOR_DELTA)
cax_b = fig.add_axes([0.918, 0.555, 0.012, 0.205])
plt.colorbar(im_dvf, cax=cax_b, label='DVF magnitude (mm)')

# ---- Row (c): sagittal dose, 3 cols --------------------------------------
DOSE_VMIN = 1.0
DOSE_VMAX = 54.0
ax_c1 = fig.add_subplot(gs[2, 0:2])
ax_c2 = fig.add_subplot(gs[2, 2:4])
ax_c3 = fig.add_subplot(gs[2, 4:6])
# Approach used by an earlier version of this figure (v10):
# plain imshow of the dose without any body mask, masking only voxels with
# dose < DOSE_VMIN to NaN. This avoids the (ct > -500) body envelope, which
# excluded lung parenchyma (HU < -500) and prevented the entrance/exit beam
# strip from appearing across the lung.
# BL: option c1 -- zoom +-30 mm around the PTV, isodoses 50/80/95/100 % of 54 Gy, sagittal aspect
# from the plan-grid spacing; same rendering as fig3_BL_panel_c.py (option sheet).
DOSE_VMAX = 58.0; RX = 54.0
# BL: four isodoses, all white and each with its own dash pattern; solid black only for the PTV
ISO = [(0.50, ':'), (0.80, (0, (4, 2))), (0.95, '-.'), (1.00, (0, (5, 1.5, 1, 1.5, 1, 1.5)))]
_sp = json.load(open(f'{RES}/{ARM_H}/frac_01/pass3_metadata.json'))['spacing']      # (sx, sy, sz)
SAG_ASPECT_BL = _sp[2] / _sp[1]
_zz, _yy = np.where(ptv[:, :, sag_x])
cz0, cz1 = max(0, _zz.min() - int(30 / _sp[2])), min(nz, _zz.max() + int(30 / _sp[2]) + 1)
cy0, cy1 = max(0, _yy.min() - int(30 / _sp[1])), min(ny, _yy.max() + int(30 / _sp[1]) + 1)
_zoom = lambda a: np.asarray(a[cz0:cz1, cy0:cy1, sag_x], np.float32)[::-1]
ct_zoom, ptv_zoom = _zoom(ct), _zoom(ptv.astype(np.float32))
for ax, title, d_arr in [
    (ax_c1, 'Heuristic-FC accumulated', dose_heu),
    (ax_c2, 'Original planned dose',     dose_plan),
    (ax_c3, 'Smolders-FC accumulated',   dose_smo)]:
    ax.imshow(ct_zoom, cmap='gray', vmin=CT_VMIN, vmax=CT_VMAX, aspect=SAG_ASPECT_BL, interpolation='bilinear')
    s = _zoom(d_arr)
    im_dose = ax.imshow(np.where(s >= DOSE_VMIN, s, np.nan), cmap='jet', vmin=DOSE_VMIN, vmax=DOSE_VMAX,
                        alpha=0.5, aspect=SAG_ASPECT_BL, interpolation='bilinear')
    for frac, ls in ISO:
        ax.contour(s, levels=[frac * RX], colors='white', linewidths=0.75, linestyles=[ls])
    ax.contour(ptv_zoom, levels=[0.5], colors='black', linewidths=1.0, linestyles='solid')
    ax.set_title(title, fontsize=8.5)
    ax.axis('off')
# BL: legend without title, on one line across the three panels of row (c), over their lower edge (inside the
# row, clear of the titles of row (d)), font as the panel titles (8.5 pt)
_iso_h = [Line2D([0], [0], color='white', ls=ls, lw=1.3, label=f'{int(f * 100)} %') for f, ls in ISO] + \
         [Line2D([0], [0], color='black', ls='-', lw=1.3, label='PTV54')]
fig.canvas.draw()                                   # the image axes shrink to their aspect at draw time: read the real extents
_p1, _p3 = ax_c1.get_position(), ax_c3.get_position()
_lg = fig.legend(handles=_iso_h, loc='lower left', bbox_to_anchor=(_p1.x0, _p1.y0 + 0.002, _p3.x1 - _p1.x0, 0.026), mode='expand',
                 ncol=5, fontsize=8.5, frameon=True, facecolor='#8c8c8c', edgecolor='none', framealpha=0.92, handlelength=3.2,
                 borderaxespad=0.0, borderpad=0.35, columnspacing=1.0)
for _t in _lg.get_texts():
    _t.set_color('white')
cax_c = fig.add_axes([0.918, 0.335, 0.012, 0.16])
plt.colorbar(im_dose, cax=cax_c, label='Dose (Gy)')

# ---- Row (d): DVH overlay, 2 cols, dual legend (v13 style) ---------------
ax_d1 = fig.add_subplot(gs[3, 0:3])
ax_d2 = fig.add_subplot(gs[3, 3:6])

def draw_dvh(ax, title, dvh, show_ylabel):
    centers = dvh['centers']
    for i, name in enumerate(dvh['names']):
        color = dvh['colors'][i]
        ax.fill_between(centers, dvh['v_lo'][i], dvh['v_hi'][i],
                        color=color, alpha=0.20)
        ax.plot(centers, dvh['v_mean'][i], color=color, lw=1.4)
        ax.plot(centers, dvh['v_plan'][i], color=color, lw=0.9, linestyle='--')
    ax.axvline(54.0, color='gray', linestyle=':', lw=0.8)
    ax.set_xlabel('Dose (Gy)', fontsize=8.5)
    if show_ylabel:
        ax.set_ylabel('Volume (%)', fontsize=8.5)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 60); ax.set_ylim(0, 105)
    ax.tick_params(labelsize=7.5)

draw_dvh(ax_d1, 'Heuristic-FC ensemble DVH', dvh_heu, show_ylabel=True)
draw_dvh(ax_d2, 'Smolders-FC ensemble DVH',   dvh_smo, show_ylabel=False)

# Legends centred in each panel, placed at the centre of the
# data-empty region where neither PTV plateau nor OAR tails pass: around
# (Dose~25–35 Gy, Volume~30–60%). 'center' bbox anchors the legend midpoint.
roi_handles = [Line2D([0], [0], color=dvh_heu['colors'][i], lw=2.0,
                       label=dvh_heu['names'][i])
                for i in range(len(dvh_heu['names']))]
# Place the ROI legend in the high-dose mid-volume "empty" region (35-50 Gy,
# 30-70% volume): above the OAR tails (which have already dropped to 0%),
# below the PTV plateau (~100%), and to the right of the Lung_R_out tail.
leg_roi = ax_d1.legend(handles=roi_handles, loc='center',
                       bbox_to_anchor=(0.50, 0.50),
                       fontsize=6.0, frameon=True, framealpha=0.92,
                       handlelength=1.4, borderpad=0.3, labelspacing=0.25,
                       ncol=1)
ax_d1.add_artist(leg_roi)

style_handles = [
    Line2D([0], [0], color='black', lw=1.4, linestyle='-', label='ensemble mean'),
    Line2D([0], [0], color='black', lw=1.0, linestyle='--', label='planned'),
    Patch(facecolor='gray', alpha=0.30, edgecolor='none', label='95 % band'),
]
ax_d2.legend(handles=style_handles, loc='center',
             bbox_to_anchor=(0.50, 0.50),
             fontsize=6.0, frameon=True, framealpha=0.92,
             handlelength=1.6, borderpad=0.3, labelspacing=0.25)

# Row letters (a) (b) (c) (d) in left column, plain weight, double font,
# positioned in the title strip of each row (slightly above the axis top edge)
# and pushed all the way left in the figure margin.
fig.canvas.draw()   # ensure positions are committed before reading bboxes

LETTER_X       = 0.005     # x in figure coords (figure left edge)
LETTER_Y_OFF   = 0.018     # gap above axis top, ~ subplot-title baseline
LETTER_FS      = 17
for letter, ax in (('(a)', ax_a1), ('(b)', ax_b1), ('(c)', ax_c1), ('(d)', ax_d1)):
    bb = ax.get_position()
    y_title = bb.y0 + bb.height + LETTER_Y_OFF
    fig.text(LETTER_X, y_title, letter,
             fontsize=LETTER_FS, ha='left', va='center')

def _save(path, **kw):
    """BL: Windows -- the target may be open (locked) in a viewer: write a temp file, replace; if locked, keep a timestamped copy."""
    root, ext = os.path.splitext(path); tmp = root + '_tmp' + ext
    fig.savefig(tmp, bbox_inches='tight', **kw)
    try:
        os.replace(tmp, path); return path
    except OSError:
        import time as _t
        alt = root + _t.strftime('_%H%M%S') + ext; os.replace(tmp, alt)
        print(f'  [locked] {os.path.basename(path)} is open elsewhere: saved as {os.path.basename(alt)}'); return alt


print(f'\nSaved preview: {_save(OUT_PREVIEW, dpi=200)}')

# Submission-ready output: TIFF LZW at 800 dpi (PMB max recommended) + PNG 400 dpi
# Override via env var: a regeneration must NOT write into the submission
# folder (the material already shipped is frozen).
SUBMISSION = os.environ.get('FIG3_SUBMISSION_DIR', OUT_DIR)                                         # BL: never the submission folder
os.makedirs(SUBMISSION, exist_ok=True)
print(f"Saved: {_save(os.path.join(SUBMISSION, f'{BASENAME}.png'), dpi=400)} (400 dpi)")
_tif = _save(os.path.join(SUBMISSION, f'{BASENAME}.tiff'), dpi=800, format='tiff', pil_kwargs={'compression': 'tiff_lzw'})
print(f"Saved: {_tif} (800 dpi LZW, {os.path.getsize(_tif) / 1024 / 1024:.1f} MB)")
