"""Build Figure 3 v15 — 4 rows with post-fix Smolders data + restored μ DVF row.

Row (a)  axial   σ-maps            : Heuristic σ_h | Smolders σ_DL,  0–5 mm
Row (b)  axial   μ DVF (Option B)  : Heuristic ⟨μ⟩ | Original μ | Smolders ⟨μ⟩
                                     heatmap turbo + sparse white quiver
Row (c)  sagittal dose             : Heuristic-FC | Plan | Smolders-FC,  0–55 Gy
Row (d)  DVH                       : Heuristic-FC | Smolders-FC, 6 ROIs
         dual legend matching v13 (ROI in left panel, line-style in right panel)
"""
from dirunc_paths import LIB, LUNG_SBRT, RESULTS  # data roots, see README
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
RES = os.environ.get('FIG3_RES', rf'{LUNG_SBRT}/Results100')
OUT_PREVIEW = os.environ.get(
    'FIG3_OUT',
    RESULTS + '/_preview_Figure3_v15.png')
# Same figure on the in-house registration chain: the deterministic field of the
# central panel of (b) is then no longer the Velocity one, and the output files
# must not overwrite the existing Figure3 in the same folder.
MU_LABEL = os.environ.get('FIG3_MU_LABEL', 'Velocity')
BASENAME = os.environ.get('FIG3_BASENAME', 'Figure3')
# Names of the two arm subfolders: used to compare two f calibrations
# (e.g. Heuristic_f211) without junctions or copies on the filesystem.
ARM_H = os.environ.get('FIG3_ARM_H', 'Heuristic')
ARM_S = os.environ.get('FIG3_ARM_S', 'Smolders')
# Label of f in the row (a) title: changes with the calibration used.
F_LABEL = os.environ.get('FIG3_F_LABEL', '3.86')

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
body = (ct > -500)

print('Resampling plan dose ...')
dose_plan = resample_plan_dose(RD_PLAN, ct, v2w_post, inv_predef)

# ============================================================================
# 2) σ-maps + dose + DVF + DVH cache
# ============================================================================
print('Loading σ-maps + accumulated FC dose ...')
sigma_heu = np.load(f'{RES}/{ARM_H}/frac_01/uncertainty_dir.npy')
sigma_smo = np.load(f'{RES}/{ARM_S}/frac_01/uncertainty_dir.npy')
dose_heu  = np.load(f'{RES}/{ARM_H}/dose_accumulated_fc.npy')
dose_smo  = np.load(f'{RES}/{ARM_S}/dose_accumulated_fc.npy')

print('Loading μ DVFs ...')
def load_dvf(path):
    m = np.load(path)
    if m.shape[1:] == (nx, ny, nz):
        m = np.transpose(m, (0, 3, 2, 1))   # (3, X, Y, Z) -> (3, Z, Y, X)
    return m
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
slc_body  = body[zc]
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
zs_body = np.where(body[:, :, sag_x].any(axis=1))[0]
zs_lo = max(0, int(zs_body.min()) - 2); zs_hi = min(nz - 1, int(zs_body.max()) + 2)
ys_sag = np.where(body[:, :, sag_x].any(axis=0))[0]
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
SIGMA_VMAX = 4.0   # reduced from 5.0; σ_DL p99=2.71, σ_h max=3.69
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
plt.colorbar(im_sig, cax=cax_a, label='σ (mm)', ticks=[0, 1, 2, 3, 4])

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
FACTOR_BULK  = 2.0          # for Velocity quiver in centre panel
FACTOR_DELTA = 15.0         # for perturbation quiver in left/right panels
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
def _dose_sag(arr3d):
    s = arr3d[zs_lo:zs_hi, ys_lo:ys_hi, sag_x].astype(np.float32)
    s = np.where(s >= DOSE_VMIN, s, np.nan)
    return s[::-1, :]

for ax, title, d_arr in [
    (ax_c1, 'Heuristic-FC accumulated', _dose_sag(dose_heu)),
    (ax_c2, 'Original planned dose',     _dose_sag(dose_plan)),
    (ax_c3, 'Smolders-FC accumulated',   _dose_sag(dose_smo))]:
    ax.imshow(ct_sag, cmap='gray', vmin=CT_VMIN, vmax=CT_VMAX,
              aspect=SAG_ASPECT, interpolation='bilinear')
    im_dose = ax.imshow(d_arr, cmap='jet', vmin=DOSE_VMIN, vmax=DOSE_VMAX,
                        alpha=0.55, aspect=SAG_ASPECT, interpolation='bilinear')
    if ptv_sag.any():
        ax.contour(ptv_sag, levels=[0.5], colors='yellow', linewidths=1.1,
                   linestyles='dashed')
    ax.set_title(title, fontsize=8.5)
    ax.axis('off')
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

fig.savefig(OUT_PREVIEW, dpi=200, bbox_inches='tight')
print(f'\nSaved preview: {OUT_PREVIEW}')

# Submission-ready output: TIFF LZW at 800 dpi (PMB max recommended) + PNG 400 dpi
# Override via env var: a regeneration must NOT write into the submission
# folder (the material already shipped is frozen).
SUBMISSION = os.environ.get(
    'FIG3_SUBMISSION_DIR',
    RESULTS + '/Figures')
os.makedirs(SUBMISSION, exist_ok=True)
import shutil
if BASENAME == 'Figure3':          # the backup applies only to the canonical name
    for ext in ('png', 'tiff'):
        src = os.path.join(SUBMISSION, f'Figure3.{ext}')
        if os.path.isfile(src):
            bak = os.path.join(SUBMISSION, f'Figure3_v13_backup.{ext}')
            if not os.path.isfile(bak):
                shutil.copy2(src, bak)
                print(f'Backed up: {src} -> {bak}')
fig.savefig(os.path.join(SUBMISSION, f'{BASENAME}.png'),
            dpi=400, bbox_inches='tight')
print(f"Saved: {os.path.join(SUBMISSION, f'{BASENAME}.png')} (400 dpi)")
fig.savefig(os.path.join(SUBMISSION, f'{BASENAME}.tiff'),
            dpi=800, format='tiff',
            pil_kwargs={'compression': 'tiff_lzw'},
            bbox_inches='tight')
sz_mb = os.path.getsize(os.path.join(SUBMISSION, f'{BASENAME}.tiff'))/1024/1024
print(f"Saved: {os.path.join(SUBMISSION, f'{BASENAME}.tiff')} (800 dpi LZW, {sz_mb:.1f} MB)")
