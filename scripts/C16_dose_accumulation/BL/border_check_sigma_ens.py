"""Why σ_ens/G < 1 without reduced samples: the kernel is L2-normalised but the convolution uses zero padding
at the VOLUME borders (core/grf.py conv3d_separable), so Var(K eps)(x) = Π_d v_d(i_d) with
v_d(i) = Σ_j k_d(i-j)^2 over the interior j only. Exact computation (no sampling) on the 10 DIR-LAB cases, averaged over the body,
compared with the σ_ens/G measured by C1 (q21_sampler.csv, elastix, aniso_fixed = stationary kernel at the lengths APPLIED per case
from q21_lengths.csv, i.e. 15/10/10 mm × mean factor of Eq. 5 ≈ 19/13/13 mm). Result: mean theory 0.976, measured 0.974, |diff| ≤ 0.007 per case."""
from dirunc_paths import DATASETS  # data roots, see README
import os, sys, csv, math, numpy as np, nibabel as nib
from scipy import ndimage as ndi
DATA = rf"{DATASETS}/Castillo-DIR-LAB"; L0_ZYX = (15.0, 10.0, 10.0); KS = 61
sys.path.insert(0, os.path.abspath(".."))
def k1d(sig):
    r = KS // 2; x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / max(sig, 1e-3)) ** 2); return k / np.sqrt((k ** 2).sum())
def edge_factor(n, sig):
    """v(i) = Σ_{j=0}^{n-1} k(i-j)^2  (variance retained at index i of an axis of length n, zero padding)."""
    k2 = k1d(sig) ** 2; r = KS // 2; v = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - r), min(n - 1, i + r); v[i] = k2[(lo - i) + r:(hi - i) + r + 1].sum()
    return v
def body_mask(ct):
    m = ct > -500.0; lbl, n = ndi.label(m)
    if n > 1: s = np.bincount(lbl.ravel()); s[0] = 0; m = lbl == int(np.argmax(s))
    for z in range(m.shape[0]): m[z] = ndi.binary_fill_holes(m[z])
    return m
meas = {}
for r in csv.DictReader(open("_out_q21_BLv2/q21_sampler.csv")):
    if r["kernel"] == "aniso_fixed" and r["engine"] == "elastix": meas.setdefault(int(r["case"]), []).append(float(r["sigma_ens_over_G_body"]))
APPLIED = {int(r["case"]): tuple(float(r[k]) for k in ("applied_Z_mm", "applied_Y_mm", "applied_X_mm"))
           for r in csv.DictReader(open("_out_q21_BLv2/q21_lengths.csv")) if r["kernel"] == "aniso"}
rows = []; profiles = {}
for case in range(1, 11):
    nii = nib.load(os.path.join(DATA, "nifti", f"case{case}_image1.nii.gz"))
    ct = np.asarray(nii.get_fdata(), dtype=np.float32).transpose(2, 1, 0); sx, sy, sz = [float(z) for z in nii.header.get_zooms()[:3]]
    sp = (sz, sy, sx); Z, Y, X = ct.shape; m = body_mask(ct)
    sig = [l / s for l, s in zip(APPLIED[case], sp)]
    vz, vy, vx = edge_factor(Z, sig[0]), edge_factor(Y, sig[1]), edge_factor(X, sig[2])
    var = vz[:, None, None] * vy[None, :, None] * vx[None, None, :]
    ratio = np.sqrt(var)[m].mean()
    only_z = np.sqrt(vz[:, None, None] * np.ones((1, Y, X)))[m].mean()
    # profile along Z: body mean per slice
    prof = np.array([np.sqrt(var[z])[m[z]].mean() if m[z].any() else np.nan for z in range(Z)])
    nbody = m.sum(axis=(1, 2))
    profiles[case] = (prof, nbody, sz)
    touch = dict(z=int(m[0].sum() + m[-1].sum() > 0), y=int(m[:, 0, :].sum() + m[:, -1, :].sum() > 0), x=int(m[:, :, 0].sum() + m[:, :, -1].sum() > 0))
    rows.append((case, Z, Y, X, sz, sy, sx, ratio, only_z, np.mean(meas.get(case, [np.nan])), touch))
    print(f"case {case:2d} shape {Z}x{Y}x{X} sp {sz:.2f}/{sy:.2f}/{sx:.2f} body touches z/y/x {touch['z']}/{touch['y']}/{touch['x']}  "
          f"theory sigma_ens/G {ratio:.3f} (Z only {only_z:.3f})  measured C1 {np.mean(meas.get(case,[np.nan])):.3f}", flush=True)
th = np.mean([r[7] for r in rows]); me = np.nanmean([r[9] for r in rows]); print(f"MEAN theory {th:.3f}  measured {me:.3f}")
with open("_out_q21_BLv2/border_check/border_check.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["case", "Z", "Y", "X", "sz", "sy", "sx", "theory_ratio", "theory_ratio_Zonly", "measured_C1", "touch_z", "touch_y", "touch_x"])
    for r in rows: w.writerow(list(r[:10]) + [r[10]["z"], r[10]["y"], r[10]["x"]])
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
n = 200; d = np.arange(n) * 2.5
ax[0].plot(d, np.sqrt(edge_factor(2 * n, 19.3 / 2.5)[:n]), "k-", label="applied ℓ = 19.3 mm (Z)"); ax[0].plot(np.arange(n) * 0.97, np.sqrt(edge_factor(2 * n, 12.9 / 0.97)[:n]), "k--", label="applied ℓ = 12.9 mm (X, Y)")
ax[0].set_xlim(0, 80); ax[0].set_ylim(0.6, 1.02); ax[0].set_xlabel("distance from the volume border (mm)"); ax[0].set_ylabel("σ_ens / G (one axis)")
ax[0].axhline(1, color="0.7", lw=0.8); ax[0].legend(); ax[0].set_title("(a) variance retained near a zero-padded border")
prof, nbody, sz = profiles[1]; z = np.arange(len(prof)) * sz
ax[1].plot(z, prof, "b-", label="theory, body mean per slice"); ax[1].set_ylim(0.6, 1.02); ax[1].set_xlabel("z (mm), case 1"); ax[1].set_ylabel("σ_ens / G")
ax2 = ax[1].twinx(); ax2.fill_between(z, nbody, color="0.85", label="body voxels per slice"); ax2.set_ylabel("body voxels per slice"); ax2.set_zorder(-1); ax[1].set_facecolor("none")
ax[1].axhline(rows[0][7], color="b", ls=":", label=f"body mean {rows[0][7]:.3f}"); ax[1].axhline(rows[0][9], color="r", ls="--", label=f"measured C1 {rows[0][9]:.3f}")
ax[1].legend(loc="lower center", fontsize=8); ax[1].set_title("(b) DIR-LAB case 1 along the cranio-caudal axis")
fig.tight_layout(); fig.savefig("_out_q21_BLv2/border_check/border_check.png", dpi=130); print("PNG ok")
