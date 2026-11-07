"""P5L56 (2026-10-03): how much CT noise moves the suppression factor M of Eq. (4).
M = min-max over the whole volume of |grad I_m| (central differences, no smoothing; runner.py gradient_magnitude_central).
Per DIR-LAB volume (10 cases x 2 phases, HU as in load_pair_nifti):
  sigma_n   CT noise, robust: 1.4826*MAD of in-plane adjacent-voxel differences in soft tissue (-200..200 HU, body), / sqrt(2)
  M_noise   predicted noise floor of M: E|grad| of pure noise with central differences = sigma_n/sqrt(2) * E[chi_3] (=1.596) / range
  M_soft    observed median of M in soft tissue (noise + real small gradients) ; M_lung observed median in lung (-950..-500 HU, body)
  M_edge    99th percentile of M in the body (real edges), for contrast
CPU, numpy only. Output: _out_p5l56/gradient_noise.csv + summary on stdout."""
from dirunc_paths import DATASETS  # data roots, see README
import os, csv, numpy as np, nibabel as nib
from scipy import ndimage as ndi
DATA = rf"{DATASETS}/Castillo-DIR-LAB/nifti"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out_p5l56"); os.makedirs(OUT, exist_ok=True)
ECHI3 = 2.0 * np.sqrt(2.0 / np.pi)  # E of chi with 3 dof = 1.5958


def body_mask(ct):  # same as n11_gradient_impact.py / manuscript step (iii)
    m = ct > -500.0; lbl, n = ndi.label(m)
    if n > 1: s = np.bincount(lbl.ravel()); s[0] = 0; m = lbl == int(np.argmax(s))
    for z in range(m.shape[0]): m[z] = ndi.binary_fill_holes(m[z])
    return m


def grad_central(img):  # runner.py gradient_magnitude_central, numpy port (zero at the volume faces)
    g = []
    for ax in range(3):
        d = np.zeros_like(img)
        sl_c = [slice(None)] * 3; sl_p = [slice(None)] * 3; sl_m = [slice(None)] * 3
        sl_c[ax] = slice(1, -1); sl_p[ax] = slice(2, None); sl_m[ax] = slice(None, -2)
        d[tuple(sl_c)] = (img[tuple(sl_p)] - img[tuple(sl_m)]) / 2.0
        g.append(d)
    return np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2 + 1e-8)


rows = []
for case in range(1, 11):
    for ph in (1, 2):
        ct = np.asarray(nib.load(os.path.join(DATA, f"case{case}_image{ph}.nii.gz")).get_fdata(), dtype=np.float32)
        if ct.min() >= -200.0: ct = ct - 1024.0  # loader's HU-offset rule
        body = body_mask(ct)
        g = grad_central(ct); rng = float(g.max() - g.min()); M = (g - g.min()) / rng
        soft = body & (ct > -200) & (ct < 200); lung = body & (ct > -950) & (ct < -500)
        # noise: adjacent differences along the two in-plane axes, both voxels in soft tissue
        dd = []
        for ax in (0, 1):
            a = np.take(ct, range(0, ct.shape[ax] - 1), axis=ax); b = np.take(ct, range(1, ct.shape[ax]), axis=ax)
            ma = np.take(soft, range(0, ct.shape[ax] - 1), axis=ax) & np.take(soft, range(1, ct.shape[ax]), axis=ax)
            dd.append((b - a)[ma])
        dd = np.concatenate(dd)
        sigma_n = float(1.4826 * np.median(np.abs(dd - np.median(dd))) / np.sqrt(2.0))
        m_noise = sigma_n / np.sqrt(2.0) * ECHI3 / rng
        r = dict(case=case, phase=ph, sigma_n_HU=sigma_n, grad_range_HU=rng, M_noise_pred=m_noise,
                 M_soft_median=float(np.median(M[soft])), M_lung_median=float(np.median(M[lung])),
                 M_body_p99=float(np.percentile(M[body], 99)), M_body_median=float(np.median(M[body])))
        rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)

with open(os.path.join(OUT, "gradient_noise.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
A = {k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ("case", "phase")}
print("\nSUMMARY over 20 volumes (min / median / max)")
for k, v in A.items(): print(f"  {k:15s} {v.min():9.4f} {np.median(v):9.4f} {v.max():9.4f}")
# self-check: predicted noise floor must not exceed the observed soft-tissue median by much (noise is part of it)
assert np.all(A["M_noise_pred"] <= 1.5 * A["M_soft_median"] + 1e-3), "noise model inconsistent with data"
print("check ok")
