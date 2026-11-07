# -*- coding: utf-8 -*-
"""
q21_kernel_metrics_BLv2.py -- 2026-09-29 C1 (step 5 of ordine_ricalcoli_20260929): copy of q21_kernel_metrics_BL.py with G = f'*sigma_BL,
f' = f/sqrt3 from _out_q22_table23_v2 (R43). Output _out_q21_BLv2; the lengths/ACF rows (invariant to G) are copied from _out_q21_BL.
q21_kernel_metrics_BL.py -- STEP 7 (2026-09-28) version of q21_kernel_metrics.py: G = f_new * sigma_BL (body-local sigma,
_sigma_BL_20260927; f of the new Table 3); folding measured with the DVF IN VOXELS (N8: jacobian_determinant expects voxels;
the production sampler passes mm -> also reported as *_mm for comparison); ACF: unbiased body estimator (reported) + classical
body estimator (control only, user decision 2026-09-28); base folding of mu (App. A.3). Outputs in ./_out_q21_BL/.
Original description follows.

Reviewer-2 Q2.1, third submission: K_aniso vs K_iso on
the 10 DIR-LAB Castillo cases, with two isotropic comparators.

Kernels (same white noise within each case):
  aniso        manuscript K: l_d(x) = l0_d (1 + a tanh(|grad I_f| / 2 theta)), core.grf (tanh)
  iso_nominal  isotropic, spatially fixed, l = mean(l0) = 11.67 mm  (the submitted Figure 2)
  iso_matched  isotropic, spatially fixed, l = geometric mean of the per-direction lengths
               that K actually applies (4x4x4 block means, core volume weighted), per case
  aniso_fixed  anisotropic, spatially fixed (alpha = 0 arm), l_d = the same applied per-direction
               lengths: same mean length and anisotropy as aniso, without the anatomical modulation
  aniso_fine_sN  (only with --fine) exploratory: the same Eq. 5 modulation s(x) = 1 + a tanh(.)
               applied voxel-wise instead of per 4x4x4 block, after Gaussian smoothing of s at
               N mm (N = 0: none). Realised by a bank of stationary fields at levels s_j from the
               same eps, linearly interpolated in s and renormalised exactly to unit variance.
  --strat        also report first-candidate folding and sigma_ens/G per tercile of the Eq. 5 scale
               s(x) in the body (lo = homogeneous ... hi = strong gradients); rows go to q21_strat.csv.

Quantities:
  (a,b) realised correlation length lambda_d = 1/e width of the ensemble ACF of delta = K eps
  (c)   pre-screening folding rate: first candidate z = mu + G delta, G = f sigma_raw (Table 3 f)
  (d)   metric 1, amplitude transmitted by the diffeomorphism backstop of Sec. 2.4, replicated
        from core/sampling.py (tau0 = 1e-4, adaptive threshold, c = 0.85, 20 retries, collapse
        to mu, fresh eps per retry): retries, gain c^r, accepted at r = 0, collapses, and the
        body-mean of sigma_ens(x) / G(x), sigma_ens = RMS over samples and components of z - mu.

Body premise (user, 2026-09-27): outside the BODY, DVF and G are ignored by design; lung air belongs
to the body. Every quantity is therefore evaluated on the body mask only (HU > -500, largest
component, per-slice hole filling): theta = median |grad I_f| (central differences, as runner.py)
over the body; applied lengths weighted by body voxels; s(x) smoothed by normalised convolution
inside the body; ACF over voxel pairs both in the body; folding (first candidate, mu, backstop
test) = fraction of folded voxels in the body eroded by one voxel, so that no value outside the
body enters the finite differences of det J. G outside the body is left as stored: it influences
nothing. --whole restores the earlier whole-volume behaviour (outputs in ./_out_q21/).

Other conventions replicate the production sampler (mu in mm as stored, spatial axes permuted to
(Z,Y,X)). The submitted code under ../core is imported, never modified.

Usage:  python q21_kernel_metrics.py --smoke
        python q21_kernel_metrics.py --gpu 0 --cases 1,3,5,7,9
Outputs (resume-safe, rows already present are skipped) in ./_out_q21_body/ (--whole: ./_out_q21/).
"""
from dirunc_paths import DATASETS, LIB  # data roots, see README
import os, sys, csv, math, time, argparse
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = LIB                       # submitted code snapshot (tanh grf)
sys.path.insert(0, SCRIPTS)
import torch
import nibabel as nib
from scipy import ndimage as ndi
from core.grf import adaptive_ell_map, anisotropic_correlation, conv3d_separable, gaussian_kernel_1d
from core.jacobian import check_diffeomorphism, jacobian_determinant

DATA = rf"{DATASETS}/Castillo-DIR-LAB"
OUT = os.path.join(HERE, "_out_q21")

L0_ZYX = (15.0, 10.0, 10.0); ALPHA = 0.5; KERNEL_SIZE = 61
L_ISO_NOMINAL = float(np.mean(L0_ZYX))                # 11.67 mm, as in the submitted Figure 2
L_ISO_GEO = float(np.prod(L0_ZYX) ** (1.0 / 3.0))     # 11.45 mm: geometric mean of the NOMINAL lengths (equal volume, no Eq. 5)
import json as _json
_T3 = _json.load(open(os.path.join(HERE, "_out_q22_table23_v2", "table23_v2.json")))["results"]   # 2026-09-29 C1: R43, f' = f/sqrt3
F_TABLE3 = {k: _T3[f"T3|DIR-LAB|{lab}|sigma_h"]["f"] for k, lab in (("synra", "SyNRA"), ("convexadam", "ConvexAdam"),
            ("unigradicon", "uniGradICON"), ("elastix", "Elastix"))}          # Table 3 v2 (sigma_BL, f/sqrt3)
SIGMA_BL = rf"{DATASETS}/_sigma_BL_20260927/Castillo-DIR-LAB"
ENGINES = ["elastix", "convexadam", "unigradicon", "synra"]
TAU0, GAIN_DECAY, MAX_RETRIES = 1e-4, 0.85, 20        # core/sampling.py defaults
MAXLAG = 40

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--cases", default="1,2,3,4,5,6,7,8,9,10")
ap.add_argument("--engines", default=",".join(ENGINES))
ap.add_argument("--n", type=int, default=100)         # samples per ensemble (paper N = 100)
ap.add_argument("--m_acf", type=int, default=100)     # realisations for lambda (submitted Fig. 2b: 100)
ap.add_argument("--fine", default="")                 # e.g. "0,3,10": smoothing scales (mm) of the voxel-wise arms
ap.add_argument("--levels", type=int, default=8)      # bank size of the voxel-wise arms
ap.add_argument("--strat", action="store_true")       # per-tercile metrics, separate CSV
ap.add_argument("--kernels", default="")              # optional subset of kernel names, comma-separated
ap.add_argument("--whole", action="store_true")       # legacy: whole-volume theta / weights / ACF / folding
args = ap.parse_args()
CASES = [int(c) for c in args.cases.split(",")]
ENG = args.engines.split(",")
N, M_ACF = args.n, args.m_acf
OUT = os.path.join(HERE, "_out_q21_BLv2")                          # 2026-09-29 C1: G = f'*sigma_BL; lengths/ACF copied from _out_q21_BL (invariant to G)
if args.smoke:
    CASES, ENG, N, M_ACF = [5], ["elastix"], 4, 3
    OUT = OUT + "_smoke"
os.makedirs(OUT, exist_ok=True)
DEV = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(DEV)


def load_ct(case):
    nii = nib.load(os.path.join(DATA, "nifti", f"case{case}_image1.nii.gz"))
    arr = np.asarray(nii.get_fdata(), dtype=np.float32)                   # (X,Y,Z)
    sx, sy, sz = [float(z) for z in nii.header.get_zooms()[:3]]
    return np.ascontiguousarray(arr.transpose(2, 1, 0)), (sz, sy, sx)


def body_mask(ct):
    """HU > -500, largest component, per-slice hole filling: keeps the aerated lung."""
    m = ct > -500.0
    lbl, n = ndi.label(m)
    if n > 1:
        sizes = np.bincount(lbl.ravel()); sizes[0] = 0
        m = lbl == int(np.argmax(sizes))
    for z in range(m.shape[0]):
        m[z] = ndi.binary_fill_holes(m[z])
    return m


def applied_lengths_mm(ell_vox, spacing, body=None):
    """Per-direction lengths as K applies them: block means over the 4x4x4 tiling of
    core.grf.anisotropic_correlation (expanded block, margin = kernel//2), weighted by
    core-block volume. Returns (lz, ly, lx) in mm."""
    _, Z, Y, X = ell_vox.shape
    tz, ty, tx = (int(np.ceil(s / 4)) for s in (Z, Y, X)); mg = KERNEL_SIZE // 2
    acc = np.zeros(3); wsum = 0.0
    for iz in range(4):
        for iy in range(4):
            for ix in range(4):
                z0, z1 = iz * tz, min((iz + 1) * tz, Z)
                y0, y1 = iy * ty, min((iy + 1) * ty, Y)
                x0, x1 = ix * tx, min((ix + 1) * tx, X)
                if z1 <= z0 or y1 <= y0 or x1 <= x0:
                    continue
                blk = ell_vox[:, max(0, z0 - mg):min(Z, z1 + mg), max(0, y0 - mg):min(Y, y1 + mg),
                              max(0, x0 - mg):min(X, x1 + mg)]
                w = (z1 - z0) * (y1 - y0) * (x1 - x0) if body is None else int(body[z0:z1, y0:y1, x0:x1].sum())
                acc += w * np.array([float(blk[d].mean()) * spacing[d] for d in range(3)]); wsum += w
    return tuple(acc / wsum)


def body_fold_rate(dvf, keep):
    """Fraction of voxels with det J <= 0 among `keep` (body eroded by one voxel)."""
    return float(((jacobian_determinant(dvf) <= 0) & keep).sum()) / float(keep.sum())


def _l1e(ac, spacing_axis):
    thr = 1.0 / math.e
    if not np.any(ac < thr):
        return (len(ac) - 1) * spacing_axis
    j = int(np.argmax(ac < thr)); return ((j - 1) + (ac[j - 1] - thr) / (ac[j - 1] - ac[j])) * spacing_axis


def masked_acf_both(d, m, axis, spacing_axis):
    """Body ACF along `axis`: unbiased (divided by the number of body pairs at each lag; REPORTED) and classical (divided by
    the number of body voxels at every lag; CONTROL ONLY). Returns (l1e_unbiased, ac_unbiased, l1e_classical)."""
    mf = m.float(); n = m.shape[axis]; other = tuple(i for i in range(3) if i != axis)
    Fm = torch.fft.rfft(mf, n=2 * n, dim=axis)
    cnt = torch.fft.irfft((Fm * Fm.conj()).real, n=2 * n, dim=axis).sum(dim=other)
    cu, cc = [], []
    for c in range(3):
        v = (d[c] - d[c][m].mean()) * mf
        F = torch.fft.rfft(v, n=2 * n, dim=axis)
        S = torch.fft.irfft((F * F.conj()).real, n=2 * n, dim=axis).sum(dim=other)
        au = (S / cnt.clamp(min=1))[:min(n, MAXLAG + 1)]; ac_ = S[:min(n, MAXLAG + 1)]
        cu.append((au / au[0]).cpu().numpy()); cc.append((ac_ / ac_[0]).cpu().numpy())
    au = np.mean(cu, axis=0); acl = np.mean(cc, axis=0)
    return _l1e(au, spacing_axis), au, _l1e(acl, spacing_axis)


def masked_acf_1e_mm(d, m, axis, spacing_axis):
    """As acf_1e_mm, but over voxel pairs that are both in the body m (bool, same shape)."""
    mf = m.float(); n = m.shape[axis]; other = tuple(i for i in range(3) if i != axis)
    Fm = torch.fft.rfft(mf, n=2 * n, dim=axis)
    cnt = torch.fft.irfft((Fm * Fm.conj()).real, n=2 * n, dim=axis).sum(dim=other)
    curves = []
    for c in range(3):
        v = (d[c] - d[c][m].mean()) * mf
        F = torch.fft.rfft(v, n=2 * n, dim=axis)
        ac = torch.fft.irfft((F * F.conj()).real, n=2 * n, dim=axis).sum(dim=other) / cnt.clamp(min=1)
        ac = ac[:min(n, MAXLAG + 1)]
        curves.append((ac / ac[0]).cpu().numpy())
    ac = np.mean(curves, axis=0); thr = 1.0 / math.e
    if not np.any(ac < thr):
        return (len(ac) - 1) * spacing_axis, ac
    j = int(np.argmax(ac < thr)); lag = (j - 1) + (ac[j - 1] - thr) / (ac[j - 1] - ac[j])
    return lag * spacing_axis, ac


def acf_1e_mm(d, roi, axis, spacing_axis):
    """1/e width (mm) of the ACF of delta along `axis`, averaged over the 3 components."""
    curves = []
    for c in range(3):
        v = d[c][roi]; v = v - v.mean(); n = v.shape[axis]
        F = torch.fft.rfft(v, n=2 * n, dim=axis)
        ac = torch.fft.irfft((F * F.conj()).real, n=2 * n, dim=axis)
        idx = [slice(None)] * 3; idx[axis] = slice(0, min(n, MAXLAG + 1))
        ac = ac[tuple(idx)].mean(dim=tuple(i for i in range(3) if i != axis))
        curves.append((ac / ac[0]).cpu().numpy())
    ac = np.mean(curves, axis=0); thr = 1.0 / math.e
    if not np.any(ac < thr):
        return (len(ac) - 1) * spacing_axis, ac
    j = int(np.argmax(ac < thr)); lag = (j - 1) + (ac[j - 1] - thr) / (ac[j - 1] - ac[j])
    return lag * spacing_axis, ac


class FineK:
    """Voxel-wise l_d(x) = l0v_d s(x): bank of stationary fields F_j = K(l0v s_j) eps from the same
    eps; at each voxel F = ((1-w) F_j + w F_j+1) / sqrt((1-w)^2 + w^2 + 2w(1-w) rho_j), with rho_j the
    exact correlation of F_j and F_j+1 (product over axes of the dot products of the L2-unit 1-D
    kernels), so Var F = 1 away from the zero-padded border, as for the stationary kernels.
    Simplification: linear interpolation in s between neighbouring levels; ceiling = rho_j -> 1 is needed,
    i.e. enough levels (8 over s in [1, 1.5] gives rho_j > 0.996)."""

    def __init__(self, s, l0v, levels):
        lo, hi = float(s.min()), float(s.max())
        self.lv = np.linspace(lo, hi, levels) if hi - lo > 1e-6 else np.array([lo, lo + 1e-6])
        ds = self.lv[1] - self.lv[0]
        self.j = ((s - lo) / ds).floor().clamp(0, len(self.lv) - 2).long()
        self.w = ((s - lo) / ds - self.j).clamp(0, 1)
        self.sig = [tuple(l * v for l in l0v) for v in self.lv]
        self.rho = []
        for a, b in zip(self.sig[:-1], self.sig[1:]):
            self.rho.append(math.prod(float((gaussian_kernel_1d(p, KERNEL_SIZE, DEV) *
                                             gaussian_kernel_1d(q, KERNEL_SIZE, DEV)).sum()) for p, q in zip(a, b)))

    def __call__(self, eps):
        e = eps.unsqueeze(0); out = torch.zeros_like(eps)
        prev = conv3d_separable(e, self.sig[0], KERNEL_SIZE)[0]
        for j in range(len(self.lv) - 1):
            nxt = conv3d_separable(e, self.sig[j + 1], KERNEL_SIZE)[0]
            m = self.j == j
            if m.any():
                w = self.w[m]; nrm = torch.sqrt((1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * self.rho[j])
                out[:, m] = ((1 - w) * prev[:, m] + w * nxt[:, m]) / nrm
            prev = nxt
        return out


def corr(eps, ell):
    return ell(eps) if callable(ell) else anisotropic_correlation(eps, ell, kernel_size=KERNEL_SIZE)


def noise(seed, shape):
    g = torch.Generator(device=DEV); g.manual_seed(int(seed))
    return torch.randn((3,) + shape, generator=g, device=DEV)


def done_keys(path, keys):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {tuple(r[k] for k in keys) for r in csv.DictReader(f)}


def append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


P_LEN = os.path.join(OUT, "q21_lengths.csv")
P_SMP = os.path.join(OUT, "q21_strat.csv" if args.strat else "q21_sampler.csv")
t_start = time.time()
for case in CASES:
    ct_np, sp = load_ct(case); Z, Y, X = ct_np.shape
    mask = body_mask(ct_np); mask_t = torch.from_numpy(mask).to(DEV)
    ct = torch.from_numpy(ct_np).to(DEV)
    gz = torch.diff(ct, dim=0, append=ct[-1:]); gy = torch.diff(ct, dim=1, append=ct[:, -1:])
    gx = torch.diff(ct, dim=2, append=ct[:, :, -1:])
    if args.whole:
        theta = float(torch.sqrt(gz ** 2 + gy ** 2 + gx ** 2 + 1e-12).median())   # earlier: forward diff, whole volume
    else:
        theta = float(np.median(np.sqrt(sum(g ** 2 for g in np.gradient(ct_np)))[mask]))   # runner.py estimator, body
    del gz, gy, gx
    keep = mask_t if args.whole else torch.from_numpy(ndi.binary_erosion(mask, border_value=1)).to(DEV)
    ell_a = adaptive_ell_map(ct.unsqueeze(0), sp, l0=L0_ZYX, alpha=ALPHA, theta=theta)
    lz, ly, lx = applied_lengths_mm(ell_a.cpu().numpy(), sp, None if args.whole else mask)
    l_match = (lz * ly * lx) ** (1.0 / 3.0)
    l_grad = math.sqrt(3.0 / (1 / lz ** 2 + 1 / ly ** 2 + 1 / lx ** 2))        # equal mean |grad|^2, for reference
    kernels = {"aniso": ell_a,
               "iso_nominal": tuple(L_ISO_NOMINAL / s for s in sp),
               "iso_geo": tuple(L_ISO_GEO / s for s in sp),
               "iso_matched": tuple(l_match / s for s in sp),
               "aniso_fixed": (lz / sp[0], ly / sp[1], lx / sp[2])}
    l0v = tuple(l / s for l, s in zip(L0_ZYX, sp))
    s_vox = (ell_a[0] / l0v[0]).cpu().numpy()                                  # Eq. 5 scale factor s(x)
    strata = {}
    if args.strat:                                                             # terciles of s(x) in the body
        q1, q2 = np.quantile(s_vox[mask], [1 / 3, 2 / 3]); s_t = torch.from_numpy(s_vox).to(DEV)
        strata = {"lo": mask_t & (s_t <= q1), "mid": mask_t & (s_t > q1) & (s_t <= q2), "hi": mask_t & (s_t > q2)}
        print(f"  strata: s terciles {q1:.3f} / {q2:.3f}, voxels " + " ".join(f"{n}={int(m.sum())}" for n, m in strata.items()), flush=True)
    for sm in [float(v) for v in args.fine.split(",") if v != ""]:
        sd = [sm / s for s in sp]
        if sm == 0:
            s_sm = s_vox
        elif args.whole:
            s_sm = ndi.gaussian_filter(s_vox, sd, mode="nearest")
        else:                                                                  # normalised convolution in the body
            mf = mask.astype(np.float32)
            s_sm = np.where(mask, ndi.gaussian_filter(s_vox * mf, sd, mode="nearest") /
                            np.maximum(ndi.gaussian_filter(mf, sd, mode="nearest"), 1e-6), s_vox)
        kernels[f"aniso_fine_s{sm:g}"] = FineK(torch.from_numpy(s_sm).float().to(DEV), l0v, args.levels)
        sb = float(s_sm[mask].mean())
        print(f"  aniso_fine_s{sm:g}: s in [{s_sm.min():.3f},{s_sm.max():.3f}], body-mean l "
              f"(Z,Y,X)=({L0_ZYX[0] * sb:.2f},{L0_ZYX[1] * sb:.2f},{L0_ZYX[2] * sb:.2f}) mm", flush=True)
    if args.kernels:
        kernels = {k: v for k, v in kernels.items() if k in args.kernels.split(",")}
    print(f"[case{case}] theta={theta:.1f} applied (Z,Y,X)=({lz:.2f},{ly:.2f},{lx:.2f}) mm  "
          f"iso_matched={l_match:.2f} mm (grad-matched {l_grad:.2f})", flush=True)

    # ---- (a,b): realised correlation lengths ----
    len_done = done_keys(P_LEN, ["case", "kernel"])
    todo = {k: ell for k, ell in kernels.items() if (str(case), k) not in len_done}
    if todo:
        zz, yy, xx = np.where(mask)
        roi = (slice(zz.min(), zz.max() + 1), slice(yy.min(), yy.max() + 1), slice(xx.min(), xx.max() + 1))
        m_roi = mask_t[roi]
        lam = {k: [[], [], []] for k in todo}; lamc = {k: [[], [], []] for k in todo}; acfz = {k: 0.0 for k in todo}
        for m in range(M_ACF):
            eps = noise(1000 + 1000 * case + m, (Z, Y, X))
            for k, ell in todo.items():
                d = corr(eps, ell)
                for ax in range(3):
                    l1e, ac, l1c = masked_acf_both(d[:, roi[0], roi[1], roi[2]], m_roi, ax, sp[ax])
                    lam[k][ax].append(l1e); lamc[k][ax].append(l1c)
                    if ax == 0:
                        acfz[k] = acfz[k] + ac / M_ACF
        for k in todo:
            np.save(os.path.join(OUT, f"acfZ_case{case}_{k}.npy"), acfz[k])
            append(P_LEN, {"case": case, "kernel": k,
                           "lambda_Z_mm": round(float(np.mean(lam[k][0])), 3),
                           "lambda_Y_mm": round(float(np.mean(lam[k][1])), 3),
                           "lambda_X_mm": round(float(np.mean(lam[k][2])), 3),
                           "ratio_Z_over_X": round(float(np.mean(lam[k][0]) / np.mean(lam[k][2])), 4),
                           "CONTROL_classical_lambda_Z_mm": round(float(np.mean(lamc[k][0])), 3),
                           "CONTROL_classical_lambda_X_mm": round(float(np.mean(lamc[k][2])), 3),
                           "CONTROL_classical_ratio_Z_over_X": round(float(np.mean(lamc[k][0]) / np.mean(lamc[k][2])), 4),
                           "applied_Z_mm": round(lz, 3), "applied_Y_mm": round(ly, 3), "applied_X_mm": round(lx, 3),
                           "iso_matched_mm": round(l_match, 3), "iso_gradmatched_mm": round(l_grad, 3),
                           "theta_HU": round(theta, 2), "spacing_zyx": "%.3f/%.3f/%.3f" % sp, "m_acf": M_ACF})
        print(f"  lengths done ({time.time() - t_start:.0f}s)", flush=True)

    # ---- (c,d): sampler with the backstop ----
    done = done_keys(P_SMP, ["case", "engine", "kernel"])
    for eng in ENG:
        cdir = os.path.join(DATA, f"Results_{eng}_procrustes", f"case{case}")
        mu = torch.from_numpy(np.ascontiguousarray(np.load(os.path.join(cdir, "dvf_mu.npy")).transpose(0, 3, 2, 1))).float().to(DEV)
        sg = torch.from_numpy(np.ascontiguousarray(np.load(os.path.join(SIGMA_BL, f"Results_{eng}_procrustes", f"case{case}", "sigma_raw.npy")).transpose(2, 1, 0))).float().to(DEV)
        VOX = torch.tensor(sp, device=DEV, dtype=torch.float32).view(3, 1, 1, 1)   # (sz, sy, sx) matches channels [dz, dy, dx]
        G = (F_TABLE3[eng] * sg).unsqueeze(0)                                  # (1,Z,Y,X), mm
        base_rate = body_fold_rate(mu / VOX, keep)                             # voxels, body (N8-corrected)
        base_rate_mm = body_fold_rate(mu, keep)                                # as the production sampler (mm)
        base_whole_vox = float((jacobian_determinant(mu / VOX) <= 0).float().mean())   # App. A.3 (whole volume, voxels)
        thr = max(TAU0, base_rate + 0.0005)                                    # core/sampling.py
        gpos = mask_t & (G[0] > 1e-6)
        for k, ell in kernels.items():
            if (str(case), eng, k) in done:
                continue
            t0 = time.time(); first_fold, first_fold_mm, retries, gains, collapsed = [], [], [], [], 0
            ff_st = {n: [] for n in strata}
            s2 = torch.zeros_like(mu)                                          # sum over samples of (z - mu)^2
            for s in range(N):
                gain, r = 1.0, 0
                while r < MAX_RETRIES:
                    eps = noise(10_000_000 + 100_000 * case + 100 * s + r, (Z, Y, X))   # same across kernels/engines
                    delta = G * gain * corr(eps, ell)
                    if args.whole:
                        acc, rate = check_diffeomorphism(mu + delta, threshold=thr)
                    else:
                        rate = body_fold_rate((mu + delta) / VOX, keep); acc = True if rate <= thr else None
                    if r == 0:
                        first_fold.append(rate); first_fold_mm.append(body_fold_rate(mu + delta, keep))
                        if strata:
                            det = jacobian_determinant((mu + delta) / VOX)
                            for n, m in strata.items():
                                mk = m & keep
                                ff_st[n].append(float((det[mk] <= 0).float().mean()))
                            del det
                    if acc is not None:
                        break
                    gain *= GAIN_DECAY; r += 1
                if acc is None:                                                # collapse to mu
                    collapsed += 1; gains.append(0.0)
                else:
                    s2 += delta ** 2; gains.append(gain)
                retries.append(r)
            sig_ens = torch.sqrt(s2.mean(dim=0) / N)                           # RMS over components and samples
            ratio = float((sig_ens[gpos] / G[0][gpos]).mean())
            g = np.array(gains)
            extra = {}
            for n, m in strata.items():
                extra[f"first_folding_{n}"] = float(np.mean(ff_st[n]))
                extra[f"sigma_ens_over_G_{n}"] = float((sig_ens[m & gpos] / G[0][m & gpos]).mean())
            append(P_SMP, {"case": case, "engine": eng, "kernel": k, "f": F_TABLE3[eng], "n": N,
                           "threshold": thr, "first_folding_mean": float(np.mean(first_fold)),
                           "first_folding_mm_production": float(np.mean(first_fold_mm)), "base_rate_mu_body_vox": base_rate,
                           "base_rate_mu_body_mm": base_rate_mm, "base_rate_mu_whole_vox": base_whole_vox,
                           "mean_retries": float(np.mean(retries)), "frac_r0": float(np.mean(np.array(retries) == 0)),
                           "frac_collapsed": collapsed / N, "gain_mean": float(g.mean()),
                           "gain_rms": float(np.sqrt(np.mean(g ** 2))), "sigma_ens_over_G_body": ratio,
                           "G_body_mean_mm": float(G[0][mask_t].mean()), "sec": round(time.time() - t0, 1), **extra})
            print(f"  {eng:11s} {k:11s} fold0={np.mean(first_fold):.4f} retries={np.mean(retries):.2f} "
                  f"r0={np.mean(np.array(retries) == 0):.2f} coll={collapsed}/{N} gain_rms={np.sqrt(np.mean(g ** 2)):.3f} "
                  f"sig_ens/G={ratio:.3f} ({time.time() - t0:.0f}s)", flush=True)
            # self-check: K is L2-normalised, so sigma_ens/G must track gain_rms (boundary padding aside)
            grms = float(np.sqrt(np.mean(g ** 2)))
            if g.any() and abs(ratio - grms) > 0.15 * max(grms, 0.05):
                print(f"  WARNING sigma_ens/G={ratio:.3f} does not track gain_rms={grms:.3f}", flush=True)
        del mu, sg, G                                     # s2 may not exist if every kernel was already done
        torch.cuda.empty_cache()
print(f"[done] {time.time() - t_start:.0f}s")
