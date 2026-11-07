# -*- coding: utf-8 -*-
"""IV round, R2.P14L21 (approved 2026-10-18): Figure 2(d) with the REALISED in-plane ACF of the kernels, i.e. the exact ACF of the
discrete L2-normalised Gaussian truncated to 61 voxels (core.grf.gaussian_kernel_1d, KERNEL_SIZE = 61), averaged over the in-plane
spacings of the ten DIR-LAB cases, instead of the untruncated analytic exp(-h^2/(4 l^2)). Runs q25_figure2_A2.py UNCHANGED on disk with
string patches (assert count == 1); output to results/q32_figure2/ (draft, NOT Figures/).
  --mode repro : only the output path is patched -> must equal the published Figures/Figure3.png pixel by pixel (control);
  --mode trunc : also the two in-plane model curves of panel (d) (Smolders' reference curve stays nominal).
Usage: python q32_figure2_trunc.py --mode repro|trunc"""
import os, sys, csv, argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["repro", "trunc"], required=True); args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); IVL = os.path.dirname(HERE)
SRC = os.path.join(HERE, "q25_figure2_A2.py")
OUTD = os.path.join(IVL, "results", "q32_figure2"); os.makedirs(OUTD, exist_ok=True)
KERNEL_SIZE = 61
src = open(SRC, encoding="utf-8").read()


def patch(old, new):
    global src
    assert src.count(old) == 1, (src.count(old), old)
    src = src.replace(old, new)


def trunc_acf(ell_mm, sp_mm, grid):
    """Exact ACF (normalised) of K eps along one axis for the 61-voxel truncated, L2-normalised discrete Gaussian, on grid (mm)."""
    s = ell_mm / sp_mm; h = KERNEL_SIZE // 2; x = np.arange(-h, h + 1)
    k = np.exp(-x ** 2 / (2 * s ** 2)); k /= np.sqrt((k ** 2).sum())
    ac = np.correlate(k, k, "full")[KERNEL_SIZE - 1:]; ac /= ac[0]
    return np.interp(grid, np.arange(len(ac)) * sp_mm, ac, right=0.0)


def l1e(grid, ac):
    t = 1 / np.e; j = int(np.argmax(ac < t)); return grid[j - 1] + (ac[j - 1] - t) / (ac[j - 1] - ac[j]) * (grid[j] - grid[j - 1])


L = list(csv.DictReader(open(os.path.join(IVL, "results", "q25_const_lengths", "q21_lengths.csv"), encoding="utf-8")))
SPX = {}
for r in L:
    SPX[r["case"]] = float(r["spacing_zyx"].split("/")[2])
assert len(SPX) == 10
MEAS = {k: np.mean([(float(r["lambda_X_mm"]) + float(r["lambda_Y_mm"])) / 2 for r in L if r["kernel"] == k])
        for k in ("aniso_const", "iso_const")}

patch('base = os.path.join(FIGDIR, "Figure3")', 'base = os.path.join(OUTD, "Figure2_%s" % MODE)')
if args.mode == "trunc":
    patch('d.plot(grid, theo(lX), color=COL["aniso_const"], lw=1.8); d.plot(grid, theo(liso), color=COL["iso_const"], lw=2.2, ls="--")',
          'd.plot(grid, TRUNC(lX), color=COL["aniso_const"], lw=1.8); d.plot(grid, TRUNC(liso), color=COL["iso_const"], lw=2.2, ls="--")')
g = {"__file__": SRC, "__name__": "q32", "OUTD": OUTD, "MODE": args.mode,
     "TRUNC": lambda l: np.mean([trunc_acf(l, sp, np.linspace(0, 100, 201)) for sp in SPX.values()], axis=0)}
exec(compile(src, SRC, "exec"), g)

grid = g["grid"]
for k, l in (("aniso_const", 13.0), ("iso_const", (19.5 * 13 * 13) ** (1 / 3))):
    t = l1e(grid, g["TRUNC"](l))
    print("[check] %s in-plane 1/e: truncated model %.2f mm, measured (samples) %.2f mm, nominal %.2f mm" % (k, t, MEAS[k], 2 * l))
    assert abs(t - MEAS[k]) < 0.3, (k, t, MEAS[k])
if args.mode == "repro":
    from PIL import Image
    a = np.asarray(Image.open(os.path.join(OUTD, "Figure2_repro.png")).convert("RGB")).astype(int)
    b = np.asarray(Image.open(os.path.join(os.path.dirname(IVL), "Figures", "Figure3.png")).convert("RGB")).astype(int)
    nd = int((a != b).any(axis=2).sum()) if a.shape == b.shape else -1
    print("[control] repro vs published Figure3.png: shape %s vs %s, differing pixels %d" % (a.shape, b.shape, nd))
