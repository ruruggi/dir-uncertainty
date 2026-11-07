# -*- coding: utf-8 -*-
"""IV round: canonical script for the number added to Section 3.2 on 2026-10-22: «For the isotropic kernel
truncated in this way the expected ratio is 1.06» (R2.P14L21: «a Gaussian kernel truncated in this way gives an expected ratio of 1.06,
against the 1.07 measured»). Expected lambda_z/lambda_x per DIR-LAB case from the exact ACF of the 61-voxel truncated, L2-normalised
discrete Gaussian (same trunc_acf and 1/e interpolation as q32_figure2_trunc.py), with the case spacings of q21_lengths.csv; mean over
the ten cases, for the isotropic comparator (l = 14.9 mm) and for K (19.5 / 13 mm). Also prints the measured ratios (samples) of the same
csv. Values obtained (reply to reviewer item P14L21): theory 1.058 (iso) / 1.537 (K) vs measured 1.066 / 1.554.
Usage: python q33_trunc_ratio.py"""
import os, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); IVL = os.path.dirname(HERE)
KERNEL_SIZE = 61


def trunc_acf(ell_mm, sp_mm, grid):
    """as in q32_figure2_trunc.py"""
    s = ell_mm / sp_mm; h = KERNEL_SIZE // 2; x = np.arange(-h, h + 1)
    k = np.exp(-x ** 2 / (2 * s ** 2)); k /= np.sqrt((k ** 2).sum())
    ac = np.correlate(k, k, "full")[KERNEL_SIZE - 1:]; ac /= ac[0]
    return np.interp(grid, np.arange(len(ac)) * sp_mm, ac, right=0.0)


def l1e(grid, ac):
    t = 1 / np.e; j = int(np.argmax(ac < t)); return grid[j - 1] + (ac[j - 1] - t) / (ac[j - 1] - ac[j]) * (grid[j] - grid[j - 1])


L = list(csv.DictReader(open(os.path.join(IVL, "results", "q25_const_lengths", "q21_lengths.csv"), encoding="utf-8")))
SP = {r["case"]: [float(v) for v in r["spacing_zyx"].split("/")] for r in L}
assert len(SP) == 10
grid = np.arange(0.0, 200.0, 0.01)
L_ISO = (19.5 * 13 * 13) ** (1 / 3)
for name, lz, lx, meas_key in (("iso", L_ISO, L_ISO, "iso_const"), ("K", 19.5, 13.0, "aniso_const")):
    th = [l1e(grid, trunc_acf(lz, sp[0], grid)) / l1e(grid, trunc_acf(lx, sp[2], grid)) for sp in SP.values()]
    ms = [float(r["ratio_Z_over_X"]) for r in L if r["kernel"] == meas_key]
    print(f"{name:3s} l = {lz:.1f}/{lx:.1f} mm: expected lambda_z/lambda_x (truncated) mean {np.mean(th):.3f} "
          f"[{min(th):.3f}, {max(th):.3f}]; measured mean {np.mean(ms):.3f}")
