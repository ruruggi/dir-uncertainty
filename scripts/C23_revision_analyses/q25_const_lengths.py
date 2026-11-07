# -*- coding: utf-8 -*-
"""IV round, Q2.5 option A2 check (2026-10-17): Figure 2 metrics (lambda, first-candidate folding, backstop) with ONE constant
anisotropic length triple for all ten DIR-LAB cases, instead of the per-case lengths applied by Eq. 11 (alpha = 0.5).
Runs the III script q21_kernel_metrics_BLv2.py UNCHANGED on disk (exec of its source with two string patches, assert count == 1):
  - OUT -> results/q25_const_lengths
  - two extra kernel arms: aniso_const = L_CONST (19.5/13/13 mm = 15/10/10 x 1.3), iso_const = geometric mean of L_CONST (14.88 mm).
Same white noise (seeded per case) as the III arms, so rows are paired with _out_q21_BLv2. Control: rerun aniso_fixed and compare.
Usage: python q25_const_lengths.py --gpu 0 --kernels aniso_fixed,aniso_const,iso_const [--cases 1,...]"""
from dirunc_paths import LIB, script  # data roots, see README
import os, sys
III = LIB  # was the III Scripts/_da_approvare folder; its scripts are found by name (dirunc_paths)
SRC = script("q21_kernel_metrics_BLv2.py")
OUT_IV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "q25_const_lengths")
L_CONST = (19.5, 13.0, 13.0)                                   # (Z, Y, X) mm
L_ISO_CONST = (L_CONST[0] * L_CONST[1] * L_CONST[2]) ** (1.0 / 3.0)

s = open(SRC, encoding="utf-8").read()


def patch(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old)
    s = s.replace(old, new)


patch('OUT = os.path.join(HERE, "_out_q21_BLv2")', "OUT = OUT_IV")
patch('"aniso_fixed": (lz / sp[0], ly / sp[1], lx / sp[2])}',
      '"aniso_fixed": (lz / sp[0], ly / sp[1], lx / sp[2]), '
      '"aniso_const": tuple(l / q for l, q in zip(L_CONST, sp)), "iso_const": tuple(L_ISO_CONST / q for q in sp)}')
g = {"__file__": SRC, "__name__": "__main__", "OUT_IV": OUT_IV, "L_CONST": L_CONST, "L_ISO_CONST": L_ISO_CONST}
print(f"[q25] L_CONST = {L_CONST} mm, iso_const = {L_ISO_CONST:.2f} mm, out = {OUT_IV}", flush=True)
exec(compile(s, SRC, "exec"), g)
