# -*- coding: utf-8 -*-
"""IV round, internal critical review 1.1 (2026-10-24): Figure 3(a,b) and Sec. 3.2 sampled SyNRA with f = 3.8216 (III Table 3
fit on all 10 DIR-LAB cases, F_TABLE3 of q21_kernel_metrics_BLv2.py) instead of the f of Table 2 (2.9764, cases retained at q <= 10 mm).
Same as q25_const_lengths.py (III script q21_kernel_metrics_BLv2.py UNCHANGED on disk, exec of its source with string patches,
assert count == 1), plus ONE patch: F_TABLE3["synra"] = F_SYNRA (value read from results/q31_lece_perdir_all.json, Table 2 source).
Output in a NEW folder results/q25_const_lengths_synra298 (the old CSV stays as it is). The III script appends rows and skips the keys
already in its CSV, so an interrupted run resumes by launching the same command again.
Usage: python q25_const_lengths_synra298.py --gpu 0 --engines synra --kernels aniso_const,iso_const"""
from dirunc_paths import LIB, script  # data roots, see README
import os, sys, json
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
III = LIB  # was the III Scripts/_da_approvare folder; its scripts are found by name (dirunc_paths)
SRC = script("q21_kernel_metrics_BLv2.py")
OUT_IV = os.path.join(HERE, "results", "q25_const_lengths_synra298")
F_SYNRA = json.load(open(os.path.join(HERE, "results", "q31_lece_perdir_all.json")))["cells"]["DIR-LAB|SyNRA"]["norm"]["f"]
assert abs(F_SYNRA - 2.98) < 0.005, F_SYNRA                     # Table 2: 2.98
L_CONST = (19.5, 13.0, 13.0)                                   # (Z, Y, X) mm, as q25_const_lengths.py
L_ISO_CONST = (L_CONST[0] * L_CONST[1] * L_CONST[2]) ** (1.0 / 3.0)
os.makedirs(OUT_IV, exist_ok=True)

s = open(SRC, encoding="utf-8").read()


def patch(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old)
    s = s.replace(old, new)


patch('OUT = os.path.join(HERE, "_out_q21_BLv2")', "OUT = OUT_IV")
patch('"aniso_fixed": (lz / sp[0], ly / sp[1], lx / sp[2])}',
      '"aniso_fixed": (lz / sp[0], ly / sp[1], lx / sp[2]), '
      '"aniso_const": tuple(l / q for l, q in zip(L_CONST, sp)), "iso_const": tuple(L_ISO_CONST / q for q in sp)}')
patch('ENGINES = ["elastix", "convexadam", "unigradicon", "synra"]',
      'F_TABLE3["synra"] = F_SYNRA\nENGINES = ["elastix", "convexadam", "unigradicon", "synra"]')
g = {"__file__": SRC, "__name__": "__main__", "OUT_IV": OUT_IV, "L_CONST": L_CONST, "L_ISO_CONST": L_ISO_CONST, "F_SYNRA": F_SYNRA}
print(f"[q25-synra298] f_SyNRA = {F_SYNRA:.4f}, L_CONST = {L_CONST} mm, iso_const = {L_ISO_CONST:.2f} mm, out = {OUT_IV}", flush=True)
exec(compile(s, SRC, "exec"), g)
print("[q25-synra298] DONE", flush=True)
