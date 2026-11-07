# -*- coding: utf-8 -*-
"""IV round, Q2.5 option A2 check (2026-10-17): Figure 3 (Lung SBRT 54 Gy / 3 fx) re-sampled with a CONSTANT anisotropic kernel,
alpha = 0 and l0 = 19.5/13/13 mm (= 15/10/10 x 1.3), instead of alpha = 0.5 on l0 = 15/10/10 (applied 18.6/12.4/12.4 mm here).
Runs the III driver fig3_BL_runner.py UNCHANGED on disk (exec of its source, one string patch, assert count == 1); everything else
(f = 2.23, N = 100, FC shared noise seed 12345, backstop, dose pipeline) as in the published Figure 3. Output in a NEW folder
(<RESULTS_DATA>\\Fig3_A2_20261017); the published <RESULTS_DATA>\\Fig3_BL_20260929 is only read.
Usage (venv): python q25_fig3_const.py --arm Heuristic_aniso --gpu 0 [--smoke]   (also --arm Smolders)"""
from dirunc_paths import LIB, RESULTS_DATA, script  # data roots, see README
import os, sys
III = LIB  # was the III Scripts/_da_approvare folder; its scripts are found by name (dirunc_paths)
SRC = script("fig3_BL_runner.py")
OUT_A2 = rf"{RESULTS_DATA}/Fig3_A2_20261017"
assert not any(a.startswith("--out") for a in sys.argv), "output folder is fixed by this wrapper"
sys.argv += ["--out", OUT_A2]

s = open(SRC, encoding="utf-8").read()
old = "L0_ZYX = (15.0, 10.0, 10.0); ALPHA = 0.5; KS = 61"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, "L0_ZYX = (19.5, 13.0, 13.0); ALPHA = 0.0; KS = 61")
sys.path.insert(0, III)
print(f"[q25] Figure 3 A2: l0 = (19.5, 13, 13) mm, alpha = 0, out = {OUT_A2}", flush=True)
exec(compile(s, SRC, "exec"), {"__file__": SRC, "__name__": "__main__"})
