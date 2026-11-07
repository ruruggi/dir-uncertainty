# -*- coding: utf-8 -*-
"""IV round, Q2.5 (A2): DVH band width per structure on the A2 Figure 3 runs, with the III script fig3_BL_oar_bands.py run UNCHANGED
on disk (exec of its source, output path patched to IV results/fig3_A2). Prints the sigma_DL / sigma_h band ratio per structure
(manuscript §3.6: «the band is wider with sigma_DL on every structure (1.2--1.9x)»; published runs: 1.24--1.93).
Usage (venv): python q25_oar_bands_A2.py"""
from dirunc_paths import LIB, RESULTS_DATA, script  # data roots, see README
import os, sys, json
III = LIB  # was the III Scripts/_da_approvare folder; its scripts are found by name (dirunc_paths)
SRC = script("fig3_BL_oar_bands.py")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "fig3_A2", "dvh_bands_by_structure_A2.json")
os.environ["FIG3_RES"] = rf"{RESULTS_DATA}/Fig3_A2_20261017"
sys.argv = [SRC, "Heuristic_aniso,Heuristic_iso,Smolders"]
s = open(SRC, encoding="utf-8").read()
old = 'os.path.join(HERE, "_out_fig3_BL", f"dvh_bands_by_structure{TAG}.json")'
assert s.count(old) == 1
s = s.replace(old, "OUT_A2")
exec(compile(s, SRC, "exec"), {"__file__": SRC, "__name__": "__main__", "OUT_A2": OUT})
d = json.load(open(OUT))
r = {k: d["Smolders"][k]["band_mean_volpct"] / d["Heuristic_aniso"][k]["band_mean_volpct"] for k in d["Heuristic_aniso"]}
print("A2 band ratio sigma_DL / sigma_h:", {k: round(v, 2) for k, v in r.items()}, "range %.2f-%.2f" % (min(r.values()), max(r.values())))
