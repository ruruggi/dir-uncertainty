# -*- coding: utf-8 -*-
"""IV round, Q2.6.d (2026-10-17): Figure 4 redrawn with the case-level 5-fold LECE at q <= 10 mm (results/wsweep_5fold_q10.json, from
q22_wsweep_5fold.py with QMAX=10) instead of the in-sample LECE of step6.json. Same drawing code as the approved Figure4_v3 (III
q22_figure4_v3.py, through _paper_scripts/make_figure_wsweep.build("A"), unchanged); only the two data arrays change: LECE = 5-fold mean,
rho = rho_pool (identical to the in-sample rho, which does not depend on f). Output (draft, not installed): IV results/fig4_q10/Figure4.{png,tiff}.
Usage: python q26d_figure4_q10.py"""
from dirunc_paths import LIB  # data roots, see README
import os, sys, json
IVL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PS = LIB + "/validation"  # was the III Scripts/_paper_scripts folder
sys.path.insert(0, PS)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import make_figure_wsweep as F4

J = json.load(open(os.path.join(IVL, "results", "wsweep_5fold_q10.json")))
assert J["meta"]["qmax_mm"] == 10.0
for e in F4.ENG:
    F4.LECE[e] = [J["results"][e][str(round(w, 1))]["lece5_mean"] for w in F4.w]
    F4.RHO[e] = [J["results"][e][str(round(w, 1))]["rho_pool"] for w in F4.w]
MK = {"SyNRA": "s", "ConvexAdam": "D", "uniGradICON": "^", "Elastix": "o"}
LS = {"SyNRA": "-", "ConvexAdam": (0, (5, 2)), "uniGradICON": (0, (5, 1.5, 1, 1.5)), "Elastix": (0, (1, 1.2))}
MS = {"SyNRA": 6.5, "ConvexAdam": 6, "uniGradICON": 7.5, "Elastix": 6.5}
fig = F4.build("A"); axL, axR = fig.axes[:2]; n = 0
for ax in (axL, axR):
    for ln in ax.lines:
        e = ln.get_label()
        if e in MK: ln.set_marker(MK[e]); ln.set_linestyle(LS[e]); ln.set_markersize(MS[e]); n += 1
    for t in ax.texts:
        if t.get_text() in ("(a)", "(b)"): t.set_position((0.02, 0.02)); t.set_va("bottom"); n += 1
assert n == 10, n
for ax in (axL, axR):
    for t in ax.texts:
        if "default" in t.get_text(): t.set_text(t.get_text().replace("default ", ""))   # w = 0.7 is fixed a priori (Q2.6.d)
axL.set_ylabel("LECE, 5-fold (mm)")
if axL.get_legend(): axL.get_legend().remove()          # in (a) the legend covered the SyNRA curve at w >= 0.6
h, l = axL.get_legend_handles_labels()
axR.legend(h, l, fontsize=F4.LEG, loc="lower left", bbox_to_anchor=(0.10, 0.03), framealpha=0.92, handlelength=2.6, borderpad=0.5)
out = os.path.join(IVL, "results", "fig4_q10"); os.makedirs(out, exist_ok=True); base = os.path.join(out, "Figure4")
fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
fig.savefig(base + ".tiff", dpi=800, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"}); plt.close(fig)
print("->", base)
