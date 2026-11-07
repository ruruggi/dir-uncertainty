# -*- coding: utf-8 -*-
"""
q22_figure4_v3.py -- Figure 4 v3 (approved 2026-10-04): same data and layout as Figure4_v2 (q22_figure4_from_json.py: step6.json
through make_figure_wsweep.build("A")), but each engine also gets its own marker and line style, so the four curves stay
distinguishable in black and white. Markers as in Figure 2 (SyNRA square, uniGradICON triangle, Elastix circle, ConvexAdam diamond).
Panel letters moved to the lower left corner (empty there; in v2 they covered the first point of a curve).
Writes ../../Figures_modificabili/Figure4_v3.png (200 dpi, as v2) + .tiff (800 dpi, LZW). Nothing else.
"""
from dirunc_paths import LIB  # data roots, see README
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); PS = LIB + "/validation"; sys.path.insert(0, PS)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import make_figure_wsweep as F4

J = json.load(open(os.path.join(HERE, "_out_q22_step6", "step6.json")))
assert J["controls_pass"], "step 6 controls failed"
for e in F4.ENG:
    F4.LECE[e] = [J["sweep_w"][e][str(round(w, 1))]["lece"] for w in F4.w]
    F4.RHO[e] = [J["sweep_w"][e][str(round(w, 1))]["rho"] for w in F4.w]
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
axL.legend(fontsize=F4.LEG, loc="upper right", framealpha=0.92, handlelength=2.6, borderpad=0.5)  # rebuilt: handles show the new styles
base = os.path.normpath(os.path.join(HERE, "..", "..", "Figures_modificabili", "Figure4_v3"))
fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
fig.savefig(base + ".tiff", dpi=800, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"}); plt.close(fig)
print("->", base)
