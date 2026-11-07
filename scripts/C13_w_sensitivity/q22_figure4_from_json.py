# -*- coding: utf-8 -*-
"""
q22_figure4_from_json.py -- Figure 4 (w-sweep, fig:wsweep) drawn from _out_q22_step6/step6.json (sigma_BL, step 6) instead of the
numbers hard-coded in make_figure_wsweep.py. Same layout: reuses make_figure_wsweep.build() with its LECE / RHO dicts filled from
the JSON. Writes ../../Figures/_da_approvare/Figure4_BL_draft.{png,tiff} (variant A = legend inside panel a). Nothing else is written.
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
FIG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Figures", "_da_approvare"); os.makedirs(FIG, exist_ok=True)
fig = F4.build("A")
fig.savefig(os.path.join(FIG, "Figure4_BL_draft.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(FIG, "Figure4_BL_draft.tiff"), dpi=800, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
print("->", os.path.join(FIG, "Figure4_BL_draft.png"))
