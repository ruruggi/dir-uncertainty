# -*- coding: utf-8 -*-
"""Figure 1 v3 (2026-10-03, reviewer comments on Figure 1): same data and panels as Figure 1 v2 (q22_fig1_v2.py), with
  - 'lm' replaced by l_# (# as subscript) = number of landmarks (user choice; 'N' already used elsewhere);
  - larger lettering: tick labels, column titles, row/axis labels (the v2 ones were nearly unreadable once scaled to \\textwidth);
  - smaller canvas (13 x 11.5 in instead of 16 x 13.6) so that the print scaling is milder;
  - count-axis labels 1.0k / 1.2k / 1.4k instead of the repeated '1k'.
The shared drawing function make_figure1_4x4._draw_panel is NOT modified (repository code); sizes are adjusted after the call.
Control: annotated f' and LECE = Table 2 v2 (table23_v2.json) q<=10 columns (16/16).
Output: ../../Figures_modificabili/Figure1_v3.{png,tiff}. Usage (venv): python q22_fig1_v3.py"""
from dirunc_paths import DATASETS, LIB  # data roots, see README
import os, sys, types, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = LIB; PS = LIB + "/validation"
sys.path.insert(0, SCRIPTS); sys.path.insert(0, PS)
if "validation" not in sys.modules:
    _v = types.ModuleType("validation"); _v.__path__ = [PS]; sys.modules["validation"] = _v
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from validation.build_v11_data import ENGINE_PATHS
import make_figure1_4x4 as F1
DS_ROOT = f"{DATASETS}"; BL_ROOT = f"{DS_ROOT}/_sigma_BL_20260927"
bl = lambda p: os.path.join(BL_ROOT, os.path.relpath(p, DS_ROOT)).replace("\\", "/")
DATASETS = ["Lung-DIR-QA", "Abdominal-DIR-QA", "Liver-DIR-QA", "DIR-LAB"]; ENGINES = ["SyNRA", "ConvexAdam", "uniGradICON", "Elastix"]
new = {e: {ds: bl(p) for ds, p in ENGINE_PATHS[e]["datasets"].items()} for e in ENGINES}
F1COL = {"Lung-DIR-QA": "Lung", "Abdominal-DIR-QA": "Abdomen", "Liver-DIR-QA": "Liver", "DIR-LAB": "DIR-LAB"}
t2 = json.load(open(os.path.join(HERE, "_out_q22_table23_v2", "table23_v2.json")))
FIG = os.path.join(os.path.dirname(SCRIPTS), "Figures_modificabili"); ok = 0
TICK, TICK2, TITLE, LABEL, BOX = 12, 11, 15, 13, 12
kfmt = plt.FuncFormatter(lambda x, _: (f"{x / 1000:.1f}k" if x % 1000 else f"{int(x / 1000)}k") if x >= 1000 else f"{int(x)}")
fig, axes = plt.subplots(4, 4, figsize=(13.0, 11.5), constrained_layout=True)
for r, (eng, color, _) in enumerate(F1.ENGINE_ROWS):
    for c, ds in enumerate(DATASETS):
        ax = axes[r][c]; res = F1._compute_panel(new[eng][ds], 10.0, 0.25); fp = res["f_global"] / math.sqrt(3.0)
        ax2 = F1._draw_panel(ax, res["diag"], color,
                             f"n = {res['n_cases']}\n$l_{{\\#}}$ = {res['n_landmarks']:,}\nf = {fp:.2f}\nLECE = {res['lece_global']:.2f}")
        ax.tick_params(axis="both", labelsize=TICK); ax2.tick_params(axis="y", labelsize=TICK2)
        if res["diag"]["bin_counts"].max() >= 1000: ax2.yaxis.set_major_formatter(kfmt)
        for t in ax.texts: t.set_fontsize(BOX)
        if r == 0: ax.set_title(F1.COL_TITLE[F1COL[ds]], fontsize=TITLE, fontweight="bold")
        if c == 0: ax.set_ylabel(f"{eng}\nRMSE (mm)", fontsize=LABEL, fontweight="bold")
        if r == 3: ax.set_xlabel(r"Predicted $\sigma_p$ (mm)", fontsize=LABEL, fontweight="bold")
        t = t2["results"][f"T2|{ds}|{eng}|10.0"]; tf = next(v for k, v in t.items() if k in ("f_prime", "f", "fp", "f_sqrt3")) if isinstance(t, dict) else None
        same = isinstance(t, dict) and abs(float(tf) - fp) < 5e-3
        ok += bool(same); print(f"{ds:17s} {eng:12s} n {res['n_cases']:2d} f' {fp:.2f} LECE {res['lece_global']:.3f}  T2v2 {tf} {'OK' if same else '??'}")
assert ok == 16, ok
fig.savefig(os.path.join(FIG, "Figure1_v3.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(FIG, "Figure1_v3.tiff"), dpi=800, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"}); plt.close(fig)
print("saved", FIG, "| controls f' = Table 2 v2:", ok, "/ 16")
