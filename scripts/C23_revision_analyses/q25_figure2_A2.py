# -*- coding: utf-8 -*-
"""IV round, Q2.5 (A2, approved 2026-10-17): Figure 2 regenerated with the A2 arms (constant K 19.5/13/13 mm = aniso_const, isotropic
kernel of equal volume 14.9 mm = iso_const; results/q25_const_lengths from q25_const_lengths.py). Same drawing code and layout as the
approved Figure2_v6 (III make_figure2_v5.py loaders + v6 panel layout); only the data folder, the two arms and the in-plane lengths change.
The correlogram of the landmark error (III _out_corr_probe/corr_pooled.json) is unchanged. Output: IV Figures/Figure3.png (300 dpi) and
Figure3.tiff (800 dpi, LZW). Usage: python q25_figure2_A2.py"""
from dirunc_paths import LIB, script  # data roots, see README
import os
III = LIB  # was the III Scripts/_da_approvare folder; its scripts are found by name (dirunc_paths)
IVL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(IVL, "results", "q25_const_lengths")
FIGDIR = os.path.join(os.path.dirname(IVL), "Figures")
SRC = script("make_figure2_v5.py")
src = open(SRC, encoding="utf-8").read().split("# ---- option B (final)")[0]


def patch(old, new):
    global src
    assert src.count(old) == 1, (src.count(old), old)
    src = src.replace(old, new)


patch('OUT = os.path.join(HERE, "_out_q21_BLv2")', "OUT = DATA_A2")
patch('K = ["aniso", "iso_geo"]; COL = {"aniso": "#8e1b12", "iso_geo": "#a9c4db"}; LS = {"aniso": "-", "iso_geo": "--"}',
      'K = ["aniso_const", "iso_const"]; COL = {"aniso_const": "#8e1b12", "iso_const": "#a9c4db"}; LS = {"aniso_const": "-", "iso_const": "--"}')
patch('LBL = {"aniso": r"aniso, adaptive ($\\alpha$ = 0.5; $\\ell_0$ = 15/10/10 mm)", "iso_geo": r"iso, spatially fixed ($\\ell$ = 11.5 mm)"}',
      'LBL = {"aniso_const": r"K, anisotropic ($\\ell$ = 19.5/13/13 mm)", "iso_const": r"iso ($\\ell$ = 14.9 mm)"}')
patch('lX = np.mean([float(r["applied_X_mm"]) for r in L if r["kernel"] == "aniso"]); liso = (15 * 10 * 10) ** (1 / 3)',
      'lX = 13.0; liso = (19.5 * 13.0 * 13.0) ** (1 / 3)')
g = {"__file__": SRC, "__name__": "fig2_A2", "DATA_A2": DATA}
exec(compile(src, SRC, "exec"), g)
import matplotlib.pyplot as plt
from PIL import Image
COL, grid, acfm, theo, lX, liso = g["COL"], g["grid"], g["acfm"], g["theo"], g["lX"], g["liso"]
assert len(g["S"]) == 120 and abs(liso - 14.88) < 0.01
# 2026-10-17 (Q2.6.f): panels (a, b) on the cases retained at q <= 10 mm, as Table 2: SyNRA without DIR-LAB case 8 (RMS TRE > 10 mm)
g["S"][:] = [r for r in g["S"] if not (r["engine"] == "synra" and r["case"] == "8")]
assert len(g["S"]) == 117, len(g["S"])                              # 3 arms in the CSV (aniso_fixed, aniso_const, iso_const) x case 8

# ---- layout of the approved Figure2_v6
fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(7.09, 5.45)); g["bars"](a, b)
a.set_ylim(0, 1.35 * a.get_ylim()[1]); a.legend(loc="upper left", ncol=1, frameon=False, fontsize=6.5)
c.plot(grid, acfm["aniso_const"], color=COL["aniso_const"], lw=1.8, label="K (aniso)")
c.plot(grid, acfm["iso_const"], color=COL["iso_const"], lw=2.2, ls="--", label="iso")
c.plot(grid, theo(30.0), color="#d07a00", lw=1.6, ls="-.", label="Smolders et al., fixed kernel")
nm = len(c.lines); g["data"](c, "CC"); c.set_ylabel("correlation"); g["finish"](c, "(c) cranio-caudal: model vs landmark error")
d.plot(grid, theo(lX), color=COL["aniso_const"], lw=1.8); d.plot(grid, theo(liso), color=COL["iso_const"], lw=2.2, ls="--")
d.plot(grid, theo(29.25), color="#d07a00", lw=1.6, ls="-.")
g["data"](d, "in-plane"); d.set_ylabel("correlation"); g["finish"](d, "(d) in-plane: model vs landmark error")
hc, lc = c.get_legend_handles_labels()
c.legend(hc[:nm], lc[:nm], loc="upper right", title="correlation model", frameon=False, fontsize=6.5, title_fontsize=6.5)
d.legend(hc[nm:], lc[nm:], loc="upper right", title="landmark error", frameon=False, fontsize=6.5, title_fontsize=6.5)
assert nm == 3 and len(hc) == 7, (nm, len(hc))
fig.tight_layout()
base = os.path.join(FIGDIR, "Figure3"); fig.savefig(base + ".png", dpi=300, bbox_inches="tight"); fig.savefig(base + "_master.png", dpi=800, bbox_inches="tight")
Image.open(base + "_master.png").convert("RGB").save(base + ".tiff", compression="tiff_lzw", dpi=(800, 800)); os.remove(base + "_master.png")
print("[out]", base, "| K 1/e along z: %.1f mm, iso %.1f mm" % (grid[(acfm["aniso_const"] < 1 / 2.718281828).argmax()], grid[(acfm["iso_const"] < 1 / 2.718281828).argmax()]))
