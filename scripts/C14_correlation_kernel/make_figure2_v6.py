# -*- coding: utf-8 -*-
"""Figure 2 v6 (approved 2026-10-04): same data and plots as Figure2_v5 (whose loading/plotting helpers are reused from make_figure2_v5.py),
legends moved inside the panels to save vertical space: (a) kernel legend (aniso / iso, one per row) upper left, y-axis headroom so it clears
the SyNRA bars; (c) correlation models (K, iso, Smolders) upper right; (d) landmark error of the four engines upper right. No figure-level
legends; canvas 7.09 x 5.45 in (v5: 6.3 in). Output: ../../Figures_modificabili/Figure2_v6.png (300 dpi) + .tiff (800 dpi, LZW)."""
import os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
# reuse the data loading and plotting helpers of v5 without running its figure section
src = open(os.path.join(HERE, "make_figure2_v5.py"), encoding="utf-8").read().split("# ---- option B (final)")[0]
g = {"__file__": os.path.join(HERE, "make_figure2_v5.py")}; exec(compile(src, "make_figure2_v5.py", "exec"), g)
COL, grid, acfm, theo, lX, liso = g["COL"], g["grid"], g["acfm"], g["theo"], g["lX"], g["liso"]

fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(7.09, 5.45)); g["bars"](a, b)
a.set_ylim(0, 1.35 * a.get_ylim()[1]); a.legend(loc="upper left", ncol=1, frameon=False, fontsize=6.5)  # headroom: legend clears the SyNRA bars
c.plot(grid, acfm["aniso"], color=COL["aniso"], lw=1.8, label="K (aniso)"); c.plot(grid, acfm["iso_geo"], color=COL["iso_geo"], lw=2.2, ls="--", label="iso"); c.plot(grid, theo(30.0), color="#d07a00", lw=1.6, ls="-.", label="Smolders et al., fixed kernel")
nm = len(c.lines); g["data"](c, "CC"); c.set_ylabel("correlation"); g["finish"](c, "(c) cranio-caudal: model vs landmark error")
d.plot(grid, theo(lX), color=COL["aniso"], lw=1.8); d.plot(grid, theo(liso), color=COL["iso_geo"], lw=2.2, ls="--"); d.plot(grid, theo(29.25), color="#d07a00", lw=1.6, ls="-.")
g["data"](d, "in-plane"); d.set_ylabel("correlation"); g["finish"](d, "(d) in-plane: model vs landmark error")
hc, lc = c.get_legend_handles_labels()
c.legend(hc[:nm], lc[:nm], loc="upper right", title="correlation model", frameon=False, fontsize=6.5, title_fontsize=6.5)
d.legend(hc[nm:], lc[nm:], loc="upper right", title="landmark error", frameon=False, fontsize=6.5, title_fontsize=6.5)
assert nm == 3 and len(hc) == 7, (nm, len(hc))
fig.tight_layout()
base = os.path.join(g["FIG"], "Figure2_v6"); fig.savefig(base + ".png", dpi=300, bbox_inches="tight"); fig.savefig(base + "_master.png", dpi=800, bbox_inches="tight")
from PIL import Image; Image.open(base + "_master.png").convert("RGB").save(base + ".tiff", compression="tiff_lzw", dpi=(800, 800)); os.remove(base + "_master.png"); print("[out]", base)
