# -*- coding: utf-8 -*-
"""Figure 2 v5 (approved 2026-10-02, Q2.4.d ii): v4 + reference curve of the fixed isotropic Gaussian kernel of Smolders et al. (sigma_c = 15 voxels on their 1.95x1.95x2.0 mm grid = 30.0 mm CC, 29.25 mm in-plane; theory exp(-h^2/(4 sigma^2)), 1/e at 2 sigma = 60 / 58.5 mm); one shared legend below (c,d).
v4 docstring: Figure 2 v4 (approved 2026-10-01, option B, all four engines): panels (a) folding and (b) full-amplitude fraction as in Figure2_v3; panel (d) anisotropy ratio removed;
the empirical correlogram (corr_landmarks_pooled.py -> _out_corr_probe/corr_pooled.json) enters the figure in two alternative layouts:
  option A: (c) ensemble ACF along Z of K and iso (as in v3); (d) correlogram of the landmark DIR error, CC and in-plane pairs, 3 engines, with the K / iso curves of both directions.
  option B: (c) cranio-caudal: ACF of K and iso along Z + CC correlograms; (d) in-plane: theory of K and iso + in-plane correlograms.
ConvexAdam (no spatial structure) left out of the plots (caption). Output: ../../Figures_modificabili/_da_approvare/Figure2_v4_preview{A,B}.png"""
import os, csv, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "_out_q21_BLv2"); CORR = os.path.join(HERE, "_out_corr_probe", "corr_pooled.json")
FIG = os.path.normpath(os.path.join(HERE, "..", "..", "Figures_modificabili")); CASE_REP = 5
K = ["aniso", "iso_geo"]; COL = {"aniso": "#8e1b12", "iso_geo": "#a9c4db"}; LS = {"aniso": "-", "iso_geo": "--"}
LBL = {"aniso": r"aniso, adaptive ($\alpha$ = 0.5; $\ell_0$ = 15/10/10 mm)", "iso_geo": r"iso, spatially fixed ($\ell$ = 11.5 mm)"}
ENG = ["elastix", "convexadam", "unigradicon", "synra"]; ENG_LBL = ["Elastix", "ConvexAdam", "uniGradICON", "SyNRA"]
DATA_ENG = ["SyNRA", "uniGradICON", "Elastix", "ConvexAdam"]; ECOL = {"SyNRA": "#1b1b1b", "uniGradICON": "#5e5e5e", "Elastix": "#8f8f8f", "ConvexAdam": "#bdbdbd"}; EMK = {"SyNRA": "s", "uniGradICON": "^", "Elastix": "o", "ConvexAdam": "D"}
def rows(n):
    seen, out = set(), []
    for r in csv.DictReader(open(os.path.join(OUT, n), encoding="utf-8")):
        k = (r["case"], r.get("engine", ""), r["kernel"])
        if k not in seen: seen.add(k); out.append(r)
    return out
L, S = rows("q21_lengths.csv"), rows("q21_sampler.csv"); C = json.load(open(CORR))
MID = 0.5 * (np.array(C["bins_mm"][1:]) + np.array(C["bins_mm"][:-1]))
grid = np.linspace(0, 100, 201); acfm = {}
for k in K:
    rr = []
    for c in range(1, 11):
        spz = float([r for r in L if r["case"] == str(c)][0]["spacing_zyx"].split("/")[0]); a = np.load(os.path.join(OUT, f"acfZ_case{c}_{k}.npy"))
        rr.append(np.interp(grid, np.arange(len(a)) * spz, a, right=0.0))
    acfm[k] = np.mean(rr, 0)
lX = np.mean([float(r["applied_X_mm"]) for r in L if r["kernel"] == "aniso"]); liso = (15 * 10 * 10) ** (1 / 3); theo = lambda l: np.exp(-grid ** 2 / (4 * l ** 2))
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 6.5})

def bars(a, b):
    x = np.arange(len(ENG)); w = 0.36
    pe = lambda col: {k: [100 * np.mean([float(r[col]) for r in S if r["engine"] == e and r["kernel"] == k]) for e in ENG] for k in K}
    fold, full = pe("first_folding_mean"), pe("frac_r0")
    for i, k in enumerate(K):
        a.bar(x + (i - 0.5) * w, fold[k], w, color=COL[k], edgecolor="black", linewidth=0.5, label=LBL[k]); b.bar(x + (i - 0.5) * w, full[k], w, color=COL[k], edgecolor="black", linewidth=0.5)
    for ax, ttl, yl in ((a, "(a) diffeomorphism", "pre-screening folding (% body voxels)"), (b, "(b) amplitude backstop", "samples at full amplitude (%)")):
        ax.set_xticks(x); ax.set_xticklabels(ENG_LBL, rotation=25, ha="right", fontsize=7.5); ax.set_ylabel(yl); ax.set_title(ttl)
    b.set_ylim(0, 105); b.axhline(100, color="grey", ls=":", lw=0.8)

def data(ax, cls, ls="-", band=True, suffix=""):
    for e in DATA_ENG:
        d = C["engines"][e]["corr"][cls]; r = np.array(d["r"], float); lo = np.array(d["lo"], float); hi = np.array(d["hi"], float)
        if band: ax.fill_between(MID, lo, hi, color=ECOL[e], alpha=0.12, lw=0)
        ax.plot(MID, r, ls=ls, marker=EMK[e], color=ECOL[e], ms=3.5, lw=1.1, label=f"{e}{suffix}")

def finish(ax, ttl, xl="separation h (mm)"):
    ax.axhline(0, color="grey", lw=0.6); ax.axhline(1 / np.e, color="grey", ls=":", lw=0.7); ax.set_xlim(0, 100); ax.set_ylim(-0.15, 1.02)
    ax.set_title(ttl); ax.set_xlabel(xl)

# ---- option B (final)
fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(7.09, 6.3)); bars(a, b)
c.plot(grid, acfm["aniso"], color=COL["aniso"], lw=1.8, label="K (aniso)"); c.plot(grid, acfm["iso_geo"], color=COL["iso_geo"], lw=2.2, ls="--", label="iso"); c.plot(grid, theo(30.0), color="#d07a00", lw=1.6, ls="-.", label="Smolders et al., fixed kernel")
data(c, "CC"); c.set_ylabel("correlation"); finish(c, "(c) cranio-caudal: model vs landmark error")
d.plot(grid, theo(lX), color=COL["aniso"], lw=1.8); d.plot(grid, theo(liso), color=COL["iso_geo"], lw=2.2, ls="--"); d.plot(grid, theo(29.25), color="#d07a00", lw=1.6, ls="-.")
data(d, "in-plane"); d.set_ylabel("correlation"); finish(d, "(d) in-plane: model vs landmark error")
h, l = a.get_legend_handles_labels(); fig.tight_layout(rect=(0, 0.07, 1, 0.93)); fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
hc, lc = c.get_legend_handles_labels(); from matplotlib.lines import Line2D; blank = Line2D([], [], lw=0)
# column-major fill, ncol=4: row 1 = the three correlation models, row 2 = the four engines (landmark error)
H = [hc[0], hc[3], hc[1], hc[4], hc[2], hc[5], blank, hc[6]]; LL = ["correlation model: " + lc[0], "landmark error: " + lc[3], lc[1], lc[4], lc[2], lc[5], "", lc[6]]
fig.legend(H, LL, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=6.8)
assert len(c.lines) >= 3 and len(d.lines) >= 3 and any(l.get_color() == "#d07a00" for l in d.lines) and any(l.get_color() == "#d07a00" for l in c.lines), "Smolders curve missing in (c) or (d)"
base = os.path.join(FIG, "Figure2_v5"); fig.savefig(base + ".png", dpi=300, bbox_inches="tight"); fig.savefig(base + "_master.png", dpi=800, bbox_inches="tight")
from PIL import Image; Image.open(base + "_master.png").convert("RGB").save(base + ".tiff", compression="tiff_lzw", dpi=(800, 800)); os.remove(base + "_master.png"); print("[out]", base)
