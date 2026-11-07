# -*- coding: utf-8 -*-
"""
make_figure2_q21.py -- draft of the new Figure 2 (Reviewer-2 Q2.1) from the CSVs of
q21_kernel_metrics_BL.py (default _out_q21_BL; 2026-09-29 A2): aniso K (adaptive, alpha = 0.5), the stationary
aniso K with the block-mean applied lengths (NOT alpha = 0: the lengths K applies on average, 2026-09-29 A2) and the
length-matched isotropic kernel, side by side
(user, 2026-09-27: the 11.7 mm isotropic comparator of the submitted Figure 2 is dropped).

  top row, impact of K:     (a) pre-screening folding rate per engine   (b) amplitude transmitted, sigma_ens / G
  bottom row, consistency:  (c) ensemble ACF along Z, representative case  (d) realised lambda_Z / lambda_X, 10 cases
      (d) reference lines = truncated-kernel theory (61-tap L2-normalised Gaussian, 1/e length of its ACF per axis),
      mean over the cases of the applied lengths in q21_lengths.csv (2026-09-29 A2; before: 1.5 and 1.0 = untruncated 2*ell)

Writes Figure2_BL_draft.png (300 dpi) + _master.png (800 dpi) + .tiff (PMB: RGB, LZW)
into ../../Figures/_da_approvare/ (drafts not yet approved; 2026-09-29 A2: was Figure2_q21_draft3). The submitted Figure2.tiff
is not touched. Env FIG2_OUT_DIR (optional) redirects the figure files only (checks); default behaviour unchanged.
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "_out_q21_BL")   # 2026-09-29 A2: default _out_q21_BL (was _out_q21_body, pre-N8 data)
FIG = OUT if len(sys.argv) > 1 else os.path.normpath(os.path.join(HERE, "..", "..", "Figures", "_da_approvare"))
FIG = os.environ.get("FIG2_OUT_DIR") or FIG   # 2026-09-29 A2: optional redirection of the figure files (checks); unset = unchanged
CASE_REP = 5
K = ["aniso", "aniso_fixed", "iso_matched"]
ENG = ["elastix", "convexadam", "unigradicon", "synra"]
ENG_LBL = ["Elastix", "ConvexAdam", "uniGradICON", "SyNRA"]
COL = {"aniso": "#c0392b", "aniso_fixed": "#e59866", "iso_matched": "#5d8aa8"}


def rows(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dedup(rs, keys):
    """Two GPU processes may both compute the case where they meet: keep the first row per key."""
    seen, out = set(), []
    for r in rs:
        k = tuple(r[c] for c in keys)
        if k not in seen:
            seen.add(k); out.append(r)
    return out


L = dedup(rows("q21_lengths.csv"), ["case", "kernel"]); S = dedup(rows("q21_sampler.csv"), ["case", "engine", "kernel"])
assert len({r["case"] for r in S}) == len({r["case"] for r in L}), "cases missing in one of the CSVs"
l_match = np.mean([float(r["iso_matched_mm"]) for r in L if r["kernel"] == "aniso"])
LBL = {"aniso": r"aniso, adaptive ($\alpha$ = 0.5)", "aniso_fixed": "aniso, stationary (block-mean lengths)",   # 2026-09-29 A2: not alpha = 0
       "iso_matched": f"iso, matched length ({l_match:.1f} mm)"}
sp_z = float([r for r in L if r["case"] == str(CASE_REP)][0]["spacing_zyx"].split("/")[0])


# 2026-09-29 A2: theory of the kernel actually applied (core.grf.gaussian_kernel_1d: 61 taps, L2-normalised), not 2*ell
def l1e_vox(sig_vox, ks=61):
    """1/e lag (voxels) of the ACF of the truncated, L2-normalised 1-D Gaussian kernel of sigma sig_vox (voxels)."""
    h = ks // 2; x = np.arange(-h, h + 1, dtype=float); k = np.exp(-x**2 / (2 * sig_vox**2)); k /= np.sqrt((k**2).sum())
    ac = np.correlate(k, k, "full"); ac = ac[len(k) - 1:] / ac[len(k) - 1]
    thr = 1 / np.e; j = int(np.argmax(ac < thr)); return (j - 1) + (ac[j - 1] - thr) / (ac[j - 1] - ac[j])


def theory_ratio(ell_z_mm, ell_x_mm, sz, sx):
    """lambda_Z / lambda_X of the truncated kernel, lambda_axis = l1e_vox(ell_axis / spacing_axis) * spacing_axis."""
    return (l1e_vox(ell_z_mm / sz) * sz) / (l1e_vox(ell_x_mm / sx) * sx)


TH = {"aniso": [], "iso_matched": []}
for r in L:
    if r["kernel"] != "aniso":
        continue
    sz_, _, sx_ = (float(v) for v in r["spacing_zyx"].split("/"))            # per-case spacing from the CSV
    TH["aniso"].append(theory_ratio(float(r["applied_Z_mm"]), float(r["applied_X_mm"]), sz_, sx_))
    TH["iso_matched"].append(theory_ratio(float(r["iso_matched_mm"]), float(r["iso_matched_mm"]), sz_, sx_))
print("truncated-kernel theory per case (spacing from the CSV): aniso", [round(v, 3) for v in TH["aniso"]],
      "iso_matched", [round(v, 3) for v in TH["iso_matched"]])
TH = {k: float(np.mean(v)) for k, v in TH.items()}
print("truncated-kernel theory, mean over cases: aniso %.4f  iso_matched %.4f" % (TH["aniso"], TH["iso_matched"]))


def per_engine(col):
    return {k: [np.mean([float(r[col]) for r in S if r["engine"] == e and r["kernel"] == k]) for e in ENG] for k in K}


plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 7.5})
fig, ax = plt.subplots(2, 2, figsize=(7.09, 5.2))
c, d, a, b = ax.ravel()   # variables keep their role; the grid puts impact first (user, 2026-09-26)

for k in K:                                                            # (a)
    acf = np.load(os.path.join(OUT, f"acfZ_case{CASE_REP}_{k}.npy"))
    a.plot(np.arange(len(acf)) * sp_z, acf, color=COL[k], lw=1.6, ls={"aniso": "-", "aniso_fixed": ":", "iso_matched": "--"}[k], label=LBL[k])
a.axhline(1 / np.e, color="grey", ls=":", lw=0.8)
a.set_xlabel("lag along Z, cranio-caudal (mm)"); a.set_ylabel("autocorrelation")
a.set_title(f"(c) ensemble ACF, case {CASE_REP}")

bp = b.boxplot([[float(r["ratio_Z_over_X"]) for r in L if r["kernel"] == k] for k in K],   # (b)
               tick_labels=["aniso\nadaptive", "aniso\nstationary", "iso\nmatched"], widths=0.55, patch_artist=True, showfliers=False)   # 2026-09-29 A2
for p, k in zip(bp["boxes"], K):
    p.set_facecolor(COL[k]); p.set_alpha(0.35)
for m in bp["medians"]:
    m.set_color("black")
# 2026-09-29 A2: reference lines = truncated-kernel theory (was 1.5 and 1.0, the untruncated 2*ell ratios)
b.axhline(TH["aniso"], color=COL["aniso"], ls=":", lw=0.9, label=f"truncated-kernel theory, aniso ({TH['aniso']:.2f})")
b.axhline(TH["iso_matched"], color=COL["iso_matched"], ls=":", lw=0.9, label=f"truncated-kernel theory, iso ({TH['iso_matched']:.2f})")
b.legend(loc="center right", fontsize=6, frameon=False)
b.set_ylabel(r"realised $\lambda_Z/\lambda_X$"); b.set_title("(d) anisotropy, 10 cases")

x = np.arange(len(ENG)); w = 0.27
for panel, col, scale, ylab, title in ((c, "first_folding_mean", 100, "pre-screening folding (%)", "(a) diffeomorphism"),
                                       (d, "sigma_ens_over_G_body", 1, r"$\sigma_{ens}/G$ (body mean)", "(b) amplitude transmitted")):
    v = per_engine(col)
    for i, k in enumerate(K):
        panel.bar(x + (i - 1) * w, np.array(v[k]) * scale, w, color=COL[k], label=LBL[k])
    panel.set_xticks(x); panel.set_xticklabels(ENG_LBL, rotation=25, ha="right", fontsize=7.5)
    panel.set_ylabel(ylab); panel.set_title(title)
h, l = c.get_legend_handles_labels()
d.set_ylim(0, 1)

fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))   # one legend, clear of the bars
os.makedirs(FIG, exist_ok=True)
base = os.path.join(FIG, "Figure2_BL_draft")   # 2026-09-29 A2: was Figure2_q21_draft3
fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
fig.savefig(base + "_master.png", dpi=800, bbox_inches="tight")
from PIL import Image
Image.open(base + "_master.png").convert("RGB").save(base + ".tiff", compression="tiff_lzw", dpi=(800, 800))
print("[out]", base + ".png")

# numbers behind the panels, for the text
for col in ("first_folding_mean", "mean_retries", "frac_r0", "frac_collapsed", "gain_rms", "sigma_ens_over_G_body"):
    v = per_engine(col)
    print(col, {k: [round(x, 4) for x in v[k]] for k in K})
for k in K:
    r = [float(q["ratio_Z_over_X"]) for q in L if q["kernel"] == k]
    print("ratio", k, round(np.mean(r), 3), "+-", round(np.std(r, ddof=1), 3))
print("iso_matched mm per case", [q["iso_matched_mm"] for q in L if q["kernel"] == "aniso"])
