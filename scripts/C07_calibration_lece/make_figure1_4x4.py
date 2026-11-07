# -*- coding: utf-8 -*-
"""
make_figure1_4x4.py  (PMB revision, Q1.3 / alpha)

Headless regeneration of manuscript Figure 1 (reliability diagrams), 4x4 grid
= 4 DIR engines (rows) x 4 datasets (columns), under the alpha presentation:
  * NO LIM pre-filter on any engine (SyNRA on the full set like the others);
  * shared convergence filter: cases with per-case post-DIR RMSE TRE <= q
    (default q = 10 mm), matching the primary regime of Table 2;
  * bin width 0.25 mm (consistent with section 2.5 / Table 3);
  * per-panel annotation: n cases, n landmarks, global-fit f, global LECE
    (= the "LECE global @ q<=10" column of Table 2).

The previous committed Figure1.png was assembled with the interactive
reliability_editor.py + manual montage; this script reproduces the same
information programmatically with clean styling (no manual tweaks).

Outputs (to figures/):
  Figure1_alpha.png   (compile / preview)
  Figure1_alpha.tiff  (800 dpi, LZW; PMB submittable spec is applied at copy time)

Run with the project venv (numpy/skimage pinned):
  venv/Scripts/python.exe -m validation.make_figure1_4x4 --q 10
"""
from __future__ import annotations
from dirunc_paths import DATASETS  # data roots, see README
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from validation.calibrate import load_all_landmarks, stack_cases, fit_c_grid
from validation.lece import compute_lece

_CASTILLO = f"{DATASETS}/Castillo-DIR-LAB"

# rows = engines (order as in the manuscript); cols = datasets
ENGINE_ROWS = [
    ("SyNRA", "#1f77b4", {
        "Lung":    f"{DATASETS}/Results_SynRA/Lung",
        "Abdomen": f"{DATASETS}/Results_SynRA/Abdomen",
        "Liver":   f"{DATASETS}/Results_SynRA/Liver",
        "DIR-LAB": f"{_CASTILLO}/Results_synra_procrustes",
    }),
    ("ConvexAdam", "#d62728", {
        "Lung":    f"{DATASETS}/Lung-DIR-QA/Results_adam_procrustes",
        "Abdomen": f"{DATASETS}/AbdominalDIR-QA/Results_adam_procrustes",
        "Liver":   f"{DATASETS}/Liver-DIR-QA/Results_adam_procrustes",
        "DIR-LAB": f"{_CASTILLO}/Results_convexadam_procrustes",
    }),
    ("uniGradICON", "#2ca02c", {
        "Lung":    f"{DATASETS}/Lung-DIR-QA/Results_unigradicon_procrustes",
        "Abdomen": f"{DATASETS}/AbdominalDIR-QA/Annotated Image Pairs/Results_unigradicon_procrustes",
        "Liver":   f"{DATASETS}/Liver-DIR-QA/Results_unigradicon_procrustes",
        "DIR-LAB": f"{_CASTILLO}/Results_unigradicon_procrustes",
    }),
    ("Elastix", "#9467bd", {
        "Lung":    f"{DATASETS}/Lung-DIR-QA/Results_elastix_procrustes",
        "Abdomen": f"{DATASETS}/AbdominalDIR-QA/Annotated Image Pairs/Results_elastix_procrustes",
        "Liver":   f"{DATASETS}/Liver-DIR-QA/Results_elastix_procrustes",
        "DIR-LAB": f"{_CASTILLO}/Results_elastix_procrustes",
    }),
]
COLS = ["Lung", "Abdomen", "Liver", "DIR-LAB"]
COL_TITLE = {"Lung": "Lung-DIR-QA", "Abdomen": "Abdominal-DIR-QA",
             "Liver": "Liver-DIR-QA", "DIR-LAB": "DIR-LAB Castillo"}


def _compute_panel(out_root, q_mm, bin_width=0.25):
    """No-LIM; keep cases with per-case post-DIR RMSE TRE <= q_mm."""
    cases = load_all_landmarks(out_root)
    if not cases:
        return None
    if q_mm is not None:
        cases = [c for c in cases
                 if float(np.sqrt(np.mean(c.tre_mm ** 2))) <= q_mm]
    if not cases:
        return None
    tre, sig_raw = stack_cases(cases)
    f_global, _ = fit_c_grid(tre, sig_raw, bin_width=bin_width)
    lece, diag = compute_lece(tre, f_global * sig_raw, bin_width=bin_width)
    return {"n_cases": len(cases), "n_landmarks": int(tre.size),
            "f_global": float(f_global), "lece_global": float(lece), "diag": diag}


def _draw_panel(ax, diag, color, stats_text):
    bc, counts, rmse, bw = (diag["bin_centers"], diag["bin_counts"],
                            diag["bin_rmse"], diag["bin_width"])
    ne = counts > 0
    bc_ne, rmse_ne, counts_ne = bc[ne], rmse[ne], counts[ne]
    if counts.sum() > 0:
        cum = np.cumsum(counts) / counts.sum()
        sigma_max = bc[min(int(np.searchsorted(cum, 0.999)), len(bc) - 1)] + bw
    else:
        sigma_max = bc.max() if bc.size else 1.0
    ax2 = ax.twinx()
    ax2.bar(bc_ne, counts_ne, width=bw * 0.9, color=color, alpha=0.25, edgecolor="none")
    ax2.tick_params(axis="y", labelsize=7, colors="0.45")
    if counts_ne.size and counts_ne.max() >= 1000:
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}"))
    rmse_max = rmse_ne.max() if rmse_ne.size else sigma_max
    plot_max = max(sigma_max, rmse_max) * 1.1
    ax.plot([0, plot_max], [0, plot_max], "k-", lw=0.8, alpha=0.6, zorder=1)
    ax.plot(bc_ne, rmse_ne, "o-", color=color, lw=1.4, ms=3.5, zorder=3)
    ax.set_xlim(0, sigma_max); ax.set_ylim(0, plot_max)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes, fontsize=15,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.6", alpha=0.85))
    return ax2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=10.0, help="convergence threshold (mm)")
    ap.add_argument("--bin_width", type=float, default=0.25)
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "Submission", "Submission PMB_20260601",
                           "re_Submissions PMB", "Figures")
    os.makedirs(out_dir, exist_ok=True)

    nrow, ncol = len(ENGINE_ROWS), len(COLS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.4 * nrow),
                             constrained_layout=True)
    summary = {}
    for r, (eng, color, paths) in enumerate(ENGINE_ROWS):
        summary[eng] = {}
        for c, ds in enumerate(COLS):
            ax = axes[r][c]
            res = _compute_panel(paths[ds], args.q, args.bin_width)
            if res is None:
                ax.text(0.5, 0.5, "no cases", ha="center", va="center",
                        transform=ax.transAxes); continue
            stats = (f"n = {res['n_cases']}\n"
                     f"lm = {res['n_landmarks']:,}\n"
                     f"f = {res['f_global']:.2f}\n"
                     f"LECE = {res['lece_global']:.2f}")
            _draw_panel(ax, res["diag"], color, stats)
            if r == 0:
                ax.set_title(COL_TITLE[ds], fontsize=11, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{eng}\nRMSE (mm)", fontsize=10, fontweight="bold")
            if r == nrow - 1:
                ax.set_xlabel(r"Predicted $\sigma_p$ (mm)", fontsize=10, fontweight="bold")
            summary[eng][ds] = {k: res[k] for k in
                                ("n_cases", "n_landmarks", "f_global", "lece_global")}
            print(f"  {eng:<11} {ds:<8} n={res['n_cases']:>2} "
                  f"f={res['f_global']:5.2f} LECE={res['lece_global']:.3f}")

    png = os.path.join(out_dir, "Figure1_alpha.png")
    tif = os.path.join(out_dir, "Figure1_alpha.tiff")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(tif, dpi=800, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    with open(os.path.join(out_dir, "Figure1_alpha_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"q_mm": args.q, "bin_width": args.bin_width, "data": summary}, f, indent=2)
    print(f"\nSaved {png}\nSaved {tif} (800 dpi LZW)")


if __name__ == "__main__":
    main()
