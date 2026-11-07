# -*- coding: utf-8 -*-
"""
make_figure_wsweep.py  (PMB revision, R2.Table5) — w-ablation Table 5 as a figure.
Two-panel: LECE vs w and Spearman rho vs w on DIR-LAB Castillo, one curve per DIR
engine, default w=0.7 marked. Data = published Table-5 values (hard-coded).

Usage:
  python -m validation.make_figure_wsweep --variant A   # legend inside panel (a)
  python -m validation.make_figure_wsweep --variant B   # shared legend on top
  python -m validation.make_figure_wsweep --final A      # write Figure4.{png,tiff}
Previews go to re_Submissions Figures as Figure4_preview_<V>.png (no overwrite).
"""
import os, sys, shutil, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "Submission", "Submission PMB_20260601",
                    "re_Submissions PMB", "Figures")
PNG_COMPILE = os.path.join(ROOT, "Submission", "Figures", "png", "Figure4.png")

w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LECE = {
    "SyNRA":       [2.253, 1.837, 1.400, 1.030, 0.695, 0.345, 0.256, 0.310, 0.491, 0.643, 0.152],
    "ConvexAdam":  [0.756, 0.570, 0.423, 0.314, 0.209, 0.157, 0.117, 0.088, 0.122, 0.053, 0.021],
    "uniGradICON": [2.049, 1.662, 1.286, 0.961, 0.697, 0.467, 0.299, 0.171, 0.084, 0.078, 0.025],
    "Elastix":     [1.147, 0.897, 0.673, 0.475, 0.306, 0.174, 0.144, 0.192, 0.190, 0.171, 0.035],
}
RHO = {
    "SyNRA":       [0.265, 0.265, 0.265, 0.266, 0.267, 0.269, 0.269, 0.272, 0.277, 0.281, 0.099],
    "ConvexAdam":  [0.105, 0.105, 0.105, 0.106, 0.106, 0.107, 0.108, 0.110, 0.112, 0.117, 0.050],
    "uniGradICON": [0.046, 0.047, 0.048, 0.050, 0.052, 0.055, 0.058, 0.065, 0.073, 0.089, 0.083],
    "Elastix":     [0.193, 0.195, 0.197, 0.198, 0.201, 0.203, 0.207, 0.212, 0.220, 0.233, 0.168],
}
COLOR = {"SyNRA": "#1f77b4", "ConvexAdam": "#d62728",
         "uniGradICON": "#2ca02c", "Elastix": "#9467bd"}
ENG = ["SyNRA", "ConvexAdam", "uniGradICON", "Elastix"]
WDEF = 0.7

# enlarged fonts
LABEL, TICK, LEG, LETTER, ANNOT = 15, 13, 14, 18, 13


def build(variant):
    top_legend = (variant == "B")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                   constrained_layout=(not top_legend))
    handles = None
    for ax, data, ylab, letter in [
            (axL, LECE, "LECE (mm)", "(a)"),
            (axR, RHO,  r"Spearman $\rho_{\mathrm{pool}}(\sigma_p,\mathrm{TRE})$", "(b)")]:
        hs = []
        for e in ENG:
            h, = ax.plot(w, data[e], "o-", color=COLOR[e], lw=2.0, ms=5, label=e)
            hs.append(h)
        handles = hs
        ax.axvline(WDEF, color="0.4", ls="--", lw=1.1)
        ax.text(WDEF - 0.018, 0.5, "default $w=0.7$", rotation=90,
                transform=ax.get_xaxis_transform(), fontsize=ANNOT, color="0.35",
                ha="right", va="center")
        ax.set_xlabel("data-fidelity weight $w$", fontsize=LABEL, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=LABEL, fontweight="bold")
        ax.set_xlim(-0.03, 1.03)
        ax.tick_params(labelsize=TICK)
        ax.grid(True, alpha=0.25, ls=":")
        ax.text(0.02, 0.98, letter, transform=ax.transAxes, fontsize=LETTER,
                fontweight="bold", va="top")
    if top_legend:
        fig.legend(handles, ENG, loc="upper center", ncol=4, fontsize=LEG,
                   frameon=True, framealpha=0.9, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=[0, 0, 1, 0.90])
    else:
        axL.legend(fontsize=LEG, loc="upper right", framealpha=0.92,
                   handlelength=1.6, borderpad=0.5)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["A", "B"])
    ap.add_argument("--final", choices=["A", "B"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.final:
        fig = build(args.final)
        png = os.path.join(OUT, "Figure4.png"); tif = os.path.join(OUT, "Figure4.tiff")
        fig.savefig(png, dpi=200, bbox_inches="tight")
        fig.savefig(tif, dpi=800, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig); shutil.copyfile(png, PNG_COMPILE)
        print("FINAL variant", args.final, "->", png, "+ tiff + compile PNG")
    else:
        variants = [args.variant] if args.variant else ["A", "B"]
        for v in variants:
            fig = build(v)
            p = os.path.join(OUT, f"Figure4_preview_{v}.png")
            fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
            print("preview ->", p)


if __name__ == "__main__":
    main()
