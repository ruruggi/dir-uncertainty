# -*- coding: utf-8 -*-
"""IV round, critical review 1.2 (user decision 2026-10-24, option B): Figure 4(b) showed z_k - (ensemble mean) under the labels
"delta_h - mu" / "delta_DL - mu" and a "79th / 18th sample" index (0-based), while the caption says z_k - mu. Panel (b) now draws the
perturbation z_k - mu of the same samples (the one whose L2 distance from the ensemble mean is the median of the K_s = 100), labelled
"z_k - mu (sigma_h)", "mu (Velocity)", "z_k - mu (sigma_DL)", without the sample index. After the reviewer (approved 2026-10-24):
sigma_DL upright (\mathrm{DL}, as in the text) in panels (a) and (b); in (d) the right DVH panel starts its x tick labels at 10, so its
"0" no longer touches the "60" of the left panel (read as "600"). Everything else unchanged.
Runs the III builder fig3_BL_build.py UNCHANGED on disk (exec of its source with string patches, assert count == 1) on the A2 outputs
(<RESULTS_DATA>/Fig3_A2_20261017, only read); outputs go to results/q36_figure4/ through the builder's own env overrides
(FIG3_OUT, FIG3_SUBMISSION_DIR, FIG3_BASENAME), never into the III folder.
  --mode control : no patch -> must equal the published Figures/Figure4.png pixel by pixel;
  --mode new     : panel (b) as above.
Usage: python q36_figure4_panel_b.py --mode control|new"""
from dirunc_paths import LIB, RESULTS_DATA, script  # data roots, see README
import os, sys, argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["control", "new"], required=True); args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); IVL = os.path.dirname(HERE)
III = LIB  # was the III Scripts/_da_approvare folder; its scripts are found by name (dirunc_paths)
SRC = script("fig3_BL_build.py")
OUTD = os.path.join(IVL, "results", "q36_figure4"); os.makedirs(OUTD, exist_ok=True)
PUBLISHED = os.path.join(os.path.dirname(IVL), "Figures", "Figure4.png")
BASE = "Figure4_%s" % args.mode
os.environ.update(FIG3_RES=rf"{RESULTS_DATA}/Fig3_A2_20261017", FIG3_SUBMISSION_DIR=OUTD, FIG3_BASENAME=BASE,
                  FIG3_OUT=os.path.join(OUTD, "_preview_%s.png" % BASE))
s = open(SRC, encoding="utf-8").read()


def patch(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old)
    s = s.replace(old, new)


if args.mode == "new":
    # the deterministic field is the same Velocity DVF in both arms: check it, then subtract it
    patch("delta_heu = sample_heu - mu_heu_mean", "delta_heu = sample_heu - mu_orig")
    patch("delta_smo = sample_smo - mu_smo_mean",
          "assert np.array_equal(load_dvf(f'{RES}/{ARM_S}/frac_01/mu_dvf.npy'), mu_orig)\ndelta_smo = sample_smo - mu_orig")
    patch("rf'$\\delta_h - \\mu$' + '\\n'\n    + rf'({_kh}$^{{\\mathrm{{th}}}}$ sample; arrows ×{FACTOR_DELTA:.0f})',",
          "rf'$z_k - \\mu$  ($\\sigma_h$)' + '\\n'\n    + rf'(arrows ×{FACTOR_DELTA:.0f})',")
    patch("rf'$\\mu_{{\\mathrm{{{MU_LABEL}}}}}$' + '\\n'", "rf'$\\mu$  ({MU_LABEL})' + '\\n'")
    patch("rf'$\\delta_{{DL}} - \\mu$' + '\\n'\n    + rf'({_ks}$^{{\\mathrm{{th}}}}$ sample; arrows ×{FACTOR_DELTA:.0f})',",
          "rf'$z_k - \\mu$  ($\\sigma_{{\\mathrm{{DL}}}}$)' + '\\n'\n    + rf'(arrows ×{FACTOR_DELTA:.0f})',")
    patch("r'Smolders  $\\sigma_{DL}$  (L20S3-4)'", "r'Smolders  $\\sigma_{\\mathrm{DL}}$  (L20S3-4)'")
    patch("draw_dvh(ax_d2, 'Smolders-FC ensemble DVH',   dvh_smo, show_ylabel=False)",
          "draw_dvh(ax_d2, 'Smolders-FC ensemble DVH',   dvh_smo, show_ylabel=False)\nax_d2.set_xticks(range(10, 61, 10))")
sys.path.insert(0, III)
print(f"[q36] Figure 4, mode {args.mode}, out {OUTD}", flush=True)
exec(compile(s, SRC, "exec"), {"__file__": SRC, "__name__": "__main__"})

png = os.path.join(OUTD, BASE + ".png")
if args.mode == "control":
    from PIL import Image
    a = np.asarray(Image.open(png).convert("RGB")).astype(int); b = np.asarray(Image.open(PUBLISHED).convert("RGB")).astype(int)
    nd = int((a != b).any(axis=2).sum()) if a.shape == b.shape else -1
    print("[control] old labels and data vs published Figure4.png: shape %s vs %s, differing pixels %d" % (a.shape, b.shape, nd))
    assert nd == 0
print("[out]", png)
