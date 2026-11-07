# -*- coding: utf-8 -*-
"""IV round, critical review 1.1 (user decision 2026-10-24): Figure 3 with SyNRA sampled at the f of Table 2 (2.98; rows of
q25_const_lengths_synra298.py) and without panel (b), whose eight bars are all at 100 % once f is calibrated (stated in the text).
Runs q32_figure2_trunc.py UNCHANGED on disk (exec of its source with string patches, assert count == 1), which in turn runs
q25_figure2_A2.py with the truncated in-plane ACF of panel (d). Output to results/q35_figure3/ (draft, NOT Figures/).
  --mode control : old data, four panels -> must equal the published Figures/Figure3.png pixel by pixel;
  --mode new     : merged data (old CSV with the SyNRA aniso_const/iso_const rows replaced by the f = 2.98 rows), three panels
                   (a) folding, (b) cranio-caudal ACF, (c) in-plane ACF.
The merged data folder results/q35_fig3_data/ is rebuilt at each run; the old and new data folders are only read.
Usage: python q35_figure3_3panels.py --mode control|new"""
import os, sys, csv, shutil, argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["control", "new"], required=True); args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); IVL = os.path.dirname(HERE); RES = os.path.join(IVL, "results")
OLD, NEW = os.path.join(RES, "q25_const_lengths"), os.path.join(RES, "q25_const_lengths_synra298")
MERGED, OUTD = os.path.join(RES, "q35_fig3_data"), os.path.join(RES, "q35_figure3")
PUBLISHED = os.path.join(os.path.dirname(IVL), "Figures", "Figure3.png")
SRC = os.path.join(HERE, "q32_figure2_trunc.py")
os.makedirs(OUTD, exist_ok=True)
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))

if args.mode == "new":
    os.makedirs(MERGED, exist_ok=True)
    # lengths and ACFs depend on case and kernel only (same noise seeds): the new run must reproduce them exactly
    Lo, Ln = rd(os.path.join(OLD, "q21_lengths.csv")), rd(os.path.join(NEW, "q21_lengths.csv"))
    key = lambda r: (r["case"], r["kernel"])
    lo = {key(r): r for r in Lo}
    assert len(Ln) == 20 and all(lo[key(r)] == r for r in Ln), "q21_lengths of the new run differ from the old ones"
    for c in range(1, 11):
        for k in ("aniso_const", "iso_const"):
            fn = f"acfZ_case{c}_{k}.npy"
            assert np.array_equal(np.load(os.path.join(OLD, fn)), np.load(os.path.join(NEW, fn))), fn
            shutil.copyfile(os.path.join(OLD, fn), os.path.join(MERGED, fn))
    shutil.copyfile(os.path.join(OLD, "q21_lengths.csv"), os.path.join(MERGED, "q21_lengths.csv"))
    So, Sn = rd(os.path.join(OLD, "q21_sampler.csv")), rd(os.path.join(NEW, "q21_sampler.csv"))
    repl = lambda r: r["engine"] == "synra" and r["kernel"] in ("aniso_const", "iso_const")
    assert len(Sn) == 20 and all(repl(r) for r in Sn) and all(abs(float(r["f"]) - 2.9764) < 1e-4 for r in Sn)
    Sm = [r for r in So if not repl(r)] + Sn
    assert len(Sm) == len(So) == 120, (len(Sm), len(So))
    with open(os.path.join(MERGED, "q21_sampler.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(So[0].keys())); w.writeheader(); w.writerows(Sm)
    DATA = MERGED
else:
    DATA = OLD


def LAYOUT(patch):
    """Extra patches on the q25_figure2_A2.py source, applied by q32 before its exec."""
    patch('DATA = os.path.join(IVL, "results", "q25_const_lengths")', "DATA = DATA_Q35")
    if args.mode == "new":
        patch('fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(7.09, 5.45)); g["bars"](a, b)',
              'fig = plt.figure(figsize=(7.09, 5.45)); gs = fig.add_gridspec(2, 4); a = fig.add_subplot(gs[0, 1:3])\n'
              'c = fig.add_subplot(gs[1, :2]); d = fig.add_subplot(gs[1, 2:]); b = fig.add_axes([0, 0, 0.01, 0.01]); g["bars"](a, b); b.remove()\n'
              'a.set_title("(a) folding before screening")')   # critical review 1.3 (2026-10-24): "diffeomorphism" is not a term of the text
        patch('"(c) cranio-caudal: model vs landmark error"', '"(b) cranio-caudal: model vs landmark error"')
        patch('"(d) in-plane: model vs landmark error"', '"(c) in-plane: model vs landmark error"')


s = open(SRC, encoding="utf-8").read()


def patch32(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old)
    s = s.replace(old, new)


patch32('OUTD = os.path.join(IVL, "results", "q32_figure2")', "OUTD = OUTD_Q35")
patch32('g = {"__file__": SRC, "__name__": "q32", "OUTD": OUTD, "MODE": args.mode,',
        'LAYOUT(patch)\ng = {"__file__": SRC, "__name__": "q32", "OUTD": OUTD, "MODE": MODE_Q35, "DATA_Q35": DATA_Q35,')
sys.argv = [SRC, "--mode", "trunc"]
g32 = {"__file__": SRC, "__name__": "q35", "OUTD_Q35": OUTD, "MODE_Q35": args.mode, "DATA_Q35": DATA, "LAYOUT": LAYOUT}
exec(compile(s, SRC, "exec"), g32)

png = os.path.join(OUTD, "Figure2_%s.png" % args.mode)
if args.mode == "control":
    from PIL import Image
    a = np.asarray(Image.open(png).convert("RGB")).astype(int); b = np.asarray(Image.open(PUBLISHED).convert("RGB")).astype(int)
    nd = int((a != b).any(axis=2).sum()) if a.shape == b.shape else -1
    print("[control] four panels, old data vs published Figure3.png: shape %s vs %s, differing pixels %d" % (a.shape, b.shape, nd))
    assert nd == 0
else:
    S = rd(os.path.join(MERGED, "q21_sampler.csv"))
    m = lambda col, k: np.mean([float(r[col]) for r in S if r["engine"] == "synra" and r["kernel"] == k and r["case"] != "8"])
    print("[check] SyNRA (9 cases): folding K %.4f %% iso %.4f %%, full amplitude K %.0f %% iso %.0f %%"
          % (100 * m("first_folding_mean", "aniso_const"), 100 * m("first_folding_mean", "iso_const"),
             100 * m("frac_r0", "aniso_const"), 100 * m("frac_r0", "iso_const")))
    assert abs(100 * m("first_folding_mean", "aniso_const") - 0.0073) < 0.0001 and m("frac_r0", "aniso_const") == 1.0
print("[out]", png)
