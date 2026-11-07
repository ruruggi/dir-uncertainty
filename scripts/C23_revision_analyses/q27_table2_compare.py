# -*- coding: utf-8 -*-
"""IV round, P1L43 option (B): Table 2 (q <= 10 mm) with and without the 99th-percentile clip of R.
Input: the per-pair outputs of q27_rclip_check.py (local results/q27_rclip/<rel>/<case>/rclip.npz and the cluster SyNRA cells copied to
results/q27_rclip/Results_SynRA/{Lung,Abdomen,Liver}/<case>/). For each of the 16 cells and each variant (099 = published, none):
cases with RMS TRE <= 10 mm in natural order; f fitted on all of them (calibrate.fit_c_grid on the unscaled sigma, f' = f/sqrt3, as
q22_table23_v2.py); 5-fold LECE repeated over seeds 0..49 (calibrate.kfold_calibrate, held-out grid LECE; mean +- s.d. over folds);
rho_pool = Spearman rho of sigma_p and TRE on the pooled landmarks.
CONTROL: variant 099 must reproduce Table 2 of the manuscript (f, LECE mean and s.d., rho_pool, to the printed decimals) in every cell.
Output results/q27_table2_compare.json. Usage (venv): python q27_table2_compare.py [--workers 24]"""
from dirunc_paths import LIB  # data roots, see README
import os, sys, types, json, re, argparse
from concurrent.futures import ProcessPoolExecutor
III = LIB  # was the III Scripts folder (lib/ + scripts/, see dirunc_paths)
PS = LIB + "/validation"; sys.path.insert(0, III); sys.path.insert(0, PS)
if "validation" not in sys.modules:
    _v = types.ModuleType("validation"); _v.__path__ = [PS]; sys.modules["validation"] = _v
import numpy as np
from scipy.stats import spearmanr
from validation.calibrate import CaseLandmarks, fit_c_grid, stack_cases, kfold_calibrate

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results", "q27_rclip"); TEX = os.path.join(os.path.dirname(HERE), "Manuscript_DVFuncertainty_2.tex")
OUT = os.path.join(HERE, "results", "q27_table2_compare.json")
SQ3 = float(np.sqrt(3.0)); BIN = 0.25; SEEDS = list(range(50))
DATASETS = ["Lung-DIR-QA", "Abdominal-DIR-QA", "Liver-DIR-QA", "DIR-LAB"]; ENGINES = ["SyNRA", "ConvexAdam", "uniGradICON", "Elastix"]
DIRS = {  # cell -> output folder of q27_rclip_check (local rel dirs as in build_v11_data; SyNRA QA from the cluster)
    ("Lung-DIR-QA", "SyNRA"): "Results_SynRA/Lung", ("Abdominal-DIR-QA", "SyNRA"): "Results_SynRA/Abdomen",
    ("Liver-DIR-QA", "SyNRA"): "Results_SynRA/Liver"}
natkey = lambda c: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", c)]


def one_rep(args):
    cases, seed = args
    return [float(fl.lece_test_grid) for fl in kfold_calibrate(cases, n_splits=min(5, len(cases)), seed=seed, bin_width=BIN)]


def find_dirs():
    """map the 13 local cells to their output folders: q27_rclip_check writes <rel dir of the results root>/<case>"""
    m = dict(DIRS)
    keys = {"SyNRA": "synra", "ConvexAdam": ("convexadam", "adam"), "uniGradICON": "unigradicon", "Elastix": "elastix"}
    dsk = {"Lung-DIR-QA": "Lung-DIR-QA", "Abdominal-DIR-QA": "AbdominalDIR-QA", "Liver-DIR-QA": "Liver-DIR-QA", "DIR-LAB": "Castillo-DIR-LAB"}
    for root, dirs, files in os.walk(RES):
        if "rclip.json" in files:
            rel = os.path.relpath(os.path.dirname(root), RES).replace("\\", "/")
            if rel.startswith("Results_SynRA/") or rel.startswith("_"):
                continue
            ds = next(d for d, k in dsk.items() if rel.startswith(k))
            low = rel.lower().split("/")[-1]
            eng = next(e for e, k in keys.items() if any(x in low for x in ((k,) if isinstance(k, str) else k)))
            prev = m.get((ds, eng))
            assert prev in (None, rel), (ds, eng, prev, rel)
            m[(ds, eng)] = rel
    assert len(m) == 16, sorted(m)
    return m


def tex_table2():
    s = open(TEX, encoding="utf-8").read(); s = s[s.index(r"\label{tab:calibration}"):]; s = s[:s.index(r"\end{tabular}")]
    rows, ds = {}, None
    for line in s.splitlines():
        m = re.match(r"\s*([A-Za-z\-]*)\s*&\s*(SyNRA|ConvexAdam|uniGradICON|Elastix)\s*&\s*(\d+)\s*&\s*[\d\\,]+\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*\$\\pm\$\s*([\d.]+)\s*&\s*\$([+\-][\d.]+)\$", line)
        if m:
            ds = m.group(1) or ds
            rows[(ds, m.group(2))] = dict(n=int(m.group(3)), f=float(m.group(4)), lece=float(m.group(5)), sd=float(m.group(6)), rho=float(m.group(7)))
    assert len(rows) == 16, len(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=24); a = ap.parse_args()
    dirs, ref = find_dirs(), tex_table2()
    cells = {}
    for (ds, eng), rel in dirs.items():
        base = os.path.join(RES, rel); cases = {"099": [], "none": []}
        for c in sorted(os.listdir(base), key=natkey):
            fz = os.path.join(base, c, "rclip.npz")
            if not os.path.isfile(fz):
                continue
            z = np.load(fz); tre = np.asarray(z["tre_mm"], np.float64)
            if np.sqrt(np.mean(tre ** 2)) <= 10.0:
                for v in cases:
                    cases[v].append(CaseLandmarks(c, tre, np.asarray(z["sigma_p_" + v], np.float64)))
        cells[(ds, eng)] = cases
    tasks = [((ds, eng, v), s) for (ds, eng), cs in cells.items() for v in cs for s in SEEDS]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        outs = list(ex.map(one_rep, [(cells[(k[0], k[1])][k[2]], s) for k, s in tasks], chunksize=4))
    folds = {}
    for (k, s), o in zip(tasks, outs):
        folds.setdefault(k, []).extend(o)
    res, ok = {}, True
    print(f"{'cell':32s} {'n':>3s} | {'f 099':>6s} {'none':>6s} | {'LECE 099':>13s} {'none':>13s} | {'rho 099':>7s} {'none':>7s} | control")
    for ds in DATASETS:
        for eng in ENGINES:
            r = {}
            for v in ("099", "none"):
                cs = cells[(ds, eng)][v]; tre, sig = stack_cases(cs); f, _ = fit_c_grid(tre, sig, bin_width=BIN)
                lf = np.array(folds[(ds, eng, v)])
                r[v] = dict(n=len(cs), f=f / SQ3, lece=float(lf.mean()), sd=float(lf.std(ddof=1)), rho=float(spearmanr(sig, tre)[0]))
            t = ref[(ds, eng)]; o = r["099"]
            c_ok = (o["n"] == t["n"] and round(o["f"], 2) == t["f"] and round(o["lece"], 2) == t["lece"] and round(o["sd"], 2) == t["sd"]
                    and round(o["rho"], 3) == t["rho"])
            ok &= c_ok
            res[f"{ds}|{eng}"] = {**r, "table2": t, "control": c_ok}
            print(f"{ds + ' ' + eng:32s} {o['n']:3d} | {o['f']:6.3f} {r['none']['f']:6.3f} | {o['lece']:.4f}+-{o['sd']:.3f} {r['none']['lece']:.4f}+-{r['none']['sd']:.3f}"
                  f" | {o['rho']:+.4f} {r['none']['rho']:+.4f} | {'PASS' if c_ok else 'FAIL'}", flush=True)
    d = {k: (abs(v["none"]["f"] - v["099"]["f"]), abs(v["none"]["lece"] - v["099"]["lece"]), abs(v["none"]["rho"] - v["099"]["rho"])) for k, v in res.items()}
    print(f"\nCONTROL (099 = Table 2): {'PASS' if ok else 'FAIL'}; max |delta| none vs 099: f {max(x[0] for x in d.values()):.4f}, "
          f"LECE {max(x[1] for x in d.values()):.4f} mm, rho {max(x[2] for x in d.values()):.4f}")
    json.dump({"control_pass": bool(ok), "cells": res}, open(OUT, "w"), indent=1); print("->", OUT)


if __name__ == "__main__":
    main()
