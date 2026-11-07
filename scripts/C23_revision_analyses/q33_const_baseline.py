# -*- coding: utf-8 -*-
"""IV round, internal critical review 2.2 (2026-10-23): constant-sigma baseline for Table 2.
Question: does sigma_h calibrate better than a sigma that is the same at every landmark of a cell (sigma = f * 1)?
Protocol fixed before seeing the result, identical to Table 2 / q31 (calibrate.fit_c_grid / kfold_calibrate, seeds 0..49, bin 0.25 mm,
cases with RMSE TRE <= 10 mm, natural case order), on the norm (TRE) and per direction (|e_d|, 3 samples per landmark):
  sigma_h  : sigma_p_mm_raw at the landmark (as published);
  constant : sigma = 1 at every landmark (f then equals the training RMSE; all landmarks fall in one bin, LECE = |RMSE_test - f|).
Same seed -> same KFold split for both predictors (same case list), so the comparison is paired fold by fold:
  d = LECE_const - LECE_h  (> 0: sigma_h better); reported: mean +- s.d. of d over the folds, fraction of folds with d > 0.
CONTROL (abort if it fails): sigma_h reproduces q31_lece_perdir_all.json exactly (norm = Table 2, per direction = Table 2 red column).
(q27.tex_table2() no longer parses Table 2 since the per-direction column was added, so the reference is the q31 output.)
Usage (venv): python q33_const_baseline.py [--datasets all] [--workers 24]   Output: ../results/q33_const_baseline_<datasets>.json"""
import os, sys, json, argparse
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import q27_table2_compare as T2
from q27_table2_compare import CaseLandmarks, SEEDS
from q31_lece_perdir import CELLS, find_qa_dirs, load_cell

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q31 = os.path.join(HERE, "results", "q31_lece_perdir_all.json")
DS = ["Lung-DIR-QA", "Abdominal-DIR-QA", "Liver-DIR-QA", "DIR-LAB"]; EN = ["SyNRA", "ConvexAdam", "uniGradICON", "Elastix"]


def const(cases):
    return [CaseLandmarks(c.case_id, c.tre_mm, np.ones_like(c.sigma_raw)) for c in cases]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", default="all"); ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    dirs = dict(CELLS); dirs.update(find_qa_dirs()); assert len(dirs) == 16, sorted(dirs)
    keep = sorted([k for k in dirs if a.datasets == "all" or k[0] in a.datasets.split(",")], key=lambda x: (DS.index(x[0]), EN.index(x[1])))
    q31 = json.load(open(Q31))["cells"]
    cells = {}
    for k in keep:
        c = load_cell(dirs[k])
        cells[k] = {(v, "h"): c[v] for v in ("norm", "perdir")}
        cells[k].update({(v, "c"): const(c[v]) for v in ("norm", "perdir")})
    tasks = [(k, key, s) for k in keep for key in cells[k] for s in SEEDS]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        outs = list(ex.map(T2.one_rep, [(cells[k][key], s) for k, key, s in tasks], chunksize=4))
    folds = {}
    for (k, key, s), o in zip(tasks, outs):
        folds.setdefault((k, key), []).extend(o)
    res, ok = {}, True
    print(f"{'cell':30s} {'n':>3s} | {'conv':6s} {'LECE sigma_h':>14s} {'LECE const':>14s} {'d=const-h':>15s} {'d>0':>5s} | control")
    for k in keep:
        r = {}
        for v in ("norm", "perdir"):
            h, c = np.array(folds[(k, (v, "h"))]), np.array(folds[(k, (v, "c"))]); d = c - h
            assert len(h) == len(c)
            r[v] = dict(n=len(cells[k][(v, "h")]), folds=len(h), lece_h=float(h.mean()), sd_h=float(h.std(ddof=1)), lece_c=float(c.mean()),
                        sd_c=float(c.std(ddof=1)), d_mean=float(d.mean()), d_sd=float(d.std(ddof=1)), frac_h_better=float((d > 0).mean()))
        q = q31[f"{k[0]}|{k[1]}"]
        c_ok = all(r[v]["n"] == q[v]["n"] and abs(r[v]["lece_h"] - q[v]["lece"]) < 1e-9 and abs(r[v]["sd_h"] - q[v]["sd"]) < 1e-9
                   for v in ("norm", "perdir")) and q["control"]
        ok &= c_ok
        res[f"{k[0]}|{k[1]}"] = {**r, "control": c_ok}
        for v in ("norm", "perdir"):
            x = r[v]
            print(f"{(k[0] + ' ' + k[1]) if v == 'norm' else '':30s} {x['n']:3d} | {v:6s} {x['lece_h']:6.3f}+-{x['sd_h']:.3f} {x['lece_c']:6.3f}+-{x['sd_c']:.3f}"
                  f" {x['d_mean']:+7.3f}+-{x['d_sd']:.3f} {x['frac_h_better']:5.2f} | {'PASS' if c_ok else 'FAIL'}", flush=True)
    assert ok, "CONTROL FAILED: sigma_h does not reproduce q31 (Table 2)"
    for v in ("norm", "perdir"):
        nb = sum(x[v]["d_mean"] > 0 for x in res.values())
        print(f"{v}: sigma_h better than constant (mean paired d > 0) in {nb}/{len(res)} cells")
    out = os.path.join(HERE, "results", f"q33_const_baseline_{a.datasets.replace(',', '_')}.json")
    json.dump({"control_pass": True, "cells": res}, open(out, "w"), indent=1); print("->", out)


if __name__ == "__main__":
    main()
