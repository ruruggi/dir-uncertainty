# -*- coding: utf-8 -*-
"""IV round, internal critical review 2.2, step 2 (2026-10-23): proper scoring rule, sigma_h vs constant sigma.
q33 showed that a constant sigma has a 5-fold LECE <= sigma_h in 14/16 cells; the binned LECE is minimised by any well-calibrated
constant, so it cannot show whether the variation of sigma_h carries information. The held-out log-likelihood can.
Model (as calibrate.fit_c_mle): isotropic 3-D Gaussian error with s.d. f*sigma per axis -> TRE ~ Maxwell(scale f*sigma);
NLL per landmark = 3 log(f sigma) + TRE^2 / (2 f^2 sigma^2) - log(TRE^2) + const (the per-direction Gaussian differs only by a
term that does not depend on sigma, so the difference between predictors is the same in both conventions).
f fitted on the training cases by MLE, closed form f^2 = mean(TRE^2 / sigma^2) / 3 (CONTROL: equals calibrate.fit_c_mle).
Same protocol and folds as Table 2 / q33 (sklearn KFold(min(5, n), shuffle, seeds 0..49) on the retained cases, q <= 10 mm).
Paired per fold: dNLL = NLL_const - NLL_h (nats per landmark; > 0: sigma_h better). Protocol fixed before seeing the result.
Usage (venv): python q34_nll_baseline.py [--workers 24]   Output: ../results/q34_nll_baseline.json"""
import os, sys, json, argparse
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.model_selection import KFold
import q27_table2_compare as T2                     # sets the III paths for validation.calibrate
from validation.calibrate import fit_c_mle, stack_cases
from q27_table2_compare import SEEDS
from q31_lece_perdir import CELLS, find_qa_dirs, load_cell

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = ["Lung-DIR-QA", "Abdominal-DIR-QA", "Liver-DIR-QA", "DIR-LAB"]; EN = ["SyNRA", "ConvexAdam", "uniGradICON", "Elastix"]


def f_mle(r, s):
    return float(np.sqrt(np.mean((r / s) ** 2) / 3.0))


def nll(r, s, f):                                   # Maxwell NLL per landmark, sigma-independent constants dropped
    a = f * s
    return float(np.mean(3 * np.log(a) + r ** 2 / (2 * a ** 2)))


def one_seed(args):
    cases, seed = args
    out = []
    for tr, te in KFold(n_splits=min(5, len(cases)), shuffle=True, random_state=seed).split([c.case_id for c in cases]):
        r_tr, s_tr = stack_cases([cases[i] for i in tr]); r_te, s_te = stack_cases([cases[i] for i in te])
        one_tr, one_te = np.ones_like(s_tr), np.ones_like(s_te)
        fh, fc = f_mle(r_tr, s_tr), f_mle(r_tr, one_tr)
        out.append((nll(r_te, s_te, fh), nll(r_te, one_te, fc), fh, fc))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=24); a = ap.parse_args()
    dirs = dict(CELLS); dirs.update(find_qa_dirs()); assert len(dirs) == 16, sorted(dirs)
    keep = sorted(dirs, key=lambda x: (DS.index(x[0]), EN.index(x[1])))
    cells = {k: load_cell(dirs[k])["norm"] for k in keep}
    for k in keep:                                  # CONTROL: closed form = published fit_c_mle; sigma_h > 0
        r, s = stack_cases(cells[k]); assert s.min() > 0 and r.min() > 0, k
        cm, _ = fit_c_mle(r, s); assert abs(cm - f_mle(r, s)) / cm < 1e-3, (k, cm, f_mle(r, s))
    tasks = [(k, s) for k in keep for s in SEEDS]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        outs = list(ex.map(one_seed, [(cells[k], s) for k, s in tasks], chunksize=4))
    folds = {}
    for (k, s), o in zip(tasks, outs):
        folds.setdefault(k, []).extend(o)
    res = {}
    print(f"{'cell':30s} {'n':>3s} {'folds':>5s} | {'NLL sigma_h':>11s} {'NLL const':>10s} | {'dNLL=const-h':>15s} {'d>0':>5s}")
    for k in keep:
        x = np.array(folds[k]); d = x[:, 1] - x[:, 0]
        res[f"{k[0]}|{k[1]}"] = dict(n=len(cells[k]), folds=len(d), nll_h=float(x[:, 0].mean()), nll_c=float(x[:, 1].mean()),
                                     d_mean=float(d.mean()), d_sd=float(d.std(ddof=1)), frac_h_better=float((d > 0).mean()),
                                     f_h_mean=float(x[:, 2].mean()), f_c_mean=float(x[:, 3].mean()))
        v = res[f"{k[0]}|{k[1]}"]
        print(f"{k[0] + ' ' + k[1]:30s} {v['n']:3d} {v['folds']:5d} | {v['nll_h']:11.4f} {v['nll_c']:10.4f} | {v['d_mean']:+8.4f}+-{v['d_sd']:.4f} {v['frac_h_better']:5.2f}", flush=True)
    nb = sum(v["d_mean"] > 0 for v in res.values())
    print(f"\nCONTROL PASS (closed-form MLE = fit_c_mle). sigma_h better than constant (mean paired dNLL > 0) in {nb}/{len(res)} cells")
    out = os.path.join(HERE, "results", "q34_nll_baseline.json")
    json.dump({"control_pass": True, "unit": "nats per landmark", "cells": res}, open(out, "w"), indent=1); print("->", out)


if __name__ == "__main__":
    main()
