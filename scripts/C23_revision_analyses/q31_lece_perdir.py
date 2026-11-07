# -*- coding: utf-8 -*-
"""IV round, P9L12 (R2: "in Smolders the LECE is per direction, maybe this affects the results"), 2026-10-18.
Smolders et al. (2023) evaluate the landmarks per direction: README of github.com/AndreasSmolders/DIRUncertainty (Validation > Landmarks):
"calculate the error in the x, y and z direction individually ... 3000 landmarks with each 3 directions (i.e. 9000 data points)", bins of
0.25 mm, RMSE per bin vs mean predicted sigma. We evaluate on the norm (TRE vs sigma_p = sqrt(3) sigma_h).
This script recomputes Table 2's 5-fold LECE in Smolders' convention, objective fixed before seeing the result:
  per-direction: each landmark gives 3 samples (|e_d|, sigma_d), e_d = predicted - true moving landmark position along axis d (mm),
  sigma_d = f * sigma_h(f=1) at the landmark (sigma_h is isotropic: the same value on the three axes); f fitted by minimising the per-direction LECE (same grid search and 5-fold protocol
  as Table 2: calibrate.fit_c_grid / kfold_calibrate, seeds 0..49, bin 0.25 mm, cases with RMSE TRE <= 10 mm, natural case order).
CONTROL (abort if it fails): the norm variant (TRE, sigma_p) reproduces Table 2 (n, f, LECE mean and s.d., rho_pool) in every cell run.
Also checked per case: |norm(e) - TRE| < 1e-9.
Data: published landmarks.npz (sigma_p_mm_raw, lm_moving_world_pred, lm_moving_world, tre_mm) under
<DATASETS>/_sigma_BL_20260927/<cell dir>/<case>/ (13 local cells + Results_SynRA/{Lung,Abdomen,Liver}).
Usage (venv): python q31_lece_perdir.py [--datasets DIR-LAB] [--workers 24]   (default: DIR-LAB only; --datasets all for the 16 cells)
Output: ../results/q31_lece_perdir_<datasets>.json"""
from dirunc_paths import DATASETS  # data roots, see README
import os, sys, re, json, argparse
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import spearmanr
import q27_table2_compare as T2          # III paths, CaseLandmarks, fit_c_grid, stack_cases, kfold_calibrate, tex_table2, one_rep
from q27_table2_compare import CaseLandmarks, fit_c_grid, stack_cases, SQ3, BIN, SEEDS, natkey

REF = f"{DATASETS}/_sigma_BL_20260927"
CELLS = {  # (dataset, engine) -> folder under REF
    ("DIR-LAB", "SyNRA"): "Castillo-DIR-LAB/Results_synra_procrustes", ("DIR-LAB", "ConvexAdam"): "Castillo-DIR-LAB/Results_convexadam_procrustes",
    ("DIR-LAB", "uniGradICON"): "Castillo-DIR-LAB/Results_unigradicon_procrustes", ("DIR-LAB", "Elastix"): "Castillo-DIR-LAB/Results_elastix_procrustes",
    ("Lung-DIR-QA", "SyNRA"): "Results_SynRA/Lung", ("Abdominal-DIR-QA", "SyNRA"): "Results_SynRA/Abdomen", ("Liver-DIR-QA", "SyNRA"): "Results_SynRA/Liver"}
LOW, HIGH = 0.41, 1.34                    # Smolders et al. (2023), DIR-LAB, B-spline/demons/Velocity


def find_qa_dirs():
    """the 9 local QA cells (ConvexAdam, uniGradICON, Elastix x Lung/Abdominal/Liver): any folder under REF/<dataset> holding caseN/landmarks.npz"""
    m = {}
    dsk = {"Lung-DIR-QA": "Lung-DIR-QA", "Abdominal-DIR-QA": "AbdominalDIR-QA", "Liver-DIR-QA": "Liver-DIR-QA"}
    eng = {"ConvexAdam": ("convexadam", "adam"), "uniGradICON": ("unigradicon",), "Elastix": ("elastix",)}
    for ds, k in dsk.items():
        for root, dirs, files in os.walk(os.path.join(REF, k)):
            if "landmarks.npz" in files:
                rel = os.path.relpath(os.path.dirname(root), REF).replace("\\", "/"); low = rel.lower().split("/")[-1]
                e = [x for x, keys in eng.items() if any(t in low for t in keys) and "synra" not in low]
                if e:
                    prev = m.get((ds, e[0])); assert prev in (None, rel), (ds, e[0], prev, rel); m[(ds, e[0])] = rel
    return m


def load_cell(rel):
    norm, perdir = [], []
    base = os.path.join(REF, rel)
    for c in sorted(os.listdir(base), key=natkey):
        fz = os.path.join(base, c, "landmarks.npz")
        if not os.path.isfile(fz):
            continue
        z = np.load(fz, allow_pickle=True)
        tre = np.asarray(z["tre_mm"], np.float64); sp = np.asarray(z["sigma_p_mm_raw"], np.float64)
        e = np.asarray(z["lm_moving_world_pred"], np.float64) - np.asarray(z["lm_moving_world"], np.float64)
        assert np.abs(np.linalg.norm(e, axis=1) - tre).max() < 1e-9, (rel, c)
        if np.sqrt(np.mean(tre ** 2)) <= 10.0:
            norm.append(CaseLandmarks(c, tre, sp))
            # sigma_p_mm_raw is sigma_h at f = 1 per component (Table 2: sigma_p = sqrt3 f raw, f = g / sqrt3), so sigma_d = f * raw
            perdir.append(CaseLandmarks(c, np.abs(e).reshape(-1), np.repeat(sp, 3)))   # 3 samples per landmark
    return {"norm": norm, "perdir": perdir}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", default="DIR-LAB"); ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    dirs = dict(CELLS); dirs.update(find_qa_dirs())
    assert len(dirs) == 16, sorted(dirs)
    keep = [k for k in dirs if a.datasets == "all" or k[0] in a.datasets.split(",")]
    ref = T2.tex_table2()
    cells = {k: load_cell(dirs[k]) for k in keep}
    tasks = [((k, v), s) for k in keep for v in ("norm", "perdir") for s in SEEDS]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        outs = list(ex.map(T2.one_rep, [(cells[k][v], s) for (k, v), s in tasks], chunksize=4))
    folds = {}
    for ((k, v), s), o in zip(tasks, outs):
        folds.setdefault((k, v), []).extend(o)
    res, ok = {}, True
    print(f"{'cell':30s} {'n':>3s} | norm: {'f':>6s} {'LECE5':>13s} | per-direction: {'f':>6s} {'LECE5':>13s} | in [{LOW}, {HIGH}]: norm perdir | control")
    for k in sorted(keep, key=lambda x: (["Lung-DIR-QA", "Abdominal-DIR-QA", "Liver-DIR-QA", "DIR-LAB"].index(x[0]),
                                          ["SyNRA", "ConvexAdam", "uniGradICON", "Elastix"].index(x[1]))):
        r = {}
        for v in ("norm", "perdir"):
            cs = cells[k][v]; tre, sig = stack_cases(cs); g, _ = fit_c_grid(tre, sig, bin_width=BIN)
            lf = np.array(folds[(k, v)])
            r[v] = dict(n=len(cs), f=float(g / SQ3) if v == "norm" else float(g), lece=float(lf.mean()), sd=float(lf.std(ddof=1)),
                        rho=float(spearmanr(sig, tre)[0]), within=bool(lf.mean() <= HIGH))
        t = ref[k]; o = r["norm"]
        c_ok = (o["n"] == t["n"] and round(o["f"], 2) == t["f"] and round(o["lece"], 2) == t["lece"] and round(o["sd"], 2) == t["sd"]
                and round(o["rho"], 3) == t["rho"])
        ok &= c_ok
        res[f"{k[0]}|{k[1]}"] = {**r, "table2": t, "control": c_ok}
        p = r["perdir"]
        print(f"{k[0] + ' ' + k[1]:30s} {o['n']:3d} | {o['f']:6.3f} {o['lece']:.3f}+-{o['sd']:.3f} | {p['f']:6.3f} {p['lece']:.3f}+-{p['sd']:.3f}"
              f" |   {'Y' if o['within'] else 'n'}     {'Y' if p['within'] else 'n'}   | {'PASS' if c_ok else 'FAIL'}", flush=True)
    assert ok, "CONTROL FAILED: the norm variant does not reproduce Table 2"
    nn = sum(v["norm"]["within"] for v in res.values()); npd = sum(v["perdir"]["within"] for v in res.values())
    print(f"\nCONTROL PASS. Cells with 5-fold LECE <= {HIGH} mm: norm {nn}/{len(res)}, per-direction {npd}/{len(res)}")
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", f"q31_lece_perdir_{a.datasets.replace(',', '_')}.json")
    json.dump({"control_pass": True, "range": [LOW, HIGH], "cells": res}, open(out, "w"), indent=1); print("->", out)


if __name__ == "__main__":
    main()
