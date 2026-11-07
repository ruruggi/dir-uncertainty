# -*- coding: utf-8 -*-
"""
q22_table23_v2.py -- step 1 of the 2026-09-29 order (Memoria/ordine_ricalcoli_20260929.md): Table 2 and Table 3 rebuilt with
  R43  sigma_p of the heuristic sigma = sqrt(3) * sigma_h (Euclidean norm of the isotropic per-direction vector, .tex r. 210)
       -> f' = f / sqrt(3) in every cell; LECE, rho, folds unchanged; sampler unchanged (f' * sigma_h per component).
       IMPLEMENTATION: the fit runs on the UNSCALED sigma_h and f' = f / sqrt3 is taken afterwards (identical model: f' * sqrt3 * sigma_h
       = f * sigma_h). Fitting on sqrt3 * sigma_h instead moves f by up to 10 % and LECE by up to 0.03 mm, because the log grid of
       fit_c_grid is not scale invariant and the LECE landscape is rugged (see _out_q22_table23_v2/_control_scaled_fit_FAILED.log).
  R45  repeated k-fold: 5-fold (n_splits = min(5, n)) repeated over N_REP seeds, cases in NATURAL (numeric) order in every cell and
       both arms; reported mean +- s.d. over all held-out folds (and, separately, s.d. of the per-repetition means).
  R44  f of the Figure 3 heuristic arm = mean of the four Elastix f' at q <= 5 (rule recovered on 2026-09-29; submitted: 3.862).
Inputs: sigma_BL cells (<DATASETS>/_sigma_BL_20260927, as step 4), sigma_DL corrected (./_out_q22_sigmaDL, Tnew_IC_Snet).
Controls: (1) f (unscaled fit) = step 4 / table3.json; (2) LECE global identical; (3) seed 42, numeric order,
Table 3: sigma_h 2.651/0.351/1.245/0.707 and sigma_DL 2.607/0.533/1.485/0.828 (audit rev2, C1); (4) seed 42 lexicographic order =
table3.json sigma_h (3.117/0.334/1.297/0.511). Read-only on inputs. Output ./_out_q22_table23_v2/ (table23_v2.json, Table2_rows_v2.tex,
Table3_rows_v2.tex). Usage (venv): python q22_table23_v2.py [--reps 50] [--workers 40]
"""
from dirunc_paths import DATASETS, LIB  # data roots, see README
import os, sys, types, json, glob, re, time, argparse
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = LIB; PS = LIB + "/validation"
sys.path.insert(0, SCRIPTS); sys.path.insert(0, PS)
if "validation" not in sys.modules:
    _v = types.ModuleType("validation"); _v.__path__ = [PS]; sys.modules["validation"] = _v
import numpy as np
from validation.build_v11_data import ENGINE_PATHS
from validation.calibrate import load_all_landmarks, CaseLandmarks, fit_c_grid, stack_cases, kfold_calibrate
from validation.lece import compute_lece

SQ3 = float(np.sqrt(3.0)); BIN = 0.25
DS_ROOT = f"{DATASETS}"; BL_ROOT = f"{DS_ROOT}/_sigma_BL_20260927"
DL = os.path.join(HERE, "_out_q22_sigmaDL"); OUT = os.path.join(HERE, "_out_q22_table23_v2")
DATASETS = ["Lung-DIR-QA", "Abdominal-DIR-QA", "Liver-DIR-QA", "DIR-LAB"]
ENGINES = ["SyNRA", "ConvexAdam", "uniGradICON", "Elastix"]
ENG_KEY = {"SyNRA": "synra", "ConvexAdam": "convexadam", "uniGradICON": "unigradicon", "Elastix": "elastix"}
bl = lambda p: os.path.join(BL_ROOT, os.path.relpath(p, DS_ROOT)).replace("\\", "/")
natkey = lambda c: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", c.case_id)]


def scaled(cases, k):
    return [CaseLandmarks(c.case_id, c.tre_mm, k * c.sigma_raw) for c in cases]


def dl_cases(key):
    out = []
    for f in glob.glob(os.path.join(DL, f"{key}_case*.npz")):
        r = np.load(f); cid = "case" + f.split("case")[-1][:-4]
        tre = np.asarray(r["tre_new"], np.float64).ravel(); sig = np.asarray(r["sp_C_net"], np.float64).ravel()
        keep = np.isfinite(tre) & np.isfinite(sig) & (sig > 1e-9) & (tre >= 0)
        out.append(CaseLandmarks(cid, tre[keep], sig[keep]))
    return out


def one_rep(args):
    """one k-fold repetition -> list of held-out LECE (grid f)"""
    cases, seed = args
    n = min(5, len(cases))
    if n < 2:
        return []
    return [float(fl.lece_test_grid) for fl in kfold_calibrate(cases, n_splits=n, seed=seed, bin_width=BIN)]


def cell_global(cases):
    tre, sig = stack_cases(cases); f, _ = fit_c_grid(tre, sig, bin_width=BIN); lg, _ = compute_lece(tre, f * sig, bin_width=BIN)
    return float(f), float(lg)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=50); ap.add_argument("--workers", type=int, default=40)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); t0 = time.time()
    seeds = list(range(a.reps))
    ref4 = json.load(open(os.path.join(HERE, "_out_q22_step4", "step4.json")))["table2_new"]
    ref3 = json.load(open(os.path.join(HERE, "_out_q22_table3", "table3.json")))["results"]
    ok_all = True

    def check(tag, ok, detail=""):
        nonlocal ok_all
        ok_all &= bool(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {tag} {detail}", flush=True)

    # ---- build the cells (natural order; sigma_h scaled by sqrt3) -------------------------------------------------------------
    cells = {}                                                      # (table, ds, eng, q) -> cases
    for eng in ENGINES:
        for ds in DATASETS:
            cases = sorted(load_all_landmarks(bl(ENGINE_PATHS[eng]["datasets"][ds])), key=natkey)
            rmse = {c.case_id: float(np.sqrt(np.mean(c.tre_mm ** 2))) for c in cases}
            for q in (10.0, 5.0):
                cells[("T2", ds, eng, q)] = [c for c in cases if rmse[c.case_id] <= q]                   # unscaled (see header)
            if ds == "DIR-LAB":
                cells[("T3", "DIR-LAB", eng, "sigma_h")] = list(cases)                             # all 10 cases, as Table 3
                cells[("T3", "DIR-LAB", eng, "sigma_DL")] = sorted(dl_cases(ENG_KEY[eng]), key=natkey)   # norm already, no sqrt3
    # ---- controls 1-2: global f' and LECE -----------------------------------------------------------------------------------
    print("Controls 1-2: f' * sqrt3 = f (step4 / table3), LECE global identical", flush=True)
    res = {}
    for key, cases in cells.items():
        tab, ds, eng, q = key
        if not cases:
            res[key] = {"n": 0}; continue
        f, lg = cell_global(cases)
        if tab == "T2":
            r = ref4[f"{ds}|{eng}"][f"{q:.1f}"]; fref, lref = r["f"], r["global"]
        else:
            r = ref3[eng]["sigma_h_BL" if q == "sigma_h" else "sigma_DL_corrected"]; fref, lref = r["f_global"], r["lece_global"]
        k = SQ3 if (tab == "T2" or q == "sigma_h") else 1.0
        check(f"{tab} {ds:17s} {eng:12s} {str(q):8s}", abs(f - fref) < 1e-6 * fref and abs(lg - lref) < 1e-9,
              f"f {f:.4f} vs {fref:.4f}; f' = f/{k:.3f} = {f / k:.4f}; LECE {lg:.4f} vs {lref:.4f}")
        res[key] = {"n": len(cases), "f": f / k, "f_fit_unscaled": f, "global": lg}
    # ---- control 3-4: seed 42 with numeric / lexicographic order on Table 3 ------------------------------------------------------
    print("Controls 3-4: Table 3, seed 42, numeric order = audit rev2 C1; lexicographic = table3.json (sigma_h)", flush=True)
    C1 = {"sigma_h": {"SyNRA": 2.651, "ConvexAdam": 0.351, "uniGradICON": 1.245, "Elastix": 0.707},
          "sigma_DL": {"SyNRA": 2.607, "ConvexAdam": 0.533, "uniGradICON": 1.485, "Elastix": 0.828}}
    for eng in ENGINES:
        for arm in ("sigma_h", "sigma_DL"):
            cs = cells[("T3", "DIR-LAB", eng, arm)]
            m = float(np.mean(one_rep((cs, 42)))); check(f"T3 {eng:12s} {arm:8s} seed42 numeric", abs(m - C1[arm][eng]) < 0.002, f"{m:.3f} vs {C1[arm][eng]:.3f}")
            if arm == "sigma_h":
                ml = float(np.mean(one_rep((sorted(cs, key=lambda c: c.case_id), 42)))); ref = ref3[eng]["sigma_h_BL"]["lece_5fold_mean"]
                check(f"T3 {eng:12s} {arm:8s} seed42 lexicographic", abs(ml - ref) < 0.002, f"{ml:.3f} vs {ref:.3f}")
    print(f"\n  CONTROLS: {'PASS' if ok_all else 'FAIL'}  ({time.time() - t0:.0f}s)\n", flush=True)
    # ---- repeated k-fold ---------------------------------------------------------------------------------------------------------
    tasks = [(key, s) for key, cases in cells.items() if len(cases) >= 2 for s in seeds]
    print(f"Repeated k-fold: {len(tasks)} repetitions on {len([k for k in cells if len(cells[k]) >= 2])} cells, {a.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        outs = list(ex.map(one_rep, [(cells[key], s) for key, s in tasks], chunksize=4))
    folds = {}
    for (key, s), o in zip(tasks, outs):
        folds.setdefault(key, []).append(o)
    for key, reps in folds.items():
        allf = np.concatenate([np.array(r) for r in reps]); means = np.array([np.mean(r) for r in reps])
        res[key].update({"kf_mean": float(allf.mean()), "kf_sd_folds": float(allf.std(ddof=1)), "kf_sd_reps": float(means.std(ddof=1)),
                         "kf_min_rep": float(means.min()), "kf_max_rep": float(means.max()), "n_reps": len(reps), "n_splits": min(5, res[key]["n"])})
    # ---- report ---------------------------------------------------------------------------------------------------------------
    fmt = lambda r: f"${r['kf_mean']:.2f} \\pm {r['kf_sd_folds']:.2f}$" if r["n"] >= 5 else "N.A."
    rows2 = []
    print("Table 2 v2 (f' = f/sqrt3; repeated 5-fold mean +- sd over folds [sd of rep means; seed-42 value in step4]):")
    for ds in DATASETS:
        for i, eng in enumerate(ENGINES):
            r10, r5 = res[("T2", ds, eng, 10.0)], res[("T2", ds, eng, 5.0)]
            s = lambda r, q: (f"n {r['n']:2d} f' {r['f']:5.2f} glob {r['global']:.2f} 5f {r['kf_mean']:.2f}+-{r['kf_sd_folds']:.2f} "
                              f"[{r['kf_sd_reps']:.2f}; s42 {ref4[f'{ds}|{eng}'][f'{q:.1f}']['kf_mean']:.2f}]") if r["n"] >= 5 else f"n {r['n']:2d} f' {r.get('f', float('nan')):5.2f} N.A."
            print(f"  {ds:17s} {eng:12s} q10: {s(r10, 10.0)} | q5: {s(r5, 5.0)}")
            rows2.append(f"{ds if i == 0 else '':17s}& {eng:12s}& {r10['n']:2d} & {r10['f']:5.2f} & {r10['global']:.2f} & {fmt(r10)} & "
                         f"{r5['n']:2d} & {r5['f']:5.2f} & {r5['global']:.2f} & {fmt(r5)} \\\\")
        rows2.append("\\midrule")
    rows3 = []
    print("\nTable 3 v2:")
    for eng in ENGINES:
        for arm, name in (("sigma_h", r"heuristic-residual $\sigma_{h}$"), ("sigma_DL", "DL Unsupervised L20S3-4        ")):
            r = res[("T3", "DIR-LAB", eng, arm)]
            print(f"  {eng:12s} {arm:8s} f {r['f']:5.2f} glob {r['global']:.3f} 5f {r['kf_mean']:.2f}+-{r['kf_sd_folds']:.2f} [{r['kf_sd_reps']:.2f}; reps {r['kf_min_rep']:.2f}-{r['kf_max_rep']:.2f}]")
            rows3.append(f"{eng + ' (this work)':23s} & {name} & {r['f']:.2f} & {r['global']:.3f} & {fmt(r)} \\\\")
    fe = [res[("T2", ds, "Elastix", 5.0)]["f"] for ds in DATASETS]
    f_fig3 = float(np.mean(fe))
    kf10 = [res[("T2", ds, e, 10.0)]["kf_mean"] for ds in DATASETS for e in ENGINES if res[("T2", ds, e, 10.0)]["n"] >= 5]
    print(f"\nR44: Elastix f' at q<=5 (Lung/Abdominal/Liver/DIR-LAB) = {', '.join(f'{x:.3f}' for x in fe)} -> mean = {f_fig3:.3f} (submitted rule: 3.862)")
    print(f"5-fold interval over the 16 cells, q<=10: {min(kf10):.2f}-{max(kf10):.2f} (step4 seed 42: 0.33-2.37)")
    json.dump({"controls_pass": bool(ok_all), "n_reps": a.reps, "sqrt3": SQ3, "f_fig3_rule_mean_elastix_q5": f_fig3, "elastix_q5_f": fe,
               "kf10_interval_16cells": [min(kf10), max(kf10)],
               "results": {"|".join(str(x) for x in k): v for k, v in res.items()}}, open(os.path.join(OUT, "table23_v2.json"), "w"), indent=2)
    open(os.path.join(OUT, "Table2_rows_v2.tex"), "w", encoding="utf-8").write("\n".join(rows2) + "\n")
    open(os.path.join(OUT, "Table3_rows_v2.tex"), "w", encoding="utf-8").write("\n".join(rows3) + "\n")
    print(f"\n-> {OUT}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
