# -*- coding: utf-8 -*-
"""IV round, Q2.6.b (2026-10-17): sigma_h and sigma_DL on DIR-LAB at the single threshold q <= 10 mm of Table 2 (Q2.6 f+a), so that the
four sigma_DL LECE values can be compared directly with the sigma_h values of Table 2 (R2: «compare the LECE values in table 2 for the
DIRLAB to the 4 values that would be achieved with Smolders' model»). Table 3 of the III used all 10 cases for both arms; at q <= 10 only
SyNRA loses a case (9 of 10). Uses the III module q22_table23_v2.py UNCHANGED (imported: cells, one_rep, cell_global, same seeds 0..49,
natural case order, 5-fold, bin 0.25 mm). Control: sigma_h SyNRA at q <= 10 must reproduce Table 2 (n 9, f' 2.98, 5-fold 1.83 +- 1.12).
Output: IV results/q26b_dirlab_q10.json. Usage (system python): python q26b_dirlab_q10.py [--reps 50] [--workers 20]"""
from dirunc_paths import LIB  # data roots, see README
import os, sys, json, argparse
import numpy as np
III = LIB  # was the III Scripts/_da_approvare folder; its scripts are found by name (dirunc_paths)
sys.path.insert(0, III)
import q22_table23_v2 as T                       # noqa: E402  (module-level code only defines paths and helpers)
from concurrent.futures import ProcessPoolExecutor
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "q26b_dirlab_q10.json")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=50); ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    res = {}
    for eng in T.ENGINES:
        cases = sorted(T.load_all_landmarks(T.bl(T.ENGINE_PATHS[eng]["datasets"]["DIR-LAB"])), key=T.natkey)
        rmse = {c.case_id: float(np.sqrt(np.mean(c.tre_mm ** 2))) for c in cases}
        keep = {cid for cid, v in rmse.items() if v <= 10.0}
        arms = {"sigma_h": [c for c in cases if c.case_id in keep],
                "sigma_DL": [c for c in sorted(T.dl_cases(T.ENG_KEY[eng]), key=T.natkey) if c.case_id in keep]}
        for arm, cs in arms.items():
            f, lg = T.cell_global(cs); k = T.SQ3 if arm == "sigma_h" else 1.0
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                reps = list(ex.map(T.one_rep, [(cs, s) for s in range(a.reps)]))
            allf = np.concatenate([np.array(r) for r in reps])
            res[f"{eng}|{arm}"] = {"n": len(cs), "cases": sorted(c.case_id for c in cs), "f": f / k, "kf_mean": float(allf.mean()),
                                   "kf_sd_folds": float(allf.std(ddof=1))}
            print(f"{eng:12s} {arm:8s} n {len(cs):2d}  f {f / k:5.2f}  5-fold {allf.mean():.2f} +- {allf.std(ddof=1):.2f}", flush=True)
    c = res["SyNRA|sigma_h"]
    ok = c["n"] == 9 and abs(c["f"] - 2.976) < 0.005 and abs(c["kf_mean"] - 1.833) < 0.005 and abs(c["kf_sd_folds"] - 1.118) < 0.005
    print("CONTROL (sigma_h SyNRA = Table 2):", "PASS" if ok else "FAIL")
    json.dump({"control_pass": ok, "reps": a.reps, "results": res}, open(OUT, "w"), indent=1)
    print("->", OUT)


if __name__ == "__main__":
    main()
