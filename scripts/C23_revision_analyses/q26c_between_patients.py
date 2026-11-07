# -*- coding: utf-8 -*-
"""IV round, Q2.6.c (2026-10-17): the «between patients» numbers of §3.4 (cited by Limitations (ii)) at q <= 10 mm: per cell, CV across
patients of the mean sigma_h at the landmarks, CV across patients of the RMS TRE, and Spearman rho across patients between the two;
plus, for Liver-DIR-QA / uniGradICON, the number of cases with negative per-case rho. Control: the q <= 10 values = the text
(CV 2--6 % vs 23--97 %, rho across patients positive in 15 of 16); unfiltered: CV of the RMS TRE 24--261 %. Uses III module q22_table23_v2 unchanged.
Output: IV results/q26c_between_patients.json. Usage: python q26c_between_patients.py"""
from dirunc_paths import LIB  # data roots, see README
import os, sys, json
import numpy as np
from scipy.stats import spearmanr
III = LIB  # was the III Scripts/_da_approvare folder; its scripts are found by name (dirunc_paths)
sys.path.insert(0, III)
import q22_table23_v2 as T                       # noqa: E402
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "q26c_between_patients.json")


def between(cases):
    ms = np.array([np.mean(c.sigma_raw) for c in cases]); rt = np.array([np.sqrt(np.mean(c.tre_mm ** 2)) for c in cases])
    return {"n": len(cases), "cv_sigma": float(ms.std() / ms.mean()), "cv_tre": float(rt.std() / rt.mean()), "rho": float(spearmanr(ms, rt)[0]),
            "neg_pc": int(sum(spearmanr(c.sigma_raw, c.tre_mm)[0] < 0 for c in cases))}


res = {"q10": {}, "all": {}}
for ds in T.DATASETS:
    for eng in T.ENGINES:
        cases = sorted(T.load_all_landmarks(T.bl(T.ENGINE_PATHS[eng]["datasets"][ds])), key=T.natkey)
        rmse = {c.case_id: float(np.sqrt(np.mean(c.tre_mm ** 2))) for c in cases}
        res["all"][f"{ds}|{eng}"] = between(cases); res["q10"][f"{ds}|{eng}"] = between([c for c in cases if rmse[c.case_id] <= 10.0])
for tag in ("all", "q10"):
    v = res[tag].values()
    print(f"{tag:4s}: CV sigma {100 * min(x['cv_sigma'] for x in v):.0f}-{100 * max(x['cv_sigma'] for x in v):.0f} %, "
          f"CV TRE {100 * min(x['cv_tre'] for x in v):.0f}-{100 * max(x['cv_tre'] for x in v):.0f} %, "
          f"rho across patients > 0 in {sum(x['rho'] > 0 for x in v)} of 16; Liver/uGI negative per-case rho in "
          f"{res[tag]['Liver-DIR-QA|uniGradICON']['neg_pc']} of {res[tag]['Liver-DIR-QA|uniGradICON']['n']} cases")
json.dump(res, open(OUT, "w"), indent=1); print("->", OUT)
