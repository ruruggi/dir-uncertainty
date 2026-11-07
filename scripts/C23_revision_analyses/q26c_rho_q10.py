# -*- coding: utf-8 -*-
"""IV round, Q2.6.c (2026-10-17): Spearman rho(sigma_p, TRE) of the heuristic sigma (sigma_BL, the sigma of Table 2) per (dataset, engine)
cell at the single threshold q <= 10 mm (Q2.6 f+a), with the perfect-sigma ceiling of each cell and the ratio rho / ceiling
(R2: «report rho relative to the corresponding ideal situation»). Ceiling = mean Spearman rho(sigma_p, sigma_p * ||N(0, I3)||) over K
draws, i.e. the rho of a perfectly calibrated Gaussian sigma with the observed spread of sigma_p (definition of III step 5 block B /
_ideal_rho_ceiling.py; rho is invariant to f, so sigma_raw is used). Also rho_pool of sigma_DL on DIR-LAB at q <= 10 (for §3.4).
Controls (unfiltered): rho_pool and per-case rho = current Table 3 (ex Table 4; e.g. DIR-LAB SyNRA +0.141, Lung Elastix +0.354);
DIR-LAB ceilings = III step5.json B.new (0.236 / 0.216 / 0.198 / 0.231) within Monte-Carlo error.
Uses III modules unchanged (q22_table23_v2: paths, landmark loader, sigma_DL loader). Output: IV results/q26c_rho_q10.json.
Usage (system python): python q26c_rho_q10.py [--K 400]"""
from dirunc_paths import LIB  # data roots, see README
import os, sys, json, argparse
import numpy as np
from scipy.stats import spearmanr
III = LIB  # was the III Scripts/_da_approvare folder; its scripts are found by name (dirunc_paths)
sys.path.insert(0, III)
import q22_table23_v2 as T                       # noqa: E402
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "q26c_rho_q10.json")


def ceiling(sig, K, rng):
    r = [spearmanr(sig, sig * np.linalg.norm(rng.standard_normal((sig.size, 3)), axis=1))[0] for _ in range(K)]
    return float(np.mean(r)), float(np.std(r, ddof=1))


def stats(cases, K, rng):
    t = np.concatenate([c.tre_mm for c in cases]); s = np.concatenate([c.sigma_raw for c in cases])
    pc = np.array([spearmanr(c.sigma_raw, c.tre_mm)[0] for c in cases])
    ok = np.isfinite(s) & (s > 0)
    ce, ce_sd = ceiling(s[ok], K, rng) if K else (float("nan"), float("nan"))
    rho = float(spearmanr(s, t)[0])
    return {"n": len(cases), "n_lm": int(t.size), "rho_pool": rho, "pc_mean": float(np.nanmean(pc)), "pc_sd": float(np.nanstd(pc, ddof=0)),
            "cv": float(np.std(s[ok]) / np.mean(s[ok])), "ceil": ce, "ceil_sd": ce_sd, "ratio": rho / ce if K else float("nan")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=400); a = ap.parse_args()
    rng = np.random.default_rng(0); res = {"q10": {}, "all": {}, "dl_dirlab_q10": {}}
    for ds in T.DATASETS:
        for eng in T.ENGINES:
            cases = sorted(T.load_all_landmarks(T.bl(T.ENGINE_PATHS[eng]["datasets"][ds])), key=T.natkey)
            rmse = {c.case_id: float(np.sqrt(np.mean(c.tre_mm ** 2))) for c in cases}
            keep = [c for c in cases if rmse[c.case_id] <= 10.0]
            res["all"][f"{ds}|{eng}"] = stats(cases, a.K if ds == "DIR-LAB" else 0, rng)
            res["q10"][f"{ds}|{eng}"] = r = stats(keep, a.K, rng)
            print(f"{ds:17s} {eng:12s} n {r['n']:2d} lm {r['n_lm']:6d}  rho {r['rho_pool']:+.3f}  pc {r['pc_mean']:+.3f}+-{r['pc_sd']:.3f}  "
                  f"CV {r['cv']:.2f}  ceil {r['ceil']:.3f}+-{r['ceil_sd']:.3f}  ratio {r['ratio']:+.2f}", flush=True)
            if ds == "DIR-LAB":
                ids = {c.case_id for c in keep}
                dl = [c for c in sorted(T.dl_cases(T.ENG_KEY[eng]), key=T.natkey) if c.case_id in ids]
                t = np.concatenate([c.tre_mm for c in dl]); s = np.concatenate([c.sigma_raw for c in dl])
                res["dl_dirlab_q10"][eng] = {"n": len(dl), "rho_pool": float(spearmanr(s, t)[0])}
                print(f"   sigma_DL DIR-LAB q<=10: n {len(dl)} rho_pool {res['dl_dirlab_q10'][eng]['rho_pool']:+.3f}", flush=True)
    ref = {"DIR-LAB|SyNRA": 0.141, "DIR-LAB|ConvexAdam": 0.047, "DIR-LAB|uniGradICON": 0.213, "DIR-LAB|Elastix": 0.107,
           "Lung-DIR-QA|Elastix": 0.354, "Liver-DIR-QA|uniGradICON": -0.136, "Abdominal-DIR-QA|SyNRA": 0.263}
    c1 = all(abs(res["all"][k]["rho_pool"] - v) < 0.0006 for k, v in ref.items())
    ceil_ref = {"SyNRA": 0.2356, "ConvexAdam": 0.2161, "uniGradICON": 0.1981, "Elastix": 0.2308}
    c2 = all(abs(res["all"][f"DIR-LAB|{e}"]["ceil"] - v) < 0.006 for e, v in ceil_ref.items())
    print("CONTROL rho unfiltered = Table:", "PASS" if c1 else "FAIL", "| CONTROL ceilings DIR-LAB = step5:", "PASS" if c2 else "FAIL")
    res["controls"] = {"rho_unfiltered": c1, "ceil_dirlab": c2, "K": a.K}
    json.dump(res, open(OUT, "w"), indent=1); print("->", OUT)


if __name__ == "__main__":
    main()
