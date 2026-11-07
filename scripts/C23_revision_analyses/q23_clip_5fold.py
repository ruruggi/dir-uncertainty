# -*- coding: utf-8 -*-
"""IV round (2026-10-16, Q2.3 "relevance of the twists not quantified"): effect of the outlier clip of R (q_R) and of d (q_J) on the
case-level 5-fold LECE and on rho_pool, DIR-LAB, 4 engines x 10 cases. Same data and sigma construction as q22_wsweep_5fold.py
(III _out_q22_step6/cache.npz); "none" = no clip (quantile replaced by the body maximum, Rstat[1] / Jstat[1]).
CONTROL (abort if it fails): default (w 0.7, q_R 0.99, q_J 0.95) 5-fold == Table 3 sigma_h (3.00/0.33/1.59/0.66).
Paired: same KFold partitions (seeds 0..49) for every configuration. Output ../results/clip_5fold.json. Usage (venv)."""
from dirunc_paths import LIB, RESULTS  # data roots, see README
import os, sys, types, json, time
import numpy as np
from multiprocessing import Pool
III = LIB  # was the III Scripts folder
PS = LIB + "/validation"; sys.path.insert(0, PS)
v = types.ModuleType("validation"); v.__path__ = [PS]; sys.modules["validation"] = v
from validation.calibrate import fit_c_grid
from lece import compute_lece
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
STEP6 = RESULTS + "/_out_q22_step6"
QMAX = float(os.environ.get("QMAX", "inf"))   # 2026-10-17 (Q2.6 f+a): QMAX=10 keeps only cases with RMS TRE <= 10 mm, as Table 2
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "clip_5fold.json" if QMAX == float("inf") else f"clip_5fold_q{QMAX:g}.json")
C = dict(np.load(STEP6 + "/cache.npz"))
QR, QJ = [0.95, 0.97, 0.99], [0.90, 0.93, 0.95, 0.97, 0.99]
ENG = ["SyNRA", "ConvexAdam", "uniGradICON", "Elastix"]
CONF = {"default (0.99, 0.95)": (0.99, 0.95), "q_R none": (None, 0.95), "q_J none": (0.99, None), "both none": (None, None),
        "q_R 0.95": (0.95, 0.95), "q_J 0.90": (0.99, 0.90), "q_J 0.99": (0.99, 0.99)}
REPS, BW, W = 50, 0.25, 0.7
T3 = {"SyNRA": 3.00, "ConvexAdam": 0.33, "uniGradICON": 1.59, "Elastix": 0.66} if QMAX == float("inf") else {"SyNRA": 1.83, "ConvexAdam": 0.33, "uniGradICON": 1.59, "Elastix": 0.66}   # Table 2 at q <= 10


def sig(lab, c, qr, qj):
    R, J = C[f"{lab}|{c}|R"], C[f"{lab}|{c}|J"]; rs, js = C[f"{lab}|{c}|Rstat"], C[f"{lab}|{c}|Jstat"]
    qv = js[1] if qj is None else js[2 + QJ.index(qj)]; lo, hi = min(js[0], qv), min(js[1], qv)
    Jn = np.clip((np.minimum(J, qv) - lo) / (hi - lo if hi - lo > 1e-8 else 1), 0, 1)
    inb = C[f"{lab}|{c}|inb"] > 0.5
    qv = rs[1] if qr is None else rs[4 + QR.index(qr)]; lo, hi = min(rs[0], qv), min(rs[1], qv)
    Rn = np.clip((np.minimum(R, qv) - lo) / (hi - lo if hi - lo > 1e-8 else 1), 0, 1)
    return np.where(inb, W * Rn + (1 - W) * Jn, 0.0)


def job(arg):
    lab, name = arg; qr, qj = CONF[name]
    cs = []
    for c in range(1, 11):
        if np.sqrt(np.nanmean(C[f"{lab}|{c}|tre"] ** 2)) > QMAX:
            continue
        t, s = C[f"{lab}|{c}|tre"], sig(lab, c, qr, qj); k = np.isfinite(t) & np.isfinite(s) & (s > 1e-9); cs.append((t[k], s[k]))
    tt = np.concatenate([x[0] for x in cs]); ts = np.concatenate([x[1] for x in cs])
    folds = []
    for seed in range(REPS):
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(np.arange(len(cs))):
            ft, _ = fit_c_grid(np.concatenate([cs[i][0] for i in tr]), np.concatenate([cs[i][1] for i in tr]), bin_width=BW)
            folds.append(float(compute_lece(np.concatenate([cs[i][0] for i in te]), ft * np.concatenate([cs[i][1] for i in te]), bin_width=BW)[0]))
    return lab, name, folds, float(spearmanr(ts, tt)[0])


if __name__ == "__main__":
    t0 = time.time()
    with Pool(12) as p:
        res = p.map(job, [(l, n) for l in ENG for n in CONF])
    D = {(l, n): (np.array(fo), r) for l, n, fo, r in res}
    for l in ENG:
        assert round(D[(l, "default (0.99, 0.95)")][0].mean(), 2) == T3[l], ("control failed", l)
    out = {"meta": {"reps": REPS, "w": W, "bin_width": BW, "data": STEP6 + "/cache.npz", "control": "default == Table 3 sigma_h"}, "results": {}}
    print("control OK (default == Table 3)")
    print("config                 LECE 5-fold mean (SyNRA/ConvexAdam/uniGradICON/Elastix) | paired diff vs default | rho_pool")
    for n in CONF:
        row = []
        for l in ENG:
            fo, r = D[(l, n)]; d = fo - D[(l, "default (0.99, 0.95)")][0]
            out["results"].setdefault(l, {})[n] = {"lece5_mean": float(fo.mean()), "lece5_sd": float(fo.std(ddof=1)), "diff_mean": float(d.mean()),
                                                  "diff_sd": float(d.std(ddof=1)), "frac_folds_better": float((d < 0).mean()), "rho_pool": r}
            row.append(out["results"][l][n])
        print(f"{n:22s} " + " ".join(f"{x['lece5_mean']:.2f}" for x in row) + "  |  " + " ".join(f"{x['diff_mean']:+.2f}({100*x['frac_folds_better']:.0f}%)" for x in row)
              + "  |  " + " ".join(f"{x['rho_pool']:+.3f}" for x in row))
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(out, open(OUT, "w"), indent=1)
    print("saved", OUT, "| %.0f s" % (time.time() - t0))
