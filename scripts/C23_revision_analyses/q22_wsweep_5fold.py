# -*- coding: utf-8 -*-
"""IV round (2026-10-16, reviewer items Q2.1/Q2.6.d): w-sweep of Figure 4 recomputed with the case-level 5-fold LECE.
Data: III Scripts/_da_approvare/_out_q22_step6/cache.npz (R and J at the landmarks + body statistics, DIR-LAB, 4 engines x 10 cases),
the same cache from which the submitted Figure 4 (step6.json, f fitted and evaluated on the same 3000 landmarks) was drawn.
sigma(w) = w * minmax_body(min(R, q_R)) + (1-w) * minmax_body(min(J, q_J)), q_R 0.99, q_J 0.95 (as q22_step6_sensitivity.py).
CONTROLS (abort if they fail): (1) for every (engine, w) the in-sample f, LECE and rho rebuilt here == step6.json (rel. 1e-9);
(2) at w = 0.7 the 5-fold LECE == Table 3 sigma_h (3.00/0.33/1.59/0.66, 2 decimals).
5-fold: KFold(5, shuffle, seeds 0..49) over the 10 cases in fixed numeric order (as Table 2/3); for each w the same partitions, so
the difference LECE(w) - LECE(0.7) is paired fold by fold. rho_pool (Spearman sigma vs TRE, all landmarks) does not depend on f.
Output: ../results/wsweep_5fold.json (old in-sample values kept next to the new ones). Usage (venv): python q22_wsweep_5fold.py"""
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
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "wsweep_5fold.json" if QMAX == float("inf") else f"wsweep_5fold_q{QMAX:g}.json")
C = dict(np.load(STEP6 + "/cache.npz"))
QR, QJ = [0.95, 0.97, 0.99], [0.90, 0.93, 0.95, 0.97, 0.99]
ENG = ["SyNRA", "ConvexAdam", "uniGradICON", "Elastix"]; W = [round(0.1 * i, 1) for i in range(11)]
REPS, BW = 50, 0.25
T3 = {"SyNRA": 3.00, "ConvexAdam": 0.33, "uniGradICON": 1.59, "Elastix": 0.66} if QMAX == float("inf") else {"SyNRA": 1.83, "ConvexAdam": 0.33, "uniGradICON": 1.59, "Elastix": 0.66}   # Table 2 at q <= 10


def sig(lab, c, w, qr=0.99, qj=0.95):
    R, J = C[f"{lab}|{c}|R"], C[f"{lab}|{c}|J"]; rs, js = C[f"{lab}|{c}|Rstat"], C[f"{lab}|{c}|Jstat"]
    qv = js[2 + QJ.index(qj)]; lo, hi = min(js[0], qv), min(js[1], qv)
    Jn = np.clip((np.minimum(J, qv) - lo) / (hi - lo if hi - lo > 1e-8 else 1), 0, 1)
    inb = C[f"{lab}|{c}|inb"] > 0.5; qv = rs[4 + QR.index(qr)]; lo, hi = min(rs[0], qv), min(rs[1], qv)
    Rn = np.clip((np.minimum(R, qv) - lo) / (hi - lo if hi - lo > 1e-8 else 1), 0, 1)
    return np.where(inb, w * Rn + (1 - w) * Jn, 0.0)


def cases(lab, w):
    out = []
    for c in range(1, 11):
        if np.sqrt(np.nanmean(C[f"{lab}|{c}|tre"] ** 2)) > QMAX:
            continue
        t, s = C[f"{lab}|{c}|tre"], sig(lab, c, w); k = np.isfinite(t) & np.isfinite(s) & (s > 1e-9); out.append((t[k], s[k]))
    return out


def job(arg):
    lab, w = arg
    cs = cases(lab, w); tt = np.concatenate([x[0] for x in cs]); ts = np.concatenate([x[1] for x in cs])
    f, _ = fit_c_grid(tt, ts, bin_width=BW)
    insample = {"f": float(f), "lece": float(compute_lece(tt, f * ts, bin_width=BW)[0]), "rho": float(spearmanr(ts, tt)[0]), "n": int(tt.size)}
    folds = []
    for seed in range(REPS):
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(np.arange(len(cs))):
            ft, _ = fit_c_grid(np.concatenate([cs[i][0] for i in tr]), np.concatenate([cs[i][1] for i in tr]), bin_width=BW)
            ht = np.concatenate([cs[i][0] for i in te]); hs = np.concatenate([cs[i][1] for i in te])
            folds.append(float(compute_lece(ht, ft * hs, bin_width=BW)[0]))
    return lab, w, insample, folds


if __name__ == "__main__":
    t0 = time.time()
    old = json.load(open(STEP6 + "/step6.json"))["sweep_w"]
    with Pool(12) as p:
        res = p.map(job, [(l, w) for l in ENG for w in W])
    D = {(l, w): (ins, np.array(fo)) for l, w, ins, fo in res}
    bad = []
    for l in ENG:
        for w in W:
            o, ins = old[l][str(w)], D[(l, w)][0]
            if QMAX != float("inf") and l == "SyNRA":
                continue                                    # SyNRA loses one case at q <= 10: in-sample values differ by design
            for k in ("f", "lece", "rho"):
                if abs(ins[k] - o[k]) > 1e-9 * max(1.0, abs(o[k])):
                    bad.append((l, w, k, ins[k], o[k]))
    assert not bad, ("control 1 failed: in-sample values differ from step6.json", bad[:5])
    for l in ENG:
        m = D[(l, 0.7)][1].mean()
        assert round(m, 2) == T3[l], ("control 2 failed: 5-fold at w=0.7 != Table 3", l, m)
    out = {"meta": {"reps": REPS, "folds": 5, "bin_width": BW, "q_R": 0.99, "q_J": 0.95, "qmax_mm": QMAX, "data": STEP6 + "/cache.npz",
                    "controls": "in-sample f/LECE/rho == step6.json for all (engine, w); 5-fold at w=0.7 == Table 3 sigma_h"},
           "results": {}}
    for l in ENG:
        ref = D[(l, 0.7)][1]
        out["results"][l] = {}
        for w in W:
            ins, fo = D[(l, w)]; d = fo - ref
            out["results"][l][str(w)] = {"insample_submitted": ins, "lece5_mean": float(fo.mean()), "lece5_sd": float(fo.std(ddof=1)),
                                         "rep_means_sd": float(fo.reshape(REPS, 5).mean(1).std(ddof=1)),
                                         "diff_vs_w07_mean": float(d.mean()), "diff_vs_w07_sd": float(d.std(ddof=1)),
                                         "frac_folds_better_than_w07": float((d < 0).mean()), "rho_pool": ins["rho"]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("controls OK (in-sample == step6.json for 44 cells; 5-fold w=0.7 == Table 3)")
    print("w    LECE 5-fold mean+-sd (SyNRA/ConvexAdam/uniGradICON/Elastix) | paired diff vs 0.7 | rho_pool")
    for w in W:
        r = [out["results"][l][str(w)] for l in ENG]
        print(f"{w:.1f}  " + "  ".join(f"{x['lece5_mean']:.2f}+-{x['lece5_sd']:.2f}" for x in r) + "  |  "
              + " ".join(f"{x['diff_vs_w07_mean']:+.2f}({100*x['frac_folds_better_than_w07']:.0f}%)" for x in r) + "  |  "
              + " ".join(f"{x['rho_pool']:+.3f}" for x in r))
    print("saved", OUT, "| %.0f s" % (time.time() - t0))
