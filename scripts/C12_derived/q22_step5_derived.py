# -*- coding: utf-8 -*-
"""
q22_step5_derived.py -- step 5 of the recalculation order (2026-09-28): numbers derived from Tables 2-4, recomputed with sigma_BL
(<DATASETS>/_sigma_BL_20260927, same relative paths as the submitted cells). Each block first reproduces the
submitted value from the submitted data (control), then gives the new one.

  A  f-stability per engine across datasets (Table 2, q <= 10): max/min and CV          [manuscript §3.5 "1.9-3.6x, CV 32-68 %"]
  B  rho ceiling on DIR-LAB (_ideal_rho_ceiling.py: Maxwell-Boltzmann draws of scale sigma_p, K draws), CV of sigma_p, obs/ceil
                                                                                         [§3.4 "CV ~ 10 %", "0.19-0.25", "0.25-1.08"]
  C  paired-case QA (paired_case_qa.py): common converged subset per QA dataset and q; bin 0.5 (control) and 0.25 (N6)  [R1.3, P12L21]
  D  cross-dataset f-transfer Lung+Abdominal+Liver -> Castillo, training pool TRE < 20 mm (user decision 2026-09-28), holdout
     unfiltered. Control: FINAL numbers from the partE_redo path table + filter; new: the Table 2 cells (ENGINE_PATHS) with sigma_BL
                                                                                         [§3.5 "+0.40 to +4.23", Q2.4.f, R1.2]
  E  cross-engine leave-one-out f-transfer on Castillo (fit on 3 engines, LECE on the 4th), with the Table 3 sigma (landmarks)
                                                                                         [§3.5 "+0.25 to +2.53" -- from _analysis_v28 own sigma (N4)]
Output ./_out_q22_step5/step5.json + printed summary. Read-only. Usage (venv): python q22_step5_derived.py
"""
from dirunc_paths import DATASETS, LIB, RESULTS  # data roots, see README
import os, sys, types, json, math
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = LIB; PS = LIB + "/validation"
sys.path.insert(0, SCRIPTS); sys.path.insert(0, PS)
if "validation" not in sys.modules:
    _v = types.ModuleType("validation"); _v.__path__ = [PS]; sys.modules["validation"] = _v
import numpy as np
from validation.build_v11_data import ENGINE_PATHS, _calibrate_subset
from validation.calibrate import load_all_landmarks, stack_cases, fit_c_grid
from validation.lece import compute_lece

DS = f"{DATASETS}"; BLR = f"{DS}/_sigma_BL_20260927"
bl = lambda p: os.path.join(BLR, os.path.relpath(p, DS)).replace("\\", "/")
ENG = ["SyNRA", "ConvexAdam", "uniGradICON", "Elastix"]; QA = ["Lung-DIR-QA", "Abdominal-DIR-QA", "Liver-DIR-QA"]
SUB = {e: ENGINE_PATHS[e]["datasets"] for e in ENG}; NEW = {e: {d: bl(p) for d, p in SUB[e].items()} for e in ENG}
OUT = os.path.join(HERE, "_out_q22_step5"); os.makedirs(OUT, exist_ok=True)
S4 = json.load(open(os.path.join(HERE, "_out_q22_step4", "step4.json")))
R = {}


# ---------------- A: f-stability ----------------
print("A  f-stability across datasets (Table 2, q<=10)")
R["A"] = {}
for tag, key in (("submitted", "table2_submitted"), ("new", "table2_new")):
    rows = {}
    for e in ENG:
        f = np.array([S4[key][f"{d}|{e}"]["10.0"]["f"] for d in QA + ["DIR-LAB"]])
        rows[e] = {"ratio": float(f.max() / f.min()), "cv_pop": float(f.std() / f.mean()), "cv_samp": float(f.std(ddof=1) / f.mean()),
                   "argmax": (QA + ["DIR-LAB"])[int(f.argmax())]}
    R["A"][tag] = rows
    rr = [v["ratio"] for v in rows.values()]; c0 = [v["cv_pop"] for v in rows.values()]; c1 = [v["cv_samp"] for v in rows.values()]
    print(f"  {tag:9s} ratio {min(rr):.2f}-{max(rr):.2f}x | CV pop {100*min(c0):.0f}-{100*max(c0):.0f} % | CV sample {100*min(c1):.0f}-{100*max(c1):.0f} % | "
          f"largest f: " + ", ".join(f"{e} {v['argmax']}" for e, v in rows.items()))


# ---------------- B: rho ceiling (DIR-LAB) ----------------
def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / (math.sqrt((rx ** 2).sum() * (ry ** 2).sum()) + 1e-12))


print("\nB  rho ceiling on DIR-LAB (K = 4000 Maxwell-Boltzmann draws, seed 0 per engine block as _ideal_rho_ceiling.py)")
R["B"] = {}
for tag, paths, t4 in (("submitted", SUB, "table4_submitted"), ("new", NEW, "table4_new")):
    rng = np.random.default_rng(0); rows = {}
    for e in ENG:
        sig = np.concatenate([c.sigma_raw for c in load_all_landmarks(paths[e]["DIR-LAB"])])
        sig = sig[np.isfinite(sig) & (sig > 0)]
        rh = np.array([spearman(sig, sig * np.sqrt((g * g).sum(axis=1))) for g in (rng.standard_normal((sig.size, 3)) for _ in range(4000))])
        obs = S4[t4][f"DIR-LAB|{e}"]["pool"]
        rows[e] = {"cv": float(sig.std() / sig.mean()), "ceil": float(rh.mean()), "ceil_sd": float(rh.std()), "obs": obs, "obs_over_ceil": obs / float(rh.mean())}
        print(f"  {tag:9s} {e:12s} CV {rows[e]['cv']:.3f} ceiling {rows[e]['ceil']:.3f}±{rows[e]['ceil_sd']:.3f} observed {obs:+.3f} obs/ceil {rows[e]['obs_over_ceil']:+.2f}")
    R["B"][tag] = rows


# ---------------- C: paired-case QA ----------------
print("\nC  paired-case QA (common converged subset), LECE global")
R["C"] = {}
for tag, paths in (("submitted", SUB), ("new", NEW)):
    R["C"][tag] = {}
    for bw in (0.5, 0.25):
        blk = {}
        for d in QA:
            ec = {e: {c.case_id: c for c in load_all_landmarks(paths[e][d])} for e in ENG}
            rm = {e: {k: float(np.sqrt(np.mean(c.tre_mm ** 2))) for k, c in ec[e].items()} for e in ENG}
            for ql, qv in (("<=10", 10.0), ("<=5", 5.0)):
                common = sorted(set.intersection(*[{k for k, r in rm[e].items() if r <= qv} for e in ENG]))
                row = {"n_common": len(common)}
                for e in ENG:
                    x = _calibrate_subset([ec[e][k] for k in common], bin_width=bw) if common else None
                    row[e] = None if x is None else {"lece": x["lece_global"], "f": x["f_global"]}
                blk[f"{d}|{ql}"] = row
        R["C"][tag][str(bw)] = blk
pc_path = RESULTS + "/paired_case_qa.json"
if os.path.isfile(pc_path):
    ref = json.load(open(pc_path)); dmax = 0.0
    for d in QA:
        for ql in ("<=10", "<=5"):
            rr = ref[d]["by_threshold"][ql]
            for e in ENG:
                a = R["C"]["submitted"]["0.5"][f"{d}|{ql}"][e]; b = rr["engines"][e]["paired_lece_global"]
                if a is not None and b is not None:
                    dmax = max(dmax, abs(a["lece"] - b))
    print(f"  [control] submitted paths, bin 0.5 vs paired_case_qa.json: max |dLECE| {dmax:.2e} {'PASS' if dmax < 1e-9 else 'FAIL'}")
for d in QA:
    for ql in ("<=10", "<=5"):
        k = f"{d}|{ql}"; n = R["C"]["new"]["0.25"][k]["n_common"]
        f = lambda t, bw: " ".join(f"{R['C'][t][bw][k][e]['lece']:.2f}" if R['C'][t][bw][k][e] else "-" for e in ENG)
        print(f"  {d:17s} q{ql:5s} n_common {n:2d} | submitted bin0.5 {f('submitted','0.5')} | submitted bin0.25 {f('submitted','0.25')} | NEW bin0.25 {f('new','0.25')}  (SyN/CA/uGI/Elx)")


# ---------------- D: cross-dataset transfer ----------------
def pool(folders, tre_max=None, q_case=None):
    """tre_max: landmark filter TRE < tre_max (submitted FINAL); q_case: keep only cases with RMS TRE <= q_case (Table 2 criterion)."""
    T, S, n = [], [], 0
    for fo in folders:
        if not fo or not os.path.isdir(fo):
            continue
        for c in load_all_landmarks(fo):
            if q_case is not None and float(np.sqrt(np.mean(c.tre_mm ** 2))) > q_case:
                continue
            k = np.ones(c.tre_mm.size, bool) if tre_max is None else (c.tre_mm < tre_max)
            T.append(c.tre_mm[k]); S.append(c.sigma_raw[k]); n += 1
    return np.concatenate(T), np.concatenate(S), n


def transfer(train, hold, tre_max=None, q_case=None):
    tt, ts, n = pool(train, tre_max, q_case); ht, hs, _ = pool([hold])
    ft, _ = fit_c_grid(tt, ts); lh, _ = compute_lece(ht, ft * hs, bin_width=0.25)
    fn, _ = fit_c_grid(ht, hs); ln, _ = compute_lece(ht, fn * hs, bin_width=0.25)
    return {"n_cases_train": n, "n_lm_train": int(tt.size), "f_train": float(ft), "lece_holdout": float(lh), "f_native": float(fn), "lece_native": float(ln), "dlece": float(lh - ln)}


print("\nD  cross-dataset f-transfer (train Lung+Abdominal+Liver; holdout Castillo unfiltered). NEW = training restricted to the\n"
      "   converged cases, q <= 10 mm as in Table 2 (user decision 2026-09-28, replaces the landmark filter TRE < 20 mm);\n"
      "   control and references keep the submitted landmark filter TRE < 20 mm")
V2 = {"SyNRA": [f"{DS}/Lung-DIR-QA/Results_synra_mindssc", f"{DS}/AbdominalDIR-QA/Annotated Image Pairs/Results_synra_mindssc", None],
      "ConvexAdam": [f"{DS}/Lung-DIR-QA/Results_convexadam_mindssc", f"{DS}/AbdominalDIR-QA/Annotated Image Pairs/Results_convexadam_mindssc", f"{DS}/Liver-DIR-QA/Results_convexadam_mindssc"],
      "uniGradICON": [SUB["uniGradICON"][d] for d in QA], "Elastix": [SUB["Elastix"][d] for d in QA]}
FINAL = {"SyNRA": 4.2343, "ConvexAdam": 3.2967, "uniGradICON": 1.4874, "Elastix": 0.3955}
R["D"] = {}
for tag, tr, ho, kw in (("control_FINAL_paths", V2, SUB, {"tre_max": 20.0}),
                        ("submitted_cells", {e: [SUB[e][d] for d in QA] for e in ENG}, SUB, {"tre_max": 20.0}),
                        ("new_TRE20_reference", {e: [NEW[e][d] for d in QA] for e in ENG}, NEW, {"tre_max": 20.0}),
                        ("no_filter_reference", {e: [NEW[e][d] for d in QA] for e in ENG}, NEW, {}),
                        ("new", {e: [NEW[e][d] for d in QA] for e in ENG}, NEW, {"q_case": 10.0})):
    R["D"][tag] = {e: transfer(tr[e], ho[e]["DIR-LAB"], **kw) for e in ENG}
    for e in ENG:
        x = R["D"][tag][e]
        chk = f"  (FINAL {FINAL[e]:+.4f} {'PASS' if abs(x['dlece'] - FINAL[e]) < 1e-3 else 'FAIL'})" if tag.startswith("control") else ""
        print(f"  {tag:20s} {e:12s} cases {x['n_cases_train']:3d} lm {x['n_lm_train']:6d} f_train {x['f_train']:6.3f} LECE_hold {x['lece_holdout']:.3f} "
              f"f_nat {x['f_native']:.3f} LECE_nat {x['lece_native']:.3f} dLECE {x['dlece']:+.3f}{chk}")


# ---------------- E: cross-engine LOO on Castillo ----------------
print("\nE  cross-engine LOO f-transfer on Castillo (Table 3 sigma)")
R["E"] = {}
for tag, paths in (("submitted_T3sigma", SUB), ("new", NEW)):
    data = {e: stack_cases(load_all_landmarks(paths[e]["DIR-LAB"])) for e in ENG}; R["E"][tag] = {}
    for h in ENG:
        tt = np.concatenate([data[e][0] for e in ENG if e != h]); ts = np.concatenate([data[e][1] for e in ENG if e != h])
        fl, _ = fit_c_grid(tt, ts); ht, hs = data[h]
        ll, _ = compute_lece(ht, fl * hs, bin_width=0.25); fn, _ = fit_c_grid(ht, hs); ln, _ = compute_lece(ht, fn * hs, bin_width=0.25)
        R["E"][tag][h] = {"f_loo": float(fl), "lece_loo": float(ll), "f_native": float(fn), "lece_native": float(ln), "dlece": float(ll - ln)}
        print(f"  {tag:18s} {h:12s} f_native {fn:.3f} LECE_native {ln:.3f} f_LOO {fl:.3f} LECE_LOO {ll:.3f} dLECE {ll - ln:+.3f}")
print("  (submitted text +0.25..+2.53 comes from _analysis_v28's own sigma: SyNRA +2.529, ConvexAdam +2.329, uniGradICON +0.248, Elastix +0.888)")

json.dump(R, open(os.path.join(OUT, "step5.json"), "w"), indent=2)
print(f"\n-> {OUT}")
