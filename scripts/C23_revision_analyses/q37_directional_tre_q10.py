# -*- coding: utf-8 -*-
"""IV round, critical review 1.4 (2026-10-24): Limitations (i) quotes a per-axis RMSE of the landmark error on DIR-LAB (CC/lateral
2.0-5.2x, SyNRA 5.6 vs 1.1 mm) computed on all 10 cases, including SyNRA case 8 (RMS TRE > 10 mm), against the single threshold
q <= 10 mm declared in R2.6.f. Runs the collection script scripts/C17_directional_tre/directional_tre_castillo.py UNCHANGED on disk
(exec of its source with string patches, assert count == 1).
  --mode control : no change of cases -> must reproduce CC/lateral 5.23/2.01/3.35/2.60 and SyNRA CC 5.63, lateral 1.08 mm;
  --mode q10     : per engine, only the cases whose RMS TRE (from landmarks.npz) is <= 10 mm, as Table 2 (Table 1 check skipped:
                   Table 1 reports all cases).
Output: results/q37_directional_tre/directional_tre_<mode>.json. Usage (venv): python q37_directional_tre_q10.py --mode control|q10"""
from dirunc_paths import DATASETS, script  # data roots, see README
import os, sys, json, argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["control", "q10"], required=True); args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); IVL = os.path.dirname(HERE)
SRC = script("directional_tre_castillo.py")
OUTD = os.path.join(IVL, "results", "q37_directional_tre"); os.makedirs(OUTD, exist_ok=True)
BASE = f"{DATASETS}/Castillo-DIR-LAB"
Q = 10.0
s = open(SRC, encoding="utf-8").read()


def patch(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old)
    s = s.replace(old, new)


patch('OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out_C17")', "OUT_DIR = OUTD_Q37")
patch('os.path.join(OUT_DIR, "directional_tre.json")', 'os.path.join(OUT_DIR, "directional_tre_%s.json" % MODE_Q37)')
KEEP = {}
if args.mode == "q10":
    for key in ("synra", "convexadam", "unigradicon", "elastix"):
        rms = {c: float(np.sqrt(np.mean(np.load(os.path.join(BASE, f"Results_{key}_procrustes", f"case{c}", "landmarks.npz"))["tre_mm"] ** 2)))
               for c in range(1, 11)}
        KEEP[key] = [c for c in range(1, 11) if rms[c] <= Q]
        print(f"[q37] {key}: retained cases {KEEP[key]} (dropped: {[(c, round(rms[c], 2)) for c in range(1, 11) if rms[c] > Q]})")
    patch("        resid, per_case = [], []\n        for c in CASES:", "        resid, per_case = [], []\n        for c in KEEP_Q37[engine_key]:")
    patch('assert abs(case_mean - TABLE1[label]) < 0.02,', 'assert True or abs(case_mean - TABLE1[label]) < 0.02,')
    patch('json.dump({"dataset": "DIR-LAB Castillo", "cases": list(CASES),',
          'json.dump({"dataset": "DIR-LAB Castillo", "cases": {k: v for k, v in KEEP_Q37.items()}, "q_mm": 10.0,')
g = {"__file__": SRC, "__name__": "__main__", "OUTD_Q37": OUTD, "MODE_Q37": args.mode, "KEEP_Q37": KEEP}
exec(compile(s, SRC, "exec"), g)

res = json.load(open(os.path.join(OUTD, "directional_tre_%s.json" % args.mode)))["per_engine"]
r = {e["engine"]: e for e in res}
if args.mode == "control":
    exp = {"SyNRA": 5.23, "ConvexAdam": 2.01, "uniGradICON": 3.35, "Elastix": 2.60}
    assert all(abs(round(r[k]["anis_CC_over_lat"], 2) - v) < 1e-9 for k, v in exp.items()), {k: r[k]["anis_CC_over_lat"] for k in exp}
    assert round(r["SyNRA"]["rms_CC_z_mm"], 2) == 5.63 and round(r["SyNRA"]["rms_lateral_x_mm"], 2) == 1.08
    print("[control] published per-axis numbers reproduced")
print("[summary] CC/lat:", {k: round(v["anis_CC_over_lat"], 2) for k, v in r.items()},
      "| CC/AP:", {k: round(v["anis_CC_over_AP"], 2) for k, v in r.items()},
      "| SyNRA CC %.2f lat %.2f AP %.2f mm" % (r["SyNRA"]["rms_CC_z_mm"], r["SyNRA"]["rms_lateral_x_mm"], r["SyNRA"]["rms_AP_y_mm"]))
