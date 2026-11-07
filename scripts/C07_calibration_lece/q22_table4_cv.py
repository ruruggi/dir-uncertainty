# -*- coding: utf-8 -*-
"""Table 4 v3 (2026-10-01, R2.2 option A approved): body-local coefficient of variation of the predicted sigma_p per (dataset, engine) cell,
added as a column to Table 4. Definition = step 5 block B (rho ceiling): CV = population sd / mean of the per-landmark sigma_p (sigma_raw,
i.e. before the scalar f, to which the CV is invariant) pooled over all landmarks of all cases of the cell (no convergence filter, as Table 4).
Control: the four DIR-LAB values must equal step5.json B.new cv (0.111/0.103/0.094/0.107).
Output: _out_q22_step4/table4_cv.json; ../../Tables_modificabili/Table4_v3.tex (= Table4_v2 + CV column, new cells in red).
Usage (venv): python q22_table4_cv.py"""
from dirunc_paths import DATASETS, LIB  # data roots, see README
import os, sys, types, json, re
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = LIB; PS = LIB + "/validation"
sys.path.insert(0, SCRIPTS); sys.path.insert(0, PS)
if "validation" not in sys.modules:
    _v = types.ModuleType("validation"); _v.__path__ = [PS]; sys.modules["validation"] = _v
import numpy as np
from validation.build_v11_data import ENGINE_PATHS
from validation.calibrate import load_all_landmarks
DS = f"{DATASETS}"; BLR = f"{DS}/_sigma_BL_20260927"
bl = lambda p: os.path.join(BLR, os.path.relpath(p, DS)).replace("\\", "/")
ENG = ["SyNRA", "ConvexAdam", "uniGradICON", "Elastix"]; DATASETS = ["Lung-DIR-QA", "Abdominal-DIR-QA", "Liver-DIR-QA", "DIR-LAB"]
TAB = os.path.normpath(os.path.join(SCRIPTS, "..", "Tables_modificabili")); OUT = os.path.join(HERE, "_out_q22_step4")
S5 = json.load(open(os.path.join(HERE, "_out_q22_step5", "step5.json")))["B"]["new"]
res = {}
for ds in DATASETS:
    for e in ENG:
        cases = load_all_landmarks(bl(ENGINE_PATHS[e]["datasets"][ds]))
        sig = np.concatenate([c.sigma_raw for c in cases]); sig = sig[np.isfinite(sig) & (sig > 0)]
        pc = np.array([float(c.sigma_raw.std() / c.sigma_raw.mean()) for c in cases])
        res[f"{ds}|{e}"] = {"n_cases": len(cases), "n_lm": int(sig.size), "cv_pooled": float(sig.std() / sig.mean()),
                            "cv_percase_mean": float(pc.mean()), "cv_percase_sd": float(pc.std())}
        print(f"{ds:17s} {e:12s} n_lm {sig.size:6d} CV pooled {res[f'{ds}|{e}']['cv_pooled']:.3f}  per-case {pc.mean():.3f} ± {pc.std():.3f}")
for e in ENG:
    a, b = res[f"DIR-LAB|{e}"]["cv_pooled"], S5[e]["cv"]
    assert abs(a - b) < 5e-4, (e, a, b)
print("control DIR-LAB CV = step5 B: PASS")
json.dump(res, open(os.path.join(OUT, "table4_cv.json"), "w"), indent=1)

src = open(os.path.join(TAB, "Table4_v2.tex"), encoding="utf-8").read().splitlines()
out = []
for line in src:
    if line.startswith(r"\begin{tabular}"):
        line = line.replace("{llrrcc}", "{llrrccc}")
    elif line.startswith("Dataset &"):
        line = line.replace(r"$\rho$ per-case (mean $\pm$ sd) \\", r"$\rho$ per-case (mean $\pm$ sd) & \textcolor{red}{CV of $\sigma_{p}$} \\")
    elif "&" in line and line.rstrip().endswith(r"\\"):
        cells = [c.strip() for c in line.split("\\\\")[0].split("&")]
        ds = re.sub(r"\s*\(Castillo\)", "", cells[0]).strip(); e = cells[1]
        cv = res[f"{ds}|{e}"]["cv_pooled"]
        line = line.rstrip()[:-2].rstrip() + f" & \\textcolor{{red}}{{{cv:.2f}}} \\\\"
    out.append(line)
open(os.path.join(TAB, "Table4_v3.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("[out]", os.path.join(TAB, "Table4_v3.tex"))
cvs = [v["cv_pooled"] for v in res.values()]; print(f"CV range over 16 cells: {min(cvs):.3f}-{max(cvs):.3f}")
