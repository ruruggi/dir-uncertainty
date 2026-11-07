# -*- coding: utf-8 -*-
"""
q22_step4_table2_table4_fig1.py -- step 4 of the recalculation order (2026-09-28): Table 2 (tab:calibration), Table 4
(tab:spearman) and Figure 1 (reliability diagrams) rebuilt with the body-local sigma (sigma_BL) of all 16 (dataset, engine) cells,
<DATASETS>/_sigma_BL_20260927/<same relative path as the submitted results>.

Same computations as the submitted generators: rebuild_table2_bin025_noLIM.cal (no LIM, bin 0.25, q <= 10 / 5 mm on the per-case
final RMSE TRE, 5-fold seed 42), Spearman rho on the pooled landmarks + per-case mean +- sd (all cases), make_figure1_4x4 panels
(q <= 10 mm). Controls: (1) submitted paths reproduce Tables/Table2.tex and Tables/Table4.tex; (2) TRE identical per case between
submitted and sigma_BL folders; (3) Figure 1 panel values = Table 2 q<=10 columns.
Outputs: ./_out_q22_step4/ (step4.json, Table2_rows_draft.tex, Table4_rows_draft.tex), ../../Figures/_da_approvare/Figure1_BL_draft.{png,tiff}.
Usage (venv): python q22_step4_table2_table4_fig1.py
"""
from dirunc_paths import DATASETS, LIB  # data roots, see README
import os, sys, types, json, re
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = LIB; PS = LIB + "/validation"
sys.path.insert(0, SCRIPTS); sys.path.insert(0, PS)
if "validation" not in sys.modules:
    _v = types.ModuleType("validation"); _v.__path__ = [PS]; sys.modules["validation"] = _v
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from validation.build_v11_data import ENGINE_PATHS
from validation.calibrate import load_all_landmarks
from rebuild_table2_bin025_noLIM import cal
import make_figure1_4x4 as F1

DS_ROOT = f"{DATASETS}"; BL_ROOT = f"{DS_ROOT}/_sigma_BL_20260927"
TABLES = os.path.join(os.path.dirname(SCRIPTS), "Tables")
OUT = os.path.join(HERE, "_out_q22_step4"); os.makedirs(OUT, exist_ok=True)
FIG = os.path.join(os.path.dirname(SCRIPTS), "Figures", "_da_approvare"); os.makedirs(FIG, exist_ok=True)
DATASETS = ["Lung-DIR-QA", "Abdominal-DIR-QA", "Liver-DIR-QA", "DIR-LAB"]
ENGINES = ["SyNRA", "ConvexAdam", "uniGradICON", "Elastix"]
F1COL = {"Lung-DIR-QA": "Lung", "Abdominal-DIR-QA": "Abdomen", "Liver-DIR-QA": "Liver", "DIR-LAB": "DIR-LAB"}
bl = lambda p: os.path.join(BL_ROOT, os.path.relpath(p, DS_ROOT)).replace("\\", "/")


def num(s):
    s = s.strip().replace("$", "").replace("\\pm", "±").replace(" ", "")
    return None if s in ("N.A.", "") else s


def parse_table(path, ncols):
    """rows of a booktabs table -> {(dataset, engine): [cells]} (dataset carried over when blank)."""
    rows, ds = {}, None
    for line in open(path, encoding="utf-8"):
        if "&" not in line or "\\\\" not in line or "Dataset" in line or "multicolumn" in line:
            continue
        c = [x.strip() for x in line.split("\\\\")[0].split("&")]
        if len(c) != ncols:
            continue
        ds = c[0] or ds; d = re.sub(r"\s*\(Castillo\)", "", ds).strip()
        rows[(d, c[1])] = c[2:]
    return rows


def table2(paths):
    out = {}
    for ds in DATASETS:
        for eng in ENGINES:
            cases = load_all_landmarks(paths[eng][ds])
            rmse = {c.case_id: float(np.sqrt(np.mean(c.tre_mm ** 2))) for c in cases}
            out[(ds, eng)] = {q: cal([c for c in cases if rmse[c.case_id] <= q]) for q in (10.0, 5.0)}
    return out


def table4(paths):
    out = {}
    for ds in DATASETS:
        for eng in ENGINES:
            cases = load_all_landmarks(paths[eng][ds])
            t = np.concatenate([c.tre_mm for c in cases]); s = np.concatenate([c.sigma_raw for c in cases])
            pc = np.array([spearmanr(c.sigma_raw, c.tre_mm)[0] for c in cases])
            out[(ds, eng)] = {"n": len(cases), "n_lm": int(t.size), "pool": float(spearmanr(s, t)[0]),
                              "pc_mean": float(np.nanmean(pc)), "pc_sd": float(np.nanstd(pc, ddof=0))}   # population sd, as the submitted Table 4
    return out


sub = {e: ENGINE_PATHS[e]["datasets"] for e in ENGINES}
new = {e: {ds: bl(p) for ds, p in sub[e].items()} for e in ENGINES}
for e in ENGINES:
    for ds in DATASETS:
        assert os.path.isdir(new[e][ds]), f"missing sigma_BL cell {new[e][ds]}"
ok_all = True


def check(tag, ok, detail=""):
    global ok_all
    ok_all &= bool(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {tag} {detail}")


print("Control 2: same TRE per case, submitted vs sigma_BL")
for e in ENGINES:
    for ds in DATASETS:
        a = {c.case_id: c.tre_mm for c in load_all_landmarks(sub[e][ds])}; b = {c.case_id: c.tre_mm for c in load_all_landmarks(new[e][ds])}
        d = max(float(np.max(np.abs(a[k] - b[k]))) if a[k].size == b[k].size else np.inf for k in a) if set(a) == set(b) else np.inf
        check(f"{ds:17s} {e:12s}", d < 1e-9, f"cases {len(a)}/{len(b)} max|dTRE| {d:.1e}")

print("Control 1: submitted paths reproduce Table2.tex / Table4.tex")
T2s, T4s = table2(sub), table4(sub)
tex2, tex4 = parse_table(os.path.join(TABLES, "Table2.tex"), 10), parse_table(os.path.join(TABLES, "Table4.tex"), 6)
for (ds, e), r in T2s.items():
    c = tex2[(ds, e)]
    for j, q in ((0, 10.0), (4, 5.0)):
        x = r[q]; kf = num(c[j + 3])
        got = (x["n"], round(x["f"], 2), round(x["global"], 2), f"{x['kf_mean']:.2f}±{x['kf_std']:.2f}" if x["n"] >= 5 else None)
        ref = (int(c[j]), float(c[j + 1]), float(c[j + 2]), kf)
        check(f"T2 {ds:17s} {e:12s} q<={q:.0f}", got == ref, f"got {got} tex {ref}" if got != ref else "")
for (ds, e), r in T4s.items():
    c = tex4[(ds, e)]
    got = (r["n"], f"{r['pool']:+.3f}", f"{r['pc_mean']:+.3f}±{r['pc_sd']:.3f}")
    ref = (int(c[0]), num(c[2]), num(c[3]))
    check(f"T4 {ds:17s} {e:12s}", got == ref, f"got {got} tex {ref}" if got != ref else "")

T2n, T4n = table2(new), table4(new)
fmt_kf = lambda x: f"${x['kf_mean']:.2f} \\pm {x['kf_std']:.2f}$" if x["n"] >= 5 else "N.A."
rows2, rows4 = [], []
print("\nTable 2 NEW (sigma_BL) vs submitted:  n  f  global  5-fold   | q<=5 ...")
for ds in DATASETS:
    for i, e in enumerate(ENGINES):
        a, b = T2n[(ds, e)], T2s[(ds, e)]
        print(f"  {ds:17s} {e:12s} " + " | ".join(f"q{q:.0f}: n {a[q]['n']:2d} f {a[q]['f']:5.2f} ({b[q]['f']:5.2f}) glob {a[q]['global']:.2f} ({b[q]['global']:.2f}) "
                                                     f"5f {a[q]['kf_mean']:.2f} ({b[q]['kf_mean']:.2f})" for q in (10.0, 5.0)))
        rows2.append(f"{ds if i == 0 else '':17s}& {e:12s}& {a[10.0]['n']:2d} & {a[10.0]['f']:5.2f} & {a[10.0]['global']:.2f} & {fmt_kf(a[10.0])} & "
                     f"{a[5.0]['n']:2d} & {a[5.0]['f']:5.2f} & {a[5.0]['global']:.2f} & {fmt_kf(a[5.0])} \\\\")
    rows2.append("\\midrule")
print("\nTable 4 NEW (sigma_BL) vs submitted:  rho_pool  per-case")
for ds in DATASETS:
    for e in ENGINES:
        a, b = T4n[(ds, e)], T4s[(ds, e)]
        print(f"  {ds:17s} {e:12s} pool {a['pool']:+.3f} ({b['pool']:+.3f})  per-case {a['pc_mean']:+.3f} ± {a['pc_sd']:.3f} ({b['pc_mean']:+.3f} ± {b['pc_sd']:.3f})")
        dsl = ds if ds != "DIR-LAB" else "DIR-LAB (Castillo)"
        rows4.append(f"{dsl} & {e:12s}& {a['n']} & {a['n_lm']:,} & ${a['pool']:+.3f}$ & ${a['pc_mean']:+.3f} \\pm {a['pc_sd']:.3f}$ \\\\".replace(",", "\\,"))

# Figure 1 (sigma_BL), q <= 10 mm, same panels as make_figure1_4x4
fig, axes = plt.subplots(4, 4, figsize=(16.0, 13.6), constrained_layout=True)
print("\nControl 3: Figure 1 panels = Table 2 q<=10")
for r, (eng, color, _) in enumerate(F1.ENGINE_ROWS):
    for c, ds in enumerate(DATASETS):
        ax = axes[r][c]; res = F1._compute_panel(new[eng][ds], 10.0, 0.25)
        F1._draw_panel(ax, res["diag"], color, f"n = {res['n_cases']}\nlm = {res['n_landmarks']:,}\nf = {res['f_global']:.2f}\nLECE = {res['lece_global']:.2f}")
        if r == 0: ax.set_title(F1.COL_TITLE[F1COL[ds]], fontsize=11, fontweight="bold")
        if c == 0: ax.set_ylabel(f"{eng}\nRMSE (mm)", fontsize=10, fontweight="bold")
        if r == 3: ax.set_xlabel(r"Predicted $\sigma_p$ (mm)", fontsize=10, fontweight="bold")
        t = T2n[(ds, eng)][10.0]
        check(f"F1 {ds:17s} {eng:12s}", res["n_cases"] == t["n"] and abs(res["f_global"] - t["f"]) < 1e-9 and abs(res["lece_global"] - t["global"]) < 1e-9)
fig.savefig(os.path.join(FIG, "Figure1_BL_draft.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(FIG, "Figure1_BL_draft.tiff"), dpi=800, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"}); plt.close(fig)

ser = lambda T: {f"{k[0]}|{k[1]}": v for k, v in T.items()}
json.dump({"controls_pass": bool(ok_all), "table2_new": {k: {str(q): x for q, x in v.items()} for k, v in ser(T2n).items()},
           "table2_submitted": {k: {str(q): x for q, x in v.items()} for k, v in ser(T2s).items()},
           "table4_new": ser(T4n), "table4_submitted": ser(T4s)}, open(os.path.join(OUT, "step4.json"), "w"), indent=2)
open(os.path.join(OUT, "Table2_rows_draft.tex"), "w", encoding="utf-8").write("\n".join(rows2[:-1]) + "\n")
open(os.path.join(OUT, "Table4_rows_draft.tex"), "w", encoding="utf-8").write("\n".join(rows4) + "\n")
print(f"\nOVERALL CONTROLS: {'PASS' if ok_all else 'FAIL'}  -> {OUT} ; Figure 1 draft -> {FIG}")
