#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
format_table2_tex.py  (PMB revision, Q1.3 / alpha layout)

Reads the canonical bin-0.25, no-LIM calibration JSON
    _synra_calibration_no_lim/table2_bin025_noLIM.json
and emits the manuscript Table 2 (`table2_recomputed.tex`) under the
"alpha" presentation agreed during revision round 1:

  * SyNRA reported on the FULL set (no LIM filter); the SyNRA<->other-engine
    advantage that the LIM filter created is gone (see R1.3 / paired-case).
  * Anchored on the convergence-filtered q regimes (q<=10 mm, q<=5 mm); the
    unfiltered "all" column is dropped because, when the DIR itself diverges,
    a calibration error is meaningless (DIR failure, not sigma_h failure).
  * Non-convergence is reported as a failure-mode via the per-regime counts n
    (out of 30 QA cases, 10 DIR-LAB cases).
  * Two f columns (f@q<=10, f@q<=5) since there is no longer a single "all" f.
  * Two LECE columns per regime: global (pooled, matches Table 3 method and the
    "within Smolders range" comparison) and 5-fold (generalisation check).
  * bin width 0.25 mm  -> consistent with section 2.5 and Table 3
    (the original Table 2 had inadvertently used 0.5 mm).
  * 5-fold LECE is reported N.A. where n<5 (a 5-fold CV is degenerate);
    only Abdominal-DIR-QA / SyNRA at q<=5 (n=3) triggers this.

Run via the project venv (numpy/skimage pinned); this script is pure-stdlib.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JSON_PATH = os.path.join(ROOT, "_synra_calibration_no_lim", "table2_bin025_noLIM.json")
OUT_PATH = os.path.join(ROOT, "table2_recomputed.tex")

DATASETS = ["Lung-DIR-QA", "Abdominal-DIR-QA", "Liver-DIR-QA", "DIR-LAB"]
ENGINES = ["SyNRA", "ConvexAdam", "uniGradICON", "Elastix"]
DATASET_LABEL = {
    "Lung-DIR-QA": "Lung-DIR-QA",
    "Abdominal-DIR-QA": "Abdominal-DIR-QA",
    "Liver-DIR-QA": "Liver-DIR-QA",
    "DIR-LAB": "DIR-LAB",
}
ENGINE_LABEL = {
    "SyNRA": "SyNRA",
    "ConvexAdam": "ConvexAdam",
    "uniGradICON": "uniGradICON",
    "Elastix": "Elastix",
}
MIN_N_5FOLD = 5  # below this a 5-fold CV is degenerate -> N.A.


def fmt_lece(cell, kind):
    """kind='global' -> pooled LECE; kind='5fold' -> mean+/-std (N.A. if n<5)."""
    if kind == "global":
        return f"{cell['global']:.2f}"
    if cell["n"] < MIN_N_5FOLD:
        return "N.A."
    return f"{cell['kf_mean']:.2f} $\\pm$ {cell['kf_std']:.2f}"


def main():
    with open(JSON_PATH, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    assert blob.get("bin_width") == 0.25, blob.get("bin_width")
    assert blob.get("lim_filter") is False, blob.get("lim_filter")
    data = blob["data"]

    L = []
    L.append(r"\setlength{\tabcolsep}{4pt}%")
    L.append(r"\resizebox{\textwidth}{!}{%")
    L.append(r"\begin{tabular}{ll cccc cccc}")
    L.append(r"\toprule")
    L.append(r" &  & \multicolumn{4}{c}{Landmarks converged at $q\le 10$\,mm}"
             r" & \multicolumn{4}{c}{Landmarks converged at $q\le 5$\,mm} \\")
    L.append(r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}")
    L.append(r" &  &  &  & \multicolumn{2}{c}{LECE (mm)} &  &  &"
             r" \multicolumn{2}{c}{LECE (mm)} \\")
    L.append(r"\cmidrule(lr){5-6}\cmidrule(lr){9-10}")
    L.append(r"Dataset & DIR engine & $n$ & $f$ & global & 5-fold"
             r" & $n$ & $f$ & global & 5-fold \\")
    L.append(r"\midrule")

    for di, ds in enumerate(DATASETS):
        for ei, eng in enumerate(ENGINES):
            c10 = data[ds][eng]["<=10"]
            c5 = data[ds][eng]["<=5"]
            dlabel = DATASET_LABEL[ds] if ei == 0 else ""
            row = (
                f"{dlabel:<16} & {ENGINE_LABEL[eng]:<11}"
                f" & {c10['n']:>2d} & {c10['f']:5.2f}"
                f" & {fmt_lece(c10, 'global')} & {fmt_lece(c10, '5fold')}"
                f" & {c5['n']:>2d} & {c5['f']:5.2f}"
                f" & {fmt_lece(c5, 'global')} & {fmt_lece(c5, '5fold')} \\\\"
            )
            L.append(row)
        if di < len(DATASETS) - 1:
            L.append(r"\midrule")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")

    out = "\n".join(L) + "\n"
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"Wrote {OUT_PATH}")
    print(out)


if __name__ == "__main__":
    main()
