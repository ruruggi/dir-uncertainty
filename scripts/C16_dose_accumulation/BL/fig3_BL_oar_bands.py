# -*- coding: utf-8 -*-
"""
fig3_BL_oar_bands.py -- 2026-09-29 D: DVH band width per structure from the _dvh_cache.npz of each arm (400 bins, 0-60 Gy, band p2.5/p97.5),
as the check of 28/9 (dvh_bands_by_structure.json, done ad hoc then). Reports, per arm and structure: mean band width in volume-% over the
dose axis where the plan DVH is between 1 and 99 %, and the band of D2% (Gy) read on the lo/hi curves.
Usage: FIG3_RES=<RESULTS_DATA>/Fig3_BL_20260929 python fig3_BL_oar_bands.py [arm1,arm2,...]   -> ./_out_fig3_BL/dvh_bands_by_structure<FIG3_TAG>.json
"""
from dirunc_paths import RESULTS_DATA  # data roots, see README
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("FIG3_RES", rf"{RESULTS_DATA}/Fig3_BL_20260929"); TAG = os.environ.get("FIG3_TAG", "_v2")
ARMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Heuristic_aniso", "Heuristic_iso", "Smolders"]


def d_at(v, centers, level):
    """dose at which the (decreasing) DVH crosses `level` (volume-%), linear interpolation."""
    v = np.asarray(v, float); idx = np.where(v <= level)[0]
    if idx.size == 0:
        return float(centers[-1])
    j = int(idx[0])
    if j == 0:
        return float(centers[0])
    return float(centers[j - 1] + (v[j - 1] - level) / (v[j - 1] - v[j] + 1e-12) * (centers[j] - centers[j - 1]))


out = {}
for arm in ARMS:
    p = os.path.join(RES, arm, "_dvh_cache.npz")
    if not os.path.exists(p):
        print(f"[skip] {arm}: no cache"); continue
    z = np.load(p, allow_pickle=True); c = z["centers"]; names = [str(n) for n in z["names"]]
    out[arm] = {}
    for i, n in enumerate(names):
        vp, lo, hi, vm = (np.asarray(z[k][i], float) for k in ("v_plan", "v_lo", "v_hi", "v_mean"))
        vp = vp * 100 if vp.max() <= 1.0 else vp; lo = lo * 100 if lo.max() <= 1.0 else lo; hi = hi * 100 if hi.max() <= 1.0 else hi
        vm = vm * 100 if vm.max() <= 1.0 else vm
        sel = (vp > 1) & (vp < 99)
        out[arm][n] = {"band_mean_volpct": float((hi - lo)[sel].mean()) if sel.any() else 0.0,
                       "D2_band_Gy": abs(d_at(hi, c, 2.0) - d_at(lo, c, 2.0)), "D95_plan": d_at(vp, c, 95.0), "D95_mean": d_at(vm, c, 95.0),
                       "D95_band_Gy": abs(d_at(hi, c, 95.0) - d_at(lo, c, 95.0))}
json.dump(out, open(os.path.join(HERE, "_out_fig3_BL", f"dvh_bands_by_structure{TAG}.json"), "w"), indent=2)
print(f"{'structure':14s}" + "".join(f"{a:>26s}" for a in out))
for n in next(iter(out.values())):
    print(f"{n:14s}" + "".join(f"  band {out[a][n]['band_mean_volpct']:5.1f} %  D2 {out[a][n]['D2_band_Gy']:4.1f} Gy" for a in out))
print("D95 (plan / mean / band):", {a: (round(out[a]["PTV54"]["D95_plan"], 2), round(out[a]["PTV54"]["D95_mean"], 2), round(out[a]["PTV54"]["D95_band_Gy"], 2)) for a in out if "PTV54" in out[a]})
