"""compare_d95_fx2fix.py -- DVH comparison: baseline (wrong Fx_2 DVF) vs corrected.

Reuses the ROI rasterisation and the plan-dose resampling of
_lung_batch/fig_dvh_overlay.py: the evaluation geometry is identical to the one
used to produce the published numbers, so the comparison is
apples-to-apples.

The band is built as in the original diagnostic (D95% read on the p2.5 / p97.5
DVH curves), so the values are comparable with the 8.56 / 10.94 Gy of that
diagnostic. In addition it reports the band on the metric (D95% per sample, then
percentiles), which is the correct construction if an interval on D95% is wanted.

Usage:
    python compare_d95_fx2fix.py
    python compare_d95_fx2fix.py --baseline <dir> --fixed-dir <dir> --struct PTV54
"""
from __future__ import annotations

from dirunc_paths import LIB, LUNG_SBRT  # data roots, see README
import argparse
import os
import sys

import numpy as np

_LUNG_BATCH = LIB
sys.path.insert(0, _LUNG_BATCH)
os.environ.setdefault("MPLBACKEND", "Agg")

import pydicom  # noqa: E402
from fig_dvh_overlay import (  # noqa: E402
    STRUCTURES, RS_PATH, RD_PLAN,
    load_pct_post_rigid, rasterize_roi, resample_plan_dose, dvh_cumulative,
)

ROOT = rf"{LUNG_SBRT}"


def d_at_vol(centers, vol, target=95.0):
    """Dose at which the cumulative volume drops to `target` %."""
    c = np.asarray(centers, float)
    v = np.asarray(vol, float)
    idx = np.argsort(c)
    c, v = c[idx], v[idx]
    if v.max() < target:
        return float("nan")
    return float(np.interp(target, v[::-1], c[::-1]))


def dvh_stats(ens_dir, masks, dose_plan, edges, centers, target=95.0):
    """Returns {struct: dict(...)} for an accumulation out_dir."""
    mean_path = os.path.join(ens_dir, "dose_accumulated_fc.npy")
    per_path = os.path.join(ens_dir, "dose_acc_per_sample_fc.npy")
    for p in (mean_path, per_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    dose_mean = np.load(mean_path).astype(np.float32)
    dose_per = np.load(per_path, mmap_mode="r")
    K = int(dose_per.shape[0])

    names = [n for n, _ in STRUCTURES if n in masks]
    # OUTER k loop: the 9.4 GB memmap is read only once, not once
    # per structure.
    V = {n: np.empty((K, len(centers)), dtype=np.float32) for n in names}
    for k in range(K):
        dk = np.asarray(dose_per[k])
        for n in names:
            V[n][k] = dvh_cumulative(dk, masks[n], edges)

    out = {}
    for name in names:
        m = masks[name]
        v_plan = dvh_cumulative(dose_plan, m, edges)
        Vn = V[name]
        d95_per = np.array([d_at_vol(centers, Vn[k], target) for k in range(K)])
        v_mean = Vn.mean(axis=0)
        v_lo = np.percentile(Vn, 2.5, axis=0)
        v_hi = np.percentile(Vn, 97.5, axis=0)

        out[name] = dict(
            d_plan=d_at_vol(centers, v_plan, target),
            d_mean=d_at_vol(centers, v_mean, target),
            d_from_lo=d_at_vol(centers, v_lo, target),
            d_from_hi=d_at_vol(centers, v_hi, target),
            d95_per_lo=float(np.nanpercentile(d95_per, 2.5)),
            d95_per_hi=float(np.nanpercentile(d95_per, 97.5)),
            mean_dose_ens=float(dose_mean[m].mean()),
            mean_dose_plan=float(dose_plan[m].mean()),
            K=K,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=f"{ROOT}/ensemble100")
    ap.add_argument("--fixed-dir", default=f"{ROOT}/ensemble100_fx2fix")
    ap.add_argument("--bins", type=int, default=400)
    ap.add_argument("--dose-max", type=float, default=60.0)
    ap.add_argument("--target", type=float, default=95.0)
    args = ap.parse_args()

    print("Loading post-rigid pCT + structures...")
    ct, v2w_post, inv_predef = load_pct_post_rigid()
    nz, ny, nx = ct.shape
    rs = pydicom.dcmread(RS_PATH)
    masks = {}
    for name, _ in STRUCTURES:
        m = rasterize_roi(rs, name, ny, nx, nz, inv_predef, v2w_post)
        if m is None or int(m.sum()) == 0:
            print(f"  [warn] {name}: missing or empty, skipping")
            continue
        masks[name] = m
        print(f"  {name:>15s}: {int(m.sum())} voxels")

    dose_plan = resample_plan_dose(RD_PLAN, ct, v2w_post, inv_predef)
    print(f"  plan dose: [{dose_plan.min():.2f}, {dose_plan.max():.2f}] Gy")

    edges = np.linspace(0, args.dose_max, args.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    runs = {}
    for label, d in (("baseline (DVF Fx_3 su Fx_2)", args.baseline),
                     ("corretto (DVF Fx_2)", args.fixed_dir)):
        print(f"\nDVH on {d} ...")
        try:
            runs[label] = dvh_stats(d, masks, dose_plan, edges, centers, args.target)
        except FileNotFoundError as e:
            print(f"  [skip] missing {os.path.basename(str(e))}")

    t = int(args.target)
    print("\n" + "=" * 96)
    print(f"D{t}% [Gy] -- 'curve' band = D{t}% read on DVH p2.5/p97.5 "
          f"(as in the published numbers); 'metric' band = percentiles of D{t}% per sample")
    print("=" * 96)
    for name, _ in STRUCTURES:
        if name not in masks:
            continue
        rows = [(lab, r[name]) for lab, r in runs.items() if name in r]
        if not rows:
            continue
        print(f"\n{name}")
        print(f"  {'':30s} {'D'+str(t)+'% plan':>12s} {'D'+str(t)+'% acc':>12s} "
              f"{'bias':>9s} {'curve band':>13s} {'metric band':>15s}")
        for lab, s in rows:
            band_c = abs(s["d_from_lo"] - s["d_from_hi"])
            band_m = abs(s["d95_per_hi"] - s["d95_per_lo"])
            bias = s["d_mean"] - s["d_plan"]
            ratio = abs(bias) / band_c if band_c > 1e-9 else float("inf")
            print(f"  {lab:30s} {s['d_plan']:12.2f} {s['d_mean']:12.2f} "
                  f"{bias:+9.2f} {band_c:13.2f} {band_m:15.2f}   "
                  f"bias/band={ratio:.2f}x")
        if len(rows) == 2:
            (_, a), (_, b) = rows
            print(f"  {'--> change in D'+str(t)+'%':30s} "
                  f"{b['d_mean'] - a['d_mean']:+.2f} Gy "
                  f"({a['d_mean']:.2f} -> {b['d_mean']:.2f})")

    print("\n" + "-" * 96)
    print("Mean dose per structure [Gy]")
    for name, _ in STRUCTURES:
        if name not in masks:
            continue
        line = f"  {name:>15s}  piano={next(iter(runs.values()))[name]['mean_dose_plan']:6.2f}"
        for lab, r in runs.items():
            if name in r:
                line += f"   {lab.split(' ')[0]}={r[name]['mean_dose_ens']:6.2f}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
