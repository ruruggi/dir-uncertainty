"""For each ensemble (Heuristic, Smolders), find k such that
||sample_k - mean||_2 is at the 50th percentile of the K=100 distribution.
Save the result + the actual sample DVF for fast re-use by the figure script.

Outputs:
  <method>/frac_01/dvf_sample_median.npy   shape = mean (3, X, Y, Z)
  <method>/frac_01/dvf_sample_median.json  {"k_median": int, "l2_dist": ..., ...}
"""
from dirunc_paths import LUNG_SBRT  # data roots, see README
import os, sys, time, json
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError: pass
import numpy as np

# Override via env var to regenerate from an alternative run directory.
# Default = original behaviour.
ROOT = os.environ.get('FIG3_RES', rf'{LUNG_SBRT}/Results100')

# The arm names can be overridden, to compare two f calibrations
# (e.g. FIG3_ARMS='Heuristic_f211') without duplicating the results folder.
ARMS = [a for a in os.environ.get('FIG3_ARMS', 'Heuristic,Smolders').split(',') if a]

for method in ARMS:
    ens_path  = os.path.join(ROOT, method, 'frac_01', 'dvf_ensemble.npy')
    mean_path = os.path.join(ROOT, method, 'frac_01', 'dvf_mean_vec.npy')
    out_npy   = os.path.join(ROOT, method, 'frac_01', 'dvf_sample_median.npy')
    out_json  = os.path.join(ROOT, method, 'frac_01', 'dvf_sample_median.json')
    if os.path.isfile(out_npy) and os.path.isfile(out_json):
        meta = json.load(open(out_json))
        print(f'[skip] {method}: k_median={meta["k_median"]} '
              f'l2_dist={meta["l2_dist_at_median"]:.3f} mm '
              f'(min={meta["l2_dist_min"]:.3f}, max={meta["l2_dist_max"]:.3f})')
        continue
    print(f'\n=== {method} ===')
    print(f'Loading: {ens_path}')
    ens  = np.load(ens_path,  mmap_mode='r')
    mean = np.load(mean_path)
    K = ens.shape[0]
    print(f'  shape = {ens.shape}, K = {K}')
    dists = np.empty(K, dtype=np.float64)
    t0 = time.time()
    for k in range(K):
        diff = np.asarray(ens[k], dtype=np.float64) - mean
        dists[k] = float(np.sqrt(np.sum(diff * diff)))
        if (k + 1) % 10 == 0 or k == 0:
            print(f'  {k+1}/{K} samples — dist[{k}]={dists[k]:.3f}  ({time.time()-t0:.1f}s)',
                  flush=True)
    sorted_idx = np.argsort(dists)
    k_median = int(sorted_idx[K // 2])
    print(f'\n  L2 distances over K={K}: '
          f'min={dists.min():.3f}, p25={np.percentile(dists,25):.3f}, '
          f'median={np.median(dists):.3f}, p75={np.percentile(dists,75):.3f}, '
          f'max={dists.max():.3f}')
    print(f'  k_median = {k_median}  (l2_dist = {dists[k_median]:.3f})')
    print(f'  Saving sample to {out_npy} ...')
    sample = np.asarray(ens[k_median], dtype=np.float32)
    np.save(out_npy, sample)
    json.dump({
        'k_median': k_median,
        'l2_dist_at_median': float(dists[k_median]),
        'l2_dist_min': float(dists.min()),
        'l2_dist_max': float(dists.max()),
        'l2_dist_mean': float(dists.mean()),
        'l2_dist_p25': float(np.percentile(dists, 25)),
        'l2_dist_p75': float(np.percentile(dists, 75)),
        'K': K,
    }, open(out_json, 'w'), indent=2)
    print(f'  Saved.')

print('\nDONE.')
