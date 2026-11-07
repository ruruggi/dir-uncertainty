"""Compute the ensemble mean DVF (3, Z, Y, X) for Heuristic and Smolders
(frac_01 only). Streamed average across K=100 samples to limit RAM.

Outputs: <method_dir>/frac_01/dvf_mean_vec.npy
"""
from dirunc_paths import LUNG_SBRT  # data roots, see README
import os, sys, time
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
    ens_path = os.path.join(ROOT, method, 'frac_01', 'dvf_ensemble.npy')
    out_path = os.path.join(ROOT, method, 'frac_01', 'dvf_mean_vec.npy')
    if os.path.isfile(out_path):
        print(f'[skip] {out_path} already exists')
        continue
    print(f'\n=== {method} ===')
    print(f'Loading memmap: {ens_path}')
    ens = np.load(ens_path, mmap_mode='r')
    K = ens.shape[0]
    print(f'  shape = {ens.shape}, K = {K}, dtype = {ens.dtype}')
    # Streaming mean: sum chunks of 10 samples, divide at the end
    out = np.zeros(ens.shape[1:], dtype=np.float64)
    chunk = 5
    t0 = time.time()
    for k0 in range(0, K, chunk):
        k1 = min(K, k0 + chunk)
        block = np.asarray(ens[k0:k1], dtype=np.float64)
        out += block.sum(axis=0)
        print(f'  processed {k1}/{K} samples  ({time.time()-t0:.1f}s)', flush=True)
    out /= K
    out = out.astype(np.float32)
    np.save(out_path, out)
    print(f'Saved: {out_path}  ({os.path.getsize(out_path)/1024/1024:.0f} MB)')
print('\nDONE.')
