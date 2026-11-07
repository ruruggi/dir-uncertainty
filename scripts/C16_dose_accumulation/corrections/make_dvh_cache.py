"""make_dvh_cache.py -- (re)generates _dvh_cache.npz in one or more accumulation out_dirs.

Needed by `_build_figure3_v15.py`, which reads `<RES>/{Heuristic,Smolders}/_dvh_cache.npz`.
Reuses `compute_or_load_cache` of `_lung_batch/fig_dvh_overlay_interactive.py` with
force_recompute=True, instead of launching the Tkinter viewer.

Parameters identical to those of the published caches (verified by reading
Results100/*/_dvh_cache.npz): 400 bins over 0-60 Gy, p2.5/p97.5 band.

Usage:
    python make_dvh_cache.py <dir1> [<dir2> ...]
"""
from __future__ import annotations

from dirunc_paths import LIB  # data roots, see README
import os
import sys

sys.path.insert(0, LIB)
sys.path.insert(0, LIB)
os.environ.setdefault("MPLBACKEND", "Agg")

BINS = 400
DOSE_MAX = 60.0


def main(dirs) -> int:
    from fig_dvh_overlay_interactive import compute_or_load_cache
    rc = 0
    for d in dirs:
        if not os.path.isdir(d):
            print(f"[skip] {d}: does not exist")
            rc = 1
            continue
        needed = ("dose_accumulated_fc.npy", "dose_acc_per_sample_fc.npy")
        if not all(os.path.exists(os.path.join(d, n)) for n in needed):
            print(f"[skip] {d}: missing {needed}")
            rc = 1
            continue
        print(f"=== {d}")
        compute_or_load_cache(d, BINS, DOSE_MAX, True)
        print(f"    written {os.path.join(d, '_dvh_cache.npz')}")
    return rc


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
