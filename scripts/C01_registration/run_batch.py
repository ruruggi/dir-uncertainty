"""
validation/run_batch.py — Runs the run_pair pipeline on all cases of the
Abdominal-DIR-QA dataset in parallel, with automatic resume.

Parallelism strategy (CPU-only, ANTs SyN):
    - N workers = ProcessPoolExecutor with the 'spawn' start method (Windows-safe)
    - each worker sets ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS BEFORE importing
      antspyx, so as to limit the ITK thread pool and not saturate the CPU
      when several workers run at the same time.
    - default for a 28-core / 128 GB workstation: 7 workers x 4 threads = 28 threads

Resume:
    - for each case, if <out_root>/<case_id>/summary.json exists and contains
      n_landmarks > 0, the case is skipped.
    - use --force to redo it.
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# make the script runnable standalone
if __package__ is None or __package__ == "":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    sys.path.insert(0, os.path.join(os.path.dirname(here), "dl_arch"))


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------
def _worker(
    case_id: str,
    case_dir: str,
    out_dir: str,
    type_of_transform: str,
    reg_iterations,
    S_scaling: float,
    save_dicom: bool,
    itk_threads: int,
    prealign: str = "none",
) -> dict:
    """Module-level function so that it is picklable by ProcessPoolExecutor."""
    # IMPORTANT: set ALL the threading env vars BEFORE importing antspyx,
    # because ANTs/ITK read them when the thread pool is initialised. Forcing
    # ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS alone is NOT ENOUGH: in production
    # we observed massive thread oversubscription when only this variable
    # was set. Also setting OMP/MKL/OpenBLAS/NUMEXPR brings the effective
    # thread count back to the requested value.
    n = str(int(itk_threads))
    for var in ("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "NUMEXPR_MAX_THREADS",
                "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = n

    # path import
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(here))
    sys.path.insert(0, os.path.join(os.path.dirname(here), "dl_arch"))

    # Force CPU for the heuristic in the worker (a GPU is often fine with 1 worker;
    # when 7 workers run together, running them all on GPU 0 is problematic
    # for memory. The heuristic is fast on CPU anyway, ~30-60 s/case.)
    import torch
    if torch.cuda.is_available():
        # GPU 0 could be made visible to the first worker only, or the GPU
        # disabled altogether. For simplicity we use the CPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from validation.run_pair import run_one

    case_out = os.path.join(out_dir, case_id)
    os.makedirs(case_out, exist_ok=True)
    summary = run_one(
        case_dir=case_dir,
        out_dir=case_out,
        case_id=case_id,
        type_of_transform=type_of_transform,
        reg_iterations=reg_iterations,
        S_scaling=S_scaling,
        prealign=prealign,
        save_dicom=save_dicom,
        device="cpu",
        verbose=True,
    )
    return summary


def _is_done(case_out: str) -> bool:
    summary_path = os.path.join(case_out, "summary.json")
    landmarks_path = os.path.join(case_out, "landmarks.npz")
    if not (os.path.isfile(summary_path) and os.path.isfile(landmarks_path)):
        return False
    try:
        with open(summary_path) as f:
            s = json.load(f)
        return s.get("n_landmarks", 0) > 0
    except Exception:
        return False


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run ANTs SyN + heuristic+residual on all cases of the "
                    "Abdominal-DIR-QA dataset, in parallel."
    )
    parser.add_argument("--root_dir", required=True,
                        help="Dataset root folder (flat NIfTI or subfolders).")
    parser.add_argument("--out_root", required=True,
                        help="Output folder for the per-case results.")
    parser.add_argument("--cases", default=None,
                        help="Comma-separated list of case_ids to process. "
                        "Default: all detected cases.")
    parser.add_argument("--n_workers", type=int, default=7,
                        help="Number of parallel ANTs processes (default 7).")
    parser.add_argument("--itk_threads", type=int, default=4,
                        help="ITK threads per worker (default 4). "
                        "n_workers x itk_threads should be ~= n_physical_cores.")
    parser.add_argument("--type_of_transform", default="SyN")
    parser.add_argument("--reg_iterations", default="100,70,50,20",
                        help="Comma-separated, e.g. '100,70,50,20'.")
    parser.add_argument("--S_scaling", "--c_scaling", type=float, default=1.0,
                        dest="S_scaling",
                        help="Calibration factor (default 1.0 = none). "
                        "--c_scaling is a deprecated alias.")
    parser.add_argument("--no_dicom", action="store_true",
                        help="Disable saving the DVF as DICOM (saves ~300 MB/case).")
    parser.add_argument("--force", action="store_true",
                        help="Also rerun cases that are already completed.")
    parser.add_argument("--prealign", choices=["none", "procrustes"],
                        default="none",
                        help="Rigid pre-alignment for each case before "
                        "SyN. 'procrustes' = Kabsch SVD on the landmarks (auto-batch, "
                        "requires an annotated dataset). 'none' = no prealign.")
    args = parser.parse_args()

    # discover
    from validation.load_abdominal_dir_qa import discover_cases
    all_cases = discover_cases(args.root_dir)

    if args.cases:
        wanted = set(c.strip() for c in args.cases.split(","))
        all_cases = [c for c in all_cases if c[0] in wanted]
        if not all_cases:
            print(f"ERROR: no case found for --cases={args.cases}")
            return

    # filter for resume
    todo = []
    skipped = []
    for case_id, case_dir in all_cases:
        case_out = os.path.join(args.out_root, case_id)
        if not args.force and _is_done(case_out):
            skipped.append(case_id)
            continue
        todo.append((case_id, case_dir))

    reg_iterations = tuple(int(x) for x in args.reg_iterations.split(","))

    print("=" * 70)
    print(f"Abdominal-DIR-QA batch run")
    print("=" * 70)
    print(f"  root_dir   : {args.root_dir}")
    print(f"  out_root   : {args.out_root}")
    print(f"  total cases: {len(all_cases)}")
    print(f"  skipped    : {len(skipped)} ({skipped[:5]}{'...' if len(skipped) > 5 else ''})")
    print(f"  todo       : {len(todo)} ({[c[0] for c in todo[:5]]}{'...' if len(todo) > 5 else ''})")
    print(f"  workers    : {args.n_workers} x {args.itk_threads} ITK threads "
          f"= {args.n_workers * args.itk_threads} threads in total")
    print(f"  iterations : {reg_iterations}")
    print(f"  transform  : {args.type_of_transform}")
    print(f"  S_scaling  : {args.S_scaling}")
    print(f"  prealign   : {args.prealign}")
    print(f"  save_dicom : {not args.no_dicom}")
    print()

    if not todo:
        print("Nothing to do.")
        return

    os.makedirs(args.out_root, exist_ok=True)

    # ProcessPoolExecutor with spawn for Windows
    ctx = mp.get_context("spawn")
    t0 = time.time()
    completed = 0
    failed = []

    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx) as ex:
        futures = {
            ex.submit(_worker,
                      case_id, case_dir, args.out_root,
                      args.type_of_transform, reg_iterations,
                      args.S_scaling, not args.no_dicom, args.itk_threads,
                      args.prealign): case_id
            for case_id, case_dir in todo
        }

        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                s = fut.result()
                completed += 1
                elapsed = time.time() - t0
                avg = elapsed / completed
                eta = avg * (len(todo) - completed)
                print(f"[{completed}/{len(todo)}] OK {cid:<10} "
                      f"(initial RMSE TRE {s['initial_rmse_tre_mm']:.2f} -> "
                      f"final {s['final_rmse_tre_mm']:.2f} mm in "
                      f"{s['total_time_s']:.0f}s) | "
                      f"elapsed {elapsed/60:.1f}min, ETA {eta/60:.1f}min",
                      flush=True)
            except Exception as e:
                completed += 1
                failed.append((cid, str(e)))
                print(f"[{completed}/{len(todo)}] FAIL {cid}: {e}", flush=True)

    total_time = time.time() - t0
    print()
    print("=" * 70)
    print(f"Batch finished in {total_time/60:.1f} min")
    print(f"  completed: {completed - len(failed)}")
    print(f"  failed    : {len(failed)}")
    for cid, err in failed:
        print(f"    {cid}: {err}")

    # Aggregate summary
    aggregate_path = os.path.join(args.out_root, "batch_summary.json")
    aggregate = {
        "root_dir": args.root_dir,
        "out_root": args.out_root,
        "n_cases_total": len(all_cases),
        "n_cases_done": completed - len(failed) + len(skipped),
        "n_cases_failed": len(failed),
        "failed_cases": [f[0] for f in failed],
        "n_workers": args.n_workers,
        "itk_threads": args.itk_threads,
        "type_of_transform": args.type_of_transform,
        "reg_iterations": list(reg_iterations),
        "S_scaling": args.S_scaling,
        "prealign": args.prealign,
        "elapsed_minutes": total_time / 60,
    }
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nbatch_summary saved -> {aggregate_path}")


if __name__ == "__main__":
    main()
