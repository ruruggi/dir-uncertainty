"""
convert_castillo_to_nifti.py — Convert DIR-LAB Castillo 4D-CT raw .img + landmark
.txt -> NIfTI + landmark layout compatible with discover_cases.

For each Case[1-10]Pack.zip in --raw_dir:
    1. Open the zip in memory; read the 10 phase volumes T00..T90 + landmarks
    2. For the inhale/exhale pair (T00, T50):
        - Reshape raw uint16 little-endian to (X, Y, Z) using CASE_META[N]
        - HU conversion: HU = raw - 1000 (DIR-LAB standard offset; verified
          on case1: air@raw=0 -> HU=-1000, water@raw=1000 -> HU=0, bone@raw=2000)
        - Build the LPS affine: diag(-sx, -sy, sz) + origin = (cx, cy, 0) with
          cx = +(X-1)*sx/2, cy = +(Y-1)*sy/2 for medical NIfTI-style centring
        - Save as <out_dir>/caseN_image1.nii.gz (T00) and
                  <out_dir>/caseN_image2.nii.gz (T50)
    3. Landmarks: read tab-separated integers (1-based MATLAB convention per
       Castillo PMB 2009) -> subtract 1 (-> 0-based numpy/itk) -> save in
       comma-separated decimal format compatible with load_pair (
       caseN_landmarks1.txt = T00 fixed, caseN_landmarks2.txt = T50 moving GT).

CASE_META: dimensions + voxel spacing from Castillo PMB 2009 Table 1 (verified
against file size: case 1 .img = 11.75 MB = 256x256x94 uint16 ✓).

Usage:
    python -m validation.convert_castillo_to_nifti \\
        --raw_dir "<DATASETS>/Castillo-DIR-LAB/raw" \\
        --out_dir "<DATASETS>/Castillo-DIR-LAB/nifti"
"""
from __future__ import annotations

import os
import argparse
import zipfile
import numpy as np
import nibabel as nib

# Castillo PMB 2009 Table 1 (verified vs file size of Case[1-10]Pack.zip
# unzipped contents): per-case dimensions (X, Y, Z in voxel)
# and voxel spacing (sx, sy, sz in mm).
CASE_META = {
    1:  {"dims": (256, 256,  94), "spacing": (0.97, 0.97, 2.5)},
    2:  {"dims": (256, 256, 112), "spacing": (1.16, 1.16, 2.5)},
    3:  {"dims": (256, 256, 104), "spacing": (1.15, 1.15, 2.5)},
    4:  {"dims": (256, 256,  99), "spacing": (1.13, 1.13, 2.5)},
    5:  {"dims": (256, 256, 106), "spacing": (1.10, 1.10, 2.5)},
    6:  {"dims": (512, 512, 128), "spacing": (0.97, 0.97, 2.5)},
    7:  {"dims": (512, 512, 136), "spacing": (0.97, 0.97, 2.5)},
    8:  {"dims": (512, 512, 128), "spacing": (0.97, 0.97, 2.5)},
    9:  {"dims": (512, 512, 128), "spacing": (0.97, 0.97, 2.5)},
   10:  {"dims": (512, 512, 120), "spacing": (0.97, 0.97, 2.5)},
}

HU_OFFSET = -1000   # raw_uint16 + HU_OFFSET = HU (DIR-LAB convention)


def _build_lps_affine(dims, spacing):
    """LPS-style affine: diag(-sx, -sy, +sz) + centred origin."""
    sx, sy, sz = spacing
    nx, ny, _ = dims
    A = np.zeros((4, 4), dtype=np.float64)
    A[0, 0] = -sx
    A[1, 1] = -sy
    A[2, 2] =  sz
    A[0, 3] = +(nx - 1) * sx / 2.0   # origin shift -> volume centered
    A[1, 3] = +(ny - 1) * sy / 2.0
    A[2, 3] =  0.0
    A[3, 3] = 1.0
    return A


def _read_img_from_zip(zf: zipfile.ZipFile, member: str, dims) -> np.ndarray:
    """Read a raw uint16 little-endian .img from the zip and reshape to (X, Y, Z)."""
    nx, ny, nz = dims
    expected_bytes = nx * ny * nz * 2
    raw = zf.read(member)
    if len(raw) != expected_bytes:
        raise ValueError(f"{member}: bytes={len(raw)}, expected {expected_bytes} "
                         f"per dims {dims}")
    arr_zyx = np.frombuffer(raw, dtype="<u2").reshape((nz, ny, nx))
    # raw layout slice-major (Z slowest, Y row-major, X col-major) -> transpose to (X, Y, Z)
    return np.ascontiguousarray(np.transpose(arr_zyx, (2, 1, 0)))


def _convert_to_hu(arr_uint16: np.ndarray) -> np.ndarray:
    """raw uint16 -> int16 HU."""
    return arr_uint16.astype(np.int32) + HU_OFFSET  # int32 to avoid overflow


def _read_landmarks_zip(zf: zipfile.ZipFile, member: str) -> np.ndarray:
    """Read a landmark file of 1-based integers (Castillo MATLAB convention) and convert
    to 0-based float voxel coordinates."""
    text = zf.read(member).decode("utf-8", errors="replace")
    pts = []
    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 3:
            pts.append([float(parts[0]) - 1.0,
                         float(parts[1]) - 1.0,
                         float(parts[2]) - 1.0])
    if not pts:
        raise ValueError(f"{member}: no landmark parsed")
    return np.asarray(pts, dtype=np.float64)


def convert_one_case(case_id: int, raw_dir: str, out_dir: str,
                       phase_fixed: str = "T00",
                       phase_moving: str = "T50",
                       n_landmarks: int = 300,
                       verbose: bool = True) -> dict:
    """Convert CaseNPack.zip -> caseN_image{1,2}.nii.gz + caseN_landmarks{1,2}.txt"""
    if case_id not in CASE_META:
        raise ValueError(f"case {case_id} not in CASE_META")
    meta = CASE_META[case_id]
    dims = meta["dims"]
    spacing = meta["spacing"]

    zip_path = os.path.join(raw_dir, f"Case{case_id}Pack.zip")
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(zip_path)

    if verbose:
        print(f"[case{case_id}] dims={dims} spacing={spacing}")

    affine = _build_lps_affine(dims, spacing)

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

        def _find_img(case_id_, phase):
            # Cases 1: case1_T00_s.img; cases 2-5: case2_T00-ssm.img;
            # cases 6-10: case6_T00.img. Match: contains "/Images/case{N}_{phase}"
            # and ends with .img.
            import re
            pat = re.compile(rf"/Images/case{case_id_}_{phase}[^/]*\.img$",
                              re.IGNORECASE)
            for n in names:
                if pat.search(n):
                    return n
            raise FileNotFoundError(f"img for case{case_id_} {phase} not found in {zip_path}")

        def _find_landmarks(case_id_, phase, n_lm):
            # Cases 1-5: ExtremePhases/Case1_300_T00_xyz.txt
            # Cases 6-10: extremePhases/case6_dirLab300_T00_xyz.txt
            import re
            pat = re.compile(rf"/[Ee]xtremePhases/[Cc]ase{case_id_}_(?:dirLab)?{n_lm}_{phase}_xyz\.txt$")
            for n in names:
                if pat.search(n):
                    return n
            raise FileNotFoundError(f"landmarks for case{case_id_} {phase} {n_lm} not found in {zip_path}")

        # Volumes
        img_t00_member = _find_img(case_id, phase_fixed)
        img_t50_member = _find_img(case_id, phase_moving)
        if verbose:
            print(f"  img1: {img_t00_member}")
            print(f"  img2: {img_t50_member}")

        arr_t00 = _read_img_from_zip(zf, img_t00_member, dims)
        arr_t50 = _read_img_from_zip(zf, img_t50_member, dims)
        hu_t00 = _convert_to_hu(arr_t00).astype(np.int16)
        hu_t50 = _convert_to_hu(arr_t50).astype(np.int16)

        if verbose:
            print(f"  HU T00: min={hu_t00.min()} mean={hu_t00.mean():.1f} max={hu_t00.max()}")
            print(f"  HU T50: min={hu_t50.min()} mean={hu_t50.mean():.1f} max={hu_t50.max()}")

        # Landmarks (300 ExtremePhases per extreme phase)
        lm_t00_member = _find_landmarks(case_id, phase_fixed, n_landmarks)
        lm_t50_member = _find_landmarks(case_id, phase_moving, n_landmarks)
        lm_t00 = _read_landmarks_zip(zf, lm_t00_member)
        lm_t50 = _read_landmarks_zip(zf, lm_t50_member)

        if verbose:
            print(f"  landmarks1 ({phase_fixed}): {len(lm_t00)} pts, "
                  f"voxel range [{lm_t00.min(axis=0).tolist()}, "
                  f"{lm_t00.max(axis=0).tolist()}]")
            print(f"  landmarks2 ({phase_moving}): {len(lm_t50)} pts")
            assert len(lm_t00) == len(lm_t50), "landmark count mismatch T00 vs T50"

    # Save
    os.makedirs(out_dir, exist_ok=True)
    img1_path = os.path.join(out_dir, f"case{case_id}_image1.nii.gz")
    img2_path = os.path.join(out_dir, f"case{case_id}_image2.nii.gz")
    lm1_path = os.path.join(out_dir, f"case{case_id}_landmarks1.txt")
    lm2_path = os.path.join(out_dir, f"case{case_id}_landmarks2.txt")

    nib.save(nib.Nifti1Image(hu_t00, affine), img1_path)
    nib.save(nib.Nifti1Image(hu_t50, affine), img2_path)
    np.savetxt(lm1_path, lm_t00, fmt="%.6f", delimiter=",")
    np.savetxt(lm2_path, lm_t50, fmt="%.6f", delimiter=",")

    if verbose:
        print(f"  saved {img1_path} ({os.path.getsize(img1_path)/1024/1024:.1f} MB)")
        print(f"  saved {img2_path} ({os.path.getsize(img2_path)/1024/1024:.1f} MB)")
        print(f"  saved {lm1_path}, {lm2_path}")

    return {
        "case_id": f"case{case_id}",
        "dims": dims,
        "spacing": spacing,
        "n_landmarks": int(len(lm_t00)),
        "img1_path": img1_path,
        "img2_path": img2_path,
        "lm1_path": lm1_path,
        "lm2_path": lm2_path,
    }


def main():
    p = argparse.ArgumentParser(description="Convert DIR-LAB Castillo 4D-CT raw -> NIfTI")
    p.add_argument("--raw_dir", required=True,
                    help="Dir containing CaseNPack.zip (1..10)")
    p.add_argument("--out_dir", required=True,
                    help="Output dir for caseN_image{1,2}.nii.gz + caseN_landmarks{1,2}.txt")
    p.add_argument("--cases", default=None,
                    help="Comma-separated list of case_ids (default: 1..10)")
    p.add_argument("--phase_fixed", default="T00")
    p.add_argument("--phase_moving", default="T50")
    p.add_argument("--n_landmarks", type=int, default=300)
    args = p.parse_args()

    if args.cases:
        ids = [int(c.strip().replace("case", "")) for c in args.cases.split(",")]
    else:
        ids = list(range(1, 11))

    print(f"=== Castillo conversion: {len(ids)} cases ===")
    print(f"  raw_dir: {args.raw_dir}")
    print(f"  out_dir: {args.out_dir}")
    results = []
    for cid in ids:
        try:
            r = convert_one_case(cid, args.raw_dir, args.out_dir,
                                  phase_fixed=args.phase_fixed,
                                  phase_moving=args.phase_moving,
                                  n_landmarks=args.n_landmarks)
            results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[case{cid}] FAIL {e}")

    print(f"\n=== DONE: {len(results)}/{len(ids)} converted ===")
    for r in results:
        print(f"  {r['case_id']}: {r['dims']} sp={r['spacing']} "
              f"n_landmarks={r['n_landmarks']}")


if __name__ == "__main__":
    main()
