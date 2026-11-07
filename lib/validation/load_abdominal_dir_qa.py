"""
validation/load_abdominal_dir_qa.py — Loader for the Abdominal-DIR-QA dataset
(Criscuolo et al. 2025, Med. Phys.; Zenodo record 14362785).

Supports both file formats published by the dataset:

    - MATLAB .mat (Tutorial archive):
        img1.mat, img2.mat (variables 'img1', 'img2')
        example_landmarks.mat (variables landmarks1, landmarks2,
                               voxelsize1, voxelsize2)
        image dims = [coronal, sagittal, axial] = (Y, X, Z)
        landmarks  = (N, 3) pixel indices, 1-based, same axis order

    - NIfTI archive (Zenodo nifti_format.zip), flat layout:
        caseXX_img1.nii.gz                    fixed CT
        caseXX_img2.nii.gz                    moving CT
        caseXX_landmarks1.txt                 fixed lm  (CSV, 0-based, X Y Z)
        caseXX_landmarks2.txt                 moving lm (CSV, 0-based, X Y Z)
        caseXX_flagged_landmarks{1,2}.txt     QA-flagged lm (excluded by default)

NB on intensity: despite the README mentioning "HU + 1000", inspection of both
.mat and NIfTI archives shows values are already in true HU (air at -1024).
The loader therefore does NOT apply any offset.

The loader normalises everything to the project's convention:
    - image array shape (X, Y, Z) in HU, float32
    - voxel-to-world matrix (axis-aligned, origin = (0,0,0))
    - landmarks both as 0-based voxel indices (X, Y, Z) and as world mm coords

References:
    Criscuolo T et al. A vessel bifurcation landmark pair dataset for
    abdominal CT deformable image registration (DIR) validation. Medical
    Physics, 2025. DOI: 10.1002/mp.17907
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.io import loadmat


@dataclass
class AbdominalDIRQAPair:
    """Container for one (fixed, moving) pair plus landmarks, in project convention."""
    case_id: str
    fixed_arr: np.ndarray
    moving_arr: np.ndarray
    spacing_fixed: Tuple[float, float, float]
    spacing_moving: Tuple[float, float, float]
    lm_fixed_voxel: np.ndarray
    lm_moving_voxel: np.ndarray
    lm_fixed_world: np.ndarray
    lm_moving_world: np.ndarray
    # Full NIfTI affine (4x4) for fixed and moving. Allows consumers
    # (run_pair.py, lece.py) to handle origin and direction correctly
    # when they differ from identity (e.g. Liver-DIR-QA: different Z origin
    # between fixed and moving, and flipped direction).
    fixed_affine: Optional[np.ndarray] = None
    moving_affine: Optional[np.ndarray] = None
    # HU offset applied by the loader (added to the array to bring air
    # back to -1024). 0 for datasets already in true HU (e.g. Abdominal-DIR-QA),
    # -1024 for datasets stored as unsigned int12 (e.g. Liver-DIR-QA).
    hu_offset_applied: float = 0.0

    @property
    def n_landmarks(self) -> int:
        return self.lm_fixed_voxel.shape[0]


# ----------------------------------------------------------------------
# .mat format loader (Tutorial archive: example pair only)
# ----------------------------------------------------------------------
def load_pair_mat(case_dir: str, case_id: Optional[str] = None) -> AbdominalDIRQAPair:
    if case_id is None:
        case_id = os.path.basename(os.path.normpath(case_dir))

    fixed_path = _find_one(case_dir, ["img1.mat", "*img1*.mat", "*_fixed*.mat"],
                           exclude_keywords=("landmark", "lm_"))
    moving_path = _find_one(case_dir, ["img2.mat", "*img2*.mat", "*_moving*.mat"],
                            exclude_keywords=("landmark", "lm_"))
    lm_path = _find_one(case_dir, ["*landmarks*.mat", "*landmark*.mat"])

    fixed_dict = loadmat(fixed_path)
    moving_dict = loadmat(moving_path)
    lm_dict = loadmat(lm_path)

    fixed_yxz = _pick_image_array(fixed_dict, prefer="img1")
    moving_yxz = _pick_image_array(moving_dict, prefer="img2")

    # (Y, X, Z) → (X, Y, Z) per project convention. No HU offset.
    fixed_arr = np.transpose(fixed_yxz, (1, 0, 2)).astype(np.float32)
    moving_arr = np.transpose(moving_yxz, (1, 0, 2)).astype(np.float32)

    # Landmarks stored as [Y, X, Z] 1-based → reordered to [X, Y, Z] 0-based
    lm1 = np.asarray(lm_dict["landmarks1"], dtype=np.float64)
    lm2 = np.asarray(lm_dict["landmarks2"], dtype=np.float64)
    lm_fixed_voxel = np.column_stack([lm1[:, 1], lm1[:, 0], lm1[:, 2]]) - 1.0
    lm_moving_voxel = np.column_stack([lm2[:, 1], lm2[:, 0], lm2[:, 2]]) - 1.0

    # Voxel sizes in [coronal, sagittal, axial] = (Y, X, Z) → swap the first two
    vs1 = np.asarray(lm_dict["voxelsize1"], dtype=np.float64).ravel()
    vs2 = np.asarray(lm_dict["voxelsize2"], dtype=np.float64).ravel()
    spacing_fixed = (float(vs1[1]), float(vs1[0]), float(vs1[2]))
    spacing_moving = (float(vs2[1]), float(vs2[0]), float(vs2[2]))

    lm_fixed_world = lm_fixed_voxel * np.array(spacing_fixed)
    lm_moving_world = lm_moving_voxel * np.array(spacing_moving)

    return AbdominalDIRQAPair(
        case_id=case_id,
        fixed_arr=fixed_arr,
        moving_arr=moving_arr,
        spacing_fixed=spacing_fixed,
        spacing_moving=spacing_moving,
        lm_fixed_voxel=lm_fixed_voxel,
        lm_moving_voxel=lm_moving_voxel,
        lm_fixed_world=lm_fixed_world,
        lm_moving_world=lm_moving_world,
    )


# ----------------------------------------------------------------------
# NIfTI loader (Zenodo nifti_format archive)
# ----------------------------------------------------------------------
def load_pair_nifti(
    case_dir: str,
    case_id: Optional[str] = None,
    include_flagged: bool = False,
) -> AbdominalDIRQAPair:
    """Load a pair from the flat Zenodo NIfTI layout.

    File patterns:
        <case_id>_img1.nii.gz, <case_id>_img2.nii.gz
        <case_id>_landmarks1.txt, <case_id>_landmarks2.txt
        <case_id>_flagged_landmarks{1,2}.txt   (excluded if include_flagged=False)

    Zenodo conventions:
        - NIfTI dims = [sagittal, coronal, axial] = (X, Y, Z) → already the project convention
        - landmarks = comma-separated CSV, 0-based indices, order (X, Y, Z)
        - intensity = true HU (no offset)
    """
    import nibabel as nib

    if case_id is None:
        # if case_dir is a folder like .../caseN_files, try to infer from basename
        case_id = os.path.basename(os.path.normpath(case_dir))

    # case_id-specific patterns if we are in a flat layout with prefixes.
    # Known suffixes: Abdominal uses _img1.nii.gz, Liver uses _img1.nii,
    # Lung uses _image1.nii.
    fixed_path = _find_with_prefix(case_dir, case_id,
                                    ["_img1.nii.gz", "_img1.nii",
                                     "_image1.nii.gz", "_image1.nii"])
    moving_path = _find_with_prefix(case_dir, case_id,
                                     ["_img2.nii.gz", "_img2.nii",
                                      "_image2.nii.gz", "_image2.nii"])
    lm1_path = _find_with_prefix(case_dir, case_id, ["_landmarks1.txt"])
    lm2_path = _find_with_prefix(case_dir, case_id, ["_landmarks2.txt"])

    fixed_nii = nib.load(fixed_path)
    moving_nii = nib.load(moving_path)
    fixed_arr = np.asarray(fixed_nii.get_fdata(), dtype=np.float32)
    moving_arr = np.asarray(moving_nii.get_fdata(), dtype=np.float32)
    spacing_fixed = tuple(float(v) for v in fixed_nii.header.get_zooms()[:3])
    spacing_moving = tuple(float(v) for v in moving_nii.header.get_zooms()[:3])
    fixed_affine = np.asarray(fixed_nii.affine, dtype=np.float64)
    moving_affine = np.asarray(moving_nii.affine, dtype=np.float64)

    # --- HU offset auto-detection -----------------------------------------
    # Conventions observed in the DIR-QA datasets:
    #   - Abdominal-DIR-QA: true HU (air ~ -1024).
    #   - Liver-DIR-QA:    HU + 1024 (uint12, range [0, 4095], air ~ 0).
    # Heuristic: if the minimum of the fixed array is >= -200 it is assumed
    # shifted and an offset of -1024 is applied to bring air back to true HU.
    hu_offset_applied = 0.0
    if fixed_arr.size > 0 and float(fixed_arr.min()) >= -200.0:
        hu_offset_applied = -1024.0
        fixed_arr = fixed_arr + hu_offset_applied
        moving_arr = moving_arr + hu_offset_applied

    # CSV with comma delimiter (Liver) or whitespace (Abdominal-style)
    lm_fixed_voxel = _load_landmark_csv_or_ws(lm1_path)
    lm_moving_voxel = _load_landmark_csv_or_ws(lm2_path)

    if include_flagged:
        try:
            lm1_fl = _find_with_prefix(case_dir, case_id, ["_flagged_landmarks1.txt"])
            lm2_fl = _find_with_prefix(case_dir, case_id, ["_flagged_landmarks2.txt"])
            f1 = _load_landmark_csv_or_ws(lm1_fl)
            f2 = _load_landmark_csv_or_ws(lm2_fl)
            lm_fixed_voxel = np.vstack([lm_fixed_voxel, f1])
            lm_moving_voxel = np.vstack([lm_moving_voxel, f2])
        except FileNotFoundError:
            pass

    # --- World-coordinate landmarks via full NIfTI affine ------------------
    # NIfTI affine: P_world = affine[:3,:3] @ voxel + affine[:3,3]
    # Needed for datasets with a different origin or non-identity direction,
    # such as Liver-DIR-QA.
    lm_fixed_world = _voxel_to_world(lm_fixed_voxel, fixed_affine)
    lm_moving_world = _voxel_to_world(lm_moving_voxel, moving_affine)

    return AbdominalDIRQAPair(
        case_id=case_id,
        fixed_arr=fixed_arr,
        moving_arr=moving_arr,
        spacing_fixed=spacing_fixed,
        spacing_moving=spacing_moving,
        lm_fixed_voxel=lm_fixed_voxel,
        lm_moving_voxel=lm_moving_voxel,
        lm_fixed_world=lm_fixed_world,
        lm_moving_world=lm_moving_world,
        fixed_affine=fixed_affine,
        moving_affine=moving_affine,
        hu_offset_applied=hu_offset_applied,
    )


# ----------------------------------------------------------------------
# Auto-detect format
# ----------------------------------------------------------------------
def load_pair(case_dir: str, case_id: Optional[str] = None,
              include_flagged: bool = False) -> AbdominalDIRQAPair:
    """Auto-detect the format (.mat or flat NIfTI layout) and dispatch."""
    if case_id is not None:
        # if a case_id was given and case_dir contains <case_id>_img1.nii* files,
        # it is a flat NIfTI layout
        if glob.glob(os.path.join(case_dir, f"{case_id}_img1.nii*")):
            return load_pair_nifti(case_dir, case_id, include_flagged=include_flagged)
        if glob.glob(os.path.join(case_dir, f"{case_id}*.mat")):
            return load_pair_mat(case_dir, case_id)
    has_mat = bool(glob.glob(os.path.join(case_dir, "*.mat")))
    has_nii = bool(glob.glob(os.path.join(case_dir, "*.nii*")))
    if has_mat:
        return load_pair_mat(case_dir, case_id)
    if has_nii:
        return load_pair_nifti(case_dir, case_id, include_flagged=include_flagged)
    raise FileNotFoundError(
        f"No .mat or .nii files found in {case_dir}. Cannot identify format."
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _find_one(folder, patterns, exclude_keywords=()):
    for pat in patterns:
        matches = glob.glob(os.path.join(folder, pat))
        if exclude_keywords:
            matches = [m for m in matches
                       if not any(k in os.path.basename(m).lower() for k in exclude_keywords)]
        if matches:
            return sorted(matches)[0]
    raise FileNotFoundError(f"In {folder}: no file found for pattern {patterns}")


def _find_with_prefix(folder, prefix, suffixes):
    """Look for <folder>/<prefix><suffix> files over a list of suffixes."""
    for suf in suffixes:
        cand = os.path.join(folder, f"{prefix}{suf}")
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"In {folder}: no file found for prefix='{prefix}' suffixes={suffixes}"
    )


def _load_landmark_csv_or_ws(path: str) -> np.ndarray:
    """Load a landmark .txt file, trying comma first (Abdominal/Liver)
    and then whitespace (alternative format).

    Always returns an (N, 3) float64 array.
    """
    for delim in [",", None]:
        try:
            arr = np.loadtxt(path, delimiter=delim, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            if arr.shape[1] == 3:
                return arr
        except (ValueError, IndexError):
            continue
    raise ValueError(f"Cannot parse {path} as (N, 3) voxel indices.")


def _voxel_to_world(voxel_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Transform voxel indices to world coordinates (mm) via a 4x4 NIfTI affine.

    P_world = affine[:3,:3] @ voxel + affine[:3,3]

    Parameters
    ----------
    voxel_coords : (N, 3) 0-based voxel indices.
    affine : (4, 4) NIfTI matrix.

    Returns
    -------
    (N, 3) world coordinates in mm.
    """
    if voxel_coords.size == 0:
        return voxel_coords
    n = voxel_coords.shape[0]
    homog = np.column_stack([voxel_coords, np.ones(n)])
    return (affine @ homog.T).T[:, :3]


def _pick_image_array(mat_dict: dict, prefer: str) -> np.ndarray:
    if prefer in mat_dict and hasattr(mat_dict[prefer], "ndim") and mat_dict[prefer].ndim == 3:
        return mat_dict[prefer]
    for k, v in mat_dict.items():
        if k.startswith("__"):
            continue
        if hasattr(v, "ndim") and v.ndim == 3:
            return v
    raise ValueError(f"No 3D array found; available keys: {list(mat_dict.keys())}")


def discover_cases(root_dir: str) -> list:
    """List of (case_id, case_dir) for the batch run.

    Recognises three layouts:
      1. flat Zenodo NIfTI: root_dir/caseN_img1.nii.gz, ... (all files mixed)
         → returns [(caseN, root_dir), ...]
      2. folders: root_dir/caseN/img1.mat (or .nii)
         → returns [(caseN, root_dir/caseN), ...]
      3. single case: root_dir directly contains img1.mat / img2.mat
         → [(basename(root_dir), root_dir)]
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"{root_dir} does not exist")

    # Layout 1 — flat NIfTI with caseN_ prefix
    # Alternative patterns: _img1 (Abdominal/Liver) or _image1 (Lung).
    flat_imgs = sorted(
        glob.glob(os.path.join(root_dir, "*_img1.nii*")) +
        glob.glob(os.path.join(root_dir, "*_image1.nii*"))
    )
    if flat_imgs:
        cases = []
        for p in flat_imgs:
            base = os.path.basename(p)
            cid = (base.replace("_img1.nii.gz", "")
                       .replace("_img1.nii", "")
                       .replace("_image1.nii.gz", "")
                       .replace("_image1.nii", ""))
            cases.append((cid, root_dir))
        # sort numerically by caseN: case1 < case2 < case10
        def _key(item):
            cid = item[0]
            digits = "".join(ch for ch in cid if ch.isdigit())
            return (int(digits) if digits else 0, cid)
        return sorted(cases, key=_key)

    # Layout 2 — subfolders
    sub = sorted([
        os.path.join(root_dir, d) for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
        and (glob.glob(os.path.join(root_dir, d, "*.mat"))
             or glob.glob(os.path.join(root_dir, d, "*.nii*")))
    ])
    if sub:
        return [(os.path.basename(d), d) for d in sub]

    # Layout 3 — single case in the root
    if glob.glob(os.path.join(root_dir, "*.mat")) or glob.glob(os.path.join(root_dir, "*.nii*")):
        return [(os.path.basename(os.path.normpath(root_dir)), root_dir)]
    raise FileNotFoundError(f"No case found under {root_dir}")
