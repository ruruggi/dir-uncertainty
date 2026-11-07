"""
core/rigid_align.py — 6-DOF rigid transform for manual pre-alignment.

Conventions:
- The transform is applied to the MOVING image to bring it into the reference
  frame of the FIXED. Mathematically: x_fixed = R*(x_moving - c) + c + t, where
  c = centroid_world of the fixed (rotation pivot), t in mm, R = Rz*Ry*Rx
  with angles in degrees (intrinsic Euler XYZ).
- World coordinates in mm. The voxel<->world conversion is done by the callers
  via the origin/spacing of the respective Scan.

Output JSON (versione 1):
    {
      "version": 1,
      "tx_mm": float, "ty_mm": float, "tz_mm": float,   # translation
      "rx_deg": float, "ry_deg": float, "rz_deg": float, # rotation (intrinsic XYZ)
      "center_world_mm": [cx, cy, cz],                   # rotation pivot
      "fixed_path": "...",                               # reference (info-only)
      "moving_path": "...",
    }

Implementation notes:
- Applying the transform to the volume for the PREVIEW uses
  scipy.ndimage.affine_transform on subsampled arrays to stay responsive.
- For full-resolution use (downstream of a DIR engine) it is better to compose
  with an ANTs initial transform; that integration is deferred to a later
  iteration.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------
def _rotation_x(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rotation_y(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rotation_z(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def build_rigid_matrix(tx_mm: float, ty_mm: float, tz_mm: float,
                       rx_deg: float, ry_deg: float, rz_deg: float,
                       center_world_mm: Optional[Tuple[float, float, float]] = None
                       ) -> np.ndarray:
    """Build the 4x4 homogeneous matrix for the rigid transformation.

    Convenzione: y = R*(x - c) + c + t, con R = Rz @ Ry @ Rx (intrinsic XYZ).
    If center_world_mm is None, c = (0, 0, 0). Otherwise the rotation is about
    center_world_mm (e.g. the fixed centroid in world coordinates).
    """
    R = _rotation_z(math.radians(rz_deg)) @ _rotation_y(math.radians(ry_deg)) @ _rotation_x(math.radians(rx_deg))
    t = np.array([tx_mm, ty_mm, tz_mm], dtype=np.float64)
    if center_world_mm is None:
        c = np.zeros(3, dtype=np.float64)
    else:
        c = np.asarray(center_world_mm, dtype=np.float64)

    # y = R*(x - c) + c + t = R*x + (c - R*c + t)
    offset = c - R @ c + t

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = offset
    return M


def matrix_to_params(M: np.ndarray, center_world_mm: Optional[Tuple[float, float, float]] = None
                     ) -> dict:
    """Partial inverse: extract (tx, ty, tz, rx, ry, rz) from a 4x4 affine
    matrix, given the pivot c. Useful in tests and for round-trip debugging.
    """
    R = M[:3, :3]
    offset = M[:3, 3]
    c = np.zeros(3) if center_world_mm is None else np.asarray(center_world_mm, dtype=np.float64)
    t = offset - (c - R @ c)

    # Decomposition of Rz @ Ry @ Rx (intrinsic XYZ).
    # Reference: Diebel 2006, eq. (1.2.6) gimbal-lock-aware.
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock; ry = +/- pi/2.
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0.0

    return {
        "tx_mm": float(t[0]),
        "ty_mm": float(t[1]),
        "tz_mm": float(t[2]),
        "rx_deg": math.degrees(rx),
        "ry_deg": math.degrees(ry),
        "rz_deg": math.degrees(rz),
    }


# ---------------------------------------------------------------------------
# Landmark-based rigid (Procrustes / Kabsch SVD)
# ---------------------------------------------------------------------------
def procrustes_rigid(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Optimal rigid transform (Kabsch SVD, no scaling) mapping Q -> P.

    Args:
        P: (N, 3) target points (e.g. fixed landmarks, world XYZ mm).
        Q: (N, 3) source points (e.g. moving landmarks, world XYZ mm).

    Returns:
        M_4x4: 4x4 matrix in XYZ convention such that M @ [Q; 1] ~= [P; 1].
        Equivalently: y = R @ x + t with R a pure rotation (det=+1), t in mm.

    Reference: Arun, Huang, Blostein (1987) — Least-Squares Fitting of Two
    3-D Point Sets. The correction `diag(1, 1, sign(det(V U^T)))` prevents
    reflections in degenerate configurations.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"P, Q must be (N, 3); got {P.shape}, {Q.shape}")
    if len(P) < 3:
        raise ValueError("At least 3 landmarks are required for Kabsch.")
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    Pc = P - cP
    Qc = Q - cQ
    H = Qc.T @ Pc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d if d != 0.0 else 1.0])
    R = Vt.T @ D @ U.T
    t = cP - R @ cQ
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------
def save_rigid_json(path: str, params: dict, fixed_path: str = "",
                    moving_path: str = "") -> None:
    out = {
        "version": 1,
        "tx_mm": float(params.get("tx_mm", 0.0)),
        "ty_mm": float(params.get("ty_mm", 0.0)),
        "tz_mm": float(params.get("tz_mm", 0.0)),
        "rx_deg": float(params.get("rx_deg", 0.0)),
        "ry_deg": float(params.get("ry_deg", 0.0)),
        "rz_deg": float(params.get("rz_deg", 0.0)),
        "center_world_mm": list(params.get("center_world_mm") or [0.0, 0.0, 0.0]),
        "fixed_path": fixed_path,
        "moving_path": moving_path,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def load_rigid_json(path: str) -> dict:
    with open(path, "r") as f:
        d = json.load(f)
    if d.get("version", 1) != 1:
        raise ValueError(f"Unsupported rigid_align JSON version: {d.get('version')}")
    return d


# ---------------------------------------------------------------------------
# Volume application (per preview)
# ---------------------------------------------------------------------------
def apply_rigid_to_volume(moving_arr: np.ndarray,
                          moving_origin: np.ndarray,
                          moving_spacing: np.ndarray,
                          fixed_shape: Tuple[int, int, int],
                          fixed_origin: np.ndarray,
                          fixed_spacing: np.ndarray,
                          M_world: np.ndarray,
                          order: int = 1,
                          cval: float = -1000.0,
                          ) -> np.ndarray:
    """Apply the rigid transformation M (in world coordinates) to the moving volume,
    resampling onto the fixed grid. All arrays use the (Z, Y, X) convention.

    M_world: 4x4 matrix mapping x_moving_world -> x_fixed_world.
    For resampling, scipy.ndimage.affine_transform needs an
    (output_index) -> (input_index) map, so we need the inverse M^-1 that
    maps x_fixed_world -> x_moving_world; we then convert world<->voxel on
    both sides.

    Output: array with shape == fixed_shape, dtype == moving_arr.dtype.
    """
    from scipy.ndimage import affine_transform

    # Index order in scipy: (Z, Y, X). Spacing/origin are passed as (Z, Y, X),
    # already consistent with the array. So the voxel<->world conversion for axis i is:
    # world_i = origin_i + voxel_i * spacing_i.

    M_inv = np.linalg.inv(M_world)

    # Build the mapping matrices fixed_voxel -> moving_voxel.
    # x_fixed_world = origin_f + diag(spacing_f) @ x_fixed_vox
    # x_moving_world = M_inv @ x_fixed_world  (converted to 4-homogeneous)
    # x_moving_vox = (x_moving_world - origin_m) / spacing_m

    Sf = np.diag(np.asarray(fixed_spacing, dtype=np.float64))
    Sm_inv = np.diag(1.0 / np.asarray(moving_spacing, dtype=np.float64))

    # Linear part: Sm_inv @ R_inv @ Sf
    R_inv = M_inv[:3, :3]
    A = Sm_inv @ R_inv @ Sf

    # Offset: Sm_inv @ (R_inv @ origin_f + t_inv - origin_m)
    t_inv = M_inv[:3, 3]
    of = np.asarray(fixed_origin, dtype=np.float64)
    om = np.asarray(moving_origin, dtype=np.float64)
    offset = Sm_inv @ (R_inv @ of + t_inv - om)

    out = affine_transform(
        moving_arr.astype(np.float32, copy=False),
        matrix=A,
        offset=offset,
        output_shape=tuple(int(s) for s in fixed_shape),
        order=order,
        mode="constant",
        cval=float(cval),
        prefilter=(order > 1),
    )
    return out


def compute_world_centroid(shape_zyx: Tuple[int, int, int],
                           origin_zyx: np.ndarray,
                           spacing_zyx: np.ndarray) -> Tuple[float, float, float]:
    """Volume centroid in world coordinates (Z, Y, X order)."""
    Z, Y, X = shape_zyx
    cx_vox = np.array([(Z - 1) / 2.0, (Y - 1) / 2.0, (X - 1) / 2.0], dtype=np.float64)
    cx_world = np.asarray(origin_zyx, dtype=np.float64) + cx_vox * np.asarray(spacing_zyx, dtype=np.float64)
    return tuple(float(v) for v in cx_world)
