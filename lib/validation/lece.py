"""
validation/lece.py — Landmark Expected Calibration Error (LECE) and reliability
diagram in the style of Smolders 2023, Fig. 4.

Definizione (Smolders 2023, Eq. 9):

    LECE = sum_{m=1..M} (|B_m|/n) * | RMSE(B_m) - mean(sigma_p, B_m) |

where landmarks are grouped into bins by predicted σ_p (default bin width
0.25 mm), |B_m| is the number of landmarks in the bin and n the total.

Riferimenti:
    Smolders A et al. Deep learning based uncertainty prediction of deformable
    image registration for contour propagation and dose accumulation in
    online adaptive radiotherapy. Phys Med Biol 68:245027, 2023.
    DOI: 10.1088/1361-6560/ad0282

    Guo C, Pleiss G, Sun Y, Weinberger KQ. On Calibration of Modern Neural
    Networks. ICML 2017 (original definition of Expected Calibration Error).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np


@dataclass
class LandmarkEvalResult:
    case_id: str
    tre_mm: np.ndarray              # (N,) landmark positional errors in mm
    sigma_p_mm: np.ndarray          # (N,) predicted sigma at the landmark in mm (3D norm)
    lm_fixed_world: np.ndarray      # (N, 3)
    lm_moving_world_pred: np.ndarray  # (N, 3)
    lm_moving_world_true: np.ndarray  # (N, 3)


def predict_landmark_endpoints(
    dvf_mm: np.ndarray,
    spacing_fixed: Tuple[float, float, float],
    lm_fixed_voxel: np.ndarray,
    fixed_affine: Optional[np.ndarray] = None,
) -> np.ndarray:
    """For each fixed landmark (in voxel index of the fixed image), return the
    predicted position in world coordinates (mm) as l_f_world + R @ u(l_f),
    where R is the direction matrix of the NIfTI affine.

    Parameters
    ----------
    dvf_mm : ndarray (3, X, Y, Z)
        Project convention: channels [dz, dy, dx] in mm, displacement
        expressed along the VOXEL axes (not world). (see utils/dicom_vf.py)
    spacing_fixed : (sx, sy, sz) mm
    lm_fixed_voxel : (N, 3) 0-based voxel coordinates in (X, Y, Z) order.
    fixed_affine : ndarray (4, 4), optional
        NIfTI affine matrix of the fixed image. If given, the world position
        of the landmark is computed as `affine @ [voxel; 1]` and the
        displacement is rotated by the direction matrix R = affine[:3,:3]
        / spacing. If None (default), origin=(0,0,0) and direction=I are
        assumed (correct for Abdominal-DIR-QA, wrong for Liver/Lung-DIR-QA,
        which have a flipped X axis).

    Returns
    -------
    ndarray (N, 3): world coordinates in mm.
    """
    X, Y, Z = dvf_mm.shape[1:]
    sx, sy, sz = spacing_fixed
    # Trilinear interpolation of the DVF at the landmarks (in voxel)
    lm_x = np.clip(lm_fixed_voxel[:, 0], 0, X - 1)
    lm_y = np.clip(lm_fixed_voxel[:, 1], 0, Y - 1)
    lm_z = np.clip(lm_fixed_voxel[:, 2], 0, Z - 1)

    # 8 corners trilinear interpolation
    x0 = np.floor(lm_x).astype(np.int64); x1 = np.minimum(x0 + 1, X - 1)
    y0 = np.floor(lm_y).astype(np.int64); y1 = np.minimum(y0 + 1, Y - 1)
    z0 = np.floor(lm_z).astype(np.int64); z1 = np.minimum(z0 + 1, Z - 1)
    xd = lm_x - x0
    yd = lm_y - y0
    zd = lm_z - z0

    def trilinear(channel: np.ndarray) -> np.ndarray:
        c000 = channel[x0, y0, z0]; c100 = channel[x1, y0, z0]
        c010 = channel[x0, y1, z0]; c110 = channel[x1, y1, z0]
        c001 = channel[x0, y0, z1]; c101 = channel[x1, y0, z1]
        c011 = channel[x0, y1, z1]; c111 = channel[x1, y1, z1]
        c00 = c000 * (1 - xd) + c100 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c11 = c011 * (1 - xd) + c111 * xd
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        return c0 * (1 - zd) + c1 * zd

    # channels in order [dz, dy, dx]
    dz = trilinear(dvf_mm[0])
    dy = trilinear(dvf_mm[1])
    dx = trilinear(dvf_mm[2])

    if fixed_affine is None:
        # Backward-compat: origin = 0, direction = I (Abdominal-DIR-QA case)
        pred_x = lm_fixed_voxel[:, 0] * sx + dx
        pred_y = lm_fixed_voxel[:, 1] * sy + dy
        pred_z = lm_fixed_voxel[:, 2] * sz + dz
        return np.column_stack([pred_x, pred_y, pred_z])

    # Full affine: NIfTI A = [R*S | T]; lm_world_init = A @ [voxel;1].
    # The displacement (dx, dy, dz) is expressed along the VOXEL AXES in mm,
    # so it must be rotated by the direction matrix R to obtain the delta in
    # world coordinates: disp_world = R @ disp_voxel_aligned.
    A = np.asarray(fixed_affine, dtype=np.float64)
    if A.shape != (4, 4):
        raise ValueError(f"fixed_affine must be 4x4, got {A.shape}")
    spacing_arr = np.array([sx, sy, sz], dtype=np.float64)
    # R = A[:3,:3] / spacing (per column). Works for orthogonal affines
    # (standard NIfTI); for non-orthogonal affines the correct formula would be
    # R = A[:3,:3] @ diag(1/spacing), which is equivalent for orthogonal axes.
    R = A[:3, :3] / spacing_arr[None, :]

    N = lm_fixed_voxel.shape[0]
    homo = np.column_stack([lm_fixed_voxel, np.ones(N)])
    lm_world_init = (A @ homo.T).T[:, :3]               # (N, 3)
    disp_voxel = np.column_stack([dx, dy, dz])           # (N, 3)
    disp_world = (R @ disp_voxel.T).T                     # (N, 3)
    return lm_world_init + disp_world


def predict_landmark_sigma(
    sigma_mm: np.ndarray,
    lm_fixed_voxel: np.ndarray,
) -> np.ndarray:
    """For each fixed landmark, return sigma_p = ||sigma(l_f)|| (3D norm).

    Parameters
    ----------
    sigma_mm : ndarray
        Supported shapes:
            (X, Y, Z)         -> scalar uncertainty (e.g. norm of the vector sigma)
            (3, X, Y, Z)      -> vector sigma, norm taken over the channels
    lm_fixed_voxel : (N, 3) 0-based voxel coordinates in (X, Y, Z) order

    Returns
    -------
    ndarray (N,) predicted sigma in mm
    """
    if sigma_mm.ndim == 4:
        # channels [dz, dy, dx]; 3D norm in mm
        sigma_scalar = np.sqrt(
            sigma_mm[0] ** 2 + sigma_mm[1] ** 2 + sigma_mm[2] ** 2
        )
    elif sigma_mm.ndim == 3:
        sigma_scalar = sigma_mm
    else:
        raise ValueError(f"sigma_mm shape expected (X,Y,Z) or (3,X,Y,Z), got {sigma_mm.shape}")

    X, Y, Z = sigma_scalar.shape
    lm_x = np.clip(lm_fixed_voxel[:, 0], 0, X - 1)
    lm_y = np.clip(lm_fixed_voxel[:, 1], 0, Y - 1)
    lm_z = np.clip(lm_fixed_voxel[:, 2], 0, Z - 1)
    x0 = np.floor(lm_x).astype(np.int64); x1 = np.minimum(x0 + 1, X - 1)
    y0 = np.floor(lm_y).astype(np.int64); y1 = np.minimum(y0 + 1, Y - 1)
    z0 = np.floor(lm_z).astype(np.int64); z1 = np.minimum(z0 + 1, Z - 1)
    xd = lm_x - x0; yd = lm_y - y0; zd = lm_z - z0

    c000 = sigma_scalar[x0, y0, z0]; c100 = sigma_scalar[x1, y0, z0]
    c010 = sigma_scalar[x0, y1, z0]; c110 = sigma_scalar[x1, y1, z0]
    c001 = sigma_scalar[x0, y0, z1]; c101 = sigma_scalar[x1, y0, z1]
    c011 = sigma_scalar[x0, y1, z1]; c111 = sigma_scalar[x1, y1, z1]
    c00 = c000 * (1 - xd) + c100 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c11 = c011 * (1 - xd) + c111 * xd
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd
    return (c0 * (1 - zd) + c1 * zd).astype(np.float64)


def evaluate_landmarks(
    dvf_mm: np.ndarray,
    sigma_mm: np.ndarray,
    spacing_fixed: Tuple[float, float, float],
    lm_fixed_voxel: np.ndarray,
    lm_moving_world_true: np.ndarray,
    case_id: str = "case",
    fixed_affine: Optional[np.ndarray] = None,
) -> LandmarkEvalResult:
    """Compute TRE and sigma_p per landmark given the DVF mu and the sigma map.

    If `fixed_affine` is given, world positions are computed via
    `affine @ [voxel; 1]` and the displacement is rotated by the direction
    matrix R = affine[:3,:3] / spacing — needed for datasets with a
    non-trivial affine (e.g. Liver/Lung-DIR-QA with flipped X axis in NIfTI).
    """
    pred = predict_landmark_endpoints(
        dvf_mm, spacing_fixed, lm_fixed_voxel,
        fixed_affine=fixed_affine,
    )
    tre = np.sqrt(np.sum((pred - lm_moving_world_true) ** 2, axis=1)).astype(np.float64)
    sigma_p = predict_landmark_sigma(sigma_mm, lm_fixed_voxel)

    sx, sy, sz = spacing_fixed
    if fixed_affine is None:
        # World position of the fixed landmark (origin = 0, direction = I)
        lm_fixed_world = lm_fixed_voxel * np.array([sx, sy, sz])
    else:
        # World position via full affine
        N = lm_fixed_voxel.shape[0]
        homo = np.column_stack([lm_fixed_voxel, np.ones(N)])
        lm_fixed_world = (np.asarray(fixed_affine, dtype=np.float64) @ homo.T).T[:, :3]

    return LandmarkEvalResult(
        case_id=case_id,
        tre_mm=tre,
        sigma_p_mm=sigma_p,
        lm_fixed_world=lm_fixed_world,
        lm_moving_world_pred=pred,
        lm_moving_world_true=lm_moving_world_true,
    )


# ----------------------------------------------------------------------
# LECE — Smolders 2023, Eq. 9
# ----------------------------------------------------------------------
def compute_lece(
    tre_mm: np.ndarray,
    sigma_p_mm: np.ndarray,
    bin_width: float = 0.25,
    sigma_max: Optional[float] = None,
) -> Tuple[float, dict]:
    """Compute the bin-weighted LECE as in Smolders 2023, Eq. 9.

    Returns
    -------
    lece : float
        LECE in mm.
    diag : dict with
        bin_centers, bin_counts, bin_rmse, bin_mean_sigma, bin_edges
    """
    tre = np.asarray(tre_mm, dtype=np.float64).ravel()
    sigma_p = np.asarray(sigma_p_mm, dtype=np.float64).ravel()
    assert tre.shape == sigma_p.shape, f"shape mismatch: {tre.shape} vs {sigma_p.shape}"

    if sigma_max is None:
        sigma_max = float(np.percentile(sigma_p, 99.9))
    sigma_max = max(sigma_max, bin_width)

    edges = np.arange(0.0, sigma_max + bin_width, bin_width)
    bin_idx = np.digitize(sigma_p, edges) - 1   # bin index 0..len(edges)-2
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)

    n = len(tre)
    lece = 0.0
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    bin_counts = np.zeros(len(bin_centers), dtype=np.int64)
    bin_rmse = np.zeros(len(bin_centers), dtype=np.float64)
    bin_mean_sigma = np.zeros(len(bin_centers), dtype=np.float64)

    for m in range(len(bin_centers)):
        mask = bin_idx == m
        bm = mask.sum()
        bin_counts[m] = bm
        if bm == 0:
            continue
        rmse_m = float(np.sqrt(np.mean(tre[mask] ** 2)))
        mean_s = float(np.mean(sigma_p[mask]))
        bin_rmse[m] = rmse_m
        bin_mean_sigma[m] = mean_s
        lece += (bm / n) * abs(rmse_m - mean_s)

    diag = {
        "bin_edges": edges,
        "bin_centers": bin_centers,
        "bin_counts": bin_counts,
        "bin_rmse": bin_rmse,
        "bin_mean_sigma": bin_mean_sigma,
        "n_landmarks": n,
        "bin_width": bin_width,
    }
    return float(lece), diag


# ----------------------------------------------------------------------
# Reliability diagram, Smolders 2023 Fig. 4 style
# ----------------------------------------------------------------------
def plot_reliability_diagram(
    diag: dict,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    label: str = "model",
    color: str = "#1f77b4",
    show_diagonal: bool = True,
):
    """Generate a diagram in the style of Smolders Fig. 4.
    X axis: predicted uncertainty σ_p (mm).
    Left Y axis: RMSE per bin (mm) as scatter+line.
    Right Y axis: number of landmarks per bin (histogram).
    """
    import matplotlib.pyplot as plt

    bc = diag["bin_centers"]
    counts = diag["bin_counts"]
    rmse = diag["bin_rmse"]
    nz = counts > 0
    if not np.any(nz):
        raise ValueError("All bins empty — no landmarks to plot.")

    fig, ax_left = plt.subplots(figsize=(6, 4.5), dpi=110)
    ax_right = ax_left.twinx()

    # Histogram
    ax_right.bar(bc, counts, width=diag["bin_width"], color="#cccccc",
                 edgecolor="#888888", alpha=0.7, label="n landmarks/bin")
    ax_right.set_ylabel("n_landmarks within bin", color="#666666")
    ax_right.tick_params(axis="y", colors="#666666")

    # RMSE points + line
    ax_left.plot(bc[nz], rmse[nz], "o-", color=color, label=label,
                 linewidth=1.6, markersize=6)

    # Ideal diagonal
    if show_diagonal:
        lim = max(np.max(bc[nz]), np.max(rmse[nz]))
        ax_left.plot([0, lim], [0, lim], color="black", linewidth=1.0,
                     linestyle="-", label="ideal (RMSE = sigma_p)")

    ax_left.set_xlabel("Predicted uncertainty $\\sigma_p$ [mm]")
    ax_left.set_ylabel("RMSE (target) of landmarks in bin [mm]")
    ax_left.set_xlim(left=0)
    ax_left.set_ylim(bottom=0)
    ax_left.legend(loc="upper left", fontsize=9)
    if title:
        ax_left.set_title(title, fontsize=11)
    fig.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150)
    return fig


def aggregate_results(results: List[LandmarkEvalResult],
                      bin_width: float = 0.25) -> Tuple[float, dict]:
    """LECE over many concatenated cases."""
    tre = np.concatenate([r.tre_mm for r in results])
    sig = np.concatenate([r.sigma_p_mm for r in results])
    return compute_lece(tre, sig, bin_width=bin_width)


# ----------------------------------------------------------------------
# Discrimination metric — Spearman rank correlation σ_p vs TRE
# ----------------------------------------------------------------------
def compute_spearman(tre_mm: np.ndarray, sigma_p_mm: np.ndarray) -> float:
    """Spearman rank correlation ρ(σ_p, TRE).

    Complementary to LECE: LECE measures whether σ_p has the right *level*
    (bin-mean σ_p ≈ bin-RMSE TRE), Spearman ρ measures whether σ_p has the
    right *ordering* across landmarks (higher σ_p → larger TRE).

    Robust to heavy tails (rank-based; invariant to monotone transforms of σ_p).
    Returns NaN if input has < 3 pairs or no variance.

    Returns
    -------
    rho : float in [-1, +1]. ρ = 0 → no monotone association.
    """
    from scipy.stats import spearmanr
    tre = np.asarray(tre_mm, dtype=np.float64).ravel()
    sp  = np.asarray(sigma_p_mm, dtype=np.float64).ravel()
    if tre.size != sp.size:
        raise ValueError(f"size mismatch: {tre.size} vs {sp.size}")
    mask = np.isfinite(tre) & np.isfinite(sp)
    if mask.sum() < 3 or sp[mask].std() == 0 or tre[mask].std() == 0:
        return float("nan")
    return float(spearmanr(tre[mask], sp[mask]).statistic)


def compute_auc_discrim(tre_mm: np.ndarray, sigma_p_mm: np.ndarray,
                        threshold: Optional[float] = None) -> float:
    """Discrimination AUC = P(σ_p(hard) > σ_p(easy)).

    Implemented via the normalised Mann-Whitney U. Default threshold = median
    of tre_mm. Equivalent to ROC-AUC with label hard = (TRE > threshold).

    Interpretation: 0.5 = random ordering, 1.0 = σ_p ranks all hard landmarks
    above all easy landmarks. Optional helper; the pipeline uses Spearman ρ.
    """
    from scipy.stats import mannwhitneyu
    tre = np.asarray(tre_mm, dtype=np.float64).ravel()
    sp  = np.asarray(sigma_p_mm, dtype=np.float64).ravel()
    if tre.size != sp.size:
        raise ValueError(f"size mismatch: {tre.size} vs {sp.size}")
    thr = float(np.median(tre)) if threshold is None else float(threshold)
    hard = sp[tre > thr]
    easy = sp[tre <= thr]
    if len(hard) == 0 or len(easy) == 0:
        return float("nan")
    u, _ = mannwhitneyu(hard, easy, alternative="greater")
    return float(u / (len(hard) * len(easy)))
