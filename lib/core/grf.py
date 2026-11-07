"""
core/grf.py — Gaussian Random Field utilities for DIR Uncertainty v2.

Implements:
  - L2-normalised anisotropic Gaussian kernel (preserves variance)
  - Spatially adaptive correlation lengths (Eq. 5)
  - Anisotropic convolution via separable 1-D filters with 4×4×4 block tiling
  - Continuous behaviour for vanishing correlation lengths (no hard threshold)
"""

import numpy as np
import torch
import torch.nn.functional as F
from utils.jacobian import jacobian_check


# ---------------------------------------------------------------------------
# 1-D Gaussian kernel — L2 normalised (preserves variance)
# ---------------------------------------------------------------------------

def gaussian_kernel_1d(sigma: float, kernel_size: int, device='cuda') -> torch.Tensor:
    """
    1-D Gaussian kernel normalised so that sum(k^2) = 1 (L2 norm = 1).

    For sigma <= 0.1 voxel, returns a Dirac delta (single 1 at center) to
    represent the limit of zero correlation length in a continuous way.

    Parameters
    ----------
    sigma : float
        Standard deviation in voxels. Clamped to ≥ 0.0.
    kernel_size : int
        Size of the kernel (will be forced to odd).
    device : str
        Torch device.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = max(sigma, 0.0)
    half = kernel_size // 2

    # Dirac delta limit (continuous transition, not a hard skip)
    if sigma <= 0.1:
        kernel = torch.zeros(kernel_size, device=device)
        kernel[half] = 1.0
        return kernel

    x = torch.linspace(-half, half, kernel_size, device=device)
    kernel = torch.exp(-x ** 2 / (2.0 * sigma ** 2))

    # L2 normalisation: sum of squares = 1
    kernel = kernel / torch.sqrt(torch.sum(kernel ** 2))
    return kernel


# ---------------------------------------------------------------------------
# Separable 3-D convolution (GPU, grouped by channel)
# ---------------------------------------------------------------------------

def conv3d_separable(x: torch.Tensor,
                     sigmas: tuple,
                     kernel_size: int,
                     device=None) -> torch.Tensor:
    """
    Apply a separable 3-D Gaussian convolution to *x* using three 1-D passes.

    Always performs convolution (no early exit), letting the kernel shape
    naturally produce the correct correlation even for very small sigmas.

    Parameters
    ----------
    x : Tensor (B, C, Z, Y, X)
    sigmas : (σ_z, σ_y, σ_x) in voxels
    kernel_size : int
    """
    if device is None:
        device = x.device
    B, C, Z, Y, X = x.shape

    pad = kernel_size // 2

    # Build 1-D kernels for each spatial axis
    kz = gaussian_kernel_1d(sigmas[0], kernel_size, device).to(x.dtype)
    ky = gaussian_kernel_1d(sigmas[1], kernel_size, device).to(x.dtype)
    kx = gaussian_kernel_1d(sigmas[2], kernel_size, device).to(x.dtype)

    # Reshape for grouped conv3d: weight shape (C_out, C_in/groups, kD, kH, kW)
    kz = kz.view(1, 1, -1, 1, 1).repeat(C, 1, 1, 1, 1)
    ky = ky.view(1, 1, 1, -1, 1).repeat(C, 1, 1, 1, 1)
    kx = kx.view(1, 1, 1, 1, -1).repeat(C, 1, 1, 1, 1)

    # Sequential 1-D convolutions (groups=C → per-channel)
    out = F.conv3d(x, kz, padding=(pad, 0, 0), groups=C)
    out = F.conv3d(out, ky, padding=(0, pad, 0), groups=C)
    out = F.conv3d(out, kx, padding=(0, 0, pad), groups=C)

    # *** NO renormalisation ***  — preserves E[ε_corr ε_corr^T] = C_aniso
    return out


# ---------------------------------------------------------------------------
# Adaptive correlation length map — Eq. 5
# ---------------------------------------------------------------------------

def adaptive_ell_map(ct_image: torch.Tensor,
                     spacing: tuple,
                     l0: tuple = (19.5, 13.0, 13.0),
                     alpha: float = 0.0,
                     theta: float = 50.0) -> torch.Tensor:
    """
    Compute spatially varying correlation lengths l_d(x) for d ∈ {z, y, x}.

    Eq. 5:  l_d(x) = l0_d · (1 + α · tanh(‖∇I(x)‖ / (2·θ)))

    where tanh(·) is the hyperbolic tangent and θ is the gradient threshold in HU.

    Pipeline default for θ
    ----------------------
    The signature default ``theta=50.0`` is a **direct-call convenience fallback
    only**: when this function is invoked through the canonical pipeline
    (``runner.py`` or the GUI ``app.py``), the caller intercepts ``theta == 0``
    and replaces it with ``median(|∇I_f|)`` of the fixed CT *before* calling
    here (see ``runner.py:1022-1030`` and the GUI label "θ (HU, 0=auto)"). The
    effective default of the pipeline is therefore the per-patient
    median-gradient, not 50 HU. The hard-coded 50 HU only applies to bare
    imports of ``adaptive_ell_map`` from outside the pipeline.

    Parameters
    ----------
    ct_image : Tensor (1, Z, Y, X) — fixed CT in HU
    spacing : (sz, sy, sx) voxel spacing in mm
    l0 : (l0_z, l0_y, l0_x) baseline correlation lengths in mm
    alpha : float ∈ [-1, 1], anatomical scaling factor
    theta : float, gradient threshold in HU. See the *Pipeline default for θ*
        note above: this signature default of 50.0 is NOT the pipeline default.

    Returns
    -------
    ell_map : Tensor (3, Z, Y, X) — correlation lengths in **voxels**
              ell_map[0] = l_z, ell_map[1] = l_y, ell_map[2] = l_x
    """
    # Compute gradient magnitude of CT image
    ct = ct_image.squeeze(0) if ct_image.dim() == 4 else ct_image  # (Z, Y, X)

    gz = torch.diff(ct, dim=0, append=ct[-1:])
    gy = torch.diff(ct, dim=1, append=ct[:, -1:])
    gx = torch.diff(ct, dim=2, append=ct[:, :, -1:])
    grad_mag = torch.sqrt(gz ** 2 + gy ** 2 + gx ** 2 + 1e-12)

    # tanh scaling: tanh(‖∇I‖ / (2·θ))
    scaled = torch.tanh(grad_mag / (2.0 * max(theta, 1e-6)))

    # Per-direction correlation lengths in mm, then convert to voxels
    spacing_t = torch.tensor(spacing, device=ct.device, dtype=ct.dtype)  # (sz, sy, sx)
    l0_t = torch.tensor(l0, device=ct.device, dtype=ct.dtype)            # (l0_z, l0_y, l0_x)

    # l_d(x) = l0_d * (1 + α * tanh) / spacing_d   → in voxels
    ell_map = torch.stack([
        l0_t[d] * (1.0 + alpha * scaled) / spacing_t[d]
        for d in range(3)
    ], dim=0)  # (3, Z, Y, X)

    return ell_map


# ---------------------------------------------------------------------------
# Anisotropic correlation — block-tiled non-stationary convolution
# ---------------------------------------------------------------------------

def anisotropic_correlation(epsilon: torch.Tensor,
                            ell_map: torch.Tensor,
                            kernel_size: int = 61) -> torch.Tensor:
    """
    Apply spatially varying anisotropic Gaussian correlation to white noise ε.

    ε_corr = K * ε   (Eq. 7)

    For non-stationary fields, the volume is tiled into 4×4×4 overlapping blocks,
    each processed with its own locally averaged kernel.

    Always applies convolution (no early exit), ensuring continuous behaviour
    even when correlation lengths are very small.

    Parameters
    ----------
    epsilon : Tensor (3, Z, Y, X) — white Gaussian noise
    ell_map : Tensor (3, Z, Y, X) — per-voxel correlation lengths in voxels
              OR tuple (σ_z, σ_y, σ_x) for stationary case
    kernel_size : int

    Returns
    -------
    correlated : Tensor (3, Z, Y, X)
    """
    device = epsilon.device

    # Ensure 5-D: (B, C, Z, Y, X)
    if epsilon.ndim == 4:
        epsilon = epsilon.unsqueeze(0)   # (1, 3, Z, Y, X)
    B, C, Z, Y, X = epsilon.shape

    is_stationary = not isinstance(ell_map, torch.Tensor) or ell_map.ndim <= 1

    # Shape validation: non-stationary ell_map must be (3, Z, Y, X) — one
    # correlation-length channel per spatial axis. Catch silent miscalls (e.g.
    # passing a 4-channel or 2-channel tensor) that would otherwise fall back
    # to stationary convolution with arbitrary length picked from the first 3
    # tensor entries.
    if isinstance(ell_map, torch.Tensor) and not is_stationary:
        if ell_map.ndim != 4 or ell_map.shape[0] != 3:
            raise ValueError(
                f"ell_map for non-stationary anisotropic correlation must have "
                f"shape (3, Z, Y, X); got shape {tuple(ell_map.shape)}."
            )

    if is_stationary:
        # Global convolution with fixed sigmas
        if isinstance(ell_map, torch.Tensor):
            sigmas = ell_map.flatten()[:3].tolist()
        else:
            sigmas = list(ell_map)[:3]
        correlated = conv3d_separable(epsilon, sigmas, kernel_size)
    else:
        # Non-stationary: 4×4×4 tiling with overlap
        n_tiles = (4, 4, 4)
        tile_sz = (
            int(np.ceil(Z / n_tiles[0])),
            int(np.ceil(Y / n_tiles[1])),
            int(np.ceil(X / n_tiles[2])),
        )

        correlated = torch.zeros_like(epsilon)
        margin = kernel_size // 2

        for tz in range(n_tiles[0]):
            for ty in range(n_tiles[1]):
                for tx in range(n_tiles[2]):
                    # Core region
                    z0 = tz * tile_sz[0]; z1 = min((tz + 1) * tile_sz[0], Z)
                    y0 = ty * tile_sz[1]; y1 = min((ty + 1) * tile_sz[1], Y)
                    x0 = tx * tile_sz[2]; x1 = min((tx + 1) * tile_sz[2], X)
                    if z1 <= z0 or y1 <= y0 or x1 <= x0:
                        continue

                    # Expanded region (with margin for convolution)
                    mz0 = max(0, z0 - margin); mz1 = min(Z, z1 + margin)
                    my0 = max(0, y0 - margin); my1 = min(Y, y1 + margin)
                    mx0 = max(0, x0 - margin); mx1 = min(X, x1 + margin)

                    block_eps = epsilon[:, :, mz0:mz1, my0:my1, mx0:mx1]
                    block_ell = ell_map[:, mz0:mz1, my0:my1, mx0:mx1]

                    # Local average of correlation lengths for this block
                    local_sigmas = [float(torch.mean(block_ell[d])) for d in range(3)]

                    # Always perform convolution (no early exit)
                    block_corr = conv3d_separable(block_eps, local_sigmas, kernel_size)

                    # Crop back to core region
                    rz0 = z0 - mz0; rz1 = z1 - mz0
                    ry0 = y0 - my0; ry1 = y1 - my0
                    rx0 = x0 - mx0; rx1 = x1 - mx0
                    correlated[:, :, z0:z1, y0:y1, x0:x1] = \
                        block_corr[:, :, rz0:rz1, ry0:ry1, rx0:rx1]

    # Return (3, Z, Y, X) — squeeze batch dim
    return correlated.squeeze(0)