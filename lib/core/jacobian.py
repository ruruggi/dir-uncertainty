"""
core/jacobian.py — Full 3×3 Jacobian determinant check for diffeomorphism control.

Implements Section 2.3 of the method:
  - Full det(I + ∇z_k) with all 9 partial derivatives
  - Rejection sampling: accept only if folding_rate < τ (default 0.01%)
  - Up to 20 retries per sample
"""

import torch
import numpy as np


def jacobian_determinant(dvf: torch.Tensor) -> torch.Tensor:
    """
    Compute the full 3×3 Jacobian determinant of the displacement field.

    Given DVF with components (u, v, w) indexed as dvf[0], dvf[1], dvf[2],
    the Jacobian matrix at each voxel is:

        J = I + [[∂u/∂z, ∂u/∂y, ∂u/∂x],
                  [∂v/∂z, ∂v/∂y, ∂v/∂x],
                  [∂w/∂z, ∂w/∂y, ∂w/∂x]]

    Parameters
    ----------
    dvf : Tensor (3, Z, Y, X) — displacement field (in voxels)

    Returns
    -------
    det_J : Tensor (Z, Y, X) — determinant of the Jacobian
    """
    u, v, w = dvf[0], dvf[1], dvf[2]

    # Central differences for interior, forward diff at boundaries
    def grad_d(f, dim):
        """Gradient along dimension dim using central differences."""
        g = torch.zeros_like(f)
        if dim == 0:  # Z
            g[1:-1] = (f[2:] - f[:-2]) / 2.0
            g[0] = f[1] - f[0]
            g[-1] = f[-1] - f[-2]
        elif dim == 1:  # Y
            g[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / 2.0
            g[:, 0] = f[:, 1] - f[:, 0]
            g[:, -1] = f[:, -1] - f[:, -2]
        elif dim == 2:  # X
            g[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / 2.0
            g[:, :, 0] = f[:, :, 1] - f[:, :, 0]
            g[:, :, -1] = f[:, :, -1] - f[:, :, -2]
        return g

    # 9 partial derivatives
    du_dz = grad_d(u, 0); du_dy = grad_d(u, 1); du_dx = grad_d(u, 2)
    dv_dz = grad_d(v, 0); dv_dy = grad_d(v, 1); dv_dx = grad_d(v, 2)
    dw_dz = grad_d(w, 0); dw_dy = grad_d(w, 1); dw_dx = grad_d(w, 2)

    # J = I + ∇u
    # det(J) = det([[1+du/dz, du/dy, du/dx],
    #               [dv/dz, 1+dv/dy, dv/dx],
    #               [dw/dz, dw/dy, 1+dw/dx]])
    a11 = 1.0 + du_dz; a12 = du_dy;       a13 = du_dx
    a21 = dv_dz;        a22 = 1.0 + dv_dy; a23 = dv_dx
    a31 = dw_dz;        a32 = dw_dy;       a33 = 1.0 + dw_dx

    det_J = (a11 * (a22 * a33 - a23 * a32)
             - a12 * (a21 * a33 - a23 * a31)
             + a13 * (a21 * a32 - a22 * a31))

    return det_J


def folding_rate(dvf: torch.Tensor) -> float:
    """
    Compute the fraction of voxels with non-positive Jacobian determinant.

    Parameters
    ----------
    dvf : Tensor (3, Z, Y, X) — displacement in voxels

    Returns
    -------
    rate : float — N_folding / N_total
    """
    det = jacobian_determinant(dvf)
    n_folding = int(torch.sum(det <= 0).item())
    n_total = det.numel()
    return n_folding / n_total


def check_diffeomorphism(dvf: torch.Tensor,
                         threshold: float = 0.0001) -> tuple:
    """
    Check whether a DVF is diffeomorphic (folding rate below threshold).

    Parameters
    ----------
    dvf : Tensor (3, Z, Y, X)
    threshold : float — max acceptable folding rate (default 0.01%)

    Returns
    -------
    (dvf, rate) if accepted, (None, rate) if rejected
    """
    rate = folding_rate(dvf)
    if rate > threshold:
        return None, rate
    return dvf, rate


# Legacy alias
jacobian_check = check_diffeomorphism
