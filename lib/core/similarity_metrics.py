"""
core/similarity_metrics.py — Similarity metrics for the residual term of the
heuristic+residual model, alternatives to the plain |I_m(x+phi) - I_f(x)|,
to extend the method to cross-modal/MRI images where the absolute intensity
is not calibrated between fixed and moving.

Available metrics (see compute_residual() as dispatcher):

    "abs_diff"  : |I_m - I_f|                   (default, equivalent to the
                                                 original code, restricted to
                                                 CT/sCT)
    "mind_ssc"  : || MIND_SSC(I_m) - MIND_SSC(I_f) ||_1
                  Modality Independent Neighbourhood Descriptor [Heinrich 2012,
                  2013]. Robust to non-monotonic intensity transformations
                  (cross-modal MR-CT, multi-sequence MR).
    "lncc"      : 1 - LNCC(I_m, I_f, win=W)
                  Local Normalised Cross-Correlation. Invariant to
                  *linear* intensity transformations. Suited to same-
                  sequence MR-MR with residual bias.

References:
    Heinrich MP, Jenkinson M, Bhushan M, et al. MIND: Modality Independent
    Neighbourhood Descriptor for multi-modal deformable registration.
    Medical Image Analysis 2012; 16(7):1423-1435.
    DOI: 10.1016/j.media.2012.05.008

    Heinrich MP, Jenkinson M, Papiez BW, Brady M, Schnabel JA. Towards
    realtime multimodal fusion for image-guided interventions using self-
    similarities. MICCAI 2013; LNCS 8149: 187-194.
    (Defines MIND-SSC with 12 self-similarity pairs.)

    Heinrich MP, Hansen L. Voxelmorph++: Going beyond the cranial vault with
    keypoint supervision and multi-channel instance optimisation. WBIR 2022.
    (Practical use of MIND-SSC in DL registration — basis of ConvexAdam.)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MIND-SSC
# ---------------------------------------------------------------------------
def mind_ssc(image: torch.Tensor,
             radius: int = 1,
             dilation: int = 2) -> torch.Tensor:
    """Modality Independent Neighbourhood Descriptor — Self-Similarity Context.

    Computes a 12-channel per-voxel descriptor based on the self-similarity
    of 12 pairs of shifted patches in the voxel's "self-similarity context".
    The descriptor is (a) invariant to intensity transformations as long as
    the local patterns are preserved and (b) parsimonious (12 channels)
    compared with the ~50 pairs of the original MIND.

    Parameters
    ----------
    image : Tensor (1, 1, X, Y, Z) or (1, X, Y, Z) or (X, Y, Z), float32
    radius : int
        Radius of the patch box for the sum-of-squared-differences (SSD).
        Default 1 -> 3x3x3 voxel patch.
    dilation : int
        Dilation of the neighbours in the self-similarity context. Default 2 voxels.

    Returns
    -------
    Tensor (1, 12, X, Y, Z) float32, normalised MIND-SSC descriptor.

    Implementation notes
    --------------------
    The implementation follows the same approach as ConvexAdam
    (Heinrich/Hansen, github.com/multimodallearning/convexAdam) — it uses
    shifts and box filters via conv3d for efficiency.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 4:
        image = image.unsqueeze(0)
    assert image.dim() == 5 and image.shape[1] == 1, \
        f"image must have shape (1, 1, X, Y, Z), got {image.shape}"
    device = image.device
    dtype = image.dtype

    # The 12 pairs of the Self-Similarity Context (Heinrich 2013, Tab. 1).
    # Each pair (a, b) is a direction of the 6-neighborhood + one of the two
    # orthogonal neighbours, giving 12 symmetric pairs.
    six_neighbours = torch.tensor([
        [0, 1, 1], [1, 1, 0], [1, 0, 1],
        [1, 1, 2], [2, 1, 1], [1, 2, 1],
    ], dtype=torch.long, device=device)
    # 12 pair indices (a, b) as in Heinrich 2013
    pair_idx = torch.tensor([
        [0, 1], [1, 2], [2, 0],
        [3, 4], [4, 5], [5, 3],
        [0, 4], [1, 5], [2, 3],
        [3, 1], [4, 2], [5, 0],
    ], dtype=torch.long, device=device)

    # Box-filter kernel for the SSD (sum over a (2*radius+1)^3 patch)
    k = 2 * radius + 1
    box_kernel = torch.ones((1, 1, k, k, k), dtype=dtype, device=device) / float(k ** 3)

    # Padding for shift + box: total pad to avoid spatial shrinkage
    pad = dilation + radius
    img_pad = F.pad(image, [pad] * 6, mode="replicate")

    # For each neighbour, compute I(x) - I(x + d*shift) and then box-filter the SSD,
    # i.e. D_n(x) = mean_{p in patch} (I(x+p) - I(x+p+d*shift))^2
    # The six_neighbours coords are indices 0/1/2 in a 3x3x3 block centred at (1,1,1):
    # we convert them to real offsets by subtracting 1.
    def _ssd_with_shift(shift_xyz):
        """Box-smoothed SSD between image and image translated by shift_xyz*dilation."""
        sx, sy, sz = ((int(s) - 1) * dilation for s in shift_xyz)
        # Extract the "shifted" image by cropping the appropriate padding
        Z, Y, X = image.shape[2:][::-1]   # spatial dims (Z, Y, X) in torch order
        # torch.nn.functional shape: (N, C, X, Y, Z); the project convention is
        # also (N, C, X, Y, Z). Slicing must operate on dims 2,3,4.
        # img_pad has shape (1, 1, X+2pad, Y+2pad, Z+2pad)
        full = img_pad
        # centred window
        c0 = full[:, :, pad:-pad, pad:-pad, pad:-pad]
        # shifted window
        sh = full[:, :,
                  pad + sx:full.shape[2] - pad + sx,
                  pad + sy:full.shape[3] - pad + sy,
                  pad + sz:full.shape[4] - pad + sz]
        diff = (c0 - sh) ** 2
        return F.conv3d(diff, box_kernel, padding=radius)

    # Pre-compute D_n(x) for each neighbour (6 in total). We use no_grad here:
    # mind_ssc is only called to derive the metric (no backprop).
    with torch.no_grad():
        Ds = [_ssd_with_shift(six_neighbours[n]) for n in range(6)]
        # MIND-SSC: for each pair (a,b) compute D_a + D_b, normalise, softmin.
        mind_pairs = []
        for i in range(12):
            a, b = pair_idx[i]
            mind_pairs.append((Ds[a] + Ds[b]) / 2.0)   # (1,1,X,Y,Z)
        # free Ds before concatenating the 12 channels (each of the 6 Ds is ~3.4 GB
        # on 144M-voxel volumes, summing to 20+ GB; they are not needed once
        # merged into mind_pairs).
        del Ds
        mind = torch.cat(mind_pairs, dim=1)            # (1, 12, X, Y, Z)
        del mind_pairs
        # Per-voxel softmin: scale by the mean over the 12 pairs, then exp(-MIND/scale)
        voxel_mean = mind.mean(dim=1, keepdim=True).clamp(min=1e-8)
        mind = torch.exp(-mind / voxel_mean)
    return mind


def mind_ssc_distance(fixed: torch.Tensor, moving: torch.Tensor,
                      radius: int = 1, dilation: int = 2) -> torch.Tensor:
    """Per-voxel distance between the MIND descriptors of fixed and moving.

    On large volumes (Lung-DIR-QA case18+ at 144M voxels) the peak memory of
    keeping md_fixed and md_moving on the GPU SIMULTANEOUSLY exceeds the 32 GB
    of the GV100. To avoid OOM:
      1. compute md_fixed,
      2. move it to CPU,
      3. free the GPU cache,
      4. compute md_moving,
      5. move md_moving to CPU,
      6. distance and abs().mean() on CPU (light: 1 channel × 144M ≈ 0.6 GB).
    For small volumes (< ~50M voxels) everything fits on the GPU without
    problems, but the CPU ping-pong adds no significant overhead (~1-2 s/case).

    Returns
    -------
    Tensor (1, 1, X, Y, Z) float, mean absolute difference over the 12 MIND channels.
    """
    md_f_gpu = mind_ssc(fixed, radius=radius, dilation=dilation)
    md_f = md_f_gpu.detach().cpu()
    del md_f_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    md_m_gpu = mind_ssc(moving, radius=radius, dilation=dilation)
    md_m = md_m_gpu.detach().cpu()
    del md_m_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    out = (md_f - md_m).abs().mean(dim=1, keepdim=True)
    # Move back to the original device (e.g. cuda:1) for consistency with the caller.
    return out.to(fixed.device)


# ---------------------------------------------------------------------------
# LNCC (Local Normalised Cross-Correlation)
# ---------------------------------------------------------------------------
def lncc(fixed: torch.Tensor, moving: torch.Tensor,
         win: int = 5, eps: float = 1e-5) -> torch.Tensor:
    """Per-voxel local NCC over a cubic window (win, win, win).

    LNCC(x) = cov_w(F, M) / sqrt( var_w(F) * var_w(M) + eps )

    Returns Tensor (1, 1, X, Y, Z), values in [-1, 1] (1 = perfect match).
    """
    if fixed.dim() == 3:
        fixed = fixed.unsqueeze(0).unsqueeze(0)
        moving = moving.unsqueeze(0).unsqueeze(0)
    elif fixed.dim() == 4:
        fixed = fixed.unsqueeze(0)
        moving = moving.unsqueeze(0)
    assert fixed.shape == moving.shape

    pad = win // 2
    kernel = torch.ones((1, 1, win, win, win),
                        dtype=fixed.dtype, device=fixed.device) / float(win ** 3)

    f_mean = F.conv3d(fixed, kernel, padding=pad)
    m_mean = F.conv3d(moving, kernel, padding=pad)
    f_var = F.conv3d(fixed * fixed, kernel, padding=pad) - f_mean ** 2
    m_var = F.conv3d(moving * moving, kernel, padding=pad) - m_mean ** 2
    fm_cov = F.conv3d(fixed * moving, kernel, padding=pad) - f_mean * m_mean
    return fm_cov / (torch.sqrt(f_var.clamp(min=0) * m_var.clamp(min=0) + eps))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
SUPPORTED_METRICS = ("abs_diff", "mind_ssc", "lncc")


def compute_residual(fixed: torch.Tensor, moved: torch.Tensor,
                     metric: str = "abs_diff",
                     **kwargs) -> torch.Tensor:
    """Compute the 'residual' term between fixed and moved (= warped moving).

    Parameters
    ----------
    fixed, moved : Tensor di shape (1, X, Y, Z) o (1, 1, X, Y, Z) float
        Same shape required.
    metric : "abs_diff" | "mind_ssc" | "lncc"
    kwargs : passed to the chosen metric (e.g. radius, dilation, win)

    Returns
    -------
    Tensor (1, X, Y, Z): non-normalised residual. For "abs_diff" it is the
    absolute value; for "mind_ssc" it is the difference of the descriptors;
    for "lncc" it is (1 - LNCC) so that large values = large disagreement.
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"metric '{metric}' not supported; "
                         f"choose among {SUPPORTED_METRICS}")

    if metric == "abs_diff":
        return (fixed - moved).abs().squeeze(0) \
            if fixed.dim() == 4 else (fixed - moved).abs().squeeze(0).squeeze(0)

    if metric == "mind_ssc":
        radius = int(kwargs.get("radius", 1))
        dilation = int(kwargs.get("dilation", 2))
        d = mind_ssc_distance(fixed, moved, radius=radius, dilation=dilation)
        # squeeze to (1, X, Y, Z) as for abs_diff
        return d.squeeze(0)  # (1, X, Y, Z)

    if metric == "lncc":
        win = int(kwargs.get("win", 5))
        ncc = lncc(fixed, moved, win=win)         # (1,1,X,Y,Z), in [-1,1]
        return (1.0 - ncc).squeeze(0)             # (1, X, Y, Z), in [0, 2]

    raise RuntimeError("unreachable")
