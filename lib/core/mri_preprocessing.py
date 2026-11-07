"""
core/mri_preprocessing.py — Helpers to pre-process MRI images before
passing them to the DIR + heuristic+residual pipeline.

Functions:
    n4_bias_correct(arr, spacing)  -> arr corrected for B1+/B0 inhomogeneity
    histogram_match(moving, fixed) -> moving with histogram matched to fixed

Typical use (before compute_heuristic_uncertainty con metric='abs_diff'):

    fixed_n4 = n4_bias_correct(fixed_arr, spacing_fixed)
    moving_n4 = n4_bias_correct(moving_arr, spacing_moving)
    moving_matched = histogram_match(moving_n4, fixed_n4)
    # then the normal pipeline as for CT

For cross-modal pairs (T1 vs T2, MR vs CT) histogram matching is useless
(different sequences produce different histograms by construction); it is
better to use similarity_metric='mind_ssc' directly in the pipeline and skip
histogram matching, applying only N4.

References:
    Tustison NJ, Avants BB, Cook PA, et al. N4ITK: improved N3 bias correction.
    IEEE TMI 2010; 29(6):1310-1320. DOI: 10.1109/TMI.2010.2046908

    Nyul LG, Udupa JK. On standardizing the MR image intensity scale.
    Magnetic Resonance in Medicine 2000; 42(6):1072-1081.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def n4_bias_correct(arr: np.ndarray,
                    spacing: Tuple[float, float, float],
                    shrink_factor: int = 4,
                    n_iterations=(50, 50, 50, 50),
                    convergence_threshold: float = 0.001,
                    bspline_fitting_distance: float = 200.0
                    ) -> np.ndarray:
    """N4 bias-field correction via antspyx (direct wrapper of N4BiasFieldCorrection).

    Parameters
    ----------
    arr : ndarray (X, Y, Z) float, raw MR intensity.
    spacing : (sx, sy, sz) mm.
    shrink_factor : downsampling for the field estimation (default 4 = ~64x faster).
    n_iterations : tuple of iterations per level (multi-resolution).
    convergence_threshold : stopping threshold.
    bspline_fitting_distance : B-spline node spacing in mm (default 200).

    Returns
    -------
    ndarray same shape, float32, corrected intensity.

    Note: the operation typically takes 20-60 s on 256³ volumes.
    """
    try:
        import ants
    except ImportError as e:
        raise ImportError(
            "antspyx is required for n4_bias_correct. Install with:\n"
            "    python -m pip install antspyx"
        ) from e

    img = ants.from_numpy(np.ascontiguousarray(arr.astype(np.float32)),
                          spacing=tuple(float(s) for s in spacing))
    corrected = ants.n4_bias_field_correction(
        img,
        shrink_factor=int(shrink_factor),
        convergence={"iters": list(n_iterations),
                     "tol": float(convergence_threshold)},
        spline_param=float(bspline_fitting_distance),
    )
    return corrected.numpy().astype(np.float32, copy=False)


def histogram_match(moving: np.ndarray,
                    fixed: np.ndarray,
                    n_bins: int = 256,
                    n_match_points: int = 12,
                    threshold_at_mean: bool = True) -> np.ndarray:
    """Histogram matching (Nyul-style) of moving onto fixed via antspyx.

    Parameters
    ----------
    moving, fixed : ndarray same anatomic region (any shape, must be
        compatible with ants.from_numpy).
    n_bins : number of bins of the intermediate histogram.
    n_match_points : number of percentile landmarks to match.
    threshold_at_mean : if True, excludes voxels below the mean (background).

    Returns
    -------
    moving_matched: ndarray same shape as moving, float32.
    """
    try:
        import ants
    except ImportError as e:
        raise ImportError("antspyx is required for histogram_match.") from e

    f_a = ants.from_numpy(np.ascontiguousarray(fixed.astype(np.float32)))
    m_a = ants.from_numpy(np.ascontiguousarray(moving.astype(np.float32)))
    m_matched = ants.histogram_match_image(
        m_a, f_a,
        number_of_histogram_bins=int(n_bins),
        number_of_match_points=int(n_match_points),
        use_threshold_at_mean_intensity=bool(threshold_at_mean),
    )
    return m_matched.numpy().astype(np.float32, copy=False)


def preprocess_mri_pair(fixed: np.ndarray,
                        moving: np.ndarray,
                        spacing_fixed: Tuple[float, float, float],
                        spacing_moving: Tuple[float, float, float],
                        do_n4: bool = True,
                        do_histogram_match: bool = True
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Full pipeline for same-sequence MR-MR:
        N4 bias correction on both + histogram matching moving -> fixed.
    """
    if do_n4:
        fixed = n4_bias_correct(fixed, spacing_fixed)
        moving = n4_bias_correct(moving, spacing_moving)
    if do_histogram_match:
        moving = histogram_match(moving, fixed)
    return fixed, moving
