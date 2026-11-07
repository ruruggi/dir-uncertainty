"""
dl/model_loader.py — Wrapper to load the DL uncertainty model from v1.

Uses the v1 FixedMeanPatchRegisterer to produce μ (mean DVF) and σ (std).
The DL architecture files are in dl_arch/ (copied from v1 DL/).
"""

import sys
import os
import numpy as np
import torch

# Add dl_arch to path so that v1 imports (DL.Model.*, etc.) resolve
_dl_arch_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dl_arch")
if _dl_arch_dir not in sys.path:
    sys.path.insert(0, os.path.dirname(_dl_arch_dir))

# Also ensure the project root is in the path for Registration/ContourPropagation imports
_project_root = os.path.dirname(os.path.dirname(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def load_registerer(model_name="L20S3-4",
                    patch_size=(256, 256, 96),
                    voxel_spacing=(1.9531, 1.9531, 2.0)):
    """
    Load the pre-trained FixedMeanPatchRegisterer.

    Returns a registerer object whose .process(fixed, moving, dvf) method
    returns (movedScan, probabilisticDVF) where probabilisticDVF has .stdev.
    """
    from Registration.fixedMeanRegistration import FixedMeanPatchRegisterer

    model_path = os.path.join(_project_root, "TrainedModels", model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    registerer = FixedMeanPatchRegisterer(
        model_path,
        patchSize=patch_size,
        voxelSpacing=voxel_spacing,
    )
    return registerer


def extract_mu_sigma(probabilistic_dvf):
    """
    Extract mean DVF (μ) and std (σ) from a GaussianVectorField.

    Returns
    -------
    mu : Tensor (3, Z, Y, X) — mean DVF in mm
    sigma : Tensor (3, Z, Y, X) — per-voxel std in mm
    """
    mu = probabilistic_dvf.getMeanVectorField()
    sigma = probabilistic_dvf.getVectorFieldUncertainty()

    if isinstance(mu, np.ndarray):
        mu = torch.from_numpy(mu).float()
    if isinstance(sigma, np.ndarray):
        sigma = torch.from_numpy(sigma).float()

    return mu, sigma


def extract_uncertainty_magnitude(probabilistic_dvf):
    """
    Extract scalar uncertainty magnitude ‖σ‖ from probabilistic DVF.

    Returns ndarray (Z, Y, X) — transposed for viewer orientation.
    """
    sigma = probabilistic_dvf.getVectorFieldUncertainty()
    if torch.is_tensor(sigma):
        sigma = sigma.cpu().numpy()

    magnitude = np.linalg.norm(sigma, axis=0)
    if magnitude.ndim == 4:
        magnitude = magnitude[0]

    # Transpose from (X, Y, Z) to (Z, Y, X) for viewer
    return np.transpose(magnitude, (2, 1, 0))
