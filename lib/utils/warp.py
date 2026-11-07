"""
utils/warp.py — Warping utilities using torch grid_sample.
Cleaned version from v1 with improved warp_torch signature.
"""

from copy import deepcopy
import torch
import torch.nn.functional as nnf
import numpy as np
from utils.constants import DEFAULT_DEVICE
from utils.conversions import toArrayType, toNumpy, toTorch
from utils.array_utils import getDimensions


def warp_torch(image_array: torch.Tensor,
               vector_array: torch.Tensor,
               mode: str = "bilinear") -> torch.Tensor:
    """
    Warp an image using a displacement vector field via grid_sample.

    Parameters
    ----------
    image_array : Tensor — (Z, Y, X) or (C, Z, Y, X) image
    vector_array : Tensor — (3, Z, Y, X) displacement in voxels
    mode : str — 'bilinear' or 'nearest'

    Returns
    -------
    warped : Tensor — same shape as image_array
    """
    dim = getDimensions(image_array)
    shape = image_array.shape[-3:]

    grid = _get_mesh_grid(shape).to(vector_array.device)

    # Handle shape mismatch
    v_arr = vector_array.float()
    if v_arr.shape[-3:] != shape:
        v_arr = nnf.interpolate(
            v_arr.unsqueeze(0), size=shape, mode='trilinear', align_corners=True
        ).squeeze(0)

    new_locations = grid + v_arr

    # Normalise to [-1, 1] for grid_sample
    for i in range(3):
        new_locations[i] = 2.0 * (new_locations[i] / (shape[i] - 1) - 0.5)

    new_locations = new_locations.permute(1, 2, 3, 0)
    new_locations = new_locations[..., [2, 1, 0]]  # ZYX → XYZ for grid_sample
    new_locations = new_locations.unsqueeze(0)

    image_input = image_array.float().unsqueeze(0).to(vector_array.device)
    if dim == 3:
        warped = nnf.grid_sample(
            image_input.unsqueeze(0), new_locations,
            align_corners=True, mode=mode
        )[0, 0]
    else:
        warped = nnf.grid_sample(
            image_input, new_locations,
            align_corners=True, mode=mode
        )[0]

    return warped


def warp_scan(scan, vector_field, device=DEFAULT_DEVICE):
    """
    Warp a Scan object using a VectorField.
    Both must be imported from the appropriate modules.
    """
    from utils.vector_field import VectorField
    from utils.scan import Scan

    scan = deepcopy(scan)

    if isinstance(vector_field, VectorField):
        return _warp_scan_with_vf(scan, vector_field, device)
    else:
        warped_array = _warp_array(scan.getPixelArray(), vector_field)
        scan._setPixelArray(warped_array)
        return scan


def _warp_scan_with_vf(scan, vector_field, device=DEFAULT_DEVICE):
    """Warp using VectorField with world coordinate transformation."""
    from utils.transformations import transformVectorarray

    scan.to(device)
    if not vector_field.inMM:
        vector_field.toMM()

    world_transform = (
        vector_field.getWorldCoordinates(device)
        + toTorch(vector_field.getPixelArray()).to(device)
    ).float()

    world_to_normal = (
        toTorch(np.linalg.inv(scan.getNormalToWorldMatrix())).to(device).float()
    )
    normal_transform = transformVectorarray(world_to_normal, world_transform)
    normal_transform = torch.permute(
        normal_transform[[2, 1, 0]], (1, 2, 3, 0)
    ).unsqueeze(0)

    new_array = nnf.grid_sample(
        scan.getPixelArray().float().unsqueeze(0),
        normal_transform,
        align_corners=True,
        mode="bilinear",
    )[0]

    scan._setVoxelToWorldMatrix(vector_field.getVoxelToWorldMatrix())
    scan._setPixelArray(new_array)
    return scan


def _warp_array(image_array, vector_array):
    """Warp a raw array with a vector array."""
    vector_array = toTorch(vector_array)
    warped = warp_torch(
        toArrayType(image_array, vector_array).float(), vector_array
    )
    if not isinstance(image_array, torch.Tensor):
        return toNumpy(warped)
    return warped


def _get_mesh_grid(shape):
    """Create a mesh grid of voxel coordinates."""
    vectors = [torch.arange(0, s) for s in shape]
    grids = torch.meshgrid(vectors, indexing='ij')
    return torch.stack(grids).float()


# Legacy aliases
warpScan = warp_scan
warpTorch = warp_torch
warpArray = _warp_array


def warpStructureSet(structureSet, vectorField, **kwargs):
    """Apply DVF deformation to all ROIs in a StructureSet.

    NOTE: This is a placeholder stub. The contour-warping path of
    ContourPropagation/ is not exercised by the main paper validation
    pipeline (the four-engine LECE/Spearman experiments use landmarks, not
    contours). It is reserved for the downstream applicative dose-accumulation
    study with full ROI contour propagation; see paper §2.6 and §4
    Limitations. To implement: iterate ROIs in the StructureSet, fetch each
    mask, warp via warpArray(mask, vectorField), and rebuild the StructureSet
    with the warped masks (writing back via ContourPropagation.structureSet API).
    """
    raise NotImplementedError(
        "warpStructureSet is not implemented in the released Submission/Scripts/. "
        "The contour-warping path of ContourPropagation/ is reserved for the "
        "downstream dose-accumulation study. Implement by mapping each ROI mask "
        "through warpArray with the provided vector field."
    )
