"""
utils/transformations.py — Coordinate transformations.
Adapted from v1.
"""

import numpy as np
import torch
import torch.nn.functional as nnf
from utils.constants import DEFAULT_DEVICE
from utils.conversions import toTorch


def getNormalToVoxelMatrix(shape):
    normalToVoxelMatrix = np.identity(4)
    for i in range(3):
        normalToVoxelMatrix[i, i] = (shape[i] - 1) / 2
        normalToVoxelMatrix[i, -1] = (shape[i] - 1) / 2
    return normalToVoxelMatrix


def getGrid(normalToSystemMatrix, shape, device=DEFAULT_DEVICE):
    normalToSystemMatrix = normalToSystemMatrix[[2, 1, 0, 3]]
    normalToSystemMatrix = normalToSystemMatrix[:, [2, 1, 0, 3]]
    normalToSystemMatrix = toTorch(normalToSystemMatrix).to(device).float()

    worldGrid = nnf.affine_grid(
        normalToSystemMatrix[:3, :].unsqueeze(0),
        [1, 1, *shape],
        align_corners=True,
    )[0]
    worldGrid = torch.permute(worldGrid, (3, 0, 1, 2))
    worldGrid = worldGrid[[2, 1, 0]]
    return worldGrid


def transformVectorarray(transformationMatrix, vectorArray):
    return (
        torch.matmul(transformationMatrix[:3, :3], vectorArray.reshape(3, -1))
        + transformationMatrix[:3, 3:4]
    ).reshape(vectorArray.shape)
