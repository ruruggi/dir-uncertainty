"""
utils/interpolate.py — Interpolation / resampling utilities.
Adapted from v1 with updated imports.
"""

import copy
import numpy as np
import torch.nn.functional as nnf
from utils.constants import DEFAULT_DEVICE
from utils.conversions import toTorch
from utils.array_utils import expandToSize, interpolate as arrayInterpolate
from utils.transformations import getNormalToVoxelMatrix


def interpolateScan(scan, targetShape=None, targetVoxelSpacing=None):
    scan = copy.deepcopy(scan)
    targetShape, finalVoxelSpacing = getTargetShapeAndFinalVoxelSpacing(
        scan, targetShape, targetVoxelSpacing
    )
    scan._setVoxelSpacing(finalVoxelSpacing)
    scan._setPixelArray(interpolateArray(scan.getPixelArray(), targetShape))
    return scan


def interpolateVectorField(scan, targetShape=None, targetVoxelSpacing=None):
    scan = copy.deepcopy(scan)
    targetShape, finalVoxelSpacing = getTargetShapeAndFinalVoxelSpacing(
        scan, targetShape, targetVoxelSpacing
    )
    scan._setVoxelSpacing(finalVoxelSpacing)
    vectorConversionFactor = getVectorConversionFactor(scan, targetShape)
    for attribute in scan.getFullResolutionArrayAttributes():
        newArray = interpolateArray(getattr(scan, attribute), targetShape)
        if attribute not in scan._getAttributesWithoutRescaling():
            newArray = vectorConversionFactor * newArray
        setattr(scan, attribute, newArray)
    return scan


def getTargetShapeAndFinalVoxelSpacing(scan, targetShape=None, targetVoxelSpacing=None):
    if targetShape is not None and targetVoxelSpacing is not None:
        raise ValueError("Cannot both match a specified voxelspacing and shape")
    originalShape = np.array(scan.getShape())
    originalVoxelSpacing = scan.getVoxelSpacing()
    return _getTargetShapeAndFinalVoxelSpacing(
        originalVoxelSpacing, targetVoxelSpacing, originalShape, targetShape
    )


def _getTargetShapeAndFinalVoxelSpacing(originalVoxelSpacing, targetVoxelSpacing,
                                         originalShape, targetShape):
    if targetVoxelSpacing is not None:
        scaleFactor = originalVoxelSpacing / np.array(targetVoxelSpacing)
        targetShape = np.round((originalShape - 1) * scaleFactor + 1).astype(np.int16)
    finalVoxelSpacing = (
        (originalShape - 1) / (np.array(targetShape) - 1) * originalVoxelSpacing
    )
    return targetShape, finalVoxelSpacing


def getVectorConversionFactor(scan, targetshape):
    vectorConversionFactor = (np.array(targetshape) - 1) / (
        np.array(scan.getShape()) - 1
    )
    return expandToSize(vectorConversionFactor, scan.getPixelArray())


def interpolateArray(array, targetShape):
    return arrayInterpolate(array, tuple(targetShape), align_corners=True, mode="trilinear")


def resampleScan(scan, targetVoxelToWorldMatrix, targetShape,
                 device=DEFAULT_DEVICE, defaultPixelValue=0.0,
                 inplace=False, mode="bilinear"):
    alignCorners = True
    scan.to(device)
    sourceArray = scan.getPixelArray().unsqueeze(0).float() - defaultPixelValue
    targetNormalToVoxelMatrix = getNormalToVoxelMatrix(targetShape)
    sourceNormalToVoxelMatrix = getNormalToVoxelMatrix(scan.getShape())
    import numpy as np
    targetNormalToWorldMatrix = np.matmul(
        targetVoxelToWorldMatrix, targetNormalToVoxelMatrix
    )
    sourceNormalToWorldMatrix = np.matmul(
        scan.getVoxelToWorldMatrix(), sourceNormalToVoxelMatrix
    )
    transformationMatrix = np.matmul(
        np.linalg.inv(sourceNormalToWorldMatrix), targetNormalToWorldMatrix
    )
    transformationMatrix2 = transformationMatrix[[2, 1, 0, 3]]
    transformationMatrix = transformationMatrix2[:, [2, 1, 0, 3]]
    transformationMatrix = toTorch(transformationMatrix).to(device).float()

    targetGrid = nnf.affine_grid(
        transformationMatrix[:3, :].unsqueeze(0),
        [1, 1, *targetShape],
        align_corners=alignCorners,
    )
    newArray = (
        nnf.grid_sample(sourceArray, targetGrid, mode=mode, align_corners=alignCorners)
        + defaultPixelValue
    )
    if not inplace:
        scan = copy.deepcopy(scan)
    scan._setPixelArray(newArray[0])
    scan._setVoxelToWorldMatrix(copy.deepcopy(targetVoxelToWorldMatrix))
    return scan
