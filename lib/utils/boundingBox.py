from copy import deepcopy
import numpy as np
from utils.array import expandToSize, getDimensions, isArray
from utils.scan import Scan


class BoundingBox:
    def __init__(self, array=None, corner1=None, corner2=None, margin=None):
        if array is None and corner1 is not None and corner2 is not None:
            self.corner1, self.corner2 = corner1, corner2
        elif array is not None:
            self.corner1, self.corner2 = findBbox(array)
        else:
            raise ValueError("Please specify an argument")
        if margin is not None:
            self.corner1, self.corner2 = self.getCornersWithMargin(margin)
            if array is None:
                self.corner1, self.corner2 = self.correctCorners(
                    array, self.corner1, self.corner2
                )

    def adjustCorners(self, corner1=None, corner2=None):
        if corner1 is not None:
            self.corner1 = corner1
        if corner2 is not None:
            self.corner2 = corner2

    def cropWithMargin(self, array, margin):
        corner1, corner2 = self.getCornersWithMargin(margin)
        corner1, corner2 = self.correctCorners(array, corner1, corner2)
        return self.crop(array, corner1, corner2)

    def getCornersWithMargin(self, margin):
        corner1 = self.corner1 - margin
        corner2 = self.corner2 + margin
        return corner1, corner2

    def getShape(self):
        return self.corner2 - self.corner1

    def correctCorners(self, array, corner1, corner2):
        corner1, corner2 = deepcopy(corner1), deepcopy(
            corner2
        )  # to avoid inplace change of self.corner
        corner1[corner1 < 0] = 0
        shape = np.array(array.shape)[
            -len(corner1) :
        ]  # In case the array has a higher dim than corner, we ignore the initial dimensions
        corner2[np.greater(corner2, shape)] = shape[np.greater(corner2, shape)]
        return corner1, corner2

    def cropZ(self, array):
        return array[..., self.corner1[-1] : self.corner2[-1]]

    def crop(self, array, corner1=None, corner2=None):
        if corner1 is None:
            corner1 = self.corner1
        if corner2 is None:
            corner2 = self.corner2
        corner1, corner2 = self.correctCorners(array, corner1, corner2)
        arrayDim = len(array.shape)
        bboxDim = len(corner1)
        if arrayDim == bboxDim:
            return array[
                corner1[0] : corner2[0],
                corner1[1] : corner2[1],
                corner1[2] : corner2[2],
            ]
        if arrayDim == bboxDim + 1:
            return array[
                :,
                corner1[0] : corner2[0],
                corner1[1] : corner2[1],
                corner1[2] : corner2[2],
            ]
        if arrayDim == bboxDim + 2:
            return array[
                :,
                :,
                corner1[0] : corner2[0],
                corner1[1] : corner2[1],
                corner1[2] : corner2[2],
            ]
        raise ValueError(
            "Dimension of the bbox and array are not compatible with the current implementation"
        )


def cropScanWithMargin(scan: Scan, bbox: BoundingBox, margin):
    scan._setPixelArray(bbox.cropWithMargin(scan.getPixelArray(), margin))
    return scan


def cropZScan(scan: Scan, bbox: BoundingBox):
    scan._setPixelArray(bbox.cropZ(scan.getPixelArray()))
    return scan


def findBbox(array):
    if array.max() == 0 and array.min() == 0:
        return np.array([np.zeros_like(array.shape), np.array(array.shape)])
    nonzeros = np.array(np.nonzero(array))
    return np.array([nonzeros.min(axis=1), nonzeros.max(axis=1)])


def fillArray(array, subArray, bbox: BoundingBox, fillMode="sum"):
    # subArray can both be a value or an array
    corner1, corner2 = bbox.correctCorners(array, bbox.corner1, bbox.corner2)
    if isArray(subArray):
        assert np.all(subArray.shape[-3:] == corner2 - corner1)
        subArray = expandToSize(subArray, array, axis=-1)

    if fillMode == "sum":
        array[
            ...,
            corner1[0] : corner2[0],
            corner1[1] : corner2[1],
            corner1[2] : corner2[2],
        ] += subArray
    elif fillMode == "replace":
        array[
            ...,
            corner1[0] : corner2[0],
            corner1[1] : corner2[1],
            corner1[2] : corner2[2],
        ] = subArray
    else:
        raise NotImplementedError
    return array
