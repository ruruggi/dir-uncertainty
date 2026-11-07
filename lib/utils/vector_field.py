"""
utils/vector_field.py — VectorField class (simplified, no embedded sampling).
Sampling logic is now in core/sampling.py.
"""

import numpy as np
import warnings
import torch
import os

from utils.scan import Scan
from utils.array_utils import getDimensions, expandToSize, getNorm
from utils.conversions import toNumpy, toTorch
from utils.io_utils import isBdf, readBdf


class VectorField(Scan):
    """
    General class to deal with VectorFields.
    No embedded sampling logic — use core.sampling.DVFSampler for ensemble generation.
    """

    def __init__(self, filePath=None, array=None, vtwMatrix=None, inMM=None):
        if filePath is not None and isinstance(filePath, Scan):
            super().__init__(filePath=filePath, array=array, vtwMatrix=vtwMatrix)
            self.inMM = getattr(filePath, "inMM", True)
        elif filePath is not None and isBdf(filePath):
            self._loadBdf(filePath)
            self.inMM = True
        else:
            super().__init__(filePath=filePath, array=array, vtwMatrix=vtwMatrix)
            if inMM is not None:
                self.inMM = inMM
            else:
                self.inMM = filePath is not None
        assert self.pixelArray.shape[0] == getDimensions(self.pixelArray) - 1

    def getVectorField(self):
        return self.getPixelArray()

    def _loadBdf(self, filePath):
        self.pixelArray, voxelSpacing = readBdf(filePath)
        self.voxelToWorldMatrix = np.identity(4)
        self._setVoxelSpacing(voxelSpacing)

    def toVoxels(self):
        if not self.inMM:
            warnings.warn("Array is likely already in voxel coordinates.")
        else:
            self.inMM = False
        self._setPixelArray(
            self.getPixelArray()
            / expandToSize(self.getVoxelSpacing(), self.getPixelArray(), -1)
        )

    def toMM(self):
        if self.inMM:
            warnings.warn("Array is likely already in world coordinates.")
        else:
            self.inMM = True
        self._setPixelArray(
            self.getPixelArray()
            * expandToSize(self.getVoxelSpacing(), self.getPixelArray(), -1)
        )

    def _getAttributesWithoutRescaling(self):
        return []

    def _loadDicom(self, dcmPath):
        from utils.dicom_vf import readDicomVectorField
        if os.path.isdir(dcmPath):
            dcmFiles = [os.path.join(dcmPath, f) for f in os.listdir(dcmPath)
                        if f.lower().endswith(('.dcm', '.reg'))]
            if not dcmFiles:
                raise ValueError(f"No DICOM/REG files found in {dcmPath}")
            selected = dcmFiles[0]
            for f in dcmFiles:
                if "reg" in f.lower():
                    selected = f
                    break
            dcmPath = selected
        self.pixelArray, self.voxelToWorldMatrix = readDicomVectorField(dcmPath)


class GaussianVectorField(VectorField):
    """
    VectorField with associated per-voxel standard deviation (from dl_arch model).
    Used to store μ and σ — actual sampling is done by core.sampling.DVFSampler.
    """

    def __init__(self, filePath=None, array=None, stdev=None, vtwMatrix=None):
        if filePath is not None:
            super().__init__(filePath=filePath)
            self.stdev = self.pixelArray[3:6]
            self.pixelArray = self.pixelArray[:3]
            self.inMM = True
        elif stdev is None:
            self.pixelArray = array[:3]
            self.stdev = array[3:6]
            self.voxelToWorldMatrix = vtwMatrix
            self.inMM = False
        else:
            self.pixelArray = array
            self.stdev = stdev
            self.voxelToWorldMatrix = vtwMatrix
            self.inMM = False

    def getMeanVectorField(self):
        return self.getPixelArray()

    def getVectorFieldUncertainty(self):
        return self.stdev

    def getVectorFieldUncertaintyMagnitude(self):
        return getNorm(self.stdev)

    def toVoxels(self):
        super().toVoxels()
        self.stdev = self.stdev / expandToSize(self.getVoxelSpacing(), self.stdev, -1)

    def toMM(self):
        super().toMM()
        self.stdev = self.stdev * expandToSize(self.getVoxelSpacing(), self.stdev, -1)

    def _getAttributesWithoutRescaling(self):
        return ["stdev"]
