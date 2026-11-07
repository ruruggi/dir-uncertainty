"""
utils/scan.py — Scan class for volume data I/O.
Adapted from v1 with updated import paths.
"""

import numpy as np
import torch
from utils.array_utils import isArray, isSubTuple
from utils.constants import DEFAULT_DEVICE, PKL_FILE_EXTENSION
from utils.conversions import toNumpy, toTorch
from utils.io_utils import (
    isDicom, isMha, isMhd, isNumpy, isPickle, isNrrd, isNifti, isBdf, isFtppData, isGz,
    readDicom, readMha, readMhd, readNrrd, readNifti, readPickle, writePickle,
    listDirAbsolute,
)
from utils.transformations import getNormalToVoxelMatrix, getGrid
import gzip as gz


def getClassAttributeKeys(obj):
    return list(obj.__dict__.keys())


class Scan:
    """
    General class to deal with scans.
    Arrays are stored as (C, X, Y, Z), C=1 for images, C=3 for vector fields.
    """

    def __init__(self, filePath=None, array=None, vtwMatrix=None):
        if filePath is not None:
            if isinstance(filePath, Scan):
                self.pixelArray = filePath.pixelArray
                self.voxelToWorldMatrix = filePath.voxelToWorldMatrix
            elif isDicom(filePath):
                self._loadDicom(filePath)
            elif isMha(filePath):
                self._loadMha(filePath)
            elif isMhd(filePath):
                self._loadMhd(filePath)
            elif isNumpy(filePath):
                self._loadNumpy(filePath)
            elif isPickle(filePath):
                self._loadPickle(filePath)
            elif isNrrd(filePath):
                self._loadNrrd(filePath)
            elif isNifti(filePath):
                self._loadNifti(filePath)
            elif isFtppData(filePath):
                self._loadFtppData(filePath)
            elif isGz(filePath):
                with gz.GzipFile(filePath, "r") as f:
                    self.__init__(f)
            else:
                raise ValueError(f"Couldn't load filepath {filePath}")
        elif array is not None:
            self.pixelArray = array
            self.voxelToWorldMatrix = vtwMatrix
        self._setChannelMaybe()

    def getPixelArray(self):
        return self.pixelArray

    def getVoxelToWorldMatrix(self):
        return self.voxelToWorldMatrix

    def _setPixelArray(self, pixelArray):
        self.pixelArray = pixelArray

    def _setVoxelToWorldMatrix(self, voxelToWorldMatrix):
        self.voxelToWorldMatrix = voxelToWorldMatrix

    def _setOrigin(self, origin):
        self.voxelToWorldMatrix[:3, 3] = origin

    def _setVoxelSpacing(self, voxelSpacing):
        for i in range(3):
            self.voxelToWorldMatrix[i, i] = voxelSpacing[i]

    def getShape(self):
        return tuple(self.getPixelArray().shape[-3:])

    def getVoxelCoordinates(self, device=DEFAULT_DEVICE):
        return getGrid(getNormalToVoxelMatrix(self.getShape()), self.getShape(), device)

    def getWorldCoordinates(self, device=DEFAULT_DEVICE):
        return getGrid(self.getNormalToWorldMatrix(), self.getShape(), device)

    def getNormalToWorldMatrix(self):
        return np.matmul(
            self.getVoxelToWorldMatrix(), getNormalToVoxelMatrix(self.getShape())
        )

    def getVoxelSpacing(self):
        return np.diagonal(self.voxelToWorldMatrix)[:3]

    def getOrigin(self):
        return self.voxelToWorldMatrix[:3, 3]

    def getFullResolutionArrayAttributes(self):
        attributes = []
        for attribute in getClassAttributeKeys(self):
            value = getattr(self, attribute)
            if isArray(value) and isSubTuple(self.getShape(), tuple(value.shape)):
                attributes.append(attribute)
        return attributes

    def to(self, device):
        for attribute in self.getFullResolutionArrayAttributes():
            setattr(self, attribute, toTorch(getattr(self, attribute)).to(device))

    def toNumpy(self):
        for attribute in self.getFullResolutionArrayAttributes():
            setattr(self, attribute, toNumpy(getattr(self, attribute)))

    def save(self, savePath):
        assert isPickle(savePath), \
            f"Scan objects must be saved with a '{PKL_FILE_EXTENSION}' extension"
        writePickle(savePath, self)

    def _loadDicom(self, dcmPath):
        import utils.dicoms as dicom
        dcm = readDicom(dcmPath)
        self.pixelArray = dicom.getPixelArray(dcm)
        self.voxelToWorldMatrix = dicom.getVoxelToWorldMatrix(dcm)

    def _loadMha(self, mhaPath):
        import utils.mha as mhaLib
        mha = readMha(mhaPath)
        self.pixelArray = mhaLib.getPixelArray(mha)
        self.voxelToWorldMatrix = mhaLib.getVoxelToWorldMatrix(mha)

    def _loadMhd(self, mhdPath):
        import utils.mha as mhaLib
        mhd = readMhd(mhdPath)
        self.pixelArray = mhaLib.getPixelArray(mhd)
        self.voxelToWorldMatrix = mhaLib.getVoxelToWorldMatrix(mhd)

    def _loadNrrd(self, nrrdPath):
        import utils.nrrdLib as nrrdLib
        nrrd_data = readNrrd(nrrdPath)
        self.pixelArray = nrrdLib.getPixelArray(nrrd_data)
        self.voxelToWorldMatrix = nrrdLib.getVoxelToWorldMatrix(nrrd_data)

    def _loadNifti(self, niftiPath):
        import utils.nifti as niftiLib
        nifti = readNifti(niftiPath)
        self.pixelArray = niftiLib.getPixelArray(nifti)
        self.voxelToWorldMatrix = niftiLib.getVoxelToWorldMatrix(nifti)

    def _loadNumpy(self, numpyPath):
        self.pixelArray = np.load(numpyPath)
        self.voxelToWorldMatrix = np.identity(4)

    def _loadFtppData(self, ftppPath):
        import utils.ftppLib as ftppLib
        self.pixelArray, self.voxelToWorldMatrix = ftppLib.readFtppData(ftppPath)

    def _loadPickle(self, picklePath):
        scanObj = readPickle(picklePath)
        self.__dict__.update(scanObj.__dict__)

    def _setChannelMaybe(self):
        pixelArray = getattr(self, "pixelArray", None)
        if pixelArray is None:
            return
        if len(pixelArray.shape) == 3:
            if isinstance(self.pixelArray, torch.Tensor):
                self.pixelArray = self.pixelArray.unsqueeze(0)
            else:
                self.pixelArray = np.expand_dims(self.pixelArray, 0)


def readPixelArray(path):
    return Scan(path).getPixelArray()
