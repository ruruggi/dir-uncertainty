from utils.interpolate import getTargetShapeAndFinalVoxelSpacing, interpolateArray
from dl_arch.Model.samplingLayer import NormalSamplingLayer
from utils.conversion import toArrayType, toNumpy, toTorch
from dl_arch.Model.gaussianBlur import GaussianBlur
from utils.constants import DEFAULT_DEVICE
from utils.dicomVectorField import readDicomVectorField
from utils.ioUtils import isDicom, readBdf, isBdf
from utils.scan import Scan
from utils.array import getNorm, getDimensions, expandToSize, zerosLike, sqrt
import torch.nn.functional as F
import numpy as np
import os
import warnings
import torch
# from scipy.signal import fftconvolve (Moved to Utils.grf)



DEFAULT_SAMPLE_VOXEL_SPACING = (1.9531, 1.9531, 2.0)
DEFAULT_SAMPLE_PADDING = 0


class VectorField(Scan):
    """
    General class to deal with VectorFields
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
                if filePath is not None:
                    self.inMM = True
                else:
                    self.inMM = False
        assert self.pixelArray.shape[0] == getDimensions(self.pixelArray) - 1

    def getVectorField(self):
        return self.getPixelArray()

    def getVectorFieldWorldCoordinates(self):
        voxelSpacing = expandToSize(self.getVoxelSpacing(), self.getPixelArray())
        return voxelSpacing * self.getPixelArray()

    def _loadBdf(self, filePath):
        self.pixelArray, voxelSpacing = readBdf(filePath)
        self.voxelToWorldMatrix = np.identity(4)
        self._setVoxelSpacing(voxelSpacing)

    def toVoxels(self):
        if not self.inMM:
            warnings.warn(
                "The array is likely already in voxel coordinates, please verify."
            )
        else:
            self.inMM = False
        self._setPixelArray(
            self.getPixelArray()
            / expandToSize(self.getVoxelSpacing(), self.getPixelArray(), -1)
        )

    def toMM(self):
        if self.inMM:
            warnings.warn(
                "The array is likely already in world coordinates, please verify."
            )
        else:
            self.inMM = True
        self._setPixelArray(
            self.getPixelArray()
            * expandToSize(self.getVoxelSpacing(), self.getPixelArray(), -1)
        )

    def save(self, savePath):
        if not self.inMM:
            warnings.warn(
                "Vector fields should generally be saved in MM, whereas now it will be saved in voxel coordinates"
            )
        return super().save(savePath)

    def _getAttributesWithoutRescaling(self):
        return []

    def _loadDicom(self, dcmPath):
        if os.path.isdir(dcmPath):
            # If folder, pick first DICOM file (ideally registration)
            dcmFiles = [os.path.join(dcmPath, f) for f in os.listdir(dcmPath) if f.lower().endswith(('.dcm', '.reg'))]
            if not dcmFiles:
                raise ValueError(f"No DICOM/REG files found in {dcmPath}")
            # Try to find REG or choose the most likely candidate (first file)
            selected = dcmFiles[0]
            for f in dcmFiles:
                if "reg" in f.lower():
                    selected = f
                    break
            dcmPath = selected
            
        self.pixelArray, self.voxelToWorldMatrix = readDicomVectorField(dcmPath)

class ProbabilisticVectorField(VectorField):
    def getMeanVectorField(self):
        return super().getVectorField()

    def getVectorFieldSample(self):
        raise NotImplementedError

    def getVectorFieldUncertainty(self):
        raise NotImplementedError

    def getVectorFieldUncertaintyMagnitude(self):
        return getNorm(self.getVectorFieldUncertainty())


class GaussianVectorField(ProbabilisticVectorField):
    def __init__(
        self,
        filePath=None,
        array=None,
        stdev=None,
        vtwMatrix=None,
        samplingLayer: NormalSamplingLayer = NormalSamplingLayer(),
        sampleVoxelSpacing=DEFAULT_SAMPLE_VOXEL_SPACING,
    ):
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
        self.sampleVoxelSpacing = sampleVoxelSpacing
        self.samplingLayer = samplingLayer
        self.samplingLayer.nbSamples = 1

    def getVectorFieldSample(self, eps=None):
        mean = self.getMeanVectorField()
        if eps is None:
            eps = self.getStandardNormalSample()
        return VectorField(
            array=mean + self.stdev * eps,
            vtwMatrix=self.getVoxelToWorldMatrix(),
            inMM=self.inMM,
        )

    def getVectorFieldUncertainty(self):
        return self.stdev 

    def getSampledVectorFieldUncertainty(self, nbSamples=50):
        sampledStdev = zerosLike(self.stdev)
        for i in range(nbSamples):
            print(i)
            sampledStdev += (
                self.getVectorFieldSample().getPixelArray() - self.getMeanVectorField()
            ) ** 2
        return sqrt(sampledStdev / nbSamples)

    def toVoxels(self):
        super().toVoxels()
        self.stdev = self.stdev / expandToSize(self.getVoxelSpacing(), self.stdev, -1)

    def toMM(self):
        super().toMM()
        self.stdev = self.stdev * expandToSize(self.getVoxelSpacing(), self.stdev, -1)

    def getVectorFieldUncertaintyWorldCoordinates(self):
        if self.inMM:
            return self.getVectorFieldUncertainty()
        else:
            voxelSpacing = expandToSize(
                self.getVoxelSpacing(), self.getVectorFieldUncertainty()
            )
            return voxelSpacing * self.getVectorFieldUncertainty()

    def getSampleVoxelSpacing(self):
        return self.sampleVoxelSpacing

    def getStandardNormalSample(self):
        targetShape, _ = getTargetShapeAndFinalVoxelSpacing(
            self, targetVoxelSpacing=self.sampleVoxelSpacing
        )
        eps = self._getStandardNormalSampleWithShape(targetShape)
        eps = interpolateArray(eps, self.getShape())
        if isinstance(self.getPixelArray(), np.ndarray):
            eps = toNumpy(eps)
        return eps

    def _setVectorFieldUncertainty(self, newVectorfieldUncertainty):
        self.stdev = newVectorfieldUncertainty 

    def _getAttributesWithoutRescaling(self):
        return ["stdev"]

    def _getStandardNormalSampleWithShape(self, shape):
        return self.samplingLayer.getStandardSample([1, 3, *shape], device=DEFAULT_DEVICE)[0]

    def _setSampleDownSamplingFactor(self, newDownSamplingFactor):
        self.samplingLayer._setSampleDownSamplingFactor(newDownSamplingFactor)


    def _updateBlurringLayer(
        self, kernelSize, smoothingSigma, padding=DEFAULT_SAMPLE_PADDING
    ):
        if kernelSize is None or smoothingSigma is None:
            newSamplingLayer = NormalSamplingLayer(
                None, 1, self.samplingLayer.sampleDownSamplingFactor
            )
        else:
            newBlurringLayer = GaussianBlur(kernelSize, smoothingSigma, padding=padding,rescale=True)
            newSamplingLayer = NormalSamplingLayer(
                newBlurringLayer,
                nbSamples=1,
                downSampleFactor=self.samplingLayer.sampleDownSamplingFactor,
            )

        self.samplingLayer = newSamplingLayer

def getNormalSample(array, shape=None):
    if shape is None:
        shape = array.shape
    if isinstance(array, np.ndarray):
        return np.random.normal(loc=0.0, scale=1.0, size=shape)
    elif isinstance(array, torch.Tensor):
        dist = torch.distributions.Normal(
            torch.tensor([0.0]).to(array.device), torch.tensor([1.0]).to(array.device)
        )
        return dist.sample(shape)[..., 0]


def getUniformSample(array):
    if isinstance(array, np.ndarray):
        return np.random.uniform(low=-0.5, high=0.5, size=array.shape)
    elif isinstance(array, torch.Tensor):
        dist = torch.distributions.Uniform(
            torch.tensor([-0.5]).to(array.device), torch.tensor([0.5]).to(array.device)
        )
        return dist.sample(array.shape)[..., 0]




