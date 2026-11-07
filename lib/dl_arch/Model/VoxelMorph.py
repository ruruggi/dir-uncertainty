from copy import deepcopy
from utils.scan import Scan
from torch import nn
import torch
from dl_arch.Constants.variableNames import PATCH_SIZE
from dl_arch.Model.STN import SpatialTransformer
from dl_arch.Model.UNet import UNetWrapper
from dl_arch.Model.gaussianBlur import createGaussianBlurLayer
from utils.config import readCfg
from utils.array_utils import getDimensions
from ast import literal_eval
from dl_arch.Model.samplingLayer import (
    NormalSamplingLayer,
)
import numpy as np
from utils.vectorField import (
    GaussianVectorField,
    VectorField,
)

DIFFEOMORPHIC = "diffeoMorphic"
PROBABILITY_MODEL = "probabilityModel"
DIFFEOMORPHIC_INTEGRATION_STEPS = "diffeoMorphicIntegrationSteps"
NUMBER_SAMPLES = "numberOfSamples"
PREDICT_MEAN = "predictMean"
FIXED_MEAN = "fixedMean"
SAMPLE_GAUSSIAN_BLUR = "sampleGaussianBlur"
PRECISION_LOSS_MODEL = "precisionLossModel"
FIRST_ORDER = "firstOrder"
SECOND_ORDER = "secondOrder"
SAMPLE_DOWN_SAMPLING_FACTOR = "sampleDownSamplingFactor"


class VoxelMorph(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.unet = UNetWrapper(cfg)
        self.targetShape = readCfg(cfg, PATCH_SIZE, (256, 256, 96), literal_eval)
        self.transformer = SpatialTransformer(self.targetShape)
        self.predictMean = readCfg(cfg, PREDICT_MEAN, True, literal_eval)
        self.fixedMean = readCfg(cfg, FIXED_MEAN, True, literal_eval)

        self.evaluation = False
        self.vtwMatrix = np.identity(4)

    def forward(self, x):
        dim = getDimensions(x) - 2

        # print(x.shape)
        z = self.unet(x)  # z is a latent representation of the vectorfield
        z = self.correctMeanField(x, z, dim)
        displacementField = self.getZSample(z, dim)

        movingImage = x[:, 1:2]
        movedImage = self.transformer(movingImage, displacementField)

        if not self.evaluation:
            return movedImage, displacementField, z
        else:
            return self.convertToOutputTypes(movedImage, displacementField, z)

    def correctMeanField(self, x, z, dim):
        if self.fixedMean:
            fixedMean = x[:, 2:]
            z = torch.cat((fixedMean, z), dim=1)
        if not self.predictMean:
            z[:, :dim] = 0 * z[:, :dim]
        return z

    def getZSample(self, z, dim):
        return z[:, :dim]

    def convertToOutputTypes(self, movedImage, displacementField, z):
        movedImages = [
            Scan(array=movedImage[i], vtwMatrix=self.vtwMatrix)
            for i in range(movedImage.shape[0])
        ]
        vectorFields = [
            self.convertToVectorField(displacementField[i], z[i])
            for i in range(displacementField.shape[0])
        ]
        return movedImages, vectorFields

    def convertToVectorField(self, displacementField, z):
        return VectorField(array=displacementField, vtwMatrix=self.vtwMatrix)


class VoxelMorphProbabilisticDiagonal(VoxelMorph):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.nbSamples = readCfg(cfg, NUMBER_SAMPLES, 1, int)

        self.sampleGaussianBlur = readCfg(
            cfg, SAMPLE_GAUSSIAN_BLUR, False, literal_eval
        )
        self.sampleDownSamplingFactor = readCfg(
            cfg, SAMPLE_DOWN_SAMPLING_FACTOR, 1, int
        )
        if self.sampleGaussianBlur:
            self.blurringLayer = createGaussianBlurLayer(cfg)
        else:
            self.blurringLayer = None

        self.samplingLayer = self.createNormalSamplingLayer()

    def createNormalSamplingLayer(self):
        return NormalSamplingLayer(
            self.blurringLayer, self.nbSamples, self.sampleDownSamplingFactor
        )

    def getZSample(self, z, dim):
        mean = z[:, :dim]
        logSigma = z[:, dim:]
        stdev = torch.sqrt(torch.exp(logSigma))
        return self.samplingLayer(mean, stdev)

    def convertToVectorField(self, displacementField, z):
        if self.samplingLayer.blurringLayer.rescale == True:
            stdev = torch.sqrt(torch.exp(z[3:]))
            return GaussianVectorField(
                array=displacementField,
                stdev=stdev,
                vtwMatrix=self.vtwMatrix,
                samplingLayer=self.samplingLayer,
            )
        else:
            newSamplingLayer = deepcopy(self.samplingLayer)
            newSamplingLayer.blurringLayer.rescale =True
            rescaleFactor = newSamplingLayer.blurringLayer.getRescaleTensor(update=True)
            stdev = torch.sqrt(torch.exp(z[3:]))*rescaleFactor
            return GaussianVectorField(
                array=displacementField,
                stdev=stdev,
                vtwMatrix=self.vtwMatrix,
                samplingLayer=newSamplingLayer,
            )

