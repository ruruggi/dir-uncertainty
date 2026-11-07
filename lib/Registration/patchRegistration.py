from ast import literal_eval
from dl_arch.Constants.variableNames import MODEL
from dl_arch.Model.modelFactory import modelFactory
from utils.interpolate import interpolateArray, interpolateScan, interpolateVectorField
import torch
import numpy as np
from utils.array_utils import concatenate
from utils.boundingBox import BoundingBox, fillArray
from utils.config import getExperimentConfig, readCfg
from utils.constants import EPS, DEFAULT_DEVICE
from utils.normalization import normalizeCT
from utils.scan import Scan
from utils.conversions import toTorch
from copy import deepcopy
from utils.vectorField import VectorField
import copy
from utils.warp import warpScan
import os

class Registerer:
    def __init__(self, sessionDir, device=DEFAULT_DEVICE):
        self.device = device
        self.loadModel(sessionDir)
        return

    def preProcess(self, fixedScan, movingScan):
        raise NotImplementedError

    def postProcess(self, fixedScan, output):
        raise NotImplementedError

    def process(self, fixedScan, movingScan):
        fixedScan = makeScan(fixedScan)
        movingScan = makeScan(movingScan)
        inputImage = self.preProcess(fixedScan, movingScan)
        output = self.register(inputImage)
        movedScan, vectorField = self.postProcess(fixedScan, movingScan, output)
        return movedScan, vectorField

    def register(self, inputImage):
        with torch.no_grad():
            return self.model.forward(inputImage)

    def loadModel(self, sessionDir):
        self.model = modelFactory(getExperimentConfig(os.path.join(sessionDir,'Configs'),MODEL))
        
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(sessionDir,'ModelWeights','model.tr'), map_location=self.device))
        self.model.eval()
        self.model.evaluation = True

    def resampleVectorField(
        self, scan: VectorField, targetshape=None, targetVoxelSpacing=None
    ):
        return interpolateVectorField(scan, targetshape, targetVoxelSpacing)

    def resampleScan(self, scan: Scan, targetshape=None, targetVoxelSpacing=None):
        return interpolateScan(scan, targetshape, targetVoxelSpacing)

    def resampleArray(self, inputArray, targetshape):
        return interpolateArray(inputArray, targetshape)


class PatchRegisterer(Registerer):
    def __init__(self, sessionDir, patchSize=None, voxelSpacing=None, device=DEFAULT_DEVICE):
        super().__init__(sessionDir, device)
        self.patchSize = patchSize
        if voxelSpacing is None:
            self.voxelSpacing = self.getFixedVoxelSpacing(sessionDir)
        else:
            self.voxelSpacing = voxelSpacing
        self.patchWeights = self.getPatchWeights(self.patchSize)
        return

    def register(self, inputImage):
        imageShape = inputImage.shape[-3:]
        boundingBoxes = createPatchBoundingBoxes(
            imageShape, self.patchSize, minOverlap=10
        )
        if len(boundingBoxes) <= 1:
            return self.registerPatch(boundingBoxes[0].crop(inputImage))
        else:
            for i, boundingBox in enumerate(boundingBoxes):
                # print(boundingBox.crop(inputImage).shape)
                output = self.registerPatch(boundingBox.crop(inputImage))
                if i == 0:
                    vectorField = self.createFullResolutionVectorField(
                        imageShape, output
                    )
                    vectorField.to(self.device)
                    weights = torch.zeros(imageShape).to(self.device)
                vectorField, weights = self.insertSubVectorField(
                    vectorField, output, weights, self.patchWeights, boundingBox
                )
            return self.weighVectorField(vectorField, weights)

    def registerPatch(self, inputImage):
        inputImage, bbox = self.padPatch(inputImage)
        with torch.no_grad():
            # print(inputImage.shape)
            output = self.model.forward(inputImage)
        if isinstance(output, tuple):
            output = output[1][
                0
            ]  # VF info is stored as second argument, and we use batch size = 1
        else:
            output = output[0]  # we use batch size = 1
        if not isinstance(output, VectorField):
            raise NotImplementedError
        return self.cropPatch(output, bbox)

    def padPatch(self, inputPatch):
        pad = []
        for i in range(3):
            diff = self.patchSize[i] - inputPatch.shape[2 + i]
            pad.append(diff // 2)
            pad.append((diff // 2 + diff % 2))

        bbox = BoundingBox(
            corner1=np.array(pad[::2]),
            corner2=np.array(pad[::2]) + np.array(inputPatch.shape[2:]),
        )
        return torch.nn.functional.pad(inputPatch, pad[::-1]), bbox

    def cropPatch(self, output: VectorField, bbox: BoundingBox):
        for attribute in output.getFullResolutionArrayAttributes():
            setattr(output, attribute, bbox.crop(getattr(output, attribute)))
        return output

    def preProcess(self, fixedScan: Scan, movingScan: Scan):
        # could be done on the GPU
        fixedScan = self.resampleScan(fixedScan, targetVoxelSpacing=self.voxelSpacing)
        movingScan = self.resampleScan(movingScan, targetshape=fixedScan.getShape())
        inputArray = concatenate(
            (fixedScan.getPixelArray(), movingScan.getPixelArray()), 0
        )
        inputArray = (
            toTorch(normalizeCT(inputArray)).float().unsqueeze(0).to(self.device)
        )
        return inputArray

    def postProcess(self, fixedScan: Scan, movingScan: Scan, output):
        vectorField = output
        vectorField = self.resampleVectorField(output, fixedScan.getShape())
        vectorField._setVoxelToWorldMatrix(fixedScan.getVoxelToWorldMatrix())
        movedScan = warpScan(deepcopy(movingScan), vectorField.getVectorField())
        return movedScan, vectorField

    def createFullResolutionVectorField(self, shape, vectorField: VectorField):
        fullResolutionVectorField = copy.deepcopy(vectorField)
        for attribute in fullResolutionVectorField.getFullResolutionArrayAttributes():
            tmpShape = tuple(
                getattr(fullResolutionVectorField, attribute).shape[:-3]
            ) + tuple(shape)
            setattr(fullResolutionVectorField, attribute, torch.zeros(tmpShape))
        return fullResolutionVectorField

    def insertSubVectorField(
        self,
        vectorField: VectorField,
        subVectorField: VectorField,
        weights,
        subWeights,
        boundingBox: BoundingBox,
    ):
        subVectorFieldShape = subVectorField.getShape()
        if torch.is_tensor(subWeights) and subWeights.shape[-3:] != subVectorFieldShape:
            subWeights = self.getPatchWeights(subVectorFieldShape)
        for attribute in vectorField.getFullResolutionArrayAttributes():
            array = self.insertSubArray(
                getattr(vectorField, attribute),
                getattr(subVectorField, attribute),
                boundingBox,
                subWeights,
            )
            setattr(vectorField, attribute, array)
        weights = self.insertSubArray(weights, subWeights, boundingBox)
        return vectorField, weights

    def weighVectorField(self, vectorField: VectorField, weights):
        for attribute in vectorField.getFullResolutionArrayAttributes():
            array = getattr(vectorField, attribute) / weights
            setattr(vectorField, attribute, array)
        return vectorField

    def insertSubArray(self, array, subArray, boundingBox: BoundingBox, subWeights=1):
        return fillArray(array, subArray * subWeights, boundingBox, fillMode="sum")

    def getFixedVoxelSpacing(self, sessionDir):
        return DEFAULT_VOXEL_SPACING

    def getPatchWeights(self, shape):
        return getPatchWeights(shape, method="linear").to(self.device)


def createPatchBoundingBoxes(imageShape, patchSize, minOverlap=20):
    inputShape = np.array(imageShape)
    patchSize = np.array(patchSize)
    numberOfPatches = np.ceil(
        (inputShape + minOverlap) / (patchSize - minOverlap)
    ).astype(np.int16)
    realOverlap = np.zeros_like(inputShape)
    for i in range(len(inputShape)):
        if numberOfPatches[i] > 1:
            realOverlap[i] = np.floor(
                patchSize[i] - (inputShape[i] - patchSize[i]) / (numberOfPatches[i] - 1)
            ).astype(np.int16)
    boundingBoxes = []
    for x in range(numberOfPatches[0]):
        for y in range(numberOfPatches[1]):
            for z in range(numberOfPatches[2]):
                corner1 = np.array([x, y, z]) * (patchSize - realOverlap)
                corner1[corner1 < 0] = 0  # should never happen
                corner2 = corner1 + patchSize
                toLarge = corner2 - inputShape
                toLarge[toLarge < 0] = 0
                corner1, corner2 = corner1 - toLarge, corner2 - toLarge
                bbox = BoundingBox(corner1=corner1, corner2=corner2)
                corner1[corner1 < 0] = 0  # this can happen
                boundingBoxes.append(bbox)
    return boundingBoxes


def getPatchWeights(patchSize, method="linear"):
    if method == "linear":
        return getLinearPatchWeights(patchSize)
    raise NotImplementedError


def getLinearPatchWeights(patchSize, linearPart=30, offset=2):
    x = getLinSpaceVector(patchSize[0], linearPart, offset)
    y = getLinSpaceVector(patchSize[1], linearPart, offset)
    z = getLinSpaceVector(patchSize[2], linearPart, offset)
    weights = torch.stack(torch.meshgrid(x, y, z))
    return torch.min(weights, dim=0)[0]


def getLinSpaceVector(length, linearPart, offset):
    if 2 * (linearPart + offset) < length:
        return torch.cat(
            (
                torch.ones(offset) * EPS,
                torch.linspace(EPS, 1, linearPart),
                torch.ones(length - 2 * (linearPart + offset)),
                torch.linspace(1, EPS, linearPart),
                torch.ones(offset) * EPS,
            )
        )
    else:
        return torch.cat(
            (
                torch.ones(offset) * EPS,
                torch.linspace(EPS, 1, length // 2 - offset),
                torch.linspace(1, EPS, length // 2 + length % 2 - offset),
                torch.ones(offset) * EPS,
            )
        )


def makeScan(scan):
    if isinstance(scan, Scan):
        return scan
    return Scan(scan)


def makeVectorField(scan):
    if isinstance(scan, VectorField):
        return scan
    return VectorField(scan)
