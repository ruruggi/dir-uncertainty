from copy import deepcopy

import warnings
import numpy as np
import torch
from Registration.patchRegistration import PatchRegisterer, makeScan, makeVectorField
from utils.array_utils import concatenate
from utils.conversions import toTorch
from utils.interpolate import resampleScan
from utils.scan import Scan
from utils.vectorField import GaussianVectorField, VectorField
from utils.warp import warpScan
from utils.constants import DEFAULT_DEVICE


class FixedMeanPatchRegisterer(PatchRegisterer):
    def __init__(
        self,
        sessionDir,
        patchSize=None,
        voxelSpacing=None,
        device=DEFAULT_DEVICE,
        dvfInput=True,
        applyPrewarping=False,
    ):
        super().__init__(sessionDir, patchSize, voxelSpacing, device)
        self.dvfInput = dvfInput
        self.applyPrewarping = applyPrewarping

    def process(self, fixedScan, movingScan, fixedVectorField):
        fixedScan = makeScan(fixedScan)
        movingScan = makeScan(movingScan)
        fixedVectorField = makeVectorField(fixedVectorField)
        inputImage = self.preProcess(fixedScan, movingScan, fixedVectorField)
        output = self.register(inputImage)
        movedScan, vectorField = self.postProcess(
            fixedScan, movingScan, fixedVectorField, output
        )
        return movedScan, vectorField

    def preProcess(self, fixedScan: Scan, movingScan: Scan, vectorField: VectorField):

        if self.applyPrewarping:
            movingScan = warpScan(movingScan, vectorField.getPixelArray())
        inputArray = super().preProcess(fixedScan, movingScan)
        if not self.dvfInput:
            return inputArray
        vectorField = self.resampleVectorField(
            vectorField, targetshape=inputArray.shape[-3:]
        )
        vectorFieldArray = (
            toTorch(vectorField.getPixelArray()).float().unsqueeze(0).to(self.device)
        )

        return concatenate((inputArray, vectorFieldArray), 1)

    def postProcess(
        self, fixedScan: Scan, movingScan: Scan, fixedVectorField: VectorField, output
    ):
        vectorField = output
        vectorField = self.resampleVectorField(output, fixedScan.getShape())
        # Convert DL-predicted stdev (still in voxel units of the resampled
        # grid) to mm with the correct channel-vs-spacing ordering.
        # pixelArray and stdev channels follow the (dz, dy, dx) order set by
        # dicom_vf.py:105 (`vectorArray[::-1, ...]`), while voxelSpacing from
        # the v2w diagonal is (X_res, Y_res, Z_res). Reverse the spacing so
        # ch[i] is multiplied by the matching physical resolution. Without
        # this reversal, an anisotropic spacing (typical thoracic CT: 0.98 vs
        # 2.0 mm) introduces a channel-permuted scale factor that produced
        # the spurious "sigma_z >= 2*sigma_xy" artefact observed in earlier
        # ensembles.
        spacing = np.asarray(vectorField.getVoxelSpacing())[::-1].copy()
        if torch.is_tensor(vectorField.stdev):
            spacing_t = torch.tensor(
                spacing,
                dtype=vectorField.stdev.dtype,
                device=vectorField.stdev.device,
            )
            vectorField.stdev = vectorField.stdev * spacing_t.view(3, 1, 1, 1)
        else:
            vectorField.stdev = vectorField.stdev * spacing.reshape(3, 1, 1, 1)
        # Replace the DL-predicted mean DVF with the input Velocity field,
        # which is the deterministic baseline that both Heuristic and
        # Smolders ensembles must share. Resample to the target grid via the
        # same world-coord-aware resampleScan that the Heuristic path uses,
        # so the two pipelines produce identical mu_dvf. The previous
        # implementation set pixelArray directly to the unresampled input
        # array, which (combined with the downstream prob_dvf.toMM() in
        # runner.py) double-rescaled the mean by mismatched channel indices
        # and yielded mu_s != mu_h.
        fixedVectorField_resampled = resampleScan(
            fixedVectorField,
            fixedScan.getVoxelToWorldMatrix(),
            fixedScan.getShape(),
        )
        vectorField._setPixelArray(fixedVectorField_resampled.getPixelArray())
        vectorField.to(self.device)
        vectorField._setVoxelToWorldMatrix(fixedScan.getVoxelToWorldMatrix())
        vectorField.inMM = True   # both pixelArray and stdev are now in mm
        movedScan = warpScan(deepcopy(movingScan), vectorField.getVectorField())
        return movedScan, vectorField


class MultiFixedMeanPatchRegisterer:
    def __init__(
        self,
        sessionDirs,
        patchSize=None,
        voxelSpacing=None,
        device=DEFAULT_DEVICE,
        dvfInput=True,
        applyPrewarping=False,
    ):
        self.patchRegisterers = [
            FixedMeanPatchRegisterer(
                sessionDir, patchSize, voxelSpacing, device, dvfInput, applyPrewarping
            )
            for sessionDir in sessionDirs
        ]
        self.device = device

    def process(self, fixedScan, movingScan, fixedVectorField):

        movedScan, vectorField = self.patchRegisterers[0].process(
            fixedScan, movingScan, fixedVectorField
        )

        currentVectorFieldUncertainty = vectorField.getVectorFieldUncertainty()
        for alternativeRegisterer in self.patchRegisterers[1:]:
            _, alternativeVectorField = alternativeRegisterer.process(
                fixedScan, movingScan, fixedVectorField
            )
            assert type(vectorField) == type(alternativeVectorField)
            assert type(vectorField) == GaussianVectorField
            assert type(vectorField.samplingLayer) == type(
                alternativeVectorField.samplingLayer
            )
            if type(vectorField.samplingLayer.blurringLayer) != type(
                alternativeVectorField.samplingLayer.blurringLayer
            ) or vectorField.samplingLayer.blurringLayer.isEqual(
                alternativeVectorField.samplingLayer.blurringLayer
            ):
                warnings.warn(
                    "Vectorfields did not have the same blurring layer, the first one is retained."
                )
            alternativeVectorFieldUncertainty = (
                alternativeVectorField.getVectorFieldUncertainty()
            )
            currentVectorFieldUncertainty[
                alternativeVectorFieldUncertainty > currentVectorFieldUncertainty
            ] = alternativeVectorFieldUncertainty[
                alternativeVectorFieldUncertainty > currentVectorFieldUncertainty
            ]
        vectorField._setVectorFieldUncertainty(currentVectorFieldUncertainty)
        return movedScan, vectorField
