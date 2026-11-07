import numpy as np
import pydicom
import torch.nn.functional as nnf
import torch
from utils.constants import DEFAULT_DEVICE
from utils.conversions import toNumpy, toTorch

from utils.transformations import getNormalToVoxelMatrix
from utils.transformations import getGrid

IDENTITY_FRAME_TYPECODE = "125021"


def readDicomRigidTranformation(filePath):
    dcm = pydicom.dcmread(filePath)
    matrixSequence = [
        np.array(
            reg.MatrixRegistrationSequence[0]
            .MatrixSequence[0]
            .FrameOfReferenceTransformationMatrix
        ).reshape(4, 4)
        for reg in dcm.RegistrationSequence
    ]
    transformationMatrix = matrixSequence[0]
    for newMatrix in matrixSequence[1:]:
        transformationMatrix = np.matmul(
            newMatrix, transformationMatrix
        )  # NOT SURE ABOUT THE ORDER, only example had an identiy matrix
    return transformationMatrix


def readVelocityDicomVectorField(filePath):
    dcm = pydicom.dcmread(filePath, stop_before_pixels=True)
    preDeformationTransformationMatrix = np.array(
        dcm.DeformableRegistrationSequence[1]
        .PreDeformationMatrixRegistrationSequence[0]
        .FrameOfReferenceTransformationMatrix
    ).reshape((4, 4))
    postDeformationTransformationMatrix = np.array(
        dcm.DeformableRegistrationSequence[2]
        .PostDeformationMatrixRegistrationSequence[0]
        .FrameOfReferenceTransformationMatrix
    ).reshape((4, 4))
    assert np.allclose(
        np.matmul(
            preDeformationTransformationMatrix, postDeformationTransformationMatrix
        ),
        np.identity(4),
    )

    deformableVectorField = dcm.DeformableRegistrationSequence[
        3
    ].DeformableRegistrationGridSequence[0]
    gridShape = deformableVectorField.GridDimensions
    vectorArray = np.frombuffer(deformableVectorField.VectorGridData, np.float32)
    vectorArray = vectorArray.reshape(
        [gridShape[2], gridShape[1], gridShape[0], 3]
    )  # 3 components per vector
    vectorArray = np.transpose(vectorArray, (3, 2, 1, 0))
    
    # ELEKTA / ML MOD: Reverse the vector components from (x,y,z) DICOM order
    # to (z,y,x) PyTorch grid_sample order. This avoids mismatched offset addition (e.g. dx added to Z axis).
    vectorArray = vectorArray[::-1, ...]
    
    voxelToWorldMatrix = np.identity(4)
    voxelToWorldMatrix[:3, 3] = np.array(deformableVectorField.ImagePositionPatient)
    for i in range(3):
        voxelToWorldMatrix[i, i] = deformableVectorField.GridResolution[i]

    # vectorArray = vectorArray + toNumpy(
    #     getRigidTransformationGrid(preAlignmentMatrix, voxelToWorldMatrix, shape)
    # )

    return vectorArray, voxelToWorldMatrix


def readDicomVectorField(filePath):
    dcm = pydicom.dcmread(filePath, stop_before_pixels=True)
    registrationSequence = [
        reg
        for reg in dcm.DeformableRegistrationSequence
        if reg.RegistrationTypeCodeSequence[0].CodeValue != IDENTITY_FRAME_TYPECODE
    ]
    assert len(registrationSequence) == 1
    registration = registrationSequence[0]
    # assert np.all(
    #     np.array(
    #         registration.PreDeformationMatrixRegistrationSequence[
    #             0
    #         ].FrameOfReferenceTransformationMatrix
    #     ).reshape((4, 4))
    #     == np.identity(4)
    # )

    deformableVectorField = registration.DeformableRegistrationGridSequence[0]
    gridShape = deformableVectorField.GridDimensions
    vectorArray = np.frombuffer(deformableVectorField.VectorGridData, np.float32)
    vectorArray = vectorArray.reshape(
        [gridShape[2], gridShape[1], gridShape[0], 3]
    )  # 3 components per vector
    vectorArray = np.transpose(vectorArray, (3, 2, 1, 0))
    
    # ELEKTA / ML MOD: Reverse the vector components from (x,y,z) DICOM order
    # to (z,y,x) PyTorch grid_sample order. This avoids mismatched offset addition (e.g. dx added to Z axis).
    vectorArray = vectorArray[::-1, ...]
    
    voxelToWorldMatrix = np.identity(4)
    voxelToWorldMatrix[:3, 3] = np.array(deformableVectorField.ImagePositionPatient)
    for i in range(3):
        voxelToWorldMatrix[i, i] = deformableVectorField.GridResolution[i]

    # We need to add a part for the initial rigid alignment to the vectorfield
    shape = vectorArray.shape[1:]
    preAlignmentMatrix = np.array(
        registration.PreDeformationMatrixRegistrationSequence[
            0
        ].FrameOfReferenceTransformationMatrix
    ).reshape((4, 4))

    rigidArray = toNumpy(
        getRigidTransformationGrid(preAlignmentMatrix, voxelToWorldMatrix, shape)
    )
    
    # ELEKTA / ML MOD: Commented out to isolate the purely elastic deformation component
    # for the uncertainty estimation network. (Excludes rigid pre-alignment).
    # vectorArray = vectorArray + rigidArray

    return vectorArray, voxelToWorldMatrix


def getRigidTransformationGrid(
    transformationMatrix, voxelToWorldMatrix, shape, device=DEFAULT_DEVICE
):
    normalToWorldMatrix = np.matmul(voxelToWorldMatrix, getNormalToVoxelMatrix(shape))
    normalToAlignedMatrix = np.matmul(transformationMatrix, normalToWorldMatrix)
    additionalGrid = getGrid(normalToAlignedMatrix, shape, device=device) - getGrid(
        normalToWorldMatrix, shape, device=device
    )
    return additionalGrid
