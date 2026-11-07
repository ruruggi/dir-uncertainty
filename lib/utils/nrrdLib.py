import nrrd
import numpy as np
from utils.conversions import toNumpy

NRRD_ORIENTATION = "space directions"
NRRD_ORIGIN = "space origin"
NRRD_SPACE = "space"
NRRD_SPACE_DEFAULT = "left-posterior-superior"


def getPixelArray(nrrd):
    return nrrd[0]


def getVoxelToWorldMatrix(nrrd):
    voxelToWorldMatrix = np.identity(4)
    voxelToWorldMatrix[:3, :3] = nrrd[1][NRRD_ORIENTATION]
    voxelToWorldMatrix[:3, 3] = nrrd[1][NRRD_ORIGIN]
    return voxelToWorldMatrix


def writeNrrd(filename, scan):
    header = {
        NRRD_ORIENTATION: scan.getVoxelToWorldMatrix()[:3, :3],
        NRRD_ORIGIN: scan.getOrigin(),
        NRRD_SPACE: NRRD_SPACE_DEFAULT,
    }
    if scan.getPixelArray().shape[0] != 1:
        nrrd.write(filename, toNumpy(scan.getPixelArray()), header=header)
    else:
        nrrd.write(filename, toNumpy(scan.getPixelArray()[0]), header=header)
