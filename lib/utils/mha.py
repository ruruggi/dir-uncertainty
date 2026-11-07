import numpy as np
import medpy.io
from utils.conversions import toNumpy


def getPixelArray(mha):
    array = mha[0]
    dim = len(array.shape)
    if dim == 4:
        array = np.transpose(array, axes=[3, 0, 1, 2])
    return array


def getVoxelToWorldMatrix(mha):
    header = mha[1]
    voxelSpacing = getVoxelSpacing(mha)
    imagePosition = np.array(header.offset)
    imageOrientation = header.direction
    assert np.all(imageOrientation == np.identity(3))
    orientationMatrix = imageOrientation * voxelSpacing
    voxelToWorldMatrix = np.identity(4)
    voxelToWorldMatrix[:3, :3] = orientationMatrix
    voxelToWorldMatrix[:3, 3] = imagePosition
    return voxelToWorldMatrix


def getVoxelSpacing(mha):
    return np.array(mha[1].spacing)


def writeMha(filename, scan):
    # can also write mhd file
    spacing = tuple(scan.getVoxelSpacing())
    orientation = scan.getOrientation()
    offset = tuple(scan.getOrigin())
    header = medpy.io.Header(spacing=spacing, offset=offset)
    header.set_direction(orientation)
    array = toNumpy(scan.getPixelArray())
    if array.shape[0] != 1:
        if len(array.shape) == 4:
            array = np.transpose(array, axes=[1, 2, 3, 0])
        medpy.io.save(array, filename, hdr=header)
    else:
        medpy.io.save(array[0], filename, hdr=header)
