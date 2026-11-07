from re import A
import numpy as np
from numpy.core.defchararray import greater
import torch
import torch.nn.functional as nnf
from utils.conversion import toTorch, toNumpy, toArrayType


def isArray(array):
    return isinstance(array, np.ndarray) or isinstance(array, torch.Tensor)


def sort(array, axis=-1, descending=False):
    if isinstance(array, np.ndarray):
        res = np.sort(array, axis=axis)
        if descending:
            if axis == -1:
                res = res[::-1]
            else:
                res = np.flip(res, axis=axis)
        return res
    else:
        return torch.sort(array, dim=axis, descending=descending)[0]


def getStd(array, axis=0, keepdims=True, ddof=0):
    if isinstance(array, np.ndarray):
        return np.std(array, axis, keepdims=keepdims, ddof=ddof)
    else:
        return torch.std(array, dim=axis, correction=ddof, keepdim=keepdims)


def getMean(array, axis=0, keepdims=True):
    if isinstance(array, np.ndarray):
        return np.mean(array, axis, keepdims=keepdims)
    else:
        return torch.mean(array, dim=axis, keepdim=keepdims)


def getMax(array, axis=0, keepdims=True):
    if isinstance(array, np.ndarray):
        return np.max(array, axis, keepdims=keepdims)
    else:
        return torch.max(array, dim=axis, keepdim=keepdims)[0]


def getMin(array, axis=0, keepdims=True):
    if isinstance(array, np.ndarray):
        return np.min(array, axis, keepdims=keepdims)
    else:
        return torch.min(array, dim=axis, keepdim=keepdims)[0]


def concatenate(arrays, axis):
    if isinstance(arrays[0], np.ndarray):
        return np.concatenate(arrays, axis)
    else:
        return torch.cat(arrays, dim=axis)


def zerosLike(array):
    if isinstance(array, np.ndarray):
        return np.zeros_like(array)
    else:
        return torch.zeros_like(array)


def sqrt(array):
    if isinstance(array, np.ndarray):
        return np.sqrt(array)
    else:
        return torch.sqrt(array)


def getNorm(array, axis=0, keepdims=True):
    if isinstance(array, np.ndarray):
        return np.linalg.norm(array, axis=axis, keepdims=keepdims)
    else:
        return torch.norm(array, dim=axis, keepdim=keepdims)


def interpolate(array, targetshape, align_corners=True, mode="trilinear"):
    if isinstance(array, np.ndarray):
        array = toTorch(array)
        array = array.float()
        array = interpolateTorch(
            array, targetshape, align_corners=align_corners, mode=mode
        )
        return toNumpy(array)
    else:
        return interpolateTorch(
            array, targetshape, align_corners=align_corners, mode=mode
        )


def interpolateTorch(array, targetshape, align_corners=True, mode="trilinear"):
    dim = getDimensions(array)
    while getDimensions(array) < 5:
        array = array.unsqueeze(0)
    array = nnf.interpolate(array, targetshape, align_corners=align_corners, mode=mode)
    while getDimensions(array) > dim:
        array = array[0]
    return array


def expandToSize(arr1, arr2, axis=-1):
    arr1 = toArrayType(arr1, arr2)
    while getDimensions(arr1) < getDimensions(arr2):
        if isinstance(arr1, torch.Tensor):
            arr1 = arr1.unsqueeze(axis)
        else:
            arr1 = np.expand_dims(arr1, axis)
    return arr1


def getGrad3D(array):
    dim = getDimensions(array)
    if dim == 3:
        if isinstance(array, np.ndarray):
            array = np.expand_dims(array, 0)
        else:
            array = array.unsqueeze(0)
    if isinstance(array, np.ndarray):
        gradDose = np.zeros([3] + list(array.shape[1:]))
    else:
        gradDose = torch.zeros([3] + list(array.shape[1:]))
    gradDose[0, 1:-1, :, :] = array[:, 2:, :, :] - array[:, :-2, :, :]
    gradDose[1, :, 1:-1, :] = array[:, :, 2:, :] - array[:, :, :-2, :]
    gradDose[2, :, :, 1:-1] = array[:, :, :, 2:] - array[:, :, :, :-2]
    return gradDose


def getDimensions(array):
    return len(array.shape)


def isOneHot3D(array):
    return array.max() <= 1 and getDimensions(array) == 4
