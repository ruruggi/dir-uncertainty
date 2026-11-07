import numpy as np
import torch
from collections.abc import Iterable
import uuid

from utils.ioUtils import readJson, writeJson



def toRadian(angleDegree):
    return (np.pi / 180) * angleDegree


def toDegree(angleRadian):
    return (180 / np.pi) * angleRadian


def toIntMaybe(str):
    try:
        return int(str)
    except:
        return str


def toNumpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    if isinstance(arr, torch.Tensor):
        if arr.is_cuda:
            arr = arr.cpu()
        return arr.numpy()
    else:
        return np.array(arr)


def toTorch(arr):
    if isinstance(arr, np.ndarray):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return torch.from_numpy(np.ascontiguousarray(arr))
    if isinstance(arr, torch.Tensor):
        return arr
    return torch.tensor(arr)


def toArrayType(arr, referenceArray):
    if isinstance(referenceArray, np.ndarray):
        return toNumpy(arr).astype(referenceArray.dtype)
    if isinstance(referenceArray, torch.Tensor):
        return toTorch(arr).type(referenceArray.dtype).to(referenceArray.device)


def pointDictToPoints(pointDict):
    contours = []
    for slice, points in pointDict.items():
        for contour in points:
            contours.append(
                np.concatenate(
                    (contour, slice * np.ones((contour.shape[0], 1))), axis=1
                )
            )
    return np.concatenate(contours, axis=0)




def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def flattenNestedList(list1):
    res = []
    for el in list1:
        if isinstance(el, list):
            res += el
        else:
            res.append(el)
    return res



def convertOneHotToClassIndex(label):
    if hasBackGroundClass(label):
        return np.argmax(label, axis=0)
    else:
        foreGround = np.any(label, axis=0)
        indices = np.argmax(label, axis=0) + 1  # to account for the background
        indices[~foreGround] = 0
        return indices


def hasBackGroundClass(label):
    return (
        np.sum(label, axis=0).min() >= 1
    )  # each voxel is assigned to at least one class


def convertClassIndexToOneHot(label):
    labels = []
    for i in range(0, max(label) + 1):
        labels.append(label == i)
    return np.stack(labels, axis=0)
