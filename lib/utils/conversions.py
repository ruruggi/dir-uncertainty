"""
utils/conversions.py — Array type conversion utilities.
Cleaned version from v1.
"""

import numpy as np
import torch


def toNumpy(arr):
    """Convert array-like to numpy ndarray."""
    if isinstance(arr, np.ndarray):
        return arr
    if isinstance(arr, torch.Tensor):
        if arr.is_cuda:
            arr = arr.cpu()
        return arr.detach().numpy()
    return np.array(arr)


def toTorch(arr):
    """Convert array-like to torch Tensor."""
    if isinstance(arr, np.ndarray):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return torch.from_numpy(np.ascontiguousarray(arr))
    if isinstance(arr, torch.Tensor):
        return arr
    return torch.tensor(arr)


def toArrayType(arr, referenceArray):
    """Convert arr to the same type and device as referenceArray."""
    if isinstance(referenceArray, np.ndarray):
        return toNumpy(arr).astype(referenceArray.dtype)
    if isinstance(referenceArray, torch.Tensor):
        return toTorch(arr).type(referenceArray.dtype).to(referenceArray.device)


def toIntMaybe(s):
    try:
        return int(s)
    except (ValueError, TypeError):
        return s
