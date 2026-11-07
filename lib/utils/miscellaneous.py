import numpy as np



def getClassName(object):
    return type(object).__name__


def getClassAttributeKeys(object):
    return list(object.__dict__.keys())


def roundUpToOdd(f):
    return np.ceil(f) // 2 * 2 + 1
