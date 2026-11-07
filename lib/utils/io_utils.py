"""
utils/io_utils.py — File I/O utilities.
Adapted from v1 with updated imports.
"""

from configparser import ConfigParser
import pickle
import numpy as np
import pydicom
import os
import shutil
import tempfile
import json
import gzip as gz
import struct
import csv

from utils.constants import (
    CONFIG_FILE_EXTENSION, FTPP_FILE_EXTENSION, JSON_FILE_EXTENSTION,
    MHD_FILE_EXTENSION, NUMPY_ARRAY_EXTENSION, DICOM_FILE_EXTENSION,
    GZ_FILE_EXTENSION, MHA_FILE_EXTENSION, PKL_FILE_EXTENSION,
    NRRD_FILE_EXTENSION, NIFTI_FILE_EXTENSION, BDF_FILE_EXTENSION,
    TOPAS_BIN_EXTENSION, TOPAS_BIN_HEADER_EXTENSION, TXT_FILE_EXTENSION,
)


# --- File type detection ---

def isDicom(filePath):
    if isinstance(filePath, list):
        return all(isDicom(path) for path in filePath)
    if isFileType(filePath, DICOM_FILE_EXTENSION):
        return True
    if os.path.isdir(filePath):
        contents = listDirAbsolute(filePath)
        return any(isFileType(p, DICOM_FILE_EXTENSION) for p in contents if not os.path.isdir(p))
    return False

def isMha(filePath):   return isFileType(filePath, MHA_FILE_EXTENSION)
def isMhd(filePath):   return isFileType(filePath, MHD_FILE_EXTENSION)
def isNumpy(filePath):  return isFileType(filePath, NUMPY_ARRAY_EXTENSION)
def isGz(filePath):     return isFileType(filePath, GZ_FILE_EXTENSION)
def isPickle(filePath): return isFileType(filePath, PKL_FILE_EXTENSION, allowZipped=True)
def isNrrd(filePath):   return isFileType(filePath, NRRD_FILE_EXTENSION)
def isNifti(filePath):  return isFileType(filePath, NIFTI_FILE_EXTENSION, allowZipped=True)
def isBdf(filePath):    return isFileType(filePath, BDF_FILE_EXTENSION)
def isFtppData(filePath): return isFileType(filePath, FTPP_FILE_EXTENSION, allowZipped=True)
def isTxt(filePath):    return isFileType(filePath, TXT_FILE_EXTENSION)
def isConfig(file):     return isinstance(file, ConfigParser) or isFileType(file, CONFIG_FILE_EXTENSION)
def isJson(filePath):   return isFileType(filePath, JSON_FILE_EXTENSTION)


def isFileType(filePath, typeExtension, allowZipped=False):
    ext = getFileExtension(filePath)
    if ext.lower() == typeExtension.lower():
        return True
    if allowZipped and getZippedFileExtension(filePath) == typeExtension:
        return True
    return False


def getFileExtension(filepath):
    return os.path.splitext(filepath)[1][1:]

def getFileName(filepath):
    return os.path.split(os.path.splitext(filepath)[0])[1]

def getZippedFileExtension(filepath):
    return os.path.splitext(getFileName(filepath))[1][1:]


# --- Read functions ---

def readDicom(filepath):
    from utils.dicoms import slicedToSingleDicom
    if isinstance(filepath, list):
        return slicedToSingleDicom(filepath)
    elif isinstance(filepath, str):
        if os.path.isfile(filepath):
            return pydicom.dcmread(filepath, stop_before_pixels=False)
        elif os.path.isdir(filepath):
            return slicedToSingleDicom(listDirAbsolute(filepath))
        else:
            raise ValueError(f"File {filepath} does not exist")
    raise ValueError(f"Unknown dicom format for {filepath}")

def readMha(filepath):
    import medpy.io
    return medpy.io.load(filepath)

def readMhd(filepath):
    import medpy.io
    return medpy.io.load(filepath)

def readNrrd(filepath):
    import nrrd
    return nrrd.read(filepath)

def readNifti(filepath):
    import nibabel
    return nibabel.load(filepath)

def readBdf(filepath):
    file = open(filepath, "rb")
    shape = [struct.unpack("i", file.read(4))[0] for _ in range(3)]
    voxelSpacing = [struct.unpack("f", file.read(4))[0] for _ in range(3)]
    arr = np.fromfile(file, dtype=np.float32).reshape(shape[2], shape[1], shape[0], 3)
    arr = np.transpose(arr, [3, 2, 1, 0])
    file.close()
    return arr, voxelSpacing

def readPickle(filePath):
    if isGz(filePath):
        with gz.GzipFile(filePath, "rb") as pkl:
            return pickle.load(pkl)
    with open(filePath, "rb") as pkl:
        return pickle.load(pkl)

def writePickle(filePath, data):
    if isGz(filePath):
        with gz.GzipFile(filePath, "w") as pkl:
            pickle.dump(data, pkl)
    else:
        with open(filePath, "wb") as pkl:
            pickle.dump(data, pkl)

def readJson(filepath):
    with open(filepath) as f:
        return json.load(f)

def writeJson(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


# --- Directory utilities ---

def listDirAbsolute(dirPath):
    return [os.path.join(dirPath, f) for f in os.listdir(dirPath)]

def listDirRelative(dirPath):
    return os.listdir(dirPath)

def makeFolderMaybe(folderPath):
    if not os.path.isdir(folderPath):
        os.makedirs(folderPath, exist_ok=True)


def getFilePathRecursive(folder, filename, maxDepth=10, throwError=True):
    """
    Search for a file recursively up the directory tree.
    """
    filePath = os.path.join(folder, filename)
    while not os.path.exists(filePath):
        parent = os.path.split(folder)[0]
        if parent == folder: # Reached root
            if throwError:
                raise ValueError(f"Could not find {filename} in {folder} or its parents")
            return ""
        folder = parent
        filePath = os.path.join(folder, filename)
        maxDepth -= 1
        if maxDepth < 0:
            if throwError:
                raise ValueError(f"Could not find {filename} (max depth reached)")
            return ""
    return filePath

    return filePath
