from configparser import ConfigParser
import pickle
import numpy as np
import pandas as pd
import pydicom
import os, shutil, tempfile
import json
from utils.constants import (
    CONFIG_FILE_EXTENSION,
    FTPP_FILE_EXTENSION,
    JSON_FILE_EXTENSTION,
    MHD_FILE_EXTENSION,
    NUMPY_ARRAY_EXTENSION,
    DICOM_FILE_EXTENSION,
    GZ_FILE_EXTENSION,
    MHA_FILE_EXTENSION,
    PKL_FILE_EXTENSION,
    NRRD_FILE_EXTENSION,
    NIFTI_FILE_EXTENSION,
    BDF_FILE_EXTENSION,
    TOPAS_BIN_EXTENSION,
    TOPAS_BIN_HEADER_EXTENSION,
    TXT_FILE_EXTENSION,
)
from utils.dicoms import slicedToSingleDicom
import gzip as gz
import medpy.io
import nrrd
import csv
import nibabel
import struct


def isDicom(filePath):
    if isinstance(filePath, list):
        return all(isDicom(path) for path in filePath)
    if isFileType(filePath, DICOM_FILE_EXTENSION):
        return True
    if os.path.isdir(filePath):
        # A directory is considered DICOM if it contains at least one .dcm file.
        # This allows directories with ancillary files (DICOMDIR, .xml, etc.)
        contents = listDirAbsolute(filePath)
        return any(isFileType(p, DICOM_FILE_EXTENSION) for p in contents if not os.path.isdir(p))
    return False


def isMha(filePath):
    return isFileType(filePath, MHA_FILE_EXTENSION)


def isMhd(filePath):
    return isFileType(filePath, MHD_FILE_EXTENSION)


def isNumpy(filePath):
    return isFileType(filePath, NUMPY_ARRAY_EXTENSION)


def isGz(filePath):
    return isFileType(filePath, GZ_FILE_EXTENSION)


def isPickle(filePath):
    return isFileType(filePath, PKL_FILE_EXTENSION, allowZipped=True)


def isNrrd(filePath):
    return isFileType(filePath, NRRD_FILE_EXTENSION)


def isNifti(filePath):
    return isFileType(filePath, NIFTI_FILE_EXTENSION, allowZipped=True)


def isBdf(filePath):
    return isFileType(filePath, BDF_FILE_EXTENSION)



def isBin(filePath):
    return isFileType(filePath, TOPAS_BIN_EXTENSION) or isFileType(
        filePath, TOPAS_BIN_HEADER_EXTENSION
    )

def isFtppData(filePath):
    return isFileType(filePath, FTPP_FILE_EXTENSION, allowZipped=True)

def isTxt(filePath):
    return isFileType(filePath, TXT_FILE_EXTENSION)


def isConfig(file):
    return isinstance(file, ConfigParser) or isFileType(file, CONFIG_FILE_EXTENSION)


def isJson(filePath):
    return isFileType(filePath, JSON_FILE_EXTENSTION)


def isFileType(filePath, typeExtentsion, allowZipped=False):
    ext = getFileExtension(filePath)
    if ext.lower() == typeExtentsion.lower():
        return True
    if allowZipped:
        if getZippedFileExtension(filePath) == typeExtentsion:
            return True
    return False


def getFileExtension(filepath):
    return os.path.splitext(filepath)[1][1:]


def changeFileExtenstion(filepath, newExtension):
    return os.path.splitext(filepath)[0] + f".{newExtension}"


def getZippedFileExtension(filepath):
    return os.path.splitext(getFileName(filepath))[1][1:]


def getFileName(filepath):
    return os.path.split(os.path.splitext(filepath)[0])[1]


def convertExtensions(filePath, newExtension):
    folderPath = os.path.split(filePath)[0]
    fileName = os.path.split(filePath)[1]
    while len(os.path.splitext(fileName)[1]) > 0:
        fileName = os.path.splitext(fileName)[0]
    return os.path.join(folderPath, f"{fileName}.{newExtension}")


def readDicom(filepath):
    if isinstance(filepath, list):
        return slicedToSingleDicom(filepath)
    elif isinstance(filepath, str):
        if os.path.isfile(filepath):
            return pydicom.dcmread(filepath, stop_before_pixels=False)
        elif os.path.isdir(filepath):
            return slicedToSingleDicom(listDirAbsolute(filepath))
        else:
            raise ValueError("File {} does not exist".format(filepath))
    else:
        raise ValueError("Unknown dicom format for {}".format(filepath))


def readMha(filepath):
    return medpy.io.load(filepath)


def readMhd(filepath):
    return medpy.io.load(filepath)


def readImg(filepath, shape, dtype=np.int16):
    with open(filepath, "rb") as f:
        img = np.fromfile(f, dtype=dtype).reshape(shape)
    return img


def readNrrd(filepath):
    return nrrd.read(filepath)


def readNifti(filepath):
    return nibabel.load(filepath)


def readBdf(filepath):
    file = open(filepath, "rb")
    shape = []
    for i in range(3):
        shape.append(struct.unpack("i", file.read(4))[0])
    voxelSpacing = []
    for i in range(3):
        voxelSpacing.append(struct.unpack("f", file.read(4))[0])
    arr = np.fromfile(file, dtype=np.float32).reshape(shape[2], shape[1], shape[0], 3)
    arr = np.transpose(arr, [3, 2, 1, 0])
    return arr, voxelSpacing


def listDirAbsolute(dirPath):
    return [os.path.join(dirPath, filename) for filename in listDirRelative(dirPath)]


def listDirRelative(dirPath):
    return os.listdir(dirPath)


def listDirRelativeAndAbsolute(dirpath):
    return zip(listDirRelative(dirpath), listDirAbsolute(dirpath))


def getFilePathRecursive(folder, filename, maxDepth=5, throwError=True):
    filePath = os.path.join(folder, filename)
    while not os.path.exists(filePath):
        folder = os.path.split(folder)[0]
        # folder = os.path.split(folder)[0]
        # folder = os.path.join(folder, "Configs")
        filePath = os.path.join(folder, filename)
        maxDepth -= 1
        if maxDepth < 0:
            if throwError:
                raise ValueError(
                    f"Could not find {filename} in {folder} or its subfolders"
                )
            else:
                return ""
    return filePath


def readJson(filepath):
    with open(filepath) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
    return jsonObject


def readJsonMaybe(filepath):
    if os.path.isfile(filepath):
        return readJson(filepath)
    return {}


def readTxt(filePath):
    with open(filePath, "r") as f:
        data = f.read()
    return data


def writeTxt(filePath, data):
    with open(filePath, "w") as f:
        f.write(data)


def writeJson(filepath, data):
    with open(filepath, "w") as jsonFile:
        json.dump(data, jsonFile, indent=4)
        jsonFile.close()
    return


def appendRowToCsv(filepath, row):
    with open(filepath, "a", newline="") as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerow(row)
        f_object.close()


def readCsv(filepath):
    dataFrame = pd.read_csv(filepath)
    return dataFrame.to_dict()


def readPickle(filePath):
    if isGz(filePath):
        return readPickleGz(filePath)
    else:
        with open(filePath, "rb") as pkl:
            pickleObject = pickle.load(pkl)
        return pickleObject


def readPickleGz(filePath):
    with gz.GzipFile(filePath, "rb") as pkl:
        pickleObject = pickle.load(pkl)
    return pickleObject


def writePickle(filePath, data):
    if isGz(filePath):
        writePickleGz(filePath, data)
    else:
        with open(filePath, "wb") as pkl:
            pickle.dump(data, pkl)
    return


def writePickleGz(filePath, data):
    with gz.GzipFile(filePath, "w") as pkl:
        pickle.dump(data, pkl)
    return


def writeNumpyGz(arr, filename):
    with gz.GzipFile(filename, "w") as f:
        np.save(f, arr)
    return


def readNumpyGz(filename):
    if isGz(filename):
        with gz.GzipFile(filename, "r") as f:
            return np.load(f, allow_pickle=True)
    else:
        return np.load(filename, allow_pickle=True)


def joinAndMakeFolder(*args):
    folderPath = os.path.join(args[0], *args[1:])
    makeFolderMaybe(folderPath)
    return folderPath


def makeFolderMaybe(folderPath):
    if not os.path.isdir(folderPath):
        os.mkdir(folderPath)


def makeFolderMaybeRecursive(folderPath, maxDepth=2):

    if not os.path.isdir(folderPath):
        if maxDepth > 0:
            makeFolderMaybeRecursive(os.path.split(folderPath)[0], maxDepth - 1)
            makeFolderMaybe(folderPath)
        else:
            raise RecursionError


def isEmptyFolder(path):
    if os.path.isdir(path):
        return len(listDirRelative(path)) == 0
    return True


def getFinalFolders(path, folderDepth):
    if os.path.isfile(path):
        path = os.path.split(path)[0]
    return os.path.join(*pathToList(path)[-folderDepth:])


def pathToList(path):
    path = os.path.normpath(path)
    return path.split(os.sep)


def createTemporaryCopy(path):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    shutil.copy2(path, tmp.name)
    return tmp.name


def zipGz(inputFile, outputFile=None, delete=False):
    if outputFile is None:
        outputFile = inputFile + "." + GZ_FILE_EXTENSION
    with open(inputFile, "rb") as fIn:
        with gz.open(outputFile, "wb") as fOut:
            shutil.copyfileobj(fIn, fOut)
    if delete:
        os.remove(inputFile)


def unzipGz(inputFile, outputFile=None):
    if outputFile is None:
        outputFile = os.path.splitext(inputFile)[0]  # remove gz extension
    with gz.open(inputFile, "rb") as f_in:
        with open(outputFile, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return outputFile


def removeFileMaybe(filePath):
    if os.path.isfile(filePath):
        try:
            os.remove(filePath)
        except:
            print(f"Did not manage to delete {filePath}")
