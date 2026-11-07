import numpy as np
import struct

from utils.ioUtils import isGz
import gzip as gz

CT_ORIENTATIONS = ["HFS", "HFP", "FFS", "FFP"]
DEFAULT_ORIENTATION = "HFS"
DOWNSCALE_FACTOR = 10  # ftpp uses cm instead of mm


def writeFtppData(scan, filepath):
    with open(filepath, "wb") as file:
        file.write(bytes(DEFAULT_ORIENTATION, "ascii"))
        file.write((scan.getOrigin() / DOWNSCALE_FACTOR).astype(np.float32).tobytes())
        file.write(
            (scan.getVoxelSpacing() / DOWNSCALE_FACTOR).astype(np.float32).tobytes()
        )
        data = np.transpose(scan.getPixelArray()[0], (1, 0, 2))
        # data=scan.getPixelArray()[0]
        file.write(np.array(data.shape, dtype=np.int32).tobytes())
        data = data.astype("<i2")
        for z in range(scan.getShape()[2]):
            file.write(data[:, :, z].tobytes())


def writeFtppDoseData(scan, filepath):
    with open(filepath, "wb") as file:
        file.write(bytes(DEFAULT_ORIENTATION, "ascii"))
        file.write((scan.getOrigin() / DOWNSCALE_FACTOR).astype(np.float32).tobytes())
        file.write(
            (scan.getVoxelSpacing() / DOWNSCALE_FACTOR).astype(np.float32).tobytes()
        )
        data = np.transpose(scan.getPixelArray()[0], (1, 0, 2))
        # data=scan.getPixelArray()[0]
        file.write(np.array(data.shape, dtype=np.int32).tobytes())
        data = data.astype(np.float32)
        for z in range(scan.getShape()[2]):
            file.write(data[:, :, z].tobytes())


def readFtppData(filepath):
    if isGz(filepath):
        openFunction = gz.open
    else:
        openFunction = open
    with openFunction(filepath, "rb") as file:
        try:
            if peekBytes(file, 3).decode("ascii") in CT_ORIENTATIONS:
                orientation = file.read(3)
                assert orientation.decode("ascii") == DEFAULT_ORIENTATION
        except:
            print(f"Could not verify orientation of {filepath}")
        origin = np.empty(3, dtype=np.float32)
        for i in range(3):
            # little-endian 4-byte float
            origin[i] = struct.unpack("<f", file.read(4))[0]

        spacing = np.empty(3, dtype=np.float32)
        for i in range(3):
            spacing[i] = struct.unpack("<f", file.read(4))[0]

        size = np.empty(3, dtype=np.int32)
        for i in range(3):
            size[i] = struct.unpack("<i", file.read(4))[0]

        slices = []
        remainingFileSize = lengthOfFile(file) - file.tell()
        if remainingFileSize == np.product(size) * 4:
            for z in range(size[2]):
                plane = np.frombuffer(
                    # Each pixel in the slice needs two bytes.
                    file.read(size[0] * size[1] * 4),
                    dtype="float32",
                ).reshape(size[:2])
                slices.append(plane)
        elif remainingFileSize == np.product(size) * 2:
            for z in range(size[2]):
                plane = np.frombuffer(
                    # Each pixel in the slice needs two bytes.
                    file.read(size[0] * size[1] * 2),
                    dtype="int16",
                ).reshape(size[:2])
                slices.append(plane)
        else:
            raise ValueError
        data = np.expand_dims(np.stack(slices, axis=2), 0)
        data = np.swapaxes(data, 1, 2)
    voxelToWorldMatrix = np.identity(4)
    voxelToWorldMatrix[:3, :3] = spacing * voxelToWorldMatrix[:3, :3] * DOWNSCALE_FACTOR
    voxelToWorldMatrix[:3, 3] = origin * DOWNSCALE_FACTOR
    return data, voxelToWorldMatrix


def peekBytes(f, bytes):
    pos = f.tell()
    line = f.read(bytes)
    f.seek(pos)
    return line


def lengthOfFile(f):
    """Get the length of the file for a regular file (not a device file)"""
    currentPos = f.tell()
    f.seek(0, 2)  # move to end of file
    length = f.tell()  # get current position
    f.seek(currentPos, 0)  # go back to where we started
    return length
