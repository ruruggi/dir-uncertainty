import numpy as np


def getPixelArray(nifti):
    array = nifti.get_fdata()
    array = np.squeeze(array)  # get rid of time dimension
    if len(array.shape) == 4:
        array = np.transpose(array, [3, 0, 1, 2])
    return array


def getVoxelToWorldMatrix(nifti):
    # nibabel >= 3.0 ha rimosso get_affine(); usare la property .affine.
    # Copia per evitare di mutare l'oggetto nifti underlying.
    voxelToWorldMatrix = np.array(nifti.affine, copy=True)
    voxelToWorldMatrix[0, :] = -voxelToWorldMatrix[0, :]  # LPS<->RAS sign flip
    voxelToWorldMatrix[1, :] = -voxelToWorldMatrix[1, :]
    return voxelToWorldMatrix
