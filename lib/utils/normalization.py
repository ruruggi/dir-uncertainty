from utils.constants import CT_HU_AIR, CT_HU_BONE

WINDOW_TYPE_HOUNSFIELD = "hounsfield"
WINDOW_TYPE_SOFT_TISSUE = "softtissue"

CT_HU_SOFT_TISSUE_LOWEST = -180
CT_HU_SOFT_TISSUE_HIGHEST = 220


def normalizeCT(array, window=WINDOW_TYPE_HOUNSFIELD):
    if window.lower() == WINDOW_TYPE_HOUNSFIELD:
        return (array - CT_HU_AIR) / (CT_HU_BONE - CT_HU_AIR)
    if window.lower() == WINDOW_TYPE_SOFT_TISSUE:
        return (array - CT_HU_SOFT_TISSUE_LOWEST) / (
            CT_HU_SOFT_TISSUE_HIGHEST - CT_HU_SOFT_TISSUE_LOWEST
        )
    raise NotImplementedError


def unnormalizeCT(array):
    return array * (CT_HU_BONE - CT_HU_AIR) + CT_HU_AIR


def normalizeMinMax(array):
    return (array - array.min()) / (array.max() - array.min())
