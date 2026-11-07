from dl_arch.Model.UNet import UNetWrapper

from dl_arch.Model.VoxelMorph import (
    PROBABILITY_MODEL,
    VoxelMorph,
    VoxelMorphProbabilisticDiagonal,
)
from dl_arch.Constants.variableNames import MODEL_NAME, MODEL
from dl_arch.Model.samplingLayer import (
    PROBABLITY_MODEL_DIAGONAL,
)
from utils.config import readCfg



def modelFactory(cfg):
    if cfg[MODEL][MODEL_NAME] == "UNet":
        return UNetWrapper(cfg[MODEL]).float()
    elif cfg[MODEL][MODEL_NAME] == "VoxelMorph":
        # For backward compatibility
        probabilityModel = readCfg(cfg[MODEL], PROBABILITY_MODEL)
        if probabilityModel is None:
            return VoxelMorph(cfg[MODEL]).float()
        elif probabilityModel == PROBABLITY_MODEL_DIAGONAL:
            return VoxelMorphProbabilisticDiagonal(cfg[MODEL]).float()
        else:
            raise ValueError(
                f"Voxelmorph model with probability model {probabilityModel} is unknown"
            )
    else:
        raise ValueError("Model type with name {} is unknown".format(cfg[MODEL_NAME]))
