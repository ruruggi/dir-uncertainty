
from utils.constants import CONFIG_FILE_EXTENSION
from utils.io_utils import getFilePathRecursive, writeJson
import configparser
from dl_arch.Constants.variableNames import *
from configparser import ConfigParser
import os
from collections import OrderedDict




def readCfg(cfgFile, value, default=None, typeFunction=None):
    output = cfgFile.get(value, default)
    if typeFunction is not None and output != default:
        return typeFunction(output)
    else:
        return output


def getExperimentConfigFilePath(experimentPath, name):
    fileName = getConfigFileName(name)
    return getFilePathRecursive(experimentPath, fileName)


def getConfigFileName(name):
    return f"{name}.{CONFIG_FILE_EXTENSION}"


def getExperimentConfig(experimentPath, name):
    config = readConfigFile(getExperimentConfigFilePath(experimentPath, name))
    metaConfig = readMetaConfig(experimentPath)
    if metaConfig is not None:
        config = replaceMetaVariables(config, metaConfig)
    return config


def readMetaConfig(experimentPath):
    try:
        metaConfigPath = getExperimentConfigFilePath(experimentPath, META)
    except:
        return None
    metaConfig = readConfigFile(metaConfigPath)
    metaConfig[META_VARIABLES][EXPERIMENT_PATH] = experimentPath
    return metaConfig



def readConfigFile(filename):
    config = ConfigParser()
    config.read(filename)
    config = transferCommonVariables(config)
    return config


def mergeConfigFiles(config1: ConfigParser, config2: ConfigParser):
    for section in config2.sections():
        config1.add_section(section)
        for key, value in config2[section].items():
            config1.set(section, key, value)
    return config1


def transferCommonVariables(config: ConfigParser):
    if COMMON_CONFIG_VARIABLES in config.sections():
        commonSection = dict(config[COMMON_CONFIG_VARIABLES])
        config.remove_section(COMMON_CONFIG_VARIABLES)
        for section in config.sections():
            for option, value in commonSection.items():
                if not config.has_option(section, option):
                    config.set(section, option, value)
    return config


def replaceMetaVariables(config, metaconfig):
    for section in config.sections():
        for option, value in config[section].items():
            config.set(
                section, option, replaceMetaVariable(value, metaconfig[META_VARIABLES])
            )
    return config




def replaceMetaVariable(value, metaVariables):
    n = len(META_VARIABLE_INDICATOR)
    while META_VARIABLE_INDICATOR in value:
        start = value.find(META_VARIABLE_INDICATOR) + n
        end = value[start:].find(META_VARIABLE_INDICATOR) + start
        if value[start:end] in metaVariables.keys():
            value = (
                value[: start - n] + metaVariables[value[start:end]] + value[end + n :]
            )
        else:
            raise ValueError(
                "Meta variable {} not found in meta.cfg file".format(value[start:end])
            )
    return value


class multidict(OrderedDict):
    _unique = 0  # class variable

    def __setitem__(self, key, val):
        if isinstance(val, dict):
            key += str(self._unique)
            self._unique += 1
        OrderedDict.__setitem__(self, key, val)


def removeSectionOrdinals(configFilePath, config):
    with open(configFilePath, "r") as file:
        filedata = file.read()

    for i, section in enumerate(config.sections()):
        filedata = filedata.replace(
            "[{}]".format(section), "[{}]".format(section[: -len(str(i))])
        )

    with open(configFilePath, "w") as file:
        file.write(filedata)


def getConfigFromVariable(source):
    if not isinstance(source, ConfigParser):
        sourceConfig = ConfigParser()
        sourceConfig.read(source)
        return sourceConfig
    else:
        return source

