import os
import yaml

def checkFileExists(file):
    return os.path.isfile(file)

def readYaml(file):
    with open(file, "r") as handle:
        configDict = yaml.load(handle, Loader=yaml.FullLoader)
    return configDict