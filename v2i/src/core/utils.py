import yaml
from copy import deepcopy
from v2i.src.core.common import checkFileExists, readYaml, raiseValueError
from v2i.src.core.defaults import DEFAULT_DICT

class configParser:

    def __init__(self, configPath):
        self.configPath = configPath
        print(self.configPath)

        # Check if configFile exists or not
        if not (checkFileExists(self.configPath)):
            raiseValueError("Config file \"%s\" not found !")
            
        # read Config yaml file
        self.configDict = readYaml(self.configPath)["config"]
        self.defautDict = deepcopy(DEFAULT_DICT)
    
        # Merge the dicts to create a new one
        self.mergeDict()

    def mergeDict(self):
        for key in self.configDict.keys():
            if key not in self.defautDict.keys():
                raiseValueError("Invalid configuration key -> %s "%(key))
            self.defautDict[key] = self.configDict[key]
    
    
    def getValue(self, key):
        if key not in self.defautDict:
            raiseValueError("Trying to access invalid key -> %s"%(key))
        else:
            return self.defautDict[key]
    
    def getFullConfigDict(self,):
        return self.defautDict
         



