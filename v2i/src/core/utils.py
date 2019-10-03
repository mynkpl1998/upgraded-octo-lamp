import yaml
from copy import deepcopy
from gym.spaces import Discrete

from v2i.src.core.common import checkFileExists, readYaml, raiseValueError, reverseDict
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
    
    def setValue(self, key, value):
        if key not in self.defautDict:
            raiseValueError("Trying to access invalid key -> %s"%(key))
        else:
            self.defautDict[key] = value
    
    def getFullConfigDict(self,):
        return self.defautDict
         
class ActionEncoderDecoder:

    def __init__(self, planSpace, querySpace):
        self.planSpace = planSpace
        self.querySpace = querySpace
    
    def encodeActions(self):
        self.actMap = {}
        index = 0
        for planAct in self.planSpace:
            for queryAct in self.querySpace:
                self.actMap[index] = planAct + "," + queryAct
                index += 1
    
    def getActionSpace(self):
        return Discrete(len(self.actMap))
    
    def decodeAction(self, actionKey):
        action = self.actMap[actionKey]
        planAct, commAct = action.split(",")
        return planAct, commAct


def mapKeys(keys):
    mappedDict = {}
    index = 0
    for key in keys:
        mappedDict[index] = key
        index += 1
    return mappedDict, reverseDict(mappedDict)