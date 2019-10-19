import yaml
from copy import deepcopy
from gym.spaces import Discrete
import numpy as np

from v2i.src.core.common import checkFileExists, readYaml, raiseValueError
from v2i.src.core.defaults import DEFAULT_DICT
from v2i.src.core.constants import LANES, IDM_CONSTS

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


def laneMap2Dict(laneMap, key):
    d = {}
    for lane in range(0, LANES):
        for vehicle in laneMap[lane]:
            vehID = vehicle["id"]
            d[vehID] = vehicle['acc']
    return d

def idmAcc(sAlpha, speedDiff, speed, viewRange, nonEgoMaxVel):
        '''
        Make sure to edit function at idm.py if you change anything here

        IDM Equation : https://en.wikipedia.org/wiki/Intelligent_driver_model
        
        Modified IDM Equation includes the sense of local view to non-Ego vehicles
        Modified IDM Details : https://github.com/mynkpl1998/single-ring-road-with-light/blob/master/SingleLaneIDM/Results%20Analysis.ipynb
        '''
        if sAlpha > viewRange:
            sAlpha = np.clip(sAlpha, a_min=0, a_max=viewRange)
            speedDiff = speed - nonEgoMaxVel
        sStar = IDM_CONSTS['MIN_SPACING'] + (speed * IDM_CONSTS['HEADWAY_TIME']) + ((speed * speedDiff)/(2 * np.sqrt(IDM_CONSTS['MAX_ACC'] * IDM_CONSTS['DECELERATION_RATE'])))
        acc = IDM_CONSTS['MAX_ACC'] * (1 - ((speed / self.maxVel)**IDM_CONSTS['DELTA']) - ((sStar/sAlpha)**2))
        return acc