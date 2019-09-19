import numpy as np
from v2i.src.core.constants import LANES

class age:

    def __init__(self, gridHandler):
        self.gridHandler = gridHandler
        self.trueAge = np.zeros((2, self.gridHandler.numCols))
        self.agentAge = np.zeros((2, self.gridHandler.numCols))
        self.numSensors = int(self.gridHandler.totalCommView/self.gridHandler.cellSize) * LANES
        self.ageStep = 1.0/(self.numSensors * 2)
        self.commMap = self.gridHandler.commMap
        self.commIndexMap = self.gridHandler.commIndexMap
        print("Num sensors : ", self.numSensors)

    def reset(self):
        self.trueAge.fill(1.0)
        self.agentAge.fill(1.0)
    
    def step(self):
        return self.trueAge, self.agentAge
    
    def getAgeValues(self, ageVector):
        assert ageVector.shape[0] == 2
        assert ageVector.shape[1] == self.gridHandler.numCols
        return np.concatenate((ageVector[:, 0:int(self.numSensors/4)], ageVector[:, -int(self.numSensors/4):]), axis=1)
    
    def getAgentAge(self):
        return self.getAgeValues(self.agentAge).copy()
    
    def getTrueAge(self):
        return self.getAgeValues(self.trueAge).copy()