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
        self.allRegs = list(self.commIndexMap.keys())

    def reset(self):
        self.trueAge.fill(0.0)
        self.agentAge.fill(0.0)
    
    def step(self, oldOccGrid, newOccGrid, oldVelGrid, newVelGrid, query):
        # ---- Update true age ---- #
        for lane in range(0, LANES):
            for col in range(0, self.gridHandler.numCols):
                if oldOccGrid[lane][col] != newOccGrid[lane][col] or newVelGrid[lane][col] != oldVelGrid[lane][col]:
                    self.trueAge[lane][col] = 0.0
                else:
                    self.trueAge[lane][col] = np.clip(self.trueAge[lane][col] + self.ageStep, 0.0, 1.0)
        
        # ---- update agent age ---- #
        assert query in list(self.commMap.values())
        for lane in range(0, LANES):
            for col in range(0, self.gridHandler.numCols):
                if col in self.commIndexMap[query]:
                    # Copy true age of the queried region
                    self.agentAge[lane][col] = self.trueAge[lane][col]
                else:
                    self.agentAge[lane][col] = np.clip(self.agentAge[lane][col] + self.ageStep, 0.0, 1.0)
        
        occGrid = newOccGrid.copy()
        velGrid = newVelGrid.copy()
        
        for key in self.allRegs:
            if key == query:
                for col in self.commIndexMap[key]:
                    pass
            else:
                for col in self.commIndexMap[key]:
                    occGrid[:, col] = oldOccGrid[:, col]
                    velGrid[:, col] = oldVelGrid[:, col]
        
        return occGrid, velGrid, self.getAgentAge()
    
    def getAgeValues(self, ageVector):
        assert ageVector.shape[0] == 2
        assert ageVector.shape[1] == self.gridHandler.numCols
        return np.concatenate((ageVector[:, 0:int(self.numSensors/4)], ageVector[:, -int(self.numSensors/4):]), axis=1)
    
    def getAgentAge(self):
        return self.getAgeValues(self.agentAge).copy()
    
    def getTrueAge(self):
        return self.getAgeValues(self.trueAge).copy()