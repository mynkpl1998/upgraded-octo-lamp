import numpy as np
from v2i.src.core.constants import LANES

class age:

    def __init__(self, gridHandler):
        self.gridHandler = gridHandler
        self.trueAge = np.zeros((2, self.gridHandler.numCols))
        self.agentAge = np.zeros((2, self.gridHandler.numCols))
        self.numSensors = int(self.gridHandler.totalCommView/self.gridHandler.cellSize) * LANES
        self.ageStep = 1.0 / (self.numSensors * 2)
        self.commMap = self.gridHandler.commMap
        self.commIndexMap = self.gridHandler.commIndexMap
        self.allRegs = list(self.commIndexMap.keys())
        self.initLocalIndexs()
    
    def initLocalIndexs(self):
        self.localIndexs = []
        allCommIndexs = []
        for reg in self.commIndexMap.keys():
            for col in self.commIndexMap[reg]:
                allCommIndexs.append(col)
        for i in range(0, self.gridHandler.numCols):
            if i not in allCommIndexs:
                self.localIndexs.append(i)
    
    def reset(self):
        self.trueAge.fill(0.0)
        self.agentAge.fill(0.0)
    
    def buildState(self, occTrack, velTrack, occGrid, velGrid, trueOcc, trueVel, query):
        # First Copy Local View Only
        for lane in range(0, LANES):
            for col in self.localIndexs:
                occTrack[lane][col] = trueOcc[lane][col]
                velTrack[lane][col] = trueVel[lane][col]

        if query == "null":
            pass
        else:
            for lane in range(0, LANES):
                for col in self.commIndexMap[query]:
                    occTrack[lane][col] = occGrid[lane][col]
                    velTrack[lane][col] = velGrid[lane][col]

        return occTrack.copy(), velTrack.copy()
    
    def getAgeValues(self, ageVector):
        assert ageVector.shape[0] == 2
        assert ageVector.shape[1] == self.gridHandler.numCols
        return np.concatenate((ageVector[:, 0:int(self.numSensors/4)], ageVector[:, -int(self.numSensors/4):]), axis=1)
    
    def getAgentAge(self):
        return self.getAgeValues(self.agentAge).copy()
    
    def updateAgentAge(self, query):
        self.agentAge += self.ageStep
        self.agentAge = np.clip(self.agentAge, 0.0, 1.0)
        
        if query == 'null':
            pass
        else:
            for lane in range(0, LANES):
                for col in self.commIndexMap[query]:
                    self.agentAge[lane][col] = self.trueAge[lane][col]
    
    def detectChange(self, prevOcc, newOcc, prevVel, newVel):
        sensorsChange = np.zeros(prevOcc.shape)
        for lane in range(0, LANES):
            for col in range(0, prevOcc.shape[1]):
                if prevOcc[lane][col] != newOcc[lane][col] or prevVel[lane][col] != newVel[lane][col]:
                    sensorsChange[lane][col] = 1.0
        return sensorsChange

    def updateTrueAge(self, sensorsChange):
        for lane in range(0, LANES):
            for col in range(0, sensorsChange.shape[1]):
                if sensorsChange[lane][col] == 1.0:
                    self.trueAge[lane][col] = 0.0
                else:
                    self.trueAge[lane][col] = min(self.trueAge[lane][col] + self.ageStep, 1.0)
    
    def getsensorsAgentAge(self):
        agentAgeSensors = []
        for reg in self.commIndexMap.keys():
            for col in self.commIndexMap[reg]:
                agentAgeSensors.append(self.agentAge[:, col])
        agentAgeSensors = np.array(agentAgeSensors)
        return agentAgeSensors.flatten().copy()

    def frame(self, prevOcc, newOcc, prevVel, newVel, query):
        sensorsChange = self.detectChange(prevOcc, newOcc, prevVel, newVel)
        self.updateTrueAge(sensorsChange)
        self.updateAgentAge(query)
        age = self.getsensorsAgentAge()
        return age
