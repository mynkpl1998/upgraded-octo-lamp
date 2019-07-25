from gym.spaces import Discrete

from v2i.src.core.constants import IDM_CONSTS
from v2i.src.core.common import getAgentID

class egoController:

    def __init__(self, tPeriod):
        self.initActionSpace()
        self.initParams(tPeriod)

    def initActionSpace(self):
        possibleActions = ['acc', 'dec', 'do-nothing', 'lane-change']
        self.actionSpace = Discrete(len(possibleActions))
        self.planMap = self.buildPlanningActionMap(possibleActions)
    
    def buildPlanningActionMap(self, possibleActs):
        planMap = {}
        for idx, element in enumerate(possibleActs):
            planMap[idx] = element
        return planMap

    def initParams(self, tPeriod):
        self.tPeriod = tPeriod
    
    def performAccelerate(self, laneMap, agentLane, agentIDX):
        pass
    
    def executeAction(self, action, laneMap, agentLane):
        agentIDX = getAgentID(laneMap, agentLane)
        