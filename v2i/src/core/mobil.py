import numpy as np

from v2i.src.core.constants import LANES, MOBIL_CONST
from itertools import cycle

class mobil:

    def __init__(self):
        self.politeness = MOBIL_CONST["POLITENESS"]
    
    def buildlist(self, lane, laneMap, key):
        laneList = []
        for element in laneMap[lane]:
            laneList.append(element[key])
        return laneList
    
    def followerList(self, laneList, lane):
        l = {}
        for i in range(0, len(laneList)):
            index = (i - 1) % len(laneList)
            l[(laneList[i], lane)] = laneList[index]
        return l        

    def oldAndNewFollower(self, laneMap):
        lane0List = self.buildlist(0, laneMap, 'id')
        lane1List = self.buildlist(1, laneMap, 'id')
        lane0FollowerList = self.followerList(lane0List, 0)
        lane1FollowerList = self.followerList(lane1List, 1)
 
    def sortLaneMap(self, laneMap):
        for lane in range(0, LANES):
            laneMap[lane] = np.sort(laneMap[lane], order=['pos'])

    def step(self, laneMap):
        self.sortLaneMap(laneMap)
        self.oldAndNewFollower(laneMap)
