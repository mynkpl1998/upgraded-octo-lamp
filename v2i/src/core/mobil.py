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
    
    def successorList(self, targetlane, otherLane, currLaneNum, otherLaneNum):
        s = {}
        for vehicle in targetlane:
            tmpNewLane = np.append(otherLane, (vehicle))
            #tmpNewLane[-1]['lane'] = otherLaneNum
            tmpNewLane = np.sort(tmpNewLane, order=['pos'])
            vehID = np.where(tmpNewLane['pos'] == vehicle['pos'])[0]
            index =  (vehID[0] + 1) % len(otherLane)
            s[(vehicle['id'], currLaneNum)] = index
        return s

    def oldAndNewFollower(self, laneMap):
        lane0List = self.buildlist(0, laneMap, 'id')
        lane1List = self.buildlist(1, laneMap, 'id')
        lane0FollowerList = self.followerList(lane0List, 0)
        lane1FollowerList = self.followerList(lane1List, 1)
        lane0SuccessorList = self.successorList(laneMap[0], laneMap[1], 0, 1)
        lane1successorList = self.successorList(laneMap[1], laneMap[0], 1, 0)

        return (lane0FollowerList, lane1FollowerList), (lane0SuccessorList, lane1successorList)
    
    def sortLaneMap(self, laneMap):
        for lane in range(0, LANES):
            laneMap[lane] = np.sort(laneMap[lane], order=['pos'])

    def step(self, laneMap):
        self.sortLaneMap(laneMap)
        followerList, sucessorList = self.oldAndNewFollower(laneMap)
        return followerList, sucessorList
