import numpy as np

from v2i.src.core.constants import MAX_CARS_IN_LANE, LANES

class router:

    def __init__(self, removeAngle, addAngle):
        self.removeAngle = removeAngle
        self.addAngle = addAngle
    
    def laneMap2Dict(self, laneMap):
        d = {}
        for lane in range(0, LANES):
            for vehicle in laneMap[lane]:
                vehID = vehicle['id']
                d[vehID] = vehicle
        return d
    
    def step(self, currLaneMap):
        currLaneMapDict = self.laneMap2Dict(currLaneMap)

        for vehID in currLaneMapDict.keys():

            curPos = currLaneMapDict[vehID]['pos']

            currLane = currLaneMapDict[vehID]['lane']
            agentType = currLaneMapDict[vehID]['agent']
            
            if curPos >= self.removeAngle - 1.0 and curPos <= self.removeAngle + 1.0 and len(currLaneMap[currLane]) > 2 and agentType!=1:
                if np.random.rand() <= 0.5:
                    currLane = currLaneMapDict[vehID]['lane']
                    currIndex = np.where(currLaneMap[currLane]['id'] == vehID)[0][0]
                    currLaneMap[currLane] =  np.delete(currLaneMap[currLane], currIndex)
                
                '''
                randomLane = np.random.randint(0, 2)
                if np.random.rand() <= 0.5 and len(currLaneMap[randomLane]) < MAX_CARS_IN_LANE:
                    newIdlane0 = max(currLaneMap[0]['id'])
                    newIdlane1 = max(currLaneMap[1]['id'])
                    tmpVehicle = currLaneMapDict[vehID]
                    tmpVehicle['lane'] = randomLane
                    tmpVehicle['pos'] = self.addAngle
                    tmpVehicle['id'] = max(newIdlane0, newIdlane1) + 1
                    currLaneMap[randomLane] = np.append(currLaneMap[randomLane], tmpVehicle)
                '''
        return currLaneMap
