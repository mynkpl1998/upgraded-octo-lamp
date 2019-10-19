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
    
    def otherLaneFollower(self, targetlane, otherLane, currLaneNum, otherLaneNum):
        s = {}
        for vehicle in targetlane:
            tmpNewLane = np.append(otherLane, (vehicle))
            #tmpNewLane[-1]['lane'] = otherLaneNum
            tmpNewLane = np.sort(tmpNewLane, order=['pos'])
            vehID = np.where(tmpNewLane['pos'] == vehicle['pos'])[0]
            index =  (vehID[0] - 1) % len(tmpNewLane)
            s[(vehicle['id'], currLaneNum)] = tmpNewLane[index]['id']
        return s

    def oldAndNewFollower(self, laneMap):
        lane0List = self.buildlist(0, laneMap, 'id')
        lane1List = self.buildlist(1, laneMap, 'id')
        lane0FollowerList = self.followerList(lane0List, 0)
        lane1FollowerList = self.followerList(lane1List, 1)
        lane0otherLaneFollowerList = self.otherLaneFollower(laneMap[0], laneMap[1], 0, 1)
        lane1otherLaneFollowerList = self.otherLaneFollower(laneMap[1], laneMap[0], 1, 0)

        return (lane0FollowerList, lane1FollowerList), (lane0otherLaneFollowerList, lane1otherLaneFollowerList)
    
    def sortLaneMap(self, laneMap):
        for lane in range(0, LANES):
            laneMap[lane] = np.sort(laneMap[lane], order=['pos'])
    
    def execLaneChange(self, currLane, targetLane, currLaneMap, targetLaneMap, idmHandler, planAct, oldAccs, followerList, otherLaneFollowerList):
        laneRes = {}
        for idx, vehicle in enumerate(currLaneMap):
            # --- target lane --- #
            newTmpLaneMap = np.append(targetLaneMap, vehicle)
            newTmpLaneMap = np.sort(newTmpLaneMap, order=['pos'])
            vehIndex = np.where(newTmpLaneMap['id'] == vehicle['id'])[0][0]
            newTmpLaneMap[vehIndex]['lane'] = targetLane

            # --- current lane --- #
            newCurrLane = np.delete(currLaneMap, idx)
            newCurrLane = np.sort(newCurrLane, order=['pos'])
            
            newLaneMap = np.array([newCurrLane, newTmpLaneMap])
            idmHandler.step(newLaneMap, planAct)
            newAccs = self.getVehiclesAccels(newLaneMap)
            
            newFollowerAcc = newAccs[followerList[(vehicle['id'], vehicle['lane'])]]
            newOtherFollowerAcc = newAccs[otherLaneFollowerList[(vehicle['id'], vehicle['lane'])]]
            newVehicleAcc = newAccs[vehicle['id']]
            
            oldFollowerAcc = oldAccs[followerList[(vehicle['id'], vehicle['lane'])]]
            oldOtherFollowerAcc = oldAccs[otherLaneFollowerList[(vehicle['id'], vehicle['lane'])]]
            oldVehicleAcc = oldAccs[vehicle['id']]

            res = self.checkValidLaneChange(oldVehicleAcc, oldOtherFollowerAcc, oldFollowerAcc, newVehicleAcc, newOtherFollowerAcc, newFollowerAcc)
            laneRes[vehicle['id']] = res
        return laneRes
    

    def checkValidLaneChange(self, oldVehicleAcc, oldOtherFollowerAcc, oldFollowerAcc, newVehicleAcc, newOtherFollowerAcc, newFollowerAcc):
        
        if (newOtherFollowerAcc - oldOtherFollowerAcc) >= -4.0:
            pass
        else:
            return False
        
        if (newVehicleAcc - oldVehicleAcc) + 0.1*((newFollowerAcc - oldFollowerAcc) + (newOtherFollowerAcc - oldOtherFollowerAcc)) > 0.1:
            return True
        else:
            return False

    def getVehiclesAccels(self, laneMap):
        accs = {} # id - acc
        for lane in range(0, LANES):
            for vehicle in laneMap[lane]:
                accs[vehicle['id']] = vehicle['acc']
        return accs
    
    def list2Dict(self, l):
        d = {}
        for lane in range(0, LANES):
            for key in l[lane].keys():
                d[key] = l[lane][key]
        return d
    
    def merge2Dict(self, l1, l2):
        d = {}
        for key in l1.keys():
            d[key] = l1[key]
        
        for key in l2.keys():
            d[key] = l2[key]
        return d


    def step(self, laneMap, idmHandler, planAct):
        self.sortLaneMap(laneMap)
        followerList, otherLaneFollowerList = self.oldAndNewFollower(laneMap)
        oldAccs = self.getVehiclesAccels(laneMap)
        lanechangeres0 = self.execLaneChange(0, 1, laneMap[0], laneMap[1], idmHandler, planAct, oldAccs, self.list2Dict(followerList) , self.list2Dict(otherLaneFollowerList))
        lanechangeres1 = self.execLaneChange(1, 0, laneMap[1], laneMap[0], idmHandler, planAct, oldAccs, self.list2Dict(followerList), self.list2Dict(otherLaneFollowerList))
        res = self.merge2Dict(lanechangeres0, lanechangeres1)
        return followerList, otherLaneFollowerList, res
    
    def exec(self, laneMap, res, laneNum): 
        newData = {}
        newData[0] = []
        newData[1] = []

        for vehicle in laneMap:
            print(vehicle['id'])


