import numpy as np
import random
from v2i.src.core.constants import LANES, MOBIL_CONST, LANE_RADIUS, SCALE, CAR_LENGTH, MOBIL_CONST
from itertools import cycle
from v2i.src.core.common import mergeDicts, buildDictWithKeys, raiseValueError
from v2i.src.core.utils import idmAcc

class mobil:

    def __init__(self, period, idmHandler):
        self.politeness = MOBIL_CONST["POLITENESS"]
        self.period = period
        self.idmHandler = idmHandler
    
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

    def findFollower(self, vehicleLane, targetLane, vehicleLaneMap, targetLaneVehicleMap, vehID):
        vehicleIndex = np.where(vehicleLaneMap['id'] == vehID)[0][0]
        vehicle = vehicleLaneMap[vehicleIndex]
        tmpTargetLaneMap = np.append(targetLaneVehicleMap, vehicle)
        tmpTargetLaneMap = np.sort(tmpTargetLaneMap, order=['pos'])
        vehID = np.where(tmpTargetLaneMap['pos'] == vehicle['pos'])[0][0]
        index = (vehID - 1) % len(tmpTargetLaneMap)
        return tmpTargetLaneMap[index]['id']
    
    def findForward(self, vehicleLane, targetLane, vehicleLaneMap, targetLaneVehicleMap, vehID):
        vehicleIndex = np.where(vehicleLaneMap['id'] == vehID)[0][0]
        vehicle = vehicleLaneMap[vehicleIndex]
        tmpTargetLaneMap = np.append(targetLaneVehicleMap, vehicle)
        tmpTargetLaneMap = np.sort(tmpTargetLaneMap, order=['pos'])
        vehID = np.where(tmpTargetLaneMap['pos'] == vehicle['pos'])[0][0]
        index = (vehID + 1) % len(tmpTargetLaneMap)
        return tmpTargetLaneMap[index]['id']
    
    
    def searchLane(self, laneMap, vehID):
        out = np.where(laneMap[0]['id'] == vehID)[0]
        if len(out) > 0:
            return 0
        out = np.where(laneMap[1]['id'] == vehID)[0]
        if len(out) > 0:
            return 1
        raiseValueError("vehicle with id:%d not found in any lane"%(vehID))
    
    def arcLength(self, deg, lane):
        length = np.deg2rad(deg) * LANE_RADIUS[lane]
        lengthInMetre = (1.0/SCALE) * length
        return lengthInMetre
    
    def bumper2bumperDist(self, diff, carLength, lane):
        carGapInMetre = self.arcLength(diff, lane)
        return carGapInMetre - CAR_LENGTH
    
    def calAcc(self, dist, period, speed):
        return 2 * ((dist - (speed * period))/(period * period))
    
    def getFollowerAcc(self, laneMap, vehLane, otherLane, vehID, otherID):
        vehIndex = np.where(laneMap[vehLane]['id'] == vehID)[0][0]
        otherIndex = np.where(laneMap[otherLane]['id'] == otherID)[0][0]
    
        vehPos = laneMap[vehLane][vehIndex]['pos']
        vehSpeed = laneMap[vehLane][vehIndex]['speed']
        otherPos = laneMap[otherLane][otherIndex]['pos']
        otherSpeed = laneMap[otherLane][otherIndex]['speed']

        distDeg = vehPos - otherPos
        distDeg %= 360
        sAlpha = self.bumper2bumperDist(distDeg, CAR_LENGTH, otherLane)
        speedDiff = otherSpeed - vehSpeed
        if sAlpha <= 0.0:
            acc = MOBIL_CONST['B_SAFE'] - 1
        else:
            acc = self.idmHandler.idmAcc(sAlpha, speedDiff, otherSpeed)
        return acc
    
    def getFrontAcc(self, laneMap, vehLane, otherLane, vehID, otherID):
        vehIndex = np.where(laneMap[vehLane]['id'] == vehID)[0][0]
        otherIndex = np.where(laneMap[otherLane]['id'] == otherID)[0][0]
    
        vehPos = laneMap[vehLane][vehIndex]['pos']
        vehSpeed = laneMap[vehLane][vehIndex]['speed']
        otherPos = laneMap[otherLane][otherIndex]['pos']
        otherSpeed = laneMap[otherLane][otherIndex]['speed']

        distDeg = otherPos - vehPos
        distDeg %= 360
        sAlpha = self.bumper2bumperDist(distDeg, CAR_LENGTH, otherLane)
        speedDiff = vehSpeed - otherSpeed
        if sAlpha <= 0.0:
            acc = MOBIL_CONST['B_SAFE'] - 1
        else:
            acc = self.idmHandler.idmAcc(sAlpha, speedDiff, otherSpeed)
        return acc
    
    def laneChangeRules(self, newVehIDAccAfterLaneChange, oldVehIDAccBeforeLaneChange, newOtherIDAccAfterLaneChange):
        vehAccDiff = newVehIDAccAfterLaneChange - oldVehIDAccBeforeLaneChange
    
        if newOtherIDAccAfterLaneChange < MOBIL_CONST["B_SAFE"]:
            return False
        
        if (newVehIDAccAfterLaneChange - oldVehIDAccBeforeLaneChange) < MOBIL_CONST['GAIN']:
            return False
        
        if np.random.rand() <= MOBIL_CONST['RANDOMIZE_PROB']:
            return True
        else:
            return False
        
    def getTargetLane(self, lane):
        if lane == 0:
            return 1
        else:
            return 0
    
    def step(self, laneMap, idmAccs):
        keys = list(idmAccs.keys())
        random.shuffle(keys)
        visited = buildDictWithKeys(keys, False)
        assert len(visited) == laneMap[0].shape[0] + laneMap[1].shape[0]
        
        followerList = {}
        frontList = {}

        for vehID in visited.keys():
            vehLane = self.searchLane(laneMap, vehID)
            otherLane = self.getTargetLane(vehLane)
            assert otherLane != vehLane

            frontID = self.findForward(vehLane, otherLane, laneMap[vehLane], laneMap[otherLane], vehID)
            followerID = self.findFollower(vehLane, otherLane, laneMap[vehLane], laneMap[otherLane], vehID)

            followerList[vehID] = followerID
            frontList[vehID] = frontID

            # Find accleration of vehicles required to check for valid lane change
            followerAcc = self.getFollowerAcc(laneMap, vehLane, otherLane, vehID, followerID) # This is the accleration of the follower vehicle if vehID switches lane
            frontAcc = self.getFrontAcc(laneMap, vehLane, otherLane, vehID, frontID) # This is acceleration of vehID if it switches lane
            
            res = self.laneChangeRules(frontAcc, idmAccs[vehID], followerAcc)

            vehIndex = np.where(laneMap[vehLane]['id'] == vehID)[0][0]
            isEgoVehicle = laneMap[vehLane][vehIndex]['agent']

            if res and isEgoVehicle == 0:
                #print("Curr Lane : ", vehLane, "Other Lane : ", otherLane)
                laneMap[otherLane] = np.append(laneMap[otherLane], laneMap[vehLane][vehIndex])
                laneMap[otherLane][-1]['lane'] = otherLane
                laneMap[vehLane] = np.delete(laneMap[vehLane], vehIndex)
                
                # Sort the both lanes
                self.sortLaneMap(laneMap)
        
        assert len(followerList) == len(idmAccs)
        assert len(frontList) == len(idmAccs)

        return followerList, frontList, laneMap

