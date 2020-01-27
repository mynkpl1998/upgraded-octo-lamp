import numpy as np

from v2i.src.core.constants import IDM_CONSTS, SCALE, LANE_RADIUS, CAR_LENGTH, LANES

class idm:

    def __init__(self, maxVel, tPeriod, viewRange):
        self.maxVel = maxVel
        self.tPeriod = tPeriod
        self.viewRange = viewRange
        self.nonEgoMaxVel = self.maxSpeed(self.viewRange, IDM_CONSTS['DECELERATION_RATE'])

        #---- Vectoized Functions ----#
        self.vecBumpBumpDistance = np.vectorize(self.BumpBumpDist)
        self.vecidmAcc = np.vectorize(self.idmAcc)
        self.vecDistTravelled = np.vectorize(self.distTravelled)
        self.vecNewSpeed = np.vectorize(self.newSpeed)
        self.vecArc2Angle = np.vectorize(self.arc2angle)
        self.vecNewPos = np.vectorize(self.newPos)
        #---- Vectoized Functions ----#

    def getAllElementbyKeys(self, key, array):
        a = np.zeros(array.shape)
        for i in range(0, array.shape[0]):
            a[i] = array[i][key]
        return a
    
    def angleDiff(self, laneMap, lane):
        a = self.getAllElementbyKeys('pos', laneMap[lane])
        b = np.zeros(a.shape)
        b[0:-1] = a[1:]
        b[-1] = a[0]
        diff = b-a
        return diff % 360
    
    def relativeSpeed(self, laneMap, lane):
        a = self.getAllElementbyKeys('speed', laneMap[lane])
        b = np.zeros(a.shape)
        b[0:-1] = a[1:]
        b[-1] = a[0]
        diff = a-b
        return diff
    
    def arcLength(self, radius, angleDeg):
        length = np.deg2rad(angleDeg) * radius
        lengthInMetre = (1.0/SCALE) * length
        return lengthInMetre

    def BumpBumpDist(self, diff, carLength, lane):
        capGapInMetre = self.arcLength(LANE_RADIUS[lane], diff)
        return capGapInMetre - CAR_LENGTH
    
    def maxSpeed(self, distanceInMetre, decelerationRate):
        return np.sqrt(2 * distanceInMetre * decelerationRate)

    
    def idmAcc(self, sAlpha, speedDiff, speed, maxSpeed):
        '''
        Make sure to edit function at utils.py if you change anything here
        IDM Equation : https://en.wikipedia.org/wiki/Intelligent_driver_model
        
        Modified IDM Equation includes the sense of local view to non-Ego vehicles
        Modified IDM Details : https://github.com/mynkpl1998/single-ring-road-with-light/blob/master/SingleLaneIDM/Results%20Analysis.ipynb
        '''
        sStar = IDM_CONSTS['MIN_SPACING'] + (speed * IDM_CONSTS['HEADWAY_TIME']) + ((speed * speedDiff)/(2 * np.sqrt(IDM_CONSTS['MAX_ACC'] * IDM_CONSTS['DECELERATION_RATE'])))
        acc = IDM_CONSTS['MAX_ACC'] * (1 - ((speed / maxSpeed)**IDM_CONSTS['DELTA']) - ((sStar/sAlpha)**2))
        return acc
    
    def distTravelled(self, speed, acc, tPeriod):
        dist = ((speed * tPeriod) + (0.5 * acc * tPeriod * tPeriod))
        if dist > 3:
            print(dist)
        
        return dist

    def newSpeed(self, speed, acc, tPeriod):
        v = speed + (acc * tPeriod)
        return v
    
    def arc2angle(self, radius, arcLen):
        return np.rad2deg(arcLen/radius)
    
    def newPos(self, oldPos, angleDiff):
        return (oldPos + angleDiff)% 360
    
    def updateLaneMap(self, speed, pos, laneMap, acc, planAct):
        for i in range(laneMap.shape[0]):
            laneMap[i]['pos'] = pos[i]
            laneMap[i]['speed'] = speed[i]
            laneMap[i]['acc'] = acc[i]
    
    '''
    def updateLaneMap(self, speed, pos, laneMap, acc, planAct):
        for i in range(laneMap.shape[0]):
            if laneMap[i]['agent'] == 1:
                if planAct == "acc":
                    laneMap[i]['acc'] = IDM_CONSTS['MAX_ACC']
                elif planAct == "dec":
                    laneMap[i]['acc'] = -IDM_CONSTS['DECELERATION_RATE']
                else:
                    laneMap[i]['acc'] = 0.0
            else:
                laneMap[i]['pos'] = pos[i]
                laneMap[i]['speed'] = speed[i]
                laneMap[i]['acc'] = acc[i]
    '''
    def sortLaneMap(self, laneMap):
        for lane in range(0, LANES):
            laneMap[lane] = np.sort(laneMap[lane], order=['pos'])

    def step(self, laneMap, planAct):
        self.sortLaneMap(laneMap)
        '''
        Each lane has different number of cars. Hence, we need to do seperate function calls on each of them
        '''
        if laneMap[0].shape[0] == 0:
            pass
        else:
            oldPosLane0 = self.getAllElementbyKeys('pos', laneMap[0])
            carsMaxSpeeds = self.getAllElementbyKeys('max-speed', laneMap[0])
            posDiffLane0 = self.angleDiff(laneMap, 0)
            speedDiffLane0 = self.relativeSpeed(laneMap, 0)
            speedLane0 = self.getAllElementbyKeys('speed', laneMap[0])
            sAlphaLane0 = self.vecBumpBumpDistance(posDiffLane0, CAR_LENGTH, 0)
            accLane0 = self.vecidmAcc(sAlphaLane0, speedDiffLane0, speedLane0, carsMaxSpeeds)
            distLane0 = self.vecDistTravelled(speedLane0, accLane0, self.tPeriod)
            newSpeedLane0 = self.vecNewSpeed(speedLane0, accLane0, self.tPeriod)
            distLane0[distLane0 < 0] = 0.0
            newSpeedLane0[newSpeedLane0 < 0] = 0.0
            distInPixelsLane0 = distLane0 * SCALE
            distInDegLane0 = self.vecArc2Angle(LANE_RADIUS[0], distInPixelsLane0)
            '''
            for idx, car in enumerate(laneMap[0]):
                if car['agent'] == 1:
                    distInDegLane0[idx] = 0.0
            '''
            carIDsLane0 = self.getAllElementbyKeys('id', laneMap[0])
            newPosLane0 = self.vecNewPos(oldPosLane0, distInDegLane0)
            self.updateLaneMap(newSpeedLane0, newPosLane0, laneMap[0], accLane0, planAct)


        if laneMap[1].shape[0] == 0:
            pass
        else:
            oldPosLane1 = self.getAllElementbyKeys('pos', laneMap[1])
            carsMaxSpeeds = self.getAllElementbyKeys('max-speed', laneMap[1])
            posDiffLane1 = self.angleDiff(laneMap, 1)
            speedDiffLane1 = self.relativeSpeed(laneMap, 1)
            speedLane1 = self.getAllElementbyKeys('speed', laneMap[1])
            sAlphaLane1 = self.vecBumpBumpDistance(posDiffLane1, CAR_LENGTH, 1)
            accLane1 = self.vecidmAcc(sAlphaLane1, speedDiffLane1, speedLane1, carsMaxSpeeds)
            distLane1 = self.vecDistTravelled(speedLane1, accLane1, self.tPeriod)
            newSpeedLane1 = self.vecNewSpeed(speedLane1, accLane1, self.tPeriod)
            # --- Check for invalid distance and new Speed --- #
            distLane1[distLane1 < 0] = 0.0
            newSpeedLane1[newSpeedLane1 < 0] = 0.0
            # --- Check for invalid distance and new Speed --- #
            distInPixelsLane1 = distLane1 * SCALE
            distInDegLane1 = self.vecArc2Angle(LANE_RADIUS[1], distInPixelsLane1)
            '''
            for idx, car in enumerate(laneMap[1]):
                if car['agent'] == 1:
                    distInDegLane1[idx] = 0.0
            '''
            carIDsLane1 = self.getAllElementbyKeys('id', laneMap[1])
            newPosLane1 = self.vecNewPos(oldPosLane1, distInDegLane1)
            self.updateLaneMap(newSpeedLane1, newPosLane1, laneMap[1], accLane1, planAct)
        
        return (carIDsLane0.copy(), carIDsLane1.copy(), distInDegLane0.copy(), distInDegLane1.copy())
        
