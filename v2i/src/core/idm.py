import numpy as np

from v2i.src.core.constants import IDM_CONSTS, SCALE, LANE_RADIUS, CAR_LENGTH

class idm:

    def __init__(self, maxVel, tPeriod):
        self.maxVel = maxVel
        self.tPeriod = tPeriod

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
    
    def angleDiff(self, laneMap):
        a = self.getAllElementbyKeys('pos', laneMap[0])
        b = np.zeros(a.shape)
        b[0:-1] = a[1:]
        b[-1] = a[0]
        diff = b-a
        return diff % 360
    
    def relativeSpeed(self, laneMap):
        a = self.getAllElementbyKeys('speed', laneMap[0])
        b = np.zeros(a.shape)
        b[0:-1] = a[1:]
        b[-1] = a[0]
        diff = a-b
        return diff
    
    def arcLength(self, radius, angleDeg):
        length = np.deg2rad(angleDeg) * radius
        lengthInMetre = (1.0/SCALE) * length
        return lengthInMetre

    def BumpBumpDist(self, diff, carLength):
        capGapInMetre = self.arcLength(LANE_RADIUS, diff)
        return capGapInMetre - CAR_LENGTH
    
    def idmAcc(self, sAlpha, speedDiff, speed):
        '''
        IDM Equation : https://en.wikipedia.org/wiki/Intelligent_driver_model
        '''
        sStar = IDM_CONSTS['MIN_SPACING'] + (speed * IDM_CONSTS['HEADWAY_TIME']) + ((speed * speedDiff)/(2 * np.sqrt(IDM_CONSTS['MAX_ACC'] * IDM_CONSTS['DECELERATION_RATE'])))
        acc = IDM_CONSTS['MAX_ACC'] * (1 - ((speed / self.maxVel)**IDM_CONSTS['DELTA']) - ((sStar/sAlpha)**2))
        return acc
    
    def distTravelled(self, speed, acc, tPeriod):
        dist = ((speed * tPeriod) + (0.5 * acc * tPeriod * tPeriod))
        return dist

    def newSpeed(self, speed, acc, tPeriod):
        v = speed + (acc * tPeriod)
        return v
    
    def arc2angle(self, radius, arcLen):
        return np.rad2deg(arcLen/radius)
    
    def newPos(self, oldPos, angleDiff):
        return (oldPos + angleDiff)% 360
    
    def updateLaneMap(self, speed, pos, laneMap):
        for i in range(laneMap.shape[0]):
            laneMap[i]['pos'] = pos[i]
            laneMap[i]['speed'] = speed[i]
     
    def step(self, laneMap):
        laneMap[0] = np.sort(laneMap[0], order=['pos'])
        oldPos = self.getAllElementbyKeys('pos', laneMap[0])
        posDiff = self.angleDiff(laneMap)
        speedDiff = self.relativeSpeed(laneMap)
        speed = self.getAllElementbyKeys('speed', laneMap[0])
        sAlpha = self.vecBumpBumpDistance(posDiff, CAR_LENGTH)
        acc = self.vecidmAcc(sAlpha, speedDiff, speed)
        dist = self.vecDistTravelled(speed, acc, self.tPeriod)
        newSpeed = self.vecNewSpeed(speed, acc, self.tPeriod)

        # --- Check for invalid distance and new Speed --- #
        dist[dist < 0] = 0.0
        newSpeed[newSpeed < 0] = 0.0
        # --- Check for invalid distance and new Speed --- #
        
        distInPixels = dist * SCALE
        distInDeg = self.vecArc2Angle(LANE_RADIUS, distInPixels)
        newPos = self.vecNewPos(oldPos, distInDeg)
        self.updateLaneMap(newSpeed, newPos, laneMap[0])
