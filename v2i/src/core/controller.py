import numpy as np

from v2i.src.core.constants import IDM_CONSTS, LANE_RADIUS, SCALE, LANES, CAR_LENGTH
from v2i.src.core.common import getAgentID, arcAngle, raiseValueError

class egoController:

    def __init__(self, tPeriod, maxSpeed):
        # Initialize Action Space
        self.initActionSpace()

        # Initialize variables passed to the class
        self.initParams(tPeriod, maxSpeed)

        # Initialize the function to calculate min Angle required to avoid any collision
        self.initCollisionAngle()

        # Intialize the function to calculate the angle required to calculate lane change limits
        self.initLaneChangeAngle()

    def initActionSpace(self):
        self.possibleActions = ['acc', 'dec', 'do-nothing', 'lane-change']
        self.planMap = self.buildPlanningActionMap(self.possibleActions)

    def planSpace(self):
        return self.possibleActions
    
    def initLaneChangeAngle(self):
        self.laneChangeAngle = []
        for lane in range(0, LANES):
            self.laneChangeAngle.append(arcAngle(LANE_RADIUS[lane], (CAR_LENGTH) * SCALE))
    
    def initCollisionAngle(self):
        '''
        Function : Calculates the minimum angle required between two cars for no collision.
        '''
        self.minCollisionAngle = []
        for lane in range(0, LANES):
            self.minCollisionAngle.append(arcAngle(LANE_RADIUS[lane], CAR_LENGTH * SCALE))
    
    def buildPlanningActionMap(self, possibleActs):
        planMap = {}
        for idx, element in enumerate(possibleActs):
            planMap[idx] = element
        return planMap

    def initParams(self, tPeriod, maxSpeed):
        self.tPeriod = tPeriod
        self.maxSpeed = maxSpeed
    
    def distTravelled(self, acc, tPeriod, speed):
        return (speed*tPeriod) + (0.5 * acc * tPeriod * tPeriod)
    
    def newSpeed(self, acc, tPeriod, speed):
        return speed + (acc * tPeriod)
    
    def checkValidAction(self, agentLane, laneMap, agentPos, agentIDX):
        if(agentIDX < (laneMap[agentLane].shape[0]-1)):
            angleDiff = laneMap[agentLane][agentIDX+1]['pos'] - agentPos
        else:
            angleDiff = laneMap[agentLane][0]['pos'] - agentPos
            angleDiff = angleDiff % 360
        return angleDiff > self.minCollisionAngle[agentLane]
    
    def performAccelerate(self, laneMap, agentLane, agentIDX):
        agentSpeed = laneMap[agentLane][agentIDX]['speed']
        agentPos = laneMap[agentLane][agentIDX]['pos']
        acc = IDM_CONSTS["MAX_ACC"]
        if agentSpeed >= self.maxSpeed:
            acc = 0.0

        distTravelledInMetres = self.distTravelled(acc, self.tPeriod, agentSpeed)
        distTravelledInDeg = arcAngle(LANE_RADIUS[agentLane], distTravelledInMetres * SCALE)
        distTravelledInDeg %= 360
        newAgentSpeed = self.newSpeed(acc, self.tPeriod, agentSpeed)
        newAgentSpeed = np.clip(newAgentSpeed, 0, self.maxSpeed)
        '''
        Only agent new pos is required to check for collision
        '''
        collision = False
        if not self.checkValidAction(agentLane, laneMap, agentPos + distTravelledInDeg, agentIDX):
            distTravelledInDeg = 0.0
            newAgentSpeed = 0.0
            collision = True
        return distTravelledInDeg, newAgentSpeed, collision, agentLane
    

    def performDecelerate(self, laneMap, agentLane, agentIDX):
        agentSpeed = laneMap[agentLane][agentIDX]['speed']
        agentPos = laneMap[agentLane][agentIDX]['pos']
        dec = -IDM_CONSTS["DECELERATION_RATE"]
        if agentSpeed <= 0.0:
            dec = 0.0

        distTravelledInMetres = self.distTravelled(dec, self.tPeriod, agentSpeed)
        distTravelledInDeg = arcAngle(LANE_RADIUS[agentLane], distTravelledInMetres * SCALE)
        distTravelledInDeg %= 360
        newAgentSpeed = self.newSpeed(dec, self.tPeriod,agentSpeed)
        newAgentSpeed = np.clip(newAgentSpeed, 0, self.maxSpeed)
        '''
        Only agent new pos is required to check for collision
        '''
        collision = False
        if not self.checkValidAction(agentLane, laneMap, agentPos + distTravelledInDeg, agentIDX):
            distTravelledInDeg = 0.0
            newAgentSpeed = 0.0
            collision = True
        return distTravelledInDeg, newAgentSpeed, collision, agentLane
    
    def performDoNothing(self, laneMap, agentLane, agentIDX):
        agentSpeed = laneMap[agentLane][agentIDX]['speed']
        agentPos = laneMap[agentLane][agentIDX]['pos']
        
        distTravelledInMetres = self.distTravelled(0, self.tPeriod, agentSpeed)
        distTravelledInDeg = arcAngle(LANE_RADIUS[agentLane], distTravelledInMetres * SCALE)
        distTravelledInDeg %= 360
        newAgentSpeed = agentSpeed
        newAgentSpeed = np.clip(newAgentSpeed, 0, self.maxSpeed)
        '''
        Only agent new pos is required to check for collision
        '''
        collision = False
        if not self.checkValidAction(agentLane, laneMap, agentPos + distTravelledInDeg, agentIDX):
            distTravelledInDeg = 0.0
            newAgentSpeed = 0.0
            collision = True
        return distTravelledInDeg, newAgentSpeed, collision, agentLane
    
    def performLaneChange(self, laneMap, agentLane, agentIDX):
        '''
        Function : Checks for valid lane change operation and return pos, speed, lane change is valid or not.
        Criteria :
            1. If agent speed is greater than zero, only then lane change can be done.
            2. lane change just changes the lane of the agent if it is valid and keeps it speed and postion unchanged.
        '''
        
        agentSpeed = laneMap[agentLane][agentIDX]['speed']
        agentPos = laneMap[agentLane][agentIDX]['pos']

        #---- Identify the neighbouring lane ----#
        if agentLane == 0:
            laneToChange = 1
        else:
            laneToChange = 0
        #---- Identify the neighbouring lane ----#
        distTravelledInDeg = 0.0
        newAgentSpeed = agentSpeed
        newAgentSpeed = np.clip(newAgentSpeed, 0, self.maxSpeed)
        collision = False

        if agentSpeed == 0.0:
            distTravelledInDeg = 0.0
            newAgentSpeed = 0.0
            collision = True
            return distTravelledInDeg, newAgentSpeed, collision, laneToChange
        
        if not self.checkValidLaneChange(laneToChange, agentLane, laneMap, agentPos):
            distTravelledInDeg = 0.0
            newAgentSpeed = 0.0
            collision = True
        return distTravelledInDeg, newAgentSpeed, collision, laneToChange
    
    def checkValidLaneChange(self, laneToChange, agentLane, laneMap, agentPos):
        egoLowLimit = (agentPos - self.laneChangeAngle[laneToChange]) % 360
        egoHighLimit = (agentPos + self.laneChangeAngle[laneToChange]) % 360
        for idx in range(0, laneMap[laneToChange].shape[0]):
            nonEgoLowLimit = (laneMap[laneToChange][idx]['pos'] - self.laneChangeAngle[laneToChange]) % 360
            nonEgoHighLimit = (laneMap[laneToChange][idx]['pos'] + self.laneChangeAngle[laneToChange]) % 360
            
            if egoHighLimit < egoLowLimit:
                if (nonEgoHighLimit > egoLowLimit and nonEgoLowLimit < nonEgoHighLimit):
                    return False
                if nonEgoHighLimit < egoHighLimit:
                    return False
                if nonEgoLowLimit > egoLowLimit and nonEgoLowLimit < 360 and nonEgoHighLimit > egoHighLimit and nonEgoHighLimit > 0:
                    return False
            else:
                if (egoLowLimit > nonEgoLowLimit and egoLowLimit < nonEgoHighLimit):
                    return False
                if (egoHighLimit < nonEgoHighLimit and egoHighLimit > nonEgoLowLimit):
                    return False
        return True
    
    def executeAction(self, action, laneMap, agentLane):
        agentIDX = getAgentID(laneMap, agentLane)
        distTravelledInDeg = 0
        newSpeed = 0.0

        if action == "acc":
            distTravelledInDeg, newSpeed, collision, laneToChange = self.performAccelerate(laneMap, agentLane, agentIDX)
        elif action == "dec":
            distTravelledInDeg, newSpeed, collision, laneToChange = self.performDecelerate(laneMap, agentLane, agentIDX)
        elif action == "do-nothing":
            distTravelledInDeg, newSpeed, collision, laneToChange = self.performDoNothing(laneMap, agentLane, agentIDX)
        elif action == "lane-change":
            distTravelledInDeg, newSpeed, collision, laneToChange = self.performLaneChange(laneMap, agentLane, agentIDX)
        else:
            raiseValueError("%s is not a valid action !"%(action))

        return distTravelledInDeg, newSpeed, collision, laneToChange