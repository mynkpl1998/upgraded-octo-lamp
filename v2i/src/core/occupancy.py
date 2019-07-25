import math
import numpy as np

from v2i.src.core.utils import raiseValueError
from v2i.src.core.constants import CAR_LENGTH, LANE_RADIUS, SCALE, LANES
from v2i.src.core.common import getAgentID, arcAngle, arcLength

class Grid:

    def __init__(self, localView, extendedView=None, cellSize=1):
        self.localView = localView # in metre
        self.cellSize = cellSize # in metre
        self.extendedView = extendedView

        #---- Checks ----#
        if self.extendedView == None:
            self.extendedView = self.localView
        
        if self.extendedView < self.localView:
            raiseValueError("size of extended view should be greater than local view")
        #---- Checks ----#

        #---- Calculate Communicable Region ----#
        self.commView = self.extendedView - self.localView # in metre
        #---- Calculate Communicable Region ----#

        if self.extendedView % self.cellSize != 0:
            raiseValueError("communication should be completely divisible by cellsize")
        
        if self.cellSize > CAR_LENGTH:
            raiseValueError("Cell-Size (%.2f m) should be smaller than Car Length (%.2f m) "%(self.cellSize, CAR_LENGTH))

        #---- Calculate the number of columns ----#
        self.numCols = int(self.extendedView / self.cellSize)
        #---- Calculate the number of columns ----#

        self.init()
    

    def init(self):
        '''
        All common initialization goes here.
        '''
        self.halfExtendedViewInAngle = []
        self.shift = []
        for lane in range(0, LANES):
            self.halfExtendedViewInAngle.append(arcAngle(LANE_RADIUS[lane], (self.extendedView/2) * SCALE))
            self.shift.append(arcAngle(LANE_RADIUS[lane], (CAR_LENGTH/2) * SCALE))
        
    def getOccupancyGrid(self, laneMap, agentLane):
        occGrid = np.zeros((2, self.numCols))
        velGrid = np.zeros((2, self.numCols))
        
        for lane in range(0, LANES):
            laneMap[lane] = np.sort(laneMap[lane], order=['pos'])
        agentID = getAgentID(laneMap, agentLane)
        agentPos = laneMap[agentLane][agentID]['pos']
        
        for lane in range(0, LANES):
            forward_done = False
            _next_ = None

            for j in range(0, laneMap[lane].shape[0]):
                if(laneMap[lane][j]['pos'] - self.shift[lane] > agentPos):
                    _next_ = j
                    break

            if(_next_ == None):
                _next_ = 0
            
            count_forward = 0
            copy_next = _next_

            for x in range(0, laneMap[lane].shape[0]):
                next_car_angle = laneMap[lane][copy_next]['pos']
                if(next_car_angle <= agentPos):
                    next_car_angle += 360
                
                angleDiff = next_car_angle - agentPos - self.shift[lane]
                angleDiff = angleDiff % 360

                if(angleDiff < self.halfExtendedViewInAngle[lane]):
                    count_forward += 1
                    copy_next += 1
                    if(copy_next > (laneMap[lane].shape[0] - 1)):
                        copy_next = 0
                else:
                    break
            
            for x in range(0, count_forward):
                next_car_angle = laneMap[lane][_next_]['pos']
                next_car_speed = laneMap[lane][_next_]['speed']

                if(next_car_angle < agentPos):
                    next_car_angle += 360
                
                angleDiff = next_car_angle - agentPos - self.shift[lane]

                if (angleDiff <= self.halfExtendedViewInAngle[lane]):
                    dist = arcLength(LANE_RADIUS[lane], angleDiff)
                    distInMetres = dist / SCALE
                    index = distInMetres/float(self.cellSize)

                    halfLen = int(occGrid[0].shape[0]/2)
                    numIndexs = math.ceil((CAR_LENGTH * 0.5) / self.cellSize)
                    occGrid[lane][halfLen + int(index)] = 1
                    velGrid[lane][halfLen + int(index)] = next_car_speed
                    if((_next_ + 1) > (laneMap[lane].shape[0]- 1)):
                        _next_ = 0
                    else:
                        _next_ += 1
        
        numIndexesOthers = math.ceil(CAR_LENGTH/self.cellSize)
        half = int(occGrid[0].shape[0]/2) - 1
        tmpCopy = np.copy(occGrid)

        for lane in range(0, LANES):
            for idx in range(half, occGrid[lane].shape[0]):
                if(tmpCopy[lane][idx] == 1):
                    occGrid[lane][idx+1:idx + numIndexesOthers + 1] = 1
                    val = velGrid[lane][idx]
                    velGrid[lane][idx+1:idx + numIndexesOthers + 1] = val
        
        tmp = {}
        tmp[0] = np.sort(laneMap[0], order=['pos'])[::-1]
        tmp[1] = np.sort(laneMap[1], order=['pos'])[::-1]

        for lane in range(0, LANES):
            backward_done = False
            _prev_ = None

            for j in range(0, tmp[lane].shape[0]):
                if(tmp[lane][j]['pos'] + self.shift[lane] < agentPos):
                    _prev_ = [i for i, tup in enumerate(laneMap[lane]) if tup['pos'] == tmp[lane][j]['pos']][0]
                    break
                
            if(_prev_ == None):
                _prev_ = laneMap[lane].shape[0] - 1
            
            count = 0
            copy_prev = _prev_

            for x in range(0, laneMap[lane].shape[0]):

                prev_car_angle = laneMap[lane][copy_prev]['pos'] + self.shift[lane]
                angleDiff = (agentPos - prev_car_angle)
                angleDiff = angleDiff % 360

                if(angleDiff <= self.halfExtendedViewInAngle[lane]):
                    count += 1
                    copy_prev -=1
                    if(copy_prev < 0):
                        copy_prev = laneMap[lane].shape[0] - 1
                else:
                    break
            
            for x in range(0, count):

                prev_car_angle = laneMap[lane][_prev_]['pos'] + self.shift[lane]
                prev_car_speed = laneMap[lane][_prev_]['speed']
                angleDiff = (agentPos - prev_car_angle)
                angleDiff %= 360

                if(angleDiff <= self.halfExtendedViewInAngle[lane]):
                    dist = arcLength(LANE_RADIUS[lane], angleDiff)
                    distInMetres = dist / SCALE
                    index = distInMetres / float(self.cellSize)

                    halfLen = int(occGrid[0].shape[0]/2)
                    occGrid[lane][halfLen - 1 - int(index)] = 1
                    velGrid[lane][halfLen - 1 - int(index)] = prev_car_speed
                    _prev_ -= 1
        
        tmpCopy = np.copy(occGrid)
        halfLen = int(occGrid[0].shape[0]/2)
        numIndexs = math.ceil((CAR_LENGTH * 0.5)/self.cellSize)
        numIndexesOthers = math.ceil(CAR_LENGTH/self.cellSize)

        for lane in range(0, LANES):
            for idx in range(0, halfLen):
                if(tmpCopy[lane][idx] == 1):
                    if(idx - numIndexesOthers <= 0):
                        occGrid[lane][0:idx] = 1
                        val = velGrid[lane][idx]
                        velGrid[lane][0:idx] = val
                    else:
                        occGrid[lane][idx - numIndexesOthers : idx] = 1
                        val = velGrid[lane][idx]
                        velGrid[lane][idx - numIndexesOthers:idx] = val

        halfLen = int(occGrid[0].shape[0]/2)
        numIndexs = math.ceil((CAR_LENGTH * 0.5)/self.cellSize)
        occGrid[agentLane][halfLen-1] = 2
        occGrid[agentLane][halfLen-1+1:halfLen-1+numIndexs+1] = 2
        occGrid[agentLane][halfLen-1-numIndexs:halfLen-1] = 2
        velGrid[agentLane][halfLen-1] = laneMap[agentLane][agentID]['speed']
        velGrid[agentLane][halfLen-1+1:halfLen-1+numIndexs+1] = laneMap[agentLane][agentID]['speed']
        velGrid[agentLane][halfLen-1-numIndexs:halfLen-1] = laneMap[agentLane][agentID]['speed']

        lowRange = agentPos - self.shift[agentLane]
        highRange = agentPos = self.shift[agentLane]
        otherLanes = [0, 1]
        otherLanes.pop(agentLane)
        for otherLane in otherLanes:
            for car in range( laneMap[otherLane].shape[0]):

                if(laneMap[otherLane][car]['pos'] >= lowRange and laneMap[otherLane][car]['pos'] <= highRange):
                    viewAngle = agentPos - self.halfExtendedViewInAngle[lane]
                    angleDiff = laneMap[otherLane][car]['pos'] - viewAngle
                    angleDiff = angleDiff % 360

                    dist = arcLength(LANE_RADIUS[lane], angleDiff)
                    distInMetres = dist / SCALE
                    index = distInMetres / float(self.cellSize)
                    occGrid[otherLane][int(index)] = 1
                    velGrid[otherLane][int(index)] = laneMap[otherLane][car]['speed']
                    occGrid[otherLane][int(index)+1:int(index)+numIndexs+1] = 1
                    occGrid[otherLane][int(index)-numIndexs:int(index)] = 1
                    val = velGrid[otherLane][int(index)]
                    velGrid[otherLane][int(index)+1:int(index)+numIndexs+1] = val
                    velGrid[otherLane][int(index)-numIndexs:int(index)] = val
    
        return occGrid
        
    def getGrids(self, laneMap, agentLane):
        return self.getOccupancyGrid(laneMap, agentLane)

