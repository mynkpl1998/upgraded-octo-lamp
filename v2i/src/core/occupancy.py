import math
import numpy as np

from gym.spaces import Box
from v2i.src.core.utils import raiseValueError
from v2i.src.core.constants import CAR_LENGTH, LANE_RADIUS, SCALE, LANES, OCCGRID_CONSTS
from v2i.src.core.common import getAgentID, arcAngle, arcLength, reverseDict

class Grid:

    def __init__(self, localView, maxSpeed, regionWidth,k, extendedView=None, cellSize=1):
        self.totalLocalView = localView # in metre
        self.cellSize = cellSize # in metre
        self.totalExtendedView = extendedView

        self.maxSpeed = maxSpeed
        self.regWidthInMetres = regionWidth
        self.k = k

        #---- Checks ----#
        if self.totalExtendedView == None:
            self.totalExtendedView = self.totalLocalView
        
        if self.totalExtendedView < self.totalLocalView:
            raiseValueError("size of extended view should be greater than local view")
        #---- Checks ----#

        #---- Calculate Communicable Region ----#
        self.totalCommView = 2*((self.totalExtendedView/2) - (self.totalLocalView/2)) # in metre
        #---- Calculate Communicable Region ----#

        if self.totalExtendedView % self.cellSize != 0:
            raiseValueError("communication should be completely divisible by cellsize")
        
        if self.cellSize > CAR_LENGTH:
            raiseValueError("Cell-Size (%.2f m) should be smaller than Car Length (%.2f m) "%(self.cellSize, CAR_LENGTH))

        #---- Calculate the number of columns ----#
        self.numCols = int(self.totalExtendedView / self.cellSize)
        #---- Calculate the number of columns ----#

        self.init()
    
    def initObservationSpace(self, k):
        occBoundLow = np.ones((LANES, self.numCols)) * OCCGRID_CONSTS[min(OCCGRID_CONSTS, key=OCCGRID_CONSTS.get)]
        occBoundHigh = np.ones((LANES, self.numCols)) * OCCGRID_CONSTS[max(OCCGRID_CONSTS, key=OCCGRID_CONSTS.get)]
        
        velGridBoundLow = np.ones((LANES, self.numCols)) * 0.0
        velGridBoundHigh = np.ones((LANES, self.numCols)) * self.maxSpeed

        mergedBoundlow = []
        mergedBoundHigh = []

        for frame in range(0, k):
            mergedBoundlow.append(occBoundLow.copy())
            mergedBoundlow.append(velGridBoundLow.copy())

            mergedBoundHigh.append(occBoundHigh.copy())
            mergedBoundHigh.append(velGridBoundHigh.copy())
        
        mergedBoundlow = np.array(mergedBoundlow)
        mergedBoundHigh = np.array(mergedBoundHigh)
        
        obsLowBound = mergedBoundlow.flatten().astype(np.float)
        obsHighBound = mergedBoundHigh.flatten().astype(np.float)
        
        self.observation_space = Box(low=obsLowBound, high=obsHighBound, dtype=np.float)
        
        
    def initComm(self):
        numRegs = int(self.totalCommView / self.regWidthInMetres)
        regWidth = int(self.regWidthInMetres / self.cellSize)
        
        #---- Region Mapping ----#
        self.commMap = {}
        for idx in range(0, numRegs):
            regName = "reg_%d"%(idx)
            self.commMap[idx] = regName
        
        #---- Region Mapping ----#
        start = 0
        self.commIndexMap = {}
        for i in range(0, int(numRegs/2)):
            self.commIndexMap[self.commMap[i]] = list(np.linspace(start, start + regWidth-1, num=regWidth, dtype='int'))
            start += regWidth
        
        start = (int(self.numCols/2) - 1) + int((self.totalLocalView/2) / self.cellSize) + 1
        for i in range(int(numRegs/2), numRegs):
            self.commIndexMap[self.commMap[i]] = list(np.linspace(start, start + regWidth - 1, num=regWidth, dtype='int'))
            start += regWidth
        #---- Region to Index Map ----#

        #---- Add Null Query ----#
        self.commMap[numRegs] = "null"
        self.commIndexMap[self.commMap[numRegs]] = []
        #---- Add Null Query ----#

    def init(self):
        '''
        All common initialization goes here.
        '''
        self.halfExtendedViewInAngle = []
        self.shift = []
        for lane in range(0, LANES):
            self.halfExtendedViewInAngle.append(arcAngle(LANE_RADIUS[lane], (self.totalExtendedView/2) * SCALE))
            self.shift.append(arcAngle(LANE_RADIUS[lane], (CAR_LENGTH/2) * SCALE))
        
        # Intialize Observation Space 
        self.initObservationSpace(self.k)

        # Intialize Communication
        self.isCommEnabled = False
        if self.totalCommView > 0:
            if not (self.totalCommView % self.regWidthInMetres == 0):
                raiseValueError("communicable region should be completely divisible by comm-size")
            
            if not (self.regWidthInMetres % self.cellSize == 0):
                raiseValueError("comm-size should be completely divisible by cell-size")
            
            self.isCommEnabled = True
            self.initComm()
        
    def getOccupancyGrid(self, laneMap, agentLane):
        occGrid = np.zeros((2, self.numCols))
        velGrid = np.zeros((2, self.numCols))
        
        for lane in range(0, LANES):
            laneMap[lane] = np.sort(laneMap[lane], order=['pos'])
        agentID = getAgentID(laneMap, agentLane)
        agentPos = laneMap[agentLane][agentID]['pos']
        
        '''
        Manually handle the bug which prevents the detection of vehicles near ego vehicles in other lane.
        '''
        for lane in range(0, LANES):
            shift = self.shift[lane]
            forward_done = False
            _next_ = None

            for j in range(0, laneMap[lane].shape[0]):
                if(laneMap[lane][j]['pos'] - shift > agentPos):
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
                
                angleDiff = next_car_angle - agentPos -shift
                angleDiff = angleDiff % 360

                if(angleDiff <= self.halfExtendedViewInAngle[lane]):
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
                
                angleDiff = next_car_angle - agentPos - shift
                angleDiff = angleDiff % 360

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
                if(tmp[lane][j]['pos'] + shift < agentPos):
                    _prev_ = [i for i, tup in enumerate(laneMap[lane]) if tup['pos'] == tmp[lane][j]['pos']][0]
                    break
                
            if(_prev_ == None):
                _prev_ = laneMap[lane].shape[0] - 1
            
            count = 0
            copy_prev = _prev_

            for x in range(0, laneMap[lane].shape[0]):

                prev_car_angle = laneMap[lane][copy_prev]['pos'] + shift
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

                prev_car_angle = laneMap[lane][_prev_]['pos'] + shift
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
        
        tmp_copy = np.copy(occGrid)
        half_len = int(occGrid[0].shape[0]/2)
        numIndexs = math.ceil((CAR_LENGTH * 0.5)/self.cellSize)
        numIndexesOthers = math.ceil(CAR_LENGTH/self.cellSize)

        for lane in range(0, LANES):
            for i in range(0, half_len):

                if(tmp_copy[lane][i] == 1):
                    
                    if(i - numIndexesOthers <= 0):
                        occGrid[lane][0:i] = 1
                        val = velGrid[lane][i]
                        velGrid[lane][0:i] = val
                    else:
                        occGrid[lane][i - numIndexesOthers:i] = 1
                        val = velGrid[lane][i]
                        velGrid[lane][i - numIndexesOthers:i] = val
        
        half_len = int(occGrid[0].shape[0]/2)
        numIndexs = math.ceil((CAR_LENGTH * 0.5)/self.cellSize)
        occGrid[agentLane][half_len - 1] = 2
        occGrid[agentLane][half_len-1+1:half_len-1+numIndexs+1] = 2
        occGrid[agentLane][half_len-1-numIndexs:half_len-1] = 2
        velGrid[agentLane][half_len - 1] = laneMap[agentLane][agentID]['speed']
        velGrid[agentLane][half_len-1+1:half_len-1+numIndexs+1] = laneMap[agentLane][agentID]['speed']
        velGrid[agentLane][half_len-1-numIndexs:half_len-1] = laneMap[agentLane][agentID]['speed']

        low_range = agentPos - shift
        high_range = agentPos + shift
        other_lanes = [0, 1]
        other_lanes.pop(agentLane)
        for other_lane in other_lanes:
            for car in range(0, laneMap[other_lane].shape[0]):

                if(laneMap[other_lane][car]['pos'] >= low_range and laneMap[other_lane][car]['pos'] <= high_range):
                    view_angle = agentPos - self.halfExtendedViewInAngle[other_lane]
                    angleDiff = laneMap[other_lane][car]['pos'] - view_angle
                    dist = arcLength(LANE_RADIUS[other_lane], angleDiff)
                    distInMetres = dist / SCALE
                    index = distInMetres / float(self.cellSize)
                    occGrid[other_lane][int(index)] = 1
                    velGrid[other_lane][int(index)] = laneMap[other_lane][car]['speed']
                    occGrid[other_lane][int(index)+1:int(index)+numIndexs+1] = 1
                    occGrid[other_lane][int(index)-numIndexs:int(index)] = 1
                    val = velGrid[other_lane][int(index)]
                    velGrid[other_lane][int(index)+1:int(index)+numIndexs+1] = val
                    velGrid[other_lane][int(index)-numIndexs:int(index)] = val

        return occGrid, velGrid
    
    def verifyGrids(self, occGrid, velGrid):
        assert occGrid.shape[0] == velGrid.shape[0]
        assert occGrid.shape[1] == velGrid.shape[1]
        for lane in range(0, LANES):
            for col in range(0, occGrid.shape[1]):
                if occGrid[lane][col] == 0:
                    if velGrid[lane][col] > 0.0:
                        raiseValueError("mistmatch in occGrids and velocity Grids")
                else:
                    if velGrid[lane][col] > self.maxSpeed or velGrid[lane][col] < 0.0:
                        raiseValueError("speed can be greater than %.2f and less than 0.0, However, speed of vehicle is %.2f"%(self.maxSpeed, velGrid[lane][col]))
    
    def querySpace(self):
        possibleQueries = []
        if self.isCommEnabled:
            for key in self.commMap.keys():
                possibleQueries.append(self.commMap[key])
        else:
            possibleQueries.append('null')
        return possibleQueries

    def getGrids(self, laneMap, agentLane, queryAct):
        occGrid, velGrid = self.getOccupancyGrid(laneMap, agentLane)
        self.verifyGrids(occGrid, velGrid)

        if self.isCommEnabled:
            for key in self.commMap:
                    if self.commMap[key] == queryAct:
                        pass
                    else:
                        indexs = self.commIndexMap[self.commMap[key]]
                        for index in indexs:
                            occGrid[:, index] = OCCGRID_CONSTS['UNKNOWN']
                            velGrid[:, index] = 0
        
        return occGrid, velGrid
