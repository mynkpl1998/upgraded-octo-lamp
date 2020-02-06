import numpy as np
from v2i.src.core.constants import LANE_RADIUS, SCALE, LANES, CAR_LENGTH

class trueAge:

    def __init__(self, sensors_range, lv, cv):
        self.sensors_range = sensors_range
        self.lv = lv
        self.cv = cv
        self.lv_pixels = self.lv * SCALE
        self.cv_pixels = self.cv * SCALE
        
        self.sensors_range_pixels = self.sensors_range * SCALE
        self.sensors_range_deg = [self.getAngle(self.sensors_range_pixels, LANE_RADIUS[0]), self.getAngle(self.sensors_range_pixels, LANE_RADIUS[1])]
        self.lv_deg = [self.getAngle(self.lv_pixels, LANE_RADIUS[0]), self.getAngle(self.lv_pixels, LANE_RADIUS[1])]
        self.cv_deg = [self.getAngle(self.cv_pixels, LANE_RADIUS[0]), self.getAngle(self.cv_pixels, LANE_RADIUS[1])]

        self.mapSensors()
        '''
        self.initSensorsAge()
        self.gap = {}
        self.gap[0] = self.getAngle(CAR_LENGTH, LANE_RADIUS[0])
        self.gap[1] = self.getAngle(CAR_LENGTH, LANE_RADIUS[1])
        '''
    
    
    def getAngle(self, arcLength, radius):
        return np.rad2deg(arcLength / float(radius))
    
    def initTracking(self):
        self.tracker = {}
        self.lanesWiseMapping = {}
        self.lanesWiseMapping[0] = []
        self.lanesWiseMapping[1] = []

        for lane in range(0, LANES):
            for sensorID in self.sensorsMapping[lane]:
                self.tracker[self.sensorsMapping[lane][sensorID]] = []
                self.lanesWiseMapping[lane].append(self.sensorsMapping[lane][sensorID])
        print(self.lanesWiseMapping)

    def getRearOverlappingSensors(self, startRange, endRange):
        overlappingSensors = []
        for lane in range(0, LANES):
            for low, high in self.sensorsMapping[lane]:
                if startRange[lane] > endRange[lane]:
                    if(startRange[lane] >= low and low >= endRange[lane]) or (startRange[lane] >= high and high >= endRange[lane]):
                        overlappingSensors.append(self.sensorsMapping[lane][(low, high)])
                elif startRange[lane] < endRange[lane]:
                    if((low <= startRange[lane] and low >= 0) or (low >= endRange[lane] and low <= 360) or (high <= startRange[lane] and high >= 0) or (high >= endRange[lane] and high <= 360)):
                        overlappingSensors.append(self.sensorsMapping[lane][(low, high)])
                else:
                    raise ValueError("Invalid...")
        return overlappingSensors

    def getFrontOverlappingSensors(self, startRange, endRange):
        overlappingSensors = []
        for lane in range(0, LANES):
            for low, high in self.sensorsMapping[lane]:
                if startRange[lane] > endRange[lane]:
                    if (startRange[lane] >= low and low >= endRange[lane]) or (startRange[lane] >= high and high >= endRange[lane]):
                        overlappingSensors.append(self.sensorsMapping[lane][(low, high)])
                elif startRange[lane] < endRange[lane]:
                    if((low <= startRange[lane] and low >= 0) or (low >= endRange[lane] and low <= 360) or (high <= startRange[lane] and high >= 0) or (high >= endRange[lane] and high <= 360)):
                        overlappingSensors.append(self.sensorsMapping[lane][(low, high)])
                else:
                    raise ValueError("Invalid...")
        return overlappingSensors            

    def updateTracker(self, agentLoc, agentLane, agentQuery, time):
        assert agentQuery == 'reg_0' or agentQuery == 'reg_1' or agentQuery == 'null'
        if agentQuery == 'null':
            pass
        else:
            if agentQuery == "reg_0":
                startRange = agentLoc - self.lv_deg
                startRange %= 360
                endRange = agentLoc - self.lv_deg - self.cv_deg
                endRange %= 360
                overlappingSensors = self.getRearOverlappingSensors(startRange, endRange)
            
            elif agentQuery == "reg_1":
                endRange = agentLoc + self.lv_deg
                endRange %= 360
                startRange = agentLoc + self.lv_deg + self.cv_deg
                startRange %= 360
                overlappingSensors = self.getFrontOverlappingSensors(startRange, endRange)
            
            else:
                overlappingSensors = []
            
            for sensor in overlappingSensors:
                self.tracker[sensor].append(time)

    
    def mapSensors(self):
        self.sensorsMapping = {}
        senseCount = 0
        for lane in range(0, LANES):
            self.sensorsMapping[lane] = {}
            start_range = 0.0
            end_range = start_range + self.sensors_range_deg[lane]
            while start_range < 360:
                self.sensorsMapping[lane][(start_range, end_range)] = senseCount
                senseCount += 1
                start_range += self.sensors_range_deg[lane]
                end_range += self.sensors_range_deg[lane]
        self.numSensors = senseCount
    
    def initSensorsAge(self):
        self.trueAgeRegister = {}
        for lane in self.sensorsMapping.keys():
            for r in self.sensorsMapping[lane].keys():
                sensorID = self.sensorsMapping[lane][r]
                self.trueAgeRegister[sensorID] = 0.0
        assert self.numSensors == len(self.trueAgeRegister)
    
    def resetSensOcc(self):
        self.currSensorsOcc = {}
        for sensor in self.trueAgeRegister.keys():
            self.currSensorsOcc[sensor] = []

    def resetSensors(self, laneMap):
        self.resetSensOcc()
        for sensor in self.trueAgeRegister.keys():
            self.trueAgeRegister[sensor] = 0.0
        for lane in range(0, LANES):
            for veh in laneMap[lane]:
                sensorID = self.getSensorID(veh['pos'])
                t = (veh['id'])
                self.currSensorsOcc[sensorID].append(t)
        
    def getMeanAge(self):
        ageValues = []
        for sensorID in self.trueAgeRegister.keys():
            ageValues.append(self.trueAgeRegister[sensorID])
        ageValues = np.array(ageValues)
        return ageValues.mean()
    
    def getSensorID(self, vehPos):
        for lane in range(0, LANES):
            for r in self.sensorsMapping[lane]:
                start_range = r[0]
                end_range = r[1]
                if vehPos >= start_range and vehPos < end_range:
                    return self.sensorsMapping[lane][r]
        raise ValueError("invalid veh position %.2f in lane %d"%(vehPos, lane))
    
    def step(self, laneMap):
        nextOcc = {}
        for sensor in self.trueAgeRegister.keys():
            nextOcc[sensor] = []
        assert len(nextOcc) == len(self.currSensorsOcc)
        
        for lane in range(0, LANES):
            for veh in laneMap[lane]:
                sensorID = self.getSensorID(veh['pos'])
                t = (veh['id'])
                nextOcc[sensorID].append(t)
        
        for sensorID in self.trueAgeRegister.keys():
            flag = True
            if len(self.currSensorsOcc[sensorID]) != len(nextOcc[sensorID]):
                self.trueAgeRegister[sensorID] = 0.0
                flag = False
            else:
                for element in self.currSensorsOcc[sensorID]:
                    if element not in nextOcc[sensorID]:
                        self.trueAgeRegister[sensorID] = 0.0
                        flag = False
                        break
            if flag:
                self.trueAgeRegister[sensorID] += 0.1
        
        #print(self.trueAgeRegister)
        self.currSensorsOcc = nextOcc.copy()
        
        