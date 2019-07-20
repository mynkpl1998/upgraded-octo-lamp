import gym
import numpy as np
import v2i.src.core.constants as constants

from v2i.src.core.utils import configParser
from v2i.src.core.common import loadPKL, raiseValueError
from v2i.src.ui.ui import ui
from v2i.src.core.idm import idm

class V2I(gym.Env):

    def __init__(self, config):

        # Parse Config file and return the handle
        self.simArgs = configParser(config)

        # Seed the random number generator
        self.seed(self.simArgs.getValue("seed"))

        # Load Trajectories
        self.trajecDict = loadPKL('v2i/src/data/trajec.pkl')
        if self.trajecDict == None:
            raiseValueError("no or invalid trajectory file found at v2i/src/data/")
        
        self.densities = list(self.trajecDict.keys())

        # Initializes the required variables
        self.init()
    
    def seed(self, value=0):
        np.random.seed(value)
    
    def init(self):
        '''
        Function : All common variables initialization goes here.
        '''
        # Inititalize UI Handler here
        if self.simArgs.getValue("render"):
            self.uiHandler = ui(self.simArgs.getValue('fps'))
            self.ui_data = {}
        
        # Initialize IDM Handler here
        self.idmHandler = idm(self.simArgs.getValue('max-speed'), self.simArgs.getValue("t-period"))
        
        
    def buildlaneMap(self, trajec, numCars):
        laneMap = {}
        '''
        Right now only single lane is there, but in future we may add more than one lane
        '''
        carsProperties = []
        for carID in range(0, numCars):
            # Pos, Speed, Lane, Agent, CarID
            tup = (trajec[carID], 0.0, 0, 0, carID)
            carsProperties.append(tup)
        
        laneMap[0] = np.array(carsProperties, dtype=[('pos', 'f8'), ('speed', 'f8'), ('lane', 'f8'), ('agent', 'f8'), ('id', 'f8')])
        
        '''
        Randomly make one of the vehicle as Ego-Vehicle
        '''
        randomID = np.random.randint(0, numCars)
        laneMap[0][randomID]['agent'] = 1
        return laneMap
    
    def packRenderData(self, laneMap, timeElapsed, maxSpeed):
        data = {}
        agentID = np.where(self.lane_map[0]['agent'] == 1)[0]

        data["allData"] = laneMap
        data["agentSpeed"] = laneMap[0][agentID]['speed'][0]
        data["timeElapsed"] = timeElapsed
        data["maxSpeed"] = maxSpeed
        return data

    def reset(self, density=None):
        
        # ---- Density Generation ----#
        epsiodeDensity = None
        if density == None:
            randomIndex = np.random.randint(0, len(self.densities))
            epsiodeDensity = self.densities[randomIndex]
        else:
            if density not in self.densities:
                raiseValueError("invalid density -> %.1f"%(density))
            epsiodeDensity = density
        # ---- Density Generation ----#
    
        # ---- Init variables ----#
        self.time_elapsed = 0
        self.num_cars, self.num_trajec = len(self.trajecDict[epsiodeDensity][0]), len(self.trajecDict[epsiodeDensity])
        self.trajecIndex = np.random.randint(0, self.num_trajec)
        self.lane_map = self.buildlaneMap(self.trajecDict[epsiodeDensity][self.trajecIndex], self.num_cars)
        # ---- Init variables ----#

        if self.simArgs.getValue("render"):
            self.uiHandler.updateScreen(self.packRenderData(self.lane_map, self.time_elapsed, self.simArgs.getValue("max-speed")))
    
    def step(self):

        # IDM Update Step
        self.idmHandler.step(self.lane_map)

        self.time_elapsed += self.simArgs.getValue("t-period")
        
        # Update Display if render is enabled
        if self.simArgs.getValue("render"):
            self.uiHandler.updateScreen(self.packRenderData(self.lane_map, self.time_elapsed, self.simArgs.getValue("max-speed")))
        
        print(self.time_elapsed)