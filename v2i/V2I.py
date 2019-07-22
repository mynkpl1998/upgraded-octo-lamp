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
        
        self.densities = list(self.trajecDict[0].keys())

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
        self.idmHandler = idm(self.simArgs.getValue('max-speed'), self.simArgs.getValue("t-period"), self.simArgs.getValue("view-range"))
        
    def buildlaneMap(self, trajecDict, trajecIndex, epsiodeDensity, numCars):
        laneMap = {}
        for lane in range(0, constants.LANES):
            carsProperties = []
            for carID in range(0, numCars[lane]):
                # Pos, Speed, Lane, Agent, CarID
                tup = (trajecDict[lane][epsiodeDensity][trajecIndex[lane]][carID], 0.0, 0, 0, carID)
                carsProperties.append(tup)
            #print(carsProperties)
            laneMap[lane] = np.array(carsProperties, dtype=[('pos', 'f8'), ('speed', 'f8'), ('lane', 'f8'), ('agent', 'f8'), ('id', 'f8')])
        return laneMap
    
    def packRenderData(self, laneMap, timeElapsed, agentLane, maxSpeed, viewRange):
        data = {}
        agentID = np.where(self.lane_map[agentLane]['agent'] == 1)[0]
        data["allData"] = laneMap
        data["agentSpeed"] = laneMap[agentLane][agentID]['speed'][0]
        data["timeElapsed"] = timeElapsed
        data["maxSpeed"] = maxSpeed
        data["viewRange"] = viewRange
        data["agentLane"] = agentLane
        return data

    def reset(self, density=None):
        # ---- Density Generation ----#
        epsiodeDensity = None
        if density == None:
            randomIndex = np.random.randint(0, len(self.densities))
            epsiodeDensity = self.densities[randomIndex]
        else:
            if density not in self.densities:
                raiseValueError("invalid density -> %f"%(density))
            epsiodeDensity = density
        # ---- Density Generation ----#
        
        # ---- Init variables ----#
        self.time_elapsed = 0
        self.num_cars = {}
        self.num_trajec = {}
        
        for lane in range(0, constants.LANES):
            self.num_trajec[lane] = len(self.trajecDict[lane][epsiodeDensity])            
        
        self.trajecIndex = {}
        self.num_cars = {}
        for lane in range(0, constants.LANES):
            self.trajecIndex[lane] = np.random.randint(0, self.num_trajec[lane])
            self.num_cars[lane] = len(self.trajecDict[lane][epsiodeDensity][self.trajecIndex[lane]])
        
        self.lane_map = self.buildlaneMap(self.trajecDict, self.trajecIndex, epsiodeDensity, self.num_cars)
        
        ''' 
        Randomly Choose Agent Lane and Agent Car ID
        '''
        self.agent_lane = np.random.randint(0, constants.LANES)
        self.randomIDX = np.random.randint(0, self.num_cars[self.agent_lane])
        self.lane_map[self.agent_lane][self.randomIDX]['agent'] = 1
        
        # ---- Init variables ----#    
        if self.simArgs.getValue("render"):
            self.uiHandler.updateScreen(self.packRenderData(self.lane_map, self.time_elapsed, self.agent_lane, self.simArgs.getValue("max-speed"), self.simArgs.getValue("view-range")))

    def step(self):

        # IDM Update Step
        self.idmHandler.step(self.lane_map)

        self.time_elapsed += self.simArgs.getValue("t-period")
        
        # Update Display if render is enabled
        if self.simArgs.getValue("render"):
            self.uiHandler.updateScreen(self.packRenderData(self.lane_map, self.time_elapsed, self.agent_lane, self.simArgs.getValue("max-speed"), self.simArgs.getValue("view-range")))
        
        print(self.time_elapsed)