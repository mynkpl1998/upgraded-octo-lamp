import gym
import numpy as np
import v2i.src.core.constants as constants

from v2i.src.core.utils import configParser, ActionEncoderDecoder
from v2i.src.core.common import loadPKL, raiseValueError
from v2i.src.core.occupancy import Grid
from v2i.src.ui.ui import ui
from v2i.src.core.idm import idm
from v2i.src.core.controller import egoController
from v2i.src.core.common import getAgentID

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
        # Initialize IDM Handler here
        self.idmHandler = idm(self.simArgs.getValue('max-speed'), self.simArgs.getValue("t-period"), self.simArgs.getValue("local-view"))

        # Intialize Grid Handler here
        self.gridHandler = Grid(2 * self.simArgs.getValue("local-view"), self.simArgs.getValue("max-speed"), self.simArgs.getValue("reg-size"), 2 * self.simArgs.getValue("extended-view"), self.simArgs.getValue("cell-size"))

        # Inititalize UI Handler here
        if self.simArgs.getValue("render"):
            self.uiHandler = ui(self.simArgs.getValue('fps'), self.gridHandler.totalExtendedView, self.gridHandler.cellSize)
            self.ui_data = {}
        
        # Initialize Ego Vehicle controller here
        self.egoControllerHandler = egoController(self.simArgs.getValue("t-period"), self.simArgs.getValue("max-speed"))

        # Init Gym Env Properties
        self.initGymProp(self.gridHandler)
    
    def initGymProp(self, obsHandler):
        self.observation_space = obsHandler.observation_space
        
        # Intialize Action Encoder and Decoder
        self.actionEncoderDecoderHandler = ActionEncoderDecoder(self.egoControllerHandler.planSpace(), self.gridHandler.querySpace())
        
        # Encode Actions
        self.actionEncoderDecoderHandler.encodeActions()

        # Get Action Space
        self.action_space = self.actionEncoderDecoderHandler.getActionSpace()

        # Get Action Map
        self.action_map = self.actionEncoderDecoderHandler.actMap
        
    def buildlaneMap(self, trajecDict, trajecIndex, epsiodeDensity, numCars):
        laneMap = {}
        for lane in range(0, constants.LANES):
            carsProperties = []
            for carID in range(0, numCars[lane]):
                # Pos, Speed, Lane, Agent, CarID
                tup = (trajecDict[lane][epsiodeDensity][trajecIndex[lane]][carID], 0.0, 0, 0, carID)
                carsProperties.append(tup)
            laneMap[lane] = np.array(carsProperties, dtype=[('pos', 'f8'), ('speed', 'f8'), ('lane', 'f8'), ('agent', 'f8'), ('id', 'f8')])
        return laneMap
    
    def packRenderData(self, laneMap, timeElapsed, agentLane, maxSpeed, viewRange, extendedRange, occGrid, planAct, queryAct, agentReward):
        data = {}
        agentID = np.where(self.lane_map[agentLane]['agent'] == 1)[0]
        data["allData"] = laneMap
        data["agentSpeed"] = laneMap[agentLane][agentID]['speed'][0]
        data["timeElapsed"] = timeElapsed
        data["maxSpeed"] = maxSpeed
        data["viewRange"] = viewRange
        data["extendedViewRange"] = extendedRange
        data["agentLane"] = agentLane
        data["occGrid"] = occGrid
        data["planAct"] = planAct
        data["queryAct"] = queryAct
        data["agentReward"] = agentReward
        return data

    def fixIssue2(self, density):
        if density == 0.3:
            return self.densities[2]
        elif density == 0.7:
            return self.densities[6]
        else:
            return density
        
    def reset(self, density=None):
        density = self.fixIssue2(density)
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
        
        #---- Get Occupancy & Velocity Grids ----#
        occGrid, velGrid = self.gridHandler.getGrids(self.lane_map, self.agent_lane, 'null')
        #---- Get Occupancy & Velocity Grids ----#

        # ---- Init variables ----#    
        if self.simArgs.getValue("render"):
            self.uiHandler.updateScreen(self.packRenderData(self.lane_map, self.time_elapsed, self.agent_lane, self.simArgs.getValue("max-speed"), self.gridHandler.totalLocalView, self.gridHandler.totalExtendedView, occGrid, "none", "null", "none"))
        
        return self.buildObservation(occGrid, velGrid)
    
    def buildObservation(self, occGrid, velGrid):
        combinedObs = np.concatenate((occGrid.flatten(), velGrid.flatten()))
        return combinedObs.copy()
    
    def rewardFunc(self, laneMap, agentLane):
        agentIDX = getAgentID(laneMap, agentLane)
        return laneMap[agentLane][agentIDX]['speed']

    def step(self, action):

        # Decodes Action -> Plan Action, Query Action
        planAct, queryAct = self.actionEncoderDecoderHandler.decodeAction(action)
        
        # Perform the required planing action
        egodistTravelledInDeg, egoSpeed, collision, laneToChange = self.egoControllerHandler.executeAction(planAct, self.lane_map, self.agent_lane)

        # Change Lane if lane changes is asked and is valid
        if(laneToChange != self.agent_lane and collision == False):
            agentIDX = getAgentID(self.lane_map, self.agent_lane)
            egoVehicleProp = self.lane_map[self.agent_lane][agentIDX]
            self.lane_map[self.agent_lane] = np.delete(self.lane_map[self.agent_lane], agentIDX)
            self.lane_map[laneToChange] = np.append(egoVehicleProp, self.lane_map[laneToChange])
            self.agent_lane = laneToChange
            egodistTravelledInDeg, egoSpeed, collision, laneToChange = self.egoControllerHandler.executeAction("do-nothing", self.lane_map, self.agent_lane)
        
        # Update Agent Location and Speed
        self.lane_map[self.agent_lane][getAgentID(self.lane_map, self.agent_lane)]['pos'] += egodistTravelledInDeg
        self.lane_map[self.agent_lane][getAgentID(self.lane_map, self.agent_lane)]['pos'] %= 360
        self.lane_map[self.agent_lane][getAgentID(self.lane_map, self.agent_lane)]['speed'] = egoSpeed        

        # IDM Update Step
        self.idmHandler.step(self.lane_map)

        self.time_elapsed += self.simArgs.getValue("t-period")

        #---- Get Occupancy & Velocity Grids ----#
        occGrid, velGrid = self.gridHandler.getGrids(self.lane_map, self.agent_lane, queryAct)
        #---- Get Occupancy & Velocity Grids ----#

        #---- Calculate Reward ----#
        reward = self.rewardFunc(self.lane_map, self.agent_lane)
        if collision:
            reward = self.simArgs.getValue("collision-penalty")
        #---- Calculate Reward ----#

        # ---- Init variables ----#    
        if self.simArgs.getValue("render"):
            self.uiHandler.updateScreen(self.packRenderData(self.lane_map, self.time_elapsed, self.agent_lane, self.simArgs.getValue("max-speed"), self.gridHandler.totalLocalView, self.gridHandler.totalExtendedView, occGrid, planAct, queryAct, round(reward, 1)))
        
        # state, reward, done, info
        return self.buildObservation(occGrid, velGrid), reward, collision, {}