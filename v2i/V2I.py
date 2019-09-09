import os
import sys
import gym
import numpy as np

import v2i.src.core.constants as constants
from v2i.src.core.utils import configParser, ActionEncoderDecoder
from v2i.src.core.common import loadPKL, raiseValueError
from v2i.src.core.occupancy import Grid
from v2i.src.ui.ui import ui
from v2i.src.core.idm import idm
from v2i.src.core.controller import egoController
from v2i.src.core.common import getAgentID, arcLength, getTfID
from v2i.src.core.tfLights import tfController
from v2i.src.core.constants import TF_CONSTS
from v2i.src.core.obsQueue import obsWrapper


class V2I(gym.Env):

    '''
    Input Params :
        1. config - simulation config file used for training.
        2. mode - if mode is set to "train", then objects properties are set to values given in config file and disbales the rendering.
                "train" mode signifies, this object will be used for training purpose.
                If mode is set to any thing other than "train", then  it signifies that object will be created for testing mode.
                You can pass additional param dict to overwrite the values specified by config file.
        3. params : Expects a dictionay when mode is not set to "train"
    '''

    def __init__(self, config, mode, params=None):

        # Parse Config file and return the handle
        self.simArgs = configParser(config)

        # Set render based on mode
        if mode == "train":
            self.simArgs.setValue("render", False)
        else:
            if params is not None:
                for param in params.keys():
                    self.simArgs.setValue(param, params[param])

        # Seed the random number generator
        self.seed(self.simArgs.getValue("seed"))

        # Set the observation size based on memory
        if self.simArgs.getValue("enable-lstm"):
            self.simArgs.setValue("k-frames", 1)

        # Load Trajectories
        currPath = os.path.realpath(__file__)[:-6]
        self.trajecDict = loadPKL(currPath + 'src/data/trajec.pkl')
        if self.trajecDict == None:
            raiseValueError("no or invalid trajectory file found at v2i/src/data/")
        
        self.densities = list(self.trajecDict[0].keys())

        # Initializes the required variables
        self.init()

        # Check for valid frame-skip-value
        if self.simArgs.getValue("frame-skip-value") <= 0:
            raiseValueError("frame skip value should be at least one")

        # Info Dict
        self.infoDict = self.buildInfoDict(self.densities)

    def buildInfoDict(self, densities):
        infoDict = {}
        for density in densities:
            infoDict[density] = 0
        infoDict["totalEpisodes"] = 0
        return infoDict

    def seed(self, value=0):
        np.random.seed(value)
    
    def init(self):
        '''
        Function : All common variables initialization goes here.
        '''
        # Initialize IDM Handler here
        self.idmHandler = idm(self.simArgs.getValue('max-speed'), self.simArgs.getValue("t-period"), self.simArgs.getValue("local-view"))

        # Intialize Grid Handler here
        self.gridHandler = Grid(2 * self.simArgs.getValue("local-view"), self.simArgs.getValue("max-speed"), self.simArgs.getValue("reg-size"), self.simArgs.getValue("k-frames"),2 * self.simArgs.getValue("extended-view"), self.simArgs.getValue("cell-size"))
        
        # Initialize Traffic Lights
        if self.simArgs.getValue("enable-tf"):
            self.tfHandler = tfController(self.simArgs.getValue("t-period"))

        # Inititalize UI Handler here
        if self.simArgs.getValue("render"):
            self.uiHandler = ui(self.simArgs.getValue('fps'), self.gridHandler.totalExtendedView, self.gridHandler.cellSize, self.simArgs.getValue("enable-tf"))
            self.ui_data = {}
        
        # Initialize Ego Vehicle controller here
        self.egoControllerHandler = egoController(self.simArgs.getValue("t-period"), self.simArgs.getValue("max-speed"))

        # Init Gym Env Properties
        self.initGymProp(self.gridHandler)

        # Initialze Observation Wrapper if lstm is disabled
        self.obsWrapper = obsWrapper(self.simArgs.getValue('k-frames'), int(self.gridHandler.observation_space.shape[0]/self.simArgs.getValue("k-frames")))

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

        # Init TF Speed Limit
        self.tfSpeedLimit = self.initTfSpeedLimit()

    def initTfSpeedLimit(self):
        self.distanceInMetre = None
        if self.gridHandler.isCommEnabled:
            self.distanceInMetre = self.simArgs.getValue("local-view") + self.gridHandler.regWidthInMetres
        else:
            self.distanceInMetre = self.simArgs.getValue("local-view")
        return np.sqrt((2 * constants.IDM_CONSTS['DECELERATION_RATE'] * self.distanceInMetre))
        
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
            #randomIndex = np.random.randint(0, len(self.densities))
            #epsiodeDensity = self.densities[randomIndex]
            epsiodeDensity = np.random.choice(self.densities, p=constants.DENSITIES_WEIGHTS)
        else:
            if density not in self.densities:
                raiseValueError("invalid density -> %f"%(density))
            epsiodeDensity = density
        # ---- Density Generation ----#
        
        # ---- Init variables ----#
        self.time_elapsed = 0
        self.num_cars = {}
        self.num_trajec = {}
        self.num_steps = 0
        self.infoDict['totalEpisodes'] += 1
        self.infoDict[epsiodeDensity] += 1
        
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

        # ---- Set Traffic Light ----#
        if self.simArgs.getValue("enable-tf"):
            self.isLightRed = [False, False]
            if np.random.rand() <= TF_CONSTS['EPISODE_TF_GEN_PROB']:
                self.tfTogglePts = self.tfHandler.expandPts()
            else:
                self.tfTogglePts = [[], []]

        # ---- Init variables ----#
        if self.simArgs.getValue("render"):
            if self.simArgs.getValue("enable-tf"):
                self.uiHandler.updateScreen(self.packRenderData(self.lane_map, self.time_elapsed, self.agent_lane, self.simArgs.getValue("max-speed"), self.gridHandler.totalLocalView, self.gridHandler.totalExtendedView, occGrid, "none", "null", "none"), self.isLightRed)
            else:
                self.uiHandler.updateScreen(self.packRenderData(self.lane_map, self.time_elapsed, self.agent_lane, self.simArgs.getValue("max-speed"), self.gridHandler.totalLocalView, self.gridHandler.totalExtendedView, occGrid, "none", "null", "none"), None)
        
        # Reset Observation Queue
        self.obsWrapper.resetQueue()
        obs = self.buildObservation(occGrid, velGrid)
        self.obsWrapper.addObs(obs)
        return self.obsWrapper.getObs()
    
    def buildObservation(self, occGrid, velGrid):
        combinedObs = np.concatenate((occGrid.flatten(), velGrid.flatten()))
        return combinedObs.copy()
    
    def getBum2BumDist(self, laneMap, agentLane, agentIDX):
        if agentIDX == 0:
            angleDiff = laneMap[agentLane][-1]['pos'] - laneMap[agentLane][0]['pos'] 
        else:
            angleDiff = laneMap[agentLane][agentIDX-1]['pos'] - laneMap[agentLane][agentIDX]['pos']
        angleDiff %= 360
        return (arcLength(constants.LANE_RADIUS[agentLane], angleDiff) / constants.SCALE) - constants.CAR_LENGTH
    
    def rewardFunc(self, laneMap, agentLane, planAct):
        tmpLaneMap = laneMap.copy()
        tmpLaneMap[agentLane] = np.sort(tmpLaneMap[agentLane], order=['pos'])[::-1]
        agentIDX = getAgentID(tmpLaneMap, agentLane)
        bum2bumDist = self.getBum2BumDist(tmpLaneMap, agentLane, agentIDX)
        agentSpeed = tmpLaneMap[agentLane][agentIDX]['speed']
        #print(bum2bumDist)
        if agentSpeed > self.tfSpeedLimit:
            return -1
        if bum2bumDist < (constants.CAR_LENGTH + 1):
            return -1
        elif planAct == "lane-change":
            return (tmpLaneMap[agentLane][agentIDX]['speed'] / self.simArgs.getValue('max-speed'))
        else:
            return tmpLaneMap[agentLane][agentIDX]['speed'] / self.simArgs.getValue("max-speed") + 0.1
        
    def commPenalty(self, PlanReward, queryAct):
        if queryAct == "null":
            return PlanReward + self.simArgs.getValue("nocomm-incentive")
        else:
            return PlanReward
    
    def step(self, action):
        for i in range(self.simArgs.getValue("frame-skip-value")):
            observation, reward, done, infoDict = self.frame(action)
            if done:
                return (observation, reward, done, infoDict)
        return (observation, reward, done, infoDict)

    def processInfoDict(self):
        infoDict = {}
        for density in self.densities:
            infoDict[density] = self.infoDict[density] / self.infoDict["totalEpisodes"]
        return infoDict

    def frame(self, action):
        
        self.num_steps += 1
        
        # Check for turing tf to green or red
        '''
        if self.simArgs.getValue("enable-tf"):
            agentIDX = getAgentID(self.lane_map, self.agent_lane)
            agentSpeed = self.lane_map[self.agent_lane][agentIDX]['speed']
            if agentSpeed >= self.tfSpeedLimit:
                for lane in range(0, constants.LANES):
                    self.isLightRed[lane] = True
            else:
                for lane in range(0, constants.LANES):
                    self.isLightRed[lane] = False
        '''
        if self.simArgs.getValue("enable-tf"):
            for lane in range(0, constants.LANES):
                if self.num_steps in self.tfTogglePts[lane]:
                    self.isLightRed[lane] = self.tfHandler.toggle(self.isLightRed, lane)
        

        # Decodes Action -> Plan Action, Query Action
        planAct, queryAct = self.actionEncoderDecoderHandler.decodeAction(action)
        
        self.planAct = planAct
        self.queryAct = queryAct

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

        # Add a vehicle if tf light is Red
        if self.simArgs.getValue("enable-tf"):
            for lane in range(0, constants.LANES):
                if self.isLightRed[lane]:
                    self.lane_map = self.tfHandler.addDummytfVehicle(self.lane_map, lane)
        
        # IDM Update Step
        self.idmHandler.step(self.lane_map)

        # Remove Dummy Vehicle if added
        if self.simArgs.getValue("enable-tf"):
            for lane in range(0, constants.LANES):
                if self.isLightRed[lane]:
                    tfIDX = getTfID(self.lane_map, lane)
                    self.lane_map[lane] = np.delete(self.lane_map[lane], tfIDX)

        self.time_elapsed += self.simArgs.getValue("t-period")

        #---- Get Occupancy & Velocity Grids ----#
        occGrid, velGrid = self.gridHandler.getGrids(self.lane_map, self.agent_lane, queryAct)
        #---- Get Occupancy & Velocity Grids ----#

        #---- Calculate Reward ----#
        reward = self.rewardFunc(self.lane_map, self.agent_lane, planAct)
        if self.gridHandler.isCommEnabled:
            reward = self.commPenalty(reward, queryAct)
        
        if collision:
            reward = -1 * self.simArgs.getValue("collision-penalty")
        
        self.collision = collision
        #---- Calculate Reward ----#

        # ---- Init variables ----#
        if self.simArgs.getValue("render"):
            if self.simArgs.getValue("enable-tf"):
                self.uiHandler.updateScreen(self.packRenderData(self.lane_map, self.time_elapsed, self.agent_lane, self.simArgs.getValue("max-speed"), self.gridHandler.totalLocalView, self.gridHandler.totalExtendedView, occGrid, planAct, queryAct, round(reward, 3)), self.isLightRed)
            else:
                self.uiHandler.updateScreen(self.packRenderData(self.lane_map, self.time_elapsed, self.agent_lane, self.simArgs.getValue("max-speed"), self.gridHandler.totalLocalView, self.gridHandler.totalExtendedView, occGrid, planAct, queryAct, round(reward, 3)), None)
        
        # state, reward, done, info
        obs = self.buildObservation(occGrid, velGrid)
        self.obsWrapper.addObs(obs)
        return self.obsWrapper.getObs(), reward, False, self.processInfoDict()