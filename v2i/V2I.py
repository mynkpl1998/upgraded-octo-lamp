import gym
import numpy as np
import v2i.src.core.constants as constants

from v2i.src.core.utils import configParser
from v2i.src.core.common import loadPKL, raiseValueError

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
        
    
    def reset(self, density=None):
        
        epsiodeDensity = None
        if density == None:
            randomIndex = np.random.randint(0, len(self.densities))
            epsiodeDensity = self.densities[randomIndex]
        else:
            epsiodeDensity = density
        
        
        
        print(epsiodeDensity)

    def seed(self, value=0):
        np.random.seed(value)
     
