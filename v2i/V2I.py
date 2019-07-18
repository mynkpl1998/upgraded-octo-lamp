import gym
import numpy as np
import v2i.src.core.constants as constants

from v2i.src.core.utils import configParser

class V2I(gym.Env):

    def __init__(self, config):

        # Parse Config file and return the handle
        self.simArgs = configParser(config)
        
        # Seed the random number generator
        self.seed(self.simArgs.getValue("seed"))
        
    
    def reset(self, density=None):
        pass


    def seed(self, value=0):
        np.random.seed(value)
     
