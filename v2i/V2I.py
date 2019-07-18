import gym

from v2i.src.core.utils import configParser

class V2I(gym.Env):

    def __init__(self, config):
        self.simArgs = configParser(config)

