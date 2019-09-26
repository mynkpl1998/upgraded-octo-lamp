import numpy as np

class maintainer:

    def __init__(self, stateShape):
        self.stateShape = stateShape
        self.currState = None
    
    def get(self):
        assert self.currState != None
        return self.currState.copy()
    
    def update(self, obs):
        for dimIDx, dim in enumerate(self.currState):
            assert obs.shape[dimIDx] == self.currState[dimIDx]
        self.currState = obs.copy()
    
