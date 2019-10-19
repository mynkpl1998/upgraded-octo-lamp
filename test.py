from v2i import V2I
from v2i.src.core.common import getAgentID
import matplotlib.pyplot as plt
import time
import numpy as np
from tqdm import tqdm
import random

path = "/home/mayank/Documents/upgraded-octo-lamp/experiments/onlyLocalView30m/configFiles/sim-config.yml"
obj = V2I.V2I(path, mode="test")
obj.seed(10)
#print(obj.observation_space.low)
#print(obj.observation_space.high)
#print(obj.observation_space)
print(obj.action_space)
print(obj.action_map)
densities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for i in range(0, 1):
    #print("Starting episode : %d"%(i+1))
    episodeDensity = []
    for i in range(0, 2):
        episodeDensity.append(random.choice(densities))

    state = obj.reset([0.3, 0.9])
    #print(state.shape)
    count = 0
    #time.sleep(0.5)
    
    for i in range(0, 3350):
        act = 0
        state, reward, done, info = obj.step(act)
        #time.sleep(100)
        #break
        #print(state.shape)
        if done:
            print("Collision")
            break