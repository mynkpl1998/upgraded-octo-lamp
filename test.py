from v2i import V2I
from v2i.src.core.common import getAgentID
import matplotlib.pyplot as plt
import time
import numpy as np
from tqdm import tqdm
import random
np.random.seed(10)
random.seed(10)

path = "/home/mayank/Documents/upgraded-octo-lamp/experiments/onlyLocalView30m/configFiles/sim-config.yml"
obj = V2I.V2I(path, mode="test")
obj.seed(10)
#print(obj.observation_space.low)
#print(obj.observation_space.high)
#print(obj.observation_space)
print(obj.action_space)
print(obj.action_map)
densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for episode in range(0, 1000):
    #print("Starting episode : %d"%(i+1))
    episodeDensity = []
    for i in range(0, 2):
        episodeDensity.append(random.choice(densities))
    print("Episode Density : ", episodeDensity)
    state = obj.reset([0.5, 0.5])

    #print(state.shape)
    count = 0
    #time.sleep(0.5)
    print(episode)
    for i in range(0, 10000):
        act = 2
        state, reward, done, info = obj.step(act)
        #time.sleep(100)
        #break
        #print(state.shape)
        if done:
            print("Collision")
            break