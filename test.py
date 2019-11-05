from v2i import V2I
from v2i.src.core.common import getAgentID
import matplotlib.pyplot as plt
import time
import numpy as np
from tqdm import tqdm
import random
np.random.seed(10)
random.seed(10)

path = "experiments/onlyLocalView15m/configFiles/sim-config.yml"
obj = V2I.V2I(path, mode="test")
obj.seed(10)
#print(obj.observation_space.low)
#print(obj.observation_space.high)
#print(obj.observation_space)
print(obj.action_space)
print(obj.action_map)
densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

maxSteps = 1200

simDensities = [0.4]
fig, ax = plt.subplots(nrows=2, ncols=3)
finalSpeedList = []

for density in simDensities:
    print("Running for density : ", density)
    speedDict = {}
    for episode in range(0, 1):
        episodeDensity = [density, density]
        state = obj.reset(episodeDensity)

        for lane in range(0, 2):
            for car in obj.lane_map[lane]:
                carID = car['id']
                carSpeed = car['speed']
                speedDict[carID] = []
                speedDict[carID].append(carSpeed)

        for i in range(0, maxSteps):
            obj.step(0)

            for lane in range(0, 2):
                for car in obj.lane_map[lane]:
                    carID = car['id']
                    speedDict[carID].append(car['speed'])
    finalSpeedList.append(speedDict)

for i, r in enumerate(ax):
    for j, c in enumerate(r):
        index = 0
        for car in finalSpeedList[index].keys():
            c.plot(finalSpeedList[index][car])
        c.set_ylim([0, 11.11])
        c.set_xlabel('time')
        c.set_ylabel('speed (m/s)')
        c.set_title('Traffic Density : %.1f'%(simDensities[index]))

plt.show()


'''
for episode in range(0, 1):
    #print("Starting episode : %d"%(i+1))
    episodeDensity = []
    for i in range(0, 2):
        episodeDensity.append(random.choice(densities))
    print("Episode Density : ", episodeDensity)
    state = obj.reset() 
    #print(state.shape)
    count = 0
    #time.sleep(0.5)
    
    for i in range(0, maxSteps):
        act = 2
        state, reward, done, info = obj.step(0)
        #print(obj.idmHandler.getAllElementbyKeys('speed', obj.lane_map[0]))
        #time.sleep(100)
        #break
        #print(state.shape)
        if done:
            print("Collision")
            break
'''