from v2i import V2I
from v2i.src.core.common import getAgentID
import matplotlib.pyplot as plt
import time
import numpy as np
from tqdm import tqdm

path = "/home/mayank/Documents/upgraded-octo-lamp/examples/WithFullCommunication/config.yml"
obj = V2I.V2I(path, mode="test")
obj.seed(10)
#print(obj.observation_space.low)
#print(obj.observation_space.high)
#print(obj.observation_space)
print(obj.action_space)
print(obj.action_map)

for i in tqdm(range(0, 1)):
    #print("Starting episode : %d"%(i+1))
    state = obj.reset(0.2)
    print(state.shape)
    count = 0

    for i in range(0, 3350):
        act = 0
        state, reward, done, info = obj.step(act)
        print(state.shape)
        if done:
            #break
            pass
'''
print(info)
plt.bar(np.arange(0, len(info.keys())), list(info.values()))
plt.xticks(np.arange(0, len(info.keys())), list(info.keys()), rotation='vertical')
plt.show()
'''
