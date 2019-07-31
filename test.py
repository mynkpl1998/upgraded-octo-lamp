from v2i import V2I
from v2i.src.core.common import getAgentID

import time
path = "/home/mayank/Documents/upgraded-octo-lamp/examples/LocalView/config.yml"
obj = V2I.V2I(path)
obj.seed(10)
#print(obj.observation_space.low)
#print(obj.observation_space.high)
#print(obj.observation_space)
print(obj.action_space)
print(obj.action_map)

for i in range(0, 100):
    print("Starting episode : %d"%(i+1))
    obj.reset(0.2)
    count = 0
    while True:
        
        if count < 10:
            act = 0
        else:
            act = obj.action_space.sample()
            if act == 1:
                act = 2

        '''
        agentIDX = getAgentID(obj.lane_map, obj.agent_lane)
        if obj.lane_map[obj.agent_lane][agentIDX]['speed'] == 0.0:
            print("Yes")
        '''

        state, reward, done, info = obj.step(act)
    
        count += 1

        if done:
            break