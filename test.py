from v2i import V2I
from v2i.src.core.common import getAgentID

import time
path = "/home/mayank/Documents/upgraded-octo-lamp/examples/WithFullCommunication/config.yml"
obj = V2I.V2I(path, mode="test")
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
        
        if count < 100:
            act = 0
        else:
            '''
            act = obj.action_space.sample()
            if act == 1:
                act = 2
            '''
            act = obj.action_space.sample()
        '''
        agentIDX = getAgentID(obj.lane_map, obj.agent_lane)
        if obj.lane_map[obj.agent_lane][agentIDX]['speed'] == 0.0:
            print("Yes")
        '''

        state, reward, done, info = obj.step(act)
    
        count += 1
        
        if count > 1000:
            break

        if done:
            print(obj.action_map[act])
            break