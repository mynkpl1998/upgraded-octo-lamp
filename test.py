from v2i import V2I

path = "/home/mayank/Documents/upgraded-octo-lamp/examples/LocalView/config.yml"
obj = V2I.V2I(path)
obj.seed(10)
#print(obj.observation_space.low)
#print(obj.observation_space.high)
#print(obj.observation_space)
print(obj.action_space)
print(obj.action_map)

for i in range(0, 1):
    state = obj.reset(0.2)

for i in range(0, 100000):
    act = obj.action_space.sample()
    state, reward, gameOver, info = obj.step(act)