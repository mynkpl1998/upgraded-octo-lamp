from v2i import V2I
import matplotlib.pyplot as plt
import pickle
import time
import numpy as np
from tqdm import tqdm
import random
np.random.seed(10)
random.seed(10)

path = "/home/mayank/Documents/upgraded-octo-lamp/experiments/fullComm40m/configFiles/sim-config.yml"
obj = V2I.V2I(path, mode="test")
#obj.seed(10)
#print(obj.observation_space.low)
#print(obj.observation_space.high)
#print(obj.observation_space)
print(obj.action_space)
print(obj.action_map)
densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

maxSteps = 3350

