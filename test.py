from v2i import V2I
from v2i.src.core.common import getAgentID, savePKL
import matplotlib.pyplot as plt
import pickle
import time
import numpy as np
from tqdm import tqdm
import random
np.random.seed(10)
random.seed(10)

path = "/home/mayank/Documents/upgraded-octo-lamp/experiments/fullComm40m-comm-penalty/configFiles/sim-config.yml"
obj = V2I.V2I(path, mode="test")
obj.seed(10)
#print(obj.observation_space.low)
#print(obj.observation_space.high)
#print(obj.observation_space)
print(obj.action_space)
print(obj.action_map)
densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

maxSteps = 3350

'''
simDensities = [0.4]
fig, ax = plt.subplots(nrows=2, ncols=3)
finalSpeedList = []


for density in simDensities:
    print("Running for density : ", density)
    speedDict = {}
    for episode in range(0, 50):
        speedDict[episode] = {}
        episodeDensity = [density, density]
        state = obj.reset(episodeDensity)
        print(episode)
        for lane in range(0, 2):
            for car in obj.lane_map[lane]:
                carID = car['id']
                carSpeed = car['speed']
                speedDict[episode][carID] = []
                speedDict[episode][carID].append(carSpeed)
        #print(speedDict)
        for i in range(0, maxSteps):
            obj.step(0)

            for lane in range(0, 2):
                for car in obj.lane_map[lane]:
                    carID = car['id']
                    speedDict[episode][carID].append(car['speed'])
    finalSpeedList.append(speedDict)

savePath = "/home/mayank/ray_results/homogeneous-no-reaction-time-only-local-view-15m/idmData.pkl"
savePKL(finalSpeedList, savePath)


for i, r in enumerate(ax):
    for j, c in enumerate(r):
        index = 0
        for car in finalSpeedList[index].keys():
            c.plot(np.array(finalSpeedList[0][index][car]) * 3.6)
        #c.set_ylim([0, 11.11])
        c.set_xlabel('time')
        c.set_ylabel('speed (m/s)')
        c.set_title('Traffic Density : %.1f'%(simDensities[index]))

plt.show()
'''

maxSteps = 10000
numEpisodes = 1
ageData = np.zeros((numEpisodes, maxSteps))

def avgVehSpeed(laneMap):
    speeds = []
    for lane in range(0, 2):
        for veh in laneMap[lane]:
            speeds.append(veh['speed'])
    return np.array(speeds).mean()

def appendAge(trueAgeRegister, ageValuesTrack):
    for sensorID in trueAgeRegister.keys():
        ageValuesTrack[sensorID].append(trueAgeRegister[sensorID])
    return ageValuesTrack

#time.sleep(20)
epSpeed = []
ageValuesTrack = {}

for episode in range(0, numEpisodes):
    carSpeeds = []
    print(episode)
    #print("Starting episode : %d"%(i+1))
    for sensorID in obj.trueAgeHandler.trueAgeRegister:
        ageValuesTrack[sensorID] = []
    
    state = obj.reset()
    carSpeeds.append(avgVehSpeed(obj.lane_map))
    ageValuesTrack = appendAge(obj.trueAgeHandler.trueAgeRegister, ageValuesTrack)

    obj.trueAgeHandler.getMeanAge()

    for i in range(0, maxSteps):
        state, reward, done, info = obj.step(8)
        carSpeeds.append(avgVehSpeed(obj.lane_map))
        ageData[episode][i] = obj.trueAgeHandler.getMeanAge()
        ageValuesTrack = appendAge(obj.trueAgeHandler.trueAgeRegister, ageValuesTrack)

        if done:
            print("Collision")
            break
    carSpeeds = np.array(carSpeeds)
    epSpeed.append(carSpeeds.mean())

epSpeed = np.array(epSpeed)

print("True Age mean : ", ageData.mean(axis=1).mean(), "Speed : ", epSpeed.mean())

zoomSteps = 10000

sampleSensors = random.sample(list(obj.trueAgeHandler.trueAgeRegister.keys()), 10)

for sensorID in sampleSensors:
    plt.step(np.arange(0, len(ageValuesTrack[sensorID]))[0:zoomSteps], ageValuesTrack[sensorID][0:zoomSteps], label="sen. %d"%(sensorID))
    plt.xlabel("time")
    plt.ylabel("avg. age of sensors")

unwantedSensorsID = []
newRegister = {}

for sensorID in obj.trueAgeHandler.trueAgeRegister:
    age = obj.trueAgeHandler.trueAgeRegister[sensorID]
    if age > 300:
        unwantedSensorsID.append(sensorID)



plt.title("Num. Sensors %d"%(len(obj.trueAgeHandler.trueAgeRegister.keys())))
plt.legend()
plt.savefig("sensors.png")