from collections import deque
import numpy as np

class obsWrapper:

    def __init__(self, k, obsSize):
        self.k = k
        self.obsQueue = deque(maxlen=self.k)
        self.obsSize = obsSize
    
    def resetQueue(self):
        zerosObs = np.zeros((self.k, self.obsSize))
        for obs in zerosObs:
            self.obsQueue.append(obs.copy())
    
    def getObs(self):
        return np.array(self.obsQueue).flatten().copy()
    
    def addObs(self, obs):
        #print(obs.shape)
        assert obs.shape[0] == self.obsSize
        self.obsQueue.append(obs.copy())

if __name__ == "__main__":
    size = 3
    testObj = obsWrapper(3, size)
    testObj.resetQueue()
    obs = testObj.getObs()
    print(testObj.getObs())
    obs1 = np.random.rand(size)
    obs2 = np.random.rand(size)
    obs3 = np.random.rand(size)
    obs4 = np.random.rand(size) * 100
    testObj.addObs(obs1)
    print(testObj.getObs())

    testObj.addObs(obs2)
    print(testObj.getObs())

    testObj.addObs(obs3)
    print(testObj.getObs())

    testObj.addObs(obs4)
    print(testObj.getObs())

