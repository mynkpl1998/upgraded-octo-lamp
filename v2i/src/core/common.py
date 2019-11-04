import os
import yaml
import pickle
import numpy as np
from huepy import bad, bold, red

from v2i.src.core.constants import IDM_CONSTS, LANES

def checkFileExists(file):
    return os.path.isfile(file)

def readYaml(file):
    with open(file, "r") as handle:
        configDict = yaml.load(handle, Loader=yaml.FullLoader)
    return configDict

def raiseValueError(msg):
    raise ValueError(bad(bold(red(msg))))

def savePKL(data, filenamePath):
    '''

    Function : saves the data object in a serialized format at the given path
    
    Input Args : 
        1. data - the data object which needs to be serialized.
        2. filenamePath - the complete path to the location to save including file name

    Return Args:
        returns True if successfully saved else return False
    
    '''
    
    with open(filenamePath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    return False

def loadPKL(filenamePath):
    
    '''

    Function : load the serialized object from the given file.
    
    Input Args : 
        1. filenamePath - absolute path of the file to load

    Return Args:
        returns the loaded object if successfully loaded else raise value error with exception as the message.
    
    '''

    try:
        with open(filenamePath, "rb") as handle:
            trajecDict = pickle.load(handle)
        return trajecDict
    except Exception as e:
        raiseValueError(e)

def arcAngle(radius, arcLength):
    return np.rad2deg(arcLength/radius)

def getAgentID(laneMap, agentLane):
    agentID = np.where(laneMap[agentLane]['agent'] == 1)[0]
    return agentID[0]

def getTfID(laneMap, lane):
    TfID = np.where(laneMap[lane]['agent'] == 2)[0]
    return TfID[0]

def arcLength(radius, arcAngleDeg):
    return np.deg2rad(arcAngleDeg) * radius

def reverseDict(dictToReverse):
    return dict(map(reversed, dictToReverse.items()))

def buildDictWithKeys(keys, initValue):
    d = {}
    for key in keys:
        d[key] = initValue
    return d.copy()

def mergeDicts(d1, d2):
    d = {}
    for key in d1.keys():
        d[key] = d1[key]
    
    for key in d2.keys():
        d[key] = d2[key]
    
    assert len(d) == len(d1) + len(d2)
    
    return d

def getMaxSpeed(speed, deceleration, distance):
    return np.sqrt(2 * deceleration * distance)

def buildMaxSpeeds(extendedViewinMetres, localViewinMetres):
    extendedViewinMetres /= 2.0
    size = extendedViewinMetres/4.0
    possibleDist = [size, 2*size, 3*size , 4*size]
    possibleMaxSpeeds = []
    for dist in possibleDist:
        possibleMaxSpeeds.append(getMaxSpeed(0.0, IDM_CONSTS['DECELERATION_RATE'], dist))
    return possibleMaxSpeeds

def randomizeSpeeds(laneMap, maxSpeeds):
    for lane in range(0, LANES):
        for car in laneMap[lane]:
            randomSpeedIndex = np.random.randint(0, len(maxSpeeds))
            randomMaxSpeed = maxSpeeds[randomSpeedIndex]
            car['speed'] = randomMaxSpeed
    print("Switched")
    return laneMap
