import pickle
import argparse
import numpy as np

import v2i.src.core.constants as constants
from v2i.src.core.common import savePKL

MAX_ITERS = 1000

'''

This scripts generates the bunch of starting positions for the different densities of the vehicles.
This predifined generation of trajectories reduces the overhead of running a while loop at the start of new episode.
Note : Script automatically picks the constants from the src.core.constant.py

'''

parser = argparse.ArgumentParser(description="generate starting trajectories for different densities")
parser.add_argument("--file-name", default="v2i/src/data/trajec.pkl", type=str, help="output path with file name")
parser.add_argument("--num-trajec", default=2, type=int, help="num of different trajectories to generate")

def calMinAngle():
    return np.rad2deg(((constants.MIN_DISTANCE * constants.SCALE)/constants.LANE_RADIUS))

def loop(density, numcars, minAngle):
    
    if numcars < 2:
        raise ValueError("At least two cars are required, Failed for %d density with %d cars"%(density, numcars))
    
    trajecs = []

    for iter in range(0, numcars):
        
        if(len(trajecs) < 1):
            loc = np.random.uniform(0, 352)
            trajecs.append(loc)
        else:
            loc = np.random.uniform(0, 352)
            generated = False
            count = 0
            while (generated != True):

                count += 1
                res = True
                generated = False

                for trajec in trajecs:
                    max_angle = max(loc, trajec)
                    min_angle = min(loc, trajec)

                    diff = max_angle - min_angle

                    if diff > minAngle:
                        pass
                    else:
                        res = res and False
                
                if res:
                    generated = True
                    trajecs.append(loc)
                else:
                    generated = False
                    loc = np.random.uniform(0, 352)
                
                if count > MAX_ITERS:
                    break
    return trajecs

    

def generate(densities, numcars, minAngle, numTrajec=1):
    trajecDict = {}
    for index, density in enumerate(densities):
        trajecDict[density] = []
        for num in range(0, numTrajec):
            trajecDict[density].append(loop(density, numcars[index], minAngle))   
    return trajecDict
    



if __name__ == "__main__":

    # Parse Arguments
    args = parser.parse_args()

    # Calculate min gap angle between two consecutive vehicles
    minAngle = calMinAngle()

    # Discretized densities
    densities = np.linspace(0.1, 1, num=10)

    # Num of cars for each density
    numcars = densities * constants.MAX_CARS_IN_LANE
    numcars = numcars.astype('int')

    trajecDict = generate(densities, numcars, minAngle, args.num_trajec)

    # Save the Data
    savePKL(trajecDict, args.file_name)