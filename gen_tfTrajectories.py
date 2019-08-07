import pickle
import argparse
import numpy as np

from v2i.src.core.tfLights import RandomIntervalGenerator
from v2i.src.core.common import raiseValueError, savePKL

'''
This script is used to generate valid traffic light switching points and stored it in a pickable 
file which can be used during simulation.
'''

parser = argparse.ArgumentParser(description="generate traffic light trajectories")
parser.add_argument("-max", "--max-duration", default=30, type=int, help="maximum duration of red light (seconds), default : 30s")
parser.add_argument("-min", "--min-duration", default=15, type=int, help="minimum duration of red light (seconds), default : 15s")
parser.add_argument("-t", "--time-period", default=0.1, type=float, help="time period of simulations (seconds), default: 0.1s (10 Hz)")
parser.add_argument("-tries", "--num-tries", default=10, type=int, help="number of tries")
parser.add_argument("-l", "--horizon", default=10000, type=int, help="max length of episode")
parser.add_argument("-pts", "--num-pts", default=10, type=int, help="max number of points in each trajectory, default: 10")
parser.add_argument("-n", "--num-trajecs", default=20, type=int, help="number of trajectories to generated, default: 20")

def fillmetaData(args):
    trajectDict = {}
    trajectDict["metadata"] = {}
    trajectDict["metadata"]["max"] = args.max_duration
    trajectDict["metadata"]["min"] = args.min_duration
    trajectDict["metadata"]["time-period"] = args.time_period
    trajectDict["metadata"]["num-tries"] = args.num_tries
    trajectDict["metadata"]["horizon"] = args.horizon
    trajectDict["metadata"]["pts"] = args.num_pts
    return trajectDict

def fillTrajectories(numTrajecs, args, genObj, trajecDict):
    trajecDict["data"] = []
    for trajec in range(0, numTrajecs):
        genObj.reset()
        trajecDict["data"].append(genObj.gen())
    trajecDict["numTrajecs"] = len(trajecDict["data"])
    return trajecDict

if __name__ == "__main__":

    # Parser Command Line Arguments
    args = parser.parse_args()

    # Generator Object -> wrong fix min_duration/time-period
    max_dur_steps = int(args.max_duration / args.time_period)
    min_dur_steps = int(args.min_duration / args.time_period)
    gen = RandomIntervalGenerator(min_dur_steps, max_dur_steps, args.num_pts, 1, args.horizon, args.num_tries)
    
    # Create dict to store data & fill meta data
    tfTrajectDict = fillmetaData(args)

    # Fill Trajectories
    tfTrajectDict = fillTrajectories(args.num_trajecs, args, gen, tfTrajectDict)

    # Dump the dict to a file.
    savePKL(tfTrajectDict, "v2i/src/data/tftrajec.pkl")

