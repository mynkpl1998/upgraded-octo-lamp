from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Ray Imports
import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env
from ray import tune

from v2i import V2I
from v2i.src.core.common import readYaml, raiseValueError
import argparse

parser = argparse.ArgumentParser(description="Training Script for v2i simulator")
parser.add_argument("-sc", "--sim-config", type=str, required=True, help="v2i simulation configuration file")
parser.add_argument("-tc", "--training-algo-config", required=True, type=str, help="training algorithm configuration file")
parser.add_argument("-w", "--num-workers", type=int, default=4, help="number of parallel worker to use for training, (default=4)")
parser.add_argument("-n", "--name", type=str, required=True, help="experiment name")
parser.add_argument("-f", "--checkpoint-freq", type=int, default=100, help="checkpoint frequency")

def doIMPALAEssentials(algoConfig, args):
    # Set Experiment Name
    algoConfig[args.name] = algoConfig.pop("EXP_NAME")
    # Set Number of Workers
    algoConfig[args.name]["config"]["num_workers"] = int(args.num_workers)
    # Set batch size
    algoConfig[args.name]["config"]["train_batch_size"] = int(args.num_workers) * algoConfig[args.name]["config"]["sample_batch_size"] * int(algoConfig[args.name]["config"]["num_envs_per_worker"])
    # Set Environment Name
    algoConfig[args.name]["env"] = "v2i-v0"
    # Set Algorithm Here
    algoConfig[args.name]["run"] = "IMPALA"
    # Set Checkpoint Frequency
    algoConfig[args.name]["checkpoint_freq"] = args.checkpoint_freq
    return algoConfig

def doPPOEssentials(algoConfig, simConfig, args):
    # Set Experiment Name
    algoConfig[args.name] = algoConfig.pop("EXP_NAME")
    # Set Number of Workers
    #algoConfig[args.name]["config"]["num_workers"] = int(args.num_workers)
    # Set batch size
    #if algoConfig[args.name]["config"]["train_batch_size"] == None:
    #    algoConfig[args.name]["config"]["train_batch_size"] = int(args.num_workers) * algoConfig[args.name]["config"]["sgd_minibatch_size"]
    # Set Environment Name
    algoConfig[args.name]["env"] = "v2i-v0"
    # Set Algorithm Here
    algoConfig[args.name]["run"] = "PPO"
    # Set Checkpoint Frequency
    algoConfig[args.name]["checkpoint_freq"] = args.checkpoint_freq
    # Enable/Disable memory use
    if simConfig['config']['enable-lstm']:
        algoConfig[args.name]['config']['model']['use_lstm'] = True
    else:
        algoConfig[args.name]['config']['model']['use_lstm'] = False
    
    return algoConfig

if __name__ == "__main__":
    args = parser.parse_args()

    # Read Config Files
    algoConfig = readYaml(args.training_algo_config)
    simConfig = readYaml(args.sim_config)
    
    # Set essentials
    trainAlgo = args.training_algo_config.split("/")[-1].split('-')[0].upper()
    if trainAlgo == 'PPO':
        algoConfig = doPPOEssentials(algoConfig, simConfig, args)
    elif trainAlgo == 'IMPALA':
        algoConfig == doIMPALAEssentials(algoConfig, args)
    else:
        raiseValueError("invalid training algo %s"%(trainAlgo))

    # Register Environment
    register_env("v2i-v0", lambda config: V2I.V2I(args.sim_config, "train"))

    # Start the training
    ray.init()
    run_experiments(algoConfig)