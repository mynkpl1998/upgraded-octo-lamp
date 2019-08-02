from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Ray Imports
import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env
from ray import tune

from v2i import V2I
from v2i.src.core.common import readYaml
import argparse

parser = argparse.ArgumentParser(description="Training Script for v2i simulator")
parser.add_argument("-sc", "--sim-config", type=str, required=True, help="v2i simulation configuration file")
parser.add_argument("-tc", "--training-algo-config", required=True, type=str, help="training algorithm configuration file")
parser.add_argument("-w", "--num-workers", type=int, default=4, help="number of parallel worker to use for training, (default=4)")
parser.add_argument("-n", "--name", type=str, required=True, help="experiment name")
parser.add_argument("-f", "--checkpoint-freq", type=int, default=100, help="checkpoint frequency")

def doEssentials(algoConfig, args):
    # Set Experiment Name
    algoConfig[args.name] = algoConfig.pop("EXP_NAME")
    # Set Number of Workers
    algoConfig[args.name]["config"]["num_workers"] = int(args.num_workers)
    # Set batch size
    algoConfig[args.name]["config"]["train_batch_size"] = int(args.num_workers) * algoConfig[args.name]["config"]["sgd_minibatch_size"]
    # Set Environment Name
    algoConfig[args.name]["env"] = "v2i-v0"
    # Set Algorithm Here
    algoConfig[args.name]["run"] = "PPO"
    # Set Checkpoint Frequency
    algoConfig[args.name]["checkpoint_freq"] = args.checkpoint_freq
    return algoConfig

if __name__ == "__main__":
    args = parser.parse_args()

    # Read Config Files
    algoConfig = readYaml(args.training_algo_config)

    # Set Essentials
    algoConfig = doEssentials(algoConfig, args)

    # Register Environment
    register_env("v2i-v0", lambda config: V2I.V2I(args.sim_config))

    # Start the training
    ray.init()
    run_experiments(algoConfig)
    