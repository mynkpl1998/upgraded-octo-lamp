import argparse
from v2i import V2I
from v2i.src.core.common import readYaml, getAgentID, savePKL, raiseValueError, getAgentID, arcLength
from v2i.src.core.ppoController import ppoController
from v2i.src.core.impalaController import impalaController

parser = argparse.ArgumentParser(description="saliency maps script")
parser.add_argument("-d", "--density", nargs='+', type=str, default=[], help="specify density list to intialize the episodes with, default will pick any density")
parser.add_argument("-l", "--episode-length", type=int, default=10000, help="maximum length of the episode, (default : 10000)")
parser.add_argument("-c", "--checkpoint-file", type=str, required=True, help="checkpoint file to load model from")
parser.add_argument("-tc", "--training-algo-config", type=str, required=True, help="algorithm configuration file")
parser.add_argument("-sc", "--sim-config", type=str, required=True, help="simulation configuration file")
parser.add_argument("-dir", type=str, help="directory to dump data, valid only if -save is enabled")
parser.add_argument("-tf", "--enable-tf", default=0, type=int, help="enable/disbale traffic lights")
parser.add_argument("-sr", "--render-screen", default=0, type=int, help="enable/disable environment screen rendering")

if __name__ == "__main__":

    # Parse arguments
    args = parser.parse_args()

    # Read config files
    algoConfig = readYaml(args.training_algo_config)
    simConfig = readYaml(args.sim_config)

    # Build the ppo controller
    trainAlgo = args.training_algo_config.split("/")[-1].split('-')[0].upper()
    print(trainAlgo)
    if trainAlgo == "IMPALA":
        controller = impalaController(args.sim_config, algoConfig, args.checkpoint_file)
    elif trainAlgo == "PPO":
        controller = ppoController(args.sim_config, algoConfig, args.checkpoint_file)
    else:
        raiseValueError("invalid training algo %s"%(trainAlgo))

    