import random
import argparse
import numpy as np
from tqdm import tqdm
from huepy import info, red, bold
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from v2i import V2I
from v2i.src.core.common import readYaml, getAgentID, savePKL
from v2i.src.core.ppoController import ppoController

parser = argparse.ArgumentParser(description="v2i rollout script")
parser.add_argument("-n", "--num_episodes", default=10, type=int, help="number of episodes to run, (default: 10)")
parser.add_argument("-d", "--density", nargs='+', type=float, default=[], help="specify density list to intialize the episodes with, default will pick any density")
parser.add_argument("-l", "--episode-length", type=int, default=10000, help="maximum length of the episode, (default : 10000)")
parser.add_argument("-c", "--checkpoint-file", type=str, required=True, help="checkpoint file to load model from")
parser.add_argument("-tc", "--training-algo-config", type=str, required=True, help="algorithm configuration file")
parser.add_argument("-sc", "--sim-config", type=str, required=True, help="simulation configuration file")
parser.add_argument("-save", "--save-data", type=int, default=1, help="save policy data to disk (default:1)")
parser.add_argument("-dir", type=str, help="directory to dump data, valid only if -save is enabled")
parser.add_argument("-r", "--render-graphs", default=1, type=int, help="render predictions in real-time, default:0")

def getRandomDensity():
    densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    return random.sample(densities, k=1)[0]

def run_rollouts(args, env, fig, ax1, ax2):
    dataDict = {}
    densityList = args.density
    acts = list(env.actionEncoderDecoderHandler.actMap.values())
    
    # ---- Meta-data ----#
    dataDict["maxSpeed"] = env.simArgs.getValue("max-speed") * 3.6
    dataDict["plan-acts"] = env.actionEncoderDecoderHandler.planSpace
    dataDict["query-acts"] = env.actionEncoderDecoderHandler.querySpace
    dataDict["max-episode-length"] = args.episode_length
    dataDict["data"] = {}
    # ---- Meta-data ----#

    if len(densityList) == 0:
        for episode in range(0, args.num_episodes):
            densityList.append(getRandomDensity())

    for density in densityList:
        print("Running Simulation for %.1f density : "%(density))
        dataDict["data"][density] = {}

        for episode in tqdm(range(0, args.num_episodes)):
            dataDict["data"][density][episode] = {}
            dataDict["data"][density][episode]["speed"] = []
            dataDict["data"][density][episode]["rewards"] = []
            dataDict["data"][density][episode]["actions"] = []

            prev_state = env.reset(density)
            # Init variables
            lstm_state = [np.zeros(algoConfig["EXP_NAME"]["config"]["model"]["lstm_cell_size"]), np.zeros(algoConfig["EXP_NAME"]["config"]["model"]["lstm_cell_size"])]
            episodeReward = 0.0

            # Clear the figures at the start of new episode
            if fig is not None:
                plotX = []
                vf = []

            for step in range(0, args.episode_length):
                action, lstm_state, probs, vfPreds = controller.getAction(prev_state, lstm_state)
                
                #Update Graphs if and only render graphs is enabled
                if fig is not None:
                    plotX.append(step)
                    vf.append(vfPreds)
                    render(vf, probs, plotX, acts, fig, ax1, ax2)

                next_state, reward, done, info_dict = env.step(action)
                
                #--- Saving data ----#
                agentIDX = getAgentID(env.lane_map, env.agent_lane)
                dataDict["data"][density][episode]["speed"].append(env.lane_map[env.agent_lane][agentIDX]['speed'])
                dataDict["data"][density][episode]["rewards"].append(reward)
                dataDict["data"][density][episode]["actions"].append((env.planAct, env.queryAct))
                #--- Saving data ----#

                episodeReward += reward
                prev_state = next_state
                if done:
                    break
    return dataDict

def initRender():
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig.show()
    return fig, ax1, ax2

def render(vfPreds, probs, timeStep, acts, fig, ax1, ax2):
    ax1.plot(timeStep, vfPreds, color='red')
    ax1.set_ylabel("$V(s)$")
    ax2.set_ylim([0, 1.1])
    ax2.bar(np.arange(len(probs)), probs, color="blue")
    ax2.set_xticks(np.arange(len(probs)))
    ax2.set_xticklabels(acts)
    fig.canvas.update()
    plt.pause(0.001)

if __name__ == "__main__":
    
    # Parse arguments
    args = parser.parse_args()

    # Read config files
    algoConfig = readYaml(args.training_algo_config)

    # Build the ppo controller
    controller = ppoController(args.sim_config, algoConfig, args.checkpoint_file)

    # Warn user if data save is not enabled
    if(args.save_data != 1):
        print(info(bold(red("save is disabled, simulation data will not be saved to disk."))))
    
    # Local Env
    env = V2I.V2I(args.sim_config)

    # Init Render if enabled
    fig, ax1, ax2 = None, None, None
    if args.render_graphs == 1:
        fig, ax1, ax2 = initRender()

    # Start rolling out :)...
    simData = run_rollouts(args, env, fig, ax1, ax2)

    # Dump Data to file
    if(args.save_data == 1):
        savePKL(simData, args.dir+"/data.pkl")
    