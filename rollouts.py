import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from huepy import info, red, bold
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import v2i.src.core.constants as constants

from v2i import V2I
from v2i.src.core.common import readYaml, getAgentID, savePKL, raiseValueError, getAgentID, arcLength
from v2i.src.core.ppoController import ppoController
from v2i.src.core.impalaController import impalaController


parser = argparse.ArgumentParser(description="v2i rollout script")
parser.add_argument("-n", "--num_episodes", default=10, type=int, help="number of episodes to run, (default: 10)")
parser.add_argument("-d", "--density", nargs='+', type=str, default=[], help="specify density list to intialize the episodes with, default will pick any density")
parser.add_argument("-l", "--episode-length", type=int, default=10000, help="maximum length of the episode, (default : 10000)")
parser.add_argument("-c", "--checkpoint-file", type=str, required=True, help="checkpoint file to load model from")
parser.add_argument("-tc", "--training-algo-config", type=str, required=True, help="algorithm configuration file")
parser.add_argument("-sc", "--sim-config", type=str, required=True, help="simulation configuration file")
parser.add_argument("-save", "--save-data", type=int, default=1, help="save policy data to disk (default:1)")
parser.add_argument("-dir", type=str, help="directory to dump data, valid only if -save is enabled")
parser.add_argument("-r", "--render-graphs", default=1, type=int, help="render predictions in real-time, default:0")
parser.add_argument("-tf", "--enable-tf", default=0, type=int, help="enable/disbale traffic lights")
parser.add_argument("-sr", "--render-screen", default=0, type=int, help="enable/disable environment screen rendering")

def getRandomDensity():
    densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    return random.sample(densities, k=1)[0]

def getBum2BumDist(laneMap, agentLane, agentIDX):
    if agentIDX == 0:
        angleDiff = laneMap[agentLane][-1]['pos'] - laneMap[agentLane][0]['pos'] 
    else:
        angleDiff = laneMap[agentLane][agentIDX-1]['pos'] - laneMap[agentLane][agentIDX]['pos']
    angleDiff %= 360
    return (arcLength(constants.LANE_RADIUS[agentLane], angleDiff) / constants.SCALE) - constants.CAR_LENGTH

def str2Density(argsStr):
    densities = []
    for density in argsStr:
        l1, l2 = density.split(",")
        epsiodeDensity = [float(l1), float(l2)]
        densities.append(epsiodeDensity)
    return densities

def run_rollouts(args, env, fig, ax1, ax2, useLstm):
    dataDict = {}
    densityList = str2Density(args.density)
    acts = list(env.actionEncoderDecoderHandler.actMap.values())
    
    # ---- Meta-data ----#
    dataDict["maxSpeed"] = env.simArgs.getValue("max-speed") * 3.6
    dataDict["maxViewSpeed"] = env.idmHandler.nonEgoMaxVel * 3.6
    dataDict["plan-acts"] = env.actionEncoderDecoderHandler.planSpace
    dataDict["query-acts"] = env.actionEncoderDecoderHandler.querySpace
    dataDict["max-episode-length"] = args.episode_length
    dataDict["data"] = {}
    dataDict["others"] = {}
    # ---- Meta-data ----#

    if len(densityList) == 0:
        for episode in range(0, args.num_episodes):
            densityList.append(getRandomDensity())

    for density in densityList:
        print("Running Simulation for , lane 0 : %f, lane 1 : %f densities : "%(density[0], density[1]))
        densityStr = str(density[0]) + "_" + str(density[1])
        dataDict["data"][densityStr] = {}
        dataDict["others"][densityStr] = {}
        dataDict["others"][densityStr]["collision-count"] = 0

        for episode in tqdm(range(0, args.num_episodes)):

            dataDict["data"][densityStr][episode] = {}
            dataDict["data"][densityStr][episode]["speed"] = {}
            dataDict["data"][densityStr][episode]['Pos'] = []
            dataDict["data"][densityStr][episode]["rewards"] = []
            dataDict["data"][densityStr][episode]["actions"] = []
            dataDict["data"][densityStr][episode]["bum2bumdist"] = []
            dataDict["data"][densityStr][episode]["EgoMaxSpeed"] = -10
            dataDict['data'][densityStr][episode]['agentCarID'] = None
            dataDict['data'][densityStr][episode]['frontDiff'] = []
            dataDict['data'][densityStr][episode]['backDiff'] = []

            
            prev_state = env.reset(density)
            agentIDX = getAgentID(env.lane_map, env.agent_lane)

            '''
            dataDict['data'][densityStr][episode]['agentCarID'] = env.lane_map[env.agent_lane][agentIDX]['id']
            dataDict['data'][densityStr][episode]['Pos'].append(env.carPos.copy())
            for lane in range(0, 2):
                for car in env.lane_map[lane]:
                    carId = car['id']
                    carSpeed = car['speed']
                    #carPos = car['pos']
                    dataDict["data"][densityStr][episode]['speed'][carId] = []
                    #dataDict["data"][densityStr][episode]['pos'][carId] = []
                    dataDict["data"][densityStr][episode]['speed'][carId].append(carSpeed)
                    #dataDict["data"][densityStr][episode]['pos'][carId].append(carPos)
            '''

            # Init variables
            if useLstm:
                lstm_state = [np.zeros(algoConfig["EXP_NAME"]["config"]["model"]["lstm_cell_size"]), np.zeros(algoConfig["EXP_NAME"]["config"]["model"]["lstm_cell_size"])]
            else:
                lstm_state = None
            episodeReward = 0.0

            # Clear the figures at the start of new episode
            if fig is not None:
                ax1.cla()
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
                
                # Collision Count
                if env.collision == True:
                    dataDict["others"][densityStr]["collision-count"] += 1
                
                # Calculate agent IDX
                localLaneMap = env.lane_map.copy()
                localLaneMap[env.agent_lane] = np.sort(localLaneMap[env.agent_lane], order=['pos'])[::-1]
                agentIDX = getAgentID(localLaneMap, env.agent_lane)

                # Calculate bumber to bumber distance
                bum2bumdist = getBum2BumDist(localLaneMap, env.agent_lane, agentIDX)
                #--- Saving data ----#
                '''
                agentIDX = getAgentID(env.lane_map, env.agent_lane)
                for lane in range(0, 2):
                    for car in env.lane_map[lane]:
                        carID = car['id']
                        carSpeed = car['speed']
                        #carPos = car['pos']
                        dataDict['data'][densityStr][episode]['speed'][carID].append(carSpeed)
                        #dataDict['data'][densityStr][episode]['pos'][carID].append(carPos)
                dataDict['data'][densityStr][episode]['Pos'].append(env.carPos.copy())
                #dataDict['data'][densityStr][episode]['frontDiff'].append(env.front_diff)
                #dataDict['data'][densityStr][episode]['backDiff'].append(env.back_diff)
                '''
                 
                #dataDict["data"][densityStr][episode]["speed"].append(env.lane_map[env.agent_lane][agentIDX]['speed'])
                dataDict["data"][densityStr][episode]["rewards"].append(reward)
                dataDict["data"][densityStr][episode]["actions"].append((env.planAct, env.queryAct))
                dataDict["data"][densityStr][episode]["EgoMaxSpeed"] = max(dataDict["data"][densityStr][episode]["EgoMaxSpeed"], env.lane_map[env.agent_lane][agentIDX]['speed'])
                dataDict["data"][densityStr][episode]['bum2bumdist'].append(bum2bumdist)
                #--- Saving data ----#

                episodeReward += reward
                prev_state = next_state
                if done:
                    break
    return dataDict

def initRender():
    fig = plt.figure(figsize=(4, 6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig.show()
    return fig, ax1, ax2

def render(vfPreds, probs, timeStep, acts, fig, ax1, ax2):
    ax1.plot(timeStep, vfPreds, color='red')
    ax1.set_ylabel("$V(s)$")
    ax2.cla()
    ax2.set_ylim([0, 1.1])
    ax2.bar(np.arange(len(probs)), probs, color="blue")
    ax2.set_xticks(np.arange(len(probs)))
    ax2.set_xticklabels(acts, rotation='vertical')
    fig.canvas.update()
    plt.pause(0.001)

def paramDict(args):
    envParams = {}
    if args.render_screen == 1:
        envParams["render"] = True
    else:
        envParams["render"] = False
    
    if args.enable_tf == 1:
        envParams["enable-tf"] = True
    else:
        envParams["enable-tf"] = False
    return envParams

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

    # Warn user if data save is not enabled
    if(args.save_data != 1):
        print(info(bold(red("save is disabled, simulation data will not be saved to disk."))))
    
    # Local Env
    env = V2I.V2I(args.sim_config, "test", paramDict(args))

    # Init Render if enabled
    fig, ax1, ax2 = None, None, None
    if args.render_graphs == 1:
        fig, ax1, ax2 = initRender()
    
    # Use LSTM if enabled by sim-config file
    useLstm = False
    if simConfig["config"]["enable-lstm"]:
        useLstm = True

    # Start rolling out :)...
    #time.sleep(7)
    simData = run_rollouts(args, env, fig, ax1, ax2, useLstm)

    # Dump Data to file
    if(args.save_data == 1):
        fileName = None
        if args.enable_tf == 1:
            fileName = "data_tf_enabled.pkl"
        else:
            fileName = "data_tf_disabled.pkl"
        savePKL(simData, args.dir + "/" + fileName)
    
