import argparse
import numpy as np

from v2i import V2I
from v2i.src.core.common import readYaml
from v2i.src.core.ppoController import ppoController

parser = argparse.ArgumentParser(description="v2i rollout script")
parser.add_argument("-n", "--num_episodes", default=10, type=int, help="number of episodes to run, (default: 10)")
parser.add_argument("-d", "--density", type=float, default=None, help="specify density to intialize the episodes with")
parser.add_argument("-l", "--episode-length", type=int, default=10000, help="maximum length of the episode, (default : 10000)")
parser.add_argument("-c", "--checkpoint-file", type=str, required=True, help="checkpoint file to load model from")
parser.add_argument("-tc", "--training-algo-config", type=str, required=True, help="algorithm configuration file")
parser.add_argument("-sc", "--sim-config", type=str, required=True, help="simulation configuration file")

if __name__ == "__main__":
    
    # Parse arguments
    args = parser.parse_args()

    # Read config files
    algoConfig = readYaml(args.training_algo_config)

    # Build the ppo controller
    controller = ppoController(args.sim_config, algoConfig, args.checkpoint_file)

    # Local Env
    env = V2I.V2I(args.sim_config)

    print(env.action_map)

    # Loop
    for episode in range(0, args.num_episodes):
        
        # Check for specific density
        if args.density != None:
            prev_state = env.reset(args.density)
        else:
            prev_state = env.reset()
        
        # Set lstm state
        lstm_state = lstm_state = [np.zeros(algoConfig["EXP_NAME"]["config"]["model"]["lstm_cell_size"]), np.zeros(algoConfig["EXP_NAME"]["config"]["model"]["lstm_cell_size"])]
        '''
        if algoConfig["EXP_NAME"]["config"]["model"]["use_lstm"]:
            
        '''
        episode_reward = 0.0
        for step in range(0, args.episode_length):
            action, lstm_state = controller.getAction(prev_state, lstm_state)
            next_state, reward, done, info_dict = env.step(action)

            episode_reward += reward
            prev_state = next_state

            if done:
                break
        
        print("Episode %d, Total Reward : %.2f"%(episode + 1, episode_reward))