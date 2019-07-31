import argparse
import pygame
from huepy import blue, bold

from v2i import V2I
from v2i.src.core.common import raiseValueError, reverseDict

parser = argparse.ArgumentParser(description="Script to control planing actions using keyboard")
parser.add_argument("-sc", "--sim-config", type=str, help="path to simulation configuration file")
parser.add_argument("-d", "--density", type=float, default=None, help="specify episode density, (default:None)")

def controlActsMsg():
    print(bold(blue("#----- Controls ------ #")))
    print("1. Up Arrow key -> Accelerate")
    print("2. Down Arrow Key -> Decelerate")
    print("3. Left or Right Arrow Key -> Lane Change")
    print(bold(blue("#----- Controls ------ #")))

def init(env):
    #---- Check for only local view ----#
    if env.gridHandler.isCommEnabled:
        raiseValueError("Only Local View is supported in this mode.")
    #---- Check for only local view ----#

    #--- Generate Reverse Action Map ----#
    reversedActMap = reverseDict(env.action_map)
    #--- Generate Reverse Action Map ----#
    return reversedActMap

def runEpisode(env, reversedActMap, density):
    if density != None:
        env.reset(density)
    else:
        env.reset()
    
    while True:
        act = reversedActMap["do-nothing,null"]
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    act = reversedActMap["lane-change,null"]
                elif event.key == pygame.K_RIGHT:
                    act = reversedActMap["lane-change,null"]
                elif event.key == pygame.K_DOWN:
                    act = reversedActMap["dec,null"]
                elif event.key == pygame.K_UP:
                    act = reversedActMap["acc,null"]
        state, reward, done, _ = env.step(act)
        if done:
            break

if __name__ == "__main__":
    
    # Parse Argument
    args = parser.parse_args()

    # Print control info
    controlActsMsg()

    # Init environment object
    env = V2I.V2I(args.sim_config)

    # Init Function
    reversedActMap = init(env)

    # Run Episode
    runEpisode(env, reversedActMap, args.density)


