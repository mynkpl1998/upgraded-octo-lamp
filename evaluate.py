from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pygal

# Ray Imports
from v2i import V2I
from v2i.src.core.common import readYaml, loadPKL, buildDictWithKeys
from v2i.src.core.ppoController import ppoController

parser = argparse.ArgumentParser(description="script to generate graphs from data")
parser.add_argument("-file", type=str, required=True, help="file to load data from")
parser.add_argument("-out", "--out-file-path", type=str, required=True, help="path to save generated graphs")


def plot(simData, args):
    densitiesList = list(simData["data"].keys())
    avgSpeed = []
    planDist = []
    queryDist = []
    EpisodeLength = []
    maxEgoSpeed = []

    for density in densitiesList:
        densityAvgSpeed = []
        
        planDictCount = buildDictWithKeys(simData["plan-acts"], 0)
        queryDictCount = buildDictWithKeys(simData["query-acts"], 0)
        actionCounts = 0
        totalNumberSteps = 0
        gloablEgoMaxSpeed = -10

        numEpisodes = len(simData["data"][density])
        for episode in simData["data"][density]:
            episodeLength = 0

            #----- Episode Avg Speed -----#
            episodeAvgSpeed = 0.0
            for speed in simData["data"][density][episode]['speed']:
                episodeAvgSpeed += speed
                episodeLength += 1
                totalNumberSteps += 1
            #----- Episode Avg Speed -----#

            #----- Ego Max Speed -----#
            gloablEgoMaxSpeed = max(gloablEgoMaxSpeed, simData["data"][density][episode]['EgoMaxSpeed'])
            #----- Ego Max Speed -----#

            #----- Action Percentages -----#
            for action in simData["data"][density][episode]["actions"]:
                planAct, queryAct = action[0], action[1]
                planDictCount[planAct] += 1
                queryDictCount[queryAct] += 1
                actionCounts += 1

            #----- Action Percentages -----#        
            densityAvgSpeed.append((episodeAvgSpeed/episodeLength) * 3.6)
        
        # ---- Avg Speed ---- #
        speedSum = 0.0
        for avg in densityAvgSpeed:
            speedSum += avg
        avgSpeed.append(speedSum/numEpisodes)
        # ---- Avg Speed ---- #

        #---- Action Percentages ----#
        for key in planDictCount.keys():
            planDictCount[key] /= actionCounts
            planDictCount[key] *= 100
        for key in queryDictCount.keys():
            queryDictCount[key] /= actionCounts
            queryDictCount[key] *= 100
        planDist.append(planDictCount.copy())
        queryDist.append(queryDictCount.copy())
        #---- Action Percentages ----#

        #---- Episode Length ----#
        EpisodeLength.append(totalNumberSteps/numEpisodes)
        #---- Episode Length ----#

        #----- Ego Max Speed -----#
        maxEgoSpeed.append(gloablEgoMaxSpeed)
        #----- Ego Max Speed -----#        

    #---- Plot Avg Speed ----#
    avgSpeedGraph = pygal.Bar()
    avgSpeedGraph.title = "Average Agent Speed (km/hr), (max: %.2f km/hr)"%(simData["maxSpeed"])
    #avgSpeedGraph.x_labels = map(str, densitiesList)
    for i, speed in enumerate(avgSpeed):
        avgSpeedGraph.add(str(densitiesList[i]), speed)
    avgSpeedGraph.render_to_file(args.out_file_path + "/avgSpeed.svg")
    #---- Plot Avg Speed ----#

    #----- Plot Ego Max Speed -----#
    EgoMaxSpeedGraph = pygal.Bar()
    EgoMaxSpeedGraph.title = "Ego Vehicle max Speed (km/hr), (max : %.2f km/hr)"%(simData["maxViewSpeed"])
    for i, speed in enumerate(maxEgoSpeed):
        EgoMaxSpeedGraph.add(str(densitiesList[i]), speed)
    EgoMaxSpeedGraph.render_to_file(args.out_file_path + "/egoMaxSpeed.svg")
    #----- Plot Ego Max Speed -----#

    #---- Plot plan distribution ----#
    planActGraph = pygal.StackedBar()
    planActGraph.title = "Planning Action Distribution"
    planActGraph.x_labels = map(str, densitiesList)

    plan = {}
    for act in simData["plan-acts"]:
        plan[act] = []

    for data in planDist:
        for act in data.keys():
            plan[act].append(data[act])
    
    for act in plan.keys():
         planActGraph.add(act, plan[act])
    planActGraph.render_to_file(args.out_file_path + "/planDist.svg")
    #---- Plot plan distribution ----#

    #---- Plot query distribution ----#
    queryActGraph = pygal.StackedBar()
    queryActGraph.title = "Query Action Distribution"
    queryActGraph.x_labels = map(str, densitiesList)

    query = {}
    for act in simData["query-acts"]:
        query[act] = []

    for data in queryDist:
        for act in data.keys():
            query[act].append(data[act])
    
    for act in query.keys():
         queryActGraph.add(act, query[act])
    queryActGraph.render_to_file(args.out_file_path + "/queryDist.svg")
    #---- Plot plan distribution ----#

    #---- Average Episode Length ----#
    avgEpisodeLengthGraph = pygal.Bar()
    avgEpisodeLengthGraph.title = "Avg Episode Length, (Max: %d)"%(simData["max-episode-length"])
    for i, epLength in enumerate(EpisodeLength):
        avgEpisodeLengthGraph.add(str(densitiesList[i]), epLength)
    avgEpisodeLengthGraph.render_to_file(args.out_file_path + "/avgEpisodeLength.svg")
    #---- Average Episode Length ----#
    
if __name__ == "__main__":
    
    # Parse Arguments
    args = parser.parse_args()

    # Load data
    simData = loadPKL(args.file)
    
    # Save Plots
    plot(simData, args)

