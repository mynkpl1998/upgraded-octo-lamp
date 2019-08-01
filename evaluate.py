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

    for density in densitiesList:
        densityAvgSpeed = []
        planDictCount = buildDictWithKeys(simData["plan-acts"], 0)
        queryDictCount = buildDictWithKeys(simData["query-acts"], 0)
        actionCounts = 0

        numEpisodes = len(simData["data"][density])
        for episode in simData["data"][density]:
            episodeLength = 0

            #----- Episode Avg Speed -----#
            episodeAvgSpeed = 0.0
            for speed in simData["data"][density][episode]['speed']:
                episodeAvgSpeed += speed
                episodeLength += 1
            #----- Episode Avg Speed -----#

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

    #---- Plot Avg Speed ----#
    avgSpeedGraph = pygal.Bar()
    avgSpeedGraph.title = "Average Agent Speed (km/hr)"
    #avgSpeedGraph.x_labels = map(str, densitiesList)
    for i, speed in enumerate(avgSpeed):
        avgSpeedGraph.add(str(densitiesList[i]), speed)
    avgSpeedGraph.render_to_file(args.out_file_path + "/avgSpeed.svg")
    #---- Plot Avg Speed ----#

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
    
if __name__ == "__main__":
    
    # Parse Arguments
    args = parser.parse_args()

    # Load data
    simData = loadPKL(args.file)
    
    # Save Plots
    plot(simData, args)

