from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pygal
import os
import matplotlib.pyplot as plt

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

    #---- Full Epiosdes Data ---- #
    numFullEpisodesList = []
    FullEpisodesavgSpeed = []
    FullEpisodesplanDist = []
    FullEpisodesQueryDist = []
    FullEpisodesmaxEgoSpeed = []
    collisionCount = []

    #---- Full Epiosdes Data ---- #

    for denID, density in enumerate(densitiesList):
        densityAvgSpeed = []
        densityFullEpisodeAvgSpeed = []
        allEpisodesSpeed = []
        allEpisodesSpeedFullEpisodes = []
        
        planDictCount = buildDictWithKeys(simData["plan-acts"], 0)
        queryDictCount = buildDictWithKeys(simData["query-acts"], 0)

        planDictCountFullEpisodes = buildDictWithKeys(simData["plan-acts"], 0)
        queryDictCountFullEpisods = buildDictWithKeys(simData["query-acts"], 0)
        
        collisionCount.append(simData["others"][density]["collision-count"])

        actionCounts = 0
        totalNumberSteps = 0
        gloablEgoMaxSpeed = -10
        gloablEgoMaxSpeedFullEpisode = -10
        numFullEpisodes = 0

        actionCountsFullEpisodes = 0

        numEpisodes = len(simData["data"][density])
        for episode in simData["data"][density]:
            episodeLength = 0

            #----- Episode Avg Speed -----#
            episodeAvgSpeed = 0.0
            
            if len(simData["data"][density][episode]['speed']) == simData["max-episode-length"]:
                numFullEpisodes += 1

            for speed in simData["data"][density][episode]['speed']:
                episodeAvgSpeed += speed
                allEpisodesSpeed.append(speed * 3.6)
                if len(simData["data"][density][episode]['speed']) == simData["max-episode-length"]:
                    allEpisodesSpeedFullEpisodes.append(speed * 3.6)
                episodeLength += 1
                totalNumberSteps += 1
            #----- Episode Avg Speed -----#

            #----- Ego Max Speed -----#
            gloablEgoMaxSpeed = max(gloablEgoMaxSpeed, simData["data"][density][episode]['EgoMaxSpeed'])
            #----- Ego Max Speed -----#
            
            #----- Ego Max Speed Full Episode-----#
            if len(simData["data"][density][episode]['speed']) == simData["max-episode-length"]:
                gloablEgoMaxSpeedFullEpisode = max(gloablEgoMaxSpeedFullEpisode, simData["data"][density][episode]['EgoMaxSpeed'])
            #----- Ego Max Speed Full Episode-----#
            
            #----- Action Percentages -----#
            if len(simData["data"][density][episode]['speed']) == simData['max-episode-length']:
                for action in simData["data"][density][episode]["actions"]:
                    planAct, queryAct = action[0], action[1]
                    planDictCountFullEpisodes[planAct] += 1
                    queryDictCountFullEpisods[queryAct] += 1
                    actionCountsFullEpisodes += 1
            else:
                for action in simData["data"][density][episode]["actions"]:
                    planAct, queryAct = action[0], action[1]
                    planDictCount[planAct] += 1
                    queryDictCount[queryAct] += 1
                    actionCounts += 1

            #----- Action Percentages -----#        
            densityAvgSpeed.append((episodeAvgSpeed/episodeLength) * 3.6)
            if len(simData["data"][density][episode]['speed']) == simData["max-episode-length"]:
                densityFullEpisodeAvgSpeed.append((episodeAvgSpeed/episodeLength) * 3.6)

        
        plt.hist(allEpisodesSpeed, bins=200, density=True)
        plt.title("Density : %.2f, Total Steps : %d"%(density, len(allEpisodesSpeed)))
        plt.xlabel("Speed (km/hr), max allowed by LV : %.2f"%(simData["maxViewSpeed"]))
        plt.ylabel("Count (normalized) % ")
        plt.savefig(args.out_file_path + "/AllEpisodes" + "/speedHist_%.2f.png"%(density))
        plt.cla()

        assert len(allEpisodesSpeedFullEpisodes) == numFullEpisodes * simData["max-episode-length"]
        plt.hist(allEpisodesSpeedFullEpisodes, bins=200, density=True)
        plt.title("Density : %.2f, Total Steps : %d"%(density, len(allEpisodesSpeedFullEpisodes)))
        plt.xlabel("Speed (km/hr), max allowed by LV : %.2f"%(simData["maxViewSpeed"]))
        plt.ylabel("Count (normalized) % ")
        plt.savefig(args.out_file_path + "/FullEpisodes" + "/speedHist_%.2f.png"%(density))
        plt.cla()

        # ---- Avg Speed ---- #
        speedSum = 0.0
        for avg in densityAvgSpeed:
            speedSum += avg
        avgSpeed.append(speedSum/numEpisodes)
        # ---- Avg Speed ---- #

        # ---- Avg Speed Full Episode ---- #
        speedSum = 0.0
        for avg in densityFullEpisodeAvgSpeed:
            speedSum += avg
        FullEpisodesavgSpeed.append(speedSum/numFullEpisodes)
        # ---- Avg Speed Full Episode ---- #                

        #----- Full Epsiodes List ----#
        numFullEpisodesList.append(numFullEpisodes)
        #----- Full Epsiodes List ----#

        #---- Action Percentages ----#
        #print(actionCountsFullEpisodes)
        count = 0
        for key in planDictCountFullEpisodes.keys():
            count += planDictCountFullEpisodes[key]
            planDictCountFullEpisodes[key] /= actionCountsFullEpisodes
            planDictCountFullEpisodes[key] *= 100
        assert count == numFullEpisodesList[denID] * simData['max-episode-length']

        count = -0
        for key in queryDictCountFullEpisods.keys():
            count += queryDictCountFullEpisods[key]
            queryDictCountFullEpisods[key] /= actionCountsFullEpisodes
            queryDictCountFullEpisods[key] *= 100
        FullEpisodesplanDist.append(planDictCountFullEpisodes.copy())
        FullEpisodesQueryDist.append(queryDictCountFullEpisods.copy())
        assert count == numFullEpisodesList[denID] * simData['max-episode-length']
        
        if actionCounts == 0:
            actionCounts = 1
        
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
        maxEgoSpeed.append(gloablEgoMaxSpeed * 3.6)
        #----- Ego Max Speed -----#

        #----- Ego Max Speed Full Episodes-----#
        FullEpisodesmaxEgoSpeed.append(gloablEgoMaxSpeedFullEpisode * 3.6)
        #----- Ego Max Speed Full Episodes-----#

    #---- Plot Collision Count ----#
    plotCollisionGraph = pygal.Bar()
    plotCollisionGraph.title = "Number of collisions"
    for i, count in enumerate(collisionCount):
        plotCollisionGraph.add(str(densitiesList[i]), count)
    plotCollisionGraph.render_to_file(args.out_file_path + "/FullEpisodes" + "/countPlot.svg")
    #---- Plot Collision Count ----#

    #---- Plot Avg Speed ----#
    avgSpeedGraph = pygal.Bar()
    avgSpeedGraph.title = "Average Agent Speed (km/hr), (max : %.2f km/hr)"%(simData["maxSpeed"])
    #avgSpeedGraph.x_labels = map(str, densitiesList)
    for i, speed in enumerate(avgSpeed):
        avgSpeedGraph.add(str(densitiesList[i]), speed)
    avgSpeedGraph.render_to_file(args.out_file_path + "/AllEpisodes" + "/avgSpeed.svg")
    #---- Plot Avg Speed ----#

    #---- Plot Avg Speed Full Episodes----#
    avgSpeedFullEpisodesGraph = pygal.Bar()
    avgSpeedFullEpisodesGraph.title = "Average Agent Speed FullEpisodes(km/hr), (max : %.2f km/hr)"%(simData["maxSpeed"])
    #avgSpeedGraph.x_labels = map(str, densitiesList)
    for i, speed in enumerate(FullEpisodesavgSpeed):
        avgSpeedFullEpisodesGraph.add(str(densitiesList[i]), speed)
    avgSpeedFullEpisodesGraph.render_to_file(args.out_file_path + "/FullEpisodes" + "/avgSpeedFullEpisodes.svg")
    #---- Plot Avg Speed Full----#

    #----- Plot Ego Max Speed -----#
    EgoMaxSpeedGraph = pygal.Bar()
    EgoMaxSpeedGraph.title = "Ego Vehicle max Speed (km/hr), (max allowed by LV : %.2f km/hr)"%(simData["maxViewSpeed"])
    for i, speed in enumerate(maxEgoSpeed):
        EgoMaxSpeedGraph.add(str(densitiesList[i]), speed)
    EgoMaxSpeedGraph.render_to_file(args.out_file_path + "/AllEpisodes" + "/egoMaxSpeed.svg")
    #----- Plot Ego Max Speed -----#

    #----- Plot Ego Max Speed Full Episodes -----#
    EgoMaxSpeedGraphFullEpisodes = pygal.Bar()
    EgoMaxSpeedGraphFullEpisodes.title = "Full Episodes, Ego Vehicle max Speed (km/hr), (max allowed by LV : %.2f km/hr)"%(simData["maxViewSpeed"])
    for i, speed in enumerate(FullEpisodesmaxEgoSpeed):
        EgoMaxSpeedGraphFullEpisodes.add(str(densitiesList[i]), speed)
    EgoMaxSpeedGraphFullEpisodes.render_to_file(args.out_file_path + "/FullEpisodes" + "/egoMaxSpeedFullEpisodes.svg")
    #----- Plot Ego Max Speed Full Episode-----#

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
    planActGraph.render_to_file(args.out_file_path + "/AllEpisodes" + "/planDist.svg")
    #---- Plot plan distribution ----#

    #---- Plot plan distribution Full Episodes----#
    planActGraphFullEpisodes = pygal.StackedBar()
    planActGraphFullEpisodes.title = "Planning Action Distribution (Full Episodes)"
    planActGraphFullEpisodes.x_labels = map(str, densitiesList)

    plan = {}
    for act in simData["plan-acts"]:
        plan[act] = []

    for data in FullEpisodesplanDist:
        for act in data.keys():
            plan[act].append(data[act])
    
    for act in plan.keys():
        planActGraphFullEpisodes.add(act, plan[act])
    planActGraphFullEpisodes.render_to_file(args.out_file_path + "/FullEpisodes" + "/planDistFullEpisodes.svg")
    #---- Plot plan distribution Full Episodes----#

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
    queryActGraph.render_to_file(args.out_file_path + "/AllEpisodes" + "/queryDist.svg")
    #---- Plot plan distribution ----#

    #---- Plot query distribution Full Episodes----#
    queryActGraphFullEpisodes = pygal.StackedBar()
    queryActGraphFullEpisodes.title = "Query Action Distribution (Full Episodes)"
    queryActGraphFullEpisodes.x_labels = map(str, densitiesList)

    query = {}
    for act in simData["query-acts"]:
        query[act] = []

    for data in FullEpisodesQueryDist:
        for act in data.keys():
            query[act].append(data[act])
    
    for act in query.keys():
        queryActGraphFullEpisodes.add(act, query[act])
    queryActGraphFullEpisodes.render_to_file(args.out_file_path + "/FullEpisodes" + "/queryDistFullEpisodes.svg")
    #---- Plot plan distribution Full Episodes----#

    #---- Average Episode Length ----#
    avgEpisodeLengthGraph = pygal.Bar()
    avgEpisodeLengthGraph.title = "Avg Episode Length, (Max: %d)"%(simData["max-episode-length"])
    for i, epLength in enumerate(EpisodeLength):
        avgEpisodeLengthGraph.add(str(densitiesList[i]), epLength)
    avgEpisodeLengthGraph.render_to_file(args.out_file_path + "/AllEpisodes" + "/avgEpisodeLength.svg")
    #---- Average Episode Length ----#

    #----- Full Epsiodes List Plot----#
    numFullEpisodesListGraph = pygal.Bar()
    numFullEpisodesListGraph.title = "Number of Full Episodes, (Number of episodes tested : %d)"%(numEpisodes)
    for i, numEpi in enumerate(numFullEpisodesList):
        numFullEpisodesListGraph.add(str(densitiesList[i]), numEpi)
    numFullEpisodesListGraph.render_to_file(args.out_file_path + "/FullEpisodes" + "/numFullEpisodes.svg")
    #----- Full Epsiodes List Plot----#

    
def createFolders(args):
    if not os.path.exists(args.out_file_path + "/FullEpisodes/"):
        os.mkdir(args.out_file_path+"/FullEpisodes/")
    if not os.path.exists(args.out_file_path + "/AllEpisodes/"):
        os.mkdir(args.out_file_path+"/AllEpisodes/")

if __name__ == "__main__":
    
    # Parse Arguments
    args = parser.parse_args()

    # Load data
    simData = loadPKL(args.file)
    
    # Create Directories
    createFolders(args)
    
    # Save Plots
    plot(simData, args)

