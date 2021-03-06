import time
import pygame
import numpy as np

import v2i.src.core.constants as constants
from v2i.src.core.common import arcAngle, getAgentID

class ui:

    def __init__(self, fps, extendedViewInMetre, cellSizeInMetre, isTFEnabled, isCarDetailsEnabled):

        self.fps = fps
        self.extendedViewInMetre = extendedViewInMetre
        self.cellSizeInMetre = cellSizeInMetre
        self.isTFEnabled = isTFEnabled
        self.isCarDetailsEnabled = isCarDetailsEnabled

        # Init Graphics Library - PyGame
        pygame.init()
        pygame.font.init()

        # Init Colors
        self.initColors()

        # Init Screen
        self.initScreen()

        # Init Fonts
        self.initFonts()

        # Init Information Board
        self.initInfoBoard()
        
        # Init occupancy Grids
        self.initDrawGrid()

        # Init Traffic Lights
        if self.isTFEnabled:
            self.initLight()

        # Init FPS Clock
        self.initClock()
    
    def initColors(self):
        self.roadColor = (97, 106, 107)
        self.colorBG = (30, 132, 73)
        self.colorWhite = (255, 255, 255)
        self.colorRed = (236, 112, 99)
        self.colorDarkRed = (255, 0, 0)
        self.colorYellow = (247, 220, 111)
        self.colorGrey = (192, 192, 192, 80)
        self.colorLime = (128, 250, 0)
        self.colorDarkGreen = (0, 120, 0)
        self.colorBlack = (0, 0, 0)
    
    def initScreen(self):
        extraSpace = 100
        dimsX = (2 * constants.RADIUS) + extraSpace
        dimsY = (2 * constants.RADIUS)
        self.screen = pygame.display.set_mode((dimsX, dimsY))
        pygame.display.set_caption("V2I")
    
    def initInfoBoard(self):
        self.infoBoard = pygame.Surface(constants.INFO_BOARD_DIM, pygame.SRCALPHA)
        self.infoBoard.fill((128, 128, 128))
        self.infoBoard.set_alpha(80)
        self.infoBoardX = constants.RADIUS - (constants.INFO_BOARD_DIM[0]/2)
        self.infoBoardY = constants.RADIUS - (constants.INFO_BOARD_DIM[1]/2)
        self.infoBoardDim = (self.infoBoardX, self.infoBoardY)
        self.infoBoardCurX = self.infoBoardX + 10
        self.infoBoardCurY = self.infoBoardY + 10
    
    def initDrawGrid(self):
        # Calculate start position and cell size in Degrees for drawing grids
        self.extendedViewInDeg = []
        self.cellSizeInDeg = []

        for lane in range(0, constants.LANES):
            self.extendedViewInDeg.append(arcAngle(constants.LANE_RADIUS[lane], self.extendedViewInMetre * constants.SCALE))
            self.cellSizeInDeg.append(arcAngle(constants.LANE_RADIUS[lane], self.cellSizeInMetre * constants.SCALE))

    def resetInfoBoardLoc(self):
        self.infoBoardCurX = self.infoBoardX + 10
        self.infoBoardCurY = self.infoBoardY
    
    def initLight(self):
        self.redLightImage = pygame.transform.scale(pygame.image.load("v2i/src/data/images/red.png").convert_alpha(), (40, 80))
        self.greenLightImage = pygame.transform.scale(pygame.image.load("v2i/src/data/images/green.png").convert_alpha(), (40, 80))
        # Find Light Coordinates for Both the Lanes
        self.tfCoordinates = []
        self.tfCoordinates.append((constants.CENTRE[0] + constants.LANE_RADIUS[1] + 50, constants.LANE_RADIUS[0] - 100))
        self.tfCoordinates.append((constants.CENTRE[0] + constants.LANE_RADIUS[0] - 118, constants.LANE_RADIUS[0] - 100))
        
    
    def initFonts(self):
        self.font = pygame.font.Font("v2i/src/data/fonts/RobotoSlab-Bold.ttf", constants.FONT_SIZE)
        self.smallFont = pygame.font.Font("v2i/src/data/fonts/RobotoSlab-Bold.ttf", constants.SMALL_FONT_SIZE)
    
    def initClock(self):
        self.clock = pygame.time.Clock()
    
    def drawRoadBoundary(self, screen, color, radius, width, pos):
        pygame.draw.circle(screen, color, pos, radius, width)
    
    def drawRoad(self, screen, roadColor, radius, roadWidth, pos):
        pygame.draw.circle(screen, roadColor, pos, radius, roadWidth)
    
    def metre2pixel(self, m):
        return m*constants.SCALE
    
    def drawCar(self, screen, centre, color):
        pygame.draw.circle(screen, color, centre, self.metre2pixel(constants.CAR_RADIUS), 0)
    
    def getCoordinates(self, angle, radius, centre):
        X = centre[0] + (np.cos(np.deg2rad(angle)) * radius)
        Y = centre[1] + (np.sin(np.deg2rad(angle)) * radius)
        return X,Y
    
    def carInfo(self, carId, lane, frontID, followerID):
        return self.smallFont.render("ID:%d,L:%d\nF:%d,B:%d"%(carId, lane, frontID, followerID), False, self.colorDarkRed)
    
    def drawAllCars(self, carsData, followerList, frontList):
        for laneID in carsData.keys():
            for car in carsData[laneID]:
                X, Y = self.getCoordinates(car['pos'], constants.LANE_RADIUS[laneID], constants.CENTRE)
                carColor = self.colorYellow
                if car['agent'] == 1:
                    carColor = self.colorLime
                self.drawCar(self.screen, (int(X), int(Y)), carColor)
                if self.isCarDetailsEnabled:
                    if followerList == "none":
                        pass
                    else:
                        idTup = car['id']
                        followerID = followerList[idTup]
                        frontID = frontList[idTup]
                        self.screen.blit(self.carInfo(car['id'], laneID, frontID, followerID), (int(X)-5, int(Y)-5))
    
    def str2font(self, msgStr):
        return self.font.render(msgStr, False, (0, 0, 0))

    def updateInfoBoard(self, screen, agentSpeed, maxSpeed, timeElapsed, viewRange, extendedViewInMetre, agentLane, planAct, queryAct, agentReward):
        self.resetInfoBoardLoc()
        screen.blit(self.infoBoard, self.infoBoardDim)
        
        # ---- Time Elapsed ---- #
        timeString = 'Time Elapsed : \t\t%d secs'%(timeElapsed)
        timeStringText = self.str2font(timeString)
        screen.blit(timeStringText, (self.infoBoardCurX, self.infoBoardCurY))
        self.infoBoardCurY += constants.INFO_BOARD_GAP
        # ---- Time Elapsed ---- #

        # ---- Agent Speed ---- #
        speedString = 'Agent Speed : \t\t%.2f km/hr'%(round((18/5.0)*agentSpeed, 2))
        speedStringText = self.str2font(speedString)
        screen.blit(speedStringText, (self.infoBoardCurX, self.infoBoardCurY))
        self.infoBoardCurY += constants.INFO_BOARD_GAP
        # ---- Agent Speed ---- #

        # ---- Max Speed ---- #
        maxSpeedString = 'Max Speed : \t\t%.2f km/hr'%(round((18/5.0)*maxSpeed, 2))
        maxspeedStringText = self.str2font(maxSpeedString)
        screen.blit(maxspeedStringText, (self.infoBoardCurX, self.infoBoardCurY))
        self.infoBoardCurY += constants.INFO_BOARD_GAP
        # ---- Max Speed ---- #

        #---- Local Visiblity ----#
        localVisiblityString = 'Total Local Visiblity : \t\t%.2f m'%(viewRange)
        localVisibliyStringText = self.str2font(localVisiblityString)
        screen.blit(localVisibliyStringText, (self.infoBoardCurX, self.infoBoardCurY))
        self.infoBoardCurY += constants.INFO_BOARD_GAP
        #---- Local Visiblity ----#

        #---- Extended View  ----#
        extendedVisiblityString = 'Total Extended Visiblity : \t\t%.2f m'%(extendedViewInMetre)
        extendedVisibliyStringText = self.str2font(extendedVisiblityString)
        screen.blit(extendedVisibliyStringText, (self.infoBoardCurX, self.infoBoardCurY))
        self.infoBoardCurY += constants.INFO_BOARD_GAP
        #---- Extended View  ----#

        #---- Agent Lane ----#
        agentLaneString = 'Agent Lane : \t\t%d'%(agentLane)
        agentLaneStringText = self.str2font(agentLaneString)
        screen.blit(agentLaneStringText, (self.infoBoardCurX, self.infoBoardCurY))
        self.infoBoardCurY += constants.INFO_BOARD_GAP
        #---- Agent Lane ----#

        #---- Plan Action ----#
        planActString = 'Plan Action : \t\t%s'%(planAct)
        planActStringText = self.str2font(planActString)
        screen.blit(planActStringText, (self.infoBoardCurX, self.infoBoardCurY))
        self.infoBoardCurY += constants.INFO_BOARD_GAP
        #---- Plan Action ----#

        #---- Query Act ----#
        queryActString = 'Query Region : \t\t%s'%(queryAct)
        queryActStringText = self.str2font(queryActString)
        screen.blit(queryActStringText, (self.infoBoardCurX, self.infoBoardCurY))
        self.infoBoardCurY += constants.INFO_BOARD_GAP
        #---- Query Act ----#

        #---- Agent Reward ----#
        rewardString = 'Reward : \t\t%s'%(agentReward)
        rewardStringText = self.str2font(rewardString)
        screen.blit(rewardStringText, (self.infoBoardCurX, self.infoBoardCurY))
        self.infoBoardCurY += constants.INFO_BOARD_GAP
        #---- Agent Reward ----#

    def drawGrids(self, screen, color, occGrid, agentID, extendedViewInMetre, laneMap, agentLane):
        agentPos = laneMap[agentLane][agentID]['pos']
        self.startAngleDeg = []
        for lane in range(0, constants.LANES):
            startPos = (agentPos - (self.extendedViewInDeg[lane]/2)) % 360
            self.startAngleDeg.append(startPos)
        
        for lane in range(0, constants.LANES):
            for col in range(0, occGrid.shape[1]):
                currLimit = self.startAngleDeg[lane]
                nextLimit = self.startAngleDeg[lane] + self.cellSizeInDeg[lane]
                p = []

                points = list(np.linspace(currLimit, nextLimit, constants.POLYGON_POINTS))
                for point in points:
                    x, y = self.getCoordinates(point, constants.LANE_BOUNDARIES[lane][0], constants.CENTRE)
                    p.append((x,y))

                points.reverse()
                for point in points:
                    x, y = self.getCoordinates(point, constants.LANE_BOUNDARIES[lane][1], constants.CENTRE)
                    p.append((x,y))

                if occGrid[lane][col] == constants.OCCGRID_CONSTS["OCCUPIED"]:
                    pygame.draw.polygon(screen, self.colorBlack, p, 0)
                elif occGrid[lane][col] == constants.OCCGRID_CONSTS["AGENT"]:
                    pygame.draw.polygon(screen, self.colorBlack, p, 0)
                elif occGrid[lane][col] == constants.OCCGRID_CONSTS["UNKNOWN"]:
                    pygame.draw.polygon(screen, self.colorRed, p, 0)
                else:
                    pygame.draw.polygon(screen, color, p, 1)
                self.startAngleDeg[lane] += self.cellSizeInDeg[lane]
    

    def drawLights(self, red=[False, False]):
        for lane in range(0, constants.LANES):
            if red[lane] == True:
                self.screen.blit(self.redLightImage, self.tfCoordinates[lane])
            else:
                self.screen.blit(self.greenLightImage, self.tfCoordinates[lane])
                
    def updateScreen(self, data, lightStat):
        
        self.screen.fill(self.colorBG)
        
        #---- lane 1----#
        self.drawRoadBoundary(self.screen, self.colorWhite, constants.RADIUS, constants.BOUNDARY_THICKNESS, constants.CENTRE)
        self.drawRoad(self.screen, self.roadColor, constants.RADIUS - constants.BOUNDARY_THICKNESS, 0, constants.CENTRE)
        self.drawRoadBoundary(self.screen, self.colorWhite, constants.RADIUS - constants.BOUNDARY_THICKNESS - constants.LANE_WIDTH, constants.BOUNDARY_THICKNESS, constants.CENTRE)
        #---- lane 1----#

        #---- lane 2 ----#
        # 2 is subtracted because rendering was overlaping each other without it
        self.drawRoad(self.screen, self.roadColor, constants.RADIUS - (2*constants.BOUNDARY_THICKNESS) - constants.LANE_WIDTH, 0, constants.CENTRE)
        self.drawRoadBoundary(self.screen, self.colorWhite, constants.RADIUS - (2*constants.BOUNDARY_THICKNESS) - (2*constants.LANE_WIDTH), constants.BOUNDARY_THICKNESS, constants.CENTRE)
        self.drawRoad(self.screen, self.colorBG, constants.RADIUS - (3*constants.BOUNDARY_THICKNESS) - (2*constants.LANE_WIDTH), 0, constants.CENTRE)
        #---- lane 2 ----#
        
        # Draw Grids
        agentID = getAgentID(data["allData"], data["agentLane"])
        self.drawGrids(self.screen, self.colorWhite, data["occGrid"], agentID, data["extendedViewRange"], data["allData"], data["agentLane"])

        # Draw Cars in the lane
        self.drawAllCars(data["allData"], data["followerList"], data["frontList"])
        
        # Update Information Board data
        self.updateInfoBoard(self.screen, data["agentSpeed"], data["maxSpeed"], data["timeElapsed"], data["viewRange"], data["extendedViewRange"], data["agentLane"], data["planAct"], data["queryAct"], data["agentReward"])
        
        # Update TF Lights
        if self.isTFEnabled:
            self.drawLights(lightStat)

        # Update pygame screen
        pygame.display.flip()

        # FPS clock
        self.clock.tick(self.fps)