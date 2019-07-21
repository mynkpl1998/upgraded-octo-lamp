import time
import pygame
import numpy as np

import v2i.src.core.constants as constants

class ui:

    def __init__(self, fps):

        self.fps = fps

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

        # Init FPS Clock
        self.initClock()
    
    def initColors(self):
        self.roadColor = (97, 106, 107)
        self.colorBG = (30, 132, 73)
        self.colorWhite = (255, 255, 255)
        self.colorRed = (236, 112, 99)
        self.colorYellow = (247, 220, 111)
        self.colorGrey = (192, 192, 192, 80)
        self.colorLime = (128, 250, 0)
        self.colorDarkGreen = (0, 120, 0)
    
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
    
    def resetInfoBoardLoc(self):
        self.infoBoardCurX = self.infoBoardX + 10
        self.infoBoardCurY = self.infoBoardY + 10
    
    def initFonts(self):
        self.font = pygame.font.SysFont('Comic Sans MS', constants.FONT_SIZE)
    
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
    
    def drawAllCars(self, carsData):
        for laneID in carsData.keys():
            for car in carsData[laneID]:
                X, Y = self.getCoordinates(car['pos'], constants.LANE_RADIUS, constants.CENTRE)
                carColor = self.colorYellow
                if car['agent'] == 1:
                    carColor = self.colorLime
                self.drawCar(self.screen, (int(X), int(Y)), carColor)
    
    def str2font(self, msgStr):
        return self.font.render(msgStr, False, (0, 0, 0))

    def updateInfoBoard(self, screen, agentSpeed, maxSpeed, timeElapsed, viewRange):
        self.resetInfoBoardLoc()
        screen.blit(self.infoBoard, self.infoBoardDim)
        
        # ---- Time Elapsed ---- #
        timeString = 'Time Elapsed : %d secs'%(timeElapsed)
        timeStringText = self.str2font(timeString)
        screen.blit(timeStringText, (self.infoBoardCurX, self.infoBoardCurY))
        self.infoBoardCurY += 30
        # ---- Time Elapsed ---- #

        # ---- Agent Speed ---- #
        speedString = 'Agent Speed : %.2f km/hr'%(round((18/5.0)*agentSpeed, 2))
        speedStringText = self.str2font(speedString)
        screen.blit(speedStringText, (self.infoBoardCurX, self.infoBoardCurY))
        self.infoBoardCurY += 30
        # ---- Agent Speed ---- #

        # ---- Max Speed ---- #
        maxSpeedString = 'Max Speed : %.2f km/hr'%(round((18/5.0)*maxSpeed, 2))
        maxspeedStringText = self.str2font(maxSpeedString)
        screen.blit(maxspeedStringText, (self.infoBoardCurX, self.infoBoardCurY))
        self.infoBoardCurY += 30
        # ---- Max Speed ---- #

        #---- Local Visiblity ----#
        localVisiblityString = 'Local Visiblity : %.2f m'%(viewRange)
        localVisibliyStringText = self.str2font(localVisiblityString)
        screen.blit(localVisibliyStringText, (self.infoBoardCurX, self.infoBoardCurY))
        self.infoBoardCurY += 30
        #---- Local Visiblity ----#
                
    def updateScreen(self, data):
        self.screen.fill(self.colorBG)
        
        #---- lane 1----#
        self.drawRoadBoundary(self.screen, self.colorWhite, constants.RADIUS, constants.BOUNDARY_THICKNESS, constants.CENTRE)
        self.drawRoad(self.screen, self.roadColor, constants.RADIUS - constants.BOUNDARY_THICKNESS, 0, constants.CENTRE)
        self.drawRoadBoundary(self.screen, self.colorWhite, constants.RADIUS - constants.BOUNDARY_THICKNESS - constants.LANE_WIDTH, constants.BOUNDARY_THICKNESS, constants.CENTRE)
        self.drawRoad(self.screen, self.colorBG, constants.RADIUS - (2*constants.BOUNDARY_THICKNESS) - constants.LANE_WIDTH, 0, constants.CENTRE)
        #---- lane 1----#
        
        # Draw Cars in the lane
        self.drawAllCars(data["allData"])
        
        # Update Information Board data
        self.updateInfoBoard(self.screen, data["agentSpeed"], data["maxSpeed"], data["timeElapsed"], data["viewRange"])
        
        # Update pygame screen
        pygame.display.flip()

        # FPS clock
        self.clock.tick(self.fps)
    
    

