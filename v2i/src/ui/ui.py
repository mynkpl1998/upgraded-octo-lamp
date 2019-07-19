import pygame

import v2i.src.core.constants as constants

class ui:

    def __init__(self):

        # Init Graphics Library - PyGame
        pygame.init()
        pygame.font.init()

        # Init Colors
        self.initColors()

        # Init Screen
        self.initScreen()
    
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
