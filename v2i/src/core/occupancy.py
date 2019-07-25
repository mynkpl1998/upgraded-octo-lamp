import numpy as np

from v2i.src.core.utils import raiseValueError

class Grid:

    def __init__(self, localView, extendedView=None, cellSize=1):
        self.localView = localView # in metre
        self.cellSize = cellSize # in metre
        self.extendedView = extendedView

        #---- Checks ----#
        if self.extendedView == None:
            self.extendedView = self.localView
        
        if self.extendedView < self.localView:
            raiseValueError("size of extended view should be greater than local view")
        #---- Checks ----#

        #---- Calculate Communicable Region ----#
        self.commView = self.extendedView - self.localView # in metre
        #---- Calculate Communicable Region ----#

        if self.extendedView % self.cellSize != 0:
            raiseValueError("communication should be completely divisible by cellsize")

        #---- Calculate the number of columns ----#
        self.numCols = int(self.extendedView / self.cellSize)
        
        #---- Calculate the number of columns ----#
        
        #---- Calculate limit of angle ----#
        
        #---- Calculate limit of angle ----#
        
    def getOccupancyGrid(self, laneMap):
        return np.zeros((2, self.numCols))
        
    def getGrids(self, laneMap):
        return self.getOccupancyGrid(laneMap)

