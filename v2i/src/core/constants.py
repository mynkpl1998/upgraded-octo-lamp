'''

This file defines the constants which are used in the simulation.
It is recomended not to change these as the simulation environement is set up according to these.

'''

from v2i.src.core.defaults import DEFAULT_DICT

MAX_CARS_IN_LANE = 31

# x pixles = 1m, ex: 6 pixels makes up 1 metre
SCALE = 6
POLYGON_POINTS = 10

# All constants with metre units are here
CAR_RADIUS = 2
CAR_LENGTH = 2 * CAR_RADIUS
MIN_CAR_DISTANCE = 2
MIN_DISTANCE = MIN_CAR_DISTANCE + 2 * (CAR_LENGTH/2)

# All constants with pixels units are here
LANE_RADIUS = 233

# UI related constants (all in pixels)
RADIUS = 250
CENTRE = (RADIUS, RADIUS)
BOUNDARY_THICKNESS = 2
LANE_WIDTH = 30
INFO_BOARD_DIM = (250, 200)
FONT_SIZE = 25

# Intelligent Driver Model constants
IDM_CONSTS = {
    'MAX_ACC': 0.73,
    'HEADWAY_TIME': 1.5,
    'DECELERATION_RATE': 1.67,
    'MIN_SPACING': 2,
    'DELTA': 4 
}