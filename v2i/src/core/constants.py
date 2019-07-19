'''

This file defines the constants which are used int the simulation.
It is recomended not to change these as the simulation environement is set up according to these.

'''

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
INFO_BOARD_DIM = (250, 250)