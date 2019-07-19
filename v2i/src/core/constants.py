'''

This file defines the constants which are used the simulation.
It is recomended not to change these as the simulation environement is set up according to these.

'''

MAX_CARS_IN_LANE = 31

# x pixles = 1m, ex: 6p makes up 1m
SCALE = 6
POLYGON_POINTS = 10

# All in meteres goes here
CAR_WIDTH = 2
CAR_LENGTH = 2 * CAR_WIDTH
MIN_CAR_DISTANCE = 2
MIN_DISTANCE = MIN_CAR_DISTANCE + 2 * (CAR_LENGTH/2)

# All in pixels goes here
LANE_RADIUS = 233

# UI Constants
RADIUS = 250
CENTRE = (RADIUS, RADIUS)
