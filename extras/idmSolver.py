'''
This script is used to quickly check the solution of IDM Equation.
It uses sympy to solve the equation 
'''

'''
IDM Equation : https://en.wikipedia.org/wiki/Intelligent_driver_model
Modified IDM Equation includes the sense of local view to non-Ego vehicles
Modified IDM Details : https://github.com/mynkpl1998/single-ring-road-with-light/blob/master/SingleLaneIDM/Results%20Analysis.ipynb     
'''

import numpy as np
from sympy.solvers import solve
from sympy import Symbol
from sympy.functions import Abs

#---- Modify Paramters according to your case ----#
sAlpha = 10 # in metres
maxAcc = 0.73 # im m/s2
minDistance = 2.0 # in m
maxSpeed = 9 # in m/s
headwayTime = 1.5 # in secs
delta = 4
decelerationRate = 1.67

if __name__ == "__main__":

    # Solving for steady state speed
    v = Symbol('v')
    abroot = np.sqrt(maxAcc * decelerationRate)
    res = solve(maxAcc*(1 - ((v/maxSpeed)**delta) - (((minDistance + v*headwayTime + ((v*(v - 5.77927))/(2 * abroot)))/sAlpha)**2)), v)
    print(res)
    print("Possible Solutions are : ")
    for i, sol in enumerate(res):
        print("Solution %d : "%(i+1), Abs(sol), " = ", Abs(sol) * 3.6, "km/hr ")
    