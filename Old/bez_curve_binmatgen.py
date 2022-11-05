# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:02:52 2022

@author: rmish
"""

import numpy as np
from scipy.special import comb

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    nPoints = 8
    # points = np.random.rand(nPoints,2)*15
    points = np.array([[0,7],[4,13],[7,7],[12,2],[15,7]])
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]

    xvals, yvals = bezier_curve(points, nTimes=15)
    plt.plot(np.round(xvals), yvals)
    plt.plot(xpoints, ypoints, "ro")
    for nr in range(len(points)):
        plt.text(points[nr][0], points[nr][1], nr)

    plt.show()
    
    z1,z2 = np.rint(xvals),np.rint(yvals)#,dtype=int)

bezmat = np.zeros([15,15])

for i in range(15):
    bezmat[int(z2[i]),i] = 1
plt.figure()
plt.imshow(bezmat)
from scipy import ndimage
b2 = ndimage.binary_dilation(bezmat)
plt.figure()
plt.imshow(b2)
