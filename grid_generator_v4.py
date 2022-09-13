# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 12:50:39 2022

@author: rmish
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
import skimage

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


#%%xmat and bezmat
# bezmat = np.zeros([11,11])

# # points = np.array([[0,5],[2,9],[5,5],[8,2],[11,5]])
# points = np.array([[0,0],[3,10],[8,1],[10,10]])


# xpoints = [p[0] for p in points]
# ypoints = [p[1] for p in points]

# xvals, yvals = bezier_curve(points, nTimes=11)
# plt.plot(np.round(xvals), yvals)
# plt.plot(xpoints, ypoints, "ro")
# for nr in range(len(points)):
#     plt.text(points[nr][0], points[nr][1], nr)
# bz1,bz2 = np.rint(xvals),np.rint(yvals)#,dtype=int)
# for i in range(11):
#     bezmat[int(bz2[i]),i] = 1
# plt.show()

xmat = np.zeros([11,11])

points = np.array([[0,0],[10,10]])
xpoints = [p[0] for p in points]
ypoints = [p[1] for p in points]

xvals, yvals = bezier_curve(points, nTimes=11)
plt.plot(np.round(xvals), yvals)
plt.plot(xpoints, ypoints, "ro")
for nr in range(len(points)):
    plt.text(points[nr][0], points[nr][1], nr)
xz1,xz2 = np.rint(xvals),np.rint(yvals)#,dtype=int)
for i in range(11):
    xmat[int(xz2[i]),i] = 1

points = np.array([[10,0],[0,10]])
xpoints = [p[0] for p in points]
ypoints = [p[1] for p in points]

xvals, yvals = bezier_curve(points, nTimes=11)
# plt.plot(np.round(xvals), yvals)
# plt.plot(xpoints, ypoints, "ro")
for nr in range(len(points)):
    plt.text(points[nr][0], points[nr][1], nr)
xz1b,xz2b = np.rint(xvals),np.rint(yvals)#,dtype=int)

for i in range(11):
    xmat[i,int(xz1b[i])] = 1
# plt.show()


    
    
    
    
# plt.figure()
# plt.imshow(bezmat)
# from scipy import ndimage
# b2 = ndimage.binary_dilation(bezmat)
# plt.figure()
# plt.imshow(b2)
#%%Matrix instantiation

clover_mat = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                 


# dil_clover = ndimage.binary_dilation(aclover_mat)


# rand_mat = np.random.choice([0, 1], size=(11,11), p=[0.92,0.08])
rand_mat = np.load('rand_mat_v1.npy')

rr,cc = skimage.draw.circle_perimeter(5,5,3,shape=[11,11])
c_mat = np.zeros((11, 11), dtype=np.uint8)

c_mat[rr,cc] = 1
plt.figure()
plt.imshow(c_mat)
f, axarr = plt.subplots(2,2)
# axarr.axis('off')
axarr[0,0].imshow(xmat,cmap = 'Greys_r')
axarr[0, 0].set_axis_off()

axarr[0,1].imshow(clover_mat,cmap = 'Greys_r')
axarr[0, 1].set_axis_off()

axarr[1,0].imshow(rand_mat,cmap = 'Greys_r')
axarr[1, 0].set_axis_off()

axarr[1,1].imshow(c_mat,cmap = 'Greys_r')
axarr[1, 1].set_axis_off()

f.savefig('binmats.png')

f, axarr = plt.subplots(1,4)
# axarr.axis('off')
axarr[0].imshow(xmat,cmap = 'Greys_r')
axarr[0].set_axis_off()

axarr[1].imshow(clover_mat,cmap = 'Greys_r')
axarr[1].set_axis_off()

axarr[2].imshow(rand_mat,cmap = 'Greys_r')
axarr[2].set_axis_off()

axarr[3].imshow(c_mat,cmap = 'Greys_r')
axarr[3].set_axis_off()

f.savefig('binmats2.png')
plt.scatter

[x_xmat,y_xmat] = np.where(xmat)
[x_xmatp,y_xmatp] = np.where(1-xmat)

[x_clover_mat,y_clover_mat] = np.where(clover_mat)
[x_clover_matp,y_clover_matp] = np.where(1-clover_mat)

[x_rand_mat,y_rand_mat] = np.where(rand_mat)
[x_rand_matp,y_rand_matp] = np.where(1-rand_mat)

[x_c_mat,y_c_mat] = np.where(c_mat)
[x_c_matp,y_c_matp] = np.where(1-c_mat)

plt.ion()
f, axarr = plt.subplots(2,2)
# axarr.set_axis('equal')
ssize = 75
axarr[0,0].scatter(x_xmat,y_xmat,c='#6d77a2',s=ssize)
axarr[0,0].scatter(x_xmatp,y_xmatp,c='#b28280',s=ssize)
axarr[0,0].set_axis_off()
axarr[0, 0].set_aspect('equal', 'box')

axarr[1,0].scatter(x_clover_mat,y_clover_mat,c='#6d77a2',s=ssize)
axarr[1,0].scatter(x_clover_matp,y_clover_matp,c='#b28280',s=ssize)
axarr[1,0].set_axis_off()
axarr[1, 0].set_aspect('equal', 'box')

axarr[0,1].scatter(x_rand_mat,y_rand_mat,c='#6d77a2',s=ssize)
axarr[0,1].scatter(x_rand_matp,y_rand_matp,c='#b28280',s=ssize)
axarr[0,1].set_axis_off()
axarr[0, 1].set_aspect('equal', 'box')

axarr[1,1].scatter(x_c_mat,y_c_mat,c='#6d77a2',s=ssize)
axarr[1,1].scatter(x_c_matp,y_c_matp,c='#b28280',s=ssize)
axarr[1,1].set_axis_off()
axarr[1, 1].set_aspect('equal', 'box')

f.tight_layout()

f.savefig('binmats_scattered.png')





# points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
# from scipy.spatial import Delaunay
# tri = Delaunay(points)
# import matplotlib.pyplot as plt
# plt.triplot(points[:,0], points[:,1], tri.simplices)
# plt.plot(points[:,0], points[:,1], 'o')
# plt.show()

# #%%
# points = np.array([[0, 0], [0, 2], [1, 0], [1, 1], [1, 2],
#                    [2, 0], [2, 1], [2, 2]])
# from scipy.spatial import Voronoi, voronoi_plot_2d
# vor = Voronoi(points)
# import matplotlib.pyplot as plt
# fig = voronoi_plot_2d(vor)
# plt.show()

# #%%
# from scipy import ndimage
# struct = ndimage.generate_binary_structure(2, 1)