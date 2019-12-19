#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:19:09 2019

@author: bzfsechi
picking algorithm as explained in "Set-free Markov state model building "
"""
#import numpy as np
#from scipy.spatial import distance
#import matplotlib.pyplot as plt
#from scipy.stats import norm

#def picking_algorithm(_min, _max, dim, n_final, traj):
#    """ to be written """
#    #start_set = (_max - _min)* np.random.random_sample((n_final*10, dim)) + _min
#    start_set = traj
#    #first_rand = np.random.randint(0, n_final*10)
#    dist = distance.cdist(start_set, start_set, distance.sqeuclidean)
#    first_rand =  np.random.randint(0, np.shape(start_set)[0])
#    rand2 = np.argmax(dist[first_rand, :])
#    #dist[:, [first_rand, rand2]] = np.amax(dist) 
#
#    dist[:, [first_rand, rand2]] = np.inf
#    list_points = [first_rand, rand2]
#    np.fill_diagonal(dist, np.inf)
#    
##    while i < (n_final - 2):
##        max_min = 0.
##        
##        #for q_ in range(n_final*10):
##        for q_ in range(np.shape(start_set)[0]):   
##            if q_ in list_points:
##                continue                  
##        
##            if np.min(dist[q_, list_points]) > max_min:
##                max_min = np.min(dist[q_, list_points])
##                i += 1
##                list_points.append(q_)
#    
#    for i in range(n_final - 2):
#        print(i)
#        dist[:, list_points] = np.inf
#        v1 = np.argmin(dist[list_points, :], axis=1)
#        v  = np.amin(dist[list_points, :], axis=1)
#        v2 = np.argmax(v)
#        newpoint = v1[v2]
#        
#        #dist[newpoint, :] = np.inf
#        
#        list_points.append(newpoint)
#        #dist[newpoint, :] = np.amax(dist)  
#        print(v1,v,v2, list_points)
#        plt.scatter(start_set[:,0], start_set[:,1])
#        plt.scatter(start_set[list_points, 0],start_set[list_points, 1])
#        plt.show()
#    return(start_set, start_set[list_points, :], dist)#, list_points)
import numpy as np
import scipy.spatial.distance as dist
def picking_list(d, n):
    """
    Picking algorithm (Durmaz, 2016)
    Pick out n points such that the respective minimal distance
    to the other picked points is maximized.    Parameters
    ----------
    d : array
        Matrix containing the pairwise distances of the points.
    n : int
        Number of points to pick.    Returns
    -------
    inds : list[int]
        List of indices of the chosen points.    """ 
    
    d = d.copy()
    q1 = 0
    q2 = np.argmax(d[q1, :])
    qs = [q1, q2]
    for k in range(1,n):
        ss = np.argmin(d[qs,:], axis=0)
        mins = list(d[qs[j], s] for s,j in enumerate(ss))
        q = np.argmax(mins)
        qs.append(q)
    return qs


def picking_algorithm(traj, n):
    
    distances = dist.cdist(traj, traj, dist.sqeuclidean)
    
    return(traj, traj[picking_list(distances, n),:])
    
#%%
    
def brownian(x0, n, dt, delta, out=None):
    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*np.sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples. 
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out
#%%
#delta = 0.5
## Total time.
#T = 10.0
## Number of steps.
#N = 300
## Time step size
#dt = T/N
## Initial values of x.
#x = np.empty((2,N+1))
#x[:, 0] = 0.0
#
#brownian(x[:,0], N, dt, delta, out=x[:,1:])
#
### Plot the 2D trajectory.
##plt.scatter(x[0],x[1])
##plt.show()
##%%
#uni = np.random.uniform(0,2, size = (100,2))
##example
#A, B, liste = picking_algorithm(-np.pi, np.pi, 2, 30, uni)
###%%
#plt.scatter(A[:, 0], A[:, 1])
#plt.scatter(B[:, 0], B[:, 1], color = "r")
#plt.grid()
#plt.show()
#plt.imshow(liste)
#plt.colorbar()
#plt.show()