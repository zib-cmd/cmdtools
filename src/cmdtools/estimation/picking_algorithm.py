#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:19:09 2019

@author: bzfsechi
picking algorithm as explained in "Set-free Markov state model building "
"""
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

def picking_algorithm(_min, _max, dim, n_final, traj):
    """ to be written """
    #start_set = (_max - _min)* np.random.random_sample((n_final*10, dim)) + _min
    start_set = traj
    #first_rand = np.random.randint(0, n_final*10)
    first_rand =  np.random.randint(0, np.shape(start_set)[0])
    dist = distance.cdist(start_set, start_set, distance.sqeuclidean)
    rand2 = np.argmax(dist[first_rand, :])
    dist[:, [first_rand, rand2]] = np.inf
    list_points = [first_rand, rand2]
    np.fill_diagonal(dist, np.max(dist)+1)
    i = 0
#    while i < (n_final - 2):
#        max_min = 0.
#        
#        #for q_ in range(n_final*10):
#        for q_ in range(np.shape(start_set)[0]):   
#            if q_ in list_points:
#                continue                  
#        
#            if np.min(dist[q_, list_points]) > max_min:
#                max_min = np.min(dist[q_, list_points])
#                i += 1
#                list_points.append(q_)
    
    for i in range(n_final - 2):
        v1 = np.argmin(dist[list_points, :], axis=1)
        v  = np.amin(dist[list_points, :], axis=1)
        v2 = np.argmax(v)
        newpoint = v1[v2]
        list_points.append(newpoint)
        dist[:, newpoint] = np.inf
    return(start_set, start_set[list_points, :])#, list_points)

#%%
#example
#A, B, liste = picking_algorithm(-np.pi, np.pi, 2, 75)
##%%
#plt.scatter(A[:, 0], A[:, 1])
#plt.scatter(B[:, 0], B[:, 1], color = "r")
#plt.grid()