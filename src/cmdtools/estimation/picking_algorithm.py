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

def picking_algorithm(_min, _max, dim, n_final):
    """ to be written """
    start_set = (_max - _min)* np.random.random_sample((n_final*2, dim)) + _min
    first_rand = np.random.randint(0, n_final*2)
    dist = distance.cdist(start_set, start_set, distance.sqeuclidean)
    np.fill_diagonal(dist, np.max(dist)+1)
    list_points = [first_rand, np.argmax(dist[first_rand, :])]
    i = 1
    while i < n_final:
        max_min = 0.
        
        for q_ in range(n_final*2):
            
            if q_ in list_points:
                continue                  
        
            if np.min(dist[q_, list_points]) > max_min:
                max_min = np.min(dist[q_, list_points])
                i += 1
                list_points.append(q_)
    return(start_set, start_set[list_points, :])

#%%
A, B = picking_algorithm(-2, 2, 2, 5)
#%%
plt.scatter(A[:, 0], A[:, 1])
plt.scatter(B[:, 0], B[:, 1])