#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:19:09 2019

@author: bzfsechi
picking algorithm as explained in "Set-free Markov state model building "
"""
import numpy as np
from scipy.spatial import distance

def picking_algorithm( _min, _max, dim, n_final):
    
    start_set =(_max-_min)* np.random.random_sample((n_final*2, dim)) + _min
    
   # picked_points = np.zeros((n_final, dim))
    first_rand = np.random.randint(0,n_final*2)
    
    list_points = [first_rand, np.argmax(dist[first_rand,:])]
    dist = distance.euclidean(start_set, start_set)
    np.fill_diagonal(dist, np.max(dist)+1)
    i = 1
    while i < n_final:
        
        max_min = 0.
        
        for q in range(n_final*2):
            
            if q in list_points:
                continue
            
            else:
                
                if np.min(dist[q,list_points])> max_min:
                    max_min = np.min(dist[q,list_points])
                    
                    i +=1
                    list_points.append(q)
        
    
    return(start_set, start_set[list_points,:] )


#%%