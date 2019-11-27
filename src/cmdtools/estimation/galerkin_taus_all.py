#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:51:18 2019

@author: bzfsechi
"""

from cmdtools import utils
import numpy as np
from scipy.spatial import distance


def propagator_tau(timeseries, centers, sigma, max_tau= 1):
    """ Given `timeseries` data, estimate the propagator matrix.

    Uses the galerkin projection onto Gaussian ansatz functions
    with bandwidth `sigma` around the given `centers`. 
    
    Input:
        timeseries: array, each row describe the coordinates at a different
             time step
        centers: array, each row describes the coordinates of a center
        sigma: float, variance for the Gaussian function
        max_taus: maximal length of the timesteps, default = 1
        
    Output:
        array, dim= (max_tau, no_centers,no_centers) for each 
            tau in [1,max_tau] the propagator. The propagator is normed 
            row-wise.
        """
    
    max_tau = int(max_tau) # avoid float
    
    no_centers= centers.shape[0]
    
    m = get_membership(timeseries, centers, sigma)
    
    counts = np.zeros((max_tau, no_centers, no_centers))
    
    for i in range(1,max_tau+1):
        
       sum_over = np.zeros((no_centers, no_centers))
        
       for j in range(1,i):
           
           sum_over += m[j:-(i+j), :].T.dot(m[(i+j):-j, :])
       
           
       # take (i-1) for the index because it starts with zer    
       counts[i-1,:,:] =  ( m[0:-i, :].T.dot(m[i:, :]) + sum_over ) 
       
       counts[i-1,:,:] = utils.rowstochastic(counts[i-1,:,:])
       
    return counts


def get_membership(timeseries, centers, sigma):
    """ Compute the pairwise membership / probability of each datapoint
    to the Ansatz functions around each center.
    A Gaussian function is used as Ansatz function where we take the 
    Euclidean distance of the points to the centers coordinates.
    
    Input:
        timeseries: array, each row describe the coordinates at a different 
            time step
        centers: array, each row describes the coordinates of a center
        sigma: float, variance for the Gaussian function
        
    Output:
        array, row-wise normed, dim(no_timeseries, no_centers) 
        """
    sqdist = distance.cdist(timeseries, centers, distance.sqeuclidean)
    
    gausskernels = np.exp(-sqdist / (2 * sigma**2))
    
    return utils.rowstochastic(gausskernels)




