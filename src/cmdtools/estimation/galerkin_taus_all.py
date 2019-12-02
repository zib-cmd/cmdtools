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
    
    counts = np.zeros((max_tau+1, no_centers, no_centers))
    
    counts[0,:,:] = m.T.dot(m)
    counts[0,:,:] = utils.rowstochastic(counts[0,:,:])
    for i in range(1,max_tau+1):
        
       sum_over = np.zeros((no_centers, no_centers))
        
       for j in range(1,i):
           
           sum_over += m[j:-(i+j), :].T.dot(m[(i+j):-j, :])
       
           
       # take (i-1) for the index because it starts with zer    
       counts[i,:,:] =  ( m[0:-i, :].T.dot(m[i:, :]) + sum_over ) 
       
       counts[i,:,:] = utils.rowstochastic(counts[i,:,:])
       
       counts[i,:,:] = np.linalg.inv(counts[0,:,:]).dot(counts[i,:,:])
       
    counts[0,:,:] = np.linalg.inv(counts[0,:,:]).dot(counts[0,:,:])
    
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


def find_bandwidth(timeseries, centers, percentile=50): #, plot= True):
    """Find the bandwidth of the Gaussian based on: 
     
    "Stein Variational Gradient Descent: 
    A General Purpose Bayesian Inference Algorithm", 
    Qiang Liu and Dilin Wang (2016).
    
     Based on the value of the percentile is possible to decide the points to
     take into consideration for the determination of the bandwidth.
     
    Input:
        timeseries: arr, trajectory, each row is a collection of coordinates at a
            different timestep
        centers: arr, centers of the Gaussians, each row has the coordinates 
            of a different center
        percentile: int [0,100], default value = 50
        
     Output:
         sigma: float, the  variance of the Gaussian"""
         
    no_centers = np.shape(centers)[0]
    sqdist = distance.cdist(timeseries, centers, distance.sqeuclidean)

    return np.percentile(sqdist,percentile)/ np.sqrt(2*np.log(no_centers)) 

