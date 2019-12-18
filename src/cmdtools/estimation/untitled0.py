#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:19:09 2019

@author: bzfsechi
picking algorithm as explained in "Set-free Markov state model building "
"""
import numpy as np

def picking_algorithm( _min, _max, dim, n_final):
    
    start_set =(_max-_min)* np.random.random_sample((n_final, dim)) + _min
    
    return(start_set)


#%%