#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 09:57:40 2019

@author: bzfsechi
"""

import numpy as np
from optimizeMetastab import optimizeMetastab
#%%

def find_Q_red(matrix, Xmatrix, noofclus):
    """
    Input:
        matrix=2d-array,starting matrix to be clustered
        Xmatrix=2d-array,set of eigen-\Schurvectors 
        noofclus=int, number of cluster (metastable states), dimension of the final reduced matrix
    Output:
        Q_red= reduced Q matrix"""
    #3
    pinvX=np.copy(np.linalg.pinv(Xmatrix))
    Lambda_sec=np.matmul(np.matmul(pinvX,matrix),Xmatrix)
    
    #4
<<<<<<< HEAD
    A=optimizeMetastab(Xmatrix,noofclus)
    Q_red=np.matmul(np.matmul(np.linalg.inv(A),Lambda_sec),A)
    return(Q_red)
=======
    Chi=optimizeMetastab(Xmatrix,noofclus)
    Q_red=np.matmul(np.matmul(np.linalg.pinv(Chi),Lambda_sec),Chi)
    return(Q_red)
    
    #%%
>>>>>>> origin/renata
