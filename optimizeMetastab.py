#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:09:47 2019

@author: bzfsechi
"""

import numpy as np
import copy
from opt_soft import opt_soft
import matplotlib.pyplot as plt
#function chi=optimizeMetastab(X,n)
#%    X:          schur vectors
#%    n:          number of Clusters
#
#global ITERMAX
#
#
#ITERMAX=500;    %500
#
#N=size(X,1);
#opts.disp=0;
#EVS=X(:,1:n);
#
#evs=reshape(EVS,N*n,1);
#[chi,val]=opt_soft(evs, N, n);
#%%
def optimizeMetastab(X,n):
    """Input: 
        X: Schur vectors
        n:number of clusters
        Output:
            chi, matrix with the new vectors"""
    global ITERMAX
    ITERMAX=500
    N=np.shape(X)[0]
    
    EVS=X[:,:n]
   # evs=np.reshape(EVS, (N*n,1))
    chi,val=opt_soft(EVS,N,n)
    return(chi)
    #%%
plt.figure(figsize=(3,3))
plt.imshow(optimizeMetastab(evs,3), aspect="auto")
plt.colorbar()
#%%