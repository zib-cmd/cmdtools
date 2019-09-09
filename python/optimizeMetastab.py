#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:09:47 2019

@author: bzfsechi
"""

import numpy as np
import copy
from opt_soft import opt_soft
from scipy import linalg
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
    
    ITERMAX=500
    N=np.shape(X)[0]
    
    EVS=X[:,:n]
   # evs=np.reshape(EVS, (N*n,1))
    chi,A,val=opt_soft(EVS,N,n)
    return(chi,A)
    #%%
#ortho_eig1[:,0]=ortho_eig1[:,0]/ortho_eig1[0,0]
#eigB[:,0]=eigB[:,0]/eigB[0,0]
#
##%%
#chiis,A=optimizeMetastab(ortho_eig1,3)
##%%
#plt.figure(figsize=(3,3))
#plt.imshow(chiis, aspect="auto")
#plt.colorbar()
##%%
matrixB=np.reshape(np.random.uniform(0.,1.,100), (10,10))
for i in range(10):
    matrixB[i,:]=matrixB[:,i]/np.sum(matrixB[:,i])
eigwB,eigB=linalg.schur(matrixB)
#%%
newevs_B=orthogon(eigB, np. ones(10), np.diag(eigwB), 3, 1.)
