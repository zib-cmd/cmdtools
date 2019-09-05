#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:02:47 2019

@author: bzfsechi
"""

import numpy as np
import copy
from scipy import linalg
#%%
def orthogon(EVS,pi,Lambda,k, key):
    """Input:
        EVS=2D array, Schur-/eigenvectors
        pi=array, distribution
        Lambda=array,eigenvalues (Schur diagonal values)
        k=Schur/eigenvectors with same eigenvalue-- you want to orthogonalize
        key= float, 0. for generator eigenvalues, 1. for propagator /transfer operator 
    Output: 
        EVS, orthogonalized
        """
    N=np.shape(EVS)[0]
    perron=1.
    pi=copy.deepcopy(pi/np.sum(pi))
   
    for i in range(k):
        den=np.dot((EVS[:,k]*pi),EVS[:,k])
        
        EVS[:,k]=copy.deepcopy(EVS[:,k]/(np.sqrt(den*np.sign(EVS[0,k]))))
    for i in range(k):
        if np.round(Lambda[i],8)==float(key):
            perron+=1.
        else:
            break
    if perron>1.:
        
        maxscal=0.0
        for i in range(int(perron)):
            scal=np.sum(pi*EVS[:,i])
       
            if np.abs(scal)>maxscal:
                maxscal=np.abs(scal)
                maxi=i
        EVS[:,maxi]=EVS[:,0]
        EVS[:,0]=np.ones(N)
        EVS[np.argwhere(pi<=0.),:]=0.
        
        for i in range(1,k):
            for j in range(i-1):
                scal=np.dot((EVS[:,j]*pi),EVS[:,i])
               
                EVS[:,i]=EVS[:,i]-scal*EVS[:,j]
            sumval=np.sqrt(np.dot((EVS[:,i]*pi),EVS[:,i]))
            EVS[:,i]=EVS[:,i]/sumval
            
        return(EVS)
                

#%%
        
#Example
matrixA=np.reshape(np.random.uniform(0.,1.,100), (10,10))
for i in range(10):
    matrixA[i,:]=matrixA[:,i]/np.sum(matrixA[:,i])
        #%%
eigw,eig1=linalg.schur(matrixA)

#%%              
newevs=orthogon(eig1, np. ones(10)*0.1, np.diag(eigw), 3, 1.)