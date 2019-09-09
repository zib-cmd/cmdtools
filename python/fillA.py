#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:17:46 2019

@author: bzfsechi
"""
import numpy as np
import copy
from  scipy.linalg import schur
#function A=fillA(A,EVS,N,k)

#% Bestimmung der ersten Spalte von A durch Zeilensummenbedingung */
#A(2:k,1)=-sum(A(2:k,2:k),2);
#
#% Bestimmung der ersten Zeile von A durch Maximumsbedingung */
#for j=1:k
#    A(1,j)=- EVS(1,2:k)*A(2:k,j);
#    for l=2:N
#        dummy = - EVS(l,2:k)*A(2:k,j);
#        if (dummy > A(1,j))
#            A(1,j) = dummy;
#        end
#    end
#end

#% Reskalierung der Matrix A auf zulaessige Menge */
#A=A/sum(A(1,:));

def fillA(A,EVS,N,k):
    """Rewritten from fillA.m
    Input: A: square-matrix
    EVS: eigenvectors
    K: int, no of eigenvectors
    N: int, no of cells
    Output:
        newf filled A, 2D array"""
    k=int(k)
    N=int(N)
    A[1:k,0]=-np.sum(A[1:k,1:k],1)
    for j in range(k):
        A[0,j]=-np.matmul(EVS[0,1:k],A[1:k,j])
        for l in range(1,N):
            dummy=-np.matmul(EVS[l,1:k],A[1:k,j])
            if dummy>A[0,j]:
                A[0,j]=copy.deepcopy(dummy)
    return(A/np.sum(A[0,:]))
    
    #%%
#AA=np.random.uniform(0,1,100)
#AA=np.reshape(AA,(10,10))
##%%
#for i in range(10):
#    AA[i,:]=AA[i,:]/np.sum(AA[i,:])
#    
##%%
#ews, evs=schur(AA, output="real")
##%%
#fil=fillA(AA,evs,10,3)
#%%
