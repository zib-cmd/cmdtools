#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:16:36 2019

@author: bzfsechi
"""

import numpy as np
import copy 
from scipy.optimize import fmin 
#fmin uses the Nelder-Mead Simplex algorithm, as well as Matlab's fminsearch
from indexsearch import indexsearch
from fillA import fillA
from objective import objective

#%%
def opt_soft(evs,Ndim,kdim):
    
    """"% Parameter:
%    evs:        Eigenvektoren
%                evs[l+Ndim*j] l-te Komponete vom j-ten EV
% 	       Dimension Ndim*kdim
%    chi:        (Ausgabe) Soft-characteristische Funktionen
%                chi[l+Ndim*j] l-te Komponetne von j-ter Funktion
% 	       Dimension: Ndim*kdim
%    Ndim:       Anzahl der Boxen
%    kdim:       Anzahl der Cluster"""
    global NORMA
    global ITERMAX
    ITERMAX=500
    N=Ndim
    k=int(kdim)
    flag=1
    if k>1:
        index=np.zeros(k)
        A=np.zeros((k,k))
        EVS=np.zeros((N,k))
        for l in range(N):
            for j in range(k):
                #two versions: the one with unravel_index follows the original MATLAB code.                 #EVS[l,j]=evs[]# in the original Matlab- code this works with linear indeces, this is a way to do the same on Python
                EVS[l,j]=evs[l,j]# see conversion of matlab-python indices
#                EVS[l,j]=evs(np.unravel_index(l*N+j))
        index=indexsearch(EVS,N,k)
        index=index.astype(int)
        
        
        A=EVS[index,:]
        A=np.linalg.pinv(A)
        NORMA=np.linalg.norm(A[1:k,1:k])
        
        minchi_start=np.amin(np.matmul(EVS,A))
        print("Starting minChi:", minchi_start)
        if flag>0:
            alpha=np.zeros((k-1)**2)
            for i in range(k-1):
                for j in range(k-1):
                    alpha[j+(i-1)*(k-1)]=A[i+1,j+1]
            startval=-objective(alpha,EVS,N,k,A)
            print("Start value for A:", startval)
            #options=###### to do max iter
#            find_alpha= lambda alpha: objective(alpha,EVS,N,k,A)
            alpha, fret,iters,funcalls, warnflag=fmin(objective, x0=alpha,args=(EVS,N,k,A),maxiter=ITERMAX, full_output=1)###### to do
            output={"Number of iterations performed (max_default=500)": iters, "Number of functioncalls":funcalls}
            endval=-fret
            for i in range(k-1):
                for j in range(k-1):
                    A[i+1,j+1]=alpha[j+(i-1)*(k-1)]
            A=fillA(A,EVS,N,k)
            
        else:
            fret=-1
        
        chi=np.matmul(EVS,A)
    else:#special case k=1
        if flag==1:
            fret=1.0
        else:
            fret=-1.0
        chi=np.ones(N)
    return (chi,A,fret)