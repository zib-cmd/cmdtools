#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:53:25 2019

@author: bzfsechi
"""

import numpy as np
import copy
from fillA import fillA
#function optval=objective(alpha,X,N,k,A)
#
#% Hier wird die Zielfunktion berechnet, die die Spur der
#% Massenmatrix maximiert. 
#
#global NORMA  
#global OPT
#
#
#
#
#%Bestimmung der vollstaendigen Matrix A
#for i=1:k-1
#    for j=1:k-1
#      A(i+1,j+1)=alpha(j + (i-1)*(k-1));
#    end
#end
#
#normA=norm(A(2:k,2:k));
#
#%A zulässig machen
#A=fillA(A, X, N, k );
#nc=size(X,2);
#
#    J2=trace(diag(1./A(1,:))*A'*A);  %traceS
#    optval=J2;
#
#optval=-optval;
#%%
def objective(alpha,X,N,k,A):
    
    global NORMA
    global OPT
    k=int(k)
    for i in range(k-1):
        for j in range(k-1):
            A[i+1,j+1]=alpha[j+(i-1)*(k-1)]
    normA=np.linalg.norm(A[1:k,1:k])
    A=fillA(A,X,N,k)
    nc=  np.shape(X)[1]
    Adiag=np.diag(1./(A[0,:]))
    mul1A=np.matmul(Adiag,np.transpose(A))
    optval=-np.trace(np.matmul(mul1A,A)) # minus trace S
         
    return(optval)