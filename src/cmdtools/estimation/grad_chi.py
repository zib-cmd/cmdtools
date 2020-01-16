#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:06:20 2020

@author: bzfsechi
"""
import numpy as np
from scipy.linalg import pinv
 #%%
# function [gradS]=Grad(q, S, n)
#
#    %% scale
#    scale = std(q');
#    q = diag(1./scale)*q;
#
#
#    A=zeros(size(q,1),size(q,1));
#    bS=zeros(size(q,1),1);
#    v=q-(q(:,n)*ones(1,size(q,2)));
#    dist=sum(v.*v);
#
#    alpha = 1/size(q,1);
#    while (1==1)
#        w=exp(-dist*alpha);
#        if (sum(w) < (size(q,1)+1)) 
#          break;
#        end;
#        alpha=alpha + 1/size(q,1);
#    end;
#    
#    for i=1:size(q,1)
#        bS(i) = sum( w.*v(i,:).*(S-S(n))); 
#        for j=1:size(q,1)
#           A(i,j)=sum(w.*v(i,:).*v(j,:));
#        end
#    end
#    bS;
#    M=pinv(A);
#    gradS=diag(scale)*(M*bS);

def grad_at_point(q, S, n):
    """ Load centers, S and the n at with calculate the gradient at point n.
    Input : 
        q = array, the rows represents different dimensions
        S = arr, the vector to calculate the gradient
        n =  the point of the loaded one at with compute the gradient
    Output : 
        gradS = the gradient at the point n.
    Comment : 
        Make sure you start counting by zero when giving the n.
        """
  
    scale = np.std(q.T, axis=0)
    q = np.matmul(np.diag(1./scale), q)
    row_q = np.shape(q)[0]
   
    A = np.zeros((row_q, row_q))
    bS = np.zeros((row_q, 1))
    
    v = q-(np.matmul(np.reshape(q[:, n], (2, 1)), np.ones((1, np.shape(q)[1]))))
  
    dist = np.sum(v * v, axis=0)
    
    alpha = float(1/row_q)
    while True:
        
        w = np.exp(-dist*alpha)
     
        if sum(w) < (row_q +1.):
            break
        
        alpha += 1./row_q
  
    for i in range(row_q):
        bS[i] = sum(w*v[i, :]*(S-S[n]))
        for j in range(row_q):
            A[i, j] = sum(w* v[i, :]* v[j, :]) 
 
    M = pinv(A)
    gradS = np.matmul(np.diag(scale), (np.matmul(M, bS)))
    
    return gradS 


def grad_chi(q, S):
    """Iterate function 'grad_at_point' to compute the gradient of the vector
    S at all the points listed in q.
    
    Input:
        q = array, the rows represents different dimensions
        S = arr, the vector whose calculate the gradient
        
    Output : 
        grad_array = the gradient values of chi at the points listed in q. The
                     rows represent the same dimensions (e.g. first row is
                     x coordinate, second row is y coordinate etc.).

        """
    grad_array = np.zeros((2, 1))
    for i in range(np.shape(q)[1]):
        grad_array = np.hstack((grad_array, grad_at_point(q, S, i)))
    return(grad_array[:, 1:])        
