#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 11:38:43 2019

@author: bzfsechi
"""

"""
Created on Mon Dec  9 14:01:32 2019

@author: bzfsechi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.sparse import random
from cmdtools.estimation import Newton_Npoints


def make_sparse_Q(dim, dens = 0.1):
    Q__ = random (dim,dim, density = dens)
    Q = Q__.A
    for i in range(np.shape(Q)[0]):
        Q[i,i] = - np.sum(Q[i,:]) + Q[i,i]
    return(Q)


def smt(infgen):
    err_new = 0.  
    err_old = 1.
    tau = int(0)
    T = np.array([expm(infgen*0.)])
    
    while  abs(err_new-err_old) > 10**(-4) :
        tau += 1
        T = np.concatenate((T, [expm(infgen*tau)]))
        Q_ex = Newton_Npoints.Newton_N(T,1.,0.)
        err_old = err_new
        err_new = np.mean(abs(infgen - Q_ex))
     
    return( np.shape(infgen)[0], err_new, tau,  (err_new/np.shape(infgen)[0]) )
        

    
def __testNewton__(max_points = 100 , step_ = 10):
    list_out = []
    for j in range(1,10):
        max_error = np.array([0.,0.,0.,0.])
        for i in range(10, int(max_points),int (step_)):
            max_error= np.vstack((max_error,smt(make_sparse_Q(i, 0.1*j))))
        list_out.append(max_error[1:,:])
    return list_out
       
#%%
a = __testNewton__(200)

plt.figure()
for i in range(len(a)):
    array = a[i]
    plt.plot(array[:,0], array[:,1], marker= "o", label = np.round(0.1*(i+1),2))
plt.xlabel("no. of basis points")
plt.ylabel("mean abs error Q_comp-Q_initial")
plt.title("error development vs dimensionality of sparse infgen")
plt.legend(bbox_to_anchor=(1., 1.05))

#plt.savefig("test_newton_max_error.pdf",bbox_inches="tight")
plt.show()
