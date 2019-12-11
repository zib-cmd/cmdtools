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
from scipy.linalg import expm, subspace_angles
from scipy.sparse import random
from cmdtools.estimation import Newton_Npoints
from cmdtools.analysis import pcca

def compare_subspace(Q_init, Q_comp, n= 3):
    Schur_init = pcca.schurvects(Q_init,n)
    Schur_comp = pcca.schurvects(Q_comp,n)
    return(Schur_init, Schur_comp,np.rad2deg(subspace_angles(Schur_init, Schur_comp)[0]))
    
def compare_op_norm():
    pass
    

def make_sparse_Q(dim, dens = 0.1):
    Q__ = random (dim,dim, density = dens)
    Q = Q__.A
    for i in range(np.shape(Q)[0]):
        Q[i,i] = - np.sum(Q[i,:]) + Q[i,i]
    return(Q)


def smt(infgen):
    err_new = 1000.  
    err_old = 1.
    tau = int(0)
    T = np.array([expm(infgen*0.)])
    n = 2
    while  abs(err_new-err_old) > 10**(-3) :
        tau += 1
        #print(tau)
        T = np.concatenate((T, [expm(infgen*tau)]))
        Q_ex = Newton_Npoints.Newton_N(T,1.,0.)
        
        try:
            
            err_new = compare_subspace(infgen, Q_ex,n)[2]#np.mean(abs(infgen - Q_ex))
            err_old = err_new
        except AssertionError:
            n += 1
            continue
        except IndexError:
            break
        #print(err_new, err_old)
    return (np.shape(infgen)[0], err_new, tau )#,(err_new/np.shape(infgen)[0]) )
        
#%%
 
#    
def __testNewton__(min_points = 50, max_points = 100 , step_ = 10):
    list_out = []
    for j in range(2,10):
        max_error = np.array([0.,0.,0.])
        for i in range(50, int(max_points),int (step_)):
            
            max_error= np.vstack((max_error,smt(make_sparse_Q(i, 0.1*j))))
        list_out.append(max_error[1:,:])
    return list_out
       
#%%
a = __testNewton__()

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
#%%
#problem find a way to vary n after the assertion error
#%%
