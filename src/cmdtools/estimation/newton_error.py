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

def get_vectors(M_, no_clus=1):
    #while abs(np.shape(M_)[0]-no_clus) > 1:
    try:
        Schur_init = pcca.schurvects(M_, no_clus)
    except AssertionError:
        Schur_init = get_vectors (M_, no_clus+1)
    return( Schur_init)
    
def compare_subspace(Q_init, Q_comp, n= 2):

    Schur_init = get_vectors(Q_init, n)

    Schur_comp = get_vectors(Q_comp, n)
            
    n_i = np.shape(Schur_init)[1]
    n_c = np.shape(Schur_comp)[1]
        
    return(np.rad2deg(subspace_angles(Schur_init[:,:min(n_i, n_c)], Schur_comp[:,:min(n_i, n_c)])[0]))
    
def compare_op_norm():
    pass
    

def make_sparse_Q(dim, dens = 0.1):
    Q__ = random (dim,dim, density = dens)
    Q = Q__.A
    for i in range(np.shape(Q)[0]):
        Q[i,i] = - np.sum(Q[i,:]) + Q[i,i]
    return(Q)

def check_Q (Q_init, dens ):
    shape = np.shape(Q_init)[0]
    try: 
        Schur_pre_init = get_vectors(Q_init, 2)
    except IndexError:
        Q_init = make_sparse_Q(shape, dens)
    return Q_init

def smt(infgen):
    err_new = 1000.  
    err_old = 1.
    tau = int(0)
    T = np.array([expm(infgen*0.)])
  
    while  abs(err_new-err_old) > 10**(-12) :
        
        tau += 1
        #print(tau)
        T = np.concatenate((T, [expm(infgen*tau)]))
        Q_ex = Newton_Npoints.Newton_N(T,1.,0.)
        
        err_new = compare_subspace(infgen, Q_ex)#[2]#np.mean(abs(infgen - Q_ex))
        err_old = err_new

    return (np.shape(infgen)[0], err_new, tau )#,(err_new/np.shape(infgen)[0]) )

def wrap(dim, dens = 0.1):
    Q_first = make_sparse_Q(dim, dens)
    Q_second = check_Q(Q_first, dens)
    
    if (Q_first == Q_second).all():
        return Q_first
    else:
        
     
#%%
 
#    
def __testNewton__(min_points = 50, max_points = 100 , step_ = 10):
    list_out = []
    for j in range(5,10):
        max_error = np.array([0.,0.,0.])
        for i in range(min_points, int(max_points),int (step_)):
            max_error= np.vstack((max_error,smt(make_sparse_Q(i, 0.1*j))))
        
        list_out.append(max_error[1:,:])
    return list_out
       
#%%
a = __testNewton__(70,86, step_ = 5)

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
