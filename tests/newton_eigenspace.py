#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:43:59 2020

@author: bzfsechi
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import random
from scipy.linalg import expm, subspace_angles, schur
#from cmdtools.analysis import pcca
from cmdtools.estimation import Newton_Npoints, sqra
from cmdtools.analysis import pcca
#%%
def q_doublewell(dim,beta):
     # number of intervals in [0,1]-space (e.g. 30)
     # inverse temperature for Boltzmann (e.g. 10)

    # grid & flux computation according to Luca Donati
    grid = np.linspace(-2, 2, dim+1)
    phi = dim**2 / beta / 9

    # potential
    u = np.exp(-(grid**2-2)**2)
    G = nx.erdos_renyi_graph(dim+1, 0.1)
   # G = nx.les_miserables_graph()
    A = nx.adjacency_matrix(G)
    A = A.toarray()

    A[np.diag_indices(dim+1)] = 0.
  #  A = np.diag(np.ones(dim), k=1)
   # A = A + A.T

    return(sqra.sqra(u, A, beta, phi))
  
def obtain_q(dim=50, beta=1):
#    infgen = random(dim, dim, density).toarray()
    infgen = q_doublewell(dim-1, beta)
#    print(np.shape(infgen))
#  infgen = sqra.sqra()
    infgen[np.diag_indices(dim)] = - np.sum(infgen- np.diagflat(np.diagonal(infgen)), axis=1)
    return(infgen*0.005)
def obtain_k(tau_max, step, dim=50, beta=1, noise=False):
    infgen = obtain_q(dim, beta)
    k_matrix = np.zeros((int(tau_max/step) +1, dim, dim))
    tau_values = np.arange(0, tau_max+1, step)
    if noise==True:
        for i in range(np.shape(k_matrix)[0]):
            k_matrix[i,:,:] = expm(tau_values[i]*((infgen)+random(dim,dim,density = 0.1)))
            k_matrix[i,:,:] = k_matrix[i,:,:]/np.sum(k_matrix[i,:,:], axis=1)
    else:
        for i in range(np.shape(k_matrix)[0]):
            k_matrix[i,:,:] = expm(tau_values[i]*infgen)       
    return(k_matrix, infgen)
def compare_eigenspace(k_matrix, infgen):
    newton_infgen = Newton_Npoints.Newton_N(k_matrix, 1, 0)
    #schur_infgen = schur(infgen)[1]
    #schur_newton_infgen = schur(newton_infgen)[1]
    #print(schur_infgen, schur_newton_infgen)
    schur_infgen = pcca.schurvects(infgen, 3)
    schur_newton_infgen = pcca.schurvects(newton_infgen, 3)
  
    angle = subspace_angles(schur_infgen, schur_newton_infgen)
    return(angle, newton_infgen, schur_infgen, schur_newton_infgen)
    
#k , q = obtain_k(6, 5, dim=100, beta = 10, noise =False) 

#alpha, q_NEW = compare_eigenspace(k, q)

#%%
dw = q_doublewell(100,10)
#%%
q = np.array([[-3., 2., 0., 1., 0., 0.],[2.,-3.,0.5,0.,.5,0.],[0., 0., -3., 2.5, .5, 0.],[0.5, 0.,3., -4., 0., 0.5],[0., 0., 0.5,0.5,-5,4.],[0.,0.25,0.25,0.5,4.,-5.]])
plt.imshow(q)
tau = np.arange(0,9,step= 1)
k = np.zeros((9, 6,6))
for i in range(np.shape(k)[0]):
    k[i,:,:] = expm(tau[i]*q)   
q_new = Newton_Npoints.Newton_N(k, 1,0)
#beta = subspace_angles(pcca.schurvects(q,4),pcca.schurvects(q_new,4))
#%%
def sort_schur(T, n=2):
    e = np.sort(np.linalg.eigvals(T))

    v_in  = np.real(e[-n])
    v_out = np.real(e[-(n + 1)])
    cutoff = (v_in+v_out)/2
    E, X, _ = schur(T, sort=lambda x: np.real(x) > cutoff)
    return(E, X)
    
def compare_eigenvals(m1, m2, n=2):
    evals1 = sort_schur(m1)[0]
    evals2 = sort_schur(m2)[0]
    return(np.linalg.norm(evals1-evals2))
s1 = sort_schur(q, n=3)[0]
s2 = sort_schur(q_new, n=3)[0]
#%%
def dummy_compare(m1,m2):
    ew1 = np.sort(np.linalg.eigvals(m1))
    ew2 = np.sort(np.linalg.eigvals(m2))
    return( ew1-ew2, ew1,ew2)
    #%%
print(dummy_compare(q,q_new)[0])