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
from scipy.linalg import expm, schur
#from cmdtools.analysis import pcca
from cmdtools.estimation import Newton_Npoints, sqra

#%%
def q_doublewell(dim, sigma):
     # number of intervals in [0,1]-space (e.g. 30)
     # inverse temperature for Boltzmann (e.g. 10)

    # grid & flux computation according to Luca Donati
    grid = np.linspace(-2.5, 2.5, dim+1)
    #using the ideas of the presentation of Donati
    m = 1.
    gamma = 1.    
   # sigma = np.sqrt(2/(beta*m*gamma))
    phi = sigma**2/(2*abs(grid[0]-grid[1]))
    # potential
    u = 1/(50)*(grid**2-2)**2
    G = nx.erdos_renyi_graph(dim+1, 0.1)
   # G = nx.les_miserables_graph()
    A = nx.adjacency_matrix(G)
    A = A.toarray()

    A[np.diag_indices(dim+1)] = 0.
  #  A = np.diag(np.ones(dim), k=1)
  
    A = np.triu(A) + np.triu(A).T

    return sqra.sqra(u, A, 2/sigma**2, phi)
example = q_doublewell(100,10)
#%%  
def obtain_q(dim=50, sigma=1):
#    infgen = random(dim, dim, density).toarray()
    infgen = q_doublewell(dim-1, sigma)
#    print(np.shape(infgen))
#  infgen = sqra.sqra()
    infgen[np.diag_indices(dim)] = - np.sum(infgen- np.diagflat(np.diagonal(infgen)), axis=1)
    return infgen#*0.005
def obtain_k(tau_max, step, dim=50, sigma=1):#, noise=False):
    infgen = obtain_q(dim, sigma)
    k_matrix = np.zeros((int(tau_max/step) +1, dim, dim))
    tau_values = np.arange(0, tau_max+1, step)
    #if noise == True:
     #   for i in range(np.shape(k_matrix)[0]):
      #      k_matrix[i, :, :] = expm(tau_values[i]*((infgen)+random(dim, dim, density=0.1)))
       #     k_matrix[i, :, :] = k_matrix[i, :, :]/np.sum(k_matrix[i, :, :], axis=1)
  #  else:
    for i in range(np.shape(k_matrix)[0]):
        k_matrix[i, :, :] = expm(tau_values[i]*infgen)       
    return(k_matrix, infgen)


#%%
#dw = q_doublewell(100,10)
#%%
q = np.array([[-3., 2., 0., 1., 0., 0.], [2.,-3.,0.5,0.,.5,0.], [0., 0., -3., 2.5, .5, 0.], [0.5, 0.,3., -4., 0., 0.5], [0., 0., 0.5,0.5,-5,4.], [0.,0.25,0.25,0.5,4.,-5.]])
#plt.imshow(q)
tau = np.arange(0,9,step= 1)
k = np.zeros((9, 6,6))
for i in range(np.shape(k)[0]):
    k[i,:,:] = expm(tau[i]*q)   
q_new = Newton_Npoints.Newton_N(k, 1,0)
#sigma = subspace_angles(pcca.schurvects(q,4),pcca.schurvects(q_new,4))
#%%
#def sort_schur(T, n=2):
#    e = np.sort(np.linalg.eigvals(T))
#
#    v_in  = np.real(e[-n])
#    v_out = np.real(e[-(n + 1)])
#    cutoff = (v_in+v_out)/2
#    E, X, _ = schur(T, sort=lambda x: np.real(x) > cutoff)
#    return(E, X)
#    
#def compare_eigenvals(m1, m2, n=2):
#    evals1 = sort_schur(m1, n)[0]
#    evals2 = sort_schur(m2, n)[0]
#    return(np.linalg.norm(evals1-evals2))
#s1 = sort_schur(q, n=3)[0]
#s2 = sort_schur(q_new, n=3)[0]
#%%
def dummy_compare(m1,m2):
    ew1 = np.sort(np.real(np.linalg.eigvals(m1)))
    ew2 = np.sort(np.real(np.linalg.eigvals(m2)))
    return(abs(ew1-ew2), ew1,ew2)
#def find_min_error(tau_max=2, step=1, dim=50, sigma=1):
#    infgen = obtain_q(dim, sigma)
#    k_matrix = np.zeros((int(tau_max/step) +1, dim, dim))
#    tau_values = np.arange(0, tau_max+1, step)
#    for i in range(np.shape(k_matrix)[0]):
#        k_matrix[i, :, :] = expm(tau_values[i]*infgen)  
#    infgen_new = Newton_Npoints.Newton_N(k_matrix, 1.,0)
#    err_old = (dummy_compare(infgen_new, infgen)[0])[-2] 
#   # print(err_old)
#    err_new = err_old +1.
#    while abs(err_old - err_new)>10**(-8):
#        err_old = err_new
#        tau_max+=1
#        k_matrix = np.vstack((k_matrix, expm(tau_max*infgen)[None]))
#        infgen_new = Newton_Npoints.Newton_N(k_matrix, 1.,0)
#        err_new = (dummy_compare(infgen_new, infgen)[0])[-2]
#    return(tau_max-1, dummy_compare(infgen_new, infgen)[0], k_matrix, infgen, infgen_new)
    #%%
def find_min_error(tau_max=2, step=1, dim=50, sigma=1):
    infgen = obtain_q(dim, sigma)
    k_matrix = np.zeros((int(tau_max/step) +1, dim, dim))
    tau_values = np.arange(0, tau_max+1, step)
    for i in range(np.shape(k_matrix)[0]):
        k_matrix[i, :, :] = expm(tau_values[i]*infgen)  
    infgen_new = Newton_Npoints.Newton_N(k_matrix, 1.,0)
    err_old = (dummy_compare(infgen_new, infgen)[0])[-2] 
   # print(err_old)
    err_new = err_old +1.
#    while abs(err_old - err_new)>10**(-15):
#        if err_new < err_old: 
#            err_old = err_new
#        else:
#            break
#        tau_max+=1       
    return(dummy_compare(infgen_new, infgen)[0], k_matrix, infgen, infgen_new)
    
    
    #%%
#print(dummy_compare(q,q_new)[0])
#%%
#k , q = obtain_k(6, 1, dim=100, sigma = 10, noise =False) 
res = find_min_error(2,1,20, .15)
#print(extract(res))
lala = (dummy_compare(res[2],res[3])[0])/abs(dummy_compare(res[2],res[3])[1])
print(lala[-2])
#alpha, q_NEW = compare_eigenspace(k, q)
#%%
 #test for statistics of error with these parameters
#
#for k in range(2,9):
for j in np.arange(10,200, step=10):
    arrays = []
    for i in range(100):
        res = find_min_error(11,1,j,.2)
        arrays.append(res)
#    
    np.savez("test_deg11_dim%(dim)d_s015"%{'dim':j}, *arrays)
#    np.savez("test_deg%(deg)d_dim%(dim)d_s015"%{'deg':k, 'dim':j}, *arrays)
    
#%%
#file = np.load("test_newton_dim30_b1.npz", allow_pickle=True)
#snd_eigval = []
#for i in range(100):
#    snd_eigval.append(((file["arr_%d"%i])[1])[-2])
#    #%%
#plt.hist(snd_eigval, bins = 20)
#tryk = file['arr_0'][2]