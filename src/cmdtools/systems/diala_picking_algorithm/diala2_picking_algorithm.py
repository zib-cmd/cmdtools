#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:37:32 2019

@author: bzfsechi
"""
#from cmdtools.src import cmdtools.estimation.galerkin
#import cmdtools.analysis

import numpy as np

#from scipy import linalg
#from scipy.spatial import distance

import matplotlib.pyplot as plt
from cmdtools.analysis import pcca

from cmdtools.estimation import galerkin_taus_all, Newton_Npoints, picking_algorithm


#from cluster_by_isa import find_new_Q
# Considering your module contains a function called my_func, you could import it:
#from lalala import counttransition_trjs,set_rate_matrix
#from SRSchur import sort_real_schur
#%%
#load trajectory    
diala = np.load("arr_0.npy")
#centers= np.loadtxt("./Kmean_diala150.txt")
centers = picking_algorithm.picking_algorithm(-np.pi, np.pi, 2, 75)[1]

centers= centers[centers[:, 0].argsort()]
#sigma_list = np.loadtxt("sigmas_10to90_randomuniform.txt")
#%%

#estimate centers from 
#centers = np.delete(centers , [32,40,59,60,55,61,62 ,63], 0)
#plt.scatter(centers[:,0], centers[:,1])
#plt.xlim(-np.pi, np.pi)
#plt.ylim( -np.pi, np.pi)
#plt.show()
#%%
#np.savetxt("Kmean_diala250.txt", centers)
#%%
# estimate with Galerkin discretization the transfer operator
#estimate 
Koopman_mtx, m = galerkin_taus_all.propagator_tau(diala, centers, 0.1, 4)
#%%
def strip_bad(counts_tensor):
    """Find the lines in which the selfoverlap of the basis functions is not 
    good, strip those basis points from the computed transition matrix. 
    Input:
        
    Output:
        """
    S = counts_tensor[0,:,:]
    to_keep = []
    for i in range(np.shape(S)[0]):
        #if np.argmax(S[i,:]) == i:
        if S[i,i] > 0.2:
            to_keep.append(i)
                      
    #for j in range(np.shape(counts_tensor)[0]):
     #   counts_tensor[j,:,:] = utils.rowstochastic(counts_tensor[j,:,:])  
    temp = counts_tensor[:,to_keep,:]
    return temp [:,:,to_keep], to_keep

#%%
Koopman_mtx2, centers_kept = strip_bad(Koopman_mtx)
#%%
for k in range(5):
    Koopman_mtx2[k, :, :] = Koopman_mtx2[k, :, :]/np.sum(Koopman_mtx2[k, :, :], axis = 1)
    #%%
S_ = Koopman_mtx2[0, :, :]

#np.sum(Koopman_mtx[0,:,:], axis= 1)
#%%
chi = pcca.pcca(Koopman_mtx2[1, :, :], 4, S_)
#visualize pcca+ Koopman matrix 
plt.imshow( chi, aspect= "auto")
plt.colorbar()
plt.show()
    
    
#%%
#estimate generator

Infgen = Newton_Npoints.Newton_N(Koopman_mtx2, 1., 0)
chi_infgen= pcca.pcca(Infgen,4, S_)

#%%

#plt.imshow(Infgen)
#plt.colorbar()
#plt.title("Infgen")
#plt.show()
##%%
#plt.imshow(chi_infgen, aspect= "auto")
#plt.colorbar()
#plt.show()
#%%
#for i in range(4):
#    plt.plot(chi_infgen[:,i])
#plt.show()
    
#
#%%

colors = ["b","r","gold","g", "m", "purple"]
centers_1 = centers[centers_kept, :]
#plt.hist2d(diala[:,0],diala[:,1], 100)
for i in range(np.shape(chi_infgen)[0]):
    #print(colors[np.argmax(chi[i,:])])
    plt.scatter(centers_1[i,0], centers_1[i,1], color = colors[np.argmax(chi_infgen[i,:])])
plt.xlabel("$\Phi$ [rad]")
plt.ylabel("$\Psi$ [rad]")
plt.title("Alanine Dipeptide, 6 Metastable States")
plt.xlim(-np.pi, np.pi)
plt.show()
#%%
#%%
Q_c = np.linalg.pinv(chi_infgen).dot(Infgen.dot(chi_infgen))
#%%
plt.scatter(diala[:,0],diala[:,1], marker = ".")