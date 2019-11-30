#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:37:32 2019

@author: bzfsechi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:43:24 2019

@author: renat
"""

#from cmdtools.src import cmdtools.estimation.galerkin
#import cmdtools.analysis

import numpy as np

from scipy import linalg

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import cmdtools as cmd

from cmdtools.analysis import pcca

from cmdtools.estimation import galerkin_taus_all, Newton_Npoints

#from cluster_by_isa import find_new_Q
# Considering your module contains a function called my_func, you could import it:
#from lalala import counttransition_trjs,set_rate_matrix
#from SRSchur import sort_real_schur


#%%
#load trajectory    

diala = np.load("./alanine-dipeptide-3x250ns-backbone-dihedrals/arr_0.npy")
#centers= np.loadtxt("./Kmean_diala75.txt")
#centers= np.sort(centers, axis= 0)
#%%
#estimate centers from kmeans

Kmeans_centers = KMeans(n_clusters=20, init="random").fit(diala)
centers = Kmeans_centers.cluster_centers_
centers= np.sort(centers, axis= 0)
#np.savetxt("Kmean_diala75.txt",centers)
#%%%
sigma = galerkin_taus_all.find_bandwidth(diala, centers, 15 ) 
#%%
# estimate with Galerkin discretization the transfer operator
#estimate 
Koopman_mtx = galerkin_taus_all.propagator_tau(diala, centers,sigma, 3)

#%%
chi = pcca.pcca(Koopman_mtx[0,:,:], 5)
#visualize pcca+ Koopman matrix 
plt.imshow( chi, aspect= "auto")
plt.colorbar()
plt.show()
    
#%%
colors = ["b","r","gold","g", "m"]
for i in range(20):
    #print(colors[np.argmax(chi[i,:])])
    plt.scatter(centers[i,0], centers [i,1], color = colors[np.argmax(chi[i,:])])
plt.show()
    
#%%
#estimate generator

Infgen = Newton_Npoints.Newton2(Koopman_mtx, 1., 1.)
#%%
#for i in range(5):
#    QQ = Infgen_diff_sigma[i]
##    print(np.sum(QQ, axis = 1))
#    #print(pcca.schur(QQ, output="real"))
#    plt.imshow( pcca.pcca(QQ,5), aspect= "auto")
#    plt.colorbar()
#    plt.show()
##    
#   # print(np.linalg.eigvals(QQ))
##%%
##apply clustering algorithm 
#
#
#Chi_diff_sigma = []
##for  i in range(3,4):
##   print(i)
#Chi_diff_sigma.append(pcca.pcca(Infgen_diff_sigma[1],5))
##chi = pcca.pcca(Q, 3)
##%%
#for i in range(len(sigma_list)):
#    plt.imshow(Chi_diff_sigma[i], aspect= "auto")
#    plt.title("Chi vectors")
#    plt.colorbar()
##
##%%
#k = Koopman_diff_sigma[1]
#Q_log= np.log(k[0,:,:]) #natural logarithms
#chi_log = pcca.pcca(Q_log, 5)
###%%
##
##plt.imshow(chi_log, aspect= "auto")
##plt.title("Chi_log vectors")
##plt.colorbar()
##
##Q_c = np.linalg.inv(chi_log.T.dot(chi_log))*(chi_log.T.dot(np.matmul(Q_log,chi_log)))
#plt.imshow(chi_log, aspect = "auto")