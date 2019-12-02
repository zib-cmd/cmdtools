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
centers= np.loadtxt("./Kmean_diala75.txt")
#%%
centers= np.sort(centers, axis= 0)
#%%
#estimate centers from kmeans

#Kmeans_centers = KMeans(n_clusters=75, init="random").fit(diala)
#centers = Kmeans_centers.cluster_centers_
#np.savetxt("Kmean_diala75.txt",centers)
#%%
plt.scatter(centers[:,0], centers[:,1])

#%%%
#sigma_list=[]
#for i in [28,29,30,31]:
 #   sigma_list.append(galerkin_taus_all.find_bandwidth(diala, centers, i )) 
sigma_list=[0.06836646929784461,0.18296163700268772, 0.37898252218020795,0.7074982602106278, 1.4464302830941358,1.5]
#%%
# estimate with Galerkin discretization the transfer operator
#estimate 
Koopman_diff_sigma = []
for  i in range(len(sigma_list)):
    Koopman_diff_sigma.append(galerkin_taus_all.propagator_tau(diala, centers,sigma_list[i], 3)) 
    
#%%
for i in range(5):
    KK = Koopman_diff_sigma[i]
    plt.imshow( pcca.pcca(KK[0,:,:],5), aspect= "auto")
    plt.colorbar()
    plt.show()
    
    #print(np.linalg.eigvals(KK[0,:,:]))
#%%
#estimate generator

Infgen_diff_sigma = []
for  i in range(len(sigma_list)):
    Infgen_diff_sigma.append(Newton_Npoints.Newton2(Koopman_diff_sigma[i], 1., 1.))
#%%
for i in range(5):
    QQ = Infgen_diff_sigma[i]
#    print(np.sum(QQ, axis = 1))
    #print(pcca.schur(QQ, output="real"))
    plt.imshow( pcca.pcca(QQ,5), aspect= "auto")
    plt.colorbar()
    plt.show()
#    
   # print(np.linalg.eigvals(QQ))
#%%
#apply clustering algorithm 


Chi_diff_sigma = []
#for  i in range(3,4):
#   print(i)
Chi_diff_sigma.append(pcca.pcca(Infgen_diff_sigma[1],5))
#chi = pcca.pcca(Q, 3)
#%%
for i in range(len(sigma_list)):
    plt.imshow(Chi_diff_sigma[i], aspect= "auto")
    plt.title("Chi vectors")
    plt.colorbar()
#
#%%
k = Koopman_diff_sigma[1]
Q_log= np.log(k[0,:,:]) #natural logarithms
chi_log = pcca.pcca(Q_log, 5)
##%%
#
#plt.imshow(chi_log, aspect= "auto")
#plt.title("Chi_log vectors")
#plt.colorbar()
#
#Q_c = np.linalg.inv(chi_log.T.dot(chi_log))*(chi_log.T.dot(np.matmul(Q_log,chi_log)))
plt.imshow(chi_log, aspect = "auto")