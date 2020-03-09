#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:15:51 2020

@author: bzfsechi
"""
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import scipy

#%%
# take mean of second eigenvalue error and put into a list
def extract(arr):
    eig1= np.sort(np.real(np.linalg.eigvals(arr[2])))[-2]
#    print(eig1)
    eig2= np.sort(np.real(np.linalg.eigvals(arr[3])))[-2]
    print(eig1,eig2)
    return(abs(abs(eig1-eig2)/eig1))
    
#mean_diff1 = []
#for j in np.arange(10,200, step=10):
#    file = np.load("test_newton_dim%d_b1.npz"%j, allow_pickle=True)
#    snd_eigval = []
#    for i in range(100):
#        snd_eigval.append(extract(file["arr_%d"%i]))
#    mean_diff1.append(mean(snd_eigval))
mean_diff10 = []
for j in np.arange(10,200, step=10):
    file = np.load("test_deg_dim%d_b10.npz"%j, allow_pickle=True)
    snd_eigval = []
    for i in range(100):
        snd_eigval.append(extract(file["arr_%d"%i]))
    mean_diff10.append(mean(snd_eigval))
#mean_diff15 = []
#for j in np.arange(10,200, step=10):
#    file = np.load("test_newton_dim%d_b15.npz"%j, allow_pickle=True)
#    snd_eigval = []
#    for i in range(100):
#        snd_eigval.append(extract(file["arr_%d"%i]))
#    mean_diff15.append(mean(snd_eigval))
#mean_diff5 = []
#for j in np.arange(10,200, step=10):
#    file = np.load("test_newton_dim%d_b5.npz"%j, allow_pickle=True)
#    snd_eigval = []
#    for i in range(100):
#        snd_eigval.append(extract(file["arr_%d"%i]))
#    mean_diff5.append(mean(snd_eigval))
#    #%%
#mean_diff2 = []
#for j in np.arange(10,200, step=10):
#    file = np.load("test_newton_dim%d_b2.npz"%j, allow_pickle=True)
#    snd_eigval = []
#    for i in range(100):
#        snd_eigval.append(extract(file["arr_%d"%i])[-2])
#    mean_diff2.append(mean(snd_eigval))

    #%%
plt.scatter(np.arange(10,200, step=10), mean_diff1, label =1)
plt.scatter(np.arange(10,200, step=10), mean_diff10,label =10)
plt.scatter(np.arange(10,200, step=10), mean_diff5, label =5)
plt.scatter(np.arange(10,200, step=10), mean_diff15, label =15)
plt.xlabel('dim')
plt.ylabel('($\lambda_i-\lambda_f)$')
#plt.xlim(30,200)
plt.ylim(0)
plt.legend()