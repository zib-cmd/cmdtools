#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:21:02 2019

@author: bzfsechi
"""

import numpy  as np
from scipy import linalg
from optimizeMetastab import optimizeMetastab
from Q_red_pccaplus import find_Q_red
from orthogon import orthogon
#%%

def randompropagator(n):
    T = np.random.rand(n,n)
    for i in range(10):
        T[i,:] = T[:,i] / np.sum(T[:,i])
    return T

T = randompropagator(10)
v, X = linalg.schur(T)
Xo = orthogon(X, np.ones(10), np.diag(v), 3, 1.)
chi, A = optimizeMetastab(Xo, 3)
