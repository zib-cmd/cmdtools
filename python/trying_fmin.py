#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:01:58 2019

@author: bzfsechi
"""
import numpy as np
from scipy import optimize
#%%

def func(x, a,b):
    return((x)**a-x**b)
    
#%%
c=2
d=5
#%%
xopt, fopt,iters,funcalls, warnflag,=optimize.fmin(func, x0=2, args=(d,c), maxiter=500,full_output=1)
#%%
dicts={"nel mezzo del cammin di nostra vita": 3.5, "mi ritrovai per una selva oscura":4}