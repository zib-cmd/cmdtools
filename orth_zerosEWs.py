#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:03:18 2019

@author: bzfsechi
"""

import numpy as np
import copy 

def orth_zeros(EW1):
    EW1=copy.deepcopy(np.diag(EW1)) #1-dim array containing the eigenvalues
    list_zeros=[np.arg(y) if np.round(y,6)==0. for y in EW1]
    return(list_zeros)
        
    