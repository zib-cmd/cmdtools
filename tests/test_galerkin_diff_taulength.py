#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:13:36 2019

@author: bzfsechi
"""

from numpy.random import rand, randint
from cmdtools import utils
from cmdtools.estimation import galerkin_taus_all

def propagator_diff_taus():
    x = rand(100, 3)
    centers = rand(4, 3)
    random_tau = randint(1,5)
    sigma = galerkin_taus_all.find_bandwidth(x, centers)
    P = galerkin_taus_all.propagator_tau(x, centers, sigma, random_tau)
    for i in range(0, random_tau):
        assert utils.is_rowstochastic(P[i])
