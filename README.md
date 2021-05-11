[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zib-cmd/cmdtools/HEAD?filepath=examples)
[![Documentation Status](https://readthedocs.org/projects/cmdtools/badge/?version=latest)](https://cmdtools.readthedocs.io/en/latest/?badge=latest)

# cmdtools

This Python library implements a suite of tools used and/or developed in the [Computational Molecular Design](https://www.zib.de/numeric/cmd) group of the Zuse Institute Berlin.


## Installation

Install with `pip install cmdtools`
    (If you want to use the SLEPc library for sparse Schur decompositions install cmdtools with the extra slepc, i.e. `pip install "cmdtools[slepc]"`)

## Contents
*  `pcca`: An implementation of (generalized) PCCA⁺ using the Schur decomposition
*  `ajc`: A sparse implementation of the augmented jump chain
*  `diffusionmaps`: Diffusionmaps with sparse support and out of sample extensions
*  `galerkin`: Trajectory based estimation of the transfer operator using a Galerkin projection onto Gaussian RBFs
*  `gillespie`: Trajectory simulation from a generator
*  `newton_generator`: Multi-step estimation of the generator via the Newton polynomial
*  `picking_algorithm`: Given a set of datapoints, pick n points such that they are distributed as evenly / equidistant as possible
*  `sqra`: The Square Root approximation, estimating the generator for the diffusion in a given potential
*  `voronoi`: Voronoi clustering of trajectories and estimation of the transfer operator with different metrics and center strategies.
*  `diffusion`: A collection of dynamical systems (So far the double- and triple-well)
