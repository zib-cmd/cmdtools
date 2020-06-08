[![Documentation Status](https://readthedocs.org/projects/cmdtools/badge/?version=latest)](https://cmdtools.readthedocs.io/en/latest/?badge=latest)

# cmdtools

This Python library implements a suite of tools used and/or developed in the [Computational Molecular Design](https://www.zib.de/numeric/cmd) group of the Zuse Institute Berlin.


## Installation

1.  Clone the repository with `git clone https://git.zib.de/cmd/cmdtools.git`
2.  Install into your python library with `pip install cmdtools`
    (If you want to use the SLEPc library for optimized Schur decompositions install cmdtools with the extra slepc, i.e. `pip install "cmdtools[slepc]"`)

## Contents
*  `cmdtools.analysis.pcca`: An implementation of (generalized) PCCA⁺ using the Schur decomposition
*  `cmdtools.estimation.galerkin`: Estimation of the transfer operator using a Galerkin projection onto Gaussian RBFs
*  `cmdtools.systems`: A collection of dynamical systems
