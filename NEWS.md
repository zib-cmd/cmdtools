0.2.1
=====

New Features
------------
* voronoi.py
    * Precluster a trajectory into voronoi cells and compute the transfer matrix.
      Uses either predefined centers or computes them via k-means or picking_algorithm
    * Alternatively use a uniform grid and sparsify by omitting unused cells
* diffusionmaps.py
    Compute the diffusion maps of a given set of points
    Features bandwidth estimation and custom metrics as well as using only the n-nearest-neighbours.

Improvements / Bug fixes
------------------------
* galerkin.find_bandwith: Fix calculation
* picking_algorithm: Enable use of custom metrics.