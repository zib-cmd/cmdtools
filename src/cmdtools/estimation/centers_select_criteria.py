
# coding: utf-8

# In[ ]:


import numpy as np


def strip_bad(counts_tensor, param=0.2):
    """Find the lines in which the selfoverlap of the basis functions is not 
    good, strip those basis points from the computed transition matrix. 
    Input:
    counts_tensor: 2d array with the stochastic Koopman matrices
    param: float, from zero to one, minimal value that the diagonal entry 
           should have
    Output:
    temp: 2d array, with the rows and columns for with diagonal entry > param;
    not row-normed
    to_keep: list of kept centers"""
    S = counts_tensor[0, :, :]
    to_keep = []
    for i in range(np.shape(S)[0]):
        if S[i, i] > 0.2:
            to_keep.append(i)

    temp = counts_tensor[:, to_keep, :]
    return temp[:, :, to_keep], to_keep


def centers_selection(counts_tensor, param=0.2):
    """Find the lines in which the selfoverlap of the basis functions is not 
    good, strip those basis points from the computed transition matrix. 
    Input:
    counts_tensor: 2d array with the stochastic Koopman matrices
    param: float, from zero to one, minimal value that the diagonal entry 
           should have
    Output:
    koopman_selected: 2d array with the rows and columns for with diagonal entry > param;
    each martrix is row-normed
    centers_kept:list of kept centers"""
    koopman_selected, centers_kept = strip_bad(counts_tensor, param)
    for k in range(np.shape(koopman_selected)[0]):
        koopman_selected[k, :, :] = koopman_selected[k, :, :] / \
            np.sum(koopman_selected[k, :, :], axis=1)
    return(koopman_selected, centers_kept)
