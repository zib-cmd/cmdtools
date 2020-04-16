
import numpy as np
import scipy.spatial.distance as dist
def picking_list(d, n):
    """
    Picking algorithm (Durmaz, 2016)
    Pick out n points such that the respective minimal distance
    to the other picked points is maximized.    Parameters
    ----------
    d : array
        Matrix containing the pairwise distances of the points.
    n : int
        Number of points to pick.    Returns
    -------
    inds : list[int]
        List of indices of the chosen points.    """ 
    
    d = d.copy()
    q1 = 0
    q2 = np.argmax(d[q1, :])
    qs = [q1, q2]
    for k in range(1, n):
        ss = np.argmin(d[qs, :], axis=0)
        mins = list(d[qs[j], s] for s, j in enumerate(ss))
        q = np.argmax(mins)
        qs.append(q)
    return qs


def picking_algorithm(traj, n):
    """
    Return coordinates of the points picked from the 
    picking_list algorithm, using  Euclidean distances. 
    Parameters
    ------------
    traj = 2d array 
         Trajectory coordinates  
    n = int
    Number of points to pick
    
    """
    distances = dist.cdist(traj, traj, dist.sqeuclidean)
    
    return(traj, traj[picking_list(distances, n),:])
 


