#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:45:27 2019

@author: bzfsechi
"""

import numpy as np
import copy 
##%%
#function index=indexsearch(evs,N,k)
#% Zur Erzeugung einer Startloesung wird in den Daten eine
#% moegliche Simplexstruktur gefunden.
#
#maxdist=0.0;
#
#rthoSys=zeros(N,N);
#temp=zeros(1,N);
#
#OrthoSys=evs;
#
#% erste Ecke des Simplex: Normgroesster Eintrag */
#for l=1:N
#    dist = norm(OrthoSys(l,:));
#    if (dist > maxdist)
#        maxdist=dist;
#	    index(1)=l;
#    end
#end
#
#% for l=1:N
#%     OrthoSys(l,:)=OrthoSys(l,:)-evs(index(1),:);
#% end
#OrthoSys=OrthoSys-ones(N,1)*evs(index(1),:);
#
#% Alle weiteren Ecken jeweils mit maximalen Abstand zum
#% bereits gewaehlten Unterraum */
#for j=2:k
#    maxdist=0.0;
#    temp=OrthoSys(index(j-1),:);
#    for l=1:N
#        sclprod=OrthoSys(l,:)*temp';
#        OrthoSys(l,:)=OrthoSys(l,:)-sclprod*temp;
#        distt=norm(OrthoSys(l,:));
#	    if (distt > maxdist ) %&& ~ismember(l,index(1:j-1))
#            maxdist=distt;
#            index(j)=l;
#        end
#    end
#    OrthoSys = OrthoSys/maxdist;
#end

def indexsearch(EVS,N,k):
    """Input: 
        EVS:eigenvectors
        N: no of cells
        k: no of clusters
    Output:
        index: 1D array"""
        
    N=int(N)
    k=int(k)
    maxdist=0.0

    temp=np.zeros((N))
    index=np.zeros(k)
    OrthoSys=copy.deepcopy(EVS)
    
    for l in range(N):
        dist=np.linalg.norm(OrthoSys[l,:])
        if dist>maxdist: 
            maxdist=copy.deepcopy(dist)
            index[0]=copy.deepcopy(l)
    OrthoSys=copy.deepcopy(OrthoSys-np.ones((N,1))*EVS[int(index[0]),:])
    for j in range(1,k):
        maxdist=0.0
        temp=copy.deepcopy(OrthoSys[int(index[j-1]),:])
        
        for l in range(N):
            sclprod=copy.deepcopy(np.dot(OrthoSys[l,:],temp))
            OrthoSys[l,:]=copy.deepcopy(OrthoSys[l,:]-sclprod*temp)
            dist2=np.linalg.norm(OrthoSys[l,:])
            if dist2>maxdist:
                maxdist=dist2
                index[j]=l
        OrthoSys=OrthoSys/maxdist
 #   plt.imshow(OrthoSys)
    return(index)
