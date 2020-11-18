#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:19:47 2020

@author: gamorosino
"""

from nilab.load_trk import load_streamlines
import numpy as np
from dipy.tracking.streamline import set_number_of_points
from nilab.nearest_neighbors import streamlines_neighbors
from scipy import sparse
from time import time 
from matching import min_weight_full_bipartite_matching
from scipy.optimize import linear_sum_assignment

np.random.seed(0)
N_streamlines1=5000
N_streamlines2=5000
N_points=32
k=1000

# %% Load Trk
print("laod stramlines...")

filename1="data1/1M_len20-250mm_coff0001_step1_seedimage_30deg_SD_STREAM.trk"
filename2="data2/1M_len20-250mm_coff0001_step1_seedimage_30deg_SD_STREAM.trk"


track1, header1, lengths1, indices1=load_streamlines(filename1,container="array",verbose=True,idxs=N_streamlines1,apply_affine=True)
track2, header2, lengths2, indices2=load_streamlines(filename2,container="array",verbose=True,idxs=N_streamlines2,apply_affine=True)

# %% Resampling
print("setting the same number of points for both the tracts...")

track1=np.array(set_number_of_points( track1 , N_points ))
track2=np.array(set_number_of_points( track2 ,  N_points ))
# %% Nearest Neightbours
print("Nearest Neightbours")

distances,neighbours=streamlines_neighbors(track1,track2,k=k)

# %% Sparse Cost Matrix

print("Creation of the Sparse Cost Matrix")

cost = sparse.csr_matrix((N_streamlines1,N_streamlines2),dtype=np.float)

for i in range(N_streamlines1):
    print(i)
    cost[i, neighbours[i]] = distances[i]

#cost=cost.tocsr()

cost=cost.todense() 
cost[cost==0]=np.inf
print('Linear Sum Assigment')
t0 = time()
row_ind, col_ind = linear_sum_assignment(cost)
print('%s sec' % (time() - t0))

# %% MWFBM  

print('MWFBM')
t0 = time()
row_ind, col_ind = min_weight_full_bipartite_matching(cost)
print('%s sec' % (time() - t0))
total_cost = cost[row_ind, col_ind].sum()
print('mwfb cost=%s' % total_cost)
