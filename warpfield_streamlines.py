#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 17:41:59 2020

@author: gamorosino
"""

#%% Load Modules

import os
import nibabel as nib
import numpy as np
from functools import partial
from dipy.align.streamlinear import StreamlineLinearRegistration
from nilab.distances import  parallel_distance_computation
from nilab.nearest_neighbors import streamlines_neighbors
from dipy.tracking.distances import bundles_distances_mam, bundles_distances_mdf
from dipy.tracking.streamline import set_number_of_points
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KNeighborsRegressor
from nilab.load_trk import load_streamlines
from fury import actor, window
from time import sleep
from scipy import sparse
from time import time 
from matching import min_weight_full_bipartite_matching

#%% Define Varaibles

N_points=32
n_neighbors=10 #Performs KNeighborsRegressor


N_streamlines1=100000#00
N_streamlines2=100000#00

show=False
plot_flag=False
SLR_flag=False
sparse_flag=True
MATLAB_flag=False
MWFBM_flag=True
dense_flag=False

k_sparse=5000

suffix='_np'+str(N_points)

this_dir=os.path.dirname(os.path.realpath(__file__))
save_dir=this_dir+'/saved/'
try:
    os.makedirs(save_dir)
except:
        pass

#%% Define Functions


def loadTrk(filename):
    data = nib.streamlines.load(filename)
    s = data.streamlines
    aff = data.affine
    return s,aff

def show_both_bundles(bundles, colors=None, show=True, fname=None):

     scene = window.Scene()
     scene.SetBackground(1., 1, 1)
     for (i, bundle) in enumerate(bundles):
         color = colors[i]
         lines_actor = actor.streamtube(bundle, color, linewidth=0.3)
         lines_actor.RotateX(-90)
         lines_actor.RotateZ(90)
         scene.add(lines_actor)
     if show:
         window.show(scene)
     if fname is not None:
         sleep(1)
         window.record(scene, n_frames=1, out_path=fname, size=(900, 900))

#%% Load tracts

filename1="/home/gamorosino/data/APSS_Neglect/tracts/sub-01_MaRo/epo-01/trk_gabriele/slf_I_right.trk"
filename2="/home/gamorosino/data/APSS_Neglect/tracts/sub-01_MaRo/epo-00/trk/SLF_I_right.trk"

#filename1="data1/1M_len20-250mm_coff0001_step1_seedimage_30deg_SD_STREAM.trk"
#filename2="data2/1M_len20-250mm_coff0001_step1_seedimage_30deg_SD_STREAM.trk"

#prefix=

track_moving, header1, lengths1, indices1=load_streamlines(filename1,container="array",verbose=True,idxs=N_streamlines1,apply_affine=True)
track_fixed, header2, lengths2, indices2=load_streamlines(filename2,container="array",verbose=True,idxs=N_streamlines2,apply_affine=True)

#track_moving, aff_moving=loadTrk(filename1)
#track_fixed, aff_fixed=loadTrk(filename2)

#_, aff_moving=loadTrk(filename1)

if show is True:
    show_both_bundles([track_moving, track_fixed],
                   colors=[window.colors.orange, window.colors.red],
                   show=True,
                   fname=save_dir+'before_registration.png')


dcms1=len(str(N_streamlines1))
if dcms1 > 3 and dcms1 < 7:
        N_streamlines1_str=str(int(N_streamlines1*1e-3))+'K'
elif dcms1 > 6 and dcms1 < 10:
        N_streamlines1_str=str(int(N_streamlines1*1e-6))+'G'   
    
dcms2=len(str(N_streamlines2))
if dcms2 > 3 and dcms2 < 7:
        N_streamlines2_str=str(int(N_streamlines2*1e-3))+'K'
elif dcms2 > 6 and dcms2 < 10:
        N_streamlines2_str=str(int(N_streamlines2*1e-6))+'G' 

dcms3=len(str(k_sparse))
if dcms3 > 3 and dcms3 < 7:
        Ksparse_str=str((k_sparse*1e-3))+'K'

#%% Load FA

FA_filename1="data1/fa.nii.gz"

#FA_filename1="/home/gamorosino/data/APSS_Neglect/sub-01_MaRo_NifTI/epo-00/DTI/DTI_bUP/DTI_bUP_data_processed1mm/ConnectGen_files/fa.nii.gz"

FA_nib=nib.load(FA_filename1)
aff_moving=FA_nib.affine
head_moving=FA_nib.header

X_grid_size=head_moving.get_data_shape()[1]
Y_grid_size=head_moving.get_data_shape()[0]
Z_grid_size=head_moving.get_data_shape()[2]

#X_grid_size=header1['dimensions'][0]
#Y_grid_size=header1['dimensions'][1]
#Z_grid_size=header1['dimensions'][2]

#%% Resample Streamlines
print("setting the same number of points for both the tracts...")
track_moving=set_number_of_points( track_moving , N_points )
track_fixed=set_number_of_points( track_fixed ,  N_points )
#%% SLR

if SLR_flag:
    print("Linear Registration of the tractcs using SLR...")
    bounds=[(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0, 0), (0, 0), (0, 0)]
    srr = StreamlineLinearRegistration(bounds=bounds)
    srm = srr.optimize(static=track_fixed, moving=track_moving)
    track_moving = srm.transform(track_moving)
    if show is True:
        show_both_bundles([track_moving, track_fixed],
                       colors=[window.colors.orange, window.colors.red],
                       show=True,
                       fname=save_dir+'after_registration.png')
        
track_moving = np.array(track_moving)
track_fixed = np.array(track_fixed)

    #%% Sparse
suffix_list=[]
col_ind_list=[]
if sparse_flag:
    suffix=suffix+'_Ksparse'+Ksparse_str
    # %% Streamlines Nearest Neightbours
    print("Streamlines Nearest Neightbours")
    
    distances,neighbours=streamlines_neighbors(track_moving,track_fixed,k=k_sparse)
    
    # %% Sparse Cost Matrix
    
    print("Creation of the Sparse Cost Matrix")
    
    cost = sparse.csr_matrix((N_streamlines1,N_streamlines2),dtype=np.float)
    
    tmp=np.repeat(np.arange(N_streamlines1)[:, None], k_sparse, axis=1 )
    
    cost[tmp, neighbours] = distances

    if MATLAB_flag:
        # %% Save cost matrix for matlab
        
        import scipy.io as sio
        
        # get the data, shape and indices
        (sparse_m,sparse_n) = cost.shape
        sparse_s = cost.data
        sparse_i = cost.tocoo().row
        sparse_j = cost.indices
        
        #data={ 'cost': cost }
        data={ 'cost': cost ,'m': sparse_m, 'n': sparse_n , 's' : sparse_s,
             'i': sparse_i, 'j' :  sparse_j}
        
        matfile="/home/gamorosino/local/sparse_large_lap/cost_"+suffix+".mat"
        sio.savemat(matfile,data)
        
        # %% MATLAB: Fast Linear Assignment Problem using Auction Algorithm
        print('Matlab: Fast Linear Assignment Problem using Auction Algorithm')
        import subprocess
        
        fastAuction="/home/gamorosino/local/fastAuction_v2.6"
        matvar1='assignments'
        matvar2='P'
        matfile_out="/home/gamorosino/local/sparse_large_lap/fastLAP_outputs"+suffix+".mat"
                     
        cmd = """matlab -nodesktop -nosplash -r "addpath('"""+fastAuction+"""'); mat=load('"""+matfile+"""'); [assignments,P]=applyFastAuction(mat.cost) ;save('"""+matfile_out+"""','"""+matvar1+"""','"""+matvar2+"""'); quit;"   """
        
        #cmd = """matlab -nodesktop -nosplash -r "addpath('"""+fastAuction+"""'); mat=load('"""+matfile+"""');m = mat.m; n = mat.n; s = mat.s; i=mat.i + 1; j = mat.j + 1; cost = sparse(i, j, s, m, n, m*n); [assignments,P]=applyFastAuction(cost) ;  save('"""+matfile_out+"""','"""+matvar1+"""','"""+matvar2+"""'); quit;"   """
        os.system(cmd + ' > /tmp/output_matlab.txt')
        cmd="cat /tmp/output_matlab.txt"
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print('output:', output.decode("utf-8"))
        mat = sio.loadmat(matfile_out)
        col_ind=mat[matvar1]
        col_ind=col_ind[:,0] - 1
        suffix_Matlab=suffix+"_Matlab"
        suffix_list.append(suffix_Matlab)
        col_ind_list.append(col_ind)
    # %% MWFBM  
    if MWFBM_flag:
        print('MWFBM')
        t0 = time()
        row_ind, col_ind = min_weight_full_bipartite_matching(cost)
        print('%s sec' % (time() - t0))
        total_cost = cost[row_ind, col_ind].sum()
        print('mwfb cost=%s' % total_cost)
        col_ind_MWFBM = col_ind
        suffix_MWFBM = suffix+"_MWFBM"
        suffix_list.append(suffix_MWFBM)
        col_ind_list.append(col_ind)
#%%
if dense_flag:
    #%% Distance Matrix
    print("Calculation of the distance matrix...")
    distance = partial(parallel_distance_computation, distance=bundles_distances_mam)
    DMAT = distance(track_moving,track_fixed)
    print("Performs Linear Assigment Problem...")
    t0 = time()
    row_ind, col_ind=linear_sum_assignment(DMAT)
    print('%s sec' % (time() - t0))
    suffix_list.append(suffix)
    col_ind_list.append(col_ind)
    # Y_train = np.zeros([1,3])
    # X_train = np.zeros([1,3])
    # track_fixed_ord=track_fixed.copy()
    # for i,idx in enumerate(col_ind):
    #    Y_train=np.concatenate([Y_train,(track_fixed[idx] - track_moving[i])],axis=0)
    #    X_train=np.concatenate([X_train,track_moving[i]],axis=0)
    #    track_fixed_ord=track_fixed[idx]
    # #%%  KNeighborsRegressor 
    # Y_train=Y_train[1:,:]
    # X_train=X_train[1:,:]

for indx,suffix in enumerate(suffix_list):
    #%% Stack Stramlines
    col_ind = col_ind_list[indx]
    
    X_train = np.vstack(track_moving)
    Y_train = np.vstack(track_fixed[col_ind] - track_moving)
    
    #%% KNeighborsRegressor
    print("Performs KNeighborsRegressor...")
    neigh = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
    neigh.fit(X_train, Y_train)
    #%% Meshgrid
    suffix=suffix+"_nn"+str(n_neighbors)
    
    #X_offset=aff_moving[:,-1][0]*aff_moving[0,0]
    #Y_offset=aff_moving[:,-1][1]*aff_moving[1,1]
    #Z_offset=aff_moving[:,-1][2]*aff_moving[2,2]
    
    step=1
    X_range=np.arange(0,X_grid_size,step).astype(int)#+X_offset
    Y_range=np.arange(0,Y_grid_size,step).astype(int)#+Y_offset
    Z_range=np.arange(0,Z_grid_size,step).astype(int)#+Z_offset

    #Y_range=np.arange(0,X_grid_size,step)#+X_offset
    
    XX,YY,ZZ = np.meshgrid(X_range,Y_range,Z_range)
   
    X_test=(np.vstack([YY.flatten(),XX.flatten(),ZZ.flatten()]).T)

    X_test = ( np.dot(aff_moving[:-1,:-1],X_test.T) + np.expand_dims(aff_moving[:-1,-1], axis=1 )).T
    
    #%% Predict 
    print("Predict Warp Vectors...")
    Y_pred=neigh.predict(X_test)
    
    #%% Warp Field 
    
    Warp = np.zeros((len(Y_range),len(X_range),len(Z_range),3))
    Warp[:,:,:,0] = Y_pred[:,0].reshape(Warp.shape[0:3])
    Warp[:,:,:,1] = Y_pred[:,1].reshape(Warp.shape[0:3])
    Warp[:,:,:,2] = Y_pred[:,2].reshape(Warp.shape[0:3])
    
  
    
    if N_streamlines2_str == N_streamlines1_str:
        warpfiled_filename=save_dir+'/WarpField_N'+N_streamlines2_str+suffix+'.nii.gz'
    else:
         warpfiled_filename=save_dir+'/WarpField_N'+N_streamlines1_str+'vs'+N_streamlines2_str+suffix+'.nii.gz'       

    WarpNII = nib.Nifti1Image(Warp, affine=aff_moving,header=head_moving) #,header=head_moving) 
    print('save '+warpfiled_filename)
    WarpNII.to_filename(warpfiled_filename)

#%% PLOT
if plot_flag:
    import matplotlib.pyplot as plt
    #from mpl_toolkits.mplot3d import Axes3D 
     
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    
    
    it_arr=range(100000)
    
    #ax = fig.gca() 
    
    #q = ax.quiver(X_test[it_arr,0], X_test[it_arr,1], X_test[it_arr,2], Y_pred[it_arr,0]+X_test[it_arr,0],  Y_pred[it_arr,1]+X_test[it_arr,1], Y_pred[it_arr,2]+X_test[it_arr,2],arrow_length_ratio=0.1,length=0.05)
    
    fig, ax = plt.subplots()
    q = ax.quiver(X_test[:,0], X_test[:,1], Y_pred[:,0],  Y_pred[:,1]) #,arrow_length_ratio=0.1,length=0.05)
    
    plt.show()
