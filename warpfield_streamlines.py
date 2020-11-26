#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 17:41:59 2020

@author: gamorosino
"""
import os
import nibabel as nib
import numpy as np
from functools import partial
from dipy.align.streamlinear import StreamlineLinearRegistration
from nilab.distances import  parallel_distance_computation
from dipy.tracking.distances import bundles_distances_mam, bundles_distances_mdf
from dipy.tracking.streamline import set_number_of_points
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KNeighborsRegressor
from nilab.load_trk import load_streamlines
from fury import actor, window
from time import sleep

N_points=32
n_neighbors=10


N_streamlines1=40000#00
N_streamlines2=40000#00

show=False
SLR_flag=False

this_dir=os.path.dirname(os.path.realpath(__file__))
save_dir=this_dir+'/saved/'
try:
    os.makedirs(save_dir)
except:
        pass

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

#%% LOAD TRACKS

track_moving_fname="/home/gamorosino/data/APSS_Neglect/tracts/sub-01_MaRo/epo-01/trk_gabriele/slf_I_right.trk"
track_fixed_fname="/home/gamorosino/data/APSS_Neglect/tracts/sub-01_MaRo/epo-00/trk/SLF_I_right.trk"

filename1="data1/1M_len20-250mm_coff0001_step1_seedimage_30deg_SD_STREAM.trk"
filename2="data2/1M_len20-250mm_coff0001_step1_seedimage_30deg_SD_STREAM.trk"

track_moving, header1, lengths1, indices1=load_streamlines(filename1,container="array",verbose=True,idxs=N_streamlines1,apply_affine=True)
track_fixed, header2, lengths2, indices2=load_streamlines(filename2,container="array",verbose=True,idxs=N_streamlines2,apply_affine=True)

#_, aff_moving=loadTrk(filename1)

FA_filename1="data1/fa.nii.gz"
FA_nib=nib.load(FA_filename1)
aff_moving=FA_nib.affine
head_moving=FA_nib.header

X_grid_size=header1['dimensions'][0]
Y_grid_size=header1['dimensions'][1]
Z_grid_size=header1['dimensions'][2]

#track_moving, aff_moving=loadTrk(track_moving_fname)
#print('Loading %s ...' % track_fixed_fname)
#track_fixed, aff_fixed=loadTrk(track_fixed_fname)
#%%bundles before SLRL


if show is True:
    show_both_bundles([track_moving, track_fixed],
                   colors=[window.colors.orange, window.colors.red],
                   show=True,
                   fname=save_dir+'before_registration.png')
#%% Resample
print("setting the same number of points for both the tracts...")
track_moving=set_number_of_points( track_moving , N_points )
track_fixed=set_number_of_points( track_fixed ,  N_points )
##%% SLR

if SLR_flag:
    print("Linear Registration of the tractcs using SLR...")
    bounds=[(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (0, 0), (0, 0), (0, 0)]
    srr = StreamlineLinearRegistration(bounds=bounds)
    srm = srr.optimize(static=track_fixed, moving=track_moving)
    track_moving = srm.transform(track_moving)

track_moving = np.array(track_moving)
track_fixed = np.array(track_fixed)

#%%bundles after SLR aligmnet
if show is True:
    show_both_bundles([track_moving, track_fixed],
                   colors=[window.colors.orange, window.colors.red],
                   show=True,
                   fname=save_dir+'after_registration.png')
#%% Distance Matrix
print("Calculation of the distance matrix...")
distance = partial(parallel_distance_computation, distance=bundles_distances_mam)
DMAT = distance(track_moving,track_fixed)
print("Performs Linear Assigment Problem...")
row_ind, col_ind=linear_sum_assignment(DMAT)
## TODO: aggiustare
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

X_train = np.vstack(track_moving)
Y_train = np.vstack(track_fixed[col_ind] - track_moving)

print("Performs KNeighborsRegressor...")
neigh = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
neigh.fit(X_train, Y_train)

#%% Meshgrid


# X_range=np.arange(X_grid_size)
# Y_range=np.arange(Y_grid_size)
# Z_range=np.arange(Z_grid_size)

X_offset=aff_moving[:,-1][0]
Y_offset=aff_moving[:,-1][1]
Z_offset=aff_moving[:,-1][2]

step=1
X_range=np.arange(X_offset,X_grid_size+X_offset,step)
Y_range=np.arange(Y_offset,Y_grid_size+Y_offset,step)
Z_range=np.arange(Z_offset,Z_grid_size+Z_offset,step)


X_range=np.arange(-90,90,step)
Y_range=np.arange(-90,90,step)
Z_range=np.arange(-90,90,step)


XX,YY,ZZ = np.meshgrid(X_range,Y_range,Z_range)

X_test=(np.vstack([YY.flatten(),XX.flatten(),ZZ.flatten()]).T)

#%% Predict 
print("Predict Warp Vectors...")
Y_pred=neigh.predict(X_test)

#%% Warp Field 

Warp = np.zeros((len(X_range),len(Y_range),len(Z_range),3))
Warp[:,:,:,0] = Y_pred[:,0].reshape(Warp.shape[0:3])
Warp[:,:,:,1] = Y_pred[:,1].reshape(Warp.shape[0:3])
Warp[:,:,:,2] = Y_pred[:,2].reshape(Warp.shape[0:3])

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

if N_streamlines2_str == N_streamlines1_str:
    warpfiled_filename=save_dir+'/WarpField_'+N_streamlines2_str+'.nii.gz'
else:
     warpfiled_filename=save_dir+'/WarpField_'+N_streamlines1_str+'vs'+N_streamlines2_str+'nii.gz'       

WarpNII = nib.Nifti1Image(Warp, affine=aff_moving,header=head_moving) #,header=head_moving) 
WarpNII.to_filename(warpfiled_filename)


import matplotlib.pyplot as plt



#%%


#from mpl_toolkits.mplot3d import Axes3D 
 
fig = plt.figure()
#ax = fig.gca(projection='3d')


it_arr=range(100000)

#ax = fig.gca() 

#q = ax.quiver(X_test[it_arr,0], X_test[it_arr,1], X_test[it_arr,2], Y_pred[it_arr,0]+X_test[it_arr,0],  Y_pred[it_arr,1]+X_test[it_arr,1], Y_pred[it_arr,2]+X_test[it_arr,2],arrow_length_ratio=0.1,length=0.05)

fig, ax = plt.subplots()
q = ax.quiver(X_test[:,0], X_test[:,1], Y_pred[:,0],  Y_pred[:,1]) #,arrow_length_ratio=0.1,length=0.05)

plt.show()
