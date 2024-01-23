#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 17:41:59 2023

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
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from nilab.load_trk import load_streamlines
from fury import actor, window
from time import sleep
from scipy import sparse
from time import time 
#from matching import min_weight_full_bipartite_matching
from dipy.tracking.metrics import length as streamline_length
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import  save_tractogram
import sys

N_points=32
n_neighbors=250 #30 #Performs KNeighborsRegressor
n_neighbors_apply=24
radius=20

N_streamlines1=1000#00
N_streamlines2=1000#00

streamline_step=1.0

saveWarp_flag=True
apply_warp_flag=True
mask_warp_flag=False

k_sparse=int(N_streamlines1/10)

suffix='_np'+str(N_points)

def corr2disp(col_ind,track_moving,track_fixed, N_points=32, streamline_step=1.0):
        print("setting the same number of points for both the tracts...")
        track_moving=set_number_of_points( track_moving , N_points )
        
        track_fixed=set_number_of_points( track_fixed ,  N_points )
    
        dist=np.linalg.norm((track_moving - track_fixed[col_ind]).reshape(-1, N_points*3), axis=1)
        dist_flip=np.linalg.norm((track_moving - track_fixed[col_ind][:,::-1,:]).reshape(-1, N_points*3), axis=1)
        idx_flip=dist_flip<dist
        X_train = [] # np.zeros((len(track_moving) * N_points, 3))
        Y_train = [] # np.zeros_like(X_train)
        for i, streamline_moving in enumerate(track_moving):
            streamline_fixed = track_fixed[col_ind[i]]
            if idx_flip[i]:
                streamline_fixed = streamline_fixed[::-1,:]
            # resample according to the length of streamline_moving:
            N_s_points = int( streamline_length(streamline_moving) / streamline_step  )
            streamline_moving = set_number_of_points( [streamline_moving] ,  N_s_points )[0]
            streamline_fixed = set_number_of_points( [streamline_fixed] ,  N_s_points )[0]        
            
            X_train.append(streamline_moving)
            Y_train.append(streamline_fixed - streamline_moving)
        return Y_train

def disp2warp(disp,track_moving,dims,n_neighbors,warpfiled_filename=None, reference=None):
            
        #%% Resample Streamlines
        
        X_grid_size=dims[0]
        Y_grid_size=dims[1]
        Z_grid_size=dims[2]        
        

            # X_train[i*N_points:(i+1)*N_points, :] = streamline_moving
            # Y_train[i*N_points:(i+1)*N_points, :] = streamline_fixed - streamline_moving
        X_train = track_moving
        Y_train = disp
        X_train = np.vstack(X_train)
        Y_train = np.vstack(Y_train)
        
        #%% KNeighborsRegressor
        print("Performs KNeighborsRegressor...")
        neigh = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1, weights='distance')
        #neigh = RadiusNeighborsRegressor(n_jobs=1, radius=radius, weights='uniform')
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
    
        #TODO: apply_affine dipy 
        X_test = ( np.dot(aff_moving[:-1,:-1],X_test.T) + np.expand_dims(aff_moving[:-1,-1], axis=1 )).T
        
        #%% Predict 
        print("Predict Warp Vectors...")
        Y_pred=neigh.predict(X_test)
        Y_pred=np.nan_to_num(Y_pred)
        
        #%% Warp Field 
        if saveWarp_flag:
            WF_nib=nib.load(reference)
            aff_moving=WF_nib.affine
            head_moving=WF_nib.header
            Warp = np.zeros((len(Y_range),len(X_range),len(Z_range),3))
            Warp_ch1 = Y_pred[:,1].reshape(Warp.shape[0:3])
            Warp_ch2 = Y_pred[:,0].reshape(Warp.shape[0:3])
            Warp_ch3 = Y_pred[:,2].reshape(Warp.shape[0:3])
            if mask_warp_flag:
                brain_mask=1*(ref_data>0)
                Warp[:,:,:,0] = Warp_ch1 * brain_mask
                Warp[:,:,:,1] = Warp_ch2 * brain_mask
                Warp[:,:,:,2] = Warp_ch3 * brain_mask
            else:
                Warp[:,:,:,0] = Warp_ch1  
                Warp[:,:,:,1] = Warp_ch2  
                Warp[:,:,:,2] = Warp_ch3  
                
            Warp=np.expand_dims(Warp,axis=3)
            WarpNII = nib.Nifti1Image(Warp, affine=aff_moving,header=head_moving) #,header=head_moving) 
            print('save '+warpfiled_filename)
            WarpNII.to_filename(warpfiled_filename)
        



if __name__ == '__main__':

        filename1=sys.args[1]
        filename2=sys.args[2]
        col_ind = sys.args[3]
        ref_filename1= sys.args[4]
        outputdir=sys.args[5]

        
        track_moving, header1, lengths1, indices1=load_streamlines(filename1,container="array",verbose=True,idxs=N_streamlines1,apply_affine=True)
        track_fixed, header2, lengths2, indices2=load_streamlines(filename2,container="array",verbose=True,idxs=N_streamlines2,apply_affine=True)

        try:
            filename1_toapply=sys.args[5]
            track_moving_toapply, header1_toapply, lengths1_toapply, indices1_toapply=load_streamlines(filename1_toapply,container="array",verbose=True,idxs=N_streamlines1,apply_affine=True)
        except:
            pass

        
        
        ref_nib=nib.load(ref_filename1)
        ref_data=ref_nib.get_data()
        aff_moving=ref_nib.affine
        head_moving=ref_nib.header
        
        X_grid_size=head_moving.get_data_shape()[1]
        Y_grid_size=head_moving.get_data_shape()[0]
        Z_grid_size=head_moving.get_data_shape()[2]        
        
        
        save_dir=outputdir+'/saved/'
        WarpField_dir=save_dir+'/WarpFields/'
        ScreenShot_dir=save_dir+'/ScreenShot/'
        WarpedShot_dir=save_dir+'/WarpedTracts/'
        home_dir=os.environ['HOME']+'/'
        

        
        dims=[X_grid_size,Y_grid_size,Z_grid_size]
        
        #X_grid_size=header1['dimensions'][0]
        #Y_grid_size=header1['dimensions'][1]
        #Z_grid_size=header1['dimensions'][2]
        
        warpfiled_filename=WarpField_dir+'/WarpField_N'+suffix+'.nii.gz'
        reference_warp=save_dir+'/sub-1178_Reg2sub-1160_MNI/sub-1178__T1w_rigid_warped_SyN1Warp.nii.gz'
        disp = corr2disp(col_ind,track_moving,track_fixed,N_points=N_points)
        disp2warp(track_moving,disp,dims,n_neighbors,warpfiled_filename=warpfiled_filename,reference=reference_warp)
    
    #%% Apply Warp
        if apply_warp_flag:
            print("Apply Warp...")   
            t0 = time()
            track_moving_toapply=set_number_of_points( track_moving_toapply , N_points )
            #filename1=filename1_toapply
            #filename2=filename2_toapply
    
            #track_moving, aff_moving=loadTrk(filename1)
            #track_fixed, aff_fixed=loadTrk(filename2)
            
            warpNeigh = KNeighborsRegressor(n_neighbors=n_neighbors_apply, n_jobs=10 ) #, weights='distance')
            warpNeigh.fit(X_test, Y_pred)
            if False:
                moving_disp=warpNeigh.predict(X_train)
                moving_warped = X_train + moving_disp
                n_=int(len(moving_warped)/N_points)
                track_moving_warped = np.zeros([n_,N_points,3])
                for idx in range(n_):
                    track_moving_warped[idx] =  moving_warped[idx*N_points:N_points*(idx+1)]    
            else:
                track_moving_warped = track_moving_toapply.copy()
                for i, streamline in enumerate(track_moving_toapply):
                    streamline_warp = warpNeigh.predict(streamline)
                    track_moving_warped[i] += streamline_warp
                    
            warped_filename=WarpedShot_dir+'/'+os.path.basename(filename1_toapply)+suffix+'.trk'            
            track_moving_warped_sft=StatefulTractogram(track_moving_warped, ref_nib, Space.RASMM)
            idx_toremove,idx_tokeep = track_moving_warped_sft.remove_invalid_streamlines()
            save_tractogram(track_moving_warped_sft,warped_filename)  
            print("save warped tracts as: "+warped_filename )
            print('%s sec' % (time() - t0))