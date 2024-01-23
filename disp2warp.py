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
n_neighbors=1000 #30 #Performs KNeighborsRegressor
n_neighbors_apply=24
radius_to_apply=30
radius=30

N_streamlines1=1000#00
N_streamlines2=1000#00

streamline_step=1.0

saveWarp_flag=True
apply_warp_flag=True


k_sparse=int(N_streamlines1/10)

suffix='_np'+str(N_points)



def disp2warp(disp,track_moving,dims,n_neighbors,aff_moving,warpfiled_filename=None, reference=None, inv=False,mask_warp_flag=False):
            
        #%% Resample Streamlines
        
        X_grid_size=dims[0]
        Y_grid_size=dims[1]
        Z_grid_size=dims[2]        
        

            # X_train[i*N_points:(i+1)*N_points, :] = streamline_moving
            # Y_train[i*N_points:(i+1)*N_points, :] = streamline_fixed - streamline_moving
        if inv:
            disp=-disp
        X_train = track_moving
        Y_train = disp
        X_train = np.vstack(X_train)
        Y_train = np.vstack(Y_train)
        
        #%% KNeighborsRegressor
        print("Performs NeighborsRegressor...")
        #neigh = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1, weights='distance')
        neigh = RadiusNeighborsRegressor(n_jobs=-1, radius=radius, weights='distance')
        neigh.fit(X_train, Y_train)
        #%% Meshgrid
        #suffix=suffix+"_nn"+str(n_neighbors)
        
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
        #X_test=(np.vstack([XX.flatten(),YY.flatten(),ZZ.flatten()]).T)
    
        #TODO: apply_affine dipy 
        X_test = ( np.dot(aff_moving[:-1,:-1],X_test.T) + np.expand_dims(aff_moving[:-1,-1], axis=1 )).T
        
        #%% Predict 
        print("Predict Warp Vectors...")
        Y_pred=neigh.predict(X_test)
        Y_pred=np.nan_to_num(Y_pred)
       
        #%% Warp Field 
        if warpfiled_filename is not None:
            WF_nib=nib.load(reference)
            aff_moving=WF_nib.affine
            head_moving=WF_nib.header
            Warp = np.zeros((len(Y_range),len(X_range),len(Z_range),3))
            #Warp = np.zeros((len(X_range),len(Y_range),len(Z_range),3))
            Warp_ch1 = Y_pred[:,1].reshape(Warp.shape[0:3])
            #Warp_ch1 = Y_pred[:,0].reshape(Warp.shape[0:3])
            Warp_ch2 = Y_pred[:,0].reshape(Warp.shape[0:3])
            #Warp_ch2 = Y_pred[:,1].reshape(Warp.shape[0:3])
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
        return Y_pred, X_test
    
    
def applyWarp(track_moving_toapply, Y_pred,X_test, warped_filename=None):        
             print("Apply Warp...")   
             t0 = time()
             track_moving_toapply=set_number_of_points( track_moving_toapply , N_points )
            
             #warpNeigh = KNeighborsRegressor(n_neighbors=n_neighbors_apply, n_jobs=-1 ) #, weights='distance')
             warpNeigh = RadiusNeighborsRegressor(n_jobs=-1, radius=radius_to_apply, weights='distance')
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
             if warped_filename is not None:       
                track_moving_warped_sft=StatefulTractogram(track_moving_warped, ref_nib, Space.RASMM)
                idx_toremove,idx_tokeep = track_moving_warped_sft.remove_invalid_streamlines()
                save_tractogram(track_moving_warped_sft,warped_filename)  
                print("save warped tracts as: "+warped_filename )
                print('%s sec' % (time() - t0))


if __name__ == '__main__':

        filename1=sys.argv[1]
        disp_filename = sys.argv[2]
        ref_filename1= sys.argv[3]
        output=sys.argv[4]
        reference_warp=sys.argv[5]
        try:
            inverse=bool(int(sys.argv[6]))
        except:
            inverse=False
        
        if inverse:
            print('inverse Warp')
        
        #idxs=N_streamlines1,
        track_moving, header1, lengths1, indices1=load_streamlines(filename1,container="array",verbose=True,apply_affine=True)
        #track_fixed, header2, lengths2, indices2=load_streamlines(filename2,container="array",verbose=True,idxs=N_streamlines2,apply_affine=True)

        try:
            filename1_toapply=sys.argv[7]
            #track_moving_toapply, header1_toapply, lengths1_toapply, indices1_toapply=load_streamlines(filename1_toapply,container="array",verbose=True,idxs=N_streamlines1,apply_affine=True)
            track_moving_toapply, header1_toapply, lengths1_toapply, indices1_toapply=load_streamlines(filename1_toapply,container="array",verbose=True,apply_affine=True)
        except:
            pass

        try:
            output_streamlines=sys.argv[8]
        except:
            pass
        
        print(filename1_toapply)
        
        ref_nib=nib.load(ref_filename1)
        ref_data=ref_nib.get_data()
        aff_moving=ref_nib.affine
        head_moving=ref_nib.header
        
        X_grid_size=head_moving.get_data_shape()[1]
        Y_grid_size=head_moving.get_data_shape()[0]
        Z_grid_size=head_moving.get_data_shape()[2]        
        
        
        #save_dir=outputdir+'/saved/'
        #WarpField_dir=save_dir+'/WarpFields/'
        #ScreenShot_dir=save_dir+'/ScreenShot/'
        #WarpedShot_dir=save_dir+'/WarpedTracts/'
        #home_dir=os.environ['HOME']+'/'
        

        
        dims=[X_grid_size,Y_grid_size,Z_grid_size]
        
        #X_grid_size=header1['dimensions'][0]
        #Y_grid_size=header1['dimensions'][1]
        #Z_grid_size=header1['dimensions'][2]
        
        #warpfiled_filename=WarpField_dir+'/WarpField_N'+suffix+'.nii.gz'
        #reference_warp=save_dir+'/sub-1178_Reg2sub-1160_MNI/sub-1178__T1w_rigid_warped_SyN1Warp.nii.gz'
        disp=np.load(disp_filename)
        Y_pred, X_test = disp2warp(disp,track_moving,dims,n_neighbors,aff_moving,warpfiled_filename=output,reference=reference_warp,inv=inverse,mask_warp_flag=False)
        applyWarp(track_moving_toapply, Y_pred, X_test,output_streamlines)
    


    #%% Apply Warp
        # if apply_warp_flag:
        #     print("Apply Warp...")   
        #     t0 = time()
        #     track_moving_toapply=set_number_of_points( track_moving_toapply , N_points )
        #     #filename1=filename1_toapply
        #     #filename2=filename2_toapply
    
        #     #track_moving, aff_moving=loadTrk(filename1)
        #     #track_fixed, aff_fixed=loadTrk(filename2)
            
        #     warpNeigh = KNeighborsRegressor(n_neighbors=n_neighbors_apply, n_jobs=-1 ) #, weights='distance')
        #     warpNeigh.fit(X_test, Y_pred)
        #     if False:
        #         moving_disp=warpNeigh.predict(X_train)
        #         moving_warped = X_train + moving_disp
        #         n_=int(len(moving_warped)/N_points)
        #         track_moving_warped = np.zeros([n_,N_points,3])
        #         for idx in range(n_):
        #             track_moving_warped[idx] =  moving_warped[idx*N_points:N_points*(idx+1)]    
        #     else:
        #         track_moving_warped = track_moving_toapply.copy()
        #         for i, streamline in enumerate(track_moving_toapply):
        #             streamline_warp = warpNeigh.predict(streamline)
        #             track_moving_warped[i] += streamline_warp
                    
        #     warped_filename=WarpedShot_dir+'/'+os.path.basename(filename1_toapply)+suffix+'.trk'            
        #     track_moving_warped_sft=StatefulTractogram(track_moving_warped, ref_nib, Space.RASMM)
        #     idx_toremove,idx_tokeep = track_moving_warped_sft.remove_invalid_streamlines()
        #     save_tractogram(track_moving_warped_sft,warped_filename)  
        #     print("save warped tracts as: "+warped_filename )
        #     print('%s sec' % (time() - t0))