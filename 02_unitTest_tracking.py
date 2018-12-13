import time
import nrrd
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import h5py

from dipy.tracking.local import LocalTracking, ThresholdTissueClassifier
from dipy.tracking.utils import random_seeds_from_mask
from dipy.reconst.dti import TensorModel
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.segment.mask import median_otsu
from dipy.viz import actor, window
from dipy.io.image import save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core import gradients
from dipy.tracking.streamline import Streamlines
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table, gradient_table_from_bvals_bvecs
from dipy.reconst.dti import fractional_anisotropy
import dipy.reconst.dti as dti
from dipy.tracking import utils

import src.dwi_tools as dwi_tools
import src.nn_helper as nn_helper
import src.tracking as tracking

from dipy.tracking.streamline import values_from_volume
import dipy.align.vector_fields as vfu
from dipy.core.sphere import Sphere
from dipy.core import subdivide_octahedron 
from scipy.spatial import KDTree
from dipy.core import subdivide_octahedron 

from dipy.tracking.local import LocalTracking
from dipy.viz import window, actor
from dipy.viz.colormap import line_colors
from dipy.tracking.streamline import Streamlines, transform_streamlines
from dipy.tracking import metrics
from dipy.tracking.utils import random_seeds_from_mask, seeds_from_mask

import numpy as np
import warnings

import tensorflow as tf
from joblib import Parallel, delayed
import multiprocessing
from keras.models import load_model
from keras.layers import Activation

import importlib
importlib.reload(tracking)
importlib.reload(nn_helper)
import src.tracking as tracking
import tensorflow as tf
from src.nn_helper import swish, squared_cosine_proximity_2
from src.SelectiveDropout import SelectiveDropout

import os

def main():
    fa_threshold = 0.15
#    fa_threshold = 0.2
    sh_order = 8
#    b_value = 3000
    stepWidth = 0.6
    minimumStreamlineLength = 50 # mm
    
    coordinateScaling = 1
    useDTIPeakDirection = True
    useSphericalHarmonics = True
    
    pCaseID = 'HCP/100408'
    pCaseID = 'HCP/100307'
    
    pModel = 'results/train_oldSH8_step0.6_wholeBrain_b3000_2_csd_1x1x1/models/mlp_doubleIn_single_sqCos2_wb_dx_1_dy_1_dz_1_dd_45_ReLU_feat_2048_depth_3_output_3_lr_0.0001_dropout_1_bn_0_pt_0_233--0.988976.h5'
    

    
    pResult = pModel.replace('.h5','').replace('/models/','/tracking/') + '.vtk'
    

    os.makedirs(pModel.replace('.h5','').replace('/models/','/tracking/'), exist_ok=True)
    
    tracker = load_model(pModel , custom_objects={'tf':tf, 'swish':Activation(swish), 'SelectiveDropout': SelectiveDropout, 'squared_cosine_proximity_2': squared_cosine_proximity_2})
    
    tracker.summary()
    
    useBitracker = True
    useBayesianTracker = False
    use2DProjection = False
    
    if(pModel.find("cnn_special")>=0):
        use2DProjection = True

    
    if(pModel.find("b1000")>=0):
        b_value = 1000
    
    if(pModel.find("b3000")>=0):
        b_value = 3000
#    if(pModel.find("single_bayesian")>=0):
#        useBayesianTracker = True
    
    if(pModel.find("mlp_single")>=0):
        useBitracker = False

    if(useBitracker and not use2DProjection): 
        noSamples, noX, noY, noZ, noC = tracker.get_input_shape_at(0)[0]
    elif (not useBitracker and not use2DProjection):
        noSamples, noX, noY, noZ, noC = tracker.get_input_shape_at(0)
    else:
        noSamples, noX, noY, noZ, noC = tracker.get_input_shape_at(0)[0]
        noZ = 1
        useSphericalHarmonics = False

    # load DWI data
    print('Loading dataset %s at b=%d' % (pCaseID, b_value))
    bvals,bvecs,gtab,dwi,aff,t1,binarymask = dwi_tools.loadHCPData('data/%s' % (pCaseID))
    dwi_subset, gtab_subset, bvals_subset, bvecs_subset = dwi_tools.cropDatsetToBValue(b_value, bvals, bvecs, dwi)
    b0_idx = bvals < 10
    b0 = dwi[..., b0_idx].mean(axis=3)

    dwi_singleShell = np.concatenate((dwi_subset, dwi[..., b0_idx]), axis=3)
    dwi_singleShell_norm = dwi_tools.normalize_dwi(dwi_singleShell, b0)
    bvals_singleShell = np.concatenate((bvals_subset, bvals[..., b0_idx]), axis=0)
    bvecs_singleShell = np.concatenate((bvecs_subset, bvecs[b0_idx,]), axis=0)
    gtab_singleShell = gradient_table(bvals=bvals_singleShell, bvecs=bvecs_singleShell, b0_threshold = 10)

    # resampling dataset
    print('Resampling to 100 directions on repulsion100')
    dwi_singleShell_ressampled, resamplingSphere = dwi_tools.resample_dwi(dwi_subset, b0, bvals_subset, bvecs_subset, sh_order=8, smooth=0, mean_centering=False)
    
    # set data to use in tracking
    tracking_data = dwi_singleShell_ressampled

    if(useSphericalHarmonics):
        print('Spherical Harmonics (ours)')
        start_time = time.time()
        #tracking_data = dwi_tools.get_spherical_harmonics_coefficients(bvals=bvals_singleShell,bvecs=bvecs_singleShell,sh_order=sh_order, dwi=dwi_singleShell_norm, b0 = 0)
        tracking_data = dwi_tools.get_spherical_harmonics_coefficients(bvals=bvals_subset,bvecs=bvecs_subset,sh_order=sh_order, dwi=dwi_subset, b0 = b0)
        runtime = time.time() - start_time
        print('Runtime ' + str(runtime) + 's')
        #CSD/DIPY APPROACH
        #print('Spherical Harmonics (dipy)')
        #csd_model = ConstrainedSphericalDeconvModel(gtab_singleShell, None, sh_order=sh_order)
        #csd_fit = csd_model.fit(dwi_singleShell_norm, mask=binarymask)
        #tracking_data = csd_fit.shm_coeff


    # DTI 
    print('DTI fa estimation')
    start_time = time.time()
    dti_model = dti.TensorModel(gtab_singleShell, fit_method='LS')
    dti_fit = dti_model.fit(dwi_singleShell_norm)
    runtime = time.time() - start_time
    print('Runtime ' + str(runtime) + 's')

    print("Tractography")
    wholebrainseeds = seeds_from_mask(binarymask, affine=aff)
    rndseeds = random_seeds_from_mask(binarymask, seeds_count=300, seed_count_per_voxel=False, affine=aff)
    #seedsToUse = wholebrainseeds 
    seedsToUse = rndseeds 
    
    nn_helper.setAllDropoutLayers(tracker, useBayesianTracker)
    nn_helper.printDropoutLayersState(tracker)
    
   
    start_time = time.time()
    if(use2DProjection):
        streamlines_mlp_simple_sc,vNorms = tracking.startWithStoppingAnd2DProjection(printProgress = True, resamplingSphere = resamplingSphere, bayesianModel = useBayesianTracker, fa_threshold = fa_threshold, mask=binarymask,fa=dti_fit.fa, bitracker=useBitracker, inverseDirection=False, seeds=seedsToUse, data=tracking_data, affine=aff, model=tracker, stepWidth = stepWidth)
    else:
        streamlines_mlp_simple_sc,vNorms = tracking.startWithStopping(printProgress = True, bayesianModel = useBayesianTracker, fa_threshold = fa_threshold, mask=binarymask,fa=dti_fit.fa, bitracker=useBitracker, inverseDirection=False, seeds=seedsToUse, data=tracking_data, affine=aff, model=tracker, noX=noX, noY=noY, noZ=noZ, dw = noC, stepWidth = stepWidth, coordinateScaling = coordinateScaling)
    runtime = time.time() - start_time
    print('Runtime ' + str(runtime) + ' s ')

    #start_time = time.time()
    if(use2DProjection):
        streamlines_mlp_simple_sc_2,vNorms_2 = tracking.startWithStoppingAnd2DProjection(printProgress = True, resamplingSphere = resamplingSphere, bayesianModel = useBayesianTracker, fa_threshold = fa_threshold, mask=binarymask,fa=dti_fit.fa, bitracker=useBitracker, inverseDirection=True, seeds=seedsToUse, data=tracking_data, affine=aff, model=tracker, stepWidth = stepWidth)
    else:
        streamlines_mlp_simple_sc_2,vNorms_2 = tracking.startWithStopping(printProgress = True, bayesianModel = useBayesianTracker, fa_threshold = fa_threshold, mask=binarymask,fa=dti_fit.fa, bitracker=useBitracker, inverseDirection=True, seeds=seedsToUse, data=tracking_data, affine=aff, model=tracker, noX=noX, noY=noY, noZ=noZ, dw = noC, stepWidth = stepWidth, coordinateScaling = coordinateScaling)
    runtime = time.time() - start_time
    print('Runtime ' + str(runtime) + ' s ')

    # join left and right streamlines
    print("Postprocessing streamlines and removing streamlines shorter than " + str(minimumStreamlineLength) + " mm")
    streamlines_joined_sc = tracking.joinTwoAlignedStreamlineLists(streamlines_mlp_simple_sc,streamlines_mlp_simple_sc_2)
       
    streamlines_joined_sc = dwi_tools.filterStreamlinesByLength(streamlines_joined_sc, minimumStreamlineLength)

    print("Writing the data to disk")
    dwi_tools.saveVTKstreamlines(streamlines_joined_sc,pResult)
    
    streamlines_joined_sc_imageCS = transform_streamlines(streamlines_joined_sc, np.linalg.inv(aff))
    dwi_tools.visStreamlines(streamlines_joined_sc_imageCS, t1, vol_slice_idx = 75)
    
    

if __name__ == "__main__":
    main()
