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
from src.tied_layers1d import Convolution2D_tied

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
import warnings

def main():
    fa_threshold = 0.15
    #fa_threshold = 0.2
    sh_order = 8
    b_value = 1000
    minimumStreamlineLength = 40 # mm
    ### maximum lengths?
    stepWidth = 1.0 # mm
    stepWidth = 0.6 # mm
    noTrackingSteps = 200
    
    coordinateScaling = 1
    useDTIPeakDirection = True
    useSphericalHarmonics = True
    usePreviousDirection = True
    useBayesianTracker = False
    use2DProjection = False
    useDenoising = False
    resample100Directions = False
    resliceDataToHCPDimension = True
    
    pCaseID = 'ISMRM_2015_Tracto_challenge_data'
    #pCaseID = 'ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2'
       
    pModel = 'results/train_sh8_noB0InSH_step0.6_wholeBrain_b1000_2_csd_ismrm_1x1x1_noUnitTension/models/V3_mlp_single_2_sqCos2_dx_1_dy_1_dz_1_dd_45_ReLU_feat_512_depth_3_output_3_lr_0.0001_dropout_1_bn_0_pt_0_unitTension_0-zvFix2_122--0.971016.h5'
    
    pModel = 'results/train_sh8_noB0InSH_step0.6_wholeBrain_b1000_2_csd_ismrm_1x1x1_noUnitTension/models/V3_mlp_doubleIn_single_sqCos2_dx_1_dy_1_dz_1_dd_45_ReLU_feat_512_depth_3_output_3_lr_0.0001_dropout_1_bn_0_pt_0_unitTension_0-zvFix2_117--0.995002.h5'
    
    ### MODELS TO TRY 12/01/18:
    #pModel = 'results/train_sh8_noB0InSH_step0.6_wholeBrain_b1000_2_ukf_curated_1x1x1_noUnitTension/models/V3_mlp_single_2_sqCos2_dx_1_dy_1_dz_1_dd_45_ReLU_feat_512_depth_3_output_3_lr_0.0001_dropout_1_bn_0_pt_0_unitTension_0-zvFix2_973--0.799828.h5'
    
    #pModel = 'results/train_sh8_noB0InSH_step0.6_wholeBrain_b1000_2_ukf_curated_1x1x1_noUnitTension/models/V3_mlp_doubleIn_single_sqCos2_dx_1_dy_1_dz_1_dd_45_ReLU_feat_512_depth_3_output_3_lr_0.0001_dropout_1_bn_0_pt_0_unitTension_0-zvFix2_01--0.998337.h5'
    
    pModel = 'results/train_res100_30k_noB0InSH_step0.6_wholeBrain_b1000_csd_ismrm_1x1x1_noUnitTension/models/V3_mlp_doubleIn_single_sqCos2_dx_1_dy_1_dz_1_dd_100_ReLU_feat_512_depth_3_output_3_lr_0.0001_dropout_1_bn_0_pt_0_unitTension_0-zvFix2_306--0.995033.h5'
    
    #pModel = 'results/train_res100_30k_noB0InSH_step0.6_wholeBrain_b1000_csd_ismrm_1x1x1_noUnitTension/models/V3_mlp_single_sqCos2_dx_1_dy_1_dz_1_dd_100_ReLU_feat_512_depth_3_output_3_lr_0.0001_dropout_1_bn_0_pt_0_unitTension_0-zvFix2_903--0.967215.h5'
    
    pModel = 'results/train_res100_30k_noB0InSH_step0.6_wholeBrain_b1000_csd_ismrm_1x1x1_noUnitTension/models/V3_mlp_single_sqCos2_dx_1_dy_1_dz_1_dd_100_ReLU_feat_512_depth_3_output_3_lr_0.0001_dropout_1_bn_0_pt_0_unitTension_0-zvFix2_104--0.966557.h5'
    
    # try:
    # results/train_sh8_noB0InSH_step0.6_wholeBrain_b1000_2_csd_ismrm_1x1x1_noUnitTension/models/V3_mlp_single_sqCos2_dx_1_dy_1_dz_1_dd_45_ReLU_feat_512_depth_3_output_3_lr_0.0050_dropout_1_bn_0_pt_0_unitTension_0-zvFix2_197--0.968804.h5
    
    ##pModel = 'results/train_res1002D_30k_noB0InSH_step0.6_wholeBrain_b1000_csd_ismrm_1x1x1_noUnitTension/models/V3_mlp_single_sqCos2_dx_1_dy_1_dz_1_dd_100_ReLU_feat_512_depth_3_output_3_lr_0.0001_dropout_1_bn_0_pt_0_unitTension_0-zvFix2_47--0.961644.h5'

    
    #pModel = 'results/train_res1002D_16x16_30k_noB0InSH_step0.6_wholeBrain_b1000_csd_ismrm_1x1x1_noUnitTension/models/V3_cnn_special_sqCos2_dx_1_dy_1_dz_1_dd_256_ReLU_feat_128_depth_2_output_3_lr_0.0001_dropout_1_bn_0_pt_0_unitTension_0-zvFix2_23--0.960626.h5'
    
    #pModel = 'results/train_res1002D_16x16_30k_noB0InSH_step0.6_wholeBrain_b1000_csd_ismrm_1x1x1_noUnitTension/models/V3_mlp_single_sqCos2_dx_1_dy_1_dz_1_dd_256_ReLU_feat_512_depth_3_output_3_lr_0.0001_dropout_1_bn_0_pt_0_unitTension_0-zvFix2_924--0.961970.h5'
    
    pModel = 'results/train_res1002D_16x16_30k_noB0InSH_step0.6_wholeBrain_b1000_csd_ismrm_1x1x1_noUnitTension/models/V3_cnn_special_sqCos2_dx_1_dy_1_dz_1_dd_256_ReLU_feat_128_depth_2_output_3_lr_0.0001_dropout_1_bn_0_pt_0_unitTension_0-zvFix2_113--0.962577.h5'
    
    pModel = 'results/train_res1002D_16x16_30k_noB0InSH_step0.6_wholeBrain_b1000_csd_ismrm_1x1x1_noUnitTension/models/V3_cnn_special_sqCos2_dx_1_dy_1_dz_1_dd_256_ReLU_feat_32_depth_3_output_3_lr_0.0001_dropout_1_bn_0_pt_0_unitTension_0-zvFix2_233--0.963799.h5'
    
    pModel = 'results/train_res1002D_16x16_30k_noB0InSH_step0.6_wholeBrain_b1000_csd_ismrm_1x1x1_noUnitTension/models/V3_cnn_special_sqCos2_dx_1_dy_1_dz_1_dd_256_ReLU_feat_128_depth_2_output_3_lr_0.0001_dropout_1_bn_0_pt_0_unitTension_0-zvFix2_291--0.964105.h5'
    
    #pModel = 'results/train_res1002D_16x16_30k_noB0InSH_step0.6_wholeBrain_b1000_csd_ismrm_1x1x1_noUnitTension/models/V3_cnn_special_sqCos2_dx_1_dy_1_dz_1_dd_256_ReLU_feat_32_depth_3_output_3_lr_0.0001_dropout_1_bn_0_pt_0_unitTension_0-zvFix2_571--0.965189.h5'
    
    pModel = 'results/train_res1002D_16x16_30k_noB0InSH_step0.6_wholeBrain_b1000_csd_ismrm_1x1x1_noUnitTension/models/V3_rcnn_sqCos2_dx_1_dy_1_dz_1_dd_256_ReLU_feat_32_depth_3_output_3_lr_0.0001_dropout_1_bn_0_pt_0_unitTension_0-zvFix2_15--0.962247.h5'
    
    pResult = "results_tracking/" + pCaseID + os.path.sep + pModel.replace('.h5','').replace('/models/','/').replace('results','') + 'reslice-' + str(resliceDataToHCPDimension) + '-denoising-' + str(useDenoising) + '-' + str(noTrackingSteps) + 'st-fa-' + str(fa_threshold)
    
    tracker = load_model(pModel , custom_objects={'tf':tf, 'swish':Activation(swish), 'SelectiveDropout': SelectiveDropout, 'squared_cosine_proximity_2': squared_cosine_proximity_2, 'Convolution2D_tied': Convolution2D_tied})

    os.makedirs(pResult, exist_ok=True)

    tracker.summary()
    
    if(pModel.find("mlp_single")>0):
        usePreviousDirection = False
        
    if((pModel.find("cnn_special")>0) or (pModel.find("rcnn")>0)):
        use2DProjection = True
        usePreviousDirection = False
        useSphericalHarmonics = False
        noC = 1
        noSamples, noX, noY, noZ = tracker.get_input_shape_at(0)
    elif(pModel.find("res1002D")>0):
        use2DProjection = True
        useSphericalHarmonics = False
        resample100Directions = False
        if(usePreviousDirection): 
            noSamples, noX, noY, noZ, noC = tracker.get_input_shape_at(0)[0]
        else: 
            noSamples, noX, noY, noZ, noC = tracker.get_input_shape_at(0)
    elif(pModel.find("res100_")>0):
        use2DProjection = False
        useSphericalHarmonics = False
        resample100Directions = True
        if(usePreviousDirection): 
            noSamples, noX, noY, noZ, noC = tracker.get_input_shape_at(0)[0]
        else: 
            noSamples, noX, noY, noZ, noC = tracker.get_input_shape_at(0)

        
    elif(usePreviousDirection): 
        noSamples, noX, noY, noZ, noC = tracker.get_input_shape_at(0)[0]
    else: 
        noSamples, noX, noY, noZ, noC = tracker.get_input_shape_at(0)

    print('Loaded model with  (dx %d, dy %d, dz %d) and %d channels' % (noX, noY, noZ, noC))

    # load DWI data
    print('Loading dataset %s at b=%d' % (pCaseID, b_value))
    bvals,bvecs,gtab,dwi,aff,t1 = dwi_tools.loadISMRMData('data/%s' % (pCaseID), denoiseData = useDenoising, resliceToHCPDimensions=resliceDataToHCPDimension)
    b0_mask, binarymask = median_otsu(dwi[:,:,:,0], 2, 1)
    
    # crop DWI data
    dwi_subset, gtab_subset, bvals_subset, bvecs_subset = dwi_tools.cropDatsetToBValue(b_value, bvals, bvecs, dwi)
    b0_idx = bvals < 10
    b0 = dwi[..., b0_idx].mean(axis=3)

    if(useSphericalHarmonics):
        print('Spherical Harmonics (ours)')
        start_time = time.time()
        tracking_data = dwi_tools.get_spherical_harmonics_coefficients(bvals=bvals_subset,bvecs=bvecs_subset,sh_order=sh_order, dwi=dwi_subset, b0 = b0)
        runtime = time.time() - start_time
        print('Runtime ' + str(runtime) + 's')

    if(use2DProjection):
        print('2D projection')
        start_time = time.time()
        tracking_data, resamplingSphere = dwi_tools.resample_dwi_forunet(dwi_subset, b0, bvals_subset, bvecs_subset, sh_order=8, smooth=0, mean_centering=False)
        runtime = time.time() - start_time
        print('Runtime ' + str(runtime) + 's')
        #use2DProjection = False

    if(resample100Directions):
        print('Resampling to 100 directions')
        start_time = time.time()
        tracking_data, resamplingSphere = dwi_tools.resample_dwi(dwi_subset, b0, bvals_subset, bvecs_subset, sh_order=8, smooth=0, mean_centering=False)
        runtime = time.time() - start_time
        print('Runtime ' + str(runtime) + 's')
    
    print('DTI Peak Direction/odf estimation')   
    dwi_singleShell = np.concatenate((dwi_subset, dwi[..., b0_idx]), axis=3)
    bvals_singleShell = np.concatenate((bvals_subset, bvals[..., b0_idx]), axis=0)
    bvecs_singleShell = np.concatenate((bvecs_subset, bvecs[b0_idx,]), axis=0)
    gtab_singleShell = gradient_table(bvals=bvals_singleShell, bvecs=bvecs_singleShell, b0_threshold = 10)
    
    start_time = time.time()
    dti_model = dti.TensorModel(gtab_singleShell, fit_method='LS')
    dti_fit = dti_model.fit(dwi_singleShell)
    runtime = time.time() - start_time
    print('Runtime ' + str(runtime) + 's')


    # whole brain seeds
    wholebrainseeds = seeds_from_mask(binarymask, affine=aff)

    print("Tractography")
    seedsToUse = wholebrainseeds 
    
    warnings.filterwarnings("ignore")
    
    importlib.reload(tracking)
    warnings.filterwarnings("ignore")
    start_time = time.time()
    streamlines_joined_sc,vNorms,slf,slb = tracking.start(printfProfiling = False, printProgress = True, fa_threshold = fa_threshold, mask=binarymask,fa=dti_fit.fa, inverseDirection=False, seeds=seedsToUse, data=tracking_data, affine=aff, model=tracker, noX=noX, noY=noY, noZ=noZ, dw = noC, stepWidth = stepWidth, coordinateScaling = coordinateScaling, noIterations = noTrackingSteps, usePreviousDirection=usePreviousDirection, reshapeForConvNet = use2DProjection)
    runtime = time.time() - start_time
    print('Runtime ' + str(runtime) + ' s ')

#    else:
#        start_time = time.time()
#        if(use2DProjection):
#            streamlines_mlp_simple_sc,vNorms = tracking.startWithStoppingAnd2DProjection(printProgress = True, resamplingSphere = resamplingSphere, bayesianModel = useBayesianTracker, fa_threshold = fa_threshold, mask=binarymask,fa=dti_fit.fa, bitracker=usePreviousDirection, inverseDirection=False, seeds=seedsToUse, data=tracking_data, affine=aff, model=tracker, stepWidth = stepWidth, noIterations = noTrackingSteps)
#        else:
#            streamlines_mlp_simple_sc,vNorms = tracking.startWithStopping(printProgress = True, bayesianModel = useBayesianTracker, fa_threshold = fa_threshold, mask=binarymask,fa=dti_fit.fa, bitracker=usePreviousDirection, inverseDirection=False, seeds=seedsToUse, data=tracking_data, affine=aff, model=tracker, noX=noX, noY=noY, noZ=noZ, dw = noC, stepWidth = stepWidth, coordinateScaling = coordinateScaling, noIterations = noTrackingSteps)
#        runtime = time.time() - start_time
#        print('Runtime ' + str(runtime) + ' s ')

#        start_time = time.time()
#        if(use2DProjection):
#            streamlines_mlp_simple_sc_2,vNorms_2 = tracking.startWithStoppingAnd2DProjection(printProgress = True, resamplingSphere = resamplingSphere, bayesianModel = useBayesianTracker, fa_threshold = fa_threshold, mask=binarymask,fa=dti_fit.fa, bitracker=usePreviousDirection, inverseDirection=True, seeds=seedsToUse, data=tracking_data, affine=aff, model=tracker, stepWidth = stepWidth, noIterations = noTrackingSteps)
#        else:
#            streamlines_mlp_simple_sc_2,vNorms_2 = tracking.startWithStopping(printProgress = True, bayesianModel = useBayesianTracker, fa_threshold = fa_threshold, mask=binarymask,fa=dti_fit.fa, bitracker=usePreviousDirection, inverseDirection=True, seeds=seedsToUse, data=tracking_data, affine=aff, model=tracker, noX=noX, noY=noY, noZ=noZ, dw = noC, stepWidth = stepWidth, coordinateScaling = coordinateScaling, noIterations = noTrackingSteps)
#        runtime = time.time() - start_time
#        print('Runtime ' + str(runtime) + ' s ')

        ## join left and right streamlines
#        streamlines_joined_sc = tracking.joinTwoAlignedStreamlineLists(streamlines_mlp_simple_sc,streamlines_mlp_simple_sc_2)
    
    print("Postprocessing streamlines and removing streamlines shorter than " + str(minimumStreamlineLength) + " mm")
    streamlines_joined_sc = dwi_tools.filterStreamlinesByLength(streamlines_joined_sc, minimumStreamlineLength)
    streamlines_joined_sc = dwi_tools.filterStreamlinesByMaxLength(streamlines_joined_sc) # 200mm


    print("The data is being written to disk.")
    dwi_tools.saveVTKstreamlines(streamlines_joined_sc,pResult + '.vtk')
    
 
    streamlines_joined_sc_imageCS = transform_streamlines(streamlines_joined_sc, np.linalg.inv(aff))
    
    dwi_tools.visStreamlines(streamlines_joined_sc_imageCS,dwi_singleShell[:,:,:,0], vol_slice_idx = 30)

if __name__ == "__main__":
    main()
