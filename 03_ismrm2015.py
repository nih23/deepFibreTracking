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
from src.nn_helper import swish, squared_cosine_proximity_2, weighted_binary_crossentropy, mse_directionInvariant
from src.SelectiveDropout import SelectiveDropout
import os 
import warnings

import argparse

def main():
    
    pModelDefault = 'results/train_res100_all_noB0InSH_step0.6_wholeBrain_b1000_csd_ismrm_cur_1x1x1_noUnitTension/models/V3wZ_mlp_single_sqCos2WEP_dx_1_dy_1_dz_1_dd_100_ReLU_feat_512_depth_3_output_3_lr_0.0001_dropout_1_bn_0_pt_0_unitTension_0-zvFix2_13--0.658784.h5'
    
    parser = argparse.ArgumentParser(description='Deep Learning Tractography -- Tracking')
    parser.add_argument('model', help='path to trained neural network')
    parser.add_argument('-c', '--caseid', default='ISMRM_2015_Tracto_challenge_data', help='name of tracking case')
    parser.add_argument('-n', '--noSteps', default=200, type=int, help='no. tracking steps')
    parser.add_argument('-sw','--stepWidth', default=1.0, type=float, help='no. tracking steps')
    parser.add_argument('-mil','--minLength', type=int, default=20, help='minimal length of a streamline [mm]')
    parser.add_argument('-mal','--maxLength', type=int, default=200, help='maximum length of a streamline [mm]')
    parser.add_argument('-fa','--faThreshold', type=float, default=0, help='fa threshold in case of non-magical models')
    parser.add_argument('-sh', '--shOrder', type=int, default=8, help='order of spherical harmonics (if used)')
    parser.add_argument('-nt', '--noThreads', type=int, default=4, help='number of parallel threads of the data generator. Note: this also increases the memory demand.')
    parser.add_argument('--denoise', help='denoise dataset', dest='denoise' , action='store_true')
    parser.add_argument('--nodenoise', help='dont denoise dataset', dest='denoise' , action='store_false')
    parser.add_argument('--bvalue',type=int, default=1000, help='b-value of our DWI data')
    parser.add_argument('--reslice', help='reslice datase to 1.25mm^3', dest='reslice' , action='store_true')
    parser.add_argument('--rotateData', help='reslice datase to 1.25mm^3', dest='rotateData' , action='store_true')
    parser.set_defaults(denoise=False)   
    parser.set_defaults(reslice=False)   
    parser.set_defaults(rotateData=False)   
    args = parser.parse_args()

    fa_threshold=args.faThreshold
    sh_order=args.shOrder
    minimumStreamlineLength = args.minLength
    stepWidth = args.stepWidth
    noTrackingSteps = args.noSteps
    pCaseID = args.caseid
    maximumStreamlineLength = args.maxLength
    b_value = args.bvalue
    pModel = args.model
    resliceDataToHCPDimension = args.reslice
    useDenoising = args.denoise
    rotateData = args.rotateData
    
    coordinateScaling = 1
    useDTIPeakDirection = True
    useSphericalHarmonics = True
    usePreviousDirection = True
    useBayesianTracker = False
    use2DProjection = False
    #useDenoising = False
    resample100Directions = False
   # resliceDataToHCPDimension = False
    
    #pCaseID = 'ISMRM_2015_Tracto_challenge_data'
    #pCaseID = 'ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2'
       

    #pModel = 'results/train_res100_all_noB0InSH_step0.6_wholeBrain_b1000_csd_ismrm_cur_1x1x1_noUnitTension/models/V3wZ_mlp_single_sqCos2WEP_dx_1_dy_1_dz_1_dd_100_ReLU_feat_512_depth_3_output_3_lr_0.0001_dropout_1_bn_0_pt_0_unitTension_0-zvFix2_13--0.658784.h5' # ziemlich gut
    
    
    pResult = "results_tracking/" + pCaseID + os.path.sep + pModel.replace('.h5','').replace('/models/','/').replace('results','') + 'reslice-' + str(resliceDataToHCPDimension) + '-denoising-' + str(useDenoising) + '-' + str(noTrackingSteps) + 'st-20mm-fa-' + str(fa_threshold)
    
    tracker = load_model(pModel , custom_objects={'tf':tf, 'swish':Activation(swish), 'SelectiveDropout': SelectiveDropout, 'squared_cosine_proximity_2': squared_cosine_proximity_2, 'Convolution2D_tied': Convolution2D_tied, 'weighted_binary_crossentropy': weighted_binary_crossentropy, 'mse_directionInvariant': mse_directionInvariant})

    os.makedirs(pResult, exist_ok=True)

    tracker.summary()
    
    if(pModel.find("mlp_single")>0):
        usePreviousDirection = False
        
    if((pModel.find("cnn_special")>0) or (pModel.find("rcnn")>0)):
        use2DProjection = True
        usePreviousDirection = (pModel.find("cnn_special_pd")>0) or (pModel.find("rcnn_pd")>0)
        useSphericalHarmonics = False
        noSamples, noX, noY, noZ = tracker.get_input_shape_at(0)
        noC = noZ
        noX, noY, noZ = (1,1,1)
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

    magicModel = False
    
    if(pModel.find("sqCos2WEP")>0):
        print("Magic model :)")
        magicModel = True
            
    print('Loaded model %s' % (pModel))

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
    
    if(not magicModel):
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
    
    #warnings.filterwarnings("ignore")
    
    if(magicModel):
        if(pModel.find('2mlp_single')>0):
            # magic 2mlp_single with aggregation
            start_time = time.time()
            streamlines_joined_sc,vNorms,stopProb = tracking.startAggregatedMagicModel(printfProfiling = False, printProgress = True, mask=binarymask, inverseDirection=False, seeds=seedsToUse, data=tracking_data, affine=aff, model=tracker, noX=noX, noY=noY, noZ=noZ, dw = noC, stepWidth = stepWidth, coordinateScaling = coordinateScaling, noIterations = noTrackingSteps, usePreviousDirection=usePreviousDirection, reshapeForConvNet = use2DProjection, rotateData = rotateData)
            runtime = time.time() - start_time
            
        else:
            # plain magic model
            start_time = time.time()
            streamlines_joined_sc,vNorms,stopProb = tracking.startMagicModel(printfProfiling = False, printProgress = True, mask=binarymask, inverseDirection=False, seeds=seedsToUse, data=tracking_data, affine=aff, model=tracker, noX=noX, noY=noY, noZ=noZ, dw = noC, stepWidth = stepWidth, coordinateScaling = coordinateScaling, noIterations = noTrackingSteps, usePreviousDirection=usePreviousDirection, reshapeForConvNet = use2DProjection, rotateData = rotateData)
            runtime = time.time() - start_time

    else:
        # standard model without learnt stopping criteria
        start_time = time.time()
        streamlines_joined_sc,vNorms = tracking.start(printfProfiling = False, printProgress = True, fa_threshold = fa_threshold, mask=binarymask,fa=dti_fit.fa, inverseDirection=False, seeds=seedsToUse, data=tracking_data, affine=aff, model=tracker, noX=noX, noY=noY, noZ=noZ, dw = noC, stepWidth = stepWidth, coordinateScaling = coordinateScaling, noIterations = noTrackingSteps, usePreviousDirection=usePreviousDirection, reshapeForConvNet = use2DProjection, rotateData = rotateData)
        runtime = time.time() - start_time

   
    print("Postprocessing streamlines and removing streamlines shorter than " + str(minimumStreamlineLength) + " mm")
    streamlines_joined_sc = dwi_tools.filterStreamlinesByLength(streamlines_joined_sc, minimumStreamlineLength)
    streamlines_joined_sc = dwi_tools.filterStreamlinesByMaxLength(streamlines_joined_sc, maximumStreamlineLength) # 200mm


    print("The data is being written to disk.")
    dwi_tools.saveVTKstreamlines(streamlines_joined_sc,pResult + '.vtk')
    
 
    streamlines_joined_sc_imageCS = transform_streamlines(streamlines_joined_sc, np.linalg.inv(aff))
    
    #dwi_tools.visStreamlines(streamlines_joined_sc_imageCS,b0, vol_slice_idx = 30)

if __name__ == "__main__":
    main()
