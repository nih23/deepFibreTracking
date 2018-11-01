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

def main():
    
    sh_order = 4
    b_value = 1000
    stepWidth = 0.6
    
    coordinateScaling = 1
    useDTIPeakDirection = True
    useSphericalHarmonics = True
    
    pCaseID = '100307'
    pModel = 'results/train_OLDsh4_step0.6_wholeBrain_b1k_csd_1x1x1/models/doubleIn_noInputRepetition_mlp_doubleIn_single_sqCos2_wb_dx_1_dy_1_dz_1_dd_15_ReLU_feat_512_depth_3_output_3_lr_0.0001_dropout_1_bn_1_pt_0_467--0.977036.h5'
    pResult = pModel.replace('.h5','').replace('/models/','_') + '-prediction'
    
    tracker = load_model(pModel , custom_objects={'tf':tf, 'swish':Activation(swish), 'squared_cosine_proximity': squared_cosine_proximity_2, 'squared_cosine_proximity_2': squared_cosine_proximity_2})
    noSamples, noX, noY, noZ, noC = tracker.get_input_shape_at(0)[0]
    #tracker.summary()
    useBitracker = True

    
    print('Loaded model with  (dx %d, dy %d, dz %d) and %d channels' % (noX, noY, noZ, noC))

    # load DWI data
    print('Importing HCP data')
    bvals,bvecs,gtab,dwi,aff,t1,binarymask = dwi_tools.loadHCPData('data/%s' % (pCaseID))
    dwi_subset, gtab_subset, bvals_subset, bvecs_subset = dwi_tools.cropDatsetToBValue(b_value, bvals, bvecs, dwi)
    b0_idx = bvals < 10
    b0 = dwi[..., b0_idx].mean(axis=3)

    dwi_singleShell = np.concatenate((dwi_subset, dwi[..., b0_idx]), axis=3)
    dwi_singleShell_norm = dwi_tools.normalize_dwi(dwi_singleShell, b0)
    bvals_singleShell = np.concatenate((bvals_subset, bvals[..., b0_idx]), axis=0)
    bvecs_singleShell = np.concatenate((bvecs_subset, bvecs[b0_idx,]), axis=0)
    gtab_singleShell = gradient_table(bvals=bvals_singleShell, bvecs=bvecs_singleShell, b0_threshold = 10)

    tracking_data = dwi_singleShell_norm

    if(useSphericalHarmonics):
        print('Spherical Harmonics (ours)')
        start_time = time.time()
        tracking_data = dwi_tools.get_spherical_harmonics_coefficients(bvals=bvals_singleShell,bvecs=bvecs_singleShell,sh_order=sh_order, dwi=dwi_singleShell_norm, b0 = 0)
        runtime = time.time() - start_time
        print('Runtime ' + str(runtime) + 's')
        #CSD/DIPY APPROACH
        #print('Spherical Harmonics (dipy)')
        #csd_model = ConstrainedSphericalDeconvModel(gtab_singleShell, None, sh_order=sh_order)
        #csd_fit = csd_model.fit(dwi_singleShell_norm, mask=binarymask)
        #tracking_data = csd_fit.shm_coeff


    # DTI PEAK DIRECTION INPUT
    if(useDTIPeakDirection):
        print('DTI Peak Direction/odf estimation')
        start_time = time.time()
        dti_model = dti.TensorModel(gtab_singleShell, fit_method='LS')
        dti_fit = dti_model.fit(dwi_singleShell_norm)
        runtime = time.time() - start_time
        print('Runtime ' + str(runtime) + 's')

    # corpus callosum seed mask
    ccmask, options = nrrd.read('data/100307/100307-ccSegmentation.nrrd')
    ccseeds = seeds_from_mask(ccmask, affine=aff)
    # whole brain seeds
    wholebrainseeds = seeds_from_mask(binarymask, affine=aff)
    
    
    start_time = time.time()
    streamlines_mlp_simple_sc,vNorms = tracking.startWithStopping(mask=binarymask,fa=dti_fit.fa, bitracker=useBitracker, inverseDirection=False, seeds=ccseeds, data=tracking_data, affine=aff, model=tracker, noX=noX, noY=noY, noZ=noZ, dw = noC, stepWidth = 0.6, coordinateScaling = coordinateScaling)
    runtime = time.time() - start_time
    print('Runtime ' + str(runtime) + ' s ')

    start_time = time.time()
    streamlines_mlp_simple_sc_2,vNorms2 = tracking.startWithStopping(mask=binarymask,fa=dti_fit.fa, bitracker=useBitracker, inverseDirection=True, seeds=ccseeds, data=tracking_data, affine=aff, model=tracker, noX=noX, noY=noY, noZ=noZ, dw = noC, stepWidth = 0.6, coordinateScaling = coordinateScaling)
    runtime = time.time() - start_time
    print('Runtime ' + str(runtime) + ' s ')

    # join left and right streamlines
    streamlines_joined_sc = tracking.joinTwoAlignedStreamlineLists(streamlines_mlp_simple_sc,streamlines_mlp_simple_sc_2)

    streamlines_joined_sc_imageCS = transform_streamlines(streamlines_joined_sc, np.linalg.inv(aff))

    dwi_tools.saveVTKstreamlines(streamlines_joined_sc_imageCS,pResult + '.vtk')
    
    dwi_tools.visStreamlines(streamlines_joined_sc_imageCS,t1, vol_slice_idx = 75)

if __name__ == "__main__":
    main()
