import time
import nrrd
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import h5py

from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from dipy.tracking.local import LocalTracking, ThresholdTissueClassifier
from dipy.tracking.utils import random_seeds_from_mask, seeds_from_mask
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
from dipy.core import gradients
from dipy.tracking.streamline import Streamlines, transform_streamlines
from dipy.reconst.dti import fractional_anisotropy

from dipy.tracking import utils

import src.dwi_tools as dwi_tools
import src.nn_helper as nn_helper

from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation
from keras.callbacks import TensorBoard

pTrainData_fibrePrediction = 'train_prediction_grid_normalized_dti_cs1_wholebrain.h5'
pTrainData_fibreTracking = 'train_tracking_grid_normalized_dti_cs1_wholebrain.h5'
pTrainInput = 'train_input_normalized_dti_cs1_wholebrain_'
noCrossings = 3


def main():
    '''
    generate training data
    '''
    
    # spatial extent of training data
    noX = 1
    noY = 1
    noZ = 1
    coordinateScaling = 1
    
    # load HCP data
    bvals,bvecs,gtab,dwi,aff,t1,binarymask = dwi_tools.loadHCPData('100307')
    dwi_subset, gtab_subset, evals_subset, evecs_subset = dwi_tools.cropDatsetToBValue(1000, bvals, bvecs, dwi)

    # compute spherical harmonics
    
    ### exract the averaged b0.
    b0_idx = bvals < 10
    b0 = dwi[..., b0_idx].mean(axis=3)
    data_sh, weights, b0 = dwi_tools.get_spherical_harmonics_coefficients(dwi_subset, b0=b0, bvals=bvals_subset, bvecs=bvecs_subset, sh_order = 4)
    
    
    # load mask of reference streamlines    
    ccmask, options = nrrd.read('100307/100307-ccSegmentation.nrrd')
    ccseeds = seeds_from_mask(ccmask, affine=aff)
    validationSeeds = ccseeds[45:48] # three reference streamlines
    
    # single tensor model
    print('fitting tensor model')
    import dipy.reconst.dti as dti
    start_time = time.time()
    dti_wls = dti.TensorModel(gtab_subset)
    fit_wls = dti_wls.fit(dwi_subset)
    runtime = time.time() - start_time
    print('-> runtime ' + str(runtime) + 's\n')
    
    
    # create training data consisting of all three tensors and their eigenvalues of that model
    dataSz = np.append(fit_wls.evecs.shape[0:3],9)
    data_evecs = np.concatenate((np.reshape(fit_wls.evecs, dataSz), fit_wls.evals), axis=3)
    
    # extract peaks
    print('extract peaks')
    sphere = get_sphere('symmetric724')
    start_time = time.time()
    dtipeaks = peaks_from_model(model=dti_wls,
                                data=dwi_subset,
                                sphere=sphere,
                                relative_peak_threshold=.5,
                                min_separation_angle=25,
                                mask=binarymask,
                                return_odf=False,
                                parallel=True,
                                normalize_peaks=False,
                                nbr_processes=12
                               )
    runtime = time.time() - start_time
    print('-> runtime ' + str(runtime) + 's\n')
    
    # reconstruct streamlines
    from dipy.tracking.local import BinaryTissueClassifier
    #classifier = ThresholdTissueClassifier(dtipeaks.gfa, .01)
    binary_classifier = BinaryTissueClassifier(binarymask == 1) # streamlines need to touch gray matter
    streamlines_generator = LocalTracking(dtipeaks, binary_classifier, ccseeds, aff, step_size=.1)
    streamlines = Streamlines(streamlines_generator)
    streamlines_filtered = dwi_tools.filterStreamlinesByLength(streamlines, 40)
    
    # project streamlines into image coordinate system
    streamlines_imageCS = transform_streamlines(streamlines_filtered, np.linalg.inv(aff)) # project streamlines from RAS into image (voxel) coordinate system
    
    # normalize DWI by b0
    print('\n normalizing dwi dataset')
    dwi_subset = dwi_tools.normalize_dwi(dwi_subset, b0)
    
    print('\n generating training data')
    rawData = data_evecs
    #rawData = data_sh

    start_time = time.time()
    train_DWI,train_prevDirection, train_LikelyFibreDirections, train_nextDirection = dwi_tools.generateTrainingData(streamlines_imageCS, rawData, noX=noX,noY=noY,noZ=noZ,coordinateScaling=coordinateScaling,distToNeighbours=1, noCrossings = noCrossings)
    runtime = time.time() - start_time
    print('-> runtime ' + str(runtime) + ' s\n')
    
    print('storing data')
    with h5py.File(pTrainData_fibrePrediction,"w") as f:
        f.create_dataset('train_DWI',data=train_DWI)
        f.create_dataset('train_curPosition',data=train_prevDirection)   
        f.create_dataset('train_LikelyFibreDirections',data=train_LikelyFibreDirections)   
        f.create_dataset('train_NextFibreDirection',data=train_nextDirection)   
    
    
if __name__ == "__main__":
    main()
