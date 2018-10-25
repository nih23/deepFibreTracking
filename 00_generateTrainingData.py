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

noCrossings = 3


def main():
    '''
    generate training data
    '''
    
    anatomicalArea = 'wholeBrain' # or 'CorpusCallosum'
    tensorModel = 'dti'
    
    # spatial extent of training data
    noX = 1
    noY = 1
    noZ = 1
    coordinateScaling = 1
    stepWidth = 0.1
    bval = 1000
    
    pTrainData = "train_%s_%s_b%d_sw%.1f_dx%d_dy%d_dz%d_cs%.1f.h5" % (anatomicalArea,tensorModel,bval,stepWidth,noX,noY,noZ,coordinateScaling)
    pStreamlinesRaw = 'streamlines_%s_%s_b%d_sw%.1f.npy' % (anatomicalArea, tensorModel, bval, stepWidth)
    
    # load HCP data
    print('load HCP data (b=%d)' % (bval))
    bvals,bvecs,gtab,dwi,aff,t1,binarymask = dwi_tools.loadHCPData('100307')
    dwi_subset, gtab_subset, bvals_subset, bvecs_subset = dwi_tools.cropDatsetToBValue(bval, bvals, bvecs, dwi)

    ### exract the averaged b0.
    b0_idx = bvals < 10
    b0 = dwi[..., b0_idx].mean(axis=3)
        
    # normalize DWI by b0
    print('\n normalizing dwi dataset')
    dwi_subset_norm = dwi_tools.normalize_dwi(dwi_subset, b0)
    
    # compute spherical harmonics
    print('compute spherical harmonics')
    data_sh, weights, b0 = dwi_tools.get_spherical_harmonics_coefficients(dwi_subset, b0=b0, bvals=bvals_subset, bvecs=bvecs_subset, sh_order = 4)
    
    
    # load mask of reference streamlines    
    ccmask, options = nrrd.read('100307/100307-ccSegmentation.nrrd')
    ccseeds = seeds_from_mask(ccmask, affine=aff)
    validationSeeds = ccseeds[45:48] # three reference streamlines
    #rndseeds = random_seeds_from_mask(binarymask, seeds_count=4000, seed_count_per_voxel=False, affine=aff)
    
    # single tensor model
    print('fitting single tensor model (WLS)')
    import dipy.reconst.dti as dti
    start_time = time.time()
    dti_wls = dti.TensorModel(gtab_subset)
    fit_wls = dti_wls.fit(dwi_subset)
    runtime = time.time() - start_time
    print('-> runtime ' + str(runtime) + 's\n')
    
    
    # create training data consisting of all three eigenvectors and their eigenvalues of that model
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
                                nbr_processes=24
                               )
    runtime = time.time() - start_time
    print('-> runtime ' + str(runtime) + 's\n')
    
    # reconstruct streamlines
    print('tracking')
    from dipy.tracking.local import BinaryTissueClassifier
    #classifier = ThresholdTissueClassifier(dtipeaks.gfa, .01)
    start_time = time.time()
    binary_classifier = BinaryTissueClassifier(binarymask == 1) # streamlines need to touch gray matter
    ##streamlines_generator = LocalTracking(dtipeaks, binary_classifier, ccseeds, aff, step_size=.1)
    streamlines_generator = LocalTracking(dtipeaks, binary_classifier, rndseeds, aff, step_size=.1)
    streamlines = Streamlines(streamlines_generator)
    streamlines_filtered = dwi_tools.filterStreamlinesByLength(streamlines, 40)
    runtime = time.time() - start_time
    print(str(len(streamlines_filtered)) + ' streamlines')
    print('-> runtime ' + str(runtime) + 's\n')
    
    print('Saving raw streamlines to ' + pStreamlinesRaw)
    start_time = time.time()
    np.save(pStreamlinesRaw,streamlines_filtered)
    runtime = time.time() - start_time
    print('-> runtime ' + str(runtime) + 's\n')
      
    print('\n generating training data')
    rawData = data_sh

    start_time = time.time()
    train_DWI,train_prevDirection, train_LikelyFibreDirections, train_nextDirection = dwi_tools.generateTrainingData(streamlines_filtered, rawData, noX=noX,noY=noY,noZ=noZ,coordinateScaling=coordinateScaling, distToNeighbours=1, noCrossings = noCrossings, affine=aff)
    runtime = time.time() - start_time
    print('-> runtime ' + str(runtime) + ' s\n')
    
    print('Saving data to ' + pTrainData)
    with h5py.File(pTrainData,"w") as f:
        f.create_dataset('dwi_raw',data=dwi_subset)
        f.create_dataset('dwi_sh',data=data_sh)
        f.create_dataset('affine',data=aff
                        )
        f.create_dataset('coordinateScaling',data=coordinateScaling)
        f.create_dataset('noCrossings',data=noCrossings)
        f.create_dataset('noX',data=noX)
        f.create_dataset('noY',data=noY)
        f.create_dataset('noZ',data=noZ)
        f.create_dataset('dwi',data=train_DWI)
        
        f.create_dataset('prevDirection',data=train_prevDirection)   
        f.create_dataset('nextDirection',data=train_nextDirection)   
        f.create_dataset('seeds',data=rndseeds)
    
    
if __name__ == "__main__":
    main()
