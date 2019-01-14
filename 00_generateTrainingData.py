import time
import nrrd
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import h5py

from dipy.tracking import eudx
from dipy.tracking.local import LocalTracking, ThresholdTissueClassifier
from dipy.tracking.utils import random_seeds_from_mask
from dipy.reconst.dti import TensorModel, quantize_evecs
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response)
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model, DeterministicMaximumDirectionGetter
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.segment.mask import median_otsu
from dipy.viz import actor, window
from dipy.io.image import save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core import gradients
from dipy.tracking.streamline import Streamlines, transform_streamlines
from dipy.tracking.utils import random_seeds_from_mask, seeds_from_mask
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table, gradient_table_from_bvals_bvecs
from dipy.reconst.dti import fractional_anisotropy
from dipy.tracking import utils
from dipy.segment.mask import median_otsu

import src.dwi_tools as dwi_tools
import src.nn_helper as nn_helper
from dipy.segment.mask import median_otsu

import argparse

def main():
    '''
    generate training data
    '''
    coordinateScaling = 1
    noCrossingFibres = 3
    
    
    parser = argparse.ArgumentParser(description='Deep Learning Tractography -- Data Generation')
    parser.add_argument('tensorModel', help='tensormodel to use to fit the data: dti, csd')
    parser.add_argument('-nx', dest='nx', default=1, type=int, help='no of voxels in X plane for each streamline position')
    parser.add_argument('-ny', dest='ny',default=1, type=int, help='no of voxels in Y plane for each streamline position')
    parser.add_argument('-nz', dest='nz',default=1, type=int, help='no of voxels in Z plane for each streamline position')
    parser.add_argument('-b', dest='b',default=1000, type=int, help='b-value')
    parser.add_argument('-fa', dest='fa',default=0.15, type=float, help='fa threshold for the tracking')
    parser.add_argument('-sw', dest='sw',default=1.0, type=float, help='tracking step width [mm]')
    parser.add_argument('-mil', dest='minLength', type=int, default=40, help='minimal length of a streamline [mm]')
    parser.add_argument('-mal', dest='maxLength', type=int, default=200, help='maximum length of a streamline [mm]')
    parser.add_argument('-repr', dest='repr',default='res100', help='data representation: [raw,sph,res100,2D]: raw, spherical harmonics, resampled to 100 directions, 16x16 2D resampling (256 directions) ')
    parser.add_argument('-sh', dest='sh', type=int, default=8, help='order of spherical harmonics (if used)')
    parser.add_argument('-noStreamlines', dest='noStreamlines', default=0, type=int, help='specify if a random subset of the streamlines should be used to generate the training data')
    parser.add_argument('--rotateTrainingData', help='rotate data wrt. tangent [default]', dest='rotateTrainingData' , action='store_true')
    #parser.add_argument('--noRotateTrainingData', help='dont rotate data', dest='rotateTrainingData' , action='store_false')
    parser.add_argument('--unitTangent', help='unit tangent', dest='unittangent' , action='store_true')
    #parser.add_argument('--noUnitTangent', help='no unit tangent [default]', dest='unittangent' , action='store_false')
    parser.add_argument('--visStreamlines', help='visualize streamlines before proceeding with data generation', dest='visStreamlines' , action='store_true')
    parser.add_argument('--ISMRM2015data', help='generate training data for the ismrm 2015 dataset', dest='ISMRM2015data' , action='store_true')
    parser.add_argument('--HCPid', default='100307', help='case id of the HCP dataset to be used [default: 100307]')
    
    parser.set_defaults(rotateTrainingData=False)   
    parser.set_defaults(unittangent=False)   
    parser.set_defaults(visStreamlines=False)   
    parser.set_defaults(ISMRM2015data=False)
    args = parser.parse_args()
    
    tensorModel = args.tensorModel
    noX = args.nx
    noY = args.ny
    noZ = args.nz
    dataRepr = args.repr
    b_value = args.b
    stepWidth = args.sw
    minimumStreamlineLength = args.minLength
    maximumStreamlineLength = args.maxLength
    shOrder = args.sh
    noStreamlines = args.noStreamlines
    faThreshold = args.fa
    unitTangent = args.unittangent
    visStreamlines = args.visStreamlines
    useISMRM = args.ISMRM2015data
    hcpCaseID = args.HCPid
    rotateTrainingData = args.rotateTrainingData
    
    print('Parameters:')
    print(str(args))
  
    
    nameDWIDataset = ''
    
    if(useISMRM):
        nameDWIDataset = 'ISMRM_2015_Tracto_challenge_data'
        useDenoising = False
        bvals,bvecs,gtab,dwi,aff,t1 = dwi_tools.loadISMRMData('data/%s' % (nameDWIDataset), denoiseData = useDenoising, resliceToHCPDimensions=False)
        b0_mask, binarymask = median_otsu(dwi[:,:,:,0], 2, 1)
        nameDWIDataset = 'ISMRM_2015_Tracto_challenge_data_denoised_preproc'
    else:
        nameDWIDataset = 'HCP%s' % (hcpCaseID)
        bvals,bvecs,gtab,dwi,aff,t1,binarymask = dwi_tools.loadHCPData('data/HCP/%s' % (hcpCaseID))
    
    wholebrainseeds = seeds_from_mask(binarymask, affine=aff)
    
    if(tensorModel == 'dti'):
        import dipy.reconst.dti as dti
        start_time = time.time()
        dti_model = dti.TensorModel(gtab)
        dti_fit = dti_model.fit(dwi, mask=binarymask)
        dti_fit_odf = dti_fit.odf(sphere = default_sphere)
        dg = DeterministicMaximumDirectionGetter
        directionGetter = dg.from_pmf(dti_fit_odf, max_angle=30., sphere=default_sphere)
        runtime = time.time() - start_time
        print('DTI Runtime ' + str(runtime) + 's')
        

    elif(tensorModel == 'csd'):
        response, ratio = auto_response(gtab, dwi, roi_radius=10, fa_thr=0.7)
        csd_model = ConstrainedSphericalDeconvModel(gtab, response)
        sphere = get_sphere('symmetric724')
        start_time = time.time()
        csd_peaks = peaks_from_model(model=csd_model,
                                     data=dwi,
                                     sphere=sphere,
                                     mask=binarymask,
                                     relative_peak_threshold=.5,
                                     #relative_peak_threshold=.6,
                                     min_separation_angle=25,
                                     parallel=False)
        runtime = time.time() - start_time
        print('CSD Runtime ' + str(runtime) + ' s')
        import dipy.reconst.dti as dti
        start_time = time.time()
        dti_model = dti.TensorModel(gtab, fit_method='LS')
        dti_fit = dti_model.fit(dwi, mask=binarymask)
        runtime = time.time() - start_time
        print('DTI Runtime ' + str(runtime) + 's')
        start_time = time.time()       
        directionGetter = csd_peaks

    classifier = ThresholdTissueClassifier(dti_fit.fa, faThreshold)
    streamlines_generator = LocalTracking(directionGetter, classifier, wholebrainseeds, aff, step_size=stepWidth)
    streamlines = Streamlines(streamlines_generator)
    streamlines_filtered = dwi_tools.filterStreamlinesByLength(streamlines, minimumStreamlineLength)
    streamlines_filtered = dwi_tools.filterStreamlinesByMaxLength(streamlines_filtered, maximumStreamlineLength)
    runtime = time.time() - start_time
    print('LocalTracking Runtime ' + str(runtime) + 's')

    
    tInfo = '%s_sw%.1f_minL%d_maxL%d_fa%.2f' % (tensorModel,stepWidth,minimumStreamlineLength,maximumStreamlineLength,faThreshold)
    
    if(visStreamlines):
        dwi_tools.visStreamlines(streamlines_filtered)
    
    dwi_tools.saveVTKstreamlines(streamlines_filtered, 'data/%s_%s_noreslicing_mrtrixDenoised_preproc.vtk' % (nameDWIDataset,tInfo))
    
    # crop DWI data
    dwi_subset, gtab_subset, bvals_subset, bvecs_subset = dwi_tools.cropDatsetToBValue(b_value, bvals, bvecs, dwi)
    b0_idx = bvals < 10
    b0 = dwi[..., b0_idx].mean(axis=3)
    dwi_singleShell = np.concatenate((dwi_subset, dwi[..., b0_idx]), axis=3)
    #    dwi_singleShell_norm = dwi_tools.normalize_dwi(dwi_singleShell, b0)
    bvals_singleShell = np.concatenate((bvals_subset, bvals[..., b0_idx]), axis=0)
    bvecs_singleShell = np.concatenate((bvecs_subset, bvecs[b0_idx,]), axis=0)
    gtab_singleShell = gradient_table(bvals=bvals_singleShell, bvecs=bvecs_singleShell, b0_threshold = 10)
    
    if(dataRepr == 'sh'):
        t_data = dwi_tools.get_spherical_harmonics_coefficients(bvals=bvals_subset,bvecs=bvecs_subset,sh_order=shOrder, dwi=dwi_subset, b0 = b0)
    elif(dataRepr == 'res100'):
        t_data, resamplingSphere = dwi_tools.resample_dwi(dwi_subset, b0, bvals_subset, bvecs_subset, sh_order=shOrder, smooth=0, mean_centering=False)
    elif(dataRepr == 'raw'):
        t_data = dwi_subset
        
    if(noStreamlines>0):
        streamlines_filtered = np.random.choice(streamlines_filtered,noStreamlines, replace=False)
        nameDWIDataset = nameDWIDataset + "_" + str(noStreamlines) + "sl_"
   
    print('Generating training data... ')

    start_time = time.time()
    train_DWI,train_prevDirection, train_nextDirection,_ = dwi_tools.generateTrainingData(streamlines_filtered, t_data, unitTension = unitTangent, affine=aff, noX=noX,noY=noY,noZ=noZ,coordinateScaling=coordinateScaling,distToNeighbours=1, noCrossings = noCrossingFibres, step = 1, rotateTrainingData = rotateTrainingData)
    #train_DWI,train_prevDirection, train_nextDirection,_,slo,train_DWI_pastAggregated = dwi_tools.generateTrainingData(streamlines_filtered, t_data, unitTension = unitTangent, affine=aff, noX=noX,noY=noY,noZ=noZ,coordinateScaling=coordinateScaling,distToNeighbours=1, noCrossings = noCrossingFibres, step = 1)
    runtime = time.time() - start_time
    print('Runtime ' + str(runtime) + ' s ')

    pTrainData = '/data/nico/trainingdata/%s_b%d_%s_sw%.1f_%dx%dx%d_ut%d_rotateTD%d.h5' % (nameDWIDataset, b_value, dataRepr, stepWidth, noX,noY,noZ,unitTangent,rotateTrainingData)

    print('Writing training data: ' + pTrainData )
    
    with h5py.File(pTrainData,"w") as f:
        f.create_dataset('train_DWI',data=train_DWI)
    #    f.create_dataset('train_DWI_pastAgg',data=train_DWI_pastAggregated)
        f.create_dataset('train_curPosition',data=train_prevDirection)   
    #   f.create_dataset('train_LikelyFibreDirections',data=train_LikelyFibreDirections)   
        f.create_dataset('train_NextFibreDirection',data=train_nextDirection)
    
if __name__ == "__main__":
    main()
