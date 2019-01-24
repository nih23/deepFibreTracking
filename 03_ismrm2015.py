import numpy as np
import tensorflow as tf
import os
import argparse
import time
import dipy.reconst.dti as dti
import src.dwi_tools as dwi_tools
import src.tracking as tracking
from src.tied_layers1d import Convolution2D_tied
from src.state import TractographyInformation
from src.nn_helper import swish, squared_cosine_proximity_2, weighted_binary_crossentropy, mse_directionInvariant
from src.SelectiveDropout import SelectiveDropout
from dipy.segment.mask import median_otsu
from dipy.core.gradients import gradient_table, gradient_table_from_bvals_bvecs
from dipy.tracking.utils import random_seeds_from_mask, seeds_from_mask
from keras.models import load_model
from keras.layers import Activation


def main():
    parser = argparse.ArgumentParser(description='Deep Learning Tractography -- Tracking')
    parser.add_argument('model', help='path to trained neural network')
    parser.add_argument('-c', '--caseid', default='ISMRM_2015_Tracto_challenge_data', help='name of tracking case')
    parser.add_argument('-n', '--noSteps', default=200, type=int, help='no. tracking steps')
    parser.add_argument('-b', dest='b', default=1000, type=int, help='b-value')
    parser.add_argument('-sw',dest='sw',default=1.0, type=float, help='no. tracking steps')
    parser.add_argument('-mil','--minLength', type=int, default=20, help='minimal length of a streamline [mm]')
    parser.add_argument('-mal','--maxLength', type=int, default=200, help='maximum length of a streamline [mm]')
    parser.add_argument('-fa','--faThreshold', dest='fa', type=float, default=0, help='fa threshold in case of non-magical models')
    parser.add_argument('-sh', '--shOrder', dest='sh', type=int, default=4, help='order of spherical harmonics (if used)')
    parser.add_argument('-nt', '--noThreads', type=int, default=4, help='number of parallel threads of the data generator. Note: this also increases the memory demand.')
    parser.add_argument('--denoise', help='denoise dataset', dest='denoise' , action='store_true')
    parser.add_argument('--nodenoise', help='dont denoise dataset', dest='denoise' , action='store_false')
    parser.add_argument('--reslice', help='reslice datase to 1.25mm^3', dest='reslice' , action='store_true')
    parser.add_argument('--rotateData', help='reslice datase to 1.25mm^3', dest='rotateData' , action='store_true')
    parser.set_defaults(denoise=False)   
    parser.set_defaults(reslice=False)   
    parser.set_defaults(rotateData=False)
    args = parser.parse_args()

    myState = TractographyInformation()
    myState.b_value = args.b
    myState.stepWidth = args.sw
    myState.shOrder = args.sh
    myState.faThreshold = args.fa
    myState.rotateData = args.rotateData
    myState.model = args.model
    myState.hcpID = args.caseid

    minimumStreamlineLength = args.minLength
    maximumStreamlineLength = args.maxLength
    noTrackingSteps = args.noSteps
    resliceDataToHCPDimension = args.reslice
    useDenoising = args.denoise

    # load neural network
    pResult = "results_tracking/" + myState.hcpID + os.path.sep + myState.model.replace('.h5','').replace('/models/','/').replace('results','') + 'reslice-' + str(resliceDataToHCPDimension) + '-denoising-' + str(useDenoising) + '-' + str(noTrackingSteps) + 'st-20mm-fa-' + str(myState.faThreshold)
    tracker = load_model(myState.model , custom_objects={'tf':tf, 'swish':Activation(swish), 'SelectiveDropout': SelectiveDropout, 'squared_cosine_proximity_2': squared_cosine_proximity_2, 'Convolution2D_tied': Convolution2D_tied, 'weighted_binary_crossentropy': weighted_binary_crossentropy, 'mse_directionInvariant': mse_directionInvariant})
    myState = myState.parseModel(tracker)
    print('Loaded model %s' % (myState.model))
    tracker.summary()
    os.makedirs(pResult, exist_ok=True)

    # load DWI data
    print('Loading dataset %s at b=%d' % (myState.hcpID, myState.b_value))
    bvals,bvecs,gtab,dwi,aff,t1 = dwi_tools.loadISMRMData('data/%s' % (myState.hcpID), denoiseData = useDenoising, resliceToHCPDimensions=resliceDataToHCPDimension)
    b0_mask, binarymask = median_otsu(dwi[:,:,:,0], 2, 1)
   
    # crop DWI data
    dwi_subset, gtab_subset, bvals_subset, bvecs_subset = dwi_tools.cropDatsetToBValue(myState.b_value, bvals, bvecs, dwi)
    b0_idx = bvals < 10
    b0 = dwi[..., b0_idx].mean(axis=3)

    if(not myState.magicModel):
        print('FA estimation')
        dwi_singleShell = np.concatenate((dwi_subset, dwi[..., b0_idx]), axis=3)
        bvals_singleShell = np.concatenate((bvals_subset, bvals[..., b0_idx]), axis=0)
        bvecs_singleShell = np.concatenate((bvecs_subset, bvecs[b0_idx,]), axis=0)
        gtab_singleShell = gradient_table(bvals=bvals_singleShell, bvecs=bvecs_singleShell, b0_threshold = 10)

        start_time = time.time()
        dti_model = dti.TensorModel(gtab_singleShell, fit_method='LS')
        dti_fit = dti_model.fit(dwi_singleShell)
        runtime = time.time() - start_time
        print('Runtime ' + str(runtime) + 's')

    dwi_subset = dwi_tools.normalize_dwi(dwi_subset, b0)
    myState.b0 = b0
    myState.bvals = bvals_subset
    myState.bvecs = bvecs_subset


    print("Tractography")
    # whole brain seeds
    seedsToUse = seeds_from_mask(binarymask, affine=aff)


    if(myState.magicModel and myState.model.find('2mlp_single')>0):
        # magic 2mlp_single with aggregation
        start_time = time.time()
        streamlines_joined_sc,vNorms,stopProb = tracking.startAggregatedMagicModel(myState, printfProfiling = False, printProgress = True, mask=binarymask, inverseDirection=False, seeds=seedsToUse, data=dwi_subset, affine=aff, model=tracker, noIterations = noTrackingSteps)
        runtime = time.time() - start_time
    else:
        # standard models
        start_time = time.time()
        streamlines_joined_sc,vNorms = tracking.start(myState, printfProfiling = False, printProgress = True, mask=binarymask,fa=dti_fit.fa, seeds=seedsToUse, data=dwi_subset, affine=aff, model=tracker, noIterations = noTrackingSteps)
        runtime = time.time() - start_time

   
    print("Postprocessing streamlines and removing streamlines shorter than " + str(minimumStreamlineLength) + " mm")
    streamlines_joined_sc = dwi_tools.filterStreamlinesByLength(streamlines_joined_sc, minimumStreamlineLength)
    streamlines_joined_sc = dwi_tools.filterStreamlinesByMaxLength(streamlines_joined_sc, maximumStreamlineLength) # 200mm

    print("The data is being written to disk.")
    dwi_tools.saveVTKstreamlines(streamlines_joined_sc,pResult + '.vtk')


if __name__ == "__main__":
    main()
