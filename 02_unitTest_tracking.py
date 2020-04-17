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
import src.tracking as tracking
from src.state import TractographyInformation

from dipy.tracking.streamline import values_from_volume
import dipy.align.vector_fields as vfu
from dipy.core.sphere import Sphere
from dipy.core import subdivide_octahedron 
from scipy.spatial import KDTree
from dipy.core import subdivide_octahedron 

from dipy.tracking.local import LocalTracking
from dipy.tracking.streamline import Streamlines, transform_streamlines
from dipy.tracking import metrics
from dipy.tracking.utils import random_seeds_from_mask, seeds_from_mask

from src.model import ModelLSTM, ModelMLP
import torch
import torch.nn as nn
import numpy as np
import warnings

from joblib import Parallel, delayed
import multiprocessing

import src.tracking as tracking

import os

def main():
    myState = TractographyInformation()
    myState.b_value = 1000 
    myState.stepWidth = 0.6
    myState.shOrder = 8
    myState.faThreshold = 0
    myState.rotateData = False
    myState.hcpID = "HCP/100307" 
    myState.gridSpacing = 1.0
    myState.resampleDWIAfterRotation = False
    myState.pStopTracking = 0.5
    myState.predictionInIJK = True 
    myState.fa = myState.faThreshold
    myState.repr = "raw" #our data is already resampled 
    myState.dim = [3,3,3] #Grid größe?
    myState.unitTangent = False # maybe True
    computeFA = False
    minimumStreamlineLength = 50 # mm
    
    
    pModel = 'result/unit_test'

    pResult = pModel + '.vtk'

    os.makedirs(pModel.replace('.h5','').replace('/models/','/tracking/'), exist_ok=True)

    ## LOAD PYT MODEL
    model = ModelLSTM(dropout=0.06, hidden_sizes=[191,191], input_size=2700, activation_function=nn.Tanh()).cuda()
    #model = ModelMLP(hidden_sizes=[192,192,192,192], dropout=0.07, input_size=2700, activation_function=nn.Tanh()).cuda()
    model.load_state_dict(torch.load('models/model.pt.lstm', map_location='cpu'))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    if not os.path.isfile('./cache/tracking_data3x.pt'): 
        # load DWI data
        print('Loading dataset %s at b=%d' % (myState.hcpID, myState.b_value))
        bvals,bvecs,gtab,dwi,aff,t1,binarymask = dwi_tools.loadHCPData('data/%s' % (myState.hcpID))
        dwi_subset, gtab_subset, bvals_subset, bvecs_subset = dwi_tools.cropDatsetToBValue(myState.b_value, bvals, bvecs, dwi)
        b0_idx = bvals < 10
        b0 = dwi[..., b0_idx].mean(axis=3)
    
        # resample dataset
        #TODO NICO: SET SMOOTH? 

        # set data to use in tracking
        dwi_subset = dwi_tools.normalize_dwi(dwi_subset, b0)
        
        print('Resampling to 100 directions on repulsion100')
        tracking_data, resamplingSphere = dwi_tools.resample_dwi(dwi_subset, b0, bvals_subset, bvecs_subset, sh_order=myState.shOrder, smooth=0, mean_centering=False)
        torch.save((tracking_data, binarymask,aff, b0, bvals_subset, bvecs_subset,t1), './cache/tracking_data3x.pt')
    else:
        (tracking_data, binarymask,aff, b0, bvals_subset, bvecs_subset,t1) = torch.load('./cache/tracking_data3x.pt')
    # DTI 
    #tracking_data = torch.from_numpy(tracking_data).float().cuda()
    #binarymask = torch.from_numpy(binarymask).float().cuda()
    #aff = torch.from_numpy(aff).float().cuda()
    #b0 = torch.from_numpy(b0).float().cuda()
    #bvals_subset = torch.from_numpy(bvals_subset).float().cuda()
    #bvecs_subset = torch.from_numpy(bvecs_subset).float().cuda()
    if(computeFA):
        print('DTI fa estimation')
        start_time = time.time()
        dti_model = dti.TensorModel(gtab_singleShell, fit_method='LS')
        dti_fit = dti_model.fit(dwi_singleShell_norm)
        runtime = time.time() - start_time
        print('Runtime ' + str(runtime) + 's')

    myState.b0 = b0
    myState.bvals = bvals_subset
    myState.bvecs = bvecs_subset

    # DL-based Tractography
    print("Tractography")
    #wholebrainseeds = seeds_from_mask(binarymask, affine=aff)
    rndseeds = random_seeds_from_mask(binarymask, seeds_count=30000, seed_count_per_voxel=False, affine=aff)
    #seedsToUse = wholebrainseeds 
    seedsToUse = rndseeds
    seedsToUse = torch.from_numpy(seedsToUse)
    start_time = time.time()
    #streamlines_mlp_simple_sc,vNorms = tracking.start(myState = myState, printProgress = True, mask=binarymask, inverseDirection=False, seeds=seedsToUse, data=tracking_data, affine=aff, model=model)
    #streamlines, vNorms = tracking.startWithRNNAndMLPCombination(myState = myState, printProgress = True, mask=binarymask, inverseDirection=False, seeds=seedsToUse, data=tracking_data, affine=aff, model=model, mlp_model=model_mlp)
    streamlines, vNorms = tracking.startWithRNNBatches(myState = myState, printProgress = True, mask=binarymask, inverseDirection=False, seeds=seedsToUse, data=tracking_data, affine=aff, model=model)
    runtime = time.time() - start_time
    print('Runtime ' + str(runtime) + ' s ')

    #streamlines_mlp_simple_sc_2,vNorms_2 = tracking.start(myState = myState, printProgress = True, mask=binarymask, inverseDirection=True, seeds=seedsToUse, data=tracking_data, affine=aff, model=model)
    #runtime = time.time() - start_time
    #print('Runtime ' + str(runtime) + ' s ')

    # join left and right streamlines
    print("Postprocessing streamlines and removing streamlines shorter than " + str(minimumStreamlineLength) + " mm")
    #streamlines_joined_sc = tracking.joinTwoAlignedStreamlineLists(streamlines_mlp_simple_sc,streamlines_mlp_simple_sc_2)
       
    #streamlines_joined_sc = dwi_tools.filterStreamlinesByLength(streamlines_joined_sc, minimumStreamlineLength)
    streamlines_joined_sc =  dwi_tools.filterStreamlinesByLength(streamlines, minimumStreamlineLength)
    print("Writing the data to disk")
    dwi_tools.saveVTKstreamlines(streamlines_joined_sc,pResult)
    
    #streamlines_joined_sc_imageCS = transform_streamlines(streamlines_joined_sc, np.linalg.inv(aff))
    #dwi_tools.visStreamlines(streamlines_joined_sc_imageCS, t1, vol_slice_idx = 75)
    
    

if __name__ == "__main__":
    main()
