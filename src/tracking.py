import time
import nrrd
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import h5py

from dipy.tracking.local import LocalTracking, ThresholdTissueClassifier
from dipy.tracking.utils import random_seeds_from_mask
from dipy.reconst.dti import TensorModel, quantize_evecs
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

from dipy.tracking import utils

import src.dwi_tools as dwi_tools
import src.nn_helper as nn_helper

from dipy.tracking.streamline import values_from_volume
import dipy.align.vector_fields as vfu
from dipy.core.sphere import Sphere
from dipy.core import subdivide_octahedron 
from scipy.spatial import KDTree
from dipy.core import subdivide_octahedron 

from dipy.tracking.local import LocalTracking
from dipy.viz import window, actor
from dipy.viz.colormap import line_colors
from dipy.tracking.streamline import Streamlines
from dipy.tracking import metrics

import numpy as np
import warnings

import tensorflow as tf
from joblib import Parallel, delayed
import multiprocessing
from keras.models import load_model



def start(seeds, data, model, affine, noX=3, noY=3,noZ=3,dw=288,coordinateScaling = 0.1, stepWidth = 0.1, nnOutputToUse = 0, useSph = False):   
    '''
    fibre tracking using neural networks
    
    NOTE/BUG: tracking is done in image coordinate system currently
    '''    
    # assume seeds in image coordinate system
    # TODO: make steps in RAS cs
    
    noSeeds = len(seeds)
    noIterations = 1000

    # initialize streamline positions data
    vNorms = np.zeros([noSeeds,noIterations+1])
    streamlinePositions = np.zeros([noSeeds,noIterations+1,3])
    streamlinePositions[:,0,] = seeds[0:noSeeds]
    streamlinePositions[:,1,] = seeds[0:noSeeds]

    # interpolate data given these coordinates for each channel
    x = np.zeros([noSeeds,noX,noY,noZ,dw])
    x_,y_,z_ = dwi_tools._getCoordinateGrid(noX,noY,noZ,coordinateScaling)

    aff_ras_ijk = np.linalg.inv(affine) # aff: IJK -> RAS
    M = aff_ras_ijk[:3, :3]
    abc = aff_ras_ijk[:3, 3]
    abc = abc[:,None]
    
    for iter in range(1,noIterations):
        # interpolate dwi data at each point of our streamline
        for j in range(0,noSeeds):
            # project from RAS to image coordinate system
            curStreamlinePos_ras = streamlinePositions[j,iter,]
            curStreamlinePos_ras = curStreamlinePos_ras[:,None]
            curStreamlinePos_ijk = (M.dot(curStreamlinePos_ras) + abc).T
            
            coordVecs = np.vstack(np.meshgrid(x_,y_,z_)).reshape(3,-1).T + curStreamlinePos_ijk
            x[j,] = dwi_tools.interpolatePartialDWIVolume(data,coordVecs, noX = noX, noY = noY, noZ = noZ, coordinateScaling = coordinateScaling,x_ = x_,y_ = y_,z_ = z_)

                
        # predict possible directions
        #x_ext = nn_helper.normalizeDWI(x)
        #lastDirections = streamlinePositions[:,iter,] - streamlinePositions[:,iter-1,]
        #lastDirections = np.expand_dims(lastDirections, axis=1)
        
        
        predictedDirection = model.predict([x])[nnOutputToUse] # 0 -> previous direction, 1 -> next direction
        
        # depending on the coordinates change different de-normalization approach
        if(useSph == True):
            predictedDirection = dwi_tools.convAllFromSphToEuclCoords((2*np.pi)*predictedDirection + np.pi)
            vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1)) # should be unit in this case.. 
        else:
            # squash output to unit length
            vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1))
            #predictedDirection = np.nan_to_num(predictedDirection / vecNorms[:,None])   
        vNorms[:,iter,] = vecNorms
        
        
        # update next streamline position
        for j in range(0,noSeeds):
            streamlinePositions[j,iter+1,] = streamlinePositions[j,iter,] + stepWidth * predictedDirection[j,]


    return streamlinePositions, vNorms

def start_singleTracker(seeds, data, model, affine, noX=3, noY=3,noZ=3,dw=288,coordinateScaling = 0.1, stepWidth = 0.1, useSph = False,inverseDirection = False):   
    '''
    fibre tracking using neural networks
    
    NOTE/BUG: tracking is done in image coordinate system currently
    '''    
    # assume seeds in image coordinate system
    # TODO: make steps in RAS cs
    
    stepDirection = 1
    if(inverseDirection):
        stepDirection = -1
    
    noSeeds = len(seeds)
    noIterations = 1000

    # initialize streamline positions data
    vNorms = np.zeros([noSeeds,noIterations+1])
    streamlinePositions = np.zeros([noSeeds,noIterations+1,3])
    streamlinePositions[:,0,] = seeds[0:noSeeds]
    streamlinePositions[:,1,] = seeds[0:noSeeds]

    # interpolate data given these coordinates for each channel
    x = np.zeros([noSeeds,noX,noY,noZ,dw])
    x_,y_,z_ = dwi_tools._getCoordinateGrid(noX,noY,noZ,coordinateScaling)

    aff_ras_ijk = np.linalg.inv(affine) # aff: IJK -> RAS
    M = aff_ras_ijk[:3, :3]
    abc = aff_ras_ijk[:3, 3]
    abc = abc[:,None]
    
    for iter in range(1,noIterations):
        # interpolate dwi data at each point of our streamline
        for j in range(0,noSeeds):
            # project from RAS to image coordinate system
            curStreamlinePos_ras = streamlinePositions[j,iter,]
            curStreamlinePos_ras = curStreamlinePos_ras[:,None]
            curStreamlinePos_ijk = (M.dot(curStreamlinePos_ras) + abc).T
            
            coordVecs = np.vstack(np.meshgrid(x_,y_,z_)).reshape(3,-1).T + curStreamlinePos_ijk
            x[j,] = dwi_tools.interpolatePartialDWIVolume(data,coordVecs, noX = noX, noY = noY, noZ = noZ, coordinateScaling = coordinateScaling,x_ = x_,y_ = y_,z_ = z_)

                
        # predict possible directions
        #x_ext = nn_helper.normalizeDWI(x)
        #lastDirections = streamlinePositions[:,iter,] - streamlinePositions[:,iter-1,]
        #lastDirections = np.expand_dims(lastDirections, axis=1)
        
        
        predictedDirection = model.predict([x])
        
        # depending on the coordinates change different de-normalization approach
        if(useSph == True):
            predictedDirection = dwi_tools.convAllFromSphToEuclCoords((2*np.pi)*predictedDirection + np.pi)
            vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1)) # should be unit in this case.. 
        else:
            # squash output to unit length
            vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1))
            #predictedDirection = np.nan_to_num(predictedDirection / vecNorms[:,None])   
        vNorms[:,iter,] = vecNorms
        
        
        # update next streamline position
        for j in range(0,noSeeds):
            streamlinePositions[j,iter+1,] = streamlinePositions[j,iter,] + stepDirection * stepWidth * predictedDirection[j,]


    return streamlinePositions, vNorms