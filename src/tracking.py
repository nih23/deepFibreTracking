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

def start(seeds, data, model, affine, noX=3, noY=3,noZ=3,dw=288,coordinateScaling = 0.1, stepWidth = 0.1, useSph = False,inverseDirection = False, bitracker = False):   
    '''
    fibre tracking using neural networks
    '''    
    # the stepDirection is currently employed to track fibre's into the predicted as well as its opposite direction
    stepDirection = 1
    if(inverseDirection):
        stepDirection = -1
    
    noSeeds = len(seeds)
    noIterations = 100 #TODO: DONT HARDCODE THE NO OF ITERATIONS, IMPLEMENT STOPPING CRITERIA

    # initialize streamline positions data
    vNorms = np.zeros([noSeeds,noIterations+1])
    streamlinePositions = np.zeros([noSeeds,noIterations+1,3])
    streamlinePositions[:,0,] = seeds[0:noSeeds]
    streamlinePositions[:,1,] = seeds[0:noSeeds]

    # interpolate data given these coordinates for each channel
    x = np.zeros([noSeeds,noX,noY,noZ,dw])
    x_,y_,z_ = dwi_tools._getCoordinateGrid(noX,noY,noZ,coordinateScaling)

    # prepare transformations to project a streamline point from IJK into RAS coordinate system to interpolate DWI data
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

        lastDirections = (streamlinePositions[:,iter-1,] - streamlinePositions[:,iter,]) # previousPosition - currentPosition
        vecNorms = np.sqrt(np.sum(lastDirections ** 2 , axis = 1)) # make unit vector
        lastDirections = np.nan_to_num(lastDirections / vecNorms[:,None])
            
        if(bitracker):
            predictedDirection = model.predict([x, lastDirections], batch_size = 2**12)
        else:
            predictedDirection = model.predict([x],  batch_size = 2**12)
        
        # depending on the coordinates change different de-normalization approach
        if(useSph == True):
            #predictedDirection = dwi_tools.convAllFromSphToEuclCoords((2*np.pi)*predictedDirection + np.pi)
            vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1)) # should be unit in this case.. 
        else:
            # squash output to unit length
            vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1))
            #predictedDirection = np.nan_to_num(predictedDirection / vecNorms[:,None])   
        vNorms[:,iter,] = vecNorms
        
        # update next streamline position
        for j in range(0,noSeeds):
            lv1 = predictedDirection[j,]
            pv1 = lastDirections[j,]
            theta = np.arcsin(np.dot(pv1,lv1) / (np.linalg.norm(pv1)*np.linalg.norm(lv1)))            
            if(theta < 0 and iter>1):
                predictedDirection[j,] = -predictedDirection[j,]
            
            streamlinePositions[j,iter+1,] = streamlinePositions[j,iter,] - stepDirection * stepWidth * predictedDirection[j,]
        
        stepDirection = 1

    return streamlinePositions, vNorms


def joinTwoAlignedStreamlineLists(streamlines_left,streamlines_right):
    assert(len(streamlines_left) == len(streamlines_right), "The two lists of streamlines need to have the same number of elements.")
    
    streamlines_joined = []
    
    for i in range(0,len(streamlines_left)):
        sl_l = np.flipud(streamlines_left[i])
        sl_r = streamlines_right[i]
        streamlines_joined.append(np.concatenate([sl_l, sl_r]))
        
    return streamlines_joined


def startWithStopping(seeds, data, model, affine, mask, fa, fa_threshold = 0.2, noX=3, noY=3,noZ=3,dw=288,coordinateScaling = 0.1, stepWidth = 0.1, useSph = False,inverseDirection = False, bitracker = False):   
    '''
    fibre tracking using neural networks
    '''    
    # the stepDirection is currently employed to track fibre's into the predicted as well as its opposite direction
    mask = mask.astype(np.float)
    stepDirection = 1
    if(inverseDirection):
        stepDirection = -1
    
    noSeeds = len(seeds)
    noIterations = 100 #TODO: DONT HARDCODE THE NO OF ITERATIONS, IMPLEMENT STOPPING CRITERIA

    # initialize streamline positions data
    vNorms = np.zeros([noSeeds,noIterations+1])
    streamlinePositions = np.zeros([noSeeds,noIterations+1,3])
    streamlinePositions[:,0,] = seeds[0:noSeeds]
    streamlinePositions[:,1,] = seeds[0:noSeeds]
    indexLastStreamlinePosition = noIterations * np.ones([noSeeds], dtype=np.intp)
    
    # interpolate data given these coordinates for each channel
    x = np.zeros([noSeeds,noX,noY,noZ,dw])
    x_,y_,z_ = dwi_tools._getCoordinateGrid(noX,noY,noZ,coordinateScaling)

    # prepare transformations to project a streamline point from IJK into RAS coordinate system to interpolate DWI data
    aff_ras_ijk = np.linalg.inv(affine) # aff: IJK -> RAS
    M = aff_ras_ijk[:3, :3]
    abc = aff_ras_ijk[:3, 3]
    abc = abc[:,None]
    
    
    start_time = time.time()    
    for iter in range(1,noIterations):
        if((iter % 10) == 0):
            print(str(iter) + "/" + str(noIterations) + "[" + str(time.time() - start_time) + "s]")
        curStreamlinePos_ras = streamlinePositions[:,iter,].T
        curStreamlinePos_ijk = (M.dot(curStreamlinePos_ras) + abc).T
        x = dwi_tools.interpolateDWIVolume(data, curStreamlinePos_ijk, x_,y_,z_, noX = noX, noY = noY, noZ = noZ)
            
        lastDirections = (streamlinePositions[:,iter-1,] - streamlinePositions[:,iter,]) # previousPosition - currentPosition
        vecNorms = np.sqrt(np.sum(lastDirections ** 2 , axis = 1)) # make unit vector
        lastDirections = np.nan_to_num(lastDirections / vecNorms[:,None])
            
        if(bitracker):
            predictedDirection = model.predict([x, lastDirections], batch_size = 2**16)
        else:
            predictedDirection = model.predict([x], batch_size = 2**16)
        
        # depending on the coordinates change different de-normalization approach
        if(useSph == True):
            #predictedDirection = dwi_tools.convAllFromSphToEuclCoords((2*np.pi)*predictedDirection + np.pi)
            vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1)) # should be unit in this case.. 
        else:
            # squash output to unit length
            vecNorms = np.sqrt(np.sum(predictedDirection ** 2 , axis = 1))
            #predictedDirection = np.nan_to_num(predictedDirection / vecNorms[:,None])   
        vNorms[:,iter,] = vecNorms
        
        # update next streamline position
        for j in range(0,noSeeds):
            lv1 = predictedDirection[j,]
            pv1 = lastDirections[j,]
            theta = np.arcsin(np.dot(pv1,lv1) / (np.linalg.norm(pv1)*np.linalg.norm(lv1)))            
            if(theta < 0 and iter>1):
                predictedDirection[j,] = -predictedDirection[j,]
            
        candidatePosition = streamlinePositions[:,iter,] - stepDirection * stepWidth * predictedDirection
        candidatePosition_ijk = (M.dot(candidatePosition.T) + abc).T   #projectRAStoIJK(candidatePosition,M,abc)
        validPoints = areVoxelsValidStreamlinePoints(candidatePosition_ijk, mask, fa, fa_threshold)
        
        for j in range(0,noSeeds):
            if(validPoints[j]):
                streamlinePositions[j,iter+1,] = candidatePosition[j,]
            else:
                indexLastStreamlinePosition[j] = np.min((indexLastStreamlinePosition[j],iter))
        
        stepDirection = 1
        
        
    streamlinePositions = streamlinePositions.tolist()
    
    for seedIdx in range(0,noSeeds):
        currentStreamline = np.array(streamlinePositions[seedIdx])
        currentStreamline = currentStreamline[0:indexLastStreamlinePosition[seedIdx],]
        streamlinePositions[seedIdx] = currentStreamline
        

    return streamlinePositions, vNorms


def areVoxelsValidStreamlinePoints(nextCandidatePositions_ijk,mask,fa,fa_threshold):
    return np.logical_and( np.greater(vfu.interpolate_scalar_3d(mask,nextCandidatePositions_ijk)[0], 0), np.greater(vfu.interpolate_scalar_3d(fa,nextCandidatePositions_ijk)[0], fa_threshold)  )
    #return (vfu.interpolate_scalar_3d(mask,nextCandidatePositions_ijk)[0] == 0) and (vfu.interpolate_scalar_3d(fa,nextCandidatePositions_ijk)[0] < fa_threshold)
    

def isVoxelValidStreamlinePoint(nextCandidatePosition_ijk,mask,fa,fa_threshold):
    
    
    
    if(vfu.interpolate_scalar_3d(mask,nextCandidatePosition_ijk)[0] == 0):
        return False

    if(vfu.interpolate_scalar_3d(fa,nextCandidatePosition_ijk)[0] < fa_threshold):
        return False
    
    return True


def projectRAStoIJK(coordinateRAS, M, abc):
    coordinateRAS = coordinateRAS[:,None]
    return (M.dot(coordinateRAS) + abc).T
